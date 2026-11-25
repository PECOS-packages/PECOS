//! Compile the engine's loss schedule once; walk only trajectory state thereafter.
use super::branch_replay::{ReplayOptions, coin, mixed_reset, pauli_gate};
use super::branches::{FalseAlarmTag, LossMoment, RootKey, RootKind, RootPhase};
use super::{Location, LossDemError, OutputTable, ProbeId, Side, invalid};
use pecos_core::{Gate, gate_type::GateType};
use pecos_engines::noise::general::{GeneralNoiseDescription, HookDescription, LossChannelKind};
use pecos_quantum::TickCircuit;
use std::collections::{BTreeMap, BTreeSet};

type DroppedMass = BTreeMap<BTreeSet<ProbeId>, f64>;

#[cfg(test)]
thread_local! {
    pub(super) static ACTION_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) struct RootSeed {
    pub key: RootKey,
    pub prior: f64,
    pub action: usize,
    pub qubits: Vec<usize>,
}

pub(super) enum ActionKind {
    Recovery {
        tag: FalseAlarmTag,
        false_negative: f64,
        qubit: usize,
    },
    Reload {
        qubit: usize,
    },
    Gate {
        gate: Box<Gate>,
        record: Option<usize>,
        probe: Option<ProbeId>,
        noisy: bool,
    },
}

pub(super) struct Action {
    pub location: Location,
    pub phase: RootPhase,
    // Source slots carry insertion coordinates but are never traversed.
    pub kind: Option<ActionKind>,
}

pub(super) struct TickPlan {
    pub before: Vec<usize>,
    pub gates: Vec<usize>,
    pub after: Vec<usize>,
}

pub(super) struct Program {
    pub actions: Vec<Action>,
    pub ticks: Vec<TickPlan>,
    pub roots: Vec<RootSeed>,
    actions_by_qubit: BTreeMap<usize, Vec<usize>>,
    partner_probability: f64,
    partner_pauli: Option<BTreeMap<String, f64>>,
}

#[derive(Clone)]
pub(super) struct Path {
    pub weight: f64,
    pub flags: BTreeSet<ProbeId>,
    pub history: Vec<LossMoment>,
    pub records: BTreeSet<usize>,
    pub dropped: BTreeSet<Location>,
    pub removals: BTreeSet<FalseAlarmTag>,
    pub inserts: BTreeMap<usize, Vec<Gate>>,
    lost: BTreeSet<usize>,
}

impl Path {
    fn change(&mut self, action: &Action, qubit: usize, lost: bool) {
        if lost {
            self.lost.insert(qubit);
        } else {
            self.lost.remove(&qubit);
        }
        self.history.push(LossMoment {
            location: action.location,
            phase: action.phase,
            lost: self.lost.clone(),
        });
    }

    fn insert(&mut self, action: usize, gates: impl IntoIterator<Item = Gate>) {
        self.inserts.entry(action).or_default().extend(gates);
    }

    fn partner_pauli(
        &self,
        action: usize,
        qubit: usize,
        model: &BTreeMap<String, f64>,
    ) -> Vec<Self> {
        // The single-qubit Pauli group modulo phase is F2^2: X=1, Z=2, Y=3.
        let support: Vec<_> = model.iter().filter(|(_, p)| **p > 0.0).collect();
        let uniform_probability = match support.len() {
            1 => 1.0_f64,
            2 => 0.5,
            4 => 0.25,
            _ => 0.0,
        };
        // Exact bits are intentional: near-uniform models branch exactly.
        let uniform = uniform_probability > 0.0
            && support
                .iter()
                .all(|(_, p)| p.to_bits() == uniform_probability.to_bits());
        let word = |s: &str| match s {
            "X" => 1,
            "Z" => 2,
            "Y" => 3,
            _ => 0,
        };
        if uniform {
            let mut path = self.clone();
            let constant = word(support[0].0);
            path.insert(action, pauli_gate(constant, qubit));
            if support.len() == 2 {
                path.insert(action, coin(constant ^ word(support[1].0), qubit));
            } else if support.len() == 4 {
                path.insert(action, coin(1, qubit));
                path.insert(action, coin(2, qubit));
            }
            vec![path]
        } else {
            // Mutually exclusive outcomes, including identity. Each child inherits
            // the trajectory metadata; a Pauli changes neither loss nor flags.
            support
                .into_iter()
                .map(|(label, probability)| {
                    let mut child = self.clone();
                    child.weight *= probability;
                    child.insert(action, pauli_gate(word(label), qubit));
                    child
                })
                .collect()
        }
    }
}

impl Program {
    pub fn new(
        circuit: &TickCircuit,
        description: &GeneralNoiseDescription,
        outputs: &OutputTable,
    ) -> Result<Self, LossDemError> {
        let mut program = Self {
            actions: Vec::new(),
            ticks: Vec::new(),
            roots: Vec::new(),
            actions_by_qubit: BTreeMap::new(),
            partner_probability: description
                .loss
                .as_ref()
                .map_or(0.0, |l| l.lost_partner_probability),
            partner_pauli: description
                .loss
                .as_ref()
                .and_then(|l| l.lost_partner_pauli.clone()),
        };
        let records: BTreeMap<_, _> = outputs
            .record_locations
            .iter()
            .copied()
            .enumerate()
            .map(|(r, l)| (l, r))
            .collect();
        let probes: BTreeMap<_, _> = outputs.probes.iter().map(|p| (p.location, p.id)).collect();
        for (tick, gates) in circuit.iter_ticks() {
            program.ticks.push(TickPlan {
                before: Vec::new(),
                gates: Vec::new(),
                after: Vec::new(),
            });
            for (batch, stored) in gates.iter_gate_batches().enumerate() {
                let gate = stored.as_gate();
                if gate.gate_type.is_meta() {
                    continue;
                }
                let noisy = !description.is_noiseless_operation(gate);
                let arity = super::channels::site_arity(gate.gate_type);
                let base = Location {
                    tick,
                    batch,
                    operand: 0,
                    side: Side::Before,
                };
                if noisy && let Some(arity) = arity {
                    let order = if arity == 1 {
                        &description.p1_hook_order_before_gate
                    } else {
                        &description.p2_hook_order_before_gate
                    };
                    program.hooks(description, order, base, gate, arity)?;
                }
                if gate.gate_type == GateType::PZ
                    && description.loss.as_ref().is_some_and(|l| l.reload_on_prep)
                {
                    program.reload(base, gate);
                }
                let instance_arity = gate.gate_type.quantum_arity().max(1);
                for (instance, qs) in gate.qubits.chunks(instance_arity).enumerate() {
                    let operand = instance * instance_arity;
                    let location = Location { operand, ..base };
                    let mut single = gate.clone();
                    single.qubits = qs.to_vec().into();
                    if !gate.meas_ids.is_empty() {
                        single.meas_ids =
                            gate.meas_ids[operand..operand + qs.len()].to_vec().into();
                    }
                    program.push(Action {
                        location,
                        phase: RootPhase::Operation,
                        kind: Some(ActionKind::Gate {
                            gate: Box::new(single),
                            record: records.get(&location).copied(),
                            probe: probes.get(&location).copied(),
                            noisy,
                        }),
                    });
                }
                let after = Location {
                    side: Side::After,
                    ..base
                };
                if let Some(loss) = &description.loss {
                    if noisy
                        && (gate.gate_type.consumes_measurement_record()
                            || gate.gate_type.is_recordless_z_measurement())
                    {
                        program.scalar(
                            after,
                            gate,
                            RootKind::Measurement,
                            RootPhase::MeasurementLoss,
                            loss.measurement,
                        );
                    }
                    if gate.gate_type == GateType::MPZ && loss.reload_on_prep {
                        program.reload(after, gate);
                    }
                    if noisy && matches!(gate.gate_type, GateType::PZ | GateType::MPZ) {
                        program.scalar(
                            after,
                            gate,
                            RootKind::Preparation,
                            RootPhase::PreparationLoss,
                            loss.prep,
                        );
                    }
                    if noisy && gate.gate_type == GateType::Idle {
                        program.scalar(
                            after,
                            gate,
                            RootKind::Idle,
                            RootPhase::Operation,
                            (loss.idle_rate * gate.idle_duration()).clamp(0.0, 1.0),
                        );
                    }
                }
                if noisy && let Some(arity) = arity {
                    let order = if arity == 1 {
                        &description.p1_hook_order_after_gate
                    } else {
                        &description.p2_hook_order_after_gate
                    };
                    program.hooks(description, order, after, gate, arity)?;
                    if arity == 2
                        && description.idle_after_2q > f64::EPSILON
                        && let Some(loss) = &description.loss
                    {
                        program.scalar(
                            after,
                            gate,
                            RootKind::AfterTwoQubitIdle,
                            RootPhase::AfterTwoQubitIdle,
                            (loss.idle_rate * description.idle_after_2q).clamp(0.0, 1.0),
                        );
                    }
                }
            }
        }
        Ok(program)
    }

    fn push(&mut self, action: Action) -> usize {
        let index = self.actions.len();
        let tick = &mut self.ticks[action.location.tick];
        match &action.kind {
            Some(ActionKind::Gate { record, .. }) => {
                tick.gates.push(index);
                if record.is_some() {
                    tick.before.push(index);
                } else {
                    tick.after.push(index);
                }
            }
            _ if action.location.side == Side::Before => tick.before.push(index),
            _ => tick.after.push(index),
        }
        match &action.kind {
            Some(ActionKind::Gate { gate, .. }) => {
                for q in &gate.qubits {
                    self.actions_by_qubit
                        .entry(q.index())
                        .or_default()
                        .push(index);
                }
            }
            Some(ActionKind::Recovery { qubit, .. } | ActionKind::Reload { qubit }) => {
                self.actions_by_qubit.entry(*qubit).or_default().push(index);
            }
            None => {}
        }
        self.actions.push(action);
        index
    }

    fn reload(&mut self, base: Location, gate: &Gate) {
        for (operand, q) in gate.qubits.iter().enumerate() {
            self.push(Action {
                location: Location { operand, ..base },
                phase: RootPhase::Reload,
                kind: Some(ActionKind::Reload { qubit: q.index() }),
            });
        }
    }

    fn scalar(
        &mut self,
        base: Location,
        gate: &Gate,
        kind: RootKind,
        phase: RootPhase,
        prior: f64,
    ) {
        for (operand, q) in gate.qubits.iter().enumerate() {
            let location = Location { operand, ..base };
            let action = self.push(Action {
                location,
                phase,
                kind: None,
            });
            if prior > 0.0 {
                self.roots.push(RootSeed {
                    key: RootKey {
                        kind,
                        channel: None,
                        label: "L".into(),
                        location,
                        phase,
                    },
                    prior,
                    action,
                    qubits: vec![q.index()],
                });
            }
        }
    }

    fn hooks(
        &mut self,
        description: &GeneralNoiseDescription,
        order: &[HookDescription],
        base: Location,
        gate: &Gate,
        arity: usize,
    ) -> Result<(), LossDemError> {
        let Some(loss) = &description.loss else {
            return Ok(());
        };
        for (position, entry) in order.iter().enumerate() {
            let HookDescription::Loss(channel) = *entry else {
                return Err(invalid(format!(
                    "hook {entry:?} is not a loss hook; remove non-loss hooks"
                )));
            };
            let phase = match base.side {
                Side::Before => RootPhase::BeforeHook(position),
                Side::After => RootPhase::AfterHook(position),
            };
            match &loss.channels[channel].kind {
                LossChannelKind::Events {
                    probability,
                    events,
                } => {
                    for (instance, qs) in gate.qubits.chunks(arity).enumerate() {
                        let location = Location {
                            operand: instance * arity,
                            ..base
                        };
                        let action = self.push(Action {
                            location,
                            phase,
                            kind: None,
                        });
                        for (label, weight) in events {
                            let prior = probability * weight;
                            if label.contains('L') && prior > 0.0 {
                                self.roots.push(RootSeed {
                                    key: RootKey {
                                        kind: RootKind::Events,
                                        channel: Some(channel),
                                        label: label.clone(),
                                        location,
                                        phase,
                                    },
                                    prior,
                                    action,
                                    qubits: qs.iter().map(pecos_core::QubitId::index).collect(),
                                });
                            }
                        }
                    }
                }
                LossChannelKind::Recovery { false_negative, .. } => {
                    for (operand, q) in gate.qubits.iter().enumerate() {
                        let location = Location { operand, ..base };
                        let tag = FalseAlarmTag { channel, location };
                        self.push(Action {
                            location,
                            phase,
                            kind: Some(ActionKind::Recovery {
                                tag,
                                false_negative: *false_negative,
                                qubit: q.index(),
                            }),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    // The minimum indexed action across the current lost set preserves engine
    // order. Spread/recovery changes that set before the next lookup. Sources are
    // excluded: a trajectory does not draw another independent root.
    pub(super) fn next_action(
        &self,
        cursor: usize,
        lost: &BTreeSet<usize>,
    ) -> Option<(usize, &ActionKind)> {
        let index = lost
            .iter()
            .filter_map(|q| {
                let indices = self.actions_by_qubit.get(q)?;
                indices
                    .get(indices.partition_point(|&index| index < cursor))
                    .copied()
            })
            .min()?;
        let kind = self.actions[index]
            .kind
            .as_ref()
            .expect("only gates, recovery and reload are indexed");
        Some((index, kind))
    }

    pub fn walk(
        &self,
        seed: &RootSeed,
        cutoff: f64,
        cap: usize,
        options: ReplayOptions,
    ) -> Result<(Vec<Path>, DroppedMass, usize), LossDemError> {
        let mut path = Path {
            weight: seed.prior,
            flags: BTreeSet::new(),
            history: Vec::new(),
            records: BTreeSet::new(),
            dropped: BTreeSet::new(),
            removals: BTreeSet::new(),
            inserts: BTreeMap::new(),
            lost: BTreeSet::new(),
        };
        for (&q, letter) in seed.qubits.iter().zip(seed.key.label.chars()) {
            if letter == 'L' {
                path.lost.insert(q);
                path.insert(seed.action, [Gate::pz(&[q])]);
                if options.keep_dropped {
                    path.insert(seed.action, mixed_reset(q));
                }
            } else {
                path.insert(
                    seed.action,
                    pauli_gate(
                        match letter {
                            'X' => 1,
                            'Y' => 3,
                            'Z' => 2,
                            _ => 0,
                        },
                        q,
                    ),
                );
            }
        }
        path.history.push(LossMoment {
            location: seed.key.location,
            phase: seed.key.phase,
            lost: path.lost.clone(),
        });
        let mut pending = vec![(seed.action + 1, path)];
        let mut completed = Vec::new();
        let mut dropped = BTreeMap::new();
        let mut explored_paths = 0;
        while let Some((mut next, mut path)) = pending.pop() {
            explored_paths += 1;
            check_branch_cap(seed, explored_paths, cap)?;
            if path.weight < cutoff {
                *dropped.entry(path.flags).or_default() += path.weight;
                continue;
            }
            while let Some((index, kind)) = self.next_action(next, &path.lost) {
                #[cfg(test)]
                ACTION_VISITS.with(|count| count.set(count.get() + 1));
                next = index + 1;
                let action = &self.actions[index];
                match kind {
                    ActionKind::Reload { qubit } => {
                        if path.lost.contains(qubit) {
                            path.change(action, *qubit, false);
                        }
                    }
                    ActionKind::Recovery {
                        tag,
                        false_negative,
                        qubit,
                    } => {
                        if !path.lost.contains(qubit) {
                            continue;
                        }
                        path.removals.insert(*tag);
                        let mut recovered = path.clone();
                        recovered.weight *= 1.0 - false_negative;
                        recovered.change(action, *qubit, false);
                        recovered.insert(index, mixed_reset(*qubit));
                        path.weight *= false_negative;
                        if recovered.weight > 0.0 {
                            if recovered.weight < cutoff {
                                *dropped.entry(recovered.flags).or_default() += recovered.weight;
                            } else {
                                pending.push((next, recovered));
                            }
                        }
                        if path.weight <= 0.0 {
                            break;
                        }
                    }
                    ActionKind::Gate {
                        gate,
                        record,
                        probe,
                        noisy,
                    } => {
                        let qs: Vec<_> =
                            gate.qubits.iter().map(pecos_core::QubitId::index).collect();
                        if !qs.iter().any(|q| path.lost.contains(q)) {
                            continue;
                        }
                        if let Some(probe) = probe {
                            path.flags.insert(*probe);
                        }
                        if let Some(record) = record {
                            path.records.insert(*record);
                            if options.convention_coins {
                                path.insert(index, mixed_reset(qs[0]));
                            }
                        }
                        if gate.gate_type.consumes_measurement_record()
                            || gate.gate_type.is_recordless_z_measurement()
                            || gate.gate_type.is_detection()
                        {
                            continue;
                        }
                        path.dropped.insert(action.location);
                        let own_event = seed.key.kind == RootKind::Events
                            && matches!(seed.key.phase, RootPhase::BeforeHook(_))
                            && seed.key.location.tick == action.location.tick
                            && seed.key.location.batch == action.location.batch
                            && seed.key.location.operand == action.location.operand;
                        if gate.gate_type.is_two_qubit() && *noisy && !own_event {
                            for q in qs {
                                if path.lost.contains(&q) {
                                    continue;
                                }
                                let mut spread = path.clone();
                                spread.weight *= self.partner_probability;
                                spread.change(action, q, true);
                                spread.insert(index, [Gate::pz(&[q])]);
                                path.weight *= 1.0 - self.partner_probability;
                                if spread.weight > 0.0 {
                                    if spread.weight < cutoff {
                                        *dropped.entry(spread.flags).or_default() += spread.weight;
                                    } else {
                                        pending.push((next, spread));
                                    }
                                }
                                if path.weight > 0.0
                                    && let Some(model) = &self.partner_pauli
                                {
                                    // No second present leg is possible in a dropped pair.
                                    // Resume every outcome after this gate, using the same
                                    // absolute-weight cutoff and explored-path cap as spread/FN.
                                    for child in path.partner_pauli(index, q, model) {
                                        if child.weight < cutoff {
                                            *dropped.entry(child.flags).or_default() +=
                                                child.weight;
                                        } else if child.weight > 0.0 {
                                            pending.push((next, child));
                                        }
                                    }
                                    path.weight = 0.0;
                                }
                                if path.weight <= 0.0 {
                                    break;
                                }
                            }
                        }
                    }
                }
                if path.weight <= 0.0 {
                    break;
                }
                if path.weight < cutoff {
                    break;
                }
            }
            if path.weight > 0.0 {
                if path.weight < cutoff {
                    *dropped.entry(path.flags).or_default() += path.weight;
                } else {
                    completed.push(path);
                }
            }
        }
        Ok((completed, dropped, explored_paths))
    }
}

// Count every explored segment, including walks whose descendants all prune.
fn check_branch_cap(seed: &RootSeed, count: usize, cap: usize) -> Result<(), LossDemError> {
    if count > cap {
        return Err(invalid(format!(
            "root {:?} exceeds max_branches = {cap}: count {count} explored paths; increase max_branches or min_branch_weight",
            seed.key
        )));
    }
    Ok(())
}
