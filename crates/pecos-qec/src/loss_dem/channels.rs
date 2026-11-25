use super::{Location, LossDemError, Side, circuit_error, invalid};
use crate::fault_tolerance::circuit_definitions::definitions_from_dag_circuit;
use crate::fault_tolerance::dem_builder::{
    DirectSourceFamily, ExactPauliConversion, NoiseChannelKind, NoiseChannelResidual,
    SourceMetadata, circuit_influence_map, exact_pauli_terms,
};
use crate::fault_tolerance::propagator::{Direction, Pauli, apply_gate};
use crate::{DemBuilder, DetectorErrorModel, FaultMechanism, NoiseConfig};
use pecos_core::{CliffordLowering, Gate, gate_type::GateType};
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::noise::general::{LossChannelKind, LossSite};
use pecos_quantum::TickCircuit;
use pecos_simulators::PauliProp;
use std::collections::BTreeMap;

/// A categorical Pauli channel on one or two operands at an explicit location.
/// The first label digit targets `location.operand`, the second the next operand.
/// Omitted probability mass is identity. Channels supplement gate-type noise.
/// Totals use ordinary gate-channel validation's strict [0, 1] bound, not the
/// inline payload path's 1e-12 sum tolerance. Source metadata names existing
/// after-locations and the conjugated Paulis (or a fitted exclusive signature).
#[derive(Debug, Clone)]
pub struct LocationChannel {
    /// First operand and side of the channel.
    pub location: Location,
    /// Pauli labels and unconditional probabilities, including optional identity.
    pub probabilities: BTreeMap<String, f64>,
}

// This is the apply_sq_faults / two-qubit engine routing set, not GateNoiseKind::Single.
pub(super) fn site_arity(gate: GateType) -> Option<usize> {
    match gate {
        GateType::H
        | GateType::F
        | GateType::Fdg
        | GateType::SZ
        | GateType::SZdg
        | GateType::SX
        | GateType::SXdg
        | GateType::SY
        | GateType::SYdg
        | GateType::X
        | GateType::Y
        | GateType::Z
        | GateType::T
        | GateType::Tdg
        | GateType::RX
        | GateType::RY
        | GateType::RZ
        | GateType::U
        | GateType::RXY1Q => Some(1),
        gate if gate.is_two_qubit() => Some(2),
        _ => None,
    }
}

impl DemBuilder<'_> {
    /// Build ordinary gate noise plus explicit categorical location channels.
    /// No influence locations are inserted or renumbered. Nondeterministic
    /// detector parities remain accepted by the ordinary path.
    ///
    /// # Panics
    /// Panics if conversion returns a nonexistent node, a unitary operand lacks its
    /// internal fault location, or the location index exceeds residual metadata capacity.
    ///
    /// # Errors
    /// Rejects invalid locations, nonunitary channel sites, or invalid channel probabilities.
    /// Inexact categorical fits carry the ordinary DEM residual diagnostics.
    pub fn try_from_tick_circuit_with_location_channels(
        circuit: &TickCircuit,
        noise: NoiseConfig,
        channels: &[LocationChannel],
    ) -> Result<DetectorErrorModel, LossDemError> {
        if channels.is_empty() {
            return Self::try_from_tick_circuit_with_noise_config(circuit, noise)
                .map_err(LossDemError::from);
        }
        if let Some(error) =
            crate::fault_tolerance::propagator::first_unsupported_tick_gate(circuit)
        {
            return Err(circuit_error("location-channel circuit", error));
        }
        let (dag, nodes) = circuit
            .to_dag_with_node_map()
            .map_err(|error| circuit_error("tick-to-DAG conversion", error))?;
        let map = circuit_influence_map(&dag)?;
        let builder = DemBuilder::new(&map)
            .with_noise_config(noise)
            .with_exact_branch_replay_context(&dag)
            .with_circuit_definitions(definitions_from_dag_circuit(&dag)?)?;
        let mut dem = builder.try_build()?;
        let (detectors, observables) = builder.build_measurement_mappings();
        let mut coordinates = BTreeMap::new();
        for ((tick, batch), instances) in nodes {
            let mut operand = 0;
            for node in instances {
                let gate = dag
                    .gate(node)
                    .expect("conversion returns inserted DAG nodes");
                for &qubit in &gate.qubits {
                    coordinates.insert((tick, batch, operand), (node, qubit));
                    operand += 1;
                }
            }
        }
        let mut after_locations = BTreeMap::new();
        let mut cliffords = BTreeMap::new();
        for (index, loc) in map.locations.iter().enumerate() {
            cliffords.entry(loc.node).or_insert(loc.clifford);
            if !loc.before
                && let [qubit] = loc.qubits.as_slice()
            {
                after_locations.insert((loc.node, *qubit), index);
            }
        }
        for channel in channels {
            let location = channel.location;
            let &(node, _) = coordinates
                .get(&(location.tick, location.batch, location.operand))
                .ok_or_else(|| {
                    invalid(format!(
                        "location {location:?} has no operand; choose a stored batch operand"
                    ))
                })?;
            let gate = dag.gate(node).expect("mapped DAG node retains its gate");
            if site_arity(gate.gate_type).is_none() && gate.gate_type != GateType::Idle {
                return Err(invalid(format!(
                    "location {location:?} on {:?} is not a unitary/idle channel site; choose a unitary gate",
                    gate.gate_type
                )));
            }
            let arity = channel.probabilities.keys().next().map_or(1, String::len);
            if !(1..=2).contains(&arity) {
                return Err(invalid(format!(
                    "channel labels at {location:?} have arity {arity}; use one or two Pauli letters"
                )));
            }
            let targets = (0..arity)
                .map(|leg| {
                    coordinates
                        .get(&(location.tick, location.batch, location.operand + leg))
                        .filter(|(n, _)| *n == node)
                        .map(|(_, q)| *q)
                        .ok_or_else(|| invalid(format!(
                            "channel at {location:?} crosses a gate instance; keep its operands in one instance"
                        )))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut probabilities = vec![0.0; 1 << (2 * arity)];
            for (label, &probability) in &channel.probabilities {
                if label.len() != arity
                    || !probability.is_finite()
                    || !(0.0..=1.0).contains(&probability)
                {
                    return Err(invalid(format!(
                        "channel {label:?} = {probability} at {location:?} is invalid; use equal-width Pauli labels and probabilities in [0, 1]"
                    )));
                }
                let mut word = 0;
                for (leg, letter) in label.chars().enumerate() {
                    let digit = match letter {
                        'I' => 0,
                        'X' => 1,
                        'Y' => 2,
                        'Z' => 3,
                        _ => {
                            return Err(invalid(format!(
                                "label {label:?} at {location:?} is invalid; use I/X/Y/Z"
                            )));
                        }
                    };
                    word |= digit << (2 * leg);
                }
                probabilities[word] += probability;
            }
            let total: f64 = probabilities.iter().sum();
            // Match ordinary gate-channel validate_exclusive_probabilities: the
            // categorical total must lie in [0, 1], without inline's sum tolerance.
            if total > 1.0 {
                return Err(invalid(format!(
                    "channel total {total} at {location:?} exceeds 1; reduce the probabilities"
                )));
            }
            probabilities[0] += 1.0 - total;
            // Use the inline conversion for its supported full Pauli distributions.
            // Other channels are projected first: distinct labels can have the same
            // detector signature. Inexact fits retain the ordinary DEM residual metadata.
            let channel_weight = probabilities.iter().skip(1).sum();
            let conversion = exact_pauli_terms(&probabilities)
                .map_err(|error| circuit_error(format!("location {location:?}"), error))?;
            let (terms, full_exact) = match conversion {
                ExactPauliConversion::Exact(terms) => (terms, true),
                ExactPauliConversion::Unavailable(_) => (
                    probabilities
                        .into_iter()
                        .enumerate()
                        .skip(1)
                        .filter(|(_, p)| *p > 0.0)
                        .collect(),
                    false,
                ),
            };
            let mut fit_source_indices = std::collections::BTreeSet::new();
            let mut exclusive = BTreeMap::new();
            for (word, probability) in terms {
                let mut prop = PauliProp::new();
                for (leg, qubit) in targets.iter().enumerate() {
                    match (word >> (2 * leg)) & 3 {
                        1 => prop.track_x(&[qubit.index()]),
                        2 => prop.track_y(&[qubit.index()]),
                        3 => prop.track_z(&[qubit.index()]),
                        _ => {}
                    }
                }
                if location.side == Side::Before && gate.gate_type != GateType::Idle {
                    let clifford = cliffords
                        .get(&node)
                        .expect("unitary DAG node has a resolved Clifford");
                    let action = match *clifford {
                        CliffordLowering::Named(g) | CliffordLowering::PerQubit(g) => g,
                    };
                    let lowered = Gate::simple(action, gate.qubits.clone());
                    // Forward conjugation of the entire correlated Pauli through G.
                    let outcome = apply_gate(&mut prop, &lowered, Direction::Forward);
                    if matches!(
                        outcome,
                        crate::fault_tolerance::propagator::PauliPropagationOutcome::Unsupported
                    ) {
                        return Err(invalid(format!(
                            "Clifford {clifford:?} at {location:?} cannot be propagated; use a supported Clifford"
                        )));
                    }
                }
                let mut effect = FaultMechanism::new();
                let mut source_indices = Vec::new();
                let mut source_paulis = Vec::new();
                for &qubit in &gate.qubits {
                    let pauli = match (
                        prop.contains_x(qubit.index()),
                        prop.contains_z(qubit.index()),
                    ) {
                        (false, false) => continue,
                        (true, false) => Pauli::X,
                        (false, true) => Pauli::Z,
                        (true, true) => Pauli::Y,
                    };
                    let index = *after_locations
                        .get(&(node, qubit))
                        .expect("unitary operand has an existing after fault location");
                    source_indices.push(index);
                    source_paulis.push(pauli);
                    fit_source_indices.insert(index);
                    effect = effect.xor(&builder.compute_mechanism(
                        index,
                        pauli,
                        &detectors,
                        &observables,
                    ));
                }
                if !effect.is_empty() {
                    if full_exact {
                        dem.add_direct_contribution_with_source(
                            effect,
                            probability,
                            SourceMetadata::new(
                                &source_indices,
                                &source_paulis,
                                &vec![gate.gate_type; source_indices.len()],
                                &vec![false; source_indices.len()],
                            ),
                        );
                    } else {
                        *exclusive.entry(effect).or_insert(0.0) += probability;
                    }
                }
            }
            if !full_exact {
                let context = format!("location {location:?}");
                let fit = crate::fault_tolerance::dem_builder::fit_exclusive_signatures(
                    exclusive,
                    FaultMechanism::xor,
                    &context,
                )
                .map_err(|error| circuit_error(&context, error))?;
                if let Some((effect, magnitude)) = fit.residual {
                    let index = after_locations[&(node, targets[0])];
                    dem.add_idle_noise_residual(NoiseChannelResidual {
                        location_index: u32::try_from(index)
                            .expect("fault-location index must fit in residual metadata"),
                        channel_kind: if arity == 2 {
                            NoiseChannelKind::TwoQubitGate
                        } else if gate.gate_type == GateType::Idle {
                            NoiseChannelKind::Idle
                        } else {
                            NoiseChannelKind::SingleQubitGate
                        },
                        effect,
                        magnitude,
                        channel_weight,
                    });
                }
                let indices: Vec<_> = fit_source_indices.into_iter().collect();
                let gate_types = vec![gate.gate_type; indices.len()];
                let before_flags = vec![false; indices.len()];
                for (effect, probability) in fit.mechanisms {
                    dem.add_direct_contribution_with_source(
                        effect,
                        probability,
                        SourceMetadata::new(&indices, &[], &gate_types, &before_flags)
                            .with_direct_source_family(DirectSourceFamily::ExclusiveSignature),
                    );
                }
            }
        }
        Ok(dem)
    }
}

/// Lossless Pauli DEM plus present-operand recovery false alarms and pure-Pauli events.
///
/// E2 branches MUST REPLACE the false-alarm term at each site/operand that is lost;
/// a recovery there is true recovery, not an additional false alarm. Treating event
/// labels containing L as identity in this background is a first-order approximation.
/// Loss trajectories and their DEMs are intentionally not enumerated here.
///
/// # Errors
/// Rejects unsupported compiled settings and invalid categorical probabilities.
pub fn background_dem(
    circuit: &TickCircuit,
    noise: &GeneralNoiseModel,
) -> Result<DetectorErrorModel, LossDemError> {
    let description = noise.description();
    let config = super::pauli_noise_config(circuit, &description)?;
    let mut channels = Vec::new();
    if let Some(loss) = &description.loss {
        for (tick, gates) in circuit.iter_ticks() {
            for (batch, gate) in gates.iter_gate_batches().enumerate() {
                let Some(arity) = site_arity(gate.gate_type) else {
                    continue;
                };
                if description.is_noiseless_operation(gate.as_gate()) {
                    continue;
                }
                for channel in &loss.channels {
                    let (site_arity, side) = match channel.site {
                        LossSite::BeforeSingleQubit => (1, Side::Before),
                        LossSite::AfterSingleQubit => (1, Side::After),
                        LossSite::BeforeTwoQubit => (2, Side::Before),
                        LossSite::AfterTwoQubit => (2, Side::After),
                    };
                    if arity != site_arity {
                        continue;
                    }
                    match &channel.kind {
                        LossChannelKind::Recovery { false_positive, .. } => {
                            if *false_positive == 0.0 {
                                continue;
                            }
                            for operand in 0..gate.qubits.len() {
                                channels.push(LocationChannel {
                                    location: Location {
                                        tick,
                                        batch,
                                        operand,
                                        side,
                                    },
                                    probabilities: ["X", "Y", "Z"]
                                        .into_iter()
                                        .map(|p| (p.to_string(), false_positive / 4.0))
                                        .collect(),
                                });
                            }
                        }
                        LossChannelKind::Events {
                            probability,
                            events,
                        } => {
                            let probabilities = events
                                .iter()
                                .filter(|(label, weight)| {
                                    !label.contains('L') && **weight * probability > 0.0
                                })
                                .map(|(label, weight)| (label.clone(), weight * probability))
                                .collect::<BTreeMap<_, _>>();
                            if probabilities.is_empty() {
                                continue;
                            }
                            for operand in (0..gate.qubits.len()).step_by(arity) {
                                channels.push(LocationChannel {
                                    location: Location {
                                        tick,
                                        batch,
                                        operand,
                                        side,
                                    },
                                    probabilities: probabilities.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    DemBuilder::try_from_tick_circuit_with_location_channels(circuit, config, &channels)
}
