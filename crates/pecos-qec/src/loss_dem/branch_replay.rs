//! Construct record-aligned loss circuits, then extract their affine signatures.
use super::branches::BranchNormalForm;
use super::trajectory::{ActionKind, Path, Program};
use super::{LossDemError, Span, affine_parity, circuit_error, invalid};
use crate::fault_tolerance::circuit_definitions::definitions_from_tick_circuit;
use crate::fault_tolerance::fault_sampler::symbolic_measurement_history;
use pecos_core::{BitSet, Gate, gate_type::GateType};
use pecos_quantum::TickCircuit;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) struct ReplayOptions {
    pub convention_coins: bool,
    // Catalog-internal gauge mutation: retain interactions with a mixed lost qubit.
    pub keep_dropped: bool,
}
impl Default for ReplayOptions {
    fn default() -> Self {
        Self {
            convention_coins: true,
            keep_dropped: false,
        }
    }
}

pub(super) fn pauli_gate(word: usize, qubit: usize) -> Vec<Gate> {
    match word {
        1 => vec![Gate::x(&[qubit])],
        2 => vec![Gate::z(&[qubit])],
        3 => vec![Gate::y(&[qubit])],
        _ => Vec::new(),
    }
}

/// Z: MZ-unrecorded. X: H; MZ-unrecorded; H.
/// Y: `SZdg; H; MeasureZLost; H; SZ` (`U = H SZdg` maps Y to Z).
pub(super) fn coin(word: usize, qubit: usize) -> Vec<Gate> {
    let kinds: &[GateType] = match word {
        1 => &[GateType::H, GateType::MeasureZLost, GateType::H],
        2 => &[GateType::MeasureZLost],
        3 => &[
            GateType::SZdg,
            GateType::H,
            GateType::MeasureZLost,
            GateType::H,
            GateType::SZ,
        ],
        _ => &[],
    };
    kinds
        .iter()
        .map(|kind| Gate::simple(*kind, vec![qubit.into()]))
        .collect()
}

pub(super) fn mixed_reset(qubit: usize) -> Vec<Gate> {
    [GateType::PZ, GateType::H, GateType::MeasureZLost]
        .into_iter()
        .map(|kind| Gate::simple(kind, vec![qubit.into()]))
        .collect()
}

struct Definition {
    name: String,
    coordinate: usize,
    records: BitSet,
    constant: bool,
}

pub(super) struct ReplayContext {
    pub num_detectors: usize,
    pub num_observables: usize,
    definitions: Vec<Definition>,
}

impl ReplayContext {
    pub fn new(circuit: &TickCircuit) -> Result<Self, LossDemError> {
        let definitions = definitions_from_tick_circuit(circuit)?;
        let num_detectors = definitions
            .detectors
            .iter()
            .map(|d| d.id as usize + 1)
            .max()
            .unwrap_or(0);
        let num_observables = definitions
            .observables
            .iter()
            .map(|d| d.id as usize + 1)
            .max()
            .unwrap_or(0);
        let original = symbolic_measurement_history(circuit)
            .map_err(|e| circuit_error("original symbolic replay", e))?;
        let detectors = definitions
            .detectors
            .iter()
            .map(|d| (format!("detector {}", d.id), d.id as usize, &d.measurements));
        let observables = definitions.observables.iter().map(|o| {
            (
                format!("observable {}", o.id),
                num_detectors + o.id as usize,
                &o.measurements,
            )
        });
        let mut resolved = Vec::new();
        for (name, coordinate, measurements) in detectors.chain(observables) {
            // Definitions preserve multiplicity; affine_parity takes a set.
            let mut records = BitSet::new();
            for &record in measurements {
                records ^= &BitSet::single(record);
            }
            let parity = affine_parity(&original, &records).map_err(|e| circuit_error(&name, e))?;
            if !parity.variables.is_empty() {
                return Err(invalid(format!(
                    "lossless {name} is nondeterministic (variables {:?}); use deterministic detector and observable parities",
                    parity.variables
                )));
            }
            resolved.push(Definition {
                name,
                coordinate,
                records,
                constant: parity.constant,
            });
        }
        Ok(Self {
            num_detectors,
            num_observables,
            definitions: resolved,
        })
    }

    pub fn normal_form(
        &self,
        program: &Program,
        path: &Path,
        options: ReplayOptions,
    ) -> Result<BranchNormalForm, LossDemError> {
        let built = loss_circuit(program, path, options)?;
        let history = symbolic_measurement_history(&built)
            .map_err(|e| circuit_error("loss symbolic replay", e))?;
        let mut constant = BitSet::new();
        let mut signatures: BTreeMap<usize, BitSet> = BTreeMap::new();
        for definition in &self.definitions {
            let parity = affine_parity(&history, &definition.records)
                .map_err(|e| circuit_error(&definition.name, e))?;
            if parity.constant != definition.constant {
                constant.insert(definition.coordinate);
            }
            for variable in &parity.variables {
                signatures
                    .entry(variable)
                    .or_default()
                    .insert(definition.coordinate);
            }
        }
        let width = self.num_detectors + self.num_observables;
        let span = Span::new(width, &signatures.into_values().collect::<Vec<_>>())?;
        let reduced = Span::new(width, &[constant])?.quotient(&span)?;
        let constant = reduced.basis().first().cloned().unwrap_or_default();
        Ok(BranchNormalForm { constant, span })
    }
}

// Only the original emission order is retained; inserted gates consume no records.
fn loss_circuit(
    program: &Program,
    path: &Path,
    options: ReplayOptions,
) -> Result<TickCircuit, LossDemError> {
    let mut built = TickCircuit::new();
    for tick in &program.ticks {
        // All operations within the original tick are disjoint. Before/after
        // insertions on those operands commute across stored batch boundaries.
        for &index in &tick.before {
            insertions(&mut built, index, path)?;
        }
        let mut retained = Vec::new();
        for &index in &tick.gates {
            let action = &program.actions[index];
            if let Some(ActionKind::Gate { gate, .. }) = &action.kind
                && (options.keep_dropped || !path.dropped.contains(&action.location))
            {
                retained.push(gate.as_ref().clone());
            }
        }
        append_tick(&mut built, retained)?;
        for &index in &tick.after {
            insertions(&mut built, index, path)?;
        }
    }
    Ok(built)
}

fn insertions(circuit: &mut TickCircuit, action: usize, path: &Path) -> Result<(), LossDemError> {
    if let Some(gates) = path.inserts.get(&action) {
        for gate in gates {
            append_tick(circuit, vec![gate.clone()])?;
        }
    }
    Ok(())
}

fn append_tick(circuit: &mut TickCircuit, gates: Vec<Gate>) -> Result<(), LossDemError> {
    let tick = circuit.num_ticks();
    circuit.tick();
    circuit
        .get_tick_mut(tick)
        .expect("tick just appended")
        .try_add_gates(gates)
        .map_err(|e| circuit_error("loss circuit insertion", e))?;
    Ok(())
}
