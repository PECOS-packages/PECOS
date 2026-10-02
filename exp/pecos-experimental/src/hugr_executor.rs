// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Circuit execution for symbolic stabilizer simulation.
//!
//! **EXPERIMENTAL: This API is unstable and may change without notice.**
//!
//! This module executes circuits implementing [`Circuit`]
//! through the [`SymbolicSparseStab`] simulator, enabling efficient sampling from
//! the resulting measurement history.
//!
//! # Overview
//!
//! The workflow is:
//! 1. Trace Guppy programs through QIS in Python
//! 2. Convert the traced tick circuit to a DAG circuit
//! 3. Execute through [`SymbolicSparseStab`] to get symbolic measurement dependencies
//! 4. Use [`MeasurementSampler`] to efficiently generate many shots
//!
//! This approach is highly efficient because:
//! - The circuit is simulated only once symbolically
//! - Sampling is reduced to XOR operations on random bits
//! - Millions of shots can be generated in milliseconds
//!
//! # Example
//!
//! ```rust
//! use pecos_simulators::{SymbolicSparseStab, MeasurementSampler};
//! use pecos_experimental::execute_hugr;
//! use pecos_quantum::{DagCircuit, Gate};
//!
//! // Create a Bell state circuit
//! let mut circuit = DagCircuit::new();
//! circuit.add_gate(Gate::h(&[0]));
//! circuit.add_gate(Gate::cx(&[(0, 1)]));
//! circuit.add_gate(Gate::mz(&[0]));
//! circuit.add_gate(Gate::mz(&[1]));
//!
//! // Execute symbolically (once!)
//! let mut sim = SymbolicSparseStab::new(2);
//! execute_hugr(&mut sim, &circuit).unwrap();
//!
//! // Sample efficiently (millions of shots)
//! let sampler = MeasurementSampler::new(sim.measurement_history());
//! let results = sampler.sample(1_000_000);
//!
//! // Results will show Bell state correlations: 00 or 11
//! assert_eq!(results.num_measurements(), 2);
//! ```
//!
//! [`MeasurementSampler`]: pecos_simulators::MeasurementSampler

use std::fmt;

use pecos_core::gate_type::GateType;
use pecos_core::{CliffordLowering, Gate, QubitId, try_lower_rotation_to_clifford};
use pecos_quantum::Circuit;
use pecos_simulators::SymbolicSparseStab;

/// Error type for symbolic circuit execution failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HugrExecutionError {
    /// Gate type is not supported by the stabilizer simulator.
    UnsupportedGate {
        gate_type: GateType,
        gate_index: usize,
    },
    /// Gate has an unexpected number of qubits.
    InvalidQubitCount {
        gate_type: GateType,
        gate_index: usize,
        expected: usize,
        actual: usize,
    },
    /// Qubit index is out of bounds for the simulator.
    QubitOutOfBounds {
        qubit: usize,
        gate_index: usize,
        num_qubits: usize,
    },
}

impl fmt::Display for HugrExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedGate {
                gate_type,
                gate_index,
            } => {
                write!(
                    f,
                    "Gate {gate_type} at index {gate_index} is not supported by stabilizer simulation"
                )
            }
            Self::InvalidQubitCount {
                gate_type,
                gate_index,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Gate {gate_type} at index {gate_index} expected {expected} qubits but got {actual}"
                )
            }
            Self::QubitOutOfBounds {
                qubit,
                gate_index,
                num_qubits,
            } => {
                write!(
                    f,
                    "Qubit {qubit} at gate index {gate_index} is out of bounds (simulator has {num_qubits} qubits)"
                )
            }
        }
    }
}

impl std::error::Error for HugrExecutionError {}

/// Execute a circuit through a symbolic stabilizer simulator.
///
/// This function walks the circuit in topological order and applies each gate
/// to the simulator. After execution, the simulator's measurement history
/// contains the symbolic dependencies for all measurements.
///
/// # Supported Gates
///
/// The following Clifford gates are supported:
/// - Single-qubit: I, X, Y, Z, H, SX, SY, SZ and their adjoints
/// - Two-qubit: CX, CY, CZ, SXX, SYY, SZZ and their adjoints
/// - Rotations at Clifford angles accepted by the shared lowering policy
/// - Measurements: Measure, `MeasureFree`
/// - Preparations: Prep, `QAlloc` (treated as reset to |0⟩)
///
/// # Unsupported Gates
///
/// The following gates will return an error:
/// - Rotations outside the shared Clifford lowering policy, T and Tdg
/// - Other operations without a supported named Clifford action
///
/// # Arguments
///
/// * `sim` - The symbolic stabilizer simulator to execute on
/// * `hugr` - The circuit to execute (anything implementing the `Circuit` trait)
///
/// # Returns
///
/// `Ok(())` if execution succeeded, or a [`HugrExecutionError`] if a gate
/// could not be executed.
///
/// # Errors
///
/// Returns [`HugrExecutionError::UnsupportedGate`] if the circuit contains non-Clifford gates.
/// Returns [`HugrExecutionError::InvalidQubitCount`] if a gate has wrong number of qubits.
/// Returns [`HugrExecutionError::QubitOutOfBounds`] if a qubit index exceeds simulator size.
///
/// # Example
///
/// ```rust
/// use pecos_simulators::SymbolicSparseStab;
/// use pecos_experimental::execute_hugr;
/// use pecos_quantum::{DagCircuit, Gate};
///
/// // Create a simple circuit
/// let mut circuit = DagCircuit::new();
/// circuit.add_gate(Gate::h(&[0]));
/// circuit.add_gate(Gate::mz(&[0]));
///
/// let mut sim = SymbolicSparseStab::new(1);
/// execute_hugr(&mut sim, &circuit).unwrap();
///
/// // Now sim.measurement_history() contains the symbolic dependencies
/// assert_eq!(sim.measurement_history().len(), 1);
/// ```
pub fn execute_hugr<C>(sim: &mut SymbolicSparseStab, hugr: &C) -> Result<(), HugrExecutionError>
where
    C: Circuit,
{
    let num_qubits = sim.num_qubits();

    for gate_view in hugr.iter_gates_topo() {
        let gate = gate_view.gate;
        let gate_idx = gate_view.index;

        // Validate qubit bounds
        for qubit in &gate.qubits {
            let q_idx = qubit.index();
            if q_idx >= num_qubits {
                return Err(HugrExecutionError::QubitOutOfBounds {
                    qubit: q_idx,
                    gate_index: gate_idx,
                    num_qubits,
                });
            }
        }

        match clifford_action(gate, gate_idx)? {
            CliffordLowering::Named(named) => {
                execute_named_gate(sim, named, &gate.qubits, gate_idx)?;
            }
            CliffordLowering::PerQubit(pauli) => {
                validate_qubit_count(gate.gate_type, gate_idx, 2, gate.qubits.len())?;
                for &qubit in &gate.qubits {
                    execute_named_gate(sim, pauli, &[qubit], gate_idx)?;
                }
            }
        }
    }

    Ok(())
}

/// Resolve rotations once, preserving the original gate in unsupported-gate errors.
pub(crate) fn clifford_action(
    gate: &Gate,
    gate_index: usize,
) -> Result<CliffordLowering, HugrExecutionError> {
    // Global phase is irrelevant to stabilizer simulation.
    if let Some(lowering) = try_lower_rotation_to_clifford(gate) {
        validate_qubit_count(
            gate.gate_type,
            gate_index,
            gate.gate_type.quantum_arity(),
            gate.qubits.len(),
        )?;
        Ok(lowering)
    } else if pecos_core::is_lowerable_rotation(gate.gate_type) {
        Err(HugrExecutionError::UnsupportedGate {
            gate_type: gate.gate_type,
            gate_index,
        })
    } else {
        validate_named_gate(gate.gate_type, gate_index, gate.qubits.len())?;
        Ok(CliffordLowering::Named(gate.gate_type))
    }
}

fn validate_named_gate(
    gate_type: GateType,
    gate_index: usize,
    qubit_count: usize,
) -> Result<(), HugrExecutionError> {
    match gate_type {
        GateType::I
        | GateType::QFree
        | GateType::Idle
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::TrackedPauliMeta => Ok(()),
        GateType::X
        | GateType::Y
        | GateType::Z
        | GateType::H
        | GateType::SZ
        | GateType::SZdg
        | GateType::SX
        | GateType::SXdg
        | GateType::SY
        | GateType::SYdg
        | GateType::CX
        | GateType::CY
        | GateType::CZ
        | GateType::SXX
        | GateType::SXXdg
        | GateType::SYY
        | GateType::SYYdg
        | GateType::SZZ
        | GateType::SZZdg
        | GateType::PZ
        | GateType::QAlloc
        | GateType::PX
        | GateType::MX
        | GateType::MZ
        | GateType::MeasureFree
        | GateType::MeasureLeaked
        | GateType::MPZ => validate_qubit_count(
            gate_type,
            gate_index,
            gate_type.quantum_arity(),
            qubit_count,
        ),
        _ => Err(HugrExecutionError::UnsupportedGate {
            gate_type,
            gate_index,
        }),
    }
}

fn execute_named_gate(
    sim: &mut SymbolicSparseStab,
    gate_type: GateType,
    qubits: &[QubitId],
    gate_idx: usize,
) -> Result<(), HugrExecutionError> {
    match gate_type {
        // Identity, timing, release and non-quantum metadata have no stabilizer action.
        GateType::I
        | GateType::QFree
        | GateType::Idle
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::TrackedPauliMeta => {}
        GateType::X
        | GateType::Y
        | GateType::Z
        | GateType::H
        | GateType::SZ
        | GateType::SZdg
        | GateType::SX
        | GateType::SXdg
        | GateType::SY
        | GateType::SYdg => {
            validate_qubit_count(gate_type, gate_idx, 1, qubits.len())?;
            execute_single_qubit(sim, gate_type, qubits[0].index());
        }
        GateType::CX
        | GateType::CY
        | GateType::CZ
        | GateType::SXX
        | GateType::SXXdg
        | GateType::SYY
        | GateType::SYYdg
        | GateType::SZZ
        | GateType::SZZdg => {
            validate_qubit_count(gate_type, gate_idx, 2, qubits.len())?;
            execute_two_qubit(sim, gate_type, qubits[0].index(), qubits[1].index());
        }
        GateType::PZ | GateType::QAlloc | GateType::PX => {
            validate_qubit_count(gate_type, gate_idx, 1, qubits.len())?;
            let q = qubits[0].index();
            sim.pz(q);
            if gate_type == GateType::PX {
                sim.h(&[q]);
            }
        }
        GateType::MX
        | GateType::MZ
        | GateType::MeasureFree
        | GateType::MeasureLeaked
        | GateType::MPZ => {
            validate_qubit_count(gate_type, gate_idx, 1, qubits.len())?;
            let q = qubits[0].index();
            if gate_type == GateType::MX {
                sim.h(&[q]);
            }
            sim.mz(&[q]);
            if gate_type == GateType::MPZ {
                // The unconditional reset implements the outcome-conditioned X correction.
                sim.pz(q);
            }
        }
        _ => {
            return Err(HugrExecutionError::UnsupportedGate {
                gate_type,
                gate_index: gate_idx,
            });
        }
    }
    Ok(())
}

fn execute_single_qubit(sim: &mut SymbolicSparseStab, gate: GateType, q: usize) {
    match gate {
        GateType::X => {
            sim.x(&[q]);
        }
        GateType::Y => {
            sim.y(&[q]);
        }
        GateType::Z => {
            sim.z(&[q]);
        }
        GateType::H => {
            sim.h(&[q]);
        }
        GateType::SZ => {
            sim.sz(&[q]);
        }
        GateType::SZdg => {
            sim.sz(&[q]).z(&[q]);
        }
        _ => execute_sqrt_pauli(sim, gate, q),
    }
}

fn execute_two_qubit(sim: &mut SymbolicSparseStab, gate: GateType, a: usize, b: usize) {
    match gate {
        GateType::CX => {
            sim.cx(&[(a, b)]);
        }
        GateType::CY => {
            sim.sz(&[b]).cx(&[(a, b)]).sz(&[b]).z(&[b]);
        }
        GateType::CZ => {
            sim.h(&[b]).cx(&[(a, b)]).h(&[b]);
        }
        _ => execute_sqrt_pair(sim, gate, a, b),
    }
}

fn execute_sqrt_pauli(sim: &mut SymbolicSparseStab, gate: GateType, q: usize) {
    match gate {
        GateType::SX | GateType::SXdg => {
            sim.h(&[q]).sz(&[q]);
            if gate == GateType::SXdg {
                sim.z(&[q]);
            }
            sim.h(&[q]);
        }
        GateType::SY => {
            sim.z(&[q]).h(&[q]);
        }
        GateType::SYdg => {
            sim.h(&[q]).z(&[q]);
        }
        _ => unreachable!("only single-qubit square roots are dispatched here"),
    }
}

fn execute_sqrt_pair(sim: &mut SymbolicSparseStab, gate: GateType, a: usize, b: usize) {
    // Conjugate a ZZ rotation into the requested Pauli basis.
    let xx = matches!(gate, GateType::SXX | GateType::SXXdg);
    let yy = matches!(gate, GateType::SYY | GateType::SYYdg);
    if yy {
        sim.sz(&[a, b]).sz(&[a, b]).sz(&[a, b]);
    }
    if xx || yy {
        sim.h(&[a, b]);
    }
    sim.cx(&[(a, b)]).sz(&[b]);
    if matches!(gate, GateType::SXXdg | GateType::SYYdg | GateType::SZZdg) {
        sim.z(&[b]);
    }
    sim.cx(&[(a, b)]);
    if xx || yy {
        sim.h(&[a, b]);
    }
    if yy {
        sim.sz(&[a, b]);
    }
}

/// Validate the qubit count for one gate application.
fn validate_qubit_count(
    gate_type: GateType,
    gate_index: usize,
    expected: usize,
    actual: usize,
) -> Result<(), HugrExecutionError> {
    if actual != expected {
        return Err(HugrExecutionError::InvalidQubitCount {
            gate_type,
            gate_index,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::DagCircuit;
    use pecos_simulators::SymbolicSparseStab;

    #[test]
    fn test_bell_state_circuit() {
        // Build Bell state circuit using DagCircuit builder interface
        let mut circuit = DagCircuit::new();
        circuit.h(&[0]);
        circuit.cx(&[(0, 1)]);
        circuit.mz(&[0]);
        circuit.mz(&[1]);

        // Execute
        let mut sim = SymbolicSparseStab::new(2);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        // Verify measurement history
        let history = sim.measurement_history();
        assert_eq!(history.len(), 2);

        // First measurement is non-deterministic
        assert!(!history[0].is_deterministic);

        // Second measurement is deterministic and depends on the first
        assert!(history[1].is_deterministic);
        assert_eq!(history[0].outcome, history[1].outcome);
    }

    #[test]
    fn test_ghz_state_circuit() {
        // Build 3-qubit GHZ state
        let mut circuit = DagCircuit::new();
        circuit.h(&[0]);
        circuit.cx(&[(0, 1)]);
        circuit.cx(&[(1, 2)]);
        circuit.mz(&[0]);
        circuit.mz(&[1]);
        circuit.mz(&[2]);

        // Execute
        let mut sim = SymbolicSparseStab::new(3);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        // Verify
        let history = sim.measurement_history();
        assert_eq!(history.len(), 3);

        // All measurements should have same outcome dependency
        assert!(!history[0].is_deterministic);
        assert!(history[1].is_deterministic);
        assert!(history[2].is_deterministic);
        assert_eq!(history[0].outcome, history[1].outcome);
        assert_eq!(history[0].outcome, history[2].outcome);
    }

    #[test]
    fn test_deterministic_circuit() {
        // Circuit with no superposition - all measurements deterministic
        // Only measure qubit 0 to avoid order ambiguity
        let mut circuit = DagCircuit::new();
        circuit.x(&[0]); // Flip to |1⟩
        circuit.mz(&[0]);

        let mut sim = SymbolicSparseStab::new(2);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        let history = sim.measurement_history();
        assert_eq!(history.len(), 1);

        // Deterministic with flip=true (was X'd)
        assert!(history[0].is_deterministic);
        assert!(history[0].flip);
    }

    #[test]
    fn test_deterministic_circuit_multiple() {
        // Test multiple independent measurements
        // Note: Order of independent measurements in history depends on topological order
        let mut circuit = DagCircuit::new();
        circuit.x(&[0]); // Flip qubit 0 to |1⟩
        circuit.mz(&[0]);
        circuit.mz(&[1]); // Qubit 1 stays |0⟩

        let mut sim = SymbolicSparseStab::new(2);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        let history = sim.measurement_history();
        assert_eq!(history.len(), 2);

        // Both deterministic
        assert!(history[0].is_deterministic);
        assert!(history[1].is_deterministic);

        // One has flip=true, one has flip=false (order may vary)
        let num_flipped = history.iter().filter(|m| m.flip).count();
        let num_not_flipped = history.iter().filter(|m| !m.flip).count();
        assert_eq!(num_flipped, 1);
        assert_eq!(num_not_flipped, 1);
    }

    #[test]
    fn test_cz_gate() {
        use pecos_core::{Gate, QubitId};

        let mut circuit = DagCircuit::new();
        circuit.h(&[0]);
        circuit.h(&[1]);
        circuit.add_gate(Gate::simple(GateType::CZ, vec![QubitId(0), QubitId(1)]));
        circuit.h(&[0]);
        circuit.h(&[1]);
        circuit.mz(&[0]);
        circuit.mz(&[1]);

        let mut sim = SymbolicSparseStab::new(2);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        // Should work without error
        assert_eq!(sim.measurement_history().len(), 2);
    }

    #[test]
    fn test_unsupported_gate_error() {
        use pecos_core::{Angle64, Gate};

        let mut circuit = DagCircuit::new();
        circuit.add_gate(Gate::rz(Angle64::from_turns(0.125), &[0])); // Non-Clifford RZ

        let mut sim = SymbolicSparseStab::new(1);
        let result = execute_hugr(&mut sim, &circuit);

        assert!(result.is_err());
        match result {
            Err(HugrExecutionError::UnsupportedGate { gate_type, .. }) => {
                assert_eq!(gate_type, GateType::RZ);
            }
            _ => panic!("Expected UnsupportedGate error"),
        }
    }

    #[test]
    fn test_qubit_out_of_bounds() {
        let mut circuit = DagCircuit::new();
        circuit.h(&[5]); // Qubit 5 doesn't exist in a 2-qubit sim

        let mut sim = SymbolicSparseStab::new(2);
        let result = execute_hugr(&mut sim, &circuit);

        assert!(result.is_err());
        match result {
            Err(HugrExecutionError::QubitOutOfBounds { qubit, .. }) => {
                assert_eq!(qubit, 5);
            }
            _ => panic!("Expected QubitOutOfBounds error"),
        }
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = DagCircuit::new();
        let mut sim = SymbolicSparseStab::new(2);

        execute_hugr(&mut sim, &circuit).expect("empty circuit should succeed");
        assert_eq!(sim.measurement_history().len(), 0);
    }

    #[test]
    fn test_repetition_code_syndrome() {
        // 3-qubit repetition code with syndrome extraction
        // Note: The order of measurements in history depends on topological order,
        // which may differ from the circuit building order.
        let mut circuit = DagCircuit::new();

        // Encode logical |+_L⟩
        circuit.h(&[0]);
        circuit.cx(&[(0, 1)]);
        circuit.cx(&[(0, 2)]);

        // Syndrome Z0Z1 via ancilla q3
        circuit.h(&[3]);
        circuit.cx(&[(0, 3)]);
        circuit.cx(&[(1, 3)]);
        circuit.h(&[3]);
        circuit.mz(&[3]); // S0

        // Syndrome Z1Z2 via ancilla q4
        circuit.h(&[4]);
        circuit.cx(&[(1, 4)]);
        circuit.cx(&[(2, 4)]);
        circuit.h(&[4]);
        circuit.mz(&[4]); // S1

        // Measure data qubits
        circuit.mz(&[0]);
        circuit.mz(&[1]);
        circuit.mz(&[2]);

        let mut sim = SymbolicSparseStab::new(5);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        let history = sim.measurement_history();
        assert_eq!(history.len(), 5);

        // Count deterministic and non-deterministic measurements
        let det = history.deterministic();
        let nondet = history.nondeterministic();

        // In a repetition code without errors:
        // - 2 syndrome measurements are deterministic with flip=false
        // - 1 data measurement is non-deterministic (random)
        // - 2 data measurements are deterministic (depend on the random one)
        assert_eq!(det.len(), 4, "Expected 4 deterministic measurements");
        assert_eq!(nondet.len(), 1, "Expected 1 non-deterministic measurement");

        // The syndrome measurements should have flip=false (no errors)
        // We can identify them as deterministic measurements with empty outcome
        let syndromes: Vec<_> = det
            .iter()
            .filter(|m| m.outcome.is_empty() && !m.flip)
            .collect();
        assert_eq!(
            syndromes.len(),
            2,
            "Expected 2 syndrome measurements with value 0"
        );

        // The dependent data measurements should have the same outcome as the random one
        let random_outcome = &nondet[0].outcome;
        let dependent_data: Vec<_> = det
            .iter()
            .filter(|m| !m.outcome.is_empty() && m.outcome == *random_outcome)
            .collect();
        assert_eq!(
            dependent_data.len(),
            2,
            "Expected 2 data measurements depending on the random one"
        );
    }
    fn assert_rotation_matches(rotation: &Gate, named: &[Gate]) {
        use crate::{DepolarizingNoiseModel, NoisyMeasurementHistoryBuilder};
        // Vary input and readout bases so phase and two-qubit propagation matter.
        for basis in 0..9 {
            let make_circuit = |gates: &[Gate]| {
                let mut circuit = DagCircuit::new();
                circuit.h(&[0]);
                circuit.sz(&[0]);
                circuit.h(&[1]);
                // A fault before the rotation must propagate through its lowered action.
                for gate in gates {
                    circuit.add_gate_auto_wire(gate.clone());
                }
                if basis % 3 != 0 {
                    circuit.h(&[0]);
                }
                if basis % 3 == 2 {
                    circuit.sz(&[0]);
                    circuit.h(&[0]);
                }
                if basis / 3 != 0 {
                    circuit.h(&[1]);
                }
                if basis / 3 == 2 {
                    circuit.sz(&[1]);
                    circuit.h(&[1]);
                }
                // Join the readout paths so both DAGs schedule measurements identically.
                circuit.cx(&[(0, 1)]);
                circuit.mz(&[0]);
                circuit.mz(&[1]);
                circuit
            };
            let original = make_circuit(std::slice::from_ref(rotation));
            let lowered = make_circuit(named);
            let mut original_sim = SymbolicSparseStab::new(2);
            let mut lowered_sim = SymbolicSparseStab::new(2);
            execute_hugr(&mut original_sim, &original).unwrap();
            execute_hugr(&mut lowered_sim, &lowered).unwrap();
            assert_eq!(
                original_sim.measurement_history().format_all(),
                lowered_sim.measurement_history().format_all(),
                "{rotation:?}"
            );
            // Turn off noise at the rotation's own arity when comparing decompositions.
            let noise = if rotation.qubits.len() == 2 {
                DepolarizingNoiseModel::new(0.03, 0.0, 0.0, 0.0)
            } else {
                DepolarizingNoiseModel::new(0.03, 0.02, 0.0, 0.0)
            };
            if named.len() == 1 && named[0].gate_type != GateType::I {
                let builder = NoisyMeasurementHistoryBuilder::new().with_noise_model(noise);
                let a = builder
                    .build_from_circuit(&original, original_sim.measurement_history())
                    .unwrap();
                let b = builder
                    .build_from_circuit(&lowered, lowered_sim.measurement_history())
                    .unwrap();
                assert_eq!(
                    a.measurements(),
                    b.measurements(),
                    "fault propagation for {rotation:?}"
                );
                assert_eq!(
                    a.faults().iter().map(|f| f.probability).collect::<Vec<_>>(),
                    b.faults().iter().map(|f| f.probability).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn clifford_rotations_match_named_gates() {
        use GateType::{
            I, RX, RXX, RY, RYY, RZ, RZZ, SX, SXX, SXXdg, SXdg, SY, SYY, SYYdg, SYdg, SZ, SZZ,
            SZZdg, SZdg, X, Y, Z,
        };
        use pecos_core::Angle64;
        for (rotation, quarter, half, inverse, two_qubit) in [
            (RX, SX, X, SXdg, false),
            (RY, SY, Y, SYdg, false),
            (RZ, SZ, Z, SZdg, false),
            (RXX, SXX, X, SXXdg, true),
            (RYY, SYY, Y, SYYdg, true),
            (RZZ, SZZ, Z, SZZdg, true),
        ] {
            for (angle, expected) in [
                (Angle64::ZERO, I),
                (Angle64::QUARTER_TURN, quarter),
                (Angle64::HALF_TURN, half),
                (Angle64::THREE_QUARTERS_TURN, inverse),
            ] {
                let qubits = if two_qubit {
                    vec![QubitId(0), QubitId(1)]
                } else {
                    vec![QubitId(0)]
                };
                let gate = Gate::with_angles(rotation, vec![angle], qubits.clone());
                let named: Vec<Gate> = if two_qubit && angle == Angle64::HALF_TURN {
                    qubits
                        .iter()
                        .map(|&q| Gate::simple(expected, vec![q]))
                        .collect()
                } else {
                    vec![Gate::simple(expected, qubits)]
                };
                assert_rotation_matches(&gate, &named);
            }
        }
        assert_rotation_matches(
            &Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
            &[Gate::sx(&[0])],
        );
        assert_rotation_matches(
            &Gate::rxyxy2q(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN, &[(0, 1)]),
            &[Gate::syy(&[(0, 1)])],
        );
        assert_rotation_matches(
            &Gate::rxyxy2q(Angle64::HALF_TURN, Angle64::ZERO, &[(0, 1)]),
            &[Gate::x(&[0]), Gate::x(&[1])],
        );
        assert_rotation_matches(
            &Gate::u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]),
            &[Gate::sz(&[0])],
        );
    }

    #[test]
    fn rxy1q_snaps_both_angles() {
        use pecos_core::Angle64;
        assert_rotation_matches(
            &Gate::rxy1q(
                Angle64::from_turns(0.25 + 1e-12),
                Angle64::from_turns(0.5 - 1e-12),
                &[0],
            ),
            &[Gate::sxdg(&[0])],
        );
    }

    #[test]
    fn non_clifford_rotations_preserve_original_error() {
        use pecos_core::Angle64;
        let angle = Angle64::from_turns(0.125);
        for gate in [
            Gate::rx(angle, &[0]),
            Gate::ry(angle, &[0]),
            Gate::rz(angle, &[0]),
            Gate::rxx(angle, &[(0, 1)]),
            Gate::ryy(angle, &[(0, 1)]),
            Gate::rzz(angle, &[(0, 1)]),
            Gate::rxy1q(angle, Angle64::ZERO, &[0]),
            Gate::rxy1q(
                Angle64::QUARTER_TURN,
                Angle64::from_turns(0.25 + 1e-7),
                &[0],
            ),
            Gate::rxyxy2q(angle, Angle64::ZERO, &[(0, 1)]),
            Gate::u(Angle64::ZERO, Angle64::ZERO, angle, &[0]),
        ] {
            let mut circuit = DagCircuit::new();
            circuit.add_gate_auto_wire(gate.clone());
            assert!(
                matches!(execute_hugr(&mut SymbolicSparseStab::new(2), &circuit),
                Err(HugrExecutionError::UnsupportedGate { gate_type, .. }) if gate_type == gate.gate_type)
            );
        }
    }
}
