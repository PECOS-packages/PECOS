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

//! HUGR execution for symbolic stabilizer simulation.
//!
//! **⚠️ EXPERIMENTAL: This API is unstable and may change without notice.**
//!
//! This module provides functionality to execute HUGR circuits (via [`SimpleHugr`])
//! through the [`SymbolicSparseStab`] simulator, enabling efficient sampling from
//! the resulting measurement history.
//!
//! # Overview
//!
//! The workflow is:
//! 1. Compile Guppy code to HUGR bytes
//! 2. Convert to [`SimpleHugr`] (validates no control flow)
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
//! use pecos_qsim::{SymbolicSparseStab, MeasurementSampler};
//! use pecos_experimental::execute_hugr;
//! use pecos_quantum::{DagCircuit, Gate};
//!
//! // Create a Bell state circuit
//! let mut circuit = DagCircuit::new();
//! circuit.add_gate(Gate::h(&[0]));
//! circuit.add_gate(Gate::cx(&[(0, 1)]));
//! circuit.add_gate(Gate::measure(&[0]));
//! circuit.add_gate(Gate::measure(&[1]));
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
//! [`SimpleHugr`]: pecos_quantum::hugr_convert::SimpleHugr
//! [`MeasurementSampler`]: pecos_qsim::MeasurementSampler

use std::fmt;

use pecos_core::gate_type::GateType;
use pecos_qsim::SymbolicSparseStab;
use pecos_quantum::Circuit;

/// Error type for HUGR execution failures.
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

/// Execute a HUGR/circuit through a symbolic stabilizer simulator.
///
/// This function walks the circuit in topological order and applies each gate
/// to the simulator. After execution, the simulator's measurement history
/// contains the symbolic dependencies for all measurements.
///
/// # Supported Gates
///
/// The following Clifford gates are supported:
/// - Single-qubit: I, X, Y, Z, H, SZ (S gate), `SZdg` (S†)
/// - Two-qubit: CX, CY, CZ
/// - Measurements: Measure, `MeasureFree`
/// - Preparations: Prep, `QAlloc` (treated as reset to |0⟩)
///
/// # Unsupported Gates
///
/// The following gates will return an error:
/// - Rotation gates: RX, RY, RZ, RZZ, T, Tdg, U, R1XY
/// - Other: SZZ, `SZZdg`
///
/// # Arguments
///
/// * `sim` - The symbolic stabilizer simulator to execute on
/// * `hugr` - The HUGR/circuit to execute (anything implementing the `Circuit` trait)
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
/// use pecos_qsim::SymbolicSparseStab;
/// use pecos_experimental::execute_hugr;
/// use pecos_quantum::{DagCircuit, Gate};
///
/// // Create a simple circuit
/// let mut circuit = DagCircuit::new();
/// circuit.add_gate(Gate::h(&[0]));
/// circuit.add_gate(Gate::measure(&[0]));
///
/// let mut sim = SymbolicSparseStab::new(1);
/// execute_hugr(&mut sim, &circuit).unwrap();
///
/// // Now sim.measurement_history() contains the symbolic dependencies
/// assert_eq!(sim.measurement_history().len(), 1);
/// ```
#[allow(clippy::too_many_lines)]
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

        match gate.gate_type {
            // No-op gates: identity, prep/alloc (qubits start in |0⟩), dealloc, idle, crosstalk
            GateType::I
            | GateType::Prep
            | GateType::QAlloc
            | GateType::QFree
            | GateType::Idle
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload => {}

            // Single-qubit Clifford gates
            GateType::X => {
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.x(q);
            }
            GateType::Y => {
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.y(q);
            }
            GateType::Z => {
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.z(q);
            }
            GateType::H => {
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.h(q);
            }
            GateType::SZ => {
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.sz(q);
            }
            GateType::SZdg => {
                // S† = S^3, so apply S three times
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.sz(q);
                sim.sz(q);
                sim.sz(q);
            }

            // Two-qubit Clifford gates
            GateType::CX => {
                validate_qubit_count(gate.gate_type, gate_idx, 2, gate.qubits.len())?;
                let q1 = gate.qubits[0].index();
                let q2 = gate.qubits[1].index();
                sim.cx(q1, q2);
            }
            GateType::CY => {
                // CY = (I ⊗ S†) CX (I ⊗ S)
                validate_qubit_count(gate.gate_type, gate_idx, 2, gate.qubits.len())?;
                let q1 = gate.qubits[0].index();
                let q2 = gate.qubits[1].index();
                // S on target
                sim.sz(q2);
                // CX
                sim.cx(q1, q2);
                // S† on target (= S^3)
                sim.sz(q2);
                sim.sz(q2);
                sim.sz(q2);
            }
            GateType::CZ => {
                // CZ = (I ⊗ H) CX (I ⊗ H)
                validate_qubit_count(gate.gate_type, gate_idx, 2, gate.qubits.len())?;
                let q1 = gate.qubits[0].index();
                let q2 = gate.qubits[1].index();
                sim.h(q2);
                sim.cx(q1, q2);
                sim.h(q2);
            }

            // Measurements (including leaked measurement, treated as regular)
            GateType::Measure | GateType::MeasureFree | GateType::MeasureLeaked => {
                validate_qubit_count(gate.gate_type, gate_idx, 1, gate.qubits.len())?;
                let q = gate.qubits[0].index();
                sim.mz(q);
            }

            // Unsupported gates (non-Clifford)
            GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::T
            | GateType::Tdg
            | GateType::U
            | GateType::R1XY
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SWAP
            | GateType::CRZ
            | GateType::CH
            | GateType::CCX
            | GateType::Custom => {
                return Err(HugrExecutionError::UnsupportedGate {
                    gate_type: gate.gate_type,
                    gate_index: gate_idx,
                });
            }
        }
    }

    Ok(())
}

/// Helper function to validate qubit count.
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
    use pecos_qsim::SymbolicSparseStab;
    use pecos_quantum::DagCircuit;

    #[test]
    fn test_bell_state_circuit() {
        // Build Bell state circuit using DagCircuit builder interface
        let mut circuit = DagCircuit::new();
        circuit.h(0);
        circuit.cx(0, 1);
        circuit.mz(0);
        circuit.mz(1);

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
        circuit.h(0);
        circuit.cx(0, 1);
        circuit.cx(1, 2);
        circuit.mz(0);
        circuit.mz(1);
        circuit.mz(2);

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
        circuit.x(0); // Flip to |1⟩
        circuit.mz(0);

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
        circuit.x(0); // Flip qubit 0 to |1⟩
        circuit.mz(0);
        circuit.mz(1); // Qubit 1 stays |0⟩

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
        circuit.h(0);
        circuit.h(1);
        circuit.add_gate(Gate::simple(GateType::CZ, vec![QubitId(0), QubitId(1)]));
        circuit.h(0);
        circuit.h(1);
        circuit.mz(0);
        circuit.mz(1);

        let mut sim = SymbolicSparseStab::new(2);
        execute_hugr(&mut sim, &circuit).expect("execution failed");

        // Should work without error
        assert_eq!(sim.measurement_history().len(), 2);
    }

    #[test]
    fn test_unsupported_gate_error() {
        use pecos_core::{Angle64, Gate};

        let mut circuit = DagCircuit::new();
        circuit.add_gate(Gate::rz(Angle64::from_turns(0.25), &[0])); // RZ is not Clifford

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
        circuit.h(5); // Qubit 5 doesn't exist in a 2-qubit sim

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
        circuit.h(0);
        circuit.cx(0, 1);
        circuit.cx(0, 2);

        // Syndrome Z0Z1 via ancilla q3
        circuit.h(3);
        circuit.cx(0, 3);
        circuit.cx(1, 3);
        circuit.h(3);
        circuit.mz(3); // S0

        // Syndrome Z1Z2 via ancilla q4
        circuit.h(4);
        circuit.cx(1, 4);
        circuit.cx(2, 4);
        circuit.h(4);
        circuit.mz(4); // S1

        // Measure data qubits
        circuit.mz(0);
        circuit.mz(1);
        circuit.mz(2);

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
}
