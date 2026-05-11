// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Quantum computing primitives for PECOS.
//!
//! This crate provides quantum computing data structures, including:
//! - [`DagCircuit`] for representing quantum circuits as directed acyclic graphs
//! - [`TickCircuit`] for representing quantum circuits as sequences of parallel time slices
//!
//! # Example - `DagCircuit`
//!
//! ```
//! use pecos_quantum::{DagCircuit, Gate, QubitId};
//!
//! let mut circuit = DagCircuit::new();
//!
//! // Add gates
//! let h = circuit.add_gate(Gate::h(&[0]));
//! let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
//!
//! // Connect H to CX with a qubit wire
//! circuit.connect(h, cx, QubitId::from(0)).unwrap();
//!
//! // Get circuit properties
//! assert_eq!(circuit.gate_count(), 2);
//! assert_eq!(circuit.wire_count(), 1);
//! ```
//!
//! # Example - `TickCircuit`
//!
//! ```
//! use pecos_quantum::TickCircuit;
//!
//! let mut circuit = TickCircuit::new();
//!
//! // Each tick() returns a handle for adding gates
//! // Regular gates chain, but preps/measurements break the chain
//! circuit.tick().pz(&[0]);              // Tick 0: Prepare q0 (breaks chain)
//! circuit.tick().pz(&[1]);              // Tick 1: Prepare q1 (breaks chain)
//! circuit.tick().h(&[0]).x(&[1]);       // Tick 2: H and X chain together
//! circuit.tick().cx(&[(0, 1)]);         // Tick 3: CNOT
//! circuit.tick().mz(&[0]);              // Tick 4: Measure q0 (breaks chain)
//! circuit.tick().mz(&[1]);              // Tick 5: Measure q1 (breaks chain)
//!
//! assert_eq!(circuit.num_ticks(), 6);
//!
//! // All methods accept slices for bulk operations:
//! let mut circuit2 = TickCircuit::new();
//! circuit2.tick().pz(&[0, 1, 2, 3]);    // Prep multiple qubits
//! circuit2.tick().h(&[0, 1, 2, 3]);     // H on multiple qubits
//! circuit2.tick().cx(&[(0, 1), (2, 3)]); // Multiple CX gates
//! circuit2.tick().mz(&[0, 1, 2, 3]);    // Measure multiple qubits
//! ```

pub mod channel;
mod circuit;
mod circuit_display;
mod dag_circuit;
pub mod diamond_norm;
pub mod measures;
pub mod pass;
pub mod pauli_group;
pub mod pauli_sequence;
pub mod pauli_set;
pub mod stabilizer_group;
mod tick_circuit;
pub mod unitary_matrix;

#[cfg(feature = "hugr")]
pub mod hugr_convert;

pub use circuit::{Circuit, CircuitMut, GateHandle, GateView};
pub use dag_circuit::{
    AnnotationKind, Attribute, DagCircuit, DagTraversalIndex, MeasRef, PauliAnnotation,
    TraversalWorkBuffers,
};
pub use tick_circuit::{
    CustomGateError, GateSignatureMismatchError, QubitConflictError, Tick, TickCircuit,
    TickGateError, TickHandle, TickMeasRef, TickMeasureHandle, TickPrepHandle,
};

// Re-export commonly used types from dependencies
pub use pecos_core::gate_type::GateType;
pub use pecos_core::{ClassicalBitId, Gate, QubitId, TimeScale, TimeUnits};
pub use pecos_num::dag::DagWouldCycleError;

// Concrete channel representation types
pub use channel::{
    ChannelError, ChiMatrix, ChoiMatrix, DiagonalPtm, KrausOps, MatrixUnitTomographyInput,
    PauliChannel, PauliSum, ProcessTomographyDesign, Ptm, PtmBasisOrder, Stinespring, SuperOp,
    basis_bitmask, basis_digit_to_pauli, basis_element, basis_index, basis_label, bitmask_label,
    matrix_unit_basis, partial_trace, pauli_basis_len, pauli_string_to_bitmask,
    pauli_to_basis_digit, random_1q_clifford, random_2q_clifford, random_clifford,
    random_density_matrix, random_density_matrix_with_rank, random_pauli, random_quantum_channel,
};
pub use diamond_norm::{
    DiamondNormError, choi_to_watrous_row_transpose, hermitian_to_real_symmetric,
    hermitian_to_real_symmetric_with_tolerance, pauli_channel_diamond_distance,
    pauli_channel_diamond_norm, scaled_psd_triangle_len, smat_real_symmetric, svec_real_symmetric,
    svec_real_symmetric_with_tolerance,
};
pub use measures::{
    DensityMatrixPartialTrace, MeasureError, SchmidtTerm, average_gate_fidelity, concurrence,
    entanglement_of_formation, entropy, entropy_with_base, gate_error, hellinger_distance,
    hellinger_fidelity, logarithmic_negativity, mutual_information, negativity,
    partial_trace_qubits, partial_trace_subsystems, process_fidelity, purity,
    schmidt_decomposition, shannon_entropy, state_fidelity, state_fidelity_with_density_matrix,
};

// Re-export operator matrix types for convenient method-style matrix conversion
pub use unitary_matrix::{ToMatrix, UnitaryMatrix, UnitaryMatrixError, random_unitary};

// Pauli collection and stabilizer group types
pub use pauli_group::{PauliGroup, PauliGroupError};
pub use pauli_sequence::{F2Matrix, PauliSequence};
pub use pauli_set::PauliSet;
pub use stabilizer_group::{PauliStabilizerGroup, PauliStabilizerGroupError};

// Re-export HUGR types when the feature is enabled
#[cfg(feature = "hugr")]
pub use tket::hugr::Hugr;
