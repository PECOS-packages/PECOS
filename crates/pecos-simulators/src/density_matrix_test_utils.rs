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

//! Test utilities for verifying density matrix simulator implementations.
//!
//! This module provides a marker trait and test suite for any density matrix
//! simulator that implements [`CliffordGateable`], [`ArbitraryRotationGateable`],
//! and [`QuantumSimulator`].
//!
//! # Example
//!
//! ```ignore
//! use pecos_simulators::density_matrix_test_utils::*;
//! use pecos_simulators::DensityMatrix;
//!
//! #[test]
//! fn test_my_simulator() {
//!     let mut sim = DensityMatrix::with_seed(4, 42);
//!     run_density_matrix_test_suite(&mut sim, 4);
//! }
//! ```

#![allow(clippy::missing_panics_doc)]

use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};

// ============================================================================
// Density Matrix Simulator Marker Trait
// ============================================================================

/// Marker trait for density matrix simulators.
///
/// Implementing this trait indicates that a simulator:
/// - Implements all Clifford gates via [`CliffordGateable`]
/// - Supports arbitrary rotation gates via [`ArbitraryRotationGateable`]
/// - Supports basic simulator operations via [`QuantumSimulator`]
/// - Can be cloned for testing
/// - Can be constructed with a seed for reproducible tests
///
/// Simulators implementing this trait can use the [`density_matrix_test_suite!`] macro
/// to automatically generate a comprehensive test suite.
///
/// # Example
///
/// ```ignore
/// use pecos_simulators::density_matrix_test_utils::{DensityMatrixSimulator, density_matrix_test_suite};
///
/// // In your test module:
/// density_matrix_test_suite!(DensityMatrix, 4);
/// ```
pub trait DensityMatrixSimulator:
    CliffordGateable + ArbitraryRotationGateable + QuantumSimulator + Clone + Sized
{
    /// Create a new simulator with the given number of qubits and RNG seed.
    fn with_seed(num_qubits: usize, seed: u64) -> Self;
}

/// Generates a comprehensive test suite for a density matrix simulator.
///
/// This macro creates test functions that verify correct implementation of
/// all Clifford gates, rotation gates, measurement behavior, and gate identities.
///
/// # Arguments
///
/// * `$sim_type` - The type implementing [`DensityMatrixSimulator`]
/// * `$num_qubits` - Number of qubits to use for testing (default: 4)
///
/// # Example
///
/// ```ignore
/// use pecos_simulators::density_matrix_test_suite;
/// use pecos_simulators::DensityMatrix;
///
/// density_matrix_test_suite!(DensityMatrix);
///
/// // Or specify a custom qubit count
/// density_matrix_test_suite!(DensityMatrix, 3);
/// ```
///
/// # Generated Tests
///
/// The macro generates the following tests:
/// - `test_<type>_clifford_suite` - Shared Clifford gate tests
/// - `test_<type>_rotation_suite` - Shared rotation gate tests
/// - `test_<type>_full_suite` - Full suite (Clifford + rotation)
#[macro_export]
macro_rules! density_matrix_test_suite {
    ($sim_type:ty) => {
        $crate::density_matrix_test_suite!($sim_type, 4);
    };
    ($sim_type:ty, $num_qubits:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _clifford_suite>]() {
                use $crate::density_matrix_test_utils::{run_density_matrix_clifford_suite, DensityMatrixSimulator};
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_density_matrix_clifford_suite(&mut sim, $num_qubits);
            }

            #[test]
            fn [<test_ $sim_type:snake _rotation_suite>]() {
                use $crate::density_matrix_test_utils::{run_density_matrix_rotation_suite, DensityMatrixSimulator};
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_density_matrix_rotation_suite(&mut sim, $num_qubits);
            }

            #[test]
            fn [<test_ $sim_type:snake _full_suite>]() {
                use $crate::density_matrix_test_utils::{run_density_matrix_test_suite, DensityMatrixSimulator};
                let mut sim = <$sim_type>::with_seed($num_qubits, 42);
                run_density_matrix_test_suite(&mut sim, $num_qubits);
            }
        }
    };
}

// ============================================================================
// Test Suite Functions
// ============================================================================

/// Run the shared Clifford gate tests on a density matrix simulator.
pub fn run_density_matrix_clifford_suite<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
    num_qubits: usize,
) {
    crate::clifford_test_utils::run_clifford_gate_tests(sim, num_qubits);
}

/// Run the shared rotation gate tests on a density matrix simulator.
pub fn run_density_matrix_rotation_suite<
    S: CliffordGateable + ArbitraryRotationGateable + QuantumSimulator,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    crate::rotation_test_utils::run_rotation_gate_tests(sim, num_qubits);
}

/// Run the full density matrix test suite on a simulator.
///
/// This runs both the shared Clifford gate tests and the shared rotation gate tests.
pub fn run_density_matrix_test_suite<
    S: CliffordGateable + ArbitraryRotationGateable + QuantumSimulator,
>(
    sim: &mut S,
    num_qubits: usize,
) {
    run_density_matrix_clifford_suite(sim, num_qubits);
    run_density_matrix_rotation_suite(sim, num_qubits);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DensityMatrix;

    #[test]
    fn test_density_matrix_clifford_suite() {
        let mut sim = DensityMatrix::with_seed(4, 42);
        run_density_matrix_clifford_suite(&mut sim, 4);
    }

    #[test]
    fn test_density_matrix_rotation_suite() {
        let mut sim = DensityMatrix::with_seed(4, 42);
        run_density_matrix_rotation_suite(&mut sim, 4);
    }

    #[test]
    fn test_density_matrix_full_suite() {
        let mut sim = DensityMatrix::with_seed(4, 42);
        run_density_matrix_test_suite(&mut sim, 4);
    }
}
