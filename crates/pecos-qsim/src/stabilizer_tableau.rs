// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

//! Trait for stabilizer tableau-based quantum simulators.
//!
//! This trait provides a common interface for simulators that use the stabilizer/destabilizer
//! tableau formalism, allowing them to share functionality like tableau printing and manipulation.

use crate::QuantumSimulator;

/// A trait for quantum simulators that use the stabilizer tableau formalism.
///
/// This trait extends `QuantumSimulator` with methods specific to stabilizer-based
/// simulators, including tableau access and manipulation.
///
/// # Examples
/// ```rust
/// use pecos_qsim::{StabilizerTableauSimulator, CliffordGateable, StdSparseStab};
///
/// let mut sim = StdSparseStab::new(2);
/// sim.h(0).cx(0, 1);  // Create Bell state
///
/// // Print the stabilizer tableau
/// println!("{}", sim.stab_tableau());
/// ```
pub trait StabilizerTableauSimulator: QuantumSimulator {
    /// Returns a string representation of the stabilizer tableau.
    ///
    /// The tableau format shows each stabilizer generator as a Pauli string
    /// with its phase (+, -, i, or -i).
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{StabilizerTableauSimulator, StdSparseStab};
    ///
    /// let sim = StdSparseStab::new(2);
    /// let tableau = sim.stab_tableau();
    /// assert!(tableau.contains("+ZI"));
    /// assert!(tableau.contains("+IZ"));
    /// ```
    fn stab_tableau(&self) -> String;

    /// Returns a string representation of the destabilizer tableau.
    ///
    /// The tableau format shows each destabilizer generator as a Pauli string
    /// with its phase (+, -, i, or -i).
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{StabilizerTableauSimulator, StdSparseStab};
    ///
    /// let sim = StdSparseStab::new(2);
    /// let tableau = sim.destab_tableau();
    /// assert!(tableau.contains("+XI"));
    /// assert!(tableau.contains("+IX"));
    /// ```
    fn destab_tableau(&self) -> String;

    /// Returns the combined stabilizer and destabilizer tableaux.
    ///
    /// This shows the full canonical form of the stabilizer state with both
    /// stabilizers and destabilizers.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{StabilizerTableauSimulator, StdSparseStab};
    ///
    /// let sim = StdSparseStab::new(1);
    /// let full = sim.full_tableau();
    /// assert!(full.contains("Destabilizers:"));
    /// assert!(full.contains("Stabilizers:"));
    /// ```
    fn full_tableau(&self) -> String {
        format!(
            "Destabilizers:\n{}\nStabilizers:\n{}",
            self.destab_tableau(),
            self.stab_tableau()
        )
    }

    /// Checks if a given Pauli operator commutes with all stabilizers.
    ///
    /// This can be used to verify if an operator is in the stabilizer group.
    ///
    /// # Arguments
    /// * `pauli_string` - A string representation of a Pauli operator (e.g., "XIZ")
    ///
    /// # Returns
    /// `true` if the operator commutes with all stabilizers, `false` otherwise.
    fn commutes_with_stabilizers(&self, pauli_string: &str) -> bool {
        // Default implementation - derived types should override for efficiency
        let _ = pauli_string;
        unimplemented!("commutes_with_stabilizers not yet implemented")
    }

    /// Returns the number of qubits in the simulator.
    ///
    /// This method should be implemented by each simulator type to return
    /// the number of qubits being simulated.
    fn num_qubits(&self) -> usize;
}
