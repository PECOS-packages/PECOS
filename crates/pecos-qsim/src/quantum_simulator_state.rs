// Copyright 2024 The PECOS Developers
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

// Base trait for quantum simulator state management
pub trait QuantumSimulatorState {
    /// Returns the number of qubits in the system
    ///
    /// # Returns
    /// * `usize` - The total number of qubits this simulator is configured to handle
    fn num_qubits(&self) -> usize;

    /// Resets all qubits in the system to the |0⟩ state
    ///
    /// This is the standard computational basis state where all qubits are in their
    /// ground state. Any entanglement or quantum correlations between qubits are removed.
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining (builder pattern)
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{QuantumSimulatorState, CliffordGateable, StdSparseStab};
    ///
    /// let mut sim = StdSparseStab::new(2);
    /// sim.h(0)
    ///    .cx(0, 1)
    ///    .reset()  // Return to |00⟩ state
    ///    .h(1);    // Can continue chaining methods
    /// ```
    fn reset(&mut self) -> &mut Self;
}
