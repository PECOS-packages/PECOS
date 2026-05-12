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

/// Base trait for quantum simulators.
///
/// This trait defines the minimal interface that all quantum simulators must implement,
/// whether they track quantum states, observables, or other quantum mechanical properties.
pub trait QuantumSimulator {
    /// Resets the simulator to its initial configuration.
    ///
    /// The exact meaning of reset depends on the simulator type:
    /// - For state vector simulators: resets to |0⟩ state
    /// - For observable propagators: clears tracked Paulis
    /// - For stabilizer simulators: resets to trivial stabilizer group
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    ///
    /// # Examples
    /// ```rust
    /// use pecos_core::{qid, QubitId};
    /// use pecos_simulators::{QuantumSimulator, CliffordGateable, SparseStab};
    ///
    /// let mut sim = SparseStab::new(2);
    /// sim.x(&qid(0))
    ///    .cx(&[(QubitId(0), QubitId(1))])
    ///    .reset()  // Return to initial configuration
    ///    .z(&qid(1));    // Can continue chaining methods
    /// ```
    fn reset(&mut self) -> &mut Self;

    /// Returns the number of qubits in the simulator.
    fn num_qubits(&self) -> usize;
}
