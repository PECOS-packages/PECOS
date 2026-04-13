// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Gate type: ideal Hamiltonian + noise Lindbladian + duration.

use crate::lindbladian::Lindbladian;

/// A physical gate with its ideal rotation, noise model, and duration.
#[derive(Clone, Debug)]
pub struct Gate {
    pub label: String,
    pub num_qubits: usize,
    /// Noise-free part of the dynamics. Sets the interaction frame.
    pub ideal: Lindbladian,
    /// Noise (coherent + incoherent) applied during the gate.
    pub noise: Lindbladian,
    /// Gate duration in the same time units as `gamma_j` of the noise.
    pub tau_g: f64,
}

impl Gate {
    /// Identity gate (no ideal Hamiltonian) with a given noise Lindbladian
    /// and duration.
    pub fn identity(num_qubits: usize, noise: Lindbladian, tau_g: f64) -> Self {
        let d = 1 << num_qubits;
        assert_eq!(noise.d, d, "noise dim mismatch");
        Self {
            label: "I".to_string(),
            num_qubits,
            ideal: Lindbladian::zero(d),
            noise,
            tau_g,
        }
    }
}
