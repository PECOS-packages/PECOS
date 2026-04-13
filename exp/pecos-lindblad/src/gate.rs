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

use num_complex::Complex64;

use crate::lindbladian::Lindbladian;
use crate::matrix::{self, Matrix};

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

    /// 1-qubit arbitrary-angle X rotation: `X_theta = exp(-i theta/2 X)`.
    /// Parameterized by drive frequency `omega_x` and rotation angle
    /// `theta`; gate duration is `theta / omega_x`.
    pub fn x_theta(omega_x: f64, theta: f64, noise: Lindbladian) -> Self {
        assert!(omega_x > 0.0, "omega_x must be positive");
        assert_eq!(noise.d, 2, "x_theta is 1-qubit");
        let d = 2;
        // H_g = (omega_x / 2) * X
        let h_g: Matrix =
            matrix::scale(&matrix::pauli_1q(crate::basis::Pauli1::X), Complex64::new(omega_x / 2.0, 0.0));
        let ideal = Lindbladian::new(d, h_g, Vec::new());
        let tau_g = theta / omega_x;
        Self {
            label: format!("X_{{{:.4}}}", theta),
            num_qubits: 1,
            ideal,
            noise,
            tau_g,
        }
    }
}
