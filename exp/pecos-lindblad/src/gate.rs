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
    /// Construct a gate from an arbitrary ideal Hamiltonian `H_g`, a noise
    /// [`Lindbladian`], and a duration `tau_g`. Provides a general escape
    /// hatch for gate types beyond the named constructors (e.g. iSWAP,
    /// XX_theta, arbitrary SU(4)).
    ///
    /// The ideal Hamiltonian is passed as a `d x d` matrix where
    /// `d = 2^num_qubits`. It must be Hermitian (caller's responsibility;
    /// [`matrix::expm`] assumes this for unitarity).
    pub fn from_hamiltonian(
        label: impl Into<String>,
        num_qubits: usize,
        ideal_hamiltonian: Matrix,
        noise: Lindbladian,
        tau_g: f64,
    ) -> Self {
        let d = 1usize << num_qubits;
        assert_eq!(ideal_hamiltonian.len(), d * d, "ideal H wrong shape");
        assert_eq!(noise.d, d, "noise dim mismatch");
        assert!(tau_g >= 0.0, "tau_g must be non-negative, got {}", tau_g);
        // Lindbladian::new checks Hermiticity of its Hamiltonian input.
        let ideal = Lindbladian::new(d, ideal_hamiltonian, Vec::new());
        Self {
            label: label.into(),
            num_qubits,
            ideal,
            noise,
            tau_g,
        }
    }

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
        let h_g: Matrix = matrix::scale(
            &matrix::pauli_1q(crate::basis::Pauli1::X),
            Complex64::new(omega_x / 2.0, 0.0),
        );
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

    /// 3-qubit `CX_theta ⊗ I` gate with coherent IZZ crosstalk between
    /// target (qubit 1) and spectator (qubit 2). `H_g = (omega/2)(IXI - ZXI)`,
    /// `H_delta = (delta/2) IZZ`, `tau_g = theta/omega`.
    ///
    /// The spectator qubit (q2) is untouched by the ideal gate but
    /// experiences a `ZZ` interaction with the target. This is the 3Q
    /// crosstalk case from arXiv:2502.03462 eqs. 1007-1011, the only
    /// non-trivial 3Q case in the paper.
    ///
    /// Noise on this gate is **purely coherent** (zero c_ops) -- use
    /// [`crate::synthesize_exact_unitary`] to synthesize Pauli-Lindblad
    /// rates (the Omega_1 dissipative-noise path gives zero for coherent
    /// noise).
    pub fn cx_theta_with_izz_crosstalk(omega: f64, theta: f64, delta: f64) -> Self {
        use crate::basis::Pauli1;
        assert!(omega > 0.0, "omega must be positive");
        let d = 8;
        let i2 = matrix::identity(2);
        let x = matrix::pauli_1q(Pauli1::X);
        let z = matrix::pauli_1q(Pauli1::Z);
        // H_g = (omega / 2) * (IXI - ZXI)
        let ixi = matrix::kron(&matrix::kron(&i2, &x, 2, 2), &i2, 4, 2);
        let zxi = matrix::kron(&matrix::kron(&z, &x, 2, 2), &i2, 4, 2);
        let diff = matrix::sub(&ixi, &zxi);
        let h_g = matrix::scale(&diff, Complex64::new(omega / 2.0, 0.0));
        let ideal = Lindbladian::new(d, h_g, Vec::new());
        // H_delta = (delta / 2) * IZZ
        let izz = matrix::kron(&matrix::kron(&i2, &z, 2, 2), &z, 4, 2);
        let h_delta = matrix::scale(&izz, Complex64::new(delta / 2.0, 0.0));
        let noise = Lindbladian::new(d, h_delta, Vec::new());
        let tau_g = theta / omega;
        Self {
            label: format!("CX_{{{:.4}}}⊗I+IZZ({:.4})", theta, delta),
            num_qubits: 3,
            ideal,
            noise,
            tau_g,
        }
    }

    /// 2-qubit arbitrary-angle CX rotation:
    /// `CX_theta = exp(-i (theta/2) (IX - ZX))`. Block-diagonal in the
    /// computational basis with the top 2x2 block zero (identity action on
    /// `|0l>`) and the bottom block = `omega_cx * X` (X rotation on the
    /// target when control = `|1>`).
    /// Reference: arXiv:2502.03462 lines 913-924.
    pub fn cx_theta(omega_cx: f64, theta: f64, noise: Lindbladian) -> Self {
        use crate::basis::Pauli1;
        assert!(omega_cx > 0.0, "omega_cx must be positive");
        assert_eq!(noise.d, 4, "cx_theta is 2-qubit");
        let d = 4;
        let i2 = matrix::identity(2);
        let x = matrix::pauli_1q(Pauli1::X);
        let z = matrix::pauli_1q(Pauli1::Z);
        let ix = matrix::kron(&i2, &x, 2, 2);
        let zx = matrix::kron(&z, &x, 2, 2);
        // H_g = (omega_cx / 2) * (IX - ZX)
        let diff = matrix::sub(&ix, &zx);
        let h_g = matrix::scale(&diff, Complex64::new(omega_cx / 2.0, 0.0));
        let ideal = Lindbladian::new(d, h_g, Vec::new());
        let tau_g = theta / omega_cx;
        Self {
            label: format!("CX_{{{:.4}}}", theta),
            num_qubits: 2,
            ideal,
            noise,
            tau_g,
        }
    }

    /// 2-qubit arbitrary-angle CZ rotation:
    /// `CZ_theta = exp(-i (theta/2) (II - IZ - ZI + ZZ))`.
    /// In computational basis `H_g = diag(0, 0, 0, 2 * omega_cz)`.
    /// Reference: arXiv:2502.03462 lines 885-891.
    pub fn cz_theta(omega_cz: f64, theta: f64, noise: Lindbladian) -> Self {
        assert!(omega_cz > 0.0, "omega_cz must be positive");
        assert_eq!(noise.d, 4, "cz_theta is 2-qubit");
        let d = 4;
        let mut h_g = matrix::zeros(d);
        h_g[3 * d + 3] = Complex64::new(2.0 * omega_cz, 0.0);
        let ideal = Lindbladian::new(d, h_g, Vec::new());
        let tau_g = theta / omega_cz;
        Self {
            label: format!("CZ_{{{:.4}}}", theta),
            num_qubits: 2,
            ideal,
            noise,
            tau_g,
        }
    }
}
