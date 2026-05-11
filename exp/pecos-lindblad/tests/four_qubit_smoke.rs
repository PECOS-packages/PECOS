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

//! 4-qubit smoke test: exercises the `d=16` matrix-exp path and 255-Pauli
//! enumeration in the synthesis pipeline.
//!
//! Case: 4Q identity gate with AD+PD noise on a single qubit. Expected
//! result (by independence): only the 3 weight-1 rates on that qubit are
//! non-zero, all other 252 rates vanish.

use approx::assert_abs_diff_eq;
use num_complex::Complex64;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Lindbladian, Pauli1, PauliString, synthesize_exact_unitary,
    synthesize_numerical,
};

fn kron_all(ops: &[&Matrix]) -> Matrix {
    // Left-associative Kronecker fold over a non-empty slice.
    let mut acc = ops[0].clone();
    let mut d = (ops[0].len() as f64).sqrt() as usize;
    for op in &ops[1..] {
        let d2 = (op.len() as f64).sqrt() as usize;
        acc = matrix::kron(&acc, op, d, d2);
        d *= d2;
    }
    acc
}

#[test]
fn three_qubit_identity_ad_on_one_qubit_fast_smoke() {
    let d = 8;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);

    let beta_down = 1e-3;
    let beta_phi = 2e-3;
    let tau_g = 5.0;

    let sm_q1 = kron_all(&[&i2, &sm, &i2]);
    let z_q1 = kron_all(&[&i2, &z, &i2]);

    let collapse: Vec<(Matrix, f64)> = vec![(sm_q1, beta_down), (z_q1, beta_phi / 2.0)];
    let hamiltonian: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    let noise = Lindbladian::new(d, hamiltonian, collapse);

    let gate = Gate::identity(3, noise, tau_g);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);
    let pl_coarse = synthesize_numerical(&gate, 2);

    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    assert_abs_diff_eq!(rate("IXI"), beta_down * tau_g / 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rate("IYI"), beta_down * tau_g / 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rate("IZI"), beta_phi * tau_g / 2.0, epsilon = 1e-10);
    for ps in PauliString::enumerate_nonidentity(3) {
        assert_abs_diff_eq!(pl.rate(&ps), pl_coarse.rate(&ps), epsilon = 1e-14);
    }

    for ps in PauliString::enumerate_nonidentity(3) {
        let label = format!("{}", ps);
        if label == "IXI" || label == "IYI" || label == "IZI" {
            continue;
        }
        assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-10);
    }
}

#[test]
fn three_qubit_identity_coherent_zzz_fast_smoke() {
    let d = 8;
    let tau_g = 5.0;
    let delta = 1e-4;

    let z = matrix::pauli_1q(Pauli1::Z);
    let zzz = kron_all(&[&z, &z, &z]);
    let h_delta = matrix::scale(&zzz, Complex64::new(delta / 2.0, 0.0));
    let noise = Lindbladian::new(d, h_delta, Vec::new());
    let gate = Gate::identity(3, noise, tau_g);
    let pl = synthesize_exact_unitary(&gate);

    let expected = (delta * tau_g).powi(2) / 4.0;
    assert_abs_diff_eq!(
        pl.rate(&PauliString::from_label("ZZZ").unwrap()),
        expected,
        epsilon = 1e-10
    );

    for ps in PauliString::enumerate_nonidentity(3) {
        if format!("{}", ps) == "ZZZ" {
            continue;
        }
        assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-10);
    }
}

#[test]
#[ignore = "Slow 4Q validation; run explicitly with: cargo test -p pecos-lindblad --test four_qubit_smoke -- --ignored"]
fn four_qubit_identity_ad_on_one_qubit() {
    let d = 16;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);

    // AD + PD on qubit 1 only (0-indexed).
    let beta_down = 1e-3;
    let beta_phi = 2e-3;
    let tau_g = 5.0;

    let sm_q1 = kron_all(&[&i2, &sm, &i2, &i2]);
    let z_q1 = kron_all(&[&i2, &z, &i2, &i2]);

    let collapse: Vec<(Matrix, f64)> = vec![(sm_q1, beta_down), (z_q1, beta_phi / 2.0)];
    let hamiltonian: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    let noise = Lindbladian::new(d, hamiltonian, collapse);

    let gate = Gate::identity(4, noise, tau_g);
    let pl = synthesize_numerical(&gate, DEFAULT_N_STEPS);
    let pl_coarse = synthesize_numerical(&gate, 2);

    // Expected non-zero rates: lambda_{q1=X}, lambda_{q1=Y}, lambda_{q1=Z}
    // on qubit 1 (index 1 from left in "qqqq" string).
    //   lambda_IXII = lambda_IYII = beta_down * tau_g / 4
    //   lambda_IZII = beta_phi * tau_g / 2
    let rate = |s: &str| pl.rate(&PauliString::from_label(s).unwrap());
    assert_abs_diff_eq!(rate("IXII"), beta_down * tau_g / 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rate("IYII"), beta_down * tau_g / 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(rate("IZII"), beta_phi * tau_g / 2.0, epsilon = 1e-10);
    for ps in PauliString::enumerate_nonidentity(4) {
        assert_abs_diff_eq!(pl.rate(&ps), pl_coarse.rate(&ps), epsilon = 1e-14);
    }

    // All other 252 non-identity 4Q Paulis should be zero.
    for ps in PauliString::enumerate_nonidentity(4) {
        let label = format!("{}", ps);
        if label == "IXII" || label == "IYII" || label == "IZII" {
            continue;
        }
        assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-10);
    }
}

#[test]
#[ignore = "Slow 4Q validation; run explicitly with: cargo test -p pecos-lindblad --test four_qubit_smoke -- --ignored"]
fn four_qubit_identity_coherent_zzzz_smoke() {
    // 4Q identity with coherent ZZZZ noise -- since all Zs commute, each
    // lambda_{all-Z} should be non-zero, everything else zero.
    let d = 16;
    let tau_g = 5.0;
    let delta = 1e-4;

    let z = matrix::pauli_1q(Pauli1::Z);
    let zzzz = kron_all(&[&z, &z, &z, &z]);
    let h_delta = matrix::scale(&zzzz, Complex64::new(delta / 2.0, 0.0));
    let noise = Lindbladian::new(d, h_delta, Vec::new());
    let gate = Gate::identity(4, noise, tau_g);
    let pl = synthesize_exact_unitary(&gate);

    // lambda_ZZZZ = (delta * tau_g)^2 / 4  (by analogy with 1Q phase noise).
    let expected = (delta * tau_g).powi(2) / 4.0;
    assert_abs_diff_eq!(
        pl.rate(&PauliString::from_label("ZZZZ").unwrap()),
        expected,
        epsilon = 1e-10
    );

    // All other 254 non-identity 4Q Paulis zero.
    for ps in PauliString::enumerate_nonidentity(4) {
        if format!("{}", ps) == "ZZZZ" {
            continue;
        }
        assert_abs_diff_eq!(pl.rate(&ps), 0.0, epsilon = 1e-10);
    }
}
