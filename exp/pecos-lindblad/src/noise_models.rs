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

//! Device-parameter convenience constructors for noise [`Lindbladian`]s.
//!
//! Real QEC experiments are typically specified in terms of coherence
//! times `(T_1, T_2)`, not raw Lindblad rates. This module converts
//! between them and builds the tensor-product Lindbladians that the
//! paper-fixture tests would otherwise hand-roll.
//!
//! # T1/T2 convention
//!
//! Standard textbook relation:
//!
//! ```text
//! beta_down = 1 / T_1
//! 1 / T_2   = 1 / (2 T_1) + 1 / T_phi
//! beta_phi  = 1 / T_phi = 1/T_2 - 1/(2 T_1)
//! ```
//!
//! `T_2 >= 2 T_1 / (1 + 2 T_1 / T_phi)`; pure-dephasing-free limit is
//! `T_2 = 2 T_1` with `beta_phi = 0`.

use num_complex::Complex64;

use crate::basis::Pauli1;
use crate::lindbladian::Lindbladian;
use crate::matrix::{self, Matrix};

/// Convert `(T_1, T_2)` to `(beta_down, beta_phi)`. Panics if `T_2 > 2 T_1`
/// (unphysical -- dephasing would be negative).
pub fn t1_t2_to_rates(t1: f64, t2: f64) -> (f64, f64) {
    assert!(t1 > 0.0, "T_1 must be positive");
    assert!(t2 > 0.0, "T_2 must be positive");
    let beta_down = 1.0 / t1;
    let inv_tphi = 1.0 / t2 - 1.0 / (2.0 * t1);
    assert!(
        inv_tphi >= -1e-15,
        "T_2 ({}) > 2 T_1 ({}) violates 1/T_phi = 1/T_2 - 1/(2 T_1) >= 0",
        t2,
        2.0 * t1,
    );
    (beta_down, inv_tphi.max(0.0))
}

/// 1-qubit amplitude-damping + pure-dephasing Lindbladian from `(T_1, T_2)`.
///
/// Collapse operators: `sigma_- with rate 1/T_1`, `Z with rate beta_phi/2`
/// where `beta_phi = 1/T_2 - 1/(2 T_1)`.
pub fn ad_pd_1q(t1: f64, t2: f64) -> Lindbladian {
    let (beta_down, beta_phi) = t1_t2_to_rates(t1, t2);
    let d = 2;
    let hamiltonian = matrix::zeros(d);
    let collapse: Vec<(Matrix, f64)> = vec![
        (matrix::sigma_minus(), beta_down),
        (matrix::pauli_1q(Pauli1::Z), beta_phi / 2.0),
    ];
    Lindbladian::new(d, hamiltonian, collapse)
}

/// 2-qubit amplitude-damping + pure-dephasing, independently parameterised
/// on left (`l`) and right (`r`) qubits.
pub fn ad_pd_2q(t1_l: f64, t1_r: f64, t2_l: f64, t2_r: f64) -> Lindbladian {
    let (bd_l, bp_l) = t1_t2_to_rates(t1_l, t2_l);
    let (bd_r, bp_r) = t1_t2_to_rates(t1_r, t2_r);
    let d = 4;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z, 2, 2);
    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, bd_l),
        (sm_r, bd_r),
        (z_l, bp_l / 2.0),
        (z_r, bp_r / 2.0),
    ];
    Lindbladian::new(d, matrix::zeros(d), collapse)
}

/// 2-qubit coherent phase noise:
/// `H_delta = (delta_iz/2) IZ + (delta_zi/2) ZI + (delta_zz/2) ZZ`,
/// no collapse operators (use [`crate::synthesize_exact_unitary`]).
pub fn coherent_phase_2q(delta_iz: f64, delta_zi: f64, delta_zz: f64) -> Lindbladian {
    let d = 4;
    let i2 = matrix::identity(2);
    let z = matrix::pauli_1q(Pauli1::Z);
    let iz = matrix::kron(&i2, &z, 2, 2);
    let zi = matrix::kron(&z, &i2, 2, 2);
    let zz = matrix::kron(&z, &z, 2, 2);
    let half = Complex64::new(0.5, 0.0);
    let h_delta = matrix::add(
        &matrix::add(
            &matrix::scale(&iz, Complex64::new(delta_iz, 0.0) * half),
            &matrix::scale(&zi, Complex64::new(delta_zi, 0.0) * half),
        ),
        &matrix::scale(&zz, Complex64::new(delta_zz, 0.0) * half),
    );
    Lindbladian::new(d, h_delta, Vec::new())
}

/// Analytic back-solve: given measured PL rates from a 1Q identity gate
/// with AD+PD noise, recover `(T_1, T_2)`.
///
/// Inverse of [`ad_pd_1q`] applied to [`crate::synthesize_identity_1q`]
/// using the paper's closed form (line 812):
///
/// ```text
/// lambda_x = lambda_y = beta_down * tau_g / 4  =>  T_1   = tau_g / (4 lambda_x)
/// lambda_z = beta_phi  * tau_g / 2             =>  T_phi = tau_g / (2 lambda_z)
/// 1/T_2 = 1/(2 T_1) + 1/T_phi                  =>  T_2   = 1 / (1/(2 T_1) + 1/T_phi)
/// ```
///
/// Returns `None` if rates are inconsistent (e.g. negative or would imply
/// `T_2 > 2 T_1`). Averages `lambda_x` and `lambda_y` for robustness
/// against small measurement noise.
pub fn recover_t1_t2_from_identity_1q(
    model: &crate::PauliLindbladModel,
    tau_g: f64,
) -> Option<(f64, f64)> {
    use crate::{Pauli1, PauliString};
    if tau_g <= 0.0 {
        return None;
    }
    let lam_x = model.rate(&PauliString::single(Pauli1::X));
    let lam_y = model.rate(&PauliString::single(Pauli1::Y));
    let lam_z = model.rate(&PauliString::single(Pauli1::Z));
    // AD gives equal lambda_x and lambda_y; average for noise robustness.
    let lam_avg_xy = 0.5 * (lam_x + lam_y);
    if lam_avg_xy <= 0.0 || lam_z < 0.0 {
        return None;
    }
    let t1 = tau_g / (4.0 * lam_avg_xy);
    if t1 <= 0.0 {
        return None;
    }
    // lambda_z = 0 is allowed (pure-T1 limit => T_2 = 2 T_1).
    let t2 = if lam_z < 1e-30 {
        2.0 * t1
    } else {
        let t_phi = tau_g / (2.0 * lam_z);
        let inv_t2 = 1.0 / (2.0 * t1) + 1.0 / t_phi;
        if inv_t2 <= 0.0 {
            return None;
        }
        1.0 / inv_t2
    };
    Some((t1, t2))
}

/// Recovered 2-qubit coherence-time parameters for the (left, right)
/// qubits of a 2Q gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveredParams2Q {
    pub t1_l: f64,
    pub t2_l: f64,
    pub t1_r: f64,
    pub t2_r: f64,
}

/// Analytic 2Q recovery: given a measured `CZ_theta + AD+PD` PL model,
/// back-solve `(T_1, T_2)` on each qubit.
///
/// Uses paper arXiv:2502.03462 eqs. 896-906:
///
/// ```text
/// lambda_zi = (theta/2) * (beta_phi_l / omega_cz)
/// lambda_iz = (theta/2) * (beta_phi_r / omega_cz)
/// lambda_xi = lambda_yi = (2*theta + sin(2*theta))/16 * beta_down_l / omega_cz
/// lambda_ix = lambda_iy = (2*theta + sin(2*theta))/16 * beta_down_r / omega_cz
/// ```
///
/// Averages the degenerate-in-paper pairs (`lambda_xi` ≈ `lambda_yi`)
/// for robustness against noisy measurements. Returns `None` if rates
/// are inconsistent (e.g. negative) or would imply `T_2 > 2 T_1`.
pub fn recover_ad_pd_2q_from_cz_theta(
    model: &crate::PauliLindbladModel,
    omega_cz: f64,
    theta: f64,
) -> Option<RecoveredParams2Q> {
    use crate::PauliString;
    if omega_cz <= 0.0 || theta <= 0.0 {
        return None;
    }
    let r = |s: &str| model.rate(&PauliString::from_label(s).unwrap());
    let factor_weight1_amp = (2.0 * theta + (2.0 * theta).sin()) / 16.0;

    // Amplitude damping: average the two equal rates (paper's 2-fold degeneracy).
    let beta_down_r = 0.5 * (r("IX") + r("IY")) * omega_cz / factor_weight1_amp;
    let beta_down_l = 0.5 * (r("XI") + r("YI")) * omega_cz / factor_weight1_amp;

    // Dephasing: single-rate back-solve.
    let beta_phi_r = r("IZ") * 2.0 * omega_cz / theta;
    let beta_phi_l = r("ZI") * 2.0 * omega_cz / theta;

    if beta_down_l < 0.0 || beta_down_r < 0.0 || beta_phi_l < 0.0 || beta_phi_r < 0.0 {
        return None;
    }
    if beta_down_l < 1e-300 || beta_down_r < 1e-300 {
        return None;
    }

    let t1_l = 1.0 / beta_down_l;
    let t1_r = 1.0 / beta_down_r;
    // 1/T_2 = 1/(2 T_1) + 1/T_phi and 1/T_phi = beta_phi.
    let inv_t2_l = 1.0 / (2.0 * t1_l) + beta_phi_l;
    let inv_t2_r = 1.0 / (2.0 * t1_r) + beta_phi_r;
    if inv_t2_l <= 0.0 || inv_t2_r <= 0.0 {
        return None;
    }
    Some(RecoveredParams2Q {
        t1_l,
        t2_l: 1.0 / inv_t2_l,
        t1_r,
        t2_r: 1.0 / inv_t2_r,
    })
}

/// Analytic 2Q recovery: given a measured `CX_theta + AD+PD` PL model,
/// back-solve `(T_1, T_2)` on each qubit.
///
/// Uses paper arXiv:2502.03462 eqs. 929-956. `beta_down_r` and
/// `beta_phi_r` mix in `lambda_iz` / `lambda_zz`; we exploit the
/// identity
///
/// ```text
/// lambda_iz - lambda_zz = sin(2*theta)/4 * beta_phi_r / omega
/// ```
///
/// to decouple the right-qubit dephasing. Requires `sin(2 theta) != 0`
/// -- at `theta = 0, pi/2, pi` the formula is degenerate and only
/// `beta_down_r + 6 beta_phi_r` is recoverable; we return `None`.
/// `beta_down_l` and `beta_phi_l` come from clean single-unknown rates
/// (`lambda_xi`, `lambda_zi`).
pub fn recover_ad_pd_2q_from_cx_theta(
    model: &crate::PauliLindbladModel,
    omega_cx: f64,
    theta: f64,
) -> Option<RecoveredParams2Q> {
    use crate::PauliString;
    if omega_cx <= 0.0 || theta <= 0.0 {
        return None;
    }
    let s2 = (2.0 * theta).sin();
    if s2.abs() < 1e-10 {
        // Degenerate: can't separate beta_down_r from beta_phi_r.
        return None;
    }
    let r = |s: &str| model.rate(&PauliString::from_label(s).unwrap());

    // Left qubit: clean single-parameter back-solves.
    let factor_weight1_amp_l = (2.0 * theta + s2) / 16.0;
    let beta_down_l = 0.5 * (r("XI") + r("YI")) * omega_cx / factor_weight1_amp_l;
    let beta_phi_l = r("ZI") * 2.0 * omega_cx / theta;

    // Right qubit: beta_down_r from clean lambda_ix; beta_phi_r via the
    // (lambda_iz - lambda_zz) decoupling.
    let beta_down_r = r("IX") * 4.0 * omega_cx / theta;
    let beta_phi_r = (r("IZ") - r("ZZ")) * 4.0 * omega_cx / s2;

    if beta_down_l < 0.0 || beta_down_r < 0.0 || beta_phi_l < 0.0 || beta_phi_r < 0.0 {
        return None;
    }
    if beta_down_l < 1e-300 || beta_down_r < 1e-300 {
        return None;
    }
    let t1_l = 1.0 / beta_down_l;
    let t1_r = 1.0 / beta_down_r;
    let inv_t2_l = 1.0 / (2.0 * t1_l) + beta_phi_l;
    let inv_t2_r = 1.0 / (2.0 * t1_r) + beta_phi_r;
    if inv_t2_l <= 0.0 || inv_t2_r <= 0.0 {
        return None;
    }
    Some(RecoveredParams2Q {
        t1_l,
        t2_l: 1.0 / inv_t2_l,
        t1_r,
        t2_r: 1.0 / inv_t2_r,
    })
}

/// Consistency residual for a `CZ_theta + AD+PD` recovery: for the
/// recovered parameters, compute the L2 residual between observed
/// degenerate-pair rates (`lambda_xi` vs `lambda_yi`, etc.). Large
/// residuals flag model mismatch (noise source beyond AD+PD).
pub fn cz_recovery_residual(model: &crate::PauliLindbladModel) -> f64 {
    use crate::PauliString;
    let r = |s: &str| model.rate(&PauliString::from_label(s).unwrap());
    let pairs = [
        (r("IX"), r("IY")),
        (r("XI"), r("YI")),
        (r("ZX"), r("ZY")),
        (r("XZ"), r("YZ")),
    ];
    pairs
        .iter()
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Per-Pauli mean + standard-deviation statistics from a Monte-Carlo
/// uncertainty propagation.
#[derive(Clone, Debug, Default)]
pub struct RateUncertainty {
    pub paulis: Vec<crate::PauliString>,
    /// `means[i]` = mean of `rates[i]` across MC samples.
    pub means: Vec<f64>,
    /// `stds[i]` = sample standard deviation of `rates[i]`.
    pub stds: Vec<f64>,
    /// Number of samples drawn.
    pub n_samples: usize,
}

/// Propagate parameter uncertainty into the Pauli-Lindblad rates by
/// Monte-Carlo sampling.
///
/// `synthesize_sample` is called once per MC draw. The user builds the
/// `Gate` from the (possibly jittered) parameters and returns the
/// synthesized [`PauliLindbladModel`]. This routine aggregates across
/// draws: per-Pauli sample mean + sample standard deviation.
///
/// The first sample determines the support enumeration; later samples
/// are matched by support equality, so stochastic supports (same Pauli
/// but generated in a different order) are fine.
pub fn propagate_uncertainty(
    n_samples: usize,
    mut synthesize_sample: impl FnMut(usize) -> crate::PauliLindbladModel,
) -> RateUncertainty {
    assert!(n_samples >= 2, "need >=2 samples to compute std");

    // Use the first sample to fix the Pauli set.
    let first = synthesize_sample(0);
    let n_p = first.supports.len();
    let mut sum = first.rates.clone();
    let mut sum_sq: Vec<f64> = first.rates.iter().map(|r| r * r).collect();

    for k in 1..n_samples {
        let model = synthesize_sample(k);
        assert_eq!(
            model.supports.len(),
            n_p,
            "MC draw {}: support size {} != expected {}",
            k,
            model.supports.len(),
            n_p,
        );
        for (i, p) in first.supports.iter().enumerate() {
            let r = model.rate(p);
            sum[i] += r;
            sum_sq[i] += r * r;
        }
    }

    let n = n_samples as f64;
    let means: Vec<f64> = sum.iter().map(|s| s / n).collect();
    let stds: Vec<f64> = sum_sq
        .iter()
        .zip(means.iter())
        .map(|(ss, m)| {
            let var = (ss / n - m * m).max(0.0);
            var.sqrt()
        })
        .collect();

    RateUncertainty {
        paulis: first.supports,
        means,
        stds,
        n_samples,
    }
}

impl RateUncertainty {
    /// Look up mean rate for a given Pauli; returns 0 if not in support.
    pub fn mean(&self, p: &crate::PauliString) -> f64 {
        self.paulis
            .iter()
            .zip(&self.means)
            .find(|(s, _)| *s == p)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }

    /// Look up standard deviation for a given Pauli; returns 0 if not in support.
    pub fn std(&self, p: &crate::PauliString) -> f64 {
        self.paulis
            .iter()
            .zip(&self.stds)
            .find(|(s, _)| *s == p)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t1_t2_round_trip() {
        let t1 = 100e-6;
        let t2 = 80e-6; // < 2 T_1 so physical
        let (bd, bp) = t1_t2_to_rates(t1, t2);
        assert!((bd - 1.0 / t1).abs() < 1e-15);
        assert!((bp - (1.0 / t2 - 1.0 / (2.0 * t1))).abs() < 1e-15);
    }

    #[test]
    fn t2_equals_2_t1_gives_zero_dephasing() {
        // Dephasing-free limit.
        let t1 = 100e-6;
        let (bd, bp) = t1_t2_to_rates(t1, 2.0 * t1);
        assert!((bd - 1.0 / t1).abs() < 1e-15);
        assert!(bp < 1e-15, "bp should be ~0, got {}", bp);
    }

    #[test]
    #[should_panic(expected = "T_2")]
    fn unphysical_t2_panics() {
        let _ = t1_t2_to_rates(100e-6, 300e-6); // T_2 > 2 T_1
    }
}
