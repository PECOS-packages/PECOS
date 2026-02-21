// Copyright 2026 The PECOS Developers
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

//! 32-bit precision State Vector Simulator (f32 instead of f64)
//!
//! This provides ~1.6-1.8x speedup over f64 by:
//! 1. Halving memory bandwidth requirements
//! 2. Using wider SIMD (f32x8 vs f64x4)
//!
//! Trade-off: Reduced precision (~7 vs ~15 decimal digits)
//! Suitable for most quantum simulation tasks where high precision isn't critical.

use crate::clifford_gateable::MeasurementResult;
use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use num_complex::Complex;
use pecos_core::{Angle64, QubitId};
use pecos_rng::{PecosRng, Rng, RngExt};
use std::fmt::Debug;
use wide::f32x8;

type Complex32 = Complex<f32>;

// =============================================================================
// 2x2 Complex Matrix for Gate Fusion (f32 version)
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct Complex2x2_32 {
    a_re: f32,
    a_im: f32,
    b_re: f32,
    b_im: f32,
    c_re: f32,
    c_im: f32,
    d_re: f32,
    d_im: f32,
}

impl Complex2x2_32 {
    #[inline]
    fn is_identity(&self) -> bool {
        const EPS: f32 = 1e-6;
        (self.a_re - 1.0).abs() < EPS
            && self.a_im.abs() < EPS
            && self.b_re.abs() < EPS
            && self.b_im.abs() < EPS
            && self.c_re.abs() < EPS
            && self.c_im.abs() < EPS
            && (self.d_re - 1.0).abs() < EPS
            && self.d_im.abs() < EPS
    }

    #[inline]
    fn mul(&self, other: &Self) -> Self {
        Self {
            a_re: self.a_re * other.a_re - self.a_im * other.a_im + self.b_re * other.c_re
                - self.b_im * other.c_im,
            a_im: self.a_re * other.a_im
                + self.a_im * other.a_re
                + self.b_re * other.c_im
                + self.b_im * other.c_re,
            b_re: self.a_re * other.b_re - self.a_im * other.b_im + self.b_re * other.d_re
                - self.b_im * other.d_im,
            b_im: self.a_re * other.b_im
                + self.a_im * other.b_re
                + self.b_re * other.d_im
                + self.b_im * other.d_re,
            c_re: self.c_re * other.a_re - self.c_im * other.a_im + self.d_re * other.c_re
                - self.d_im * other.c_im,
            c_im: self.c_re * other.a_im
                + self.c_im * other.a_re
                + self.d_re * other.c_im
                + self.d_im * other.c_re,
            d_re: self.c_re * other.b_re - self.c_im * other.b_im + self.d_re * other.d_re
                - self.d_im * other.d_im,
            d_im: self.c_re * other.b_im
                + self.c_im * other.b_re
                + self.d_re * other.d_im
                + self.d_im * other.d_re,
        }
    }
}

mod gate_matrices_32 {
    use super::Complex2x2_32;

    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    pub const H: Complex2x2_32 = Complex2x2_32 {
        a_re: INV_SQRT2,
        a_im: 0.0,
        b_re: INV_SQRT2,
        b_im: 0.0,
        c_re: INV_SQRT2,
        c_im: 0.0,
        d_re: -INV_SQRT2,
        d_im: 0.0,
    };

    pub const X: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.0,
        a_im: 0.0,
        b_re: 1.0,
        b_im: 0.0,
        c_re: 1.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: 0.0,
    };

    pub const Y: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: -1.0,
        c_re: 0.0,
        c_im: 1.0,
        d_re: 0.0,
        d_im: 0.0,
    };

    pub const Z: Complex2x2_32 = Complex2x2_32 {
        a_re: 1.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 0.0,
        d_re: -1.0,
        d_im: 0.0,
    };

    pub const SZ: Complex2x2_32 = Complex2x2_32 {
        a_re: 1.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: 1.0,
    };

    pub const SZDG: Complex2x2_32 = Complex2x2_32 {
        a_re: 1.0,
        a_im: 0.0,
        b_re: 0.0,
        b_im: 0.0,
        c_re: 0.0,
        c_im: 0.0,
        d_re: 0.0,
        d_im: -1.0,
    };

    pub const SX: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: -0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    pub const SXDG: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: 0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: -0.5,
    };

    pub const SY: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: -0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: 0.5,
    };

    pub const SYDG: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: -0.5,
        c_im: 0.5,
        d_re: 0.5,
        d_im: -0.5,
    };

    pub const F: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.5,
        a_im: 0.5,
        b_re: 0.5,
        b_im: -0.5,
        c_re: 0.5,
        c_im: 0.5,
        d_re: -0.5,
        d_im: 0.5,
    };

    pub const FDG: Complex2x2_32 = Complex2x2_32 {
        a_re: 0.5,
        a_im: -0.5,
        b_re: 0.5,
        b_im: 0.5,
        c_re: 0.5,
        c_im: -0.5,
        d_re: -0.5,
        d_im: -0.5,
    };
}

// =============================================================================
// StateVecSoA32 - 32-bit Precision State Vector Simulator
// =============================================================================

/// 32-bit precision state vector simulator using Structure of Arrays layout.
///
/// This provides ~1.6-1.8x speedup over the f64 version at the cost of
/// reduced precision (~7 decimal digits vs ~15).
pub struct StateVecSoA32<R = PecosRng>
where
    R: Rng,
{
    /// Real components of the state vector (f32)
    pub(crate) real: Vec<f32>,
    /// Imaginary components of the state vector (f32)
    pub(crate) imag: Vec<f32>,
    /// Number of qubits
    num_qubits: usize,
    /// Random number generator for measurements
    rng: R,
    /// Pending gates for fusion (None = identity)
    pending_gates: Vec<Option<Complex2x2_32>>,
    /// Whether gate fusion is enabled
    fusion_enabled: bool,
}

impl<R> StateVecSoA32<R>
where
    R: Rng,
{
    // =========================================================================
    // Gate Fusion Support
    // =========================================================================

    /// Enable or disable gate fusion.
    #[inline]
    pub fn set_fusion(&mut self, enabled: bool) {
        if !enabled {
            self.flush();
        }
        self.fusion_enabled = enabled;
    }

    /// Check if gate fusion is enabled.
    #[inline]
    pub fn fusion_enabled(&self) -> bool {
        self.fusion_enabled
    }

    /// Queue a single-qubit gate for fusion.
    #[inline]
    fn queue_gate(&mut self, qubit: usize, gate: &Complex2x2_32) {
        if !self.fusion_enabled {
            self.apply_fused_matrix(qubit, gate);
            return;
        }

        match &mut self.pending_gates[qubit] {
            Some(accumulated) => {
                *accumulated = gate.mul(accumulated);
            }
            None => {
                self.pending_gates[qubit] = Some(*gate);
            }
        }
    }

    /// Flush pending gates for a specific qubit.
    #[inline]
    fn flush_qubit(&mut self, qubit: usize) {
        if let Some(matrix) = self.pending_gates[qubit].take()
            && !matrix.is_identity()
        {
            self.apply_fused_matrix(qubit, &matrix);
        }
    }

    /// Flush all pending gates.
    pub fn flush(&mut self) {
        for qubit in 0..self.num_qubits {
            self.flush_qubit(qubit);
        }
    }

    /// Flush pending gates for two qubits.
    #[inline]
    fn flush_two_qubit(&mut self, q1: usize, q2: usize) {
        self.flush_qubit(q1);
        self.flush_qubit(q2);
    }

    /// Apply a fused 2x2 complex matrix using SIMD.
    fn apply_fused_matrix(&mut self, q: usize, m: &Complex2x2_32) {
        let step = 1 << q;
        let n = self.real.len();

        if step < 8 {
            // Scalar fallback for small steps
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let alpha_re = self.real[j];
                    let alpha_im = self.imag[j];
                    let beta_re = self.real[p];
                    let beta_im = self.imag[p];

                    self.real[j] = (m.a_re * alpha_re - m.a_im * alpha_im)
                        + (m.b_re * beta_re - m.b_im * beta_im);
                    self.imag[j] = (m.a_re * alpha_im + m.a_im * alpha_re)
                        + (m.b_re * beta_im + m.b_im * beta_re);
                    self.real[p] = (m.c_re * alpha_re - m.c_im * alpha_im)
                        + (m.d_re * beta_re - m.d_im * beta_im);
                    self.imag[p] = (m.c_re * alpha_im + m.c_im * alpha_re)
                        + (m.d_re * beta_im + m.d_im * beta_re);
                }
            }
        } else {
            // SIMD path using f32x8
            let a_re = f32x8::splat(m.a_re);
            let a_im = f32x8::splat(m.a_im);
            let b_re = f32x8::splat(m.b_re);
            let b_im = f32x8::splat(m.b_im);
            let c_re = f32x8::splat(m.c_re);
            let c_im = f32x8::splat(m.c_im);
            let d_re = f32x8::splat(m.d_re);
            let d_im = f32x8::splat(m.d_im);

            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 8 <= i + step {
                    let p = j + step;

                    let alpha_re = f32x8::from(&self.real[j..j + 8]);
                    let alpha_im = f32x8::from(&self.imag[j..j + 8]);
                    let beta_re = f32x8::from(&self.real[p..p + 8]);
                    let beta_im = f32x8::from(&self.imag[p..p + 8]);

                    let new_alpha_re =
                        (a_re * alpha_re - a_im * alpha_im) + (b_re * beta_re - b_im * beta_im);
                    let new_alpha_im =
                        (a_re * alpha_im + a_im * alpha_re) + (b_re * beta_im + b_im * beta_re);
                    let new_beta_re =
                        (c_re * alpha_re - c_im * alpha_im) + (d_re * beta_re - d_im * beta_im);
                    let new_beta_im =
                        (c_re * alpha_im + c_im * alpha_re) + (d_re * beta_im + d_im * beta_re);

                    let arr_re_a: [f32; 8] = new_alpha_re.into();
                    let arr_im_a: [f32; 8] = new_alpha_im.into();
                    let arr_re_b: [f32; 8] = new_beta_re.into();
                    let arr_im_b: [f32; 8] = new_beta_im.into();

                    self.real[j..j + 8].copy_from_slice(&arr_re_a);
                    self.imag[j..j + 8].copy_from_slice(&arr_im_a);
                    self.real[p..p + 8].copy_from_slice(&arr_re_b);
                    self.imag[p..p + 8].copy_from_slice(&arr_im_b);

                    j += 8;
                }
            }
        }
    }

    // =========================================================================
    // Specialized Gate Implementations (used when fusion is disabled)
    // =========================================================================

    /// Specialized Z gate: negate amplitudes where qubit bit is 1.
    #[inline]
    fn apply_z_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            let neg_one = f32x8::splat(-1.0);
            for i in (0..n).step_by(step * 2) {
                let mut j = i + step;
                while j + 8 <= i + step * 2 {
                    let re = f32x8::from(&self.real[j..j + 8]);
                    let im = f32x8::from(&self.imag[j..j + 8]);
                    let neg_re: [f32; 8] = (re * neg_one).into();
                    let neg_im: [f32; 8] = (im * neg_one).into();
                    self.real[j..j + 8].copy_from_slice(&neg_re);
                    self.imag[j..j + 8].copy_from_slice(&neg_im);
                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in (i + step)..(i + step * 2) {
                    self.real[j] = -self.real[j];
                    self.imag[j] = -self.imag[j];
                }
            }
        }
    }

    /// Specialized X gate: swap amplitude pairs.
    #[inline]
    fn apply_x_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        for i in (0..n).step_by(step * 2) {
            let (left_re, right_re) = self.real[i..i + step * 2].split_at_mut(step);
            left_re.swap_with_slice(right_re);

            let (left_im, right_im) = self.imag[i..i + step * 2].split_at_mut(step);
            left_im.swap_with_slice(right_im);
        }
    }

    /// Specialized Y gate: swap with phase factors.
    #[inline]
    fn apply_y_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            for i in (0..n).step_by(step * 2) {
                let mut j = 0;
                while j + 8 <= step {
                    let idx0 = i + j;
                    let idx1 = i + j + step;

                    let re0 = f32x8::from(&self.real[idx0..idx0 + 8]);
                    let im0 = f32x8::from(&self.imag[idx0..idx0 + 8]);
                    let re1 = f32x8::from(&self.real[idx1..idx1 + 8]);
                    let im1 = f32x8::from(&self.imag[idx1..idx1 + 8]);

                    let new_re0: [f32; 8] = im1.into();
                    let new_im0: [f32; 8] = (-re1).into();
                    let new_re1: [f32; 8] = (-im0).into();
                    let new_im1: [f32; 8] = re0.into();

                    self.real[idx0..idx0 + 8].copy_from_slice(&new_re0);
                    self.imag[idx0..idx0 + 8].copy_from_slice(&new_im0);
                    self.real[idx1..idx1 + 8].copy_from_slice(&new_re1);
                    self.imag[idx1..idx1 + 8].copy_from_slice(&new_im1);

                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in 0..step {
                    let idx0 = i + j;
                    let idx1 = i + j + step;

                    let re0 = self.real[idx0];
                    let im0 = self.imag[idx0];
                    let re1 = self.real[idx1];
                    let im1 = self.imag[idx1];

                    self.real[idx0] = im1;
                    self.imag[idx0] = -re1;
                    self.real[idx1] = -im0;
                    self.imag[idx1] = re0;
                }
            }
        }
    }

    /// Specialized SZ gate: multiply by i where qubit bit is 1.
    #[inline]
    fn apply_sz_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            for i in (0..n).step_by(step * 2) {
                let mut j = i + step;
                while j + 8 <= i + step * 2 {
                    let re = f32x8::from(&self.real[j..j + 8]);
                    let im = f32x8::from(&self.imag[j..j + 8]);
                    let new_re: [f32; 8] = (-im).into();
                    let new_im: [f32; 8] = re.into();
                    self.real[j..j + 8].copy_from_slice(&new_re);
                    self.imag[j..j + 8].copy_from_slice(&new_im);
                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in (i + step)..(i + step * 2) {
                    let re = self.real[j];
                    let im = self.imag[j];
                    self.real[j] = -im;
                    self.imag[j] = re;
                }
            }
        }
    }

    /// Specialized SZDG gate: multiply by -i where qubit bit is 1.
    #[inline]
    fn apply_szdg_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            for i in (0..n).step_by(step * 2) {
                let mut j = i + step;
                while j + 8 <= i + step * 2 {
                    let re = f32x8::from(&self.real[j..j + 8]);
                    let im = f32x8::from(&self.imag[j..j + 8]);
                    let new_re: [f32; 8] = im.into();
                    let new_im: [f32; 8] = (-re).into();
                    self.real[j..j + 8].copy_from_slice(&new_re);
                    self.imag[j..j + 8].copy_from_slice(&new_im);
                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in (i + step)..(i + step * 2) {
                    let re = self.real[j];
                    let im = self.imag[j];
                    self.real[j] = im;
                    self.imag[j] = -re;
                }
            }
        }
    }

    /// Specialized H gate using SIMD.
    #[inline]
    fn apply_h_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();
        let inv_sqrt2: f32 = std::f32::consts::FRAC_1_SQRT_2;

        if step >= 8 {
            let factor = f32x8::splat(inv_sqrt2);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 8 <= i + step {
                    let paired_j = j + step;

                    let a_re = f32x8::from(&self.real[j..j + 8]);
                    let a_im = f32x8::from(&self.imag[j..j + 8]);
                    let b_re = f32x8::from(&self.real[paired_j..paired_j + 8]);
                    let b_im = f32x8::from(&self.imag[paired_j..paired_j + 8]);

                    let new_a_re: [f32; 8] = ((a_re + b_re) * factor).into();
                    let new_a_im: [f32; 8] = ((a_im + b_im) * factor).into();
                    let new_b_re: [f32; 8] = ((a_re - b_re) * factor).into();
                    let new_b_im: [f32; 8] = ((a_im - b_im) * factor).into();

                    self.real[j..j + 8].copy_from_slice(&new_a_re);
                    self.imag[j..j + 8].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 8].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 8].copy_from_slice(&new_b_im);

                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    self.real[j] = (a_re + b_re) * inv_sqrt2;
                    self.imag[j] = (a_im + b_im) * inv_sqrt2;
                    self.real[paired_j] = (a_re - b_re) * inv_sqrt2;
                    self.imag[paired_j] = (a_im - b_im) * inv_sqrt2;
                }
            }
        }
    }

    /// Specialized SX gate.
    #[inline]
    fn apply_sx_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            let half = f32x8::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 8 <= i + step {
                    let paired_j = j + step;

                    let a_re = f32x8::from(&self.real[j..j + 8]);
                    let a_im = f32x8::from(&self.imag[j..j + 8]);
                    let b_re = f32x8::from(&self.real[paired_j..paired_j + 8]);
                    let b_im = f32x8::from(&self.imag[paired_j..paired_j + 8]);

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    let new_a_re: [f32; 8] = ((sum_re - diff_im) * half).into();
                    let new_a_im: [f32; 8] = ((sum_im + diff_re) * half).into();
                    let new_b_re: [f32; 8] = ((sum_re + diff_im) * half).into();
                    let new_b_im: [f32; 8] = ((sum_im - diff_re) * half).into();

                    self.real[j..j + 8].copy_from_slice(&new_a_re);
                    self.imag[j..j + 8].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 8].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 8].copy_from_slice(&new_b_im);

                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    self.real[j] = (sum_re - diff_im) * 0.5;
                    self.imag[j] = (sum_im + diff_re) * 0.5;
                    self.real[paired_j] = (sum_re + diff_im) * 0.5;
                    self.imag[paired_j] = (sum_im - diff_re) * 0.5;
                }
            }
        }
    }

    /// Specialized SXDG gate.
    #[inline]
    fn apply_sxdg_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            let half = f32x8::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 8 <= i + step {
                    let paired_j = j + step;

                    let a_re = f32x8::from(&self.real[j..j + 8]);
                    let a_im = f32x8::from(&self.imag[j..j + 8]);
                    let b_re = f32x8::from(&self.real[paired_j..paired_j + 8]);
                    let b_im = f32x8::from(&self.imag[paired_j..paired_j + 8]);

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    let new_a_re: [f32; 8] = ((sum_re + diff_im) * half).into();
                    let new_a_im: [f32; 8] = ((sum_im - diff_re) * half).into();
                    let new_b_re: [f32; 8] = ((sum_re - diff_im) * half).into();
                    let new_b_im: [f32; 8] = ((sum_im + diff_re) * half).into();

                    self.real[j..j + 8].copy_from_slice(&new_a_re);
                    self.imag[j..j + 8].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 8].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 8].copy_from_slice(&new_b_im);

                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    self.real[j] = (sum_re + diff_im) * 0.5;
                    self.imag[j] = (sum_im - diff_re) * 0.5;
                    self.real[paired_j] = (sum_re - diff_im) * 0.5;
                    self.imag[paired_j] = (sum_im + diff_re) * 0.5;
                }
            }
        }
    }

    /// Specialized SY gate.
    #[inline]
    fn apply_sy_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            let half = f32x8::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 8 <= i + step {
                    let paired_j = j + step;

                    let a_re = f32x8::from(&self.real[j..j + 8]);
                    let a_im = f32x8::from(&self.imag[j..j + 8]);
                    let b_re = f32x8::from(&self.real[paired_j..paired_j + 8]);
                    let b_im = f32x8::from(&self.imag[paired_j..paired_j + 8]);

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    let new_a_re: [f32; 8] = ((sum_re + diff_re) * half).into();
                    let new_a_im: [f32; 8] = ((sum_im + diff_im) * half).into();
                    let new_b_re: [f32; 8] = ((sum_re - diff_re) * half).into();
                    let new_b_im: [f32; 8] = ((sum_im - diff_im) * half).into();

                    self.real[j..j + 8].copy_from_slice(&new_a_re);
                    self.imag[j..j + 8].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 8].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 8].copy_from_slice(&new_b_im);

                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    self.real[j] = (sum_re + diff_re) * 0.5;
                    self.imag[j] = (sum_im + diff_im) * 0.5;
                    self.real[paired_j] = (sum_re - diff_re) * 0.5;
                    self.imag[paired_j] = (sum_im - diff_im) * 0.5;
                }
            }
        }
    }

    /// Specialized SYDG gate.
    #[inline]
    fn apply_sydg_gate(&mut self, q: usize) {
        let step = 1 << q;
        let n = self.real.len();

        if step >= 8 {
            let half = f32x8::splat(0.5);
            for i in (0..n).step_by(step * 2) {
                let mut j = i;
                while j + 8 <= i + step {
                    let paired_j = j + step;

                    let a_re = f32x8::from(&self.real[j..j + 8]);
                    let a_im = f32x8::from(&self.imag[j..j + 8]);
                    let b_re = f32x8::from(&self.real[paired_j..paired_j + 8]);
                    let b_im = f32x8::from(&self.imag[paired_j..paired_j + 8]);

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    let new_a_re: [f32; 8] = ((sum_re - diff_re) * half).into();
                    let new_a_im: [f32; 8] = ((sum_im - diff_im) * half).into();
                    let new_b_re: [f32; 8] = ((sum_re + diff_re) * half).into();
                    let new_b_im: [f32; 8] = ((sum_im + diff_im) * half).into();

                    self.real[j..j + 8].copy_from_slice(&new_a_re);
                    self.imag[j..j + 8].copy_from_slice(&new_a_im);
                    self.real[paired_j..paired_j + 8].copy_from_slice(&new_b_re);
                    self.imag[paired_j..paired_j + 8].copy_from_slice(&new_b_im);

                    j += 8;
                }
            }
        } else {
            for i in (0..n).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;

                    let a_re = self.real[j];
                    let a_im = self.imag[j];
                    let b_re = self.real[paired_j];
                    let b_im = self.imag[paired_j];

                    let sum_re = a_re + b_re;
                    let sum_im = a_im + b_im;
                    let diff_re = a_re - b_re;
                    let diff_im = a_im - b_im;

                    self.real[j] = (sum_re - diff_re) * 0.5;
                    self.imag[j] = (sum_im - diff_im) * 0.5;
                    self.real[paired_j] = (sum_re + diff_re) * 0.5;
                    self.imag[paired_j] = (sum_im + diff_im) * 0.5;
                }
            }
        }
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get probability of a basis state.
    pub fn probability(&mut self, basis_state: usize) -> f64 {
        self.flush();
        let re = f64::from(self.real[basis_state]);
        let im = f64::from(self.imag[basis_state]);
        re * re + im * im
    }

    /// Get amplitude of a basis state.
    pub fn get_amplitude(&mut self, basis_state: usize) -> Complex32 {
        self.flush();
        Complex32::new(self.real[basis_state], self.imag[basis_state])
    }

    /// Convert to complex vector for comparison.
    pub fn to_complex_vec(&mut self) -> Vec<Complex32> {
        self.flush();
        self.real
            .iter()
            .zip(self.imag.iter())
            .map(|(&re, &im)| Complex32::new(re, im))
            .collect()
    }
}

// =============================================================================
// Constructors
// =============================================================================

// Constructors that use the default PecosRng
impl StateVecSoA32 {
    /// Creates a new state vector initialized to |0...0⟩.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> StateVecSoA32<PecosRng> {
        let rng = rand::make_rng();
        StateVecSoA32::with_rng(num_qubits, rng)
    }

    /// Creates a new state vector with a specific seed for reproducibility.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> StateVecSoA32<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        StateVecSoA32::with_rng(num_qubits, rng)
    }
}

impl StateVecSoA32<PecosRng> {
    /// Sets the random seed for measurements.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = PecosRng::seed_from_u64(seed);
    }
}

impl<R> StateVecSoA32<R>
where
    R: Rng,
{
    /// Create with specific RNG.
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let size = 1 << num_qubits;
        let mut real = vec![0.0f32; size];
        real[0] = 1.0;
        Self {
            real,
            imag: vec![0.0; size],
            num_qubits,
            rng,
            pending_gates: vec![None; num_qubits],
            fusion_enabled: true,
        }
    }

    /// Returns the number of qubits.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

// =============================================================================
// QuantumSimulator Implementation
// =============================================================================

impl<R> QuantumSimulator for StateVecSoA32<R>
where
    R: Rng,
{
    fn reset(&mut self) -> &mut Self {
        self.real.fill(0.0);
        self.imag.fill(0.0);
        self.real[0] = 1.0;
        for pg in &mut self.pending_gates {
            *pg = None;
        }
        self
    }
}

// =============================================================================
// CliffordGateable Implementation
// =============================================================================

impl<R> CliffordGateable for StateVecSoA32<R>
where
    R: Rng,
{
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::H);
            } else {
                self.apply_h_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::X);
            } else {
                self.apply_x_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::Y);
            } else {
                self.apply_y_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::Z);
            } else {
                self.apply_z_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::SZ);
            } else {
                self.apply_sz_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::SZDG);
            } else {
                self.apply_szdg_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::SX);
            } else {
                self.apply_sx_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::SXDG);
            } else {
                self.apply_sxdg_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::SY);
            } else {
                self.apply_sy_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            if self.fusion_enabled {
                self.queue_gate(q.index(), &gate_matrices_32::SYDG);
            } else {
                self.apply_sydg_gate(q.index());
            }
        }
        self
    }

    #[inline]
    fn f(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_gate(q.index(), &gate_matrices_32::F);
        }
        self
    }

    #[inline]
    fn fdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_gate(q.index(), &gate_matrices_32::FDG);
        }
        self
    }

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let control = pair[0].index();
            let target = pair[1].index();
            self.flush_two_qubit(control, target);

            let n = self.real.len();
            let (q_lo, q_hi) = if control < target {
                (control, target)
            } else {
                (target, control)
            };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let control_mask = 1 << control;
            let target_mask = 1 << target;

            if step_lo >= 8 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 8 <= step_lo {
                            let base = i_lo + offset;
                            let idx0 = base | control_mask;
                            let idx1 = idx0 | target_mask;

                            let re0 = f32x8::from(&self.real[idx0..idx0 + 8]);
                            let im0 = f32x8::from(&self.imag[idx0..idx0 + 8]);
                            let re1 = f32x8::from(&self.real[idx1..idx1 + 8]);
                            let im1 = f32x8::from(&self.imag[idx1..idx1 + 8]);

                            let arr_re0: [f32; 8] = re1.into();
                            let arr_im0: [f32; 8] = im1.into();
                            let arr_re1: [f32; 8] = re0.into();
                            let arr_im1: [f32; 8] = im0.into();

                            self.real[idx0..idx0 + 8].copy_from_slice(&arr_re0);
                            self.imag[idx0..idx0 + 8].copy_from_slice(&arr_im0);
                            self.real[idx1..idx1 + 8].copy_from_slice(&arr_re1);
                            self.imag[idx1..idx1 + 8].copy_from_slice(&arr_im1);

                            offset += 8;
                        }
                    }
                }
            } else {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx0 = base | control_mask;
                            let idx1 = idx0 | target_mask;
                            self.real.swap(idx0, idx1);
                            self.imag.swap(idx0, idx1);
                        }
                    }
                }
            }
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CZ requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();
            self.flush_two_qubit(q1, q2);

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask_11 = (1 << q1) | (1 << q2);

            if step_lo >= 8 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 8 <= step_lo {
                            let idx = (i_lo + offset) | mask_11;
                            let re = f32x8::from(&self.real[idx..idx + 8]);
                            let im = f32x8::from(&self.imag[idx..idx + 8]);
                            let arr_re: [f32; 8] = (-re).into();
                            let arr_im: [f32; 8] = (-im).into();
                            self.real[idx..idx + 8].copy_from_slice(&arr_re);
                            self.imag[idx..idx + 8].copy_from_slice(&arr_im);
                            offset += 8;
                        }
                    }
                }
            } else {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let idx = (i_lo + offset) | mask_11;
                            self.real[idx] = -self.real[idx];
                            self.imag[idx] = -self.imag[idx];
                        }
                    }
                }
            }
        }
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SWAP requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();
            self.flush_two_qubit(q1, q2);

            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            for i in 0..self.real.len() {
                let bit1 = (i >> q1) & 1;
                let bit2 = (i >> q2) & 1;
                if bit1 != bit2 && bit1 < bit2 {
                    let j = (i & !mask1 & !mask2) | (bit1 << q2) | (bit2 << q1);
                    self.real.swap(i, j);
                    self.imag.swap(i, j);
                }
            }
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.flush();

        qubits
            .iter()
            .map(|&qubit| {
                let q = qubit.index();
                let mask = 1 << q;

                // Calculate probability of |1>
                let mut prob_one = 0.0f64;
                for i in 0..self.real.len() {
                    if (i & mask) != 0 {
                        let re = f64::from(self.real[i]);
                        let im = f64::from(self.imag[i]);
                        prob_one += re * re + im * im;
                    }
                }

                // Determine outcome
                let random: f64 = self.rng.random();
                let outcome = random < prob_one;
                let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);

                // Collapse and renormalize
                let norm_factor = if outcome {
                    1.0 / prob_one.sqrt()
                } else {
                    1.0 / (1.0 - prob_one).sqrt()
                } as f32;

                for i in 0..self.real.len() {
                    let bit = (i >> q) & 1 == 1;
                    if bit == outcome {
                        self.real[i] *= norm_factor;
                        self.imag[i] *= norm_factor;
                    } else {
                        self.real[i] = 0.0;
                        self.imag[i] = 0.0;
                    }
                }

                MeasurementResult {
                    outcome,
                    is_deterministic,
                }
            })
            .collect()
    }
}

// =============================================================================
// ArbitraryRotationGateable Implementation
// =============================================================================

impl<R> ArbitraryRotationGateable for StateVecSoA32<R>
where
    R: Rng,
{
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let theta = theta as f32;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        let matrix = Complex2x2_32 {
            a_re: cos_half,
            a_im: 0.0,
            b_re: 0.0,
            b_im: -sin_half,
            c_re: 0.0,
            c_im: -sin_half,
            d_re: cos_half,
            d_im: 0.0,
        };

        for &q in qubits {
            self.queue_gate(q.index(), &matrix);
        }
        self
    }

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let theta = theta as f32;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        let matrix = Complex2x2_32 {
            a_re: cos_half,
            a_im: 0.0,
            b_re: -sin_half,
            b_im: 0.0,
            c_re: sin_half,
            c_im: 0.0,
            d_re: cos_half,
            d_im: 0.0,
        };

        for &q in qubits {
            self.queue_gate(q.index(), &matrix);
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let theta = theta as f32;
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();

        let matrix = Complex2x2_32 {
            a_re: cos_half,
            a_im: -sin_half,
            b_re: 0.0,
            b_im: 0.0,
            c_re: 0.0,
            c_im: 0.0,
            d_re: cos_half,
            d_im: sin_half,
        };

        for &q in qubits {
            self.queue_gate(q.index(), &matrix);
        }
        self
    }

    fn rxx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RXX = exp(-i * theta/2 * XX)
        // Decompose: H-H, CX, RZ, CX, H-H
        debug_assert!(qubits.len().is_multiple_of(2));
        for pair in qubits.chunks_exact(2) {
            let q0 = pair[0];
            let q1 = pair[1];
            self.h(&[q0, q1]);
            self.cx(&[q0, q1]);
            self.rz(theta, &[q1]);
            self.cx(&[q0, q1]);
            self.h(&[q0, q1]);
        }
        self
    }

    fn ryy(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RYY decomposition
        debug_assert!(qubits.len().is_multiple_of(2));
        for pair in qubits.chunks_exact(2) {
            let q0 = pair[0];
            let q1 = pair[1];
            self.rx(Angle64::QUARTER_TURN, &[q0, q1]);
            self.cx(&[q0, q1]);
            self.rz(theta, &[q1]);
            self.cx(&[q0, q1]);
            self.rx(-Angle64::QUARTER_TURN, &[q0, q1]);
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RZZ = exp(-i * theta/2 * ZZ)
        debug_assert!(qubits.len().is_multiple_of(2));
        for pair in qubits.chunks_exact(2) {
            let q0 = pair[0];
            let q1 = pair[1];
            self.cx(&[q0, q1]);
            self.rz(theta, &[q1]);
            self.cx(&[q0, q1]);
        }
        self
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn qid(q: usize) -> [QubitId; 1] {
        [QubitId(q)]
    }

    #[test]
    fn test_new() {
        let sim: StateVecSoA32 = StateVecSoA32::new(3);
        assert_eq!(sim.num_qubits(), 3);
        assert_eq!(sim.real.len(), 8);
    }

    #[test]
    fn test_h_gate() {
        let mut sim: StateVecSoA32 = StateVecSoA32::new(1);
        sim.h(&qid(0));
        sim.flush();

        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sim.real[0] - inv_sqrt2).abs() < 1e-6);
        assert!((sim.real[1] - inv_sqrt2).abs() < 1e-6);
    }

    #[test]
    fn test_x_gate() {
        let mut sim: StateVecSoA32 = StateVecSoA32::new(1);
        sim.x(&qid(0));
        sim.flush();

        assert!((sim.real[0]).abs() < 1e-6);
        assert!((sim.real[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cx_gate() {
        let mut sim: StateVecSoA32 = StateVecSoA32::new(2);
        sim.x(&qid(0)); // Set control to |1>
        sim.cx(&[QubitId(0), QubitId(1)]);
        sim.flush();

        // Should be |11>
        assert!((sim.real[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bell_state() {
        let mut sim: StateVecSoA32 = StateVecSoA32::new(2);
        sim.h(&qid(0));
        sim.cx(&[QubitId(0), QubitId(1)]);
        sim.flush();

        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((sim.real[0] - inv_sqrt2).abs() < 1e-5);
        assert!((sim.real[3] - inv_sqrt2).abs() < 1e-5);
        assert!(sim.real[1].abs() < 1e-6);
        assert!(sim.real[2].abs() < 1e-6);
    }

    #[test]
    fn test_fusion() {
        let mut sim: StateVecSoA32 = StateVecSoA32::new(1);
        sim.h(&qid(0));
        sim.z(&qid(0));
        sim.h(&qid(0));
        sim.flush();

        // H-Z-H = X, so |0> -> |1>
        assert!(sim.real[0].abs() < 1e-5);
        assert!((sim.real[1] - 1.0).abs() < 1e-5);
    }
}
