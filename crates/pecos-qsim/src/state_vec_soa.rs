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

//! Optimized State Vector Simulator combining multiple optimization strategies:
//!
//! 1. **SoA Layout**: Separate real and imaginary arrays for SIMD-friendly math
//! 2. **Strided Iteration**: Cache-efficient access patterns for two-qubit gates
//!
//! This simulator prioritizes simple, clean code that the compiler can optimize well.

use crate::clifford_gateable::MeasurementResult;
use crate::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use num_complex::Complex64;
use pecos_core::{QubitId, RngManageable};
use pecos_rng::{PecosRng, Rng, RngCore, SeedableRng};
use std::fmt::Debug;
use wide::f64x4;

// =============================================================================
// SIMD Gate Implementation Macros
// =============================================================================
//
// These macros generate SIMD-optimized single-qubit gate implementations.
// They handle the boilerplate of scalar fallback for small steps and SIMD
// processing for step >= 4.

/// Macro for phase gates that only modify the |1⟩ component.
/// Supported phases: i, neg_i, neg_one
macro_rules! apply_phase_gate_simd {
    // SZ: multiply |1⟩ by i: (re, im) -> (-im, re)
    ($self:expr, $q:expr, i) => {{
        let step = 1 << $q;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re, im) = ($self.real[p], $self.imag[p]);
                    $self.real[p] = -im;
                    $self.imag[p] = re;
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re = f64x4::from(&$self.real[p..p + 4]);
                    let im = f64x4::from(&$self.imag[p..p + 4]);
                    let nr: [f64; 4] = (-im).into();
                    let ni: [f64; 4] = re.into();
                    $self.real[p..p + 4].copy_from_slice(&nr);
                    $self.imag[p..p + 4].copy_from_slice(&ni);
                    j += 4;
                }
            }
        }
    }};
    // SZDG: multiply |1⟩ by -i: (re, im) -> (im, -re)
    ($self:expr, $q:expr, neg_i) => {{
        let step = 1 << $q;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re, im) = ($self.real[p], $self.imag[p]);
                    $self.real[p] = im;
                    $self.imag[p] = -re;
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re = f64x4::from(&$self.real[p..p + 4]);
                    let im = f64x4::from(&$self.imag[p..p + 4]);
                    let nr: [f64; 4] = im.into();
                    let ni: [f64; 4] = (-re).into();
                    $self.real[p..p + 4].copy_from_slice(&nr);
                    $self.imag[p..p + 4].copy_from_slice(&ni);
                    j += 4;
                }
            }
        }
    }};
    // Z: multiply |1⟩ by -1: (re, im) -> (-re, -im)
    ($self:expr, $q:expr, neg_one) => {{
        let step = 1 << $q;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    $self.real[p] = -$self.real[p];
                    $self.imag[p] = -$self.imag[p];
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re = f64x4::from(&$self.real[p..p + 4]);
                    let im = f64x4::from(&$self.imag[p..p + 4]);
                    let nr: [f64; 4] = (-re).into();
                    let ni: [f64; 4] = (-im).into();
                    $self.real[p..p + 4].copy_from_slice(&nr);
                    $self.imag[p..p + 4].copy_from_slice(&ni);
                    j += 4;
                }
            }
        }
    }};
}

/// Macro for full 2x2 gates with real coefficients.
/// Used for: H, and gates where matrix entries are all real.
///
/// Matrix form: [[c00, c01], [c10, c11]]
/// new_a = c00 * a + c01 * b
/// new_b = c10 * a + c11 * b
macro_rules! apply_real_2x2_gate_simd {
    ($self:expr, $q:expr, $c00:expr, $c01:expr, $c10:expr, $c11:expr) => {{
        let step = 1 << $q;
        let (c00, c01, c10, c11) = ($c00, $c01, $c10, $c11);

        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[paired_j], $self.imag[paired_j]);

                    $self.real[j] = c00 * re_a + c01 * re_b;
                    $self.imag[j] = c00 * im_a + c01 * im_b;
                    $self.real[paired_j] = c10 * re_a + c11 * re_b;
                    $self.imag[paired_j] = c10 * im_a + c11 * im_b;
                }
            }
        } else {
            let c00_vec = f64x4::splat(c00);
            let c01_vec = f64x4::splat(c01);
            let c10_vec = f64x4::splat(c10);
            let c11_vec = f64x4::splat(c11);

            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[paired_j..paired_j + 4]);
                    let im_b = f64x4::from(&$self.imag[paired_j..paired_j + 4]);

                    let new_re_a: [f64; 4] = (c00_vec * re_a + c01_vec * re_b).into();
                    let new_im_a: [f64; 4] = (c00_vec * im_a + c01_vec * im_b).into();
                    let new_re_b: [f64; 4] = (c10_vec * re_a + c11_vec * re_b).into();
                    let new_im_b: [f64; 4] = (c10_vec * im_a + c11_vec * im_b).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[paired_j..paired_j + 4].copy_from_slice(&new_re_b);
                    $self.imag[paired_j..paired_j + 4].copy_from_slice(&new_im_b);

                    j += 4;
                }
            }
        }
    }};
}

/// Macro for swap gate (X gate).
/// Swaps |0⟩ ↔ |1⟩ components.
macro_rules! apply_swap_gate_simd {
    ($self:expr, $q:expr) => {{
        let step = 1 << $q;

        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;
                    $self.real.swap(j, paired_j);
                    $self.imag.swap(j, paired_j);
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[paired_j..paired_j + 4]);
                    let im_b = f64x4::from(&$self.imag[paired_j..paired_j + 4]);

                    let arr_re_a: [f64; 4] = re_b.into();
                    let arr_im_a: [f64; 4] = im_b.into();
                    let arr_re_b: [f64; 4] = re_a.into();
                    let arr_im_b: [f64; 4] = im_a.into();

                    $self.real[j..j + 4].copy_from_slice(&arr_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&arr_im_a);
                    $self.real[paired_j..paired_j + 4].copy_from_slice(&arr_re_b);
                    $self.imag[paired_j..paired_j + 4].copy_from_slice(&arr_im_b);

                    j += 4;
                }
            }
        }
    }};
}

/// Macro for Y gate: swap with ±i factors.
/// |0⟩ → i|1⟩, |1⟩ → -i|0⟩
macro_rules! apply_y_gate_simd {
    ($self:expr, $q:expr) => {{
        let step = 1 << $q;

        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let paired_j = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[paired_j], $self.imag[paired_j]);

                    // |0⟩ → i|1⟩ means new_b gets i * old_a = (-im_a, re_a)
                    // |1⟩ → -i|0⟩ means new_a gets -i * old_b = (im_b, -re_b)
                    $self.real[j] = im_b;
                    $self.imag[j] = -re_b;
                    $self.real[paired_j] = -im_a;
                    $self.imag[paired_j] = re_a;
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let paired_j = j + step;

                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[paired_j..paired_j + 4]);
                    let im_b = f64x4::from(&$self.imag[paired_j..paired_j + 4]);

                    // new_a = -i * b = (im_b, -re_b)
                    // new_b = i * a = (-im_a, re_a)
                    let new_re_a: [f64; 4] = im_b.into();
                    let new_im_a: [f64; 4] = (-re_b).into();
                    let new_re_b: [f64; 4] = (-im_a).into();
                    let new_im_b: [f64; 4] = re_a.into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[paired_j..paired_j + 4].copy_from_slice(&new_re_b);
                    $self.imag[paired_j..paired_j + 4].copy_from_slice(&new_im_b);

                    j += 4;
                }
            }
        }
    }};
}

/// Macro for SX/SXDG/SY/SYDG gates with linear combinations.
/// These gates have the form: output = 0.5 * (±re_a ± im_a ± re_b ± im_b)
macro_rules! apply_sqrt_gate_simd {
    // SX: matrix (1/2)[[1+i, 1-i], [1-i, 1+i]]
    ($self:expr, $q:expr, sx) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a + re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b + im_b);
                    $self.real[p] = half * (re_a + im_a + re_b - im_b);
                    $self.imag[p] = half * (-re_a + im_a + re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a + im_a + re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (-re_a + im_a + re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // SXDG: matrix (1/2)[[1-i, 1+i], [1+i, 1-i]]
    ($self:expr, $q:expr, sxdg) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a + im_a + re_b - im_b);
                    $self.imag[j] = half * (-re_a + im_a + re_b + im_b);
                    $self.real[p] = half * (re_a - im_a + re_b + im_b);
                    $self.imag[p] = half * (re_a + im_a - re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a + im_a + re_b - im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a + im_a + re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a + im_a - re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // SY: matrix (1/2)[[1+i, -(1+i)], [1+i, 1+i]]
    ($self:expr, $q:expr, sy) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a - re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b - im_b);
                    $self.real[p] = half * (re_a - im_a + re_b - im_b);
                    $self.imag[p] = half * (re_a + im_a + re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a - re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b - im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a - im_a + re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a + im_a + re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // SYDG: matrix (1/2)[[1-i, 1-i], [-(1-i), 1-i]]
    ($self:expr, $q:expr, sydg) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a + im_a + re_b + im_b);
                    $self.imag[j] = half * (-re_a + im_a - re_b + im_b);
                    $self.real[p] = half * (-re_a - im_a + re_b + im_b);
                    $self.imag[p] = half * (re_a - im_a - re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a + im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a + im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (-re_a - im_a + re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a - im_a - re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F: matrix (1/2)[[1+i, 1-i], [1+i, -1+i]]
    ($self:expr, $q:expr, f) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a + re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b + im_b);
                    $self.real[p] = half * (re_a - im_a - re_b - im_b);
                    $self.imag[p] = half * (re_a + im_a + re_b - im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a - im_a - re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a + im_a + re_b - im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // FDG: matrix (1/2)[[1-i, 1-i], [1+i, -1-i]]
    ($self:expr, $q:expr, fdg) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a + im_a + re_b + im_b);
                    $self.imag[j] = half * (-re_a + im_a - re_b + im_b);
                    $self.real[p] = half * (re_a - im_a - re_b + im_b);
                    $self.imag[p] = half * (re_a + im_a - re_b - im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a + im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a + im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a - im_a - re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a + im_a - re_b - im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // H2: matrix (1/2)[[1+i, -(1+i)], [-(1+i), -(1+i)]]
    ($self:expr, $q:expr, h2) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a - re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b - im_b);
                    $self.real[p] = half * (-re_a + im_a - re_b + im_b);
                    $self.imag[p] = half * (-re_a - im_a - re_b - im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a - re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b - im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (-re_a + im_a - re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (-re_a - im_a - re_b - im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // H5: matrix (1/2)[[1+i, 1-i], [-(1-i), -(1+i)]]
    ($self:expr, $q:expr, h5) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a + re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b + im_b);
                    $self.real[p] = half * (-re_a - im_a - re_b + im_b);
                    $self.imag[p] = half * (re_a - im_a - re_b - im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (-re_a - im_a - re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a - im_a - re_b - im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // H6: matrix (1/2)[[-1-i, 1-i], [-1+i, 1+i]]
    ($self:expr, $q:expr, h6) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (-re_a + im_a + re_b + im_b);
                    $self.imag[j] = half * (-re_a - im_a - re_b + im_b);
                    $self.real[p] = half * (-re_a - im_a + re_b - im_b);
                    $self.imag[p] = half * (re_a - im_a + re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (-re_a + im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a - im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (-re_a - im_a + re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F2: matrix (1/2)[[1-i, -1+i], [1+i, 1+i]]
    ($self:expr, $q:expr, f2) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a + im_a - re_b - im_b);
                    $self.imag[j] = half * (-re_a + im_a + re_b - im_b);
                    $self.real[p] = half * (re_a - im_a + re_b - im_b);
                    $self.imag[p] = half * (re_a + im_a + re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a + im_a - re_b - im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a + im_a + re_b - im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a - im_a + re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a + im_a + re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F2DG: matrix (1/2)[[1+i, 1-i], [-1-i, 1-i]]
    ($self:expr, $q:expr, f2dg) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a + re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b + im_b);
                    $self.real[p] = half * (-re_a + im_a + re_b + im_b);
                    $self.imag[p] = half * (-re_a - im_a - re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (-re_a + im_a + re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (-re_a - im_a - re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F3: matrix (1/2)[[1-i, 1+i], [-1+i, 1+i]]
    ($self:expr, $q:expr, f3) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a + im_a + re_b - im_b);
                    $self.imag[j] = half * (-re_a + im_a + re_b + im_b);
                    $self.real[p] = half * (-re_a - im_a + re_b - im_b);
                    $self.imag[p] = half * (re_a - im_a + re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a + im_a + re_b - im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a + im_a + re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (-re_a - im_a + re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (re_a - im_a + re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F3DG: matrix (1/2)[[1+i, -1-i], [1-i, 1-i]]
    ($self:expr, $q:expr, f3dg) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a - re_b + im_b);
                    $self.imag[j] = half * (re_a + im_a - re_b - im_b);
                    $self.real[p] = half * (re_a + im_a + re_b + im_b);
                    $self.imag[p] = half * (-re_a + im_a - re_b + im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a - re_b + im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a - re_b - im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a + im_a + re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (-re_a + im_a - re_b + im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F4: matrix (1/2)[[1+i, 1+i], [1-i, -1+i]]
    ($self:expr, $q:expr, f4) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a - im_a + re_b - im_b);
                    $self.imag[j] = half * (re_a + im_a + re_b + im_b);
                    $self.real[p] = half * (re_a + im_a - re_b - im_b);
                    $self.imag[p] = half * (-re_a + im_a + re_b - im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a - im_a + re_b - im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (re_a + im_a + re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a + im_a - re_b - im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (-re_a + im_a + re_b - im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // F4DG: matrix (1/2)[[1-i, 1+i], [1-i, -1-i]]
    ($self:expr, $q:expr, f4dg) => {{
        let step = 1 << $q;
        let half = 0.5_f64;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = half * (re_a + im_a + re_b - im_b);
                    $self.imag[j] = half * (-re_a + im_a + re_b + im_b);
                    $self.real[p] = half * (re_a + im_a - re_b + im_b);
                    $self.imag[p] = half * (-re_a + im_a - re_b - im_b);
                }
            }
        } else {
            let half_v = f64x4::splat(half);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (half_v * (re_a + im_a + re_b - im_b)).into();
                    let new_im_a: [f64; 4] = (half_v * (-re_a + im_a + re_b + im_b)).into();
                    let new_re_b: [f64; 4] = (half_v * (re_a + im_a - re_b + im_b)).into();
                    let new_im_b: [f64; 4] = (half_v * (-re_a + im_a - re_b - im_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // HS (S*H): matrix 1/√2 [[1, 1], [i, -i]]
    // new_a = (a + b)/√2, new_b = i*(a - b)/√2
    ($self:expr, $q:expr, hs) => {{
        let step = 1 << $q;
        let k = std::f64::consts::FRAC_1_SQRT_2;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = k * (re_a + re_b);
                    $self.imag[j] = k * (im_a + im_b);
                    $self.real[p] = k * (-im_a + im_b);
                    $self.imag[p] = k * (re_a - re_b);
                }
            }
        } else {
            let k_v = f64x4::splat(k);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (k_v * (re_a + re_b)).into();
                    let new_im_a: [f64; 4] = (k_v * (im_a + im_b)).into();
                    let new_re_b: [f64; 4] = (k_v * (-im_a + im_b)).into();
                    let new_im_b: [f64; 4] = (k_v * (re_a - re_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // SH (H*S): matrix 1/√2 [[1, i], [1, -i]]
    // new_a = (a + i*b)/√2, new_b = (a - i*b)/√2
    ($self:expr, $q:expr, sh) => {{
        let step = 1 << $q;
        let k = std::f64::consts::FRAC_1_SQRT_2;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = k * (re_a - im_b);
                    $self.imag[j] = k * (im_a + re_b);
                    $self.real[p] = k * (re_a + im_b);
                    $self.imag[p] = k * (im_a - re_b);
                }
            }
        } else {
            let k_v = f64x4::splat(k);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (k_v * (re_a - im_b)).into();
                    let new_im_a: [f64; 4] = (k_v * (im_a + re_b)).into();
                    let new_re_b: [f64; 4] = (k_v * (re_a + im_b)).into();
                    let new_im_b: [f64; 4] = (k_v * (im_a - re_b)).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
}

/// Macro for H3/H4 gates - swap with phase multiplication.
macro_rules! apply_swap_phase_gate_simd {
    // H3: [[0, 1], [i, 0]] - new_a = b, new_b = i*a
    ($self:expr, $q:expr, h3) => {{
        let step = 1 << $q;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    // new_a = b
                    // new_b = i*a = (-im_a, re_a)
                    $self.real[j] = re_b;
                    $self.imag[j] = im_b;
                    $self.real[p] = -im_a;
                    $self.imag[p] = re_a;
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    // new_a = b, new_b = i*a
                    let new_re_a: [f64; 4] = re_b.into();
                    let new_im_a: [f64; 4] = im_b.into();
                    let new_re_b: [f64; 4] = (-im_a).into();
                    let new_im_b: [f64; 4] = re_a.into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
    // H4: [[0, i], [1, 0]] - new_a = i*b, new_b = a
    ($self:expr, $q:expr, h4) => {{
        let step = 1 << $q;
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    // new_a = i*b = (-im_b, re_b)
                    // new_b = a
                    $self.real[j] = -im_b;
                    $self.imag[j] = re_b;
                    $self.real[p] = re_a;
                    $self.imag[p] = im_a;
                }
            }
        } else {
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    // new_a = i*b, new_b = a
                    let new_re_a: [f64; 4] = (-im_b).into();
                    let new_im_a: [f64; 4] = re_b.into();
                    let new_re_b: [f64; 4] = re_a.into();
                    let new_im_b: [f64; 4] = im_a.into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
}

/// Macro for RX gate with precomputed cos/sin.
/// RX: [[cos, -i*sin], [-i*sin, cos]]
macro_rules! apply_rx_simd {
    ($self:expr, $q:expr, $cos:expr, $sin:expr) => {{
        let step = 1 << $q;
        let (cos, sin) = ($cos, $sin);
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = cos * re_a + sin * im_b;
                    $self.imag[j] = cos * im_a - sin * re_b;
                    $self.real[p] = sin * im_a + cos * re_b;
                    $self.imag[p] = -sin * re_a + cos * im_b;
                }
            }
        } else {
            let cos_v = f64x4::splat(cos);
            let sin_v = f64x4::splat(sin);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (cos_v * re_a + sin_v * im_b).into();
                    let new_im_a: [f64; 4] = (cos_v * im_a - sin_v * re_b).into();
                    let new_re_b: [f64; 4] = (sin_v * im_a + cos_v * re_b).into();
                    let new_im_b: [f64; 4] = (cos_v * im_b - sin_v * re_a).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
}

/// Macro for RY gate with precomputed cos/sin.
/// RY: [[cos, -sin], [sin, cos]]
macro_rules! apply_ry_simd {
    ($self:expr, $q:expr, $cos:expr, $sin:expr) => {{
        let step = 1 << $q;
        let (cos, sin) = ($cos, $sin);
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = cos * re_a - sin * re_b;
                    $self.imag[j] = cos * im_a - sin * im_b;
                    $self.real[p] = sin * re_a + cos * re_b;
                    $self.imag[p] = sin * im_a + cos * im_b;
                }
            }
        } else {
            let cos_v = f64x4::splat(cos);
            let sin_v = f64x4::splat(sin);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (cos_v * re_a - sin_v * re_b).into();
                    let new_im_a: [f64; 4] = (cos_v * im_a - sin_v * im_b).into();
                    let new_re_b: [f64; 4] = (sin_v * re_a + cos_v * re_b).into();
                    let new_im_b: [f64; 4] = (sin_v * im_a + cos_v * im_b).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
}

/// Macro for RZ gate with precomputed cos/sin for both phases.
/// RZ: [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
macro_rules! apply_rz_simd {
    ($self:expr, $q:expr, $cos_neg:expr, $sin_neg:expr, $cos_pos:expr, $sin_pos:expr) => {{
        let step = 1 << $q;
        let (cos_neg, sin_neg, cos_pos, sin_pos) = ($cos_neg, $sin_neg, $cos_pos, $sin_pos);
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = cos_neg * re_a - sin_neg * im_a;
                    $self.imag[j] = sin_neg * re_a + cos_neg * im_a;
                    $self.real[p] = cos_pos * re_b - sin_pos * im_b;
                    $self.imag[p] = sin_pos * re_b + cos_pos * im_b;
                }
            }
        } else {
            let cos_neg_v = f64x4::splat(cos_neg);
            let sin_neg_v = f64x4::splat(sin_neg);
            let cos_pos_v = f64x4::splat(cos_pos);
            let sin_pos_v = f64x4::splat(sin_pos);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (cos_neg_v * re_a - sin_neg_v * im_a).into();
                    let new_im_a: [f64; 4] = (sin_neg_v * re_a + cos_neg_v * im_a).into();
                    let new_re_b: [f64; 4] = (cos_pos_v * re_b - sin_pos_v * im_b).into();
                    let new_im_b: [f64; 4] = (sin_pos_v * re_b + cos_pos_v * im_b).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
}

/// Macro for R1XY gate with precomputed coefficients.
/// R1XY: [[cos, r01], [r10, cos]] where r01, r10 are complex
macro_rules! apply_r1xy_simd {
    ($self:expr, $q:expr, $cos:expr, $r01_re:expr, $r01_im:expr, $r10_re:expr, $r10_im:expr) => {{
        let step = 1 << $q;
        let (cos, r01_re, r01_im, r10_re, r10_im) = ($cos, $r01_re, $r01_im, $r10_re, $r10_im);
        if step < 4 {
            for i in (0..$self.real.len()).step_by(step * 2) {
                for j in i..(i + step) {
                    let p = j + step;
                    let (re_a, im_a) = ($self.real[j], $self.imag[j]);
                    let (re_b, im_b) = ($self.real[p], $self.imag[p]);
                    $self.real[j] = cos * re_a + r01_re * re_b - r01_im * im_b;
                    $self.imag[j] = cos * im_a + r01_re * im_b + r01_im * re_b;
                    $self.real[p] = r10_re * re_a - r10_im * im_a + cos * re_b;
                    $self.imag[p] = r10_re * im_a + r10_im * re_a + cos * im_b;
                }
            }
        } else {
            let cos_v = f64x4::splat(cos);
            let r01_re_v = f64x4::splat(r01_re);
            let r01_im_v = f64x4::splat(r01_im);
            let r10_re_v = f64x4::splat(r10_re);
            let r10_im_v = f64x4::splat(r10_im);
            for i in (0..$self.real.len()).step_by(step * 2) {
                let mut j = i;
                while j + 4 <= i + step {
                    let p = j + step;
                    let re_a = f64x4::from(&$self.real[j..j + 4]);
                    let im_a = f64x4::from(&$self.imag[j..j + 4]);
                    let re_b = f64x4::from(&$self.real[p..p + 4]);
                    let im_b = f64x4::from(&$self.imag[p..p + 4]);

                    let new_re_a: [f64; 4] = (cos_v * re_a + r01_re_v * re_b - r01_im_v * im_b).into();
                    let new_im_a: [f64; 4] = (cos_v * im_a + r01_re_v * im_b + r01_im_v * re_b).into();
                    let new_re_b: [f64; 4] = (r10_re_v * re_a - r10_im_v * im_a + cos_v * re_b).into();
                    let new_im_b: [f64; 4] = (r10_re_v * im_a + r10_im_v * re_a + cos_v * im_b).into();

                    $self.real[j..j + 4].copy_from_slice(&new_re_a);
                    $self.imag[j..j + 4].copy_from_slice(&new_im_a);
                    $self.real[p..p + 4].copy_from_slice(&new_re_b);
                    $self.imag[p..p + 4].copy_from_slice(&new_im_b);
                    j += 4;
                }
            }
        }
    }};
}

/// Optimized state vector simulator with SoA layout.
#[derive(Debug, Clone)]
pub struct StateVecSoA<R = PecosRng>
where
    R: Rng,
{
    /// Real components of the state vector
    pub(crate) real: Vec<f64>,
    /// Imaginary components of the state vector
    pub(crate) imag: Vec<f64>,
    /// Number of qubits
    num_qubits: usize,
    /// Random number generator for measurements
    rng: R,
    /// Scratch buffer for real components (used by two_qubit_unitary)
    scratch_real: Vec<f64>,
    /// Scratch buffer for imaginary components (used by two_qubit_unitary)
    scratch_imag: Vec<f64>,
}

// Constructors that use the default PecosRng
impl StateVecSoA {
    /// Creates a new state vector initialized to |0...0⟩.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> StateVecSoA<PecosRng> {
        let rng = PecosRng::from_os_rng();
        StateVecSoA::with_rng(num_qubits, rng)
    }

    /// Creates a new state vector with a specific seed for reproducibility.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> StateVecSoA<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        StateVecSoA::with_rng(num_qubits, rng)
    }
}

impl StateVecSoA<PecosRng> {
    /// Sets the random seed for measurements.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = PecosRng::seed_from_u64(seed);
    }
}

impl<R> StateVecSoA<R>
where
    R: Rng,
{
    /// Creates a new state vector with a custom RNG.
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let size = 1 << num_qubits;
        let mut real = vec![0.0; size];
        let imag = vec![0.0; size];
        real[0] = 1.0; // |0...0⟩ state

        Self {
            real,
            imag,
            num_qubits,
            rng,
            scratch_real: vec![0.0; size],
            scratch_imag: vec![0.0; size],
        }
    }

    /// Returns a reference to the real components.
    #[inline]
    #[must_use]
    pub fn real(&self) -> &[f64] {
        &self.real
    }

    /// Returns a reference to the imaginary components.
    #[inline]
    #[must_use]
    pub fn imag(&self) -> &[f64] {
        &self.imag
    }

    /// Prepare a specific computational basis state |n⟩.
    #[inline]
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        for r in &mut self.real {
            *r = 0.0;
        }
        for i in &mut self.imag {
            *i = 0.0;
        }
        self.real[basis_state] = 1.0;
        self
    }

    /// Returns the number of qubits in the state vector.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the probability of measuring a specific computational basis state.
    ///
    /// The probability is calculated as |amplitude|^2 for the given basis state.
    #[inline]
    #[must_use]
    pub fn probability(&self, basis_state: usize) -> f64 {
        let re = self.real[basis_state];
        let im = self.imag[basis_state];
        re * re + im * im
    }

    /// Returns the amplitude at the given basis state index as a Complex64.
    #[inline]
    #[must_use]
    pub fn get_amplitude(&self, index: usize) -> Complex64 {
        Complex64::new(self.real[index], self.imag[index])
    }

    /// Sets the amplitude at the given basis state index.
    #[inline]
    pub fn set_amplitude(&mut self, index: usize, value: Complex64) {
        self.real[index] = value.re;
        self.imag[index] = value.im;
    }

    /// Returns the state vector as a Vec of Complex64 for inspection.
    ///
    /// This creates a new vector by combining the real and imaginary arrays.
    #[must_use]
    pub fn to_complex_vec(&self) -> Vec<Complex64> {
        self.real
            .iter()
            .zip(&self.imag)
            .map(|(&re, &im)| Complex64::new(re, im))
            .collect()
    }

    /// Creates a state vector from a Vec of Complex64.
    ///
    /// The length of the state vector must be a power of 2.
    #[must_use]
    pub fn from_complex_state(state: Vec<Complex64>, rng: R) -> Self {
        let num_qubits = state.len().trailing_zeros() as usize;
        let size = state.len();
        assert_eq!(1 << num_qubits, size, "Invalid state vector size");

        let real: Vec<f64> = state.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = state.iter().map(|c| c.im).collect();

        Self {
            real,
            imag,
            num_qubits,
            rng,
            scratch_real: vec![0.0; size],
            scratch_imag: vec![0.0; size],
        }
    }

    /// Creates a state vector from a Vec of Complex64.
    ///
    /// Alias for `from_complex_state` for API compatibility.
    #[must_use]
    pub fn from_state(state: Vec<Complex64>, rng: R) -> Self {
        Self::from_complex_state(state, rng)
    }

    /// Returns the state vector as a Vec of Complex64.
    ///
    /// This creates a new vector by combining the real and imaginary arrays.
    /// Alias for `to_complex_vec` for API compatibility.
    #[must_use]
    pub fn state(&self) -> Vec<Complex64> {
        self.to_complex_vec()
    }

    /// Returns a reference to the random number generator.
    #[inline]
    #[must_use]
    pub fn rng(&self) -> &R {
        &self.rng
    }

    /// Prepare all qubits in the |+⟩ state, creating an equal superposition of all basis states.
    ///
    /// This operation prepares the state (1/√2^n)(|0...0⟩ + |0...1⟩ + ... + |1...1⟩)
    /// where n is the number of qubits.
    #[inline]
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        let factor = 1.0 / ((1 << self.num_qubits) as f64).sqrt();
        self.real.fill(factor);
        self.imag.fill(0.0);
        self
    }

    /// Apply a general single-qubit unitary gate given by a 2x2 complex matrix.
    ///
    /// The matrix elements are:
    /// ```text
    /// U = [[u00, u01],
    ///      [u10, u11]]
    /// ```
    ///
    /// # Example
    /// ```
    /// use pecos_qsim::StateVecSoA;
    /// use num_complex::Complex64;
    /// use std::f64::consts::FRAC_1_SQRT_2;
    ///
    /// let mut sim = StateVecSoA::new(1);
    /// // Apply Hadamard gate
    /// sim.single_qubit_unitary(0,
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u00
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u01
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u10
    ///     Complex64::new(-FRAC_1_SQRT_2, 0.0), // u11
    /// );
    /// ```
    #[inline]
    pub fn single_qubit_unitary(
        &mut self,
        qubit: usize,
        u00: Complex64,
        u01: Complex64,
        u10: Complex64,
        u11: Complex64,
    ) -> &mut Self {
        let step = 1 << qubit;
        for i in (0..self.real.len()).step_by(2 * step) {
            for offset in 0..step {
                let j = i + offset;
                let k = j ^ step;

                let a_re = self.real[j];
                let a_im = self.imag[j];
                let b_re = self.real[k];
                let b_im = self.imag[k];

                // new_j = u00 * a + u01 * b
                self.real[j] = u00.re * a_re - u00.im * a_im + u01.re * b_re - u01.im * b_im;
                self.imag[j] = u00.re * a_im + u00.im * a_re + u01.re * b_im + u01.im * b_re;

                // new_k = u10 * a + u11 * b
                self.real[k] = u10.re * a_re - u10.im * a_im + u11.re * b_re - u11.im * b_im;
                self.imag[k] = u10.re * a_im + u10.im * a_re + u11.re * b_im + u11.im * b_re;
            }
        }
        self
    }

    /// Apply a general two-qubit unitary gate given by a 4x4 complex matrix.
    ///
    /// The matrix is indexed as:
    /// ```text
    /// U = [[u[0][0], u[0][1], u[0][2], u[0][3]],
    ///      [u[1][0], u[1][1], u[1][2], u[1][3]],
    ///      [u[2][0], u[2][1], u[2][2], u[2][3]],
    ///      [u[3][0], u[3][1], u[3][2], u[3][3]]]
    /// ```
    ///
    /// where rows/columns correspond to basis states |00⟩, |01⟩, |10⟩, |11⟩.
    #[inline]
    pub fn two_qubit_unitary(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        matrix: [[Complex64; 4]; 4],
    ) -> &mut Self {
        let size = self.real.len();

        // Ensure consistent ordering for strided iteration
        let (lo, hi) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        let step_lo = 1 << lo;
        let step_hi = 1 << hi;

        // The matrix is indexed as matrix[output_basis][input_basis]
        // where basis_idx = (qubit1_bit << 1) | qubit2_bit
        //
        // Our iteration uses (lo_bit, hi_bit) ordering:
        // - idx 0: lo=0, hi=0
        // - idx 1: lo=1, hi=0
        // - idx 2: lo=0, hi=1
        // - idx 3: lo=1, hi=1
        //
        // When qubit1 < qubit2 (qubit1 is lo, qubit2 is hi):
        //   lo_bit = qubit1_bit, hi_bit = qubit2_bit
        //   our_idx -> basis_idx: 0->0, 1->2, 2->1, 3->3
        //
        // When qubit1 > qubit2 (qubit2 is lo, qubit1 is hi):
        //   lo_bit = qubit2_bit, hi_bit = qubit1_bit
        //   our_idx -> basis_idx: 0->0, 1->1, 2->2, 3->3 (identity)

        // Permutation from our iteration order to matrix basis order
        let perm: [usize; 4] = if qubit1 < qubit2 {
            [0, 2, 1, 3] // swap indices 1 and 2
        } else {
            [0, 1, 2, 3] // identity
        };

        // Process groups of 4 basis states that share the same "frame" bits
        for outer in (0..size).step_by(step_hi * 2) {
            for mid in (0..step_hi).step_by(step_lo * 2) {
                for inner in 0..step_lo {
                    let base = outer + mid + inner;

                    // The 4 indices in (lo_bit, hi_bit) order
                    let indices = [
                        base,                       // lo=0, hi=0
                        base + step_lo,             // lo=1, hi=0
                        base + step_hi,             // lo=0, hi=1
                        base + step_hi + step_lo,   // lo=1, hi=1
                    ];

                    // Load the 4 amplitudes in matrix basis order
                    let a = [
                        (self.real[indices[perm[0]]], self.imag[indices[perm[0]]]),
                        (self.real[indices[perm[1]]], self.imag[indices[perm[1]]]),
                        (self.real[indices[perm[2]]], self.imag[indices[perm[2]]]),
                        (self.real[indices[perm[3]]], self.imag[indices[perm[3]]]),
                    ];

                    // Apply the 4x4 matrix: new[j] = sum_k matrix[j][k] * old[k]
                    for (j, row) in matrix.iter().enumerate() {
                        let mut new_re = 0.0;
                        let mut new_im = 0.0;
                        for (k, &(amp_re, amp_im)) in a.iter().enumerate() {
                            let m = row[k];
                            new_re += m.re * amp_re - m.im * amp_im;
                            new_im += m.re * amp_im + m.im * amp_re;
                        }
                        // Write to the correct index using inverse permutation
                        self.scratch_real[indices[perm[j]]] = new_re;
                        self.scratch_imag[indices[perm[j]]] = new_im;
                    }
                }
            }
        }

        // Swap buffers (avoids copying)
        std::mem::swap(&mut self.real, &mut self.scratch_real);
        std::mem::swap(&mut self.imag, &mut self.scratch_imag);
        self
    }

    /// Apply a single-qubit gate using a closure that transforms (re_0, im_0, re_1, im_1).
    #[inline]
    fn apply_single_qubit<F>(&mut self, q: usize, mut f: F)
    where
        F: FnMut(f64, f64, f64, f64) -> (f64, f64, f64, f64),
    {
        let step = 1 << q;
        for i in (0..self.real.len()).step_by(step * 2) {
            for j in i..(i + step) {
                let paired_j = j + step;
                let (new_re_j, new_im_j, new_re_p, new_im_p) = f(
                    self.real[j],
                    self.imag[j],
                    self.real[paired_j],
                    self.imag[paired_j],
                );
                self.real[j] = new_re_j;
                self.imag[j] = new_im_j;
                self.real[paired_j] = new_re_p;
                self.imag[paired_j] = new_im_p;
            }
        }
    }

    // =========================================================================
    // SIMD-optimized single-qubit gate implementations using macros
    // =========================================================================

    /// Apply Hadamard gate with SIMD: H = 1/√2 [[1, 1], [1, -1]]
    #[inline]
    fn apply_h_simd(&mut self, q: usize) {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        apply_real_2x2_gate_simd!(self, q, k, k, k, -k);
    }

    /// Apply SZ gate with SIMD: |1⟩ → i|1⟩
    #[inline]
    fn apply_sz_simd(&mut self, q: usize) {
        // i * (re, im) = (-im, re)
        apply_phase_gate_simd!(self, q, i);
    }

    /// Apply SZDG gate with SIMD: |1⟩ → -i|1⟩
    #[inline]
    fn apply_szdg_simd(&mut self, q: usize) {
        // -i * (re, im) = (im, -re)
        apply_phase_gate_simd!(self, q, neg_i);
    }

    /// Apply X gate with SIMD: swap |0⟩ ↔ |1⟩
    #[inline]
    fn apply_x_simd(&mut self, q: usize) {
        apply_swap_gate_simd!(self, q);
    }

    /// Apply Y gate with SIMD: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    #[inline]
    fn apply_y_simd(&mut self, q: usize) {
        apply_y_gate_simd!(self, q);
    }

    /// Apply Z gate with SIMD: |1⟩ → -|1⟩
    #[inline]
    fn apply_z_simd(&mut self, q: usize) {
        // -1 * (re, im) = (-re, -im)
        apply_phase_gate_simd!(self, q, neg_one);
    }

    /// Apply SX gate with SIMD
    #[inline]
    fn apply_sx_simd(&mut self, q: usize) {
        apply_sqrt_gate_simd!(self, q, sx);
    }

    /// Apply SXDG gate with SIMD
    #[inline]
    fn apply_sxdg_simd(&mut self, q: usize) {
        apply_sqrt_gate_simd!(self, q, sxdg);
    }

    /// Apply SY gate with SIMD
    #[inline]
    fn apply_sy_simd(&mut self, q: usize) {
        apply_sqrt_gate_simd!(self, q, sy);
    }

    /// Apply SYDG gate with SIMD
    #[inline]
    fn apply_sydg_simd(&mut self, q: usize) {
        apply_sqrt_gate_simd!(self, q, sydg);
    }

    /// Apply F gate with SIMD
    #[inline]
    fn apply_f_simd(&mut self, q: usize) {
        apply_sqrt_gate_simd!(self, q, f);
    }

    /// Apply FDG gate with SIMD
    #[inline]
    fn apply_fdg_simd(&mut self, q: usize) {
        apply_sqrt_gate_simd!(self, q, fdg);
    }

    /// Compute the probability of measuring |1⟩ for a qubit using SIMD.
    ///
    /// This sums |amplitude|^2 for all basis states where the given qubit is |1⟩.
    #[inline]
    fn probability_one(&self, qubit: usize) -> f64 {
        let step = 1 << qubit;

        // For small step sizes, use scalar to avoid SIMD overhead
        if step < 4 {
            let mut prob = 0.0;
            for i in (0..self.real.len()).step_by(step * 2) {
                for j in (i + step)..(i + 2 * step) {
                    prob += self.real[j] * self.real[j] + self.imag[j] * self.imag[j];
                }
            }
            return prob;
        }

        // SIMD accumulator
        let mut acc = f64x4::ZERO;

        for i in (0..self.real.len()).step_by(step * 2) {
            let mut j = i + step;
            // Process 4 elements at a time
            while j + 4 <= i + 2 * step {
                let re = f64x4::from(&self.real[j..j + 4]);
                let im = f64x4::from(&self.imag[j..j + 4]);
                acc += re * re + im * im;
                j += 4;
            }
            // Handle remainder (step is power of 2 and >= 4, so remainder is 0)
        }

        // Horizontal sum
        let vals: [f64; 4] = acc.into();
        vals[0] + vals[1] + vals[2] + vals[3]
    }
}

impl<R> QuantumSimulator for StateVecSoA<R>
where
    R: Rng,
{
    fn reset(&mut self) -> &mut Self {
        for r in &mut self.real {
            *r = 0.0;
        }
        for i in &mut self.imag {
            *i = 0.0;
        }
        self.real[0] = 1.0;
        self
    }
}

impl<R> CliffordGateable for StateVecSoA<R>
where
    R: Rng,
{
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_h_simd(q.index());
        }
        self
    }

    #[inline]
    fn h2(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H2 = Z * SY = (1/2)[[1+i, -(1+i)], [-(1+i), -(1+i)]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), h2);
        }
        self
    }

    #[inline]
    fn h3(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H3 = Y * SZ = [[0, 1], [i, 0]]
        for &q in qubits {
            apply_swap_phase_gate_simd!(self, q.index(), h3);
        }
        self
    }

    #[inline]
    fn h4(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H4 = X * SZ = [[0, i], [1, 0]]
        for &q in qubits {
            apply_swap_phase_gate_simd!(self, q.index(), h4);
        }
        self
    }

    #[inline]
    fn h5(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H5 = Z * SX = (1/2)[[1+i, 1-i], [-(1-i), -(1+i)]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), h5);
        }
        self
    }

    #[inline]
    fn h6(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H6 = Y * SX = (1/2)[[-1-i, 1-i], [-1+i, 1+i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), h6);
        }
        self
    }

    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_x_simd(q.index());
        }
        self
    }

    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_y_simd(q.index());
        }
        self
    }

    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_z_simd(q.index());
        }
        self
    }

    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sz_simd(q.index());
        }
        self
    }

    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_szdg_simd(q.index());
        }
        self
    }

    #[inline]
    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sx_simd(q.index());
        }
        self
    }

    #[inline]
    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sxdg_simd(q.index());
        }
        self
    }

    #[inline]
    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sy_simd(q.index());
        }
        self
    }

    #[inline]
    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_sydg_simd(q.index());
        }
        self
    }

    #[inline]
    fn f(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_f_simd(q.index());
        }
        self
    }

    #[inline]
    fn fdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.apply_fdg_simd(q.index());
        }
        self
    }

    #[inline]
    fn f2(&mut self, qubits: &[QubitId]) -> &mut Self {
        // F2 = SY * SXDG = (1/2)[[1-i, -1+i], [1+i, 1+i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), f2);
        }
        self
    }

    #[inline]
    fn f2dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // F2DG = SX * SYDG = (1/2)[[1+i, 1-i], [-1-i, 1-i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), f2dg);
        }
        self
    }

    #[inline]
    fn f3(&mut self, qubits: &[QubitId]) -> &mut Self {
        // F3 = SZ * SXDG = (1/2)[[1-i, 1+i], [-1+i, 1+i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), f3);
        }
        self
    }

    #[inline]
    fn f3dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // F3DG = SX * SZDG = (1/2)[[1+i, -1-i], [1-i, 1-i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), f3dg);
        }
        self
    }

    #[inline]
    fn f4(&mut self, qubits: &[QubitId]) -> &mut Self {
        // F4 = SX * SZ = (1/2)[[1+i, 1+i], [1-i, -1+i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), f4);
        }
        self
    }

    #[inline]
    fn f4dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // F4DG = SZDG * SXDG = (1/2)[[1-i, 1+i], [1-i, -1-i]]
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), f4dg);
        }
        self
    }

    #[inline]
    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let control = pair[0].index();
            let target = pair[1].index();

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

            // When q_lo >= 2, indices are contiguous and we can use SIMD
            if step_lo >= 4 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx0 = base | control_mask;
                            let idx1 = idx0 | target_mask;

                            // Load both sets
                            let re0 = f64x4::from(&self.real[idx0..idx0 + 4]);
                            let im0 = f64x4::from(&self.imag[idx0..idx0 + 4]);
                            let re1 = f64x4::from(&self.real[idx1..idx1 + 4]);
                            let im1 = f64x4::from(&self.imag[idx1..idx1 + 4]);

                            // Swap by storing in opposite locations
                            let arr_re0: [f64; 4] = re1.into();
                            let arr_im0: [f64; 4] = im1.into();
                            let arr_re1: [f64; 4] = re0.into();
                            let arr_im1: [f64; 4] = im0.into();

                            self.real[idx0..idx0 + 4].copy_from_slice(&arr_re0);
                            self.imag[idx0..idx0 + 4].copy_from_slice(&arr_im0);
                            self.real[idx1..idx1 + 4].copy_from_slice(&arr_re1);
                            self.imag[idx1..idx1 + 4].copy_from_slice(&arr_im1);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step_lo
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_c1_t0 = base | control_mask;
                            let idx_c1_t1 = idx_c1_t0 | target_mask;

                            self.real.swap(idx_c1_t0, idx_c1_t1);
                            self.imag.swap(idx_c1_t0, idx_c1_t1);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CZ requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask_11 = (1 << q1) | (1 << q2);

            // When q_lo >= 2, indices are contiguous and we can use SIMD
            if step_lo >= 4 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx = base | mask_11;
                            let re = f64x4::from(&self.real[idx..idx + 4]);
                            let im = f64x4::from(&self.imag[idx..idx + 4]);
                            let neg_re: [f64; 4] = (-re).into();
                            let neg_im: [f64; 4] = (-im).into();
                            self.real[idx..idx + 4].copy_from_slice(&neg_re);
                            self.imag[idx..idx + 4].copy_from_slice(&neg_im);
                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step_lo
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_11 = base | mask_11;
                            self.real[idx_11] = -self.real[idx_11];
                            self.imag[idx_11] = -self.imag[idx_11];
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SWAP requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask_01 = 1 << q2;
            let mask_10 = 1 << q1;

            // When q_lo >= 2, indices are contiguous and we can use SIMD
            if step_lo >= 4 {
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx_01 = base | mask_01;
                            let idx_10 = base | mask_10;

                            let re01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);

                            let arr_re01: [f64; 4] = re10.into();
                            let arr_im01: [f64; 4] = im10.into();
                            let arr_re10: [f64; 4] = re01.into();
                            let arr_im10: [f64; 4] = im01.into();

                            self.real[idx_01..idx_01 + 4].copy_from_slice(&arr_re01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&arr_im01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&arr_re10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&arr_im10);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_01 = base | mask_01;
                            let idx_10 = base | mask_10;
                            self.real.swap(idx_01, idx_10);
                            self.imag.swap(idx_01, idx_10);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn cy(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CY requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let control = pair[0].index();
            let target = pair[1].index();

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

            // CY = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Y
            // When control=1: apply Y to target
            // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩

            if step_lo >= 4 {
                // SIMD version: process 4 elements at a time
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        // Check if target bit is set in i_lo - if so, skip this entire block
                        let test_idx = i_lo | control_mask;
                        if (test_idx & target_mask) != 0 {
                            continue;
                        }

                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            let idx_c1_t0 = base | control_mask;
                            let idx_c1_t1 = idx_c1_t0 | target_mask;

                            let re_t0 = f64x4::from(&self.real[idx_c1_t0..idx_c1_t0 + 4]);
                            let im_t0 = f64x4::from(&self.imag[idx_c1_t0..idx_c1_t0 + 4]);
                            let re_t1 = f64x4::from(&self.real[idx_c1_t1..idx_c1_t1 + 4]);
                            let im_t1 = f64x4::from(&self.imag[idx_c1_t1..idx_c1_t1 + 4]);

                            // new |t0⟩ = -i * old |t1⟩: -i * (re, im) = (im, -re)
                            let new_re_t0: [f64; 4] = im_t1.into();
                            let new_im_t0: [f64; 4] = (-re_t1).into();

                            // new |t1⟩ = i * old |t0⟩: i * (re, im) = (-im, re)
                            let new_re_t1: [f64; 4] = (-im_t0).into();
                            let new_im_t1: [f64; 4] = re_t0.into();

                            self.real[idx_c1_t0..idx_c1_t0 + 4].copy_from_slice(&new_re_t0);
                            self.imag[idx_c1_t0..idx_c1_t0 + 4].copy_from_slice(&new_im_t0);
                            self.real[idx_c1_t1..idx_c1_t1 + 4].copy_from_slice(&new_re_t1);
                            self.imag[idx_c1_t1..idx_c1_t1 + 4].copy_from_slice(&new_im_t1);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_c1_t0 = base | control_mask;

                            // Skip if target bit already set (we handle pairs)
                            if (idx_c1_t0 & target_mask) != 0 {
                                continue;
                            }

                            let idx_c1_t1 = idx_c1_t0 | target_mask;

                            let re_t0 = self.real[idx_c1_t0];
                            let im_t0 = self.imag[idx_c1_t0];
                            let re_t1 = self.real[idx_c1_t1];
                            let im_t1 = self.imag[idx_c1_t1];

                            // new |t0⟩ = -i * old |t1⟩
                            self.real[idx_c1_t0] = im_t1;
                            self.imag[idx_c1_t0] = -re_t1;

                            // new |t1⟩ = i * old |t0⟩
                            self.real[idx_c1_t1] = -im_t0;
                            self.imag[idx_c1_t1] = re_t0;
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn sxx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SXX requires pairs of qubits"
        );

        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SXX = exp(-i * π/4 * X⊗X) = (1/√2)(I - i*X⊗X)
            // Matrix: (1/√2) * [[1, 0, 0, -i], [0, 1, -i, 0], [0, -i, 1, 0], [-i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 + im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 - re_11)).into();

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 + im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 - re_10)).into();

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 + im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 - re_01)).into();

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 + im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 - re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            // Only process each quartet once
                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            self.real[idx_00] = K * (re_00 + im_11);
                            self.imag[idx_00] = K * (im_00 - re_11);

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            self.real[idx_01] = K * (re_01 + im_10);
                            self.imag[idx_01] = K * (im_01 - re_10);

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            self.real[idx_10] = K * (re_10 + im_01);
                            self.imag[idx_10] = K * (im_10 - re_01);

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            self.real[idx_11] = K * (re_11 + im_00);
                            self.imag[idx_11] = K * (im_11 - re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn sxxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SXXDG requires pairs of qubits"
        );

        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SXXDG = exp(+i * π/4 * X⊗X) = (1/√2)(I + i*X⊗X)
            // Matrix: (1/√2) * [[1, 0, 0, i], [0, 1, i, 0], [0, i, 1, 0], [i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 - im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 + re_11)).into();

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 - im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 + re_10)).into();

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 - im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 + re_01)).into();

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 - im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 + re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            self.real[idx_00] = K * (re_00 - im_11);
                            self.imag[idx_00] = K * (im_00 + re_11);

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            self.real[idx_01] = K * (re_01 - im_10);
                            self.imag[idx_01] = K * (im_01 + re_10);

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            self.real[idx_10] = K * (re_10 - im_01);
                            self.imag[idx_10] = K * (im_10 + re_01);

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            self.real[idx_11] = K * (re_11 - im_00);
                            self.imag[idx_11] = K * (im_11 + re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn syy(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SYY requires pairs of qubits"
        );

        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SYY = exp(-i * π/4 * Y⊗Y) = (1/√2)(I - i*Y⊗Y)
            // Y⊗Y swaps |00⟩↔-|11⟩ and |01⟩↔|10⟩
            // Matrix: (1/√2) * [[1, 0, 0, i], [0, 1, -i, 0], [0, -i, 1, 0], [i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 - im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 + re_11)).into();

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 + im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 - re_10)).into();

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 + im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 - re_01)).into();

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 - im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 + re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ + i*|11⟩)
                            self.real[idx_00] = K * (re_00 - im_11);
                            self.imag[idx_00] = K * (im_00 + re_11);

                            // new_01 = K * (|01⟩ - i*|10⟩)
                            self.real[idx_01] = K * (re_01 + im_10);
                            self.imag[idx_01] = K * (im_01 - re_10);

                            // new_10 = K * (|10⟩ - i*|01⟩)
                            self.real[idx_10] = K * (re_10 + im_01);
                            self.imag[idx_10] = K * (im_10 - re_01);

                            // new_11 = K * (|11⟩ + i*|00⟩)
                            self.real[idx_11] = K * (re_11 - im_00);
                            self.imag[idx_11] = K * (im_11 + re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn syydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SYYDG requires pairs of qubits"
        );

        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            // SYYDG = exp(+i * π/4 * Y⊗Y) = (1/√2)(I + i*Y⊗Y)
            // Matrix: (1/√2) * [[1, 0, 0, -i], [0, 1, i, 0], [0, i, 1, 0], [-i, 0, 0, 1]]

            if step_lo >= 4 {
                // SIMD version
                let k_v = f64x4::splat(K);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            let new_re_00: [f64; 4] = (k_v * (re_00 + im_11)).into();
                            let new_im_00: [f64; 4] = (k_v * (im_00 - re_11)).into();

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            let new_re_01: [f64; 4] = (k_v * (re_01 - im_10)).into();
                            let new_im_01: [f64; 4] = (k_v * (im_01 + re_10)).into();

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            let new_re_10: [f64; 4] = (k_v * (re_10 - im_01)).into();
                            let new_im_10: [f64; 4] = (k_v * (im_10 + re_01)).into();

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            let new_re_11: [f64; 4] = (k_v * (re_11 + im_00)).into();
                            let new_im_11: [f64; 4] = (k_v * (im_11 - re_00)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = K * (|00⟩ - i*|11⟩)
                            self.real[idx_00] = K * (re_00 + im_11);
                            self.imag[idx_00] = K * (im_00 - re_11);

                            // new_01 = K * (|01⟩ + i*|10⟩)
                            self.real[idx_01] = K * (re_01 - im_10);
                            self.imag[idx_01] = K * (im_01 + re_10);

                            // new_10 = K * (|10⟩ + i*|01⟩)
                            self.real[idx_10] = K * (re_10 - im_01);
                            self.imag[idx_10] = K * (im_10 + re_01);

                            // new_11 = K * (|11⟩ - i*|00⟩)
                            self.real[idx_11] = K * (re_11 + im_00);
                            self.imag[idx_11] = K * (im_11 - re_00);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn szz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SZZ requires pairs of qubits"
        );

        // SZZ = exp(-i * π/4 * Z⊗Z)
        // Z⊗Z is diagonal: diag(1, -1, -1, 1)
        // SZZ = diag(e^{-iπ/4}, e^{iπ/4}, e^{iπ/4}, e^{-iπ/4})
        // e^{-iπ/4} = (1-i)/√2: (re,im) -> K*(re+im, -re+im)
        // e^{iπ/4} = (1+i)/√2: (re,im) -> K*(re-im, re+im)
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();
            let q_lo = q1.min(q2);

            // When both qubits >= 2, consecutive indices share the same phase
            if q_lo >= 2 {
                let n = self.real.len();
                let k_v = f64x4::splat(K);
                let mut i = 0;
                while i + 4 <= n {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;

                    let re = f64x4::from(&self.real[i..i + 4]);
                    let im = f64x4::from(&self.imag[i..i + 4]);

                    let (new_re, new_im) = if bit1 == bit2 {
                        // e^{-iπ/4}: (re,im) -> K*(re+im, -re+im)
                        (k_v * (re + im), k_v * (im - re))
                    } else {
                        // e^{iπ/4}: (re,im) -> K*(re-im, re+im)
                        (k_v * (re - im), k_v * (re + im))
                    };
                    let arr_re: [f64; 4] = new_re.into();
                    let arr_im: [f64; 4] = new_im.into();
                    self.real[i..i + 4].copy_from_slice(&arr_re);
                    self.imag[i..i + 4].copy_from_slice(&arr_im);
                    i += 4;
                }
            } else {
                // Scalar fallback
                let n = self.real.len();
                let mask1 = 1 << q1;
                let mask2 = 1 << q2;
                let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
                let step_lo = 1 << q_lo;
                let step_hi = 1 << q_hi;

                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            // |00⟩ → (1-i)/√2 |00⟩
                            let (re, im) = (self.real[idx_00], self.imag[idx_00]);
                            self.real[idx_00] = K * (re + im);
                            self.imag[idx_00] = K * (-re + im);

                            // |01⟩ → (1+i)/√2 |01⟩
                            let (re, im) = (self.real[idx_01], self.imag[idx_01]);
                            self.real[idx_01] = K * (re - im);
                            self.imag[idx_01] = K * (re + im);

                            // |10⟩ → (1+i)/√2 |10⟩
                            let (re, im) = (self.real[idx_10], self.imag[idx_10]);
                            self.real[idx_10] = K * (re - im);
                            self.imag[idx_10] = K * (re + im);

                            // |11⟩ → (1-i)/√2 |11⟩
                            let (re, im) = (self.real[idx_11], self.imag[idx_11]);
                            self.real[idx_11] = K * (re + im);
                            self.imag[idx_11] = K * (-re + im);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn szzdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SZZDG requires pairs of qubits"
        );

        // SZZDG = exp(+i * π/4 * Z⊗Z)
        // SZZDG = diag(e^{iπ/4}, e^{-iπ/4}, e^{-iπ/4}, e^{iπ/4})
        // e^{iπ/4} = (1+i)/√2: (re,im) -> K*(re-im, re+im)
        // e^{-iπ/4} = (1-i)/√2: (re,im) -> K*(re+im, -re+im)
        const K: f64 = std::f64::consts::FRAC_1_SQRT_2;

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();
            let q_lo = q1.min(q2);

            // When both qubits >= 2, consecutive indices share the same phase
            if q_lo >= 2 {
                let n = self.real.len();
                let k_v = f64x4::splat(K);
                let mut i = 0;
                while i + 4 <= n {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;

                    let re = f64x4::from(&self.real[i..i + 4]);
                    let im = f64x4::from(&self.imag[i..i + 4]);

                    let (new_re, new_im) = if bit1 == bit2 {
                        // e^{iπ/4}: (re,im) -> K*(re-im, re+im)
                        (k_v * (re - im), k_v * (re + im))
                    } else {
                        // e^{-iπ/4}: (re,im) -> K*(re+im, -re+im)
                        (k_v * (re + im), k_v * (im - re))
                    };
                    let arr_re: [f64; 4] = new_re.into();
                    let arr_im: [f64; 4] = new_im.into();
                    self.real[i..i + 4].copy_from_slice(&arr_re);
                    self.imag[i..i + 4].copy_from_slice(&arr_im);
                    i += 4;
                }
            } else {
                // Scalar fallback
                let n = self.real.len();
                let mask1 = 1 << q1;
                let mask2 = 1 << q2;
                let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
                let step_lo = 1 << q_lo;
                let step_hi = 1 << q_hi;

                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);

                            if base != idx_00 {
                                continue;
                            }

                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            // |00⟩ → (1+i)/√2 |00⟩
                            let (re, im) = (self.real[idx_00], self.imag[idx_00]);
                            self.real[idx_00] = K * (re - im);
                            self.imag[idx_00] = K * (re + im);

                            // |01⟩ → (1-i)/√2 |01⟩
                            let (re, im) = (self.real[idx_01], self.imag[idx_01]);
                            self.real[idx_01] = K * (re + im);
                            self.imag[idx_01] = K * (-re + im);

                            // |10⟩ → (1-i)/√2 |10⟩
                            let (re, im) = (self.real[idx_10], self.imag[idx_10]);
                            self.real[idx_10] = K * (re + im);
                            self.imag[idx_10] = K * (-re + im);

                            // |11⟩ → (1+i)/√2 |11⟩
                            let (re, im) = (self.real[idx_11], self.imag[idx_11]);
                            self.real[idx_11] = K * (re - im);
                            self.imag[idx_11] = K * (re + im);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn iswap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "iSWAP requires pairs of qubits"
        );

        // iSWAP matrix:
        // [[1, 0, 0, 0],
        //  [0, 0, i, 0],
        //  [0, i, 0, 0],
        //  [0, 0, 0, 1]]
        // |00⟩ → |00⟩, |01⟩ → i|10⟩, |10⟩ → i|01⟩, |11⟩ → |11⟩

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            if step_lo >= 4 {
                // SIMD version: when q_lo >= 2, consecutive base indices have
                // consecutive idx_01 and idx_10 values
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let base = i_lo + offset;
                            // Since we only process base indices where both qubit bits are 0,
                            // and step_lo >= 4, consecutive bases have consecutive idx values
                            let idx_01 = base | mask2;
                            let idx_10 = base | mask1;

                            // Load 4 consecutive values for |01⟩ and |10⟩ states
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);

                            // new |01⟩ = i * old |10⟩: i * (re, im) = (-im, re)
                            let new_re_01: [f64; 4] = (-im_10).into();
                            let new_im_01: [f64; 4] = re_10.into();

                            // new |10⟩ = i * old |01⟩
                            let new_re_10: [f64; 4] = (-im_01).into();
                            let new_im_10: [f64; 4] = re_01.into();

                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_01 = (base & !(mask1 | mask2)) | mask2;
                            let idx_10 = (base & !(mask1 | mask2)) | mask1;

                            // Skip if we've already processed this pair
                            if base != (base & !(mask1 | mask2)) {
                                continue;
                            }

                            // Swap |01⟩ ↔ |10⟩ and multiply both by i
                            // i * (re, im) = (-im, re)
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);

                            // new |01⟩ = i * old |10⟩
                            self.real[idx_01] = -im_10;
                            self.imag[idx_01] = re_10;

                            // new |10⟩ = i * old |01⟩
                            self.real[idx_10] = -im_01;
                            self.imag[idx_10] = re_01;
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn g(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len().is_multiple_of(2), "G requires pairs of qubits");

        // G = CZ.H(q1).H(q2).CZ
        // Traced through the decomposition, the actual matrix is:
        // [[1,  1,  1, -1],
        //  [1, -1,  1,  1],
        //  [1,  1, -1,  1],
        //  [-1, 1,  1,  1]] / 2
        //
        // new_00 = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2
        // new_01 = (|00⟩ - |01⟩ + |10⟩ + |11⟩) / 2
        // new_10 = (|00⟩ + |01⟩ - |10⟩ + |11⟩) / 2
        // new_11 = (-|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            let n = self.real.len();
            let (q_lo, q_hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

            let step_lo = 1 << q_lo;
            let step_hi = 1 << q_hi;
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            if step_lo >= 4 {
                // SIMD version: when q_lo >= 2, consecutive base indices have consecutive idx values
                let half_v = f64x4::splat(0.5);
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        let mut offset = 0;
                        while offset + 4 <= step_lo {
                            let idx_00 = i_lo + offset;
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            let re_00 = f64x4::from(&self.real[idx_00..idx_00 + 4]);
                            let im_00 = f64x4::from(&self.imag[idx_00..idx_00 + 4]);
                            let re_01 = f64x4::from(&self.real[idx_01..idx_01 + 4]);
                            let im_01 = f64x4::from(&self.imag[idx_01..idx_01 + 4]);
                            let re_10 = f64x4::from(&self.real[idx_10..idx_10 + 4]);
                            let im_10 = f64x4::from(&self.imag[idx_10..idx_10 + 4]);
                            let re_11 = f64x4::from(&self.real[idx_11..idx_11 + 4]);
                            let im_11 = f64x4::from(&self.imag[idx_11..idx_11 + 4]);

                            // new_00 = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2
                            let new_re_00: [f64; 4] =
                                (half_v * (re_00 + re_01 + re_10 - re_11)).into();
                            let new_im_00: [f64; 4] =
                                (half_v * (im_00 + im_01 + im_10 - im_11)).into();

                            // new_01 = (|00⟩ - |01⟩ + |10⟩ + |11⟩) / 2
                            let new_re_01: [f64; 4] =
                                (half_v * (re_00 - re_01 + re_10 + re_11)).into();
                            let new_im_01: [f64; 4] =
                                (half_v * (im_00 - im_01 + im_10 + im_11)).into();

                            // new_10 = (|00⟩ + |01⟩ - |10⟩ + |11⟩) / 2
                            let new_re_10: [f64; 4] =
                                (half_v * (re_00 + re_01 - re_10 + re_11)).into();
                            let new_im_10: [f64; 4] =
                                (half_v * (im_00 + im_01 - im_10 + im_11)).into();

                            // new_11 = (-|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2
                            let new_re_11: [f64; 4] =
                                (half_v * (-re_00 + re_01 + re_10 + re_11)).into();
                            let new_im_11: [f64; 4] =
                                (half_v * (-im_00 + im_01 + im_10 + im_11)).into();

                            self.real[idx_00..idx_00 + 4].copy_from_slice(&new_re_00);
                            self.imag[idx_00..idx_00 + 4].copy_from_slice(&new_im_00);
                            self.real[idx_01..idx_01 + 4].copy_from_slice(&new_re_01);
                            self.imag[idx_01..idx_01 + 4].copy_from_slice(&new_im_01);
                            self.real[idx_10..idx_10 + 4].copy_from_slice(&new_re_10);
                            self.imag[idx_10..idx_10 + 4].copy_from_slice(&new_im_10);
                            self.real[idx_11..idx_11 + 4].copy_from_slice(&new_re_11);
                            self.imag[idx_11..idx_11 + 4].copy_from_slice(&new_im_11);

                            offset += 4;
                        }
                    }
                }
            } else {
                // Scalar fallback for small step
                for i_hi in (0..n).step_by(step_hi * 2) {
                    for i_lo in (i_hi..i_hi + step_hi).step_by(step_lo * 2) {
                        for offset in 0..step_lo {
                            let base = i_lo + offset;
                            let idx_00 = base & !(mask1 | mask2);
                            let idx_01 = idx_00 | mask2;
                            let idx_10 = idx_00 | mask1;
                            let idx_11 = idx_00 | mask1 | mask2;

                            // Skip if we've already processed this quartet
                            if base != idx_00 {
                                continue;
                            }

                            let (re_00, im_00) = (self.real[idx_00], self.imag[idx_00]);
                            let (re_01, im_01) = (self.real[idx_01], self.imag[idx_01]);
                            let (re_10, im_10) = (self.real[idx_10], self.imag[idx_10]);
                            let (re_11, im_11) = (self.real[idx_11], self.imag[idx_11]);

                            // new_00 = (|00⟩ + |01⟩ + |10⟩ - |11⟩) / 2
                            self.real[idx_00] = 0.5 * (re_00 + re_01 + re_10 - re_11);
                            self.imag[idx_00] = 0.5 * (im_00 + im_01 + im_10 - im_11);

                            // new_01 = (|00⟩ - |01⟩ + |10⟩ + |11⟩) / 2
                            self.real[idx_01] = 0.5 * (re_00 - re_01 + re_10 + re_11);
                            self.imag[idx_01] = 0.5 * (im_00 - im_01 + im_10 + im_11);

                            // new_10 = (|00⟩ + |01⟩ - |10⟩ + |11⟩) / 2
                            self.real[idx_10] = 0.5 * (re_00 + re_01 - re_10 + re_11);
                            self.imag[idx_10] = 0.5 * (im_00 + im_01 - im_10 + im_11);

                            // new_11 = (-|00⟩ + |01⟩ + |10⟩ + |11⟩) / 2
                            self.real[idx_11] = 0.5 * (-re_00 + re_01 + re_10 + re_11);
                            self.imag[idx_11] = 0.5 * (-im_00 + im_01 + im_10 + im_11);
                        }
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let q_idx = q.index();
            let step = 1 << q_idx;

            // Calculate probability of measuring |1⟩ using SIMD
            let prob_one = self.probability_one(q_idx);

            // Sample outcome
            let outcome = self.rng.random::<f64>() < prob_one;
            let is_deterministic = prob_one < 1e-10 || prob_one > 1.0 - 1e-10;

            // Collapse and renormalize
            let norm_factor = if outcome {
                1.0 / prob_one.sqrt()
            } else {
                1.0 / (1.0 - prob_one).sqrt()
            };

            // For small steps, use scalar collapse
            if step < 4 {
                for i in (0..self.real.len()).step_by(step * 2) {
                    if outcome {
                        for j in i..(i + step) {
                            self.real[j] = 0.0;
                            self.imag[j] = 0.0;
                        }
                        for j in (i + step)..(i + 2 * step) {
                            self.real[j] *= norm_factor;
                            self.imag[j] *= norm_factor;
                        }
                    } else {
                        for j in i..(i + step) {
                            self.real[j] *= norm_factor;
                            self.imag[j] *= norm_factor;
                        }
                        for j in (i + step)..(i + 2 * step) {
                            self.real[j] = 0.0;
                            self.imag[j] = 0.0;
                        }
                    }
                }
            } else {
                // SIMD collapse and renormalize
                let norm_vec = f64x4::splat(norm_factor);

                for i in (0..self.real.len()).step_by(step * 2) {
                    if outcome {
                        // Zero |0⟩ states, normalize |1⟩ states
                        let mut j = i;
                        while j + 4 <= i + step {
                            self.real[j..j + 4].copy_from_slice(&[0.0; 4]);
                            self.imag[j..j + 4].copy_from_slice(&[0.0; 4]);
                            j += 4;
                        }
                        let mut j = i + step;
                        while j + 4 <= i + 2 * step {
                            let re = f64x4::from(&self.real[j..j + 4]);
                            let im = f64x4::from(&self.imag[j..j + 4]);
                            let scaled_re: [f64; 4] = (norm_vec * re).into();
                            let scaled_im: [f64; 4] = (norm_vec * im).into();
                            self.real[j..j + 4].copy_from_slice(&scaled_re);
                            self.imag[j..j + 4].copy_from_slice(&scaled_im);
                            j += 4;
                        }
                    } else {
                        // Normalize |0⟩ states, zero |1⟩ states
                        let mut j = i;
                        while j + 4 <= i + step {
                            let re = f64x4::from(&self.real[j..j + 4]);
                            let im = f64x4::from(&self.imag[j..j + 4]);
                            let scaled_re: [f64; 4] = (norm_vec * re).into();
                            let scaled_im: [f64; 4] = (norm_vec * im).into();
                            self.real[j..j + 4].copy_from_slice(&scaled_re);
                            self.imag[j..j + 4].copy_from_slice(&scaled_im);
                            j += 4;
                        }
                        let mut j = i + step;
                        while j + 4 <= i + 2 * step {
                            self.real[j..j + 4].copy_from_slice(&[0.0; 4]);
                            self.imag[j..j + 4].copy_from_slice(&[0.0; 4]);
                            j += 4;
                        }
                    }
                }
            }

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }
        results
    }
}

impl<R> ArbitraryRotationGateable for StateVecSoA<R>
where
    R: Rng,
{
    #[inline]
    fn rx(&mut self, theta: f64, qubits: &[QubitId]) -> &mut Self {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        for &q in qubits {
            apply_rx_simd!(self, q.index(), cos, sin);
        }
        self
    }

    #[inline]
    fn ry(&mut self, theta: f64, qubits: &[QubitId]) -> &mut Self {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        for &q in qubits {
            apply_ry_simd!(self, q.index(), cos, sin);
        }
        self
    }

    #[inline]
    fn rz(&mut self, theta: f64, qubits: &[QubitId]) -> &mut Self {
        let cos_neg = (-theta / 2.0).cos();
        let sin_neg = (-theta / 2.0).sin();
        let cos_pos = (theta / 2.0).cos();
        let sin_pos = (theta / 2.0).sin();
        for &q in qubits {
            apply_rz_simd!(self, q.index(), cos_neg, sin_neg, cos_pos, sin_pos);
        }
        self
    }

    #[inline]
    fn r1xy(&mut self, theta: f64, phi: f64, qubits: &[QubitId]) -> &mut Self {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        // R1XY: [[cos, r01], [r10, cos]]
        // r01 = -i*sin*e^(-iφ) = -sin*sinφ - i*sin*cosφ
        // r10 = -i*sin*e^(iφ)  = sin*sinφ - i*sin*cosφ
        let r01_re = -sin * phi.sin();
        let r01_im = -sin * phi.cos();
        let r10_re = sin * phi.sin();
        let r10_im = -sin * phi.cos();
        for &q in qubits {
            apply_r1xy_simd!(self, q.index(), cos, r01_re, r01_im, r10_re, r10_im);
        }
        self
    }

    #[inline]
    fn rzz(&mut self, theta: f64, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RZZ requires pairs of qubits"
        );
        let cos_pos = (theta / 2.0).cos();
        let sin_pos = (theta / 2.0).sin();
        let cos_neg = (-theta / 2.0).cos();
        let sin_neg = (-theta / 2.0).sin();

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();
            let q_lo = q1.min(q2);

            // When both qubits >= 2, consecutive indices share the same phase
            if q_lo >= 2 {
                let n = self.real.len();
                let mut i = 0;
                while i + 4 <= n {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;
                    let (cos, sin) = if bit1 == bit2 {
                        (cos_neg, sin_neg)
                    } else {
                        (cos_pos, sin_pos)
                    };
                    let cos_v = f64x4::splat(cos);
                    let sin_v = f64x4::splat(sin);

                    let re = f64x4::from(&self.real[i..i + 4]);
                    let im = f64x4::from(&self.imag[i..i + 4]);
                    let new_re: [f64; 4] = (cos_v * re - sin_v * im).into();
                    let new_im: [f64; 4] = (sin_v * re + cos_v * im).into();
                    self.real[i..i + 4].copy_from_slice(&new_re);
                    self.imag[i..i + 4].copy_from_slice(&new_im);
                    i += 4;
                }
            } else {
                // Scalar fallback for small qubit indices
                for i in 0..self.real.len() {
                    let bit1 = (i >> q1) & 1;
                    let bit2 = (i >> q2) & 1;
                    let (cos, sin) = if bit1 == bit2 {
                        (cos_neg, sin_neg)
                    } else {
                        (cos_pos, sin_pos)
                    };
                    let re = self.real[i];
                    let im = self.imag[i];
                    self.real[i] = cos * re - sin * im;
                    self.imag[i] = sin * re + cos * im;
                }
            }
        }
        self
    }

    #[inline]
    fn rxx(&mut self, theta: f64, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RXX requires pairs of qubits"
        );
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            // Use strided iteration for cache efficiency
            let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
            let step_lo = 1 << lo;
            let step_hi = 1 << hi;

            // RXX matrix (in computational basis):
            // |00⟩ -> cos|00⟩ - i*sin|11⟩
            // |01⟩ -> cos|01⟩ - i*sin|10⟩
            // |10⟩ -> -i*sin|01⟩ + cos|10⟩
            // |11⟩ -> -i*sin|00⟩ + cos|11⟩

            for outer in (0..self.real.len()).step_by(step_hi * 2) {
                for mid in (0..step_hi).step_by(step_lo * 2) {
                    for inner in 0..step_lo {
                        let base = outer + mid + inner;
                        let i00 = base;
                        let i01 = base + step_lo;
                        let i10 = base + step_hi;
                        let i11 = base + step_hi + step_lo;

                        // Load amplitudes
                        let (r00, m00) = (self.real[i00], self.imag[i00]);
                        let (r01, m01) = (self.real[i01], self.imag[i01]);
                        let (r10, m10) = (self.real[i10], self.imag[i10]);
                        let (r11, m11) = (self.real[i11], self.imag[i11]);

                        // Apply RXX: multiply by -i*sin means (re, im) -> (sin*im, -sin*re)
                        // new|00⟩ = cos*|00⟩ - i*sin*|11⟩
                        self.real[i00] = cos * r00 + sin * m11;
                        self.imag[i00] = cos * m00 - sin * r11;

                        // new|01⟩ = cos*|01⟩ - i*sin*|10⟩
                        self.real[i01] = cos * r01 + sin * m10;
                        self.imag[i01] = cos * m01 - sin * r10;

                        // new|10⟩ = -i*sin*|01⟩ + cos*|10⟩
                        self.real[i10] = sin * m01 + cos * r10;
                        self.imag[i10] = -sin * r01 + cos * m10;

                        // new|11⟩ = -i*sin*|00⟩ + cos*|11⟩
                        self.real[i11] = sin * m00 + cos * r11;
                        self.imag[i11] = -sin * r00 + cos * m11;
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn ryy(&mut self, theta: f64, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RYY requires pairs of qubits"
        );
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            // Use strided iteration for cache efficiency
            let (lo, hi) = if q1 < q2 { (q1, q2) } else { (q2, q1) };
            let step_lo = 1 << lo;
            let step_hi = 1 << hi;

            // RYY matrix (in computational basis):
            // |00⟩ -> cos|00⟩ + i*sin|11⟩
            // |01⟩ -> cos|01⟩ - i*sin|10⟩
            // |10⟩ -> -i*sin|01⟩ + cos|10⟩
            // |11⟩ -> i*sin|00⟩ + cos|11⟩

            for outer in (0..self.real.len()).step_by(step_hi * 2) {
                for mid in (0..step_hi).step_by(step_lo * 2) {
                    for inner in 0..step_lo {
                        let base = outer + mid + inner;
                        let i00 = base;
                        let i01 = base + step_lo;
                        let i10 = base + step_hi;
                        let i11 = base + step_hi + step_lo;

                        // Load amplitudes
                        let (r00, m00) = (self.real[i00], self.imag[i00]);
                        let (r01, m01) = (self.real[i01], self.imag[i01]);
                        let (r10, m10) = (self.real[i10], self.imag[i10]);
                        let (r11, m11) = (self.real[i11], self.imag[i11]);

                        // Apply RYY: multiply by i*sin means (re, im) -> (-sin*im, sin*re)
                        // new|00⟩ = cos*|00⟩ + i*sin*|11⟩
                        self.real[i00] = cos * r00 - sin * m11;
                        self.imag[i00] = cos * m00 + sin * r11;

                        // new|01⟩ = cos*|01⟩ - i*sin*|10⟩
                        self.real[i01] = cos * r01 + sin * m10;
                        self.imag[i01] = cos * m01 - sin * r10;

                        // new|10⟩ = -i*sin*|01⟩ + cos*|10⟩
                        self.real[i10] = sin * m01 + cos * r10;
                        self.imag[i10] = -sin * r01 + cos * m10;

                        // new|11⟩ = i*sin*|00⟩ + cos*|11⟩
                        self.real[i11] = -sin * m00 + cos * r11;
                        self.imag[i11] = sin * r00 + cos * m11;
                    }
                }
            }
        }
        self
    }

    #[inline]
    fn u(&mut self, theta: f64, phi: f64, lambda: f64, qubits: &[QubitId]) -> &mut Self {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        // U gate matrix elements
        let u00_re = cos;
        let u01_re = -sin * lambda.cos();
        let u01_im = -sin * lambda.sin();
        let u10_re = sin * phi.cos();
        let u10_im = sin * phi.sin();
        let u11_re = cos * (phi + lambda).cos();
        let u11_im = cos * (phi + lambda).sin();

        for &q in qubits {
            let q = q.index();
            let step = 1 << q;

            if step >= 4 {
                // SIMD version
                let u00_re_v = f64x4::splat(u00_re);
                let u01_re_v = f64x4::splat(u01_re);
                let u01_im_v = f64x4::splat(u01_im);
                let u10_re_v = f64x4::splat(u10_re);
                let u10_im_v = f64x4::splat(u10_im);
                let u11_re_v = f64x4::splat(u11_re);
                let u11_im_v = f64x4::splat(u11_im);

                for i in (0..self.real.len()).step_by(step * 2) {
                    let mut j = i;
                    while j + 4 <= i + step {
                        let p = j + step;

                        let re_a = f64x4::from(&self.real[j..j + 4]);
                        let im_a = f64x4::from(&self.imag[j..j + 4]);
                        let re_b = f64x4::from(&self.real[p..p + 4]);
                        let im_b = f64x4::from(&self.imag[p..p + 4]);

                        let new_re_a: [f64; 4] =
                            (u00_re_v * re_a + u01_re_v * re_b - u01_im_v * im_b).into();
                        let new_im_a: [f64; 4] =
                            (u00_re_v * im_a + u01_re_v * im_b + u01_im_v * re_b).into();
                        let new_re_b: [f64; 4] =
                            (u10_re_v * re_a - u10_im_v * im_a + u11_re_v * re_b - u11_im_v * im_b)
                                .into();
                        let new_im_b: [f64; 4] =
                            (u10_re_v * im_a + u10_im_v * re_a + u11_re_v * im_b + u11_im_v * re_b)
                                .into();

                        self.real[j..j + 4].copy_from_slice(&new_re_a);
                        self.imag[j..j + 4].copy_from_slice(&new_im_a);
                        self.real[p..p + 4].copy_from_slice(&new_re_b);
                        self.imag[p..p + 4].copy_from_slice(&new_im_b);

                        j += 4;
                    }
                }
            } else {
                // Scalar fallback for small step
                self.apply_single_qubit(q, |re_a, im_a, re_b, im_b| {
                    (
                        u00_re * re_a + u01_re * re_b - u01_im * im_b,
                        u00_re * im_a + u01_re * im_b + u01_im * re_b,
                        u10_re * re_a - u10_im * im_a + u11_re * re_b - u11_im * im_b,
                        u10_re * im_a + u10_im * re_a + u11_re * im_b + u11_im * re_b,
                    )
                });
            }
        }
        self
    }
}

// ============================================================================
// RNG Management
// ============================================================================

impl<R> RngManageable for StateVecSoA<R>
where
    R: RngCore + SeedableRng + Debug,
{
    type Rng = R;

    #[inline]
    fn set_rng(&mut self, rng: R) {
        self.rng = rng;
    }

    #[inline]
    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

// ============================================================================
// Fused Gate Operations
// ============================================================================
//
// These fused gates combine two operations into a single pass over memory,
// reducing memory bandwidth requirements by ~50% compared to separate gates.

impl<R> StateVecSoA<R>
where
    R: Rng,
{
    /// Fused H-Z gate: applies H then Z in a single memory pass.
    ///
    /// Matrix: Z*H = 1/√2 [[1, 1], [-1, 1]] (rightmost applied first)
    ///
    /// This is ~1.5x faster than calling `h()` then `z()` separately.
    #[inline]
    pub fn hz(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        for &q in qubits {
            apply_real_2x2_gate_simd!(self, q.index(), k, k, -k, k);
        }
        self
    }

    /// Fused Z-H gate: applies Z then H in a single memory pass.
    ///
    /// Matrix: H*Z = 1/√2 [[1, -1], [1, 1]] (rightmost applied first)
    #[inline]
    pub fn zh(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        for &q in qubits {
            apply_real_2x2_gate_simd!(self, q.index(), k, -k, k, k);
        }
        self
    }

    /// Fused H-S gate: applies H then S in a single memory pass.
    ///
    /// Matrix: S*H = 1/√2 [[1, 1], [i, -i]] (rightmost applied first)
    #[inline]
    pub fn hs(&mut self, qubits: &[QubitId]) -> &mut Self {
        // S*H: new |0⟩ = (a + b)/√2, new |1⟩ = i*(a - b)/√2
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), hs);
        }
        self
    }

    /// Fused S-H gate: applies S then H in a single memory pass.
    ///
    /// Matrix: H*S = 1/√2 [[1, i], [1, -i]] (rightmost applied first)
    #[inline]
    pub fn sh(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H*S: new |0⟩ = (a + i*b)/√2, new |1⟩ = (a - i*b)/√2
        for &q in qubits {
            apply_sqrt_gate_simd!(self, q.index(), sh);
        }
        self
    }

    /// Fused H-X gate: applies H then X in a single memory pass.
    ///
    /// Matrix: X*H = 1/√2 [[1, -1], [1, 1]] (rightmost applied first)
    #[inline]
    pub fn hx(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        for &q in qubits {
            // Same as zh
            apply_real_2x2_gate_simd!(self, q.index(), k, -k, k, k);
        }
        self
    }

    /// Fused X-H gate: applies X then H in a single memory pass.
    ///
    /// Matrix: H*X = 1/√2 [[1, 1], [-1, 1]] (rightmost applied first)
    #[inline]
    pub fn xh(&mut self, qubits: &[QubitId]) -> &mut Self {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        for &q in qubits {
            // Same as hz
            apply_real_2x2_gate_simd!(self, q.index(), k, k, -k, k);
        }
        self
    }

    /// Fused H on target then CX: applies H(target) then CX(control, target) in optimized passes.
    ///
    /// This pattern is common for creating entanglement after preparing superposition.
    /// The H and CX operate on the same target qubit, allowing some optimization.
    #[inline]
    pub fn h_then_cx(&mut self, control: QubitId, target: QubitId) -> &mut Self {
        // Apply H to target first
        self.h(&[target]);
        // Then apply CX - these can't be fully fused since they have different structure
        self.cx(&[control, target]);
        self
    }

    /// Fused CX then H on target: applies CX(control, target) then H(target).
    ///
    /// This pattern is common in measurement preparation.
    #[inline]
    pub fn cx_then_h(&mut self, control: QubitId, target: QubitId) -> &mut Self {
        self.cx(&[control, target]);
        self.h(&[target]);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVecSoA;
    use num_complex::Complex64;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI};

    fn states_match(sv: &StateVecSoA, opt: &StateVecSoA, tolerance: f64) -> bool {
        sv.state().iter().enumerate().all(|(i, c)| {
            let opt_c = Complex64::new(opt.real[i], opt.imag[i]);
            (*c - opt_c).norm() < tolerance
        })
    }

    fn assert_states_match(sv: &StateVecSoA, opt: &StateVecSoA, context: &str) {
        const TOLERANCE: f64 = 1e-10;
        assert!(
            states_match(sv, opt, TOLERANCE),
            "States don't match for {context}"
        );
    }

    #[test]
    fn test_new_state() {
        let opt: StateVecSoA = StateVecSoA::new(3);
        assert_eq!(opt.num_qubits(), 3);
        assert_eq!(opt.real.len(), 8);
        assert_eq!(opt.real[0], 1.0);
        assert_eq!(opt.imag[0], 0.0);
        for i in 1..8 {
            assert_eq!(opt.real[i], 0.0);
            assert_eq!(opt.imag[i], 0.0);
        }
    }

    #[test]
    fn test_h_gate() {
        for num_qubits in 1..=5 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("H on qubit {target} of {num_qubits}"));
            }
        }
    }

    #[test]
    fn test_x_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.x(&[QubitId(target)]);
                opt.x(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("X on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_y_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.y(&[QubitId(target)]);
                opt.y(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("Y on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_z_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);
                sv.z(&[QubitId(target)]);
                opt.z(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("Z on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_cx_gate() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut sv = StateVecSoA::new(num_qubits);
                    let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                    sv.h(&[QubitId(control)]);
                    opt.h(&[QubitId(control)]);

                    sv.cx(&[QubitId(control), QubitId(target)]);
                    opt.cx(&[QubitId(control), QubitId(target)]);

                    assert_states_match(
                        &sv,
                        &opt,
                        &format!("CX({control},{target}) in {num_qubits}q"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_cz_gate() {
        for num_qubits in 2..=4 {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            for q in 0..num_qubits {
                sv.h(&[QubitId(q)]);
                opt.h(&[QubitId(q)]);
            }

            sv.cz(&[QubitId(0), QubitId(1)]);
            opt.cz(&[QubitId(0), QubitId(1)]);

            assert_states_match(&sv, &opt, &format!("CZ in {num_qubits}q"));
        }
    }

    #[test]
    fn test_swap_gate() {
        for num_qubits in 2..=4 {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            sv.x(&[QubitId(0)]);
            opt.x(&[QubitId(0)]);
            sv.h(&[QubitId(1)]);
            opt.h(&[QubitId(1)]);

            sv.swap(&[QubitId(0), QubitId(1)]);
            opt.swap(&[QubitId(0), QubitId(1)]);

            assert_states_match(&sv, &opt, &format!("SWAP in {num_qubits}q"));
        }
    }

    #[test]
    fn test_rx_gate() {
        let angles = [0.0, FRAC_PI_2, PI, 1.234];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.rx(theta, &[QubitId(0)]);
            opt.rx(theta, &[QubitId(0)]);

            assert_states_match(&sv, &opt, &format!("RX({theta})"));
        }
    }

    #[test]
    fn test_u_gate() {
        let mut sv = StateVecSoA::new(2);
        let mut opt: StateVecSoA = StateVecSoA::new(2);

        sv.u(PI / 3.0, PI / 4.0, PI / 5.0, &[QubitId(0)]);
        opt.u(PI / 3.0, PI / 4.0, PI / 5.0, &[QubitId(0)]);

        assert_states_match(&sv, &opt, "U gate");
    }

    #[test]
    fn test_ghz_state() {
        let num_qubits = 4;
        let mut sv = StateVecSoA::new(num_qubits);
        let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);
        for i in 0..(num_qubits - 1) {
            sv.cx(&[QubitId(i), QubitId(i + 1)]);
            opt.cx(&[QubitId(i), QubitId(i + 1)]);
        }

        assert_states_match(&sv, &opt, "GHZ state");
    }

    #[test]
    fn test_mz_deterministic() {
        // Test deterministic measurement of |0⟩
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        let result = opt.mz(&[QubitId(0)]);
        assert!(!result[0].outcome, "Expected 0 outcome for |0> state");

        // Test deterministic measurement of |1⟩
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.x(&[QubitId(0)]);
        let result = opt.mz(&[QubitId(0)]);
        assert!(result[0].outcome, "Expected 1 outcome for |1> state");
    }

    #[test]
    fn test_reset() {
        let mut opt: StateVecSoA = StateVecSoA::new(3);
        opt.h(&[QubitId(0), QubitId(1), QubitId(2)]);
        opt.cx(&[QubitId(0), QubitId(1)]);
        opt.reset();

        assert_eq!(opt.real[0], 1.0);
        for i in 1..opt.real.len() {
            assert_eq!(opt.real[i], 0.0);
            assert_eq!(opt.imag[i], 0.0);
        }
    }

    // Helper to compare two StateVecSoA instances
    fn opts_match(a: &StateVecSoA, b: &StateVecSoA, tolerance: f64) -> bool {
        a.real.iter().zip(&b.real).all(|(x, y)| (x - y).abs() < tolerance)
            && a.imag.iter().zip(&b.imag).all(|(x, y)| (x - y).abs() < tolerance)
    }

    fn assert_opts_match(a: &StateVecSoA, b: &StateVecSoA, context: &str) {
        const TOLERANCE: f64 = 1e-10;
        assert!(opts_match(a, b, TOLERANCE), "States don't match for {context}");
    }

    #[test]
    fn test_fused_hz() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                // Prepare non-trivial state first
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                // Put into superposition
                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply H then Z separately
                separate.h(&[QubitId(target)]);
                separate.z(&[QubitId(target)]);

                // Apply fused H-Z
                fused.hz(&[QubitId(target)]);

                assert_opts_match(&separate, &fused, &format!("HZ fused on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_fused_zh() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply Z then H separately
                separate.z(&[QubitId(target)]);
                separate.h(&[QubitId(target)]);

                // Apply fused Z-H
                fused.zh(&[QubitId(target)]);

                assert_opts_match(&separate, &fused, &format!("ZH fused on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_fused_hs() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply H then S separately
                separate.h(&[QubitId(target)]);
                separate.sz(&[QubitId(target)]);

                // Apply fused H-S
                fused.hs(&[QubitId(target)]);

                assert_opts_match(&separate, &fused, &format!("HS fused on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_fused_sh() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply S then H separately
                separate.sz(&[QubitId(target)]);
                separate.h(&[QubitId(target)]);

                // Apply fused S-H
                fused.sh(&[QubitId(target)]);

                assert_opts_match(&separate, &fused, &format!("SH fused on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_fused_hx() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply H then X separately
                separate.h(&[QubitId(target)]);
                separate.x(&[QubitId(target)]);

                // Apply fused H-X
                fused.hx(&[QubitId(target)]);

                assert_opts_match(&separate, &fused, &format!("HX fused on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_fused_xh() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                separate.h(&[QubitId(target)]);
                fused.h(&[QubitId(target)]);

                // Apply X then H separately
                separate.x(&[QubitId(target)]);
                separate.h(&[QubitId(target)]);

                // Apply fused X-H
                fused.xh(&[QubitId(target)]);

                assert_opts_match(&separate, &fused, &format!("XH fused on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_fused_h_then_cx() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                    let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                    // Apply H then CX separately
                    separate.h(&[QubitId(target)]);
                    separate.cx(&[QubitId(control), QubitId(target)]);

                    // Apply fused H-CX
                    fused.h_then_cx(QubitId(control), QubitId(target));

                    assert_opts_match(
                        &separate,
                        &fused,
                        &format!("H-CX fused c={control} t={target}"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_fused_cx_then_h() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut separate: StateVecSoA = StateVecSoA::new(num_qubits);
                    let mut fused: StateVecSoA = StateVecSoA::new(num_qubits);

                    // Prepare entangled state first
                    separate.h(&[QubitId(control)]);
                    fused.h(&[QubitId(control)]);
                    separate.cx(&[QubitId(control), QubitId(target)]);
                    fused.cx(&[QubitId(control), QubitId(target)]);

                    // Apply CX then H separately
                    separate.cx(&[QubitId(control), QubitId(target)]);
                    separate.h(&[QubitId(target)]);

                    // Apply fused CX-H
                    fused.cx_then_h(QubitId(control), QubitId(target));

                    assert_opts_match(
                        &separate,
                        &fused,
                        &format!("CX-H fused c={control} t={target}"),
                    );
                }
            }
        }
    }

    // Additional tests for parity with StateVecSoA test coverage

    #[test]
    fn test_probability() {
        let mut sv = StateVecSoA::new(1);
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        // Prepare |+⟩ state
        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);

        let sv_prob_zero = sv.probability(0);
        let sv_prob_one = sv.probability(1);

        let opt_prob_zero = opt.probability(0);
        let opt_prob_one = opt.probability(1);

        assert!((sv_prob_zero - opt_prob_zero).abs() < 1e-10);
        assert!((sv_prob_one - opt_prob_one).abs() < 1e-10);
        assert!((opt_prob_zero - 0.5).abs() < 1e-10);
        assert!((opt_prob_one - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prepare_computational_basis_all_states() {
        let num_qubits = 3;

        for basis_state in 0..(1 << num_qubits) {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            sv.prepare_computational_basis(basis_state);
            opt.prepare_computational_basis(basis_state);

            assert_states_match(
                &sv,
                &opt,
                &format!("prepare_computational_basis({basis_state})"),
            );
        }
    }

    #[test]
    fn test_sz_gate() {
        for num_qubits in 1..=4 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                // Put into superposition first to see effect
                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);
                sv.sz(&[QubitId(target)]);
                opt.sz(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("SZ on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_cy_gate() {
        for num_qubits in 2..=4 {
            for control in 0..num_qubits {
                for target in 0..num_qubits {
                    if control == target {
                        continue;
                    }

                    let mut sv = StateVecSoA::new(num_qubits);
                    let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                    // Create |+0⟩ state
                    sv.h(&[QubitId(control)]);
                    opt.h(&[QubitId(control)]);

                    sv.cy(&[QubitId(control), QubitId(target)]);
                    opt.cy(&[QubitId(control), QubitId(target)]);

                    assert_states_match(
                        &sv,
                        &opt,
                        &format!("CY({control},{target}) in {num_qubits}q"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_measurement_consistency() {
        // Measuring a deterministic state should always give the same result
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.x(&[QubitId(0)]); // Put in |1⟩

        let result1 = opt.mz(&[QubitId(0)]);
        let result2 = opt.mz(&[QubitId(0)]);

        assert!(result1[0].outcome);
        assert!(result2[0].outcome);
    }

    #[test]
    fn test_measurement_collapse() {
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        // Prepare |+⟩ = (|0⟩ + |1⟩) / √2
        opt.h(&[QubitId(0)]);

        // Simulate a measurement
        let result = opt.mz(&[QubitId(0)]);

        // State should collapse to |0⟩ or |1⟩
        if result[0].outcome {
            assert!((opt.probability(1) - 1.0).abs() < 1e-10);
        } else {
            assert!((opt.probability(0) - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pz() {
        // Use same seed for both to ensure deterministic comparison
        let seed = 42;
        let mut sv = StateVecSoA::with_seed(1, seed);
        let mut opt: StateVecSoA = StateVecSoA::with_seed(1, seed);

        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);

        sv.pz(&[QubitId(0)]);
        opt.pz(&[QubitId(0)]);

        assert_states_match(&sv, &opt, "PZ on single qubit");
    }

    #[test]
    fn test_pz_multiple_qubits() {
        // Use same seed for both to ensure deterministic comparison
        let seed = 42;
        let mut sv = StateVecSoA::with_seed(2, seed);
        let mut opt: StateVecSoA = StateVecSoA::with_seed(2, seed);

        sv.h(&[QubitId(0)]);
        opt.h(&[QubitId(0)]);
        sv.cx(&[QubitId(0), QubitId(1)]);
        opt.cx(&[QubitId(0), QubitId(1)]);

        sv.pz(&[QubitId(0)]);
        opt.pz(&[QubitId(0)]);

        assert_states_match(&sv, &opt, "PZ on entangled state");
    }

    #[test]
    fn test_ry_gate() {
        let angles = [0.0, FRAC_PI_2, PI, 1.234];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.ry(theta, &[QubitId(0)]);
            opt.ry(theta, &[QubitId(0)]);

            assert_states_match(&sv, &opt, &format!("RY({theta})"));
        }
    }

    #[test]
    fn test_rz_gate() {
        let angles = [0.0, FRAC_PI_2, PI, 1.234];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            // Put in superposition to see phase effects
            sv.h(&[QubitId(0)]);
            opt.h(&[QubitId(0)]);
            sv.rz(theta, &[QubitId(0)]);
            opt.rz(theta, &[QubitId(0)]);

            assert_states_match(&sv, &opt, &format!("RZ({theta})"));
        }
    }

    #[test]
    fn test_r1xy_gate() {
        let mut sv = StateVecSoA::new(1);
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;

        sv.r1xy(theta, phi, &[QubitId(0)]);
        opt.r1xy(theta, phi, &[QubitId(0)]);

        assert_states_match(&sv, &opt, "R1XY gate");
    }

    #[test]
    fn test_rxx_gate() {
        let angles = [FRAC_PI_2, PI, FRAC_PI_4];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.rxx(theta, &[QubitId(0), QubitId(1)]);
            opt.rxx(theta, &[QubitId(0), QubitId(1)]);

            assert_states_match(&sv, &opt, &format!("RXX({theta})"));
        }
    }

    #[test]
    fn test_ryy_gate() {
        let angles = [FRAC_PI_2, PI, FRAC_PI_4];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            sv.ryy(theta, &[QubitId(0), QubitId(1)]);
            opt.ryy(theta, &[QubitId(0), QubitId(1)]);

            assert_states_match(&sv, &opt, &format!("RYY({theta})"));
        }
    }

    #[test]
    fn test_rzz_gate() {
        let angles = [FRAC_PI_2, PI, FRAC_PI_4];
        for &theta in &angles {
            let mut sv = StateVecSoA::new(2);
            let mut opt: StateVecSoA = StateVecSoA::new(2);

            // Create non-trivial state
            sv.h(&[QubitId(0)]);
            opt.h(&[QubitId(0)]);
            sv.h(&[QubitId(1)]);
            opt.h(&[QubitId(1)]);

            sv.rzz(theta, &[QubitId(0), QubitId(1)]);
            opt.rzz(theta, &[QubitId(0), QubitId(1)]);

            assert_states_match(&sv, &opt, &format!("RZZ({theta})"));
        }
    }

    #[test]
    fn test_normalization() {
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]).sz(&[QubitId(0)]).cx(&[QubitId(0), QubitId(1)]);

        let norm: f64 = opt
            .real
            .iter()
            .zip(&opt.imag)
            .map(|(r, i)| r * r + i * i)
            .sum();
        assert!((norm - 1.0).abs() < 1e-10, "State should be normalized");
    }

    #[test]
    fn test_unitarity() {
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.h(&[QubitId(0)]);
        let initial_real = opt.real.clone();
        let initial_imag = opt.imag.clone();

        // H^2 = I
        opt.h(&[QubitId(0)]).h(&[QubitId(0)]);

        for i in 0..opt.real.len() {
            assert!(
                (opt.real[i] - initial_real[i]).abs() < 1e-10,
                "H^2 should equal I (real part)"
            );
            assert!(
                (opt.imag[i] - initial_imag[i]).abs() < 1e-10,
                "H^2 should equal I (imag part)"
            );
        }
    }

    #[test]
    fn test_pauli_relations() {
        // XYZ = iI (up to global phase)
        let mut sv = StateVecSoA::new(1);
        let mut opt: StateVecSoA = StateVecSoA::new(1);

        sv.x(&[QubitId(0)]).y(&[QubitId(0)]).z(&[QubitId(0)]);
        opt.x(&[QubitId(0)]).y(&[QubitId(0)]).z(&[QubitId(0)]);

        assert_states_match(&sv, &opt, "XYZ sequence");

        // Also verify YZX gives same result
        let mut sv2 = StateVecSoA::new(1);
        let mut opt2: StateVecSoA = StateVecSoA::new(1);

        sv2.y(&[QubitId(0)]).z(&[QubitId(0)]).x(&[QubitId(0)]);
        opt2.y(&[QubitId(0)]).z(&[QubitId(0)]).x(&[QubitId(0)]);

        assert_states_match(&sv2, &opt2, "YZX sequence");
    }

    #[test]
    fn test_bell_state_correlations() {
        // Create Bell state and verify measurement correlations
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);
        opt.cx(&[QubitId(0), QubitId(1)]);

        // Measure first qubit
        let result1 = opt.mz(&[QubitId(0)]);
        // Measure second qubit - should match first
        let result2 = opt.mz(&[QubitId(1)]);

        assert_eq!(
            result1[0].outcome, result2[0].outcome,
            "Bell state measurements should be correlated"
        );
    }

    #[test]
    fn test_sx_gate() {
        for num_qubits in 1..=3 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.sx(&[QubitId(target)]);
                opt.sx(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("SX on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_sxdg_gate() {
        for num_qubits in 1..=3 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.sxdg(&[QubitId(target)]);
                opt.sxdg(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("SXDG on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_szdg_gate() {
        for num_qubits in 1..=3 {
            for target in 0..num_qubits {
                let mut sv = StateVecSoA::new(num_qubits);
                let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

                sv.h(&[QubitId(target)]);
                opt.h(&[QubitId(target)]);
                sv.szdg(&[QubitId(target)]);
                opt.szdg(&[QubitId(target)]);

                assert_states_match(&sv, &opt, &format!("SZDG on qubit {target}"));
            }
        }
    }

    #[test]
    fn test_iswap_gate() {
        for num_qubits in 2..=3 {
            let mut sv = StateVecSoA::new(num_qubits);
            let mut opt: StateVecSoA = StateVecSoA::new(num_qubits);

            sv.x(&[QubitId(0)]);
            opt.x(&[QubitId(0)]);

            sv.iswap(&[QubitId(0), QubitId(1)]);
            opt.iswap(&[QubitId(0), QubitId(1)]);

            assert_states_match(&sv, &opt, &format!("ISWAP in {num_qubits}q"));
        }
    }

    #[test]
    fn test_measurement_superposition_statistics() {
        // Test that superposition measurements are roughly 50/50
        let mut zeros = 0;
        let trials = 1000;

        for _ in 0..trials {
            let mut opt: StateVecSoA = StateVecSoA::new(1);
            opt.h(&[QubitId(0)]);
            let result = opt.mz(&[QubitId(0)]);
            if !result[0].outcome {
                zeros += 1;
            }
        }

        let ratio = f64::from(zeros) / f64::from(trials);
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "Superposition measurements should be roughly 50/50, got {ratio}"
        );
    }

    // Tests for new convenience and inspection methods

    #[test]
    fn test_get_set_amplitude() {
        let mut opt: StateVecSoA = StateVecSoA::new(2);

        // Initial state should be |00⟩
        assert!((opt.get_amplitude(0) - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(opt.get_amplitude(1).norm() < 1e-10);

        // Set a specific amplitude
        opt.set_amplitude(1, Complex64::new(0.5, 0.5));
        let amp = opt.get_amplitude(1);
        assert!((amp.re - 0.5).abs() < 1e-10);
        assert!((amp.im - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_to_complex_vec() {
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);

        let complex_state = opt.to_complex_vec();
        assert_eq!(complex_state.len(), 4);

        // After H on qubit 0: (|00⟩ + |01⟩)/√2
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((complex_state[0].re - expected).abs() < 1e-10);
        assert!((complex_state[1].re - expected).abs() < 1e-10);
        assert!(complex_state[2].norm() < 1e-10);
        assert!(complex_state[3].norm() < 1e-10);
    }

    #[test]
    fn test_from_complex_state() {
        let state = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let opt: StateVecSoA = StateVecSoA::from_complex_state(state.clone(), PecosRng::from_os_rng());

        for (i, expected) in state.iter().enumerate() {
            let actual = opt.get_amplitude(i);
            assert!((actual - expected).norm() < 1e-10);
        }
    }

    #[test]
    fn test_prepare_plus_state() {
        let mut opt: StateVecSoA = StateVecSoA::new(3);
        opt.prepare_plus_state();

        // Verify all amplitudes are equal to 1/sqrt(2^n) for n qubits
        let expected = 1.0 / (8.0_f64).sqrt();
        for i in 0..8 {
            let amp = opt.get_amplitude(i);
            assert!((amp.re - expected).abs() < 1e-10, "Real part mismatch at index {i}");
            assert!(amp.im.abs() < 1e-10, "Imaginary part should be zero at index {i}");
        }

        // Verify normalization (sum of probabilities = 1)
        let total_prob: f64 = (0..8).map(|i| opt.probability(i)).sum();
        assert!((total_prob - 1.0).abs() < 1e-10, "State should be normalized");
    }

    #[test]
    fn test_single_qubit_unitary() {
        use std::f64::consts::FRAC_1_SQRT_2;

        // Test Hadamard gate via single_qubit_unitary
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.single_qubit_unitary(
            0,
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        );

        // Compare with H gate
        let mut opt2: StateVecSoA = StateVecSoA::new(1);
        opt2.h(&[QubitId(0)]);

        assert_opts_match(&opt, &opt2, "single_qubit_unitary (Hadamard)");

        // Test X gate via single_qubit_unitary
        let mut opt: StateVecSoA = StateVecSoA::new(1);
        opt.single_qubit_unitary(
            0,
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        );

        let mut opt2: StateVecSoA = StateVecSoA::new(1);
        opt2.x(&[QubitId(0)]);

        assert_opts_match(&opt, &opt2, "single_qubit_unitary (X)");
    }

    #[test]
    fn test_two_qubit_unitary() {
        // Test CNOT gate via two_qubit_unitary
        let cnot = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ];

        // Create Bell state using two_qubit_unitary
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);
        opt.two_qubit_unitary(0, 1, cnot);

        // Create Bell state using CX
        let mut opt2: StateVecSoA = StateVecSoA::new(2);
        opt2.h(&[QubitId(0)]);
        opt2.cx(&[QubitId(0), QubitId(1)]);

        assert_opts_match(&opt, &opt2, "two_qubit_unitary (CNOT)");

        // Test SWAP gate via two_qubit_unitary
        let swap = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.x(&[QubitId(0)]);
        opt.two_qubit_unitary(0, 1, swap);

        let mut opt2: StateVecSoA = StateVecSoA::new(2);
        opt2.x(&[QubitId(0)]);
        opt2.swap(&[QubitId(0), QubitId(1)]);

        assert_opts_match(&opt, &opt2, "two_qubit_unitary (SWAP)");
    }

    #[test]
    fn test_roundtrip_complex_state() {
        // Create a non-trivial state
        let mut opt: StateVecSoA = StateVecSoA::new(2);
        opt.h(&[QubitId(0)]);
        opt.cx(&[QubitId(0), QubitId(1)]);

        // Convert to complex vec and back
        let complex_state = opt.to_complex_vec();
        let opt2: StateVecSoA = StateVecSoA::from_complex_state(complex_state, PecosRng::from_os_rng());

        assert_opts_match(&opt, &opt2, "roundtrip complex state");
    }
}
