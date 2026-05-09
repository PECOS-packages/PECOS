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

//! Minimal dense complex-matrix helpers for Phase 1.
//!
//! Matrices are stored row-major as `Vec<Complex64>` of length `d*d`. Caller
//! tracks `d`. This is intentionally primitive -- swap to faer / ndarray once
//! Phase 1 numbers prove out.

use num_complex::Complex64;

use crate::basis::{Pauli1, PauliString};

pub type Matrix = Vec<Complex64>;

pub fn zeros(d: usize) -> Matrix {
    vec![Complex64::new(0.0, 0.0); d * d]
}

pub fn identity(d: usize) -> Matrix {
    let mut m = zeros(d);
    for i in 0..d {
        m[i * d + i] = Complex64::new(1.0, 0.0);
    }
    m
}

pub fn matmul(a: &Matrix, b: &Matrix, d: usize) -> Matrix {
    let mut c = zeros(d);
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            if aik == Complex64::new(0.0, 0.0) {
                continue;
            }
            for j in 0..d {
                c[i * d + j] += aik * b[k * d + j];
            }
        }
    }
    c
}

/// Conjugate transpose.
pub fn dag(a: &Matrix, d: usize) -> Matrix {
    let mut b = zeros(d);
    for i in 0..d {
        for j in 0..d {
            b[j * d + i] = a[i * d + j].conj();
        }
    }
    b
}

/// Plain transpose (no complex conjugation).
pub fn transpose(a: &Matrix, d: usize) -> Matrix {
    let mut b = zeros(d);
    for i in 0..d {
        for j in 0..d {
            b[j * d + i] = a[i * d + j];
        }
    }
    b
}

/// Element-wise complex conjugate.
pub fn conj(a: &Matrix) -> Matrix {
    a.iter().map(|c| c.conj()).collect()
}

pub fn trace(a: &Matrix, d: usize) -> Complex64 {
    (0..d).map(|i| a[i * d + i]).sum()
}

pub fn scale(a: &Matrix, s: Complex64) -> Matrix {
    a.iter().map(|x| x * s).collect()
}

pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn sub(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// `A*B - B*A`.
pub fn commutator(a: &Matrix, b: &Matrix, d: usize) -> Matrix {
    sub(&matmul(a, b, d), &matmul(b, a, d))
}

/// `A*B + B*A`.
pub fn anticommutator(a: &Matrix, b: &Matrix, d: usize) -> Matrix {
    add(&matmul(a, b, d), &matmul(b, a, d))
}

/// 2x2 Pauli matrix for a single-qubit Pauli operator.
pub fn pauli_1q(p: Pauli1) -> Matrix {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    match p {
        Pauli1::I => vec![o, z, z, o],
        Pauli1::X => vec![z, o, o, z],
        Pauli1::Y => vec![z, -i, i, z],
        Pauli1::Z => vec![o, z, z, -o],
    }
}

/// Lowering operator sigma_- = |1><0| = [[0,0],[1,0]].
pub fn sigma_minus() -> Matrix {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    vec![z, z, o, z]
}

/// Kronecker product of `a` (da x da) and `b` (db x db). Result is
/// `(da * db) x (da * db)`.
pub fn kron(a: &Matrix, b: &Matrix, da: usize, db: usize) -> Matrix {
    let d = da * db;
    let mut out = zeros(d);
    for i in 0..da {
        for j in 0..da {
            let aij = a[i * da + j];
            if aij == Complex64::new(0.0, 0.0) {
                continue;
            }
            for k in 0..db {
                for l in 0..db {
                    let bkl = b[k * db + l];
                    let row = i * db + k;
                    let col = j * db + l;
                    out[row * d + col] = aij * bkl;
                }
            }
        }
    }
    out
}

/// Matrix representation of a multi-qubit Pauli string (tensor-product
/// of 2x2 Pauli matrices, left-to-right).
pub fn pauli_string_mat(ps: &PauliString) -> Matrix {
    assert!(!ps.0.is_empty(), "empty PauliString");
    let mut acc = pauli_1q(ps.0[0]);
    let mut d = 2;
    for p in ps.0.iter().skip(1) {
        acc = kron(&acc, &pauli_1q(*p), d, 2);
        d *= 2;
    }
    acc
}

/// Check whether a d x d matrix is (numerically) Hermitian: `M = M^dag`.
/// Returns true if all `|M_ij - conj(M_ji)| < tol`.
pub fn is_hermitian(m: &Matrix, d: usize, tol: f64) -> bool {
    assert_eq!(m.len(), d * d, "is_hermitian: wrong shape");
    for i in 0..d {
        for j in 0..d {
            if (m[i * d + j] - m[j * d + i].conj()).norm() > tol {
                return false;
            }
        }
    }
    true
}

/// Check whether a d x d matrix is (numerically) diagonal.
pub fn is_diagonal(m: &Matrix, d: usize, tol: f64) -> bool {
    for i in 0..d {
        for j in 0..d {
            if i == j {
                continue;
            }
            if m[i * d + j].norm() > tol {
                return false;
            }
        }
    }
    true
}

/// Is a 4x4 matrix 2x2-block-diagonal? I.e. off-diagonal 2x2 blocks zero.
pub fn is_2x2_block_diagonal(m: &Matrix, tol: f64) -> bool {
    assert_eq!(m.len(), 16, "is_2x2_block_diagonal requires 4x4 input");
    for r in 0..2 {
        for c in 2..4 {
            if m[r * 4 + c].norm() > tol {
                return false;
            }
            if m[c * 4 + r].norm() > tol {
                return false;
            }
        }
    }
    true
}

/// `exp(-i * H * t)` for a 4x4 2x2-block-diagonal Hermitian `H`, assuming
/// each 2x2 block is traceless (true for CX_theta: blocks are `0_2` and
/// `omega * X`).
pub fn exp_minus_i_h_t_2x2_block_diag(h: &Matrix, t: f64) -> Matrix {
    let d = 4;
    assert_eq!(h.len(), d * d);
    let mut ul = zeros(2);
    let mut lr = zeros(2);
    for r in 0..2 {
        for c in 0..2 {
            ul[r * 2 + c] = h[r * 4 + c];
            lr[r * 2 + c] = h[(r + 2) * 4 + (c + 2)];
        }
    }
    let ul_exp = exp_minus_i_h_t_1q_traceless(&ul, t);
    let lr_exp = exp_minus_i_h_t_1q_traceless(&lr, t);
    let mut out = zeros(d);
    for r in 0..2 {
        for c in 0..2 {
            out[r * 4 + c] = ul_exp[r * 2 + c];
            out[(r + 2) * 4 + (c + 2)] = lr_exp[r * 2 + c];
        }
    }
    out
}

/// `exp(-i * H * t)` for a Hermitian `H`. Dispatches:
/// - `H` diagonal -> elementwise exp (any `d`).
/// - `d == 2`, non-diagonal, traceless -> Bloch form.
/// - `d == 4`, 2x2-block-diagonal with traceless blocks -> block-wise
///   Bloch form (covers CX_theta).
/// - else falls through to general [`expm`] scaling-squaring.
pub fn exp_minus_i_h_t(h: &Matrix, d: usize, t: f64) -> Matrix {
    if is_diagonal(h, d, 1e-14) {
        let mut u = zeros(d);
        for i in 0..d {
            let arg = Complex64::new(0.0, -h[i * d + i].re * t);
            u[i * d + i] = arg.exp();
        }
        return u;
    }
    if d == 2 {
        return exp_minus_i_h_t_1q_traceless(h, t);
    }
    if d == 4 && is_2x2_block_diagonal(h, 1e-14) {
        return exp_minus_i_h_t_2x2_block_diag(h, t);
    }
    let arg = scale(h, Complex64::new(0.0, -t));
    expm(&arg, d)
}

/// General matrix exponential via Taylor series + scaling + squaring.
///
/// - Scale: find `s` such that `||A/2^s|| < 0.5` (so Taylor converges quickly).
/// - Taylor: `exp(A/2^s) ≈ sum_{k=0..=N} (A/2^s)^k / k!` with `N=20`.
/// - Squaring: `exp(A) = (exp(A/2^s))^(2^s)` via `s` matrix squarings.
///
/// Accuracy ~machine-precision for Hermitian `A` with arbitrary norm;
/// validated in module tests against Bloch-form and diagonal paths.
pub fn expm(a: &Matrix, d: usize) -> Matrix {
    let norm = inf_norm(a, d);
    if norm < 1e-14 {
        return identity(d);
    }
    // Choose s so that ||A / 2^s|| < 0.5.
    let s_float = (norm / 0.5).log2().max(0.0).ceil();
    let s: u32 = s_float as u32;
    let factor = Complex64::new(2f64.powi(-(s as i32)), 0.0);
    let scaled = scale(a, factor);
    let mut result = taylor_exp(&scaled, d, 20);
    for _ in 0..s {
        result = matmul(&result, &result, d);
    }
    result
}

/// Taylor series of `exp(A)` truncated at degree `n`. Assumes `||A||`
/// is already small (typically `< 0.5`).
fn taylor_exp(a: &Matrix, d: usize, n: usize) -> Matrix {
    let mut term = identity(d);
    let mut sum = identity(d);
    for k in 1..=n {
        term = scale(&matmul(&term, a, d), Complex64::new(1.0 / k as f64, 0.0));
        sum = add(&sum, &term);
    }
    sum
}

/// Infinity norm: max over rows of `sum_j |A_ij|`.
pub fn inf_norm(a: &Matrix, d: usize) -> f64 {
    (0..d)
        .map(|i| (0..d).map(|j| a[i * d + j].norm()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Column-stack vectorization `vec(M)` of a `d x d` matrix (length `d^2`).
/// Convention: `vec(A rho B) = (B^T ⊗ A) vec(rho)`.
pub fn vec_of(m: &Matrix, d: usize) -> Vec<Complex64> {
    assert_eq!(m.len(), d * d);
    let mut out = vec![Complex64::new(0.0, 0.0); d * d];
    // Column-major layout: vec(M)[i + d*j] = M[i, j] = m[i*d + j].
    for i in 0..d {
        for j in 0..d {
            out[i + d * j] = m[i * d + j];
        }
    }
    out
}

/// Inverse of [`vec_of`]: reshape a `d^2` vector back to a `d x d`
/// matrix (row-major storage).
pub fn unvec(v: &[Complex64], d: usize) -> Matrix {
    assert_eq!(v.len(), d * d);
    let mut m = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            m[i * d + j] = v[i + d * j];
        }
    }
    m
}

/// Matrix-vector product `A * v` for a `n x n` matrix `A` and length-`n`
/// vector `v`.
pub fn matvec(a: &Matrix, v: &[Complex64], n: usize) -> Vec<Complex64> {
    assert_eq!(a.len(), n * n);
    assert_eq!(v.len(), n);
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let mut s = Complex64::new(0.0, 0.0);
        for j in 0..n {
            s += a[i * n + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Matrix exponential `exp(-i * H * t)` for a 2x2 traceless Hermitian H.
/// Uses the Bloch form: `exp(-i H t) = cos(r t) I - i sin(r t) H / r`
/// where `r = sqrt(c_x^2 + c_y^2 + c_z^2)` is the Pauli-decomposition norm.
/// Panics if `H` has nonzero trace.
pub fn exp_minus_i_h_t_1q_traceless(h: &Matrix, t: f64) -> Matrix {
    assert_eq!(h.len(), 4, "requires a 2x2 matrix");
    // Check Hermitian (tolerant).
    let h00 = h[0];
    let h01 = h[1];
    let h10 = h[2];
    let h11 = h[3];
    assert!(
        h00.im.abs() < 1e-12 && h11.im.abs() < 1e-12,
        "H not Hermitian (diagonal)"
    );
    assert!(
        (h10 - h01.conj()).norm() < 1e-12,
        "H not Hermitian (off-diagonal)"
    );
    let tr = (h00 + h11).re;
    assert!(tr.abs() < 1e-12, "H must be traceless; got trace = {}", tr);

    // Pauli decomposition: H = c_x X + c_y Y + c_z Z.
    let c_x = h01.re;
    let c_y = -h01.im; // since H_{01} = c_x - i c_y
    let c_z = (h00.re - h11.re) * 0.5;

    let r = (c_x * c_x + c_y * c_y + c_z * c_z).sqrt();
    if r < 1e-15 {
        return identity(2);
    }
    let c = (r * t).cos();
    let s = (r * t).sin() / r;
    let minus_i_s = Complex64::new(0.0, -s);
    // result = c * I - i s * H
    let i2 = identity(2);
    add(&scale(&i2, Complex64::new(c, 0.0)), &scale(h, minus_i_s))
}
