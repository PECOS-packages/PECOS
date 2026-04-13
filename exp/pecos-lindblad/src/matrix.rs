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

use crate::basis::Pauli1;

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

/// Matrix exponential `exp(-i * H * t)` for a 2x2 traceless Hermitian H.
/// Uses the Bloch form: `exp(-i H t) = cos(r t) I - i sin(r t) H / r`
/// where `r = sqrt(c_x^2 + c_y^2 + c_z^2)` is the Pauli-decomposition norm.
/// Panics if `H` has nonzero trace (use a dedicated impl for those cases).
pub fn exp_minus_i_h_t_1q(h: &Matrix, t: f64) -> Matrix {
    assert_eq!(h.len(), 4, "exp_minus_i_h_t_1q requires a 2x2 matrix");
    // Check Hermitian (tolerant).
    let h00 = h[0];
    let h01 = h[1];
    let h10 = h[2];
    let h11 = h[3];
    assert!(h00.im.abs() < 1e-12 && h11.im.abs() < 1e-12, "H not Hermitian (diagonal)");
    assert!((h10 - h01.conj()).norm() < 1e-12, "H not Hermitian (off-diagonal)");
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
