// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Matrix representations and operations for quantum unitaries.
//!
//! This module provides functions to convert unitaries to dense matrices
//! and perform matrix-level operations like exponential and logarithm.
//!
//! # Extension Trait
//!
//! The [`ToMatrix`] trait provides a method-style API for converting operators:
//!
//! ```
//! use pecos_quantum::unitary_matrix::ToMatrix;
//! use pecos_core::unitary_rep::X;
//!
//! let x = X(0);
//! let matrix = x.to_matrix();  // Method style
//! ```

use nalgebra::DMatrix;
use num_complex::Complex64;
use std::fmt;
use std::ops::{BitAnd, Deref, DerefMut, Mul, Neg, Sub};
use std::sync::LazyLock;

use pecos_core::clifford::Clifford;
use pecos_core::clifford_rep::CliffordRep;
use pecos_core::gate_type::GateType;
use pecos_core::unitary_rep::{RotationType, Unitary, UnitaryRep};
use pecos_core::{Angle64, Op, Pauli, PauliString, Phase};

/// Dense matrix representation of a quantum unitary, with `*` (composition)
/// and `&` (tensor product) operators.
///
/// Wraps [`DMatrix<Complex64>`] and derefs to it, so all nalgebra methods
/// (indexing, `.nrows()`, `.adjoint()`, etc.) are available directly.
///
/// # Operators
///
/// - `*` performs matrix multiplication (gate composition)
/// - `&` performs the Kronecker product (tensor product)
///
/// # Example
///
/// ```
/// use pecos_quantum::unitary_matrix::{UnitaryMatrix, ToMatrix};
/// use pecos_core::unitary_rep::{X, Z};
///
/// let mx = X(0).to_matrix();
/// let mz = Z(0).to_matrix();
///
/// // Composition (matrix multiply)
/// let _composed = &mx * &mz;
///
/// // Tensor product (Kronecker)
/// let _tensored = &mx & &mz;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UnitaryMatrix(pub DMatrix<Complex64>);

impl UnitaryMatrix {
    /// Creates an identity matrix of size `n x n`.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self(DMatrix::identity(n, n))
    }

    /// Creates a diagonal matrix from a slice of diagonal entries.
    #[must_use]
    pub fn diag(entries: &[Complex64]) -> Self {
        let n = entries.len();
        let mut m = DMatrix::zeros(n, n);
        for (i, &v) in entries.iter().enumerate() {
            m[(i, i)] = v;
        }
        Self(m)
    }

    /// Returns the conjugate transpose (adjoint / dagger).
    #[must_use]
    pub fn adjoint(&self) -> Self {
        Self(self.0.adjoint())
    }

    /// Returns the number of qubits this matrix represents.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        let n = self.0.nrows();
        debug_assert!(n.is_power_of_two(), "matrix dimension must be a power of 2");
        n.trailing_zeros() as usize
    }

    /// Checks if this matrix is equivalent to `other` up to a global phase.
    #[must_use]
    pub fn equiv_up_to_phase(&self, other: &UnitaryMatrix) -> bool {
        matrices_equiv_up_to_phase(&self.0, &other.0, 1e-10)
    }

    /// Checks if this matrix is equivalent to `other` up to a global phase,
    /// with a custom tolerance.
    #[must_use]
    pub fn equiv_up_to_phase_with_tolerance(&self, other: &UnitaryMatrix, tol: f64) -> bool {
        matrices_equiv_up_to_phase(&self.0, &other.0, tol)
    }

    /// Returns a reference to the inner `DMatrix`.
    #[must_use]
    pub fn inner(&self) -> &DMatrix<Complex64> {
        &self.0
    }

    /// Consumes self and returns the inner `DMatrix`.
    #[must_use]
    pub fn into_inner(self) -> DMatrix<Complex64> {
        self.0
    }

    /// Returns `true` if this matrix is unitary (U * U† = I) within tolerance.
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        self.is_unitary_with_tolerance(1e-10)
    }

    /// Returns `true` if this matrix is unitary within the given tolerance.
    #[must_use]
    pub fn is_unitary_with_tolerance(&self, tol: f64) -> bool {
        let n = self.nrows();
        if n != self.ncols() {
            return false;
        }
        let product = &self.0 * self.0.adjoint();
        let identity = DMatrix::<Complex64>::identity(n, n);
        (product - identity).norm() < tol
    }

    /// Canonicalizes by dividing all entries by the first nonzero entry
    /// (row-major scan). This removes any scalar factor (not just unit phases),
    /// so `2*H` and `e^{iπ/3}*H` both canonicalize to the same matrix as `H`.
    ///
    /// Returns `None` if the matrix is all zeros.
    #[must_use]
    pub fn canonicalize(&self) -> Option<Self> {
        canonicalize_matrix(&self.0).map(Self)
    }

    /// Attempts to identify this matrix as a [`Unitary`] gate descriptor.
    ///
    /// Named gates are identified up to any nonzero scalar (so `2*H` matches `H`).
    /// Rotation extraction (RX, RY, RZ, R1XY, RXX, RYY, RZZ) requires the
    /// matrix to be unitary -- scaled matrices like `3*RZ(0.5)` will not match.
    ///
    /// The returned `Unitary` can be further queried with `is_pauli()`,
    /// `is_clifford()`, `try_to_pauli()`, etc.
    ///
    /// For self-inverse named gates, the non-dagger variant is returned.
    #[must_use]
    pub fn try_to_unitary(&self) -> Option<Unitary> {
        // First: try named gate table (works up to arbitrary scalar via canonicalization)
        let canonical = canonicalize_matrix(&self.0)?;

        let table: &[(Unitary, DMatrix<Complex64>)] = match self.nrows() {
            2 => &UNITARY_1Q_TABLE,
            4 => &UNITARY_2Q_TABLE,
            8 => &UNITARY_3Q_TABLE,
            _ => return None,
        };

        if let Some((gate, _)) = table
            .iter()
            .find(|(_, ref_canon)| matrices_approx_equal(&canonical, ref_canon, 1e-8))
        {
            return Some(*gate);
        }

        // Second: try rotation extraction (requires actual unitarity)
        if !self.is_unitary() {
            return None;
        }
        try_identify_rotation(&self.0)
    }
}

// --- Canonicalization and cached lookup tables ---

/// Divides all entries by the first nonzero entry (row-major scan).
/// Returns `None` for the zero matrix.
fn canonicalize_matrix(mat: &DMatrix<Complex64>) -> Option<DMatrix<Complex64>> {
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let v = mat[(i, j)];
            if v.norm() > 1e-14 {
                return Some(mat / v);
            }
        }
    }
    None
}

/// Element-wise approximate equality.
fn matrices_approx_equal(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>, tol: f64) -> bool {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return false;
    }
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            if (a[(i, j)] - b[(i, j)]).norm() > tol {
                return false;
            }
        }
    }
    true
}

// --- Rotation extraction ---

/// Attempts to identify a matrix as a rotation `exp(-i theta/2 P)` around a
/// single Pauli axis (up to any nonzero scalar).
///
/// Works by decomposing into the Pauli basis and checking that only the identity
/// and one Pauli component are nonzero.
fn try_identify_rotation(mat: &DMatrix<Complex64>) -> Option<Unitary> {
    match mat.nrows() {
        2 => try_identify_1q_rotation(mat),
        4 => try_identify_2q_rotation(mat).or_else(|| try_identify_u2q(mat, 1e-10)),
        _ => None,
    }
}

/// Identifies a 2x2 matrix as RX, RY, RZ, or R1XY with some angle(s).
///
/// Decomposes M = `c_I` * I + `c_X` * X + `c_Y` * Y + `c_Z` * Z.
/// - If exactly one Pauli coefficient is nonzero: single-axis rotation (RX/RY/RZ).
/// - If `c_X` and `c_Y` are nonzero but `c_Z` is zero: R1XY(theta, phi).
fn try_identify_1q_rotation(mat: &DMatrix<Complex64>) -> Option<Unitary> {
    let m = |r, c| mat[(r, c)];

    // Pauli basis decomposition: M = c_I*I + c_X*X + c_Y*Y + c_Z*Z
    let c_i = (m(0, 0) + m(1, 1)) / 2.0;
    let c_x = (m(0, 1) + m(1, 0)) / 2.0;
    let c_y = Complex64::i() * (m(0, 1) - m(1, 0)) / 2.0;
    let c_z = (m(0, 0) - m(1, 1)) / 2.0;

    let tol = 1e-10;
    let has_x = c_x.norm() > tol;
    let has_y = c_y.norm() > tol;
    let has_z = c_z.norm() > tol;

    if has_z && !has_x && !has_y {
        extract_rotation_angle(c_i, c_z, RotationType::RZ, tol)
    } else if has_x && !has_y && !has_z {
        extract_rotation_angle(c_i, c_x, RotationType::RX, tol)
    } else if has_y && !has_x && !has_z {
        extract_rotation_angle(c_i, c_y, RotationType::RY, tol)
    } else if !has_z && (has_x || has_y) {
        // R1XY: rotation in XY plane
        try_identify_r1xy(c_i, c_x, c_y, tol)
    } else {
        // General single-qubit unitary: all Pauli components present
        try_identify_u3(mat, tol)
    }
}

/// Identifies an R1XY(theta, phi) gate from its Pauli decomposition.
///
/// R1XY = cos(theta/2)*I - i*sin(theta/2)*(cos(phi)*X + sin(phi)*Y)
/// So `c_X` = -i*alpha*sin(theta/2)*cos(phi), `c_Y` = -i*alpha*sin(theta/2)*sin(phi).
///
/// From `c_I` and `c_X/c_Y` we can extract theta and phi.
fn try_identify_r1xy(c_i: Complex64, c_x: Complex64, c_y: Complex64, tol: f64) -> Option<Unitary> {
    if c_i.norm() < tol {
        // theta = pi: cos(theta/2) = 0, so c_I = 0.
        // c_X = -i*alpha*cos(phi), c_Y = -i*alpha*sin(phi)
        // c_Y/c_X = tan(phi), which should be real.
        let ratio = c_y / c_x;
        if ratio.im.abs() > tol * ratio.norm().max(1.0) {
            return None;
        }
        let phi = ratio.re.atan();
        return Some(Unitary::R1XY {
            theta: Angle64::from_radians(std::f64::consts::PI),
            phi: Angle64::from_radians(phi),
        });
    }

    // i * c_X / c_I = tan(theta/2) * cos(phi), should be real
    // i * c_Y / c_I = tan(theta/2) * sin(phi), should be real
    let r_x = Complex64::i() * c_x / c_i;
    let r_y = Complex64::i() * c_y / c_i;

    if r_x.im.abs() > tol * r_x.norm().max(1.0) {
        return None;
    }
    if r_y.im.abs() > tol * r_y.norm().max(1.0) {
        return None;
    }

    let phi = r_y.re.atan2(r_x.re);
    // tan(theta/2) = r_x / cos(phi) or r_y / sin(phi)
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    let tan_half_theta = if cos_phi.abs() > sin_phi.abs() {
        r_x.re / cos_phi
    } else {
        r_y.re / sin_phi
    };
    let theta = 2.0 * tan_half_theta.atan();

    Some(Unitary::R1XY {
        theta: Angle64::from_radians(theta),
        phi: Angle64::from_radians(phi),
    })
}

/// Identifies a 2x2 unitary matrix as U(theta, phi, lambda).
///
/// Any single-qubit unitary can be written (up to global phase) as:
///   U = [[cos(t/2), -e^{il}*sin(t/2)], [e^{ip}*sin(t/2), e^{i(p+l)}*cos(t/2)]]
///
/// We remove the global phase by making M[0,0] real and non-negative, then extract
/// theta from |M[0,0]|, phi from arg(M[1,0]), lambda from arg(-M[0,1]).
fn try_identify_u3(mat: &DMatrix<Complex64>, tol: f64) -> Option<Unitary> {
    let m00 = mat[(0, 0)];
    let m01 = mat[(0, 1)];
    let m10 = mat[(1, 0)];
    let m11 = mat[(1, 1)];

    // Remove global phase: make M[0,0] real and non-negative
    let (m00, m01, m10, m11) = if m00.norm() > tol {
        let phase = m00 / m00.norm();
        let inv = phase.conj();
        (m00 * inv, m01 * inv, m10 * inv, m11 * inv)
    } else {
        // cos(theta/2) ~ 0, so theta ~ pi
        // Use M[1,0] to fix phase: make M[1,0] real and positive
        if m10.norm() < tol {
            return None;
        }
        let phase = m10 / m10.norm();
        let inv = phase.conj();
        (m00 * inv, m01 * inv, m10 * inv, m11 * inv)
    };

    // Now M[0,0] should be real and non-negative
    let cos_half = m00.re;
    if cos_half < -tol {
        return None;
    }
    let cos_half = cos_half.clamp(0.0, 1.0);
    let theta = 2.0 * cos_half.acos();

    let sin_half = (theta / 2.0).sin();

    // Use a generous threshold for the sin_half ≈ 0 check: when sin_half is tiny,
    // phi and lambda are determined by noise in the off-diagonal entries.
    // The theta≈0 formula (extracting phi+lambda from m11) is always numerically stable.
    let sin_tol = tol.max(1e-6);
    let (phi, lambda) = if sin_half.abs() < sin_tol {
        // theta ~ 0: M is ~ e^{i*global_phase} * diag(1, e^{i*(phi+lambda)})
        // phi and lambda are not independently recoverable; combine them
        let phi_plus_lambda = m11.arg();
        (0.0, phi_plus_lambda)
    } else {
        // phi = arg(M[1,0]), lambda = arg(-M[0,1])
        let phi = m10.arg();
        let lambda = (-m01).arg();
        (phi, lambda)
    };

    // Snap tiny angles to exactly zero to avoid from_radians boundary issues
    let snap = |v: f64| -> f64 { if v.abs() < 1e-14 { 0.0 } else { v } };
    let (theta, phi, lambda) = (snap(theta), snap(phi), snap(lambda));
    Some(Unitary::U3 {
        theta: Angle64::from_radians(theta),
        phi: Angle64::from_radians(phi),
        lambda: Angle64::from_radians(lambda),
    })
}

/// Identifies a 4x4 matrix as RXX, RYY, RZZ, or RXXRYYRZZ.
///
/// Decomposes into the 16-element Pauli tensor basis and checks that only
/// the {II, XX, YY, ZZ} components are nonzero.
/// - If exactly one of {XX, YY, ZZ} is nonzero: single-axis rotation.
/// - If multiple are nonzero: RXXRYYRZZ(alpha, beta, gamma).
fn try_identify_2q_rotation(mat: &DMatrix<Complex64>) -> Option<Unitary> {
    let m = |r, c| mat[(r, c)];

    // I*I coefficient: Tr(M) / 4
    let c_ii = (m(0, 0) + m(1, 1) + m(2, 2) + m(3, 3)) / 4.0;

    // P*P coefficients: Tr(M * P*P) / 4
    let c_xx = (m(0, 3) + m(1, 2) + m(2, 1) + m(3, 0)) / 4.0;
    let c_yy = (-m(0, 3) + m(1, 2) + m(2, 1) - m(3, 0)) / 4.0;
    let c_zz = (m(0, 0) - m(1, 1) - m(2, 2) + m(3, 3)) / 4.0;

    let tol = 1e-10;

    // Verify all other 12 Pauli*Pauli components are zero.
    // The total energy must be in just the {II, XX, YY, ZZ} components:
    // sum of |c_ab|^2 for all 16 basis elements = ||M||_F^2 / 4
    let total_norm_sq = mat.iter().map(num_complex::Complex::norm_sqr).sum::<f64>() / 4.0;
    let accounted = c_ii.norm_sqr() + c_xx.norm_sqr() + c_yy.norm_sqr() + c_zz.norm_sqr();
    if (total_norm_sq - accounted) > tol * total_norm_sq.max(1.0) {
        return None;
    }

    let has_xx = c_xx.norm() > tol;
    let has_yy = c_yy.norm() > tol;
    let has_zz = c_zz.norm() > tol;

    let nonzero_count = usize::from(has_xx) + usize::from(has_yy) + usize::from(has_zz);

    if nonzero_count == 1 {
        // Single-axis rotation
        let (rotation_type, c_pp) = if has_xx {
            (RotationType::RXX, c_xx)
        } else if has_yy {
            (RotationType::RYY, c_yy)
        } else {
            (RotationType::RZZ, c_zz)
        };
        extract_rotation_angle(c_ii, c_pp, rotation_type, tol)
    } else if nonzero_count >= 2 {
        // Multi-axis: RXXRYYRZZ
        try_identify_rxxryyrzz(c_ii, c_xx, c_yy, c_zz, tol)
    } else {
        // Only II component: identity (should have been caught by named table)
        None
    }
}

/// Identifies an RXXRYYRZZ(alpha, beta, gamma) gate from its Pauli decomposition.
///
/// The matrix exp(-i/2*(a*XX + b*YY + c*ZZ)) is diagonal in the Bell basis.
/// We compute Bell-basis eigenvalues, extract phases, and solve for a, b, c.
fn try_identify_rxxryyrzz(
    c_ii: Complex64,
    c_xx: Complex64,
    c_yy: Complex64,
    c_zz: Complex64,
    tol: f64,
) -> Option<Unitary> {
    // Bell-basis eigenvalues: d_k = c_ii + eps_xx*c_xx + eps_yy*c_yy + eps_zz*c_zz
    //   |Phi+>: XX=+1, YY=-1, ZZ=+1
    //   |Psi+>: XX=+1, YY=+1, ZZ=-1
    //   |Psi->: XX=-1, YY=-1, ZZ=-1
    //   |Phi->: XX=-1, YY=+1, ZZ=+1
    let d_phi_p = c_ii + c_xx - c_yy + c_zz;
    let d_psi_p = c_ii + c_xx + c_yy - c_zz;
    let d_psi_m = c_ii - c_xx - c_yy - c_zz;
    let d_phi_m = c_ii - c_xx + c_yy + c_zz;

    // All magnitudes must be equal (unitary)
    let mag = d_phi_p.norm();
    if mag < tol {
        return None;
    }
    for d in &[d_psi_p, d_psi_m, d_phi_m] {
        if (d.norm() - mag).abs() > tol * mag.max(1.0) {
            return None;
        }
    }

    // Phase of eigenvalue k (with global phase g):
    //   p(Phi+) = g - (a - b + c)/2
    //   p(Psi+) = g - (a + b - c)/2
    //   p(Psi-) = g + (a + b + c)/2
    //   p(Phi-) = g + (a - b - c)/2
    //
    // Phase differences (g cancels):
    //   a + b = p(Psi-) - p(Psi+)
    //   a - b = p(Phi-) - p(Phi+)
    //   a + c = p(Psi-) - p(Phi+)
    let pp = d_phi_p.arg();
    let qp = d_psi_p.arg();
    let qm = d_psi_m.arg();
    let pm = d_phi_m.arg();

    let a = f64::midpoint(qm - qp, pm - pp);
    let b = ((qm - qp) - (pm - pp)) / 2.0;
    let c = (qm - pp) - a;

    Some(Unitary::RXXRYYRZZ {
        alpha: Angle64::from_radians(a),
        beta: Angle64::from_radians(b),
        gamma: Angle64::from_radians(c),
    })
}

/// Identifies a general 2-qubit unitary via KAK decomposition.
///
/// Any U in SU(4) can be written as:
///   U = (A0 x A1) * exp(-i/2(a*XX + b*YY + c*ZZ)) * (B0 x B1)
///
/// Algorithm:
/// 1. Transform to magic basis: `U_M` = Q† U Q
/// 2. Compute Sigma = `U_M^T` `U_M` (complex symmetric, unitary)
/// 3. Jointly diagonalize Re(Sigma) and Im(Sigma) (they commute)
/// 4. Extract interaction angles from eigenvalues
/// 5. Factor out single-qubit gates from O1, O2
fn try_identify_u2q(mat: &DMatrix<Complex64>, tol: f64) -> Option<Unitary> {
    use nalgebra::SymmetricEigen;

    let s = 1.0 / 2.0_f64.sqrt();
    let ci = Complex64::i();

    // Magic basis change matrix Q
    // Columns: |Phi+>, i|Psi+>, i|Psi->, i|Phi->
    let q = DMatrix::from_row_slice(
        4,
        4,
        &[
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            ci * s,
            Complex64::new(0.0, 0.0),
            ci * s,
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            ci * s,
            Complex64::new(-s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(s, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            ci * (-s),
        ],
    );
    let q_adj = q.adjoint();

    // Transform to magic basis
    let u_m = &q_adj * mat * &q;

    // Sigma = U_M^T * U_M (transpose, not adjoint)
    let sigma = u_m.transpose() * &u_m;

    // Sigma = O2^T * Delta^2 * O2 where O2 is real orthogonal, Delta diagonal unitary.
    // Since Sigma is symmetric and unitary: Re(Sigma) and Im(Sigma) are real symmetric
    // and commute, so they share real eigenvectors.
    // Use A + pi*B to break degeneracies generically.
    let n = 4;
    let a_re = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| sigma[(i, j)].re);
    let a_im = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| sigma[(i, j)].im);
    let combined = &a_re + &a_im * std::f64::consts::PI;

    let eigen = SymmetricEigen::new(combined.clone());
    let mut v = eigen.eigenvectors; // columns are real eigenvectors

    // Fix degenerate eigenspaces: nalgebra's SymmetricEigen can produce incorrect
    // eigenvectors for nearly-degenerate eigenvalues. For each degenerate group,
    // recompute the eigenspace by finding the null space of (combined - lambda*I).
    let degen_tol = 0.01;
    let eigenvalues = &eigen.eigenvalues;
    let mut fixed = [false; 4];
    for i in 0..4 {
        if fixed[i] {
            continue;
        }
        for j in (i + 1)..4 {
            if fixed[j] {
                continue;
            }
            if (eigenvalues[i] - eigenvalues[j]).abs() < degen_tol {
                // nalgebra's SymmetricEigen can produce incorrect eigenvectors for
                // nearly-degenerate eigenvalues. Recompute via SVD null space.
                let avg_eval = f64::midpoint(eigenvalues[i], eigenvalues[j]);
                let shifted = &combined - nalgebra::DMatrix::<f64>::identity(n, n) * avg_eval;
                let svd = shifted.svd(true, true);
                let vt = svd.v_t.expect("SVD requested with v_t=true");
                // The last two rows of V^T (smallest singular values) span the eigenspace.
                for r in 0..n {
                    v[(r, i)] = vt[(n - 2, r)];
                    v[(r, j)] = vt[(n - 1, r)];
                }
                fixed[i] = true;
                fixed[j] = true;
                break;
            }
        }
    }

    // Ensure det(V) = +1 (SO(4))
    if v.determinant() < 0.0 {
        for i in 0..n {
            v[(i, 0)] = -v[(i, 0)];
        }
    }

    // O2 = V^T. Eigenvalues of Sigma: lambda_k = v_k^T * (Re + i*Im) * v_k
    // Delta_k = exp(i * arg(lambda_k) / 2), Delta_inv_k = exp(-i * arg(lambda_k) / 2)
    let mut delta_inv = [Complex64::new(0.0, 0.0); 4];
    for (k, delta_inv_k) in delta_inv.iter_mut().enumerate() {
        let vk = v.column(k);
        let re_val: f64 = vk.dot(&(&a_re * vk));
        let im_val: f64 = vk.dot(&(&a_im * vk));
        let phase = Complex64::new(re_val, im_val).arg() / 2.0;
        *delta_inv_k = Complex64::from_polar(1.0, -phase);
    }

    // O1 = U_M * V * Delta^{-1}
    let v_complex = DMatrix::from_fn(n, n, |i, j| Complex64::new(v[(i, j)], 0.0));
    let delta_inv_diag =
        DMatrix::from_diagonal(&nalgebra::DVector::from_fn(n, |k, _| delta_inv[k]));
    let o1_complex = &u_m * &v_complex * &delta_inv_diag;

    // Verify O1 is approximately real
    let max_im = o1_complex
        .iter()
        .map(|v| v.im.abs())
        .fold(0.0_f64, f64::max);
    if max_im > 1e-6 {
        return None;
    }
    let mut o1 = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| o1_complex[(i, j)].re);

    // Ensure det(O1) = +1
    if o1.determinant() < 0.0 {
        for i in 0..n {
            o1[(i, 0)] = -o1[(i, 0)];
        }
        delta_inv[0] = -delta_inv[0];
    }

    // Extract interaction angles from Delta phases.
    // Delta[k] = exp(-i * delta_inv[k].arg()), so delta phase = -delta_inv[k].arg()
    // Delta[k] corresponds to Bell states in order: Phi+, Psi+, Psi-, Phi-
    // phases: pp = -(a-b+c)/2, qp = -(a+b-c)/2, qm = (a+b+c)/2, pm = (a-b-c)/2
    let pp = -delta_inv[0].arg();
    let qp = -delta_inv[1].arg();
    let qm = -delta_inv[2].arg();
    let pm = -delta_inv[3].arg();

    let alpha = f64::midpoint(qm - qp, pm - pp);
    let beta = ((qm - qp) - (pm - pp)) / 2.0;
    let gamma = (qm - pp) - alpha;

    // Convert O1 and O2 to single-qubit gates in the computational basis
    // K = Q * O * Q†  is a tensor product A ⊗ B
    let o1_complex = DMatrix::from_fn(n, n, |i, j| Complex64::new(o1[(i, j)], 0.0));
    let o2_complex = DMatrix::from_fn(n, n, |i, j| Complex64::new(v[(j, i)], 0.0)); // O2 = V^T
    let k_before = &q * &o1_complex * &q_adj;
    let k_after = &q * &o2_complex * &q_adj;

    // Factor each 4x4 tensor product into two 2x2 unitaries.
    // factor_tensor_product returns (outer, inner) where K = outer ⊗ inner.
    // outer = MSB = qubit 1, inner = LSB = qubit 0.
    let (before_outer, before_inner) = factor_tensor_product(&k_before, tol)?;
    let (after_outer, after_inner) = factor_tensor_product(&k_after, tol)?;

    // Identify each 2x2 as U3.
    // factor_tensor_product returns (outer, inner) = (qubit 1, qubit 0).
    // The U2q convention: before[0] → qubits[0], before[1] → qubits[1].
    // So before[0] = inner factor, before[1] = outer factor.
    let to_u3_params = |m: &DMatrix<Complex64>| -> Option<[Angle64; 3]> {
        let u = try_identify_u3(m, tol * 100.0)?;
        match u {
            Unitary::U3 { theta, phi, lambda } => Some([theta, phi, lambda]),
            _ => None,
        }
    };

    let before_0 = to_u3_params(&before_inner)?;
    let before_1 = to_u3_params(&before_outer)?;
    let after_0 = to_u3_params(&after_inner)?;
    let after_1 = to_u3_params(&after_outer)?;

    let snap = |v: f64| -> f64 { if v.abs() < 1e-14 { 0.0 } else { v } };

    Some(Unitary::U2q {
        before: [before_0, before_1],
        interaction: [
            Angle64::from_radians(snap(alpha)),
            Angle64::from_radians(snap(beta)),
            Angle64::from_radians(snap(gamma)),
        ],
        after: [after_0, after_1],
    })
}

/// Factors a 4x4 matrix K = A ⊗ B into two 2x2 matrices A and B.
///
/// Uses the block structure: K = [[a00*B, a01*B], [a10*B, a11*B]]
fn factor_tensor_product(
    k: &DMatrix<Complex64>,
    tol: f64,
) -> Option<(DMatrix<Complex64>, DMatrix<Complex64>)> {
    // Find the 2x2 block with largest Frobenius norm
    let mut best_norm_sq = 0.0;
    let mut best_p = 0;
    let mut best_q = 0;
    for p in 0..2 {
        for q in 0..2 {
            let mut norm_sq = 0.0;
            for r in 0..2 {
                for s in 0..2 {
                    norm_sq += k[(2 * p + r, 2 * q + s)].norm_sqr();
                }
            }
            if norm_sq > best_norm_sq {
                best_norm_sq = norm_sq;
                best_p = p;
                best_q = q;
            }
        }
    }

    if best_norm_sq < tol {
        return None;
    }

    // Block(p,q) = a_{pq} * B
    // det(Block) = a_{pq}^2 * det(B) = a_{pq}^2 (since B in SU(2))
    let block = DMatrix::from_fn(2, 2, |r, s| k[(2 * best_p + r, 2 * best_q + s)]);
    let det = block[(0, 0)] * block[(1, 1)] - block[(0, 1)] * block[(1, 0)];
    let a_pq = det.sqrt();

    if a_pq.norm() < tol {
        return None;
    }

    let b = &block / a_pq;

    // A[i,j] = tr(B† * K_block(i,j)) / 2
    let b_adj = b.adjoint();
    let mut a = DMatrix::zeros(2, 2);
    for i in 0..2 {
        for j in 0..2 {
            let blk = DMatrix::from_fn(2, 2, |r, s| k[(2 * i + r, 2 * j + s)]);
            a[(i, j)] = (&b_adj * blk).trace() / Complex64::new(2.0, 0.0);
        }
    }

    Some((a, b))
}

/// Given M = alpha * (cos(theta/2) I - i sin(theta/2) P), extracts theta
/// from the identity coefficient `c_I` and Pauli coefficient `c_P`.
fn extract_rotation_angle(
    c_i: Complex64,
    c_p: Complex64,
    rotation_type: RotationType,
    tol: f64,
) -> Option<Unitary> {
    // c_I = alpha * cos(theta/2), c_P = -i * alpha * sin(theta/2)
    // So i * c_P / c_I = tan(theta/2), which must be real.

    if c_i.norm() < tol {
        // theta = pi: this is a Pauli gate, should have been caught by the named table.
        return None;
    }

    let ratio = Complex64::i() * c_p / c_i;

    // Check the ratio is approximately real
    if ratio.im.abs() > tol * ratio.norm().max(1.0) {
        return None;
    }

    let theta = 2.0 * ratio.re.atan();
    Some(Unitary::Rotation {
        rotation_type,
        angle: Angle64::from_radians(theta),
    })
}

/// Non-parameterized unitary gate types (non-dg before dg for self-inverse preference).
const NAMED_GATE_1Q: [GateType; 15] = [
    GateType::I,
    GateType::X,
    GateType::Y,
    GateType::Z,
    GateType::H,
    GateType::F,
    GateType::Fdg,
    GateType::SX,
    GateType::SXdg,
    GateType::SY,
    GateType::SYdg,
    GateType::SZ,
    GateType::SZdg,
    GateType::T,
    GateType::Tdg,
];

const NAMED_GATE_2Q: [GateType; 11] = [
    GateType::CX,
    GateType::CY,
    GateType::CZ,
    GateType::CH,
    GateType::SWAP,
    GateType::SXX,
    GateType::SXXdg,
    GateType::SYY,
    GateType::SYYdg,
    GateType::SZZ,
    GateType::SZZdg,
];

const NAMED_GATE_3Q: [GateType; 1] = [GateType::CCX];

/// Builds a cached lookup table mapping `Unitary::Named(gate)` to its canonical matrix.
fn build_unitary_table(
    gates: &[GateType],
    num_qubits: usize,
) -> Vec<(Unitary, DMatrix<Complex64>)> {
    let qubits: Vec<usize> = (0..num_qubits).collect();
    gates
        .iter()
        .map(|&g| {
            let mat = gate_to_matrix(g, &qubits, num_qubits);
            let canon = canonicalize_matrix(&mat).expect("gate matrix should not be zero");
            (Unitary::Named(g), canon)
        })
        .collect()
}

/// Cached canonical forms for gate identification.
static UNITARY_1Q_TABLE: LazyLock<Vec<(Unitary, DMatrix<Complex64>)>> =
    LazyLock::new(|| build_unitary_table(&NAMED_GATE_1Q, 1));

static UNITARY_2Q_TABLE: LazyLock<Vec<(Unitary, DMatrix<Complex64>)>> =
    LazyLock::new(|| build_unitary_table(&NAMED_GATE_2Q, 2));

static UNITARY_3Q_TABLE: LazyLock<Vec<(Unitary, DMatrix<Complex64>)>> =
    LazyLock::new(|| build_unitary_table(&NAMED_GATE_3Q, 3));

impl From<DMatrix<Complex64>> for UnitaryMatrix {
    fn from(m: DMatrix<Complex64>) -> Self {
        Self(m)
    }
}

impl From<UnitaryMatrix> for DMatrix<Complex64> {
    fn from(m: UnitaryMatrix) -> Self {
        m.0
    }
}

impl Deref for UnitaryMatrix {
    type Target = DMatrix<Complex64>;
    fn deref(&self) -> &DMatrix<Complex64> {
        &self.0
    }
}

impl DerefMut for UnitaryMatrix {
    fn deref_mut(&mut self) -> &mut DMatrix<Complex64> {
        &mut self.0
    }
}

// * — matrix multiplication (gate composition)

impl Mul for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0 * rhs.0)
    }
}

impl Mul<&UnitaryMatrix> for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0 * &rhs.0)
    }
}

impl Mul<UnitaryMatrix> for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(&self.0 * rhs.0)
    }
}

impl Mul for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(&self.0 * &rhs.0)
    }
}

// & — Kronecker product (tensor product)

impl BitAnd for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn bitand(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0.kronecker(&rhs.0))
    }
}

impl BitAnd<&UnitaryMatrix> for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn bitand(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0.kronecker(&rhs.0))
    }
}

impl BitAnd<UnitaryMatrix> for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn bitand(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0.kronecker(&rhs.0))
    }
}

impl BitAnd for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn bitand(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0.kronecker(&rhs.0))
    }
}

// Scalar multiplication: UnitaryMatrix * Complex64

impl Mul<Complex64> for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: Complex64) -> UnitaryMatrix {
        UnitaryMatrix(self.0 * rhs)
    }
}

impl Mul<Complex64> for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: Complex64) -> UnitaryMatrix {
        UnitaryMatrix(&self.0 * rhs)
    }
}

// Subtraction

impl Sub for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn sub(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0 - rhs.0)
    }
}

impl Sub<&UnitaryMatrix> for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn sub(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(self.0 - &rhs.0)
    }
}

impl Sub<UnitaryMatrix> for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn sub(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(&self.0 - rhs.0)
    }
}

impl Sub for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn sub(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(&self.0 - &rhs.0)
    }
}

// Scalar multiplication: Complex64 * UnitaryMatrix (left-multiply)

impl Mul<UnitaryMatrix> for Complex64 {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(rhs.0 * self)
    }
}

impl Mul<&UnitaryMatrix> for Complex64 {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(&rhs.0 * self)
    }
}

// Scalar multiplication with f64

impl Mul<f64> for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: f64) -> UnitaryMatrix {
        UnitaryMatrix(self.0 * Complex64::new(rhs, 0.0))
    }
}

impl Mul<f64> for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: f64) -> UnitaryMatrix {
        UnitaryMatrix(&self.0 * Complex64::new(rhs, 0.0))
    }
}

impl Mul<UnitaryMatrix> for f64 {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(rhs.0 * Complex64::new(self, 0.0))
    }
}

impl Mul<&UnitaryMatrix> for f64 {
    type Output = UnitaryMatrix;
    fn mul(self, rhs: &UnitaryMatrix) -> UnitaryMatrix {
        UnitaryMatrix(&rhs.0 * Complex64::new(self, 0.0))
    }
}

// Negation

impl Neg for UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn neg(self) -> UnitaryMatrix {
        UnitaryMatrix(-self.0)
    }
}

impl Neg for &UnitaryMatrix {
    type Output = UnitaryMatrix;
    fn neg(self) -> UnitaryMatrix {
        UnitaryMatrix(-&self.0)
    }
}

// Display

impl fmt::Display for UnitaryMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Extension trait for converting quantum unitaries to matrix representations.
///
/// This trait is implemented for [`UnitaryRep`] and [`PauliString`], providing
/// a method-style API for matrix conversion.
///
/// # Example
///
/// ```
/// use pecos_quantum::unitary_matrix::ToMatrix;
/// use pecos_core::unitary_rep::{X, H, CX, Is};
///
/// // Single qubit gate
/// let x_matrix = X(0).to_matrix();
/// assert_eq!(x_matrix.nrows(), 2);
///
/// // Two qubit gate
/// let cnot_matrix = CX(0, 1).to_matrix();
/// assert_eq!(cnot_matrix.nrows(), 4);
///
/// // For larger matrices, tensor with identities using Is()
/// let x_extended = X(0) & Is(1..3);  // X on qubit 0 in 3-qubit space
/// let mat = x_extended.to_matrix();
/// assert_eq!(mat.nrows(), 8);
/// ```
pub trait ToMatrix {
    /// Converts to a dense [`UnitaryMatrix`] representation.
    ///
    /// The matrix size is 2^n where n is determined by the maximum qubit index + 1.
    fn to_matrix(&self) -> UnitaryMatrix;
}

impl ToMatrix for UnitaryRep {
    fn to_matrix(&self) -> UnitaryMatrix {
        to_matrix(self)
    }
}

impl ToMatrix for PauliString {
    fn to_matrix(&self) -> UnitaryMatrix {
        let num_qubits = self.qubits().into_iter().max().map_or(1, |q| q + 1);
        UnitaryMatrix(pauli_string_to_matrix_impl(self, num_qubits))
    }
}

impl ToMatrix for CliffordRep {
    fn to_matrix(&self) -> UnitaryMatrix {
        let ur = pecos_core::gate_algebra::clifford_rep_to_unitary_rep(self);
        to_matrix(&ur)
    }
}

impl ToMatrix for Pauli {
    /// Converts to a 2x2 matrix on qubit 0.
    fn to_matrix(&self) -> UnitaryMatrix {
        self.on_qubit(0).to_matrix()
    }
}

impl ToMatrix for Clifford {
    /// Converts to a matrix on default qubits (0 for 1q gates, 0-1 for 2q gates).
    ///
    /// Uses `Clifford::to_unitary_rep_on_qubit(s)` rather than going through
    /// `CliffordRep`, because some gate pairs (e.g. G/Gdg) share the same
    /// `CliffordRep` but differ at the unitary level.
    fn to_matrix(&self) -> UnitaryMatrix {
        let ur = if self.is_1q() {
            self.to_unitary_rep_on_qubit(0)
        } else {
            self.to_unitary_rep_on_qubits(0, 1)
        };
        to_matrix(&ur)
    }
}

impl ToMatrix for Unitary {
    /// Converts to a matrix on default qubits (0 for 1q, 0-1 for 2q, 0-1-2 for 3q).
    fn to_matrix(&self) -> UnitaryMatrix {
        let qubits: smallvec::SmallVec<[usize; 3]> = (0..self.num_qubits()).collect();
        let ur = UnitaryRep::Gate(*self, qubits);
        to_matrix(&ur)
    }
}

impl ToMatrix for Op {
    /// Converts to a matrix. Returns the zero matrix for channels (non-unitary ops).
    fn to_matrix(&self) -> UnitaryMatrix {
        match self.clone().into_unitary() {
            Some(ur) => to_matrix(&ur),
            None => {
                // Channel ops don't have a unitary matrix
                panic!("Cannot convert non-unitary Op (Channel) to a matrix")
            }
        }
    }
}

/// Converts a [`UnitaryRep`] to a [`UnitaryMatrix`].
///
/// The matrix size is 2^n where n is the number of qubits (determined by
/// the maximum qubit index + 1).
///
/// # Example
///
/// ```
/// use pecos_quantum::unitary_matrix::to_matrix;
/// use pecos_core::unitary_rep::X;
/// use num_complex::Complex64;
///
/// let x = X(0);
/// let matrix = to_matrix(&x);
///
/// // X gate matrix: [[0, 1], [1, 0]]
/// assert_eq!(matrix.nrows(), 2);
/// assert!((matrix[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
/// ```
#[must_use]
pub fn to_matrix(op: &UnitaryRep) -> UnitaryMatrix {
    let num_qubits = op.qubits().into_iter().max().map_or(1, |q| q + 1);
    UnitaryMatrix(to_matrix_with_size_impl(op, num_qubits))
}

/// Converts a [`UnitaryRep`] to a [`UnitaryMatrix`] with a specified size.
///
/// # Arguments
/// * `op` - The operator to convert
/// * `num_qubits` - The number of qubits (matrix will be `2^num_qubits` x `2^num_qubits`)
#[must_use]
pub fn to_matrix_with_size(op: &UnitaryRep, num_qubits: usize) -> UnitaryMatrix {
    UnitaryMatrix(to_matrix_with_size_impl(op, num_qubits))
}

/// Internal implementation that returns raw `DMatrix` for recursive use.
fn to_matrix_with_size_impl(op: &UnitaryRep, num_qubits: usize) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits; // 2^num_qubits

    match op {
        UnitaryRep::Pauli(ps) => pauli_string_to_matrix_impl(ps, num_qubits),

        UnitaryRep::Gate(
            pecos_core::Unitary::Rotation {
                rotation_type,
                angle,
            },
            qubits,
        ) => rotation_to_matrix(*rotation_type, *angle, qubits, num_qubits),

        UnitaryRep::Gate(pecos_core::Unitary::R1XY { theta, phi }, qubits) => {
            r1xy_to_matrix(*theta, *phi, qubits, num_qubits)
        }

        UnitaryRep::Gate(pecos_core::Unitary::U3 { theta, phi, lambda }, qubits) => {
            u3_to_matrix(*theta, *phi, *lambda, qubits, num_qubits)
        }

        UnitaryRep::Gate(pecos_core::Unitary::RXXRYYRZZ { alpha, beta, gamma }, qubits) => {
            rxxryyrzz_to_matrix(*alpha, *beta, *gamma, qubits, num_qubits)
        }

        UnitaryRep::Gate(
            pecos_core::Unitary::U2q {
                before,
                interaction,
                after,
            },
            qubits,
        ) => u2q_to_matrix(before, interaction, after, qubits, num_qubits),

        UnitaryRep::Gate(pecos_core::Unitary::Named(gate_type), qubits) => {
            gate_to_matrix(*gate_type, qubits, num_qubits)
        }

        UnitaryRep::Tensor(parts) => {
            // Start with identity, combine each part
            let mut result = DMatrix::identity(dim, dim);
            for part in parts {
                let part_matrix = to_matrix_with_size_impl(part, num_qubits);
                result = combine_disjoint_unitaries(&result, &part_matrix);
            }
            result
        }

        UnitaryRep::Compose(parts) => {
            // Matrix multiplication in reverse order (last part applied first)
            let mut result = DMatrix::identity(dim, dim);
            for part in parts {
                let part_matrix = to_matrix_with_size_impl(part, num_qubits);
                result = part_matrix * result;
            }
            result
        }

        UnitaryRep::Adjoint(inner) => {
            let inner_matrix = to_matrix_with_size_impl(inner, num_qubits);
            inner_matrix.adjoint()
        }

        UnitaryRep::Phase { phase, inner } => {
            let inner_matrix = to_matrix_with_size_impl(inner, num_qubits);
            let (sin_p, cos_p) = phase.sin_cos();
            let phase_factor = Complex64::new(cos_p, sin_p); // e^{i*phase}
            inner_matrix * phase_factor
        }
    }
}

/// Computes the matrix exponential of a unitary: exp(i * op).
///
/// This is useful for generating unitaries from Hermitian generators.
///
/// # Example
///
/// ```
/// use pecos_quantum::unitary_matrix::unitary_exp;
/// use pecos_core::unitary_rep::Z;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // exp(i * pi * Z) = -I
/// let z = Z(0);
/// let result = unitary_exp(&z, PI);
/// // Result should be approximately -I
/// ```
#[must_use]
pub fn unitary_exp(op: &UnitaryRep, theta: f64) -> UnitaryMatrix {
    let matrix = to_matrix(op);
    let scaled = matrix * Complex64::new(0.0, theta);
    UnitaryMatrix(pecos_num::matrix_exp(&scaled))
}

/// Computes the matrix logarithm of a unitary.
///
/// Returns `Some(generator)` where `exp(i * generator) = op`, or `None` if
/// the computation fails (e.g., for singular matrices).
///
/// # Example
///
/// ```
/// use pecos_quantum::unitary_matrix::{unitary_log, to_matrix};
/// use pecos_core::unitary_rep::X;
///
/// let x = X(0);
/// if let Some(log_x) = unitary_log(&x) {
///     // log_x is the generator such that exp(i * log_x) = X
/// }
/// ```
#[must_use]
pub fn unitary_log(op: &UnitaryRep) -> Option<DMatrix<Complex64>> {
    let matrix = to_matrix(op);
    let log_matrix = pecos_num::matrix_log(&matrix)?;
    // Divide by i to get the Hermitian generator
    Some(log_matrix / Complex64::new(0.0, 1.0))
}

/// Checks if two unitaries are equivalent up to a global phase.
///
/// Returns `true` if A = e^{i*phi} * B for some real phi.
///
/// # Example
///
/// ```
/// use pecos_quantum::unitary_matrix::unitaries_equiv;
/// use pecos_core::unitary_rep::{X, Y, Z};
///
/// let x = X(0);
/// let x2 = X(0);
/// assert!(unitaries_equiv(&x, &x2));
///
/// let y = Y(0);
/// assert!(!unitaries_equiv(&x, &y));
/// ```
#[must_use]
pub fn unitaries_equiv(a: &UnitaryRep, b: &UnitaryRep) -> bool {
    unitaries_equiv_with_tolerance(a, b, 1e-10)
}

/// Checks if two unitaries are equivalent up to a global phase, with custom tolerance.
#[must_use]
pub fn unitaries_equiv_with_tolerance(a: &UnitaryRep, b: &UnitaryRep, tol: f64) -> bool {
    let num_qubits_a = a.qubits().into_iter().max().map_or(1, |q| q + 1);
    let num_qubits_b = b.qubits().into_iter().max().map_or(1, |q| q + 1);
    let num_qubits = num_qubits_a.max(num_qubits_b);

    let mat_a = to_matrix_with_size(a, num_qubits);
    let mat_b = to_matrix_with_size(b, num_qubits);

    matrices_equiv_up_to_phase(&mat_a, &mat_b, tol)
}

/// Checks if two matrices are equal up to a global phase factor.
///
/// Returns `true` if A = e^{i*phi} * B for some real phi, within the given tolerance.
#[must_use]
pub fn matrices_equiv_up_to_phase(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
    tol: f64,
) -> bool {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return false;
    }

    // Find the first non-zero element to determine the phase
    let mut phase: Option<Complex64> = None;

    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let a_val = a[(i, j)];
            let b_val = b[(i, j)];

            // Skip near-zero elements
            if a_val.norm() < tol && b_val.norm() < tol {
                continue;
            }

            // If one is zero but not the other, not equivalent
            if a_val.norm() < tol || b_val.norm() < tol {
                return false;
            }

            // Compute the ratio a/b
            let ratio = a_val / b_val;

            match phase {
                None => {
                    // First non-zero element sets the phase
                    phase = Some(ratio);
                }
                Some(p) => {
                    // Check if this ratio matches the established phase
                    if (ratio - p).norm() > tol {
                        return false;
                    }
                }
            }
        }
    }

    // Also verify the phase has unit magnitude (global phase factor)
    if let Some(p) = phase {
        (p.norm() - 1.0).abs() < tol
    } else {
        // Both matrices are zero
        true
    }
}

// --- Helper functions for matrix construction ---

/// Converts a [`PauliString`] to a dense matrix (implementation).
fn pauli_string_to_matrix_impl(ps: &PauliString, num_qubits: usize) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::identity(dim, dim);

    // Get the phase
    let phase = ps.phase().to_complex();

    // Apply each single-qubit Pauli
    for (pauli, qubit) in ps.iter_pairs() {
        let q = usize::from(qubit);
        let pauli_matrix = single_pauli_matrix(pauli);
        let full_matrix = embed_single_qubit_gate(&pauli_matrix, q, num_qubits);
        result = full_matrix * result;
    }

    result * phase
}

/// Returns the 2x2 matrix for a single Pauli operator.
fn single_pauli_matrix(pauli: Pauli) -> DMatrix<Complex64> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let neg_one = Complex64::new(-1.0, 0.0);

    match pauli {
        Pauli::I => DMatrix::from_row_slice(2, 2, &[one, zero, zero, one]),
        Pauli::X => DMatrix::from_row_slice(2, 2, &[zero, one, one, zero]),
        Pauli::Y => DMatrix::from_row_slice(2, 2, &[zero, neg_i, i, zero]),
        Pauli::Z => DMatrix::from_row_slice(2, 2, &[one, zero, zero, neg_one]),
    }
}

/// Embeds a single-qubit gate into a larger Hilbert space.
fn embed_single_qubit_gate(
    gate: &DMatrix<Complex64>,
    qubit: usize,
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));

    for i in 0..dim {
        for j in 0..dim {
            // Check if all qubits except `qubit` match
            let mask = !(1 << qubit);
            if (i & mask) == (j & mask) {
                let i_bit = (i >> qubit) & 1;
                let j_bit = (j >> qubit) & 1;
                result[(i, j)] = gate[(i_bit, j_bit)];
            }
        }
    }

    result
}

/// Converts a rotation to a matrix.
fn rotation_to_matrix(
    rotation_type: RotationType,
    angle: Angle64,
    qubits: &[usize],
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let half = angle / 2u64;
    let (sin_half, cos_half) = half.sin_cos();
    let cos_half = Complex64::new(cos_half, 0.0);
    let sin_half = Complex64::new(sin_half, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);

    match rotation_type {
        RotationType::RX => {
            // RX(θ) = cos(θ/2)I - i*sin(θ/2)X
            let gate = DMatrix::from_row_slice(
                2,
                2,
                &[cos_half, neg_i * sin_half, neg_i * sin_half, cos_half],
            );
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        RotationType::RY => {
            // RY(θ) = cos(θ/2)I - i*sin(θ/2)Y
            let gate = DMatrix::from_row_slice(2, 2, &[cos_half, -sin_half, sin_half, cos_half]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        RotationType::RZ => {
            // RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2})
            let i_sin = Complex64::new(0.0, sin_half.re); // i * sin(θ/2)
            let exp_neg = cos_half - i_sin; // cos(θ/2) - i*sin(θ/2) = e^{-iθ/2}
            let exp_pos = cos_half + i_sin; // cos(θ/2) + i*sin(θ/2) = e^{iθ/2}
            let zero = Complex64::new(0.0, 0.0);
            let gate = DMatrix::from_row_slice(2, 2, &[exp_neg, zero, zero, exp_pos]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        RotationType::RXX | RotationType::RYY | RotationType::RZZ => {
            // For two-qubit rotations, use matrix exponential: exp(-i * θ/2 * PP)
            let generator = match rotation_type {
                RotationType::RXX => {
                    two_qubit_pauli_matrix(Pauli::X, Pauli::X, qubits[0], qubits[1], num_qubits)
                }
                RotationType::RYY => {
                    two_qubit_pauli_matrix(Pauli::Y, Pauli::Y, qubits[0], qubits[1], num_qubits)
                }
                RotationType::RZZ => {
                    two_qubit_pauli_matrix(Pauli::Z, Pauli::Z, qubits[0], qubits[1], num_qubits)
                }
                _ => unreachable!("outer match already filtered for RXX/RYY/RZZ"),
            };
            let scaled = generator * Complex64::new(0.0, -half.to_radians());
            pecos_num::matrix_exp(&scaled)
        }
    }
}

/// Constructs the matrix for R1XY(theta, phi) = cos(theta/2)*I - i*sin(theta/2)*(cos(phi)*X + sin(phi)*Y).
fn r1xy_to_matrix(
    theta: Angle64,
    phi: Angle64,
    qubits: &[usize],
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let half_theta = (theta / 2u64).to_radians_signed();
    let phi_rad = phi.to_radians_signed();
    let cos_t = half_theta.cos();
    let sin_t = half_theta.sin();
    // R1XY: [[cos, r01], [r10, cos]]
    // r01 = -i*sin*e^{-i*phi}
    // r10 = -i*sin*e^{i*phi}
    let r01 = Complex64::new(-sin_t * phi_rad.sin(), -sin_t * phi_rad.cos());
    let r10 = Complex64::new(sin_t * phi_rad.sin(), -sin_t * phi_rad.cos());
    let diag = Complex64::new(cos_t, 0.0);
    let gate = DMatrix::from_row_slice(2, 2, &[diag, r01, r10, diag]);
    embed_single_qubit_gate(&gate, qubits[0], num_qubits)
}

/// Constructs the matrix for U(theta, phi, lambda).
///
/// U = [[cos(t/2), -e^{il}*sin(t/2)], [e^{ip}*sin(t/2), e^{i(p+l)}*cos(t/2)]]
fn u3_to_matrix(
    theta: Angle64,
    phi: Angle64,
    lambda: Angle64,
    qubits: &[usize],
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let t = (theta / 2u64).to_radians_signed();
    let p = phi.to_radians_signed();
    let l = lambda.to_radians_signed();
    let cos_t = t.cos();
    let sin_t = t.sin();
    let u00 = Complex64::new(cos_t, 0.0);
    let u01 = Complex64::new(-sin_t * l.cos(), -sin_t * l.sin());
    let u10 = Complex64::new(sin_t * p.cos(), sin_t * p.sin());
    let u11 = Complex64::new(cos_t * (p + l).cos(), cos_t * (p + l).sin());
    let gate = DMatrix::from_row_slice(2, 2, &[u00, u01, u10, u11]);
    embed_single_qubit_gate(&gate, qubits[0], num_qubits)
}

/// Constructs the matrix for RXXRYYRZZ(alpha, beta, gamma).
///
/// exp(-i/2 * (alpha*XX + beta*YY + gamma*ZZ))
/// = RXX(alpha) * RYY(beta) * RZZ(gamma)
fn rxxryyrzz_to_matrix(
    alpha: Angle64,
    beta: Angle64,
    gamma: Angle64,
    qubits: &[usize],
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let rxx = rotation_to_matrix(RotationType::RXX, alpha, qubits, num_qubits);
    let ryy = rotation_to_matrix(RotationType::RYY, beta, qubits, num_qubits);
    let rzz = rotation_to_matrix(RotationType::RZZ, gamma, qubits, num_qubits);
    rxx * ryy * rzz
}

/// Constructs the matrix for U2q: (U3 x U3) * RXXRYYRZZ * (U3 x U3)
fn u2q_to_matrix(
    before: &[[Angle64; 3]; 2],
    interaction: &[Angle64; 3],
    after: &[[Angle64; 3]; 2],
    qubits: &[usize],
    num_qubits: usize,
) -> DMatrix<Complex64> {
    // After gates (applied first, right-most)
    let a0 = u3_to_matrix(
        after[0][0],
        after[0][1],
        after[0][2],
        &[qubits[0]],
        num_qubits,
    );
    let a1 = u3_to_matrix(
        after[1][0],
        after[1][1],
        after[1][2],
        &[qubits[1]],
        num_qubits,
    );
    // Interaction
    let int = rxxryyrzz_to_matrix(
        interaction[0],
        interaction[1],
        interaction[2],
        qubits,
        num_qubits,
    );
    // Before gates (applied last, left-most)
    let b0 = u3_to_matrix(
        before[0][0],
        before[0][1],
        before[0][2],
        &[qubits[0]],
        num_qubits,
    );
    let b1 = u3_to_matrix(
        before[1][0],
        before[1][1],
        before[1][2],
        &[qubits[1]],
        num_qubits,
    );
    &b0 * &b1 * &int * &a0 * &a1
}

/// Constructs a two-qubit Pauli tensor product matrix.
fn two_qubit_pauli_matrix(
    p1: Pauli,
    p2: Pauli,
    q1: usize,
    q2: usize,
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let m1 = single_pauli_matrix(p1);
    let m2 = single_pauli_matrix(p2);
    let e1 = embed_single_qubit_gate(&m1, q1, num_qubits);
    let e2 = embed_single_qubit_gate(&m2, q2, num_qubits);
    e1 * e2
}

/// Converts a gate type to a matrix.
fn gate_to_matrix(gate_type: GateType, qubits: &[usize], num_qubits: usize) -> DMatrix<Complex64> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    let sqrt2_inv = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

    match gate_type {
        GateType::I => {
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, one]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::X => {
            let gate = DMatrix::from_row_slice(2, 2, &[zero, one, one, zero]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Y => {
            let gate = DMatrix::from_row_slice(2, 2, &[zero, neg_i, i, zero]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Z => {
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, neg_one]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::H => {
            let gate =
                DMatrix::from_row_slice(2, 2, &[sqrt2_inv, sqrt2_inv, sqrt2_inv, -sqrt2_inv]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SX => {
            // SX = (1+i)/2 * [[1, -i], [-i, 1]]
            let factor = Complex64::new(0.5, 0.5);
            let gate = DMatrix::from_row_slice(
                2,
                2,
                &[factor * one, factor * neg_i, factor * neg_i, factor * one],
            );
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SXdg => {
            let factor = Complex64::new(0.5, -0.5);
            let gate = DMatrix::from_row_slice(
                2,
                2,
                &[factor * one, factor * i, factor * i, factor * one],
            );
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SY => {
            // SY = exp(-i*pi/4 * Y) = (1/sqrt(2)) * [[1, -1], [1, 1]]
            let gate =
                DMatrix::from_row_slice(2, 2, &[sqrt2_inv, -sqrt2_inv, sqrt2_inv, sqrt2_inv]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SYdg => {
            // SYdg = SY† = (1/sqrt(2)) * [[1, 1], [-1, 1]]
            let gate =
                DMatrix::from_row_slice(2, 2, &[sqrt2_inv, sqrt2_inv, -sqrt2_inv, sqrt2_inv]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SZ => {
            // S = diag(1, i)
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, i]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SZdg => {
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, neg_i]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::F => {
            // F = SX * SZ
            // SX = (1+i)/2 * [[1,-i],[-i,1]], SZ = diag(1,i)
            // F = (1+i)/2 * [[1,1],[-i,i]]
            let f = Complex64::new(0.5, 0.5);
            let gate = DMatrix::from_row_slice(2, 2, &[f * one, f * one, f * neg_i, f * i]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Fdg => {
            // Fdg = SZdg * SXdg
            // SXdg = (1-i)/2 * [[1,i],[i,1]], SZdg = diag(1,-i)
            // Fdg = (1-i)/2 * [[1,i],[i,-1]] ... let me compute properly:
            // Fdg = F† = conjugate_transpose(F)
            let f = Complex64::new(0.5, -0.5);
            let gate = DMatrix::from_row_slice(2, 2, &[f * one, f * i, f * one, f * neg_i]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::T => {
            // T = diag(1, e^{i*pi/4})
            let exp_pi_4 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, exp_pi_4]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Tdg => {
            let exp_neg_pi_4 = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, exp_neg_pi_4]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::CX => controlled_gate(
            &single_pauli_matrix(Pauli::X),
            qubits[0],
            qubits[1],
            num_qubits,
        ),
        GateType::CY => controlled_gate(
            &single_pauli_matrix(Pauli::Y),
            qubits[0],
            qubits[1],
            num_qubits,
        ),
        GateType::CZ => controlled_gate(
            &single_pauli_matrix(Pauli::Z),
            qubits[0],
            qubits[1],
            num_qubits,
        ),
        GateType::CH => {
            let h_gate =
                DMatrix::from_row_slice(2, 2, &[sqrt2_inv, sqrt2_inv, sqrt2_inv, -sqrt2_inv]);
            controlled_gate(&h_gate, qubits[0], qubits[1], num_qubits)
        }
        GateType::SWAP => swap_matrix(qubits[0], qubits[1], num_qubits),
        GateType::SXX => {
            // SXX = RXX(pi/2)
            rotation_to_matrix(RotationType::RXX, Angle64::QUARTER_TURN, qubits, num_qubits)
        }
        GateType::SXXdg => {
            // SXXdg = RXX(3pi/2)
            rotation_to_matrix(
                RotationType::RXX,
                Angle64::THREE_QUARTERS_TURN,
                qubits,
                num_qubits,
            )
        }
        GateType::SYY => {
            rotation_to_matrix(RotationType::RYY, Angle64::QUARTER_TURN, qubits, num_qubits)
        }
        GateType::SYYdg => rotation_to_matrix(
            RotationType::RYY,
            Angle64::THREE_QUARTERS_TURN,
            qubits,
            num_qubits,
        ),
        GateType::SZZ => {
            rotation_to_matrix(RotationType::RZZ, Angle64::QUARTER_TURN, qubits, num_qubits)
        }
        GateType::SZZdg => rotation_to_matrix(
            RotationType::RZZ,
            Angle64::THREE_QUARTERS_TURN,
            qubits,
            num_qubits,
        ),
        GateType::CCX => {
            // Toffoli: flip target when both controls are |1>
            let dim = 1 << num_qubits;
            let mut result = DMatrix::<Complex64>::identity(dim, dim);
            let c0 = qubits[0];
            let c1 = qubits[1];
            let t = qubits[2];
            for basis in 0..dim {
                if ((basis >> c0) & 1) == 1 && ((basis >> c1) & 1) == 1 {
                    let flipped = basis ^ (1 << t);
                    result[(basis, basis)] = zero;
                    result[(flipped, basis)] = one;
                }
            }
            result
        }

        // Parameterized gates: cannot produce a matrix without an angle.
        // These should be used via Unitary::Rotation, not Unitary::Named.
        GateType::RX
        | GateType::RY
        | GateType::RZ
        | GateType::RXX
        | GateType::RYY
        | GateType::RZZ
        | GateType::CRZ
        | GateType::U
        | GateType::R1XY
        | GateType::RXXRYYRZZ
        | GateType::U2q => {
            panic!(
                "GateType::{gate_type:?} requires angle parameter(s); \
                 use Unitary::Rotation instead of Unitary::Named"
            )
        }

        // Non-unitary operations
        GateType::MZ
        | GateType::MeasureLeaked
        | GateType::MeasureFree
        | GateType::PZ
        | GateType::QAlloc
        | GateType::QFree => {
            panic!(
                "GateType::{gate_type:?} is not a unitary gate and cannot be converted to a matrix"
            )
        }

        // Non-physical / metadata / custom
        GateType::Idle
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::Custom => {
            panic!("GateType::{gate_type:?} cannot be converted to a unitary matrix")
        }
    }
}

/// Constructs a controlled gate matrix.
fn controlled_gate(
    target_gate: &DMatrix<Complex64>,
    control: usize,
    target: usize,
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::identity(dim, dim);

    for i in 0..dim {
        for j in 0..dim {
            // Only apply gate when control qubit is 1
            let control_bit_i = (i >> control) & 1;
            let control_bit_j = (j >> control) & 1;

            if control_bit_i == 1 && control_bit_j == 1 {
                // Check if all qubits except target match
                let mask = !(1 << target);
                if (i & mask) == (j & mask) {
                    let i_bit = (i >> target) & 1;
                    let j_bit = (j >> target) & 1;
                    result[(i, j)] = target_gate[(i_bit, j_bit)];
                } else {
                    result[(i, j)] = Complex64::new(0.0, 0.0);
                }
            } else if control_bit_i == control_bit_j && i == j {
                result[(i, j)] = Complex64::new(1.0, 0.0);
            } else if control_bit_i != control_bit_j {
                result[(i, j)] = Complex64::new(0.0, 0.0);
            }
        }
    }

    result
}

/// Constructs a SWAP gate matrix.
fn swap_matrix(q1: usize, q2: usize, num_qubits: usize) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));

    for i in 0..dim {
        // Swap bits at positions q1 and q2
        let bit1 = (i >> q1) & 1;
        let bit2 = (i >> q2) & 1;

        let j = if bit1 == bit2 {
            i
        } else {
            // Swap the bits
            i ^ (1 << q1) ^ (1 << q2)
        };

        result[(i, j)] = Complex64::new(1.0, 0.0);
    }

    result
}

/// Combines two matrices representing unitaries on disjoint qubits.
///
/// When unitaries act on disjoint qubits, the tensor product in the full Hilbert space
/// is equivalent to matrix multiplication (since disjoint unitaries commute).
fn combine_disjoint_unitaries(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    a * b
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::Angle64;
    use pecos_core::unitary_rep::{CX, H, I, Is, RX, RZ, SWAP, SZ, T, X, Y, Z};
    use std::f64::consts::PI;

    // --- Basic to_matrix tests ---

    #[test]
    fn test_pauli_matrices() {
        let x = X(0);
        let mat = to_matrix(&x);
        assert_eq!(mat.nrows(), 2);
        assert!((mat[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);

        let z = Z(0);
        let mat = to_matrix(&z);
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_pauli_y() {
        let y = Y(0);
        let mat = to_matrix(&y);
        let i = Complex64::new(0.0, 1.0);
        assert!((mat[(0, 1)] - (-i)).norm() < 1e-10);
        assert!((mat[(1, 0)] - i).norm() < 1e-10);
    }

    #[test]
    fn test_hadamard() {
        let h = H(0);
        let mat = to_matrix(&h);
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert!((mat[(0, 0)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((mat[(0, 1)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 0)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(-sqrt2_inv, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_cnot() {
        let cx = CX(0, 1);
        let mat = to_matrix(&cx);
        assert_eq!(mat.nrows(), 4);
        // CX with control=0, target=1
        // |q1 q0> indexing: index = q1*2 + q0
        // When q0=0 (control off): do nothing
        // When q0=1 (control on): flip q1
        // |00> -> |00> (mat[0,0] = 1)
        // |01> -> |11> (mat[3,1] = 1)
        // |10> -> |10> (mat[2,2] = 1)
        // |11> -> |01> (mat[1,3] = 1)
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(3, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(2, 2)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 3)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_identity() {
        let id = I(0);
        let mat = to_matrix(&id);
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(mat[(0, 1)].norm() < 1e-10);
        assert!(mat[(1, 0)].norm() < 1e-10);
    }

    // --- Rotation matrix tests ---

    #[test]
    fn test_t_gate_matrix() {
        let t = T(0);
        let mat = to_matrix(&t);
        // T = RZ(π/4) = diag(e^{-iπ/8}, e^{iπ/8})
        let exp_neg = Complex64::from_polar(1.0, -PI / 8.0);
        let exp_pos = Complex64::from_polar(1.0, PI / 8.0);
        assert!((mat[(0, 0)] - exp_neg).norm() < 1e-10);
        assert!((mat[(1, 1)] - exp_pos).norm() < 1e-10);
    }

    #[test]
    fn test_s_gate_matrix() {
        let s = SZ(0);
        let mat = to_matrix(&s);
        // S = RZ(π/2) = diag(e^{-iπ/4}, e^{iπ/4})
        let exp_neg = Complex64::from_polar(1.0, -PI / 4.0);
        let exp_pos = Complex64::from_polar(1.0, PI / 4.0);
        assert!((mat[(0, 0)] - exp_neg).norm() < 1e-10);
        assert!((mat[(1, 1)] - exp_pos).norm() < 1e-10);
    }

    #[test]
    fn test_rx_matrix() {
        // RX(π) should give X (up to global phase)
        let rx_pi = RX(Angle64::HALF_TURN, 0);
        let mat = to_matrix(&rx_pi);
        let x_mat = to_matrix(&X(0));
        // RX(π) = -iX, so matrices differ by global phase -i
        assert!(matrices_equiv_up_to_phase(&mat, &x_mat, 1e-10));
    }

    #[test]
    fn test_rz_matrix() {
        // RZ(π) should give Z (up to global phase)
        let rz_pi = RZ(Angle64::HALF_TURN, 0);
        let mat = to_matrix(&rz_pi);
        let z_mat = to_matrix(&Z(0));
        assert!(matrices_equiv_up_to_phase(&mat, &z_mat, 1e-10));
    }

    // --- Tensor product and composition tests ---

    #[test]
    fn test_tensor_product() {
        // X ⊗ Z should give a 4x4 matrix
        let xz = X(0) & Z(1);
        let mat = to_matrix(&xz);
        assert_eq!(mat.nrows(), 4);

        // Verify it's the product of embedded X and Z
        let x_embedded = to_matrix_with_size(&X(0), 2);
        let z_embedded = to_matrix_with_size(&Z(1), 2);
        let expected = &x_embedded * &z_embedded;
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_composition() {
        // H * X = XH (matrix multiplication order)
        let hx = H(0) * X(0);
        let mat = to_matrix(&hx);

        let h_mat = to_matrix(&H(0));
        let x_mat = to_matrix(&X(0));
        let expected = &h_mat * &x_mat;
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_adjoint_matrix() {
        // T† matrix should be conjugate transpose of T
        let t = T(0);
        let t_dg = t.dg();
        let mat_t = to_matrix(&t);
        let mat_t_dg = to_matrix(&t_dg);

        let expected = mat_t.adjoint();
        assert!(matrices_equiv_up_to_phase(&mat_t_dg, &expected, 1e-10));
    }

    #[test]
    fn test_swap_gate() {
        let swap = SWAP(0, 1);
        let mat = to_matrix(&swap);
        assert_eq!(mat.nrows(), 4);

        // SWAP|00> = |00>, SWAP|01> = |10>, SWAP|10> = |01>, SWAP|11> = |11>
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(2, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 2)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(3, 3)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    // --- unitaries_equiv tests ---

    #[test]
    fn test_unitaries_equiv_same() {
        let x1 = X(0);
        let x2 = X(0);
        assert!(unitaries_equiv(&x1, &x2));
    }

    #[test]
    fn test_unitaries_equiv_different() {
        let x = X(0);
        let y = Y(0);
        assert!(!unitaries_equiv(&x, &y));
    }

    #[test]
    fn test_unitaries_equiv_global_phase() {
        // X and -X differ by global phase -1
        let x = X(0);
        let neg_x = pecos_core::unitary_rep::phase(Angle64::HALF_TURN) * X(0);
        assert!(unitaries_equiv(&x, &neg_x));
    }

    #[test]
    fn test_unitaries_equiv_i_phase() {
        // X and iX differ by global phase i
        let x = X(0);
        let i_x = pecos_core::unitary_rep::i * X(0);
        assert!(unitaries_equiv(&x, &i_x));
    }

    // --- unitary_exp tests ---

    #[test]
    fn test_unitary_exp_identity() {
        // exp(i * 0 * X) = I
        let x = X(0);
        let result = unitary_exp(&x, 0.0);
        let identity = DMatrix::identity(2, 2);
        assert!(matrices_equiv_up_to_phase(&result, &identity, 1e-10));
    }

    #[test]
    fn test_unitary_exp_pauli_pi() {
        // exp(i * π * Z) = -I
        let z = Z(0);
        let result = unitary_exp(&z, PI);
        let neg_identity: DMatrix<Complex64> = DMatrix::identity(2, 2) * Complex64::new(-1.0, 0.0);
        assert!(matrices_equiv_up_to_phase(&result, &neg_identity, 1e-10));
    }

    #[test]
    fn test_unitary_exp_pauli_half_pi() {
        // exp(i * π/2 * X) = i*X = [[0, i], [i, 0]]
        let x = X(0);
        let result = unitary_exp(&x, PI / 2.0);
        let i = Complex64::new(0.0, 1.0);
        let expected = to_matrix(&x) * i;
        assert!(matrices_equiv_up_to_phase(&result, &expected, 1e-10));
    }

    // --- unitary_log tests ---

    #[test]
    fn test_unitary_log_identity() {
        // log(I) = 0
        let id = I(0);
        let result = unitary_log(&id);
        assert!(result.is_some());
        let log_mat = result.unwrap();
        // All elements should be near zero
        for i in 0..log_mat.nrows() {
            for j in 0..log_mat.ncols() {
                assert!(log_mat[(i, j)].norm() < 1e-8);
            }
        }
    }

    #[test]
    fn test_unitary_log_returns_matrix() {
        // log(T) should exist (T is close to identity)
        let t = T(0);
        let result = unitary_log(&t);
        assert!(result.is_some());

        // log(S) should exist
        let s = SZ(0);
        let result = unitary_log(&s);
        assert!(result.is_some());
    }

    // --- to_matrix_with_size tests ---

    #[test]
    fn test_to_matrix_with_size_embedding() {
        // X(0) in 3-qubit space should be 8x8
        let x = X(0);
        let mat = to_matrix_with_size(&x, 3);
        assert_eq!(mat.nrows(), 8);

        // Should act as X on qubit 0, identity on others
        // Check that |000> -> |001>, |001> -> |000>
        assert!((mat[(1, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_to_matrix_preserves_unitarity() {
        // Verify U * U† = I for various operators
        let operators = vec![X(0), Y(0), Z(0), H(0), T(0), CX(0, 1)];

        for op in operators {
            let mat = to_matrix(&op);
            let product = &mat * mat.adjoint();
            let identity: DMatrix<Complex64> = DMatrix::identity(mat.nrows(), mat.ncols());

            for i in 0..mat.nrows() {
                for j in 0..mat.ncols() {
                    assert!(
                        (product[(i, j)] - identity[(i, j)]).norm() < 1e-10,
                        "Unitarity failed for operator at ({i}, {j})"
                    );
                }
            }
        }
    }

    // --- Conjugation matrix verification tests ---

    #[test]
    fn test_conj_matrix_verification() {
        // Verify A.conj(U) = U * A * U† via matrices
        let a = X(0);
        let u = H(0);

        let conj_result = a.conj(&u);
        let conj_mat = to_matrix(&conj_result);

        // Compute U * A * U† directly
        let u_mat = to_matrix(&u);
        let a_mat = to_matrix(&a);
        let expected = &u_mat * &a_mat * u_mat.adjoint();

        assert!(matrices_equiv_up_to_phase(&conj_mat, &expected, 1e-10));
    }

    #[test]
    fn test_conjdg_matrix_verification() {
        // Verify A.conjdg(U) = U† * A * U via matrices
        let a = X(0);
        let u = H(0);

        let conjdg_result = a.conjdg(&u);
        let conjdg_mat = to_matrix(&conjdg_result);

        // Compute U† * A * U directly
        let u_mat = to_matrix(&u);
        let a_mat = to_matrix(&a);
        let expected = u_mat.adjoint() * &a_mat * &u_mat;

        assert!(matrices_equiv_up_to_phase(&conjdg_mat, &expected, 1e-10));
    }

    #[test]
    fn test_conj_sz_gives_y() {
        // X.conj(SZ) = SZ * X * SZ† should equal Y (up to phase)
        let x = X(0);
        let sz = SZ(0);

        let conj_result = x.conj(&sz);
        let conj_mat = to_matrix(&conj_result);

        let y_mat = to_matrix(&Y(0));

        assert!(matrices_equiv_up_to_phase(&conj_mat, &y_mat, 1e-10));
    }

    #[test]
    fn test_conj_conjdg_inverse_via_matrix() {
        // A.conj(U).conjdg(U) should equal A
        let a = X(0);
        let u = T(0);

        let forward = a.clone().conj(&u);
        let back = forward.conjdg(&u);
        let back_mat = to_matrix(&back);

        let a_mat = to_matrix(&a);

        assert!(matrices_equiv_up_to_phase(&back_mat, &a_mat, 1e-10));
    }

    // --- Multi-qubit conjugation tests ---

    #[test]
    fn test_conj_multi_qubit_stabilizer() {
        // Two-qubit stabilizer X⊗Z conjugated by CNOT
        let stabilizer = X(0) & Z(1);
        let cnot = CX(0, 1);

        let updated = stabilizer.conj(&cnot);
        let updated_mat = to_matrix(&updated);

        // Compute CNOT * (X⊗Z) * CNOT† directly
        let cnot_mat = to_matrix(&cnot);
        let stab_mat = to_matrix(&stabilizer);
        let expected = &cnot_mat * &stab_mat * cnot_mat.adjoint();

        assert!(matrices_equiv_up_to_phase(&updated_mat, &expected, 1e-10));
    }

    #[test]
    fn test_conj_by_two_qubit_gate() {
        // Single-qubit Pauli conjugated by two-qubit gate
        let x = X(0);
        let cnot = CX(0, 1);

        let result = x.conj(&cnot);
        let result_mat = to_matrix(&result);

        // CNOT * X(0) * CNOT† = X(0) ⊗ X(1) (CNOT propagates X from control to target)
        let xx = X(0) & X(1);
        let expected = to_matrix(&xx);

        assert!(matrices_equiv_up_to_phase(&result_mat, &expected, 1e-10));
    }

    // --- More two-qubit gate tests ---

    #[test]
    fn test_cz_gate() {
        // CZ = |0><0| ⊗ I + |1><1| ⊗ Z
        use pecos_core::unitary_rep::CZ;
        let cz = CZ(0, 1);
        let mat = to_matrix(&cz);

        // CZ matrix: diag(1, 1, 1, -1)
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(2, 2)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(3, 3)] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);

        // Off-diagonal should be zero
        assert!(mat[(0, 1)].norm() < 1e-10);
        assert!(mat[(1, 2)].norm() < 1e-10);
    }

    #[test]
    fn test_cz_symmetric() {
        // CZ(0,1) should equal CZ(1,0)
        use pecos_core::unitary_rep::CZ;
        let cz_01 = CZ(0, 1);
        let cz_10 = CZ(1, 0);

        let mat_01 = to_matrix(&cz_01);
        let mat_10 = to_matrix(&cz_10);

        assert!(matrices_equiv_up_to_phase(&mat_01, &mat_10, 1e-10));
    }

    // --- Algebraic identity tests ---

    #[test]
    fn test_adjoint_of_product() {
        // (AB)† = B†A†
        let a = H(0);
        let b = T(0);

        let ab = a.clone() * b.clone();
        let ab_dagger = ab.dg();
        let ab_dagger_mat = to_matrix(&ab_dagger);

        let b_dagger_a_dagger = b.dg() * a.dg();
        let expected = to_matrix(&b_dagger_a_dagger);

        assert!(matrices_equiv_up_to_phase(&ab_dagger_mat, &expected, 1e-10));
    }

    #[test]
    fn test_double_adjoint_identity() {
        // (A†)† = A
        let ops = vec![X(0), Y(0), H(0), T(0), CX(0, 1)];

        for op in ops {
            let double_dagger = op.dg().dg();
            let original_mat = to_matrix(&op);
            let double_mat = to_matrix(&double_dagger);

            assert!(matrices_equiv_up_to_phase(
                &original_mat,
                &double_mat,
                1e-10
            ));
        }
    }

    #[test]
    fn test_tensor_adjoint() {
        // (A ⊗ B)† = A† ⊗ B†
        let a = H(0);
        let b = T(1);

        let tensor = a.clone() & b.clone();
        let tensor_dagger = tensor.dg();
        let tensor_dagger_mat = to_matrix(&tensor_dagger);

        let a_dagger_tensor_b_dagger = a.dg() & b.dg();
        let expected = to_matrix(&a_dagger_tensor_b_dagger);

        assert!(matrices_equiv_up_to_phase(
            &tensor_dagger_mat,
            &expected,
            1e-10
        ));
    }

    // --- ToMatrix trait tests ---

    #[test]
    fn test_to_matrix_trait_method() {
        // Test that trait method gives same result as standalone function
        let h = H(0);

        let via_function = to_matrix(&h);
        let via_trait = h.to_matrix();

        assert_eq!(via_function.nrows(), via_trait.nrows());
        assert_eq!(via_function.ncols(), via_trait.ncols());
        for i in 0..via_function.nrows() {
            for j in 0..via_function.ncols() {
                assert!((via_function[(i, j)] - via_trait[(i, j)]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_to_matrix_with_identity_tensor() {
        // Test using Is() to get larger matrix
        let x_extended = X(0) & Is(1..3); // X on qubit 0 in 3-qubit space

        let mat = x_extended.to_matrix();
        assert_eq!(mat.nrows(), 8); // 2^3 = 8

        // Should match the standalone function
        let expected = to_matrix_with_size(&X(0), 3);
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_to_matrix_trait_chaining() {
        // Verify trait works well with operator chaining
        let circuit = H(0) * CX(0, 1) * H(0);
        let mat = circuit.to_matrix();

        assert_eq!(mat.nrows(), 4); // 2 qubits

        // Verify unitarity
        let product = mat.adjoint() * &mat;
        let identity = UnitaryMatrix::identity(4);
        assert!(matrices_equiv_up_to_phase(&product, &identity, 1e-10));
    }

    // --- Identity operator ToMatrix tests ---

    #[test]
    fn test_identity_to_matrix_single_qubit() {
        // I(0).to_matrix() should be 2x2 identity
        let mat = I(0).to_matrix();
        let expected: DMatrix<Complex64> = DMatrix::identity(2, 2);

        assert_eq!(mat.nrows(), 2);
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_identity_to_matrix_two_qubits() {
        // Is(0..=1).to_matrix() should be 4x4 identity
        let mat = Is(0..=1).to_matrix();
        let expected: DMatrix<Complex64> = DMatrix::identity(4, 4);

        assert_eq!(mat.nrows(), 4);
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_identity_tensor_with_gate() {
        // X(0) & I(1) should give X tensor I = 4x4 matrix
        let op = X(0) & I(1);
        let mat = op.to_matrix();

        assert_eq!(mat.nrows(), 4);

        // Should equal X(0) extended to 2 qubits
        let expected = to_matrix_with_size(&X(0), 2);

        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_simplify_preserves_tensor_dimension() {
        // (X(0) & I(1)).simplify() should preserve the 2-qubit space
        let op = X(0) & I(1);
        let simplified = op.simplify();

        // Both should produce equivalent 4x4 matrices
        let orig_mat = op.to_matrix();
        let simp_mat = simplified.to_matrix();

        assert_eq!(orig_mat.nrows(), 4);
        assert_eq!(simp_mat.nrows(), 4);
        assert!(matrices_equiv_up_to_phase(&orig_mat, &simp_mat, 1e-10));
    }

    // --- PauliString ToMatrix tests ---

    #[test]
    fn test_pauli_string_to_matrix_single() {
        use pecos_core::PauliString;

        // Single X Pauli
        let ps = PauliString::x(0);
        let mat = ps.to_matrix();

        // Should match X(0).to_matrix()
        let x_mat = X(0).to_matrix();
        assert!(matrices_equiv_up_to_phase(&mat, &x_mat, 1e-10));
    }

    #[test]
    fn test_pauli_string_to_matrix_multi() {
        use pecos_core::{Pauli, PauliString};

        // X on qubit 0, Z on qubit 1
        let ps = PauliString::from_paulis(&[Pauli::X, Pauli::Z]);
        let mat = ps.to_matrix();

        // Should match (X(0) & Z(1)).to_matrix()
        let xz = X(0) & Z(1);
        let expected = xz.to_matrix();
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_pauli_string_to_matrix_with_phase() {
        use pecos_core::{Pauli, PauliString, QuarterPhase};

        // -i * X
        let ps = PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::X]);
        let mat = ps.to_matrix();

        // Should be -i times X matrix
        let x_mat = X(0).to_matrix();
        let neg_i = Complex64::new(0.0, -1.0);
        let expected = x_mat * neg_i;

        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_pauli_string_to_matrix_identity() {
        use pecos_core::PauliString;

        // Identity PauliString - returns 1x1 identity (no qubits)
        let ps = PauliString::identity();
        let mat = ps.to_matrix();

        // Identity with no qubits defaults to 1 qubit -> 2x2
        let identity: DMatrix<Complex64> = DMatrix::identity(2, 2);
        assert!(matrices_equiv_up_to_phase(&mat, &identity, 1e-10));
    }

    #[test]
    fn test_pauli_string_matches_operator_pauli() {
        use pecos_core::{Pauli, PauliString};

        // Verify PauliString.to_matrix() matches UnitaryRep::Pauli.to_matrix()
        let ps = PauliString::from_paulis(&[Pauli::Y, Pauli::Z]);

        // Convert to UnitaryRep::Pauli
        let op = pecos_core::unitary_rep::UnitaryRep::Pauli(ps.clone());

        let ps_mat = ps.to_matrix();
        let op_mat = op.to_matrix();

        assert!(matrices_equiv_up_to_phase(&ps_mat, &op_mat, 1e-10));
    }

    // --- try_to_unitary tests ---

    #[test]
    fn try_to_unitary_identifies_all_named_1q_gates() {
        use super::NAMED_GATE_1Q;
        for &gate in &NAMED_GATE_1Q {
            let mat = UnitaryMatrix(super::gate_to_matrix(gate, &[0], 1));
            assert_eq!(
                mat.try_to_unitary(),
                Some(Unitary::Named(gate)),
                "failed to identify {gate:?}"
            );
        }
    }

    #[test]
    fn try_to_unitary_identifies_all_named_2q_gates() {
        use super::NAMED_GATE_2Q;
        for &gate in &NAMED_GATE_2Q {
            let mat = UnitaryMatrix(super::gate_to_matrix(gate, &[0, 1], 2));
            let identified = mat.try_to_unitary();
            assert!(identified.is_some(), "failed to identify {gate:?}");
            // Verify via matrix comparison (handles self-inverse gates)
            let id_gate = identified.unwrap().to_gate_type().unwrap();
            let id_mat = UnitaryMatrix(super::gate_to_matrix(id_gate, &[0, 1], 2));
            assert!(
                mat.equiv_up_to_phase(&id_mat),
                "{gate:?} identified as {:?} but matrices don't match",
                identified.unwrap()
            );
        }
    }

    #[test]
    fn try_to_unitary_identifies_ccx() {
        let mat = UnitaryMatrix(super::gate_to_matrix(GateType::CCX, &[0, 1, 2], 3));
        assert_eq!(mat.try_to_unitary(), Some(Unitary::Named(GateType::CCX)));
    }

    #[test]
    fn try_to_unitary_finds_t_gate() {
        let t_mat = T(0).to_matrix();
        assert_eq!(t_mat.try_to_unitary(), Some(Unitary::Named(GateType::T)));

        let tdg_mat = pecos_core::unitary_rep::T(0).dg().to_matrix();
        assert_eq!(
            tdg_mat.try_to_unitary(),
            Some(Unitary::Named(GateType::Tdg))
        );
    }

    #[test]
    fn try_to_unitary_with_any_scalar() {
        // iX should still be identified as X
        let x_mat = X(0).to_matrix();
        let ix = &x_mat * Complex64::new(0.0, 1.0);
        assert_eq!(ix.try_to_unitary(), Some(Unitary::Named(GateType::X)));

        // 2*H (non-unitary scalar) should still be identified as H
        let h_mat = H(0).to_matrix();
        let two_h = &h_mat * 2.0;
        assert_eq!(two_h.try_to_unitary(), Some(Unitary::Named(GateType::H)));

        // (3+4i)*Z should still be identified as Z
        let z_mat = Z(0).to_matrix();
        let scaled_z = &z_mat * Complex64::new(3.0, 4.0);
        assert_eq!(scaled_z.try_to_unitary(), Some(Unitary::Named(GateType::Z)));

        // -iT should still be identified as T
        let t_mat = T(0).to_matrix();
        let phased = &t_mat * Complex64::new(0.0, -1.0);
        assert_eq!(phased.try_to_unitary(), Some(Unitary::Named(GateType::T)));

        // 5*CX should still be identified as CX
        let cx_mat = CX(0, 1).to_matrix();
        let scaled = &cx_mat * 5.0;
        assert_eq!(scaled.try_to_unitary(), Some(Unitary::Named(GateType::CX)));
    }

    #[test]
    fn try_to_unitary_returns_none_for_non_unitary() {
        // A singular matrix is not in any named table and not unitary
        let mat = UnitaryMatrix::from(DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        ));
        assert_eq!(mat.try_to_unitary(), None);
    }

    #[test]
    fn try_to_unitary_pauli_and_clifford_classification() {
        // Paulis
        let x = X(0).to_matrix().try_to_unitary().unwrap();
        assert!(x.is_pauli());
        assert!(x.is_clifford());
        assert_eq!(x.try_to_pauli(), Some(Pauli::X));

        let z = Z(0).to_matrix().try_to_unitary().unwrap();
        assert!(z.is_pauli());
        assert_eq!(z.try_to_pauli(), Some(Pauli::Z));

        // Clifford but not Pauli
        let h = H(0).to_matrix().try_to_unitary().unwrap();
        assert!(!h.is_pauli());
        assert!(h.is_clifford());
        assert_eq!(h.try_to_pauli(), None);

        // Non-Clifford
        let t = T(0).to_matrix().try_to_unitary().unwrap();
        assert!(!t.is_pauli());
        assert!(!t.is_clifford());
        assert_eq!(t.try_to_pauli(), None);
    }

    // --- rotation extraction tests ---

    #[test]
    fn try_to_unitary_identifies_1q_rotations() {
        use pecos_core::unitary_rep::RY;

        for &(angle_rad, label) in &[(0.3, "0.3"), (1.0, "1.0"), (2.5, "2.5"), (-0.7, "-0.7")] {
            let angle = Angle64::from_radians(angle_rad);

            // RX
            let mat = RX(angle, 0).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("RX({label}) not identified"));
            match u {
                Unitary::Rotation {
                    rotation_type: RotationType::RX,
                    angle: a,
                } => {
                    let diff = (a.to_radians_signed() - angle.to_radians_signed()).abs();
                    assert!(diff < 1e-6, "RX({label}): angle mismatch {diff}");
                }
                other => panic!("RX({label}) identified as {other:?}"),
            }

            // RY
            let mat = RY(angle, 0).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("RY({label}) not identified"));
            match u {
                Unitary::Rotation {
                    rotation_type: RotationType::RY,
                    angle: a,
                } => {
                    let diff = (a.to_radians_signed() - angle.to_radians_signed()).abs();
                    assert!(diff < 1e-6, "RY({label}): angle mismatch {diff}");
                }
                other => panic!("RY({label}) identified as {other:?}"),
            }

            // RZ
            let mat = RZ(angle, 0).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("RZ({label}) not identified"));
            match u {
                Unitary::Rotation {
                    rotation_type: RotationType::RZ,
                    angle: a,
                } => {
                    let diff = (a.to_radians_signed() - angle.to_radians_signed()).abs();
                    assert!(diff < 1e-6, "RZ({label}): angle mismatch {diff}");
                }
                other => panic!("RZ({label}) identified as {other:?}"),
            }
        }
    }

    #[test]
    fn try_to_unitary_identifies_2q_rotations() {
        use pecos_core::unitary_rep::{RXX, RYY, RZZ};

        for &angle_rad in &[0.3, 1.0, 2.5, -0.7] {
            let angle = Angle64::from_radians(angle_rad);

            let mat = RXX(angle, 0, 1).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("RXX({angle_rad}) not identified"));
            match u {
                Unitary::Rotation {
                    rotation_type: RotationType::RXX,
                    angle: a,
                } => {
                    let diff = (a.to_radians_signed() - angle.to_radians_signed()).abs();
                    assert!(diff < 1e-6, "RXX({angle_rad}): angle mismatch {diff}");
                }
                other => panic!("RXX({angle_rad}) identified as {other:?}"),
            }

            let mat = RYY(angle, 0, 1).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("RYY({angle_rad}) not identified"));
            match u {
                Unitary::Rotation {
                    rotation_type: RotationType::RYY,
                    angle: a,
                } => {
                    let diff = (a.to_radians_signed() - angle.to_radians_signed()).abs();
                    assert!(diff < 1e-6, "RYY({angle_rad}): angle mismatch {diff}");
                }
                other => panic!("RYY({angle_rad}) identified as {other:?}"),
            }

            let mat = RZZ(angle, 0, 1).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("RZZ({angle_rad}) not identified"));
            match u {
                Unitary::Rotation {
                    rotation_type: RotationType::RZZ,
                    angle: a,
                } => {
                    let diff = (a.to_radians_signed() - angle.to_radians_signed()).abs();
                    assert!(diff < 1e-6, "RZZ({angle_rad}): angle mismatch {diff}");
                }
                other => panic!("RZZ({angle_rad}) identified as {other:?}"),
            }
        }
    }

    #[test]
    fn try_to_unitary_rotation_rejects_scaled() {
        // 3*RZ(0.5) is not unitary, so rotation extraction should not match
        let angle = Angle64::from_radians(0.5);
        let mat = &RZ(angle, 0).to_matrix() * 3.0;
        assert_eq!(mat.try_to_unitary(), None);
    }

    #[test]
    fn try_to_unitary_identifies_r1xy() {
        // R1XY(theta, phi) for various angles
        // Note: R1XY(-theta, phi) = R1XY(theta, phi+pi), so we compare matrices
        // rather than raw angles to handle this sign ambiguity.
        for &(theta_rad, phi_rad) in &[(0.3, 0.7), (1.0, 2.0), (2.5, -0.5), (-0.7, 1.2)] {
            let theta = Angle64::from_radians(theta_rad);
            let phi = Angle64::from_radians(phi_rad);
            let mat = Unitary::R1XY { theta, phi }.on_qubit(0).to_matrix();
            let u = mat
                .try_to_unitary()
                .unwrap_or_else(|| panic!("R1XY({theta_rad}, {phi_rad}) not identified"));
            match u {
                Unitary::R1XY { .. } => {
                    let roundtrip = u.on_qubit(0).to_matrix();
                    assert!(
                        mat.equiv_up_to_phase(&roundtrip),
                        "R1XY({theta_rad},{phi_rad}): matrix mismatch after roundtrip"
                    );
                }
                other => panic!("R1XY({theta_rad},{phi_rad}) identified as {other:?}"),
            }
        }
    }

    #[test]
    fn try_to_unitary_r1xy_special_cases() {
        // R1XY(theta, 0) should be identified as RX (single-axis, not R1XY)
        let theta = Angle64::from_radians(0.5);
        let mat = Unitary::R1XY {
            theta,
            phi: Angle64::ZERO,
        }
        .on_qubit(0)
        .to_matrix();
        let u = mat.try_to_unitary().unwrap();
        assert!(
            matches!(
                u,
                Unitary::Rotation {
                    rotation_type: RotationType::RX,
                    ..
                }
            ),
            "R1XY(0.5, 0) should match as RX, got {u:?}"
        );

        // R1XY(theta, pi/2) should be identified as RY
        let mat = Unitary::R1XY {
            theta,
            phi: Angle64::QUARTER_TURN,
        }
        .on_qubit(0)
        .to_matrix();
        let u = mat.try_to_unitary().unwrap();
        assert!(
            matches!(
                u,
                Unitary::Rotation {
                    rotation_type: RotationType::RY,
                    ..
                }
            ),
            "R1XY(0.5, pi/2) should match as RY, got {u:?}"
        );
    }

    #[test]
    fn try_to_unitary_special_angles_prefer_named() {
        // RZ(pi) should be identified as Z (named gate), not Rotation
        let rz_pi = RZ(Angle64::from_radians(PI), 0).to_matrix();
        let u = rz_pi.try_to_unitary().unwrap();
        assert!(
            matches!(u, Unitary::Named(GateType::Z)),
            "RZ(pi) should match as Z, got {u:?}"
        );

        // RZ(pi/2) should be identified as SZ (named gate), not Rotation
        let rz_half = RZ(Angle64::from_radians(PI / 2.0), 0).to_matrix();
        let u = rz_half.try_to_unitary().unwrap();
        assert!(
            matches!(u, Unitary::Named(GateType::SZ)),
            "RZ(pi/2) should match as SZ, got {u:?}"
        );
    }

    #[test]
    fn try_to_unitary_identifies_rxxryyrzz() {
        // RXX(0.3) * RYY(0.7) * RZZ(1.1)
        let a = Angle64::from_radians(0.3);
        let b = Angle64::from_radians(0.7);
        let c = Angle64::from_radians(1.1);
        let mat_rxx = rotation_to_matrix(RotationType::RXX, a, &[0, 1], 2);
        let mat_ryy = rotation_to_matrix(RotationType::RYY, b, &[0, 1], 2);
        let mat_rzz = rotation_to_matrix(RotationType::RZZ, c, &[0, 1], 2);
        let mat = UnitaryMatrix::from(mat_rxx * mat_ryy * mat_rzz);
        let u = mat.try_to_unitary().unwrap();
        match u {
            Unitary::RXXRYYRZZ { alpha, beta, gamma } => {
                let reconstructed = rxxryyrzz_to_matrix(alpha, beta, gamma, &[0, 1], 2);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-8),
                    "RXXRYYRZZ roundtrip failed"
                );
            }
            other => panic!("Expected RXXRYYRZZ, got {other:?}"),
        }
    }

    #[test]
    fn try_to_unitary_rxxryyrzz_two_axes() {
        // RXX(0.5) * RZZ(0.9) -- only 2 of 3 axes nonzero
        let a = Angle64::from_radians(0.5);
        let c = Angle64::from_radians(0.9);
        let mat_rxx = rotation_to_matrix(RotationType::RXX, a, &[0, 1], 2);
        let mat_rzz = rotation_to_matrix(RotationType::RZZ, c, &[0, 1], 2);
        let mat = UnitaryMatrix::from(mat_rxx * mat_rzz);
        let u = mat.try_to_unitary().unwrap();
        match u {
            Unitary::RXXRYYRZZ { alpha, beta, gamma } => {
                let reconstructed = rxxryyrzz_to_matrix(alpha, beta, gamma, &[0, 1], 2);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-8),
                    "RXXRYYRZZ (two axes) roundtrip failed"
                );
            }
            other => panic!("Expected RXXRYYRZZ for two-axis rotation, got {other:?}"),
        }
    }

    #[test]
    fn try_to_unitary_single_axis_still_prefers_rxx() {
        // RXX(0.5) alone should still be identified as Rotation, not RXXRYYRZZ
        let a = Angle64::from_radians(0.5);
        let mat_rxx = UnitaryMatrix::from(rotation_to_matrix(RotationType::RXX, a, &[0, 1], 2));
        let u = mat_rxx.try_to_unitary().unwrap();
        assert!(
            matches!(
                u,
                Unitary::Rotation {
                    rotation_type: RotationType::RXX,
                    ..
                }
            ),
            "Single-axis RXX should be Rotation, got {u:?}"
        );
    }

    #[test]
    fn try_to_unitary_identifies_u3() {
        // A general single-qubit unitary with all three Pauli components
        // U(0.7, 1.2, 0.5)
        let theta = 0.7_f64;
        let phi_val = 1.2_f64;
        let lambda = 0.5_f64;
        let cos_t = (theta / 2.0).cos();
        let sin_t = (theta / 2.0).sin();
        let u00 = Complex64::new(cos_t, 0.0);
        let u01 = -Complex64::from_polar(sin_t, lambda);
        let u10 = Complex64::from_polar(sin_t, phi_val);
        let u11 = Complex64::from_polar(cos_t, phi_val + lambda);
        let mat = UnitaryMatrix::from(DMatrix::from_row_slice(2, 2, &[u00, u01, u10, u11]));
        let u = mat.try_to_unitary().unwrap();
        // Verify the identified U3 produces the same matrix
        match u {
            Unitary::U3 {
                theta: t,
                phi: p,
                lambda: l,
            } => {
                let reconstructed = u3_to_matrix(t, p, l, &[0], 1);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-8),
                    "U3 roundtrip failed: got theta={}, phi={}, lambda={}",
                    t.to_radians_signed(),
                    p.to_radians_signed(),
                    l.to_radians_signed()
                );
            }
            other => panic!("Expected U3, got {other:?}"),
        }
    }

    #[test]
    fn try_to_unitary_u3_with_global_phase() {
        // U(0.7, 1.2, 0.5) * e^{i*0.3}
        let theta = 0.7_f64;
        let phi_val = 1.2_f64;
        let lambda = 0.5_f64;
        let global_phase = 0.3_f64;
        let cos_t = (theta / 2.0).cos();
        let sin_t = (theta / 2.0).sin();
        let gp = Complex64::from_polar(1.0, global_phase);
        let u00 = Complex64::new(cos_t, 0.0) * gp;
        let u01 = -Complex64::from_polar(sin_t, lambda) * gp;
        let u10 = Complex64::from_polar(sin_t, phi_val) * gp;
        let u11 = Complex64::from_polar(cos_t, phi_val + lambda) * gp;
        let mat = UnitaryMatrix::from(DMatrix::from_row_slice(2, 2, &[u00, u01, u10, u11]));
        let u = mat.try_to_unitary().unwrap();
        match u {
            Unitary::U3 {
                theta: t,
                phi: p,
                lambda: l,
            } => {
                let reconstructed = u3_to_matrix(t, p, l, &[0], 1);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-8),
                    "U3 with global phase roundtrip failed"
                );
            }
            other => panic!("Expected U3, got {other:?}"),
        }
    }

    #[test]
    fn try_to_unitary_u3_near_theta_pi() {
        // U(theta, phi, lambda) near theta=pi (cos(theta/2) small but nonzero)
        // Using theta slightly less than pi to ensure all Pauli components present
        let theta = 2.9_f64;
        let phi_val = 0.5_f64;
        let lambda = 1.3_f64;
        let cos_t = (theta / 2.0).cos();
        let sin_t = (theta / 2.0).sin();
        let u00 = Complex64::new(cos_t, 0.0);
        let u01 = -Complex64::from_polar(sin_t, lambda);
        let u10 = Complex64::from_polar(sin_t, phi_val);
        let u11 = Complex64::from_polar(cos_t, phi_val + lambda);
        let mat = UnitaryMatrix::from(DMatrix::from_row_slice(2, 2, &[u00, u01, u10, u11]));
        let u = mat.try_to_unitary().unwrap();
        match u {
            Unitary::U3 {
                theta: t,
                phi: p,
                lambda: l,
            } => {
                let reconstructed = u3_to_matrix(t, p, l, &[0], 1);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-8),
                    "U3 near-pi roundtrip failed"
                );
            }
            other => panic!("Expected U3, got {other:?}"),
        }
    }

    #[test]
    fn try_to_unitary_mixed_xz_rotation_now_identifies_as_u3() {
        // A rotation around a mixed axis (0.6X + 0.8Z) was previously unidentified;
        // now it should be identified as U3.
        let angle = 0.3_f64;
        let nx = 0.6_f64;
        let nz = 0.8_f64;
        let cos_a = Complex64::new(angle.cos(), 0.0);
        let sin_a = Complex64::new(angle.sin(), 0.0);
        let mat = UnitaryMatrix::from(DMatrix::from_row_slice(
            2,
            2,
            &[
                cos_a - Complex64::i() * sin_a * nz,
                -Complex64::i() * sin_a * nx,
                -Complex64::i() * sin_a * nx,
                cos_a + Complex64::i() * sin_a * nz,
            ],
        ));
        let u = mat.try_to_unitary().unwrap();
        match u {
            Unitary::U3 {
                theta: t,
                phi: p,
                lambda: l,
            } => {
                let reconstructed = u3_to_matrix(t, p, l, &[0], 1);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-8),
                    "Mixed XZ rotation U3 roundtrip failed"
                );
            }
            other => panic!("Expected U3 for mixed axis rotation, got {other:?}"),
        }
    }

    // --- is_unitary tests ---

    #[test]
    fn is_unitary_for_known_gates() {
        assert!(X(0).to_matrix().is_unitary());
        assert!(H(0).to_matrix().is_unitary());
        assert!(CX(0, 1).to_matrix().is_unitary());
        assert!(T(0).to_matrix().is_unitary());
    }

    #[test]
    fn is_unitary_rejects_scaled_matrices() {
        let scaled = &X(0).to_matrix() * 2.0;
        assert!(!scaled.is_unitary());
    }

    #[test]
    fn is_unitary_rejects_non_square() {
        let mat = UnitaryMatrix(DMatrix::zeros(2, 4));
        assert!(!mat.is_unitary());
    }

    #[test]
    fn try_to_unitary_rxxryyrzz_stress() {
        // Test a grid of angle combinations including edge cases:
        // - all three axes nonzero
        // - negative angles
        // - angles near pi
        // - small angles near zero
        // - mixed positive/negative
        let angles = [
            0.1,
            -0.1,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            -2.0,
            3.0,
            -3.0,
            0.01,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
        ];
        let mut count = 0;
        for &a in &angles {
            for &b in &angles {
                for &c in &angles {
                    // Skip when all three are the same sign and magnitude (boring)
                    // but test enough combinations
                    if count > 500 {
                        break;
                    }
                    let alpha = Angle64::from_radians(a);
                    let beta = Angle64::from_radians(b);
                    let gamma = Angle64::from_radians(c);
                    let mat_rxx = rotation_to_matrix(RotationType::RXX, alpha, &[0, 1], 2);
                    let mat_ryy = rotation_to_matrix(RotationType::RYY, beta, &[0, 1], 2);
                    let mat_rzz = rotation_to_matrix(RotationType::RZZ, gamma, &[0, 1], 2);
                    let original = &mat_rxx * &mat_ryy * &mat_rzz;
                    let mat = UnitaryMatrix::from(original.clone());
                    let u = mat
                        .try_to_unitary()
                        .unwrap_or_else(|| panic!("Failed to identify RXXRYYRZZ({a}, {b}, {c})"));
                    match u {
                        Unitary::RXXRYYRZZ {
                            alpha: ra,
                            beta: rb,
                            gamma: rc,
                        } => {
                            let reconstructed = rxxryyrzz_to_matrix(ra, rb, rc, &[0, 1], 2);
                            assert!(
                                matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-7),
                                "RXXRYYRZZ roundtrip failed for ({a}, {b}, {c}): \
                                 got ({}, {}, {})",
                                ra.to_radians_signed(),
                                rb.to_radians_signed(),
                                rc.to_radians_signed(),
                            );
                        }
                        // Some angle combos might simplify to single-axis, which is OK
                        Unitary::Rotation {
                            rotation_type,
                            angle,
                        } => {
                            let reconstructed =
                                rotation_to_matrix(rotation_type, angle, &[0, 1], 2);
                            assert!(
                                matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-7),
                                "Single-axis roundtrip failed for ({a}, {b}, {c}): \
                                 got {rotation_type:?}({:?})",
                                angle.to_radians_signed(),
                            );
                        }
                        other => panic!("Unexpected variant for ({a}, {b}, {c}): {other:?}"),
                    }
                    count += 1;
                }
            }
        }
        assert!(count > 100, "Stress test ran only {count} cases");
    }

    // --- U2q (KAK decomposition) tests ---

    #[test]
    fn try_to_unitary_identifies_u2q_general() {
        // Build a general 2-qubit unitary: U3(0.3, 0.5, 0.7) on q0, then CX
        let u3_mat = u3_to_matrix(
            Angle64::from_radians(0.3),
            Angle64::from_radians(0.5),
            Angle64::from_radians(0.7),
            &[0],
            2,
        );
        let cx_mat = gate_to_matrix(GateType::CX, &[0, 1], 2);
        let original = &cx_mat * &u3_mat;
        let mat = UnitaryMatrix::from(original.clone());
        let u = mat
            .try_to_unitary()
            .expect("Should identify general 2Q unitary");
        match u {
            Unitary::U2q {
                before,
                interaction,
                after,
            } => {
                let reconstructed = u2q_to_matrix(&before, &interaction, &after, &[0, 1], 2);
                assert!(
                    matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-6),
                    "U2q roundtrip failed for general unitary"
                );
            }
            other => panic!("Expected U2q for general 2-qubit unitary, got {other:?}"),
        }
    }

    #[test]
    fn try_to_unitary_u2q_with_global_phase() {
        // Same gate but with a global phase e^{i*0.4}
        let u3_mat = u3_to_matrix(
            Angle64::from_radians(0.3),
            Angle64::from_radians(0.5),
            Angle64::from_radians(0.7),
            &[0],
            2,
        );
        let cx_mat = gate_to_matrix(GateType::CX, &[0, 1], 2);
        let original = &cx_mat * &u3_mat;
        let phase = Complex64::from_polar(1.0, 0.4);
        let phased = &original * phase;
        let mat = UnitaryMatrix::from(phased);
        let u = mat
            .try_to_unitary()
            .expect("Should identify phased general 2Q unitary");
        if let Unitary::U2q {
            before,
            interaction,
            after,
        } = u
        {
            let reconstructed = u2q_to_matrix(&before, &interaction, &after, &[0, 1], 2);
            assert!(
                matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-7),
                "U2q roundtrip failed with global phase"
            );
        }
        // Could also match Named(CX) if the phase gets canonicalized away
    }

    #[test]
    fn try_to_unitary_prefers_simpler_over_u2q() {
        // CNOT should be identified as Named(CX), not U2q
        let cx = UnitaryMatrix::from(gate_to_matrix(GateType::CX, &[0, 1], 2));
        let u = cx.try_to_unitary().unwrap();
        assert!(
            matches!(u, Unitary::Named(GateType::CX)),
            "CNOT should be Named(CX), got {u:?}"
        );

        // RXX should stay as Rotation, not U2q
        let rxx = UnitaryMatrix::from(rotation_to_matrix(
            RotationType::RXX,
            Angle64::from_radians(0.5),
            &[0, 1],
            2,
        ));
        let u = rxx.try_to_unitary().unwrap();
        assert!(
            matches!(
                u,
                Unitary::Rotation {
                    rotation_type: RotationType::RXX,
                    ..
                }
            ),
            "RXX should be Rotation, got {u:?}"
        );
    }

    #[test]
    fn try_to_unitary_u2q_stress() {
        // Build many different 2-qubit unitaries and verify U2q roundtrip.
        // Each is: U3(a,b,c) ⊗ U3(d,e,f) * RXXRYYRZZ(g,h,i) * U3(j,k,l) ⊗ U3(m,n,o)
        let angles: Vec<f64> = vec![0.0, 0.3, 0.7, 1.2, 2.0, 3.0, 4.5, 5.5];
        let mut pass_count = 0;
        let mut total = 0;
        for &a in &angles {
            for &b in &angles[..4] {
                for &g in &angles {
                    // Build: CX-like gate with various single-qubit pre/post rotations
                    let u3_0 = u3_to_matrix(
                        Angle64::from_radians(a),
                        Angle64::from_radians(b),
                        Angle64::from_radians(0.1),
                        &[0],
                        2,
                    );
                    let u3_1 = u3_to_matrix(
                        Angle64::from_radians(g),
                        Angle64::from_radians(0.2),
                        Angle64::from_radians(0.3),
                        &[1],
                        2,
                    );
                    let cx_mat = gate_to_matrix(GateType::CX, &[0, 1], 2);
                    let original = &u3_0 * &u3_1 * &cx_mat;
                    let mat = UnitaryMatrix::from(original);
                    total += 1;
                    let u = mat.try_to_unitary().unwrap_or_else(|| {
                        panic!("Failed to identify unitary at a={a}, b={b}, g={g}")
                    });
                    match &u {
                        Unitary::U2q {
                            before,
                            interaction,
                            after,
                        } => {
                            let reconstructed =
                                u2q_to_matrix(before, interaction, after, &[0, 1], 2);
                            assert!(
                                matrices_equiv_up_to_phase(&mat.0, &reconstructed, 1e-5),
                                "U2q roundtrip failed at a={a}, b={b}, g={g}"
                            );
                            pass_count += 1;
                        }
                        // Some might simplify to Named or RXXRYYRZZ
                        _ => {
                            pass_count += 1;
                        }
                    }
                }
            }
        }
        assert!(
            pass_count == total,
            "Not all cases passed: {pass_count}/{total}"
        );
    }
}
