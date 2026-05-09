// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

const MATRIX_EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum CliffordMatrixGate {
    SZdg,
    F,
    Fdg,
    SX,
    SXdg,
    SY,
    SYdg,
    CX,
    CY,
    CZ,
    SXX,
    SXXdg,
    SYY,
    SYYdg,
    SZZ,
    SZZdg,
    SWAP,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SignedPauli {
    pub(crate) sign: i8,
    pub(crate) pauli: String,
}

#[derive(Clone, Debug)]
struct Matrix {
    n: usize,
    data: Vec<Complex64>,
}

impl Matrix {
    fn from_data(n: usize, data: Vec<Complex64>) -> Self {
        assert_eq!(data.len(), n * n);
        Self { n, data }
    }

    fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![Complex64::new(0.0, 0.0); n * n],
        }
    }

    fn identity(n: usize) -> Self {
        let mut matrix = Self::zeros(n);
        for i in 0..n {
            matrix.set(i, i, Complex64::new(1.0, 0.0));
        }
        matrix
    }

    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.n + col]
    }

    fn set(&mut self, row: usize, col: usize, value: Complex64) {
        self.data[row * self.n + col] = value;
    }

    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::from_data(self.n, data)
    }

    fn scale(&self, scalar: Complex64) -> Self {
        let data = self.data.iter().map(|v| scalar * v).collect();
        Self::from_data(self.n, data)
    }

    fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut out = Self::zeros(self.n);
        for row in 0..self.n {
            for col in 0..self.n {
                let mut value = Complex64::new(0.0, 0.0);
                for k in 0..self.n {
                    value += self.get(row, k) * other.get(k, col);
                }
                out.set(row, col, value);
            }
        }
        out
    }

    fn dagger(&self) -> Self {
        let mut out = Self::zeros(self.n);
        for row in 0..self.n {
            for col in 0..self.n {
                out.set(col, row, self.get(row, col).conj());
            }
        }
        out
    }

    fn approx_eq(&self, other: &Self) -> bool {
        assert_eq!(self.n, other.n);
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (*a - *b).norm() < MATRIX_EPS)
    }
}

pub(crate) fn all_pauli_strings(num_qubits: usize) -> Vec<String> {
    let mut strings = vec![String::new()];
    for _ in 0..num_qubits {
        let mut next = Vec::with_capacity(strings.len() * 4);
        for prefix in &strings {
            for suffix in ['I', 'X', 'Y', 'Z'] {
                let mut value = prefix.clone();
                value.push(suffix);
                next.push(value);
            }
        }
        strings = next;
    }
    strings
}

pub(crate) fn conjugate_pauli(gate: CliffordMatrixGate, input: &str) -> SignedPauli {
    let unitary = gate_matrix(gate);
    let input_matrix = pauli_string_matrix(input);
    let image = unitary.mul(&input_matrix).mul(&unitary.dagger());
    classify_signed_pauli(&image, input.len())
}

fn classify_signed_pauli(image: &Matrix, num_qubits: usize) -> SignedPauli {
    for pauli in all_pauli_strings(num_qubits) {
        let matrix = pauli_string_matrix(&pauli);
        if image.approx_eq(&matrix) {
            return SignedPauli { sign: 1, pauli };
        }
        if image.approx_eq(&matrix.scale(Complex64::new(-1.0, 0.0))) {
            return SignedPauli { sign: -1, pauli };
        }
    }
    panic!("matrix image is not a signed Pauli");
}

fn gate_matrix(gate: CliffordMatrixGate) -> Matrix {
    match gate {
        CliffordMatrixGate::SZdg => sqrt_pauli_matrix("Z", true),
        CliffordMatrixGate::F => sqrt_pauli_matrix("Z", false).mul(&sqrt_pauli_matrix("X", false)),
        CliffordMatrixGate::Fdg => sqrt_pauli_matrix("X", true).mul(&sqrt_pauli_matrix("Z", true)),
        CliffordMatrixGate::SX => sqrt_pauli_matrix("X", false),
        CliffordMatrixGate::SXdg => sqrt_pauli_matrix("X", true),
        CliffordMatrixGate::SY => sqrt_pauli_matrix("Y", false),
        CliffordMatrixGate::SYdg => sqrt_pauli_matrix("Y", true),
        CliffordMatrixGate::CX => controlled_x_matrix(),
        CliffordMatrixGate::CY => controlled_y_matrix(),
        CliffordMatrixGate::CZ => controlled_z_matrix(),
        CliffordMatrixGate::SXX => sqrt_pauli_matrix("XX", false),
        CliffordMatrixGate::SXXdg => sqrt_pauli_matrix("XX", true),
        CliffordMatrixGate::SYY => sqrt_pauli_matrix("YY", false),
        CliffordMatrixGate::SYYdg => sqrt_pauli_matrix("YY", true),
        CliffordMatrixGate::SZZ => sqrt_pauli_matrix("ZZ", false),
        CliffordMatrixGate::SZZdg => sqrt_pauli_matrix("ZZ", true),
        CliffordMatrixGate::SWAP => swap_matrix(),
    }
}

fn sqrt_pauli_matrix(pauli: &str, adjoint: bool) -> Matrix {
    let pauli = pauli_string_matrix(pauli);
    let identity = Matrix::identity(pauli.n);
    let phase_sign = if adjoint { 1.0 } else { -1.0 };
    identity
        .scale(Complex64::new(FRAC_1_SQRT_2, 0.0))
        .add(&pauli.scale(Complex64::new(0.0, phase_sign * FRAC_1_SQRT_2)))
}

fn pauli_string_matrix(pauli: &str) -> Matrix {
    let mut matrix = Matrix::from_data(1, vec![Complex64::new(1.0, 0.0)]);
    for label in pauli.chars() {
        matrix = kron(&matrix, &single_pauli_matrix(label));
    }
    matrix
}

fn single_pauli_matrix(label: char) -> Matrix {
    let one = Complex64::new(1.0, 0.0);
    let minus_one = Complex64::new(-1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let minus_i = Complex64::new(0.0, -1.0);

    match label {
        'I' => Matrix::from_data(2, vec![one, zero, zero, one]),
        'X' => Matrix::from_data(2, vec![zero, one, one, zero]),
        'Y' => Matrix::from_data(2, vec![zero, minus_i, i, zero]),
        'Z' => Matrix::from_data(2, vec![one, zero, zero, minus_one]),
        _ => panic!("invalid Pauli label {label}"),
    }
}

fn kron(left: &Matrix, right: &Matrix) -> Matrix {
    let n = left.n * right.n;
    let mut out = Matrix::zeros(n);
    for lr in 0..left.n {
        for lc in 0..left.n {
            for rr in 0..right.n {
                for rc in 0..right.n {
                    out.set(
                        lr * right.n + rr,
                        lc * right.n + rc,
                        left.get(lr, lc) * right.get(rr, rc),
                    );
                }
            }
        }
    }
    out
}

fn controlled_y_matrix() -> Matrix {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let minus_i = Complex64::new(0.0, -1.0);
    Matrix::from_data(
        4,
        vec![
            one, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, minus_i, zero, zero, i,
            zero,
        ],
    )
}

fn controlled_x_matrix() -> Matrix {
    Matrix::from_data(
        4,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
}

fn controlled_z_matrix() -> Matrix {
    Matrix::from_data(
        4,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
}

fn swap_matrix() -> Matrix {
    Matrix::from_data(
        4,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
}
