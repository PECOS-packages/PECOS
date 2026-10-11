// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Independent, <=3-qubit dense Lindbladian oracle. No EEG formula calls.

use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_eeg::eeg::EegType;

#[derive(Clone, Copy, Debug, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    pub fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

pub type Matrix = Vec<Complex>;

#[derive(Clone, Copy)]
pub struct Generator {
    pub kind: EegType,
    pub p: &'static str,
    pub q: Option<&'static str>,
    pub coeff: f64,
}

fn multiply(a: &[Complex], b: &[Complex], dim: usize) -> Matrix {
    let mut out = vec![Complex::default(); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                out[i * dim + j] = out[i * dim + j].add(a[i * dim + k].mul(b[k * dim + j]));
            }
        }
    }
    out
}

fn combination(terms: &[(&[Complex], Complex)]) -> Matrix {
    (0..terms[0].0.len())
        .map(|i| {
            terms
                .iter()
                .fold(Complex::default(), |sum, (matrix, factor)| {
                    sum.add(matrix[i].mul(*factor))
                })
        })
        .collect()
}

fn real(value: f64) -> Complex {
    Complex { re: value, im: 0.0 }
}

fn pauli(label: &str) -> Matrix {
    let dim = 1 << label.len();
    let mut matrix = vec![Complex::default(); dim * dim];
    for column in 0..dim {
        let mut row = column;
        let mut amplitude = real(1.0);
        for (q, symbol) in label.bytes().enumerate() {
            let one = column & (1 << q) != 0;
            match symbol {
                b'I' => {}
                b'X' => row ^= 1 << q,
                b'Y' => {
                    row ^= 1 << q;
                    amplitude = amplitude.mul(Complex {
                        re: 0.0,
                        im: if one { -1.0 } else { 1.0 },
                    });
                }
                b'Z' => amplitude = amplitude.scale(if one { -1.0 } else { 1.0 }),
                _ => panic!("invalid Pauli symbol"),
            }
        }
        matrix[row * dim + column] = amplitude;
    }
    matrix
}

/// Prepare |psi><psi| with explicit short Clifford state-vector operations.
pub fn prepared(n: usize, gates: &[Gate]) -> Matrix {
    assert!(n <= 3);
    let dim = 1 << n;
    let mut state = vec![Complex::default(); dim];
    state[0] = real(1.0);
    for gate in gates {
        let q = gate.qubits[0].index();
        let old = state.clone();
        for i in 0..dim {
            let bit = i & (1 << q) != 0;
            state[i] = match gate.gate_type {
                GateType::H => {
                    let low = i & !(1 << q);
                    old[low]
                        .add(old[low | (1 << q)].scale(if bit { -1.0 } else { 1.0 }))
                        .scale(std::f64::consts::FRAC_1_SQRT_2)
                }
                GateType::X => old[i ^ (1 << q)],
                GateType::Z => old[i].scale(if bit { -1.0 } else { 1.0 }),
                GateType::SZ => old[i].mul(if bit {
                    Complex { re: 0.0, im: 1.0 }
                } else {
                    real(1.0)
                }),
                GateType::CX => {
                    old[if bit {
                        i ^ (1 << gate.qubits[1].index())
                    } else {
                        i
                    }]
                }
                other => panic!("unsupported oracle preparation {other:?}"),
            };
        }
    }
    (0..dim * dim)
        .map(|i| state[i / dim].mul(state[i % dim].conj()))
        .collect()
}

/// Written directly from H=-i[P,rho], S=rho-P rho P,
/// C=P rho Q+Q rho P-{{P,Q},rho}/2,
/// A=i(P rho Q-Q rho P+{[P,Q],rho}/2).
pub fn lindbladian(rho: &[Complex], generators: &[Generator]) -> Matrix {
    let dim = rho.len().isqrt();
    let mut sum = vec![Complex::default(); rho.len()];
    for generator in generators {
        let p = pauli(generator.p);
        assert_eq!(p.len(), rho.len());
        let pr = multiply(&p, rho, dim);
        let rp = multiply(rho, &p, dim);
        let image = match generator.kind {
            EegType::H => combination(&[
                (&pr, Complex { re: 0.0, im: -1.0 }),
                (&rp, Complex { re: 0.0, im: 1.0 }),
            ]),
            EegType::S => combination(&[(rho, real(1.0)), (&multiply(&pr, &p, dim), real(-1.0))]),
            EegType::C | EegType::A => {
                let q = pauli(generator.q.expect("C/A needs Q"));
                let prq = multiply(&pr, &q, dim);
                let qrp = multiply(&q, &rp, dim);
                let pq = multiply(&p, &q, dim);
                let qp = multiply(&q, &p, dim);
                let is_c = generator.kind == EegType::C;
                let bracket =
                    combination(&[(&pq, real(1.0)), (&qp, real(if is_c { 1.0 } else { -1.0 }))]);
                let bracket_r = multiply(&bracket, rho, dim);
                let r_bracket = multiply(rho, &bracket, dim);
                if is_c {
                    combination(&[
                        (&prq, real(1.0)),
                        (&qrp, real(1.0)),
                        (&bracket_r, real(-0.5)),
                        (&r_bracket, real(-0.5)),
                    ])
                } else {
                    combination(&[
                        (&prq, Complex { re: 0.0, im: 1.0 }),
                        (&qrp, Complex { re: 0.0, im: -1.0 }),
                        (&bracket_r, Complex { re: 0.0, im: 0.5 }),
                        (&r_bracket, Complex { re: 0.0, im: 0.5 }),
                    ])
                }
            }
        };
        for (value, contribution) in sum.iter_mut().zip(image) {
            *value = value.add(contribution.scale(generator.coeff));
        }
    }
    sum
}

pub fn evolved(rho: &[Complex], generators: &[Generator]) -> Matrix {
    let mut result = rho.to_vec();
    let mut term = rho.to_vec();
    for order in 1..=200 {
        term = lindbladian(&term, generators)
            .into_iter()
            .map(|v| v.scale(1.0 / f64::from(order)))
            .collect();
        for (value, contribution) in result.iter_mut().zip(&term) {
            *value = value.add(*contribution);
        }
        if term.iter().all(|v| v.re.abs() + v.im.abs() < 1e-17) {
            return result;
        }
    }
    panic!("dense exponential did not converge");
}

pub fn trace(matrix: &[Complex]) -> Complex {
    let dim = matrix.len().isqrt();
    (0..dim).fold(Complex::default(), |sum, i| sum.add(matrix[i * dim + i]))
}

/// Project onto exactly the requested flip pattern relative to noiseless rho.
/// Applying this to L[rho] instead of exp(L)[rho] returns the derivative.
pub fn pattern(rho: &[Complex], image: &[Complex], detectors: &[(&str, bool)]) -> f64 {
    let dim = rho.len().isqrt();
    let mut projected = image.to_vec();
    for &(label, flipped) in detectors {
        let d = pauli(label);
        let eigenvalue = trace(&multiply(&d, rho, dim));
        assert!((eigenvalue.re.abs() - 1.0).abs() < 1e-12 && eigenvalue.im.abs() < 1e-12);
        let sign = if flipped {
            -eigenvalue.re
        } else {
            eigenvalue.re
        };
        projected = combination(&[
            (&projected, real(0.5)),
            (&multiply(&d, &projected, dim), real(0.5 * sign)),
        ]);
    }
    let value = trace(&projected);
    assert!(value.im.abs() < 1e-12);
    value.re
}
