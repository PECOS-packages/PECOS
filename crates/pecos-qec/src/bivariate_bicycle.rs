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

//! Bivariate-bicycle codes and their depth-eight syndrome-extraction circuit.
//!
//! The construction and circuit schedule follow Tables 4 and 5 of
//! [Bravyi et al., arXiv:2308.07915](https://arxiv.org/abs/2308.07915).

use pecos_quantum::{F2Matrix, TickCircuit, TickMeasRef};
use thiserror::Error;

use crate::memory_circuit::{
    CssMemoryCircuitFinish, discover_css_logical_operators, finish_css_memory_circuit,
};
use crate::{MemoryBasis, ParityCheckMatrix};

/// One monomial `x^a y^b` in `F_2[x, y] / (x^l - 1, y^m - 1)`.
pub type BbMonomial = (usize, usize);

/// Memory-experiment basis accepted by the bivariate-bicycle builder.
pub type BbMemoryBasis = MemoryBasis;

/// Errors reported while constructing a bivariate-bicycle code or memory circuit.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BivariateBicycleError {
    /// A torus dimension was zero.
    #[error("bivariate-bicycle dimensions must be positive, got l={l}, m={m}")]
    ZeroDimension { l: usize, m: usize },
    /// The block or circuit size overflowed `usize`.
    #[error("bivariate-bicycle dimensions l={l}, m={m} overflow the supported size")]
    SizeOverflow { l: usize, m: usize },
    /// A weight-three polynomial did not contain exactly three terms.
    #[error("{polynomial} must contain exactly three monomials, got {actual}")]
    WrongTermCount {
        polynomial: &'static str,
        actual: usize,
    },
    /// A monomial exponent did not name a canonical torus shift.
    #[error(
        "{polynomial} monomial {term_index} has exponent ({x_power}, {y_power}) outside Z_{l} x Z_{m}"
    )]
    ExponentOutOfRange {
        polynomial: &'static str,
        term_index: usize,
        x_power: usize,
        y_power: usize,
        l: usize,
        m: usize,
    },
    /// Two terms describe the same permutation matrix.
    #[error("{polynomial} monomials {first} and {second} describe the same permutation matrix")]
    DuplicateMonomial {
        polynomial: &'static str,
        first: usize,
        second: usize,
    },
    /// A generated monomial was not a permutation matrix.
    #[error("{polynomial} monomial {term_index} is not a permutation matrix")]
    NonPermutationMonomial {
        polynomial: &'static str,
        term_index: usize,
    },
    /// The CSS commutation condition failed.
    #[error("bivariate-bicycle checks do not commute: Hx * Hz^T is nonzero")]
    NonCommutingChecks,
    /// At least one syndrome cycle is required.
    #[error("bivariate-bicycle memory experiment requires at least one syndrome cycle")]
    ZeroRounds,
    /// An internally produced measurement reference could not be annotated.
    #[error("invalid bivariate-bicycle measurement annotation: {0}")]
    InvalidAnnotation(String),
}

/// A validated bivariate-bicycle CSS code `QC(A, B)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BivariateBicycleCode {
    l: usize,
    m: usize,
    a_terms: [BbMonomial; 3],
    b_terms: [BbMonomial; 3],
    hx: ParityCheckMatrix,
    hz: ParityCheckMatrix,
    logical_x: ParityCheckMatrix,
    logical_z: ParityCheckMatrix,
}

impl BivariateBicycleCode {
    /// Construct `QC(A, B)` from two weight-three bivariate polynomials.
    ///
    /// A monomial `(a, b)` denotes the permutation whose one in row `(i, j)`
    /// is at column `(i + a mod l, j + b mod m)`. Exponents must use the
    /// canonical ranges `0..l` and `0..m`; rejecting rather than reducing them
    /// catches malformed specifications. The constructor validates all six
    /// permutation matrices and `Hx * Hz^T = 0` before returning.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, non-weight-three polynomials,
    /// non-canonical or duplicate terms, size overflow, or noncommuting checks.
    ///
    /// # Panics
    ///
    /// Panics only if the internally generated rectangular binary matrices are
    /// rejected by [`ParityCheckMatrix`], which would violate this module's
    /// construction invariant.
    pub fn new(
        l: usize,
        m: usize,
        a_terms: &[BbMonomial],
        b_terms: &[BbMonomial],
    ) -> Result<Self, BivariateBicycleError> {
        let block_size = validated_block_size(l, m)?;
        let a_terms = validate_terms("A", l, m, a_terms)?;
        let b_terms = validate_terms("B", l, m, b_terms)?;
        let a_monomials = monomial_matrices("A", l, m, &a_terms)?;
        let b_monomials = monomial_matrices("B", l, m, &b_terms)?;
        let a = sum_matrices(&a_monomials);
        let b = sum_matrices(&b_monomials);
        let num_qubits = block_size
            .checked_mul(2)
            .ok_or(BivariateBicycleError::SizeOverflow { l, m })?;
        let mut hx = F2Matrix::zeros(block_size, num_qubits);
        let mut hz = F2Matrix::zeros(block_size, num_qubits);
        for row in 0..block_size {
            for column in 0..block_size {
                hx.set(row, column, a.get(row, column));
                hx.set(row, block_size + column, b.get(row, column));
                hz.set(row, column, b.get(column, row));
                hz.set(row, block_size + column, a.get(column, row));
            }
        }
        if hx.mul(&hz.transpose()) != F2Matrix::zeros(block_size, block_size) {
            return Err(BivariateBicycleError::NonCommutingChecks);
        }

        let hx = ParityCheckMatrix::from_dense(hx.rows())
            .expect("a nonempty rectangular binary matrix was generated");
        let hz = ParityCheckMatrix::from_dense(hz.rows())
            .expect("a nonempty rectangular binary matrix was generated");
        let (logical_x, logical_z) = discover_css_logical_operators(&hx, &hz);

        Ok(Self {
            l,
            m,
            a_terms,
            b_terms,
            hx,
            hz,
            logical_x,
            logical_z,
        })
    }

    /// Number of data qubits, `n = 2lm`.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.hx.num_qubits()
    }

    /// Number of encoded logical qubits, `k = n - rank(Hx) - rank(Hz)`.
    #[must_use]
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits() - self.hx.rank() - self.hz.rank()
    }

    /// X-type stabilizer parity-check matrix `Hx = [A | B]`.
    #[must_use]
    pub fn hx(&self) -> &ParityCheckMatrix {
        &self.hx
    }

    /// Z-type stabilizer parity-check matrix `Hz = [B^T | A^T]`.
    #[must_use]
    pub fn hz(&self) -> &ParityCheckMatrix {
        &self.hz
    }

    /// A basis of logical X operators, represented as binary rows.
    #[must_use]
    pub fn logical_x(&self) -> &ParityCheckMatrix {
        &self.logical_x
    }

    /// A basis of logical Z operators, represented as binary rows.
    #[must_use]
    pub fn logical_z(&self) -> &ParityCheckMatrix {
        &self.logical_z
    }

    /// Torus dimensions `(l, m)`.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.l, self.m)
    }

    /// The three canonical exponent pairs of `A`.
    #[must_use]
    pub fn a_terms(&self) -> &[BbMonomial; 3] {
        &self.a_terms
    }

    /// The three canonical exponent pairs of `B`.
    #[must_use]
    pub fn b_terms(&self) -> &[BbMonomial; 3] {
        &self.b_terms
    }
}

/// Construct the depth-eight bivariate-bicycle memory experiment.
///
/// The four physical registers are ordered `q(X), q(L), q(R), q(Z)`. The
/// initial tick prepares the data and `q(Z)` registers. Each syndrome cycle is
/// then exactly the eight depth-one rounds of Table 5 in arXiv:2308.07915.
/// A final data-measurement tick closes the matching boundary detectors and
/// defines all `k` logical observables.
///
/// # Errors
///
/// Returns any code-structure error and rejects zero syndrome cycles.
pub fn bb_memory_circuit(
    l: usize,
    m: usize,
    a_terms: &[BbMonomial],
    b_terms: &[BbMonomial],
    rounds: usize,
    basis: BbMemoryBasis,
) -> Result<TickCircuit, BivariateBicycleError> {
    if rounds == 0 {
        return Err(BivariateBicycleError::ZeroRounds);
    }
    let code = BivariateBicycleCode::new(l, m, a_terms, b_terms)?;
    build_memory_circuit(&code, rounds, basis)
}

fn build_memory_circuit(
    code: &BivariateBicycleCode,
    rounds: usize,
    basis: BbMemoryBasis,
) -> Result<TickCircuit, BivariateBicycleError> {
    let block_size = code.l * code.m;
    block_size
        .checked_mul(4)
        .ok_or(BivariateBicycleError::SizeOverflow {
            l: code.l,
            m: code.m,
        })?;
    let qx = |i| i;
    let ql = |i| block_size + i;
    let qr = |i| 2 * block_size + i;
    let qz = |i| 3 * block_size + i;
    let x_qubits: Vec<_> = (0..block_size).map(qx).collect();
    let left_qubits: Vec<_> = (0..block_size).map(ql).collect();
    let right_qubits: Vec<_> = (0..block_size).map(qr).collect();
    let z_qubits: Vec<_> = (0..block_size).map(qz).collect();
    let data_qubits: Vec<_> = left_qubits.iter().chain(&right_qubits).copied().collect();

    let mut circuit = TickCircuit::new();
    {
        let tick = circuit.tick();
        tick.pz(&z_qubits);
    }
    // Data preparation shares the pre-cycle layer with the disjoint q(Z) reset.
    let initial_tick = circuit.get_tick_mut(0).expect("the initial tick exists");
    let prep = match basis {
        BbMemoryBasis::X => pecos_core::Gate::px(&data_qubits),
        BbMemoryBasis::Z => pecos_core::Gate::pz(&data_qubits),
    };
    initial_tick.add_gate(prep);

    let mut x_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    let mut z_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        // Round 1.
        circuit.tick().px(&x_qubits);
        let r1 = (0..block_size)
            .map(|i| {
                (
                    qr(transpose_shift(i, code.l, code.m, code.a_terms[0])),
                    qz(i),
                )
            })
            .collect::<Vec<_>>();
        circuit
            .get_tick_mut(circuit.num_ticks() - 1)
            .unwrap()
            .add_gate(pecos_core::Gate::cx(&r1));
        circuit
            .get_tick_mut(circuit.num_ticks() - 1)
            .unwrap()
            .add_gate(pecos_core::Gate::idle(
                1.0,
                left_qubits
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect::<pecos_core::GateQubits>(),
            ));

        // Round 2.
        let x_left = (0..block_size)
            .map(|i| (qx(i), ql(forward_shift(i, code.l, code.m, code.a_terms[1]))))
            .collect::<Vec<_>>();
        let right_z = (0..block_size)
            .map(|i| {
                (
                    qr(transpose_shift(i, code.l, code.m, code.a_terms[2])),
                    qz(i),
                )
            })
            .collect::<Vec<_>>();
        circuit.tick().cx(&x_left).cx(&right_z);

        // Round 3.
        let x_right = (0..block_size)
            .map(|i| (qx(i), qr(forward_shift(i, code.l, code.m, code.b_terms[1]))))
            .collect::<Vec<_>>();
        let left_z = (0..block_size)
            .map(|i| {
                (
                    ql(transpose_shift(i, code.l, code.m, code.b_terms[0])),
                    qz(i),
                )
            })
            .collect::<Vec<_>>();
        circuit.tick().cx(&x_right).cx(&left_z);

        // Round 4.
        let x_right = (0..block_size)
            .map(|i| (qx(i), qr(forward_shift(i, code.l, code.m, code.b_terms[0]))))
            .collect::<Vec<_>>();
        let left_z = (0..block_size)
            .map(|i| {
                (
                    ql(transpose_shift(i, code.l, code.m, code.b_terms[1])),
                    qz(i),
                )
            })
            .collect::<Vec<_>>();
        circuit.tick().cx(&x_right).cx(&left_z);

        // Round 5.
        let x_right = (0..block_size)
            .map(|i| (qx(i), qr(forward_shift(i, code.l, code.m, code.b_terms[2]))))
            .collect::<Vec<_>>();
        let left_z = (0..block_size)
            .map(|i| {
                (
                    ql(transpose_shift(i, code.l, code.m, code.b_terms[2])),
                    qz(i),
                )
            })
            .collect::<Vec<_>>();
        circuit.tick().cx(&x_right).cx(&left_z);

        // Round 6.
        let x_left = (0..block_size)
            .map(|i| (qx(i), ql(forward_shift(i, code.l, code.m, code.a_terms[0]))))
            .collect::<Vec<_>>();
        let right_z = (0..block_size)
            .map(|i| {
                (
                    qr(transpose_shift(i, code.l, code.m, code.a_terms[1])),
                    qz(i),
                )
            })
            .collect::<Vec<_>>();
        circuit.tick().cx(&x_left).cx(&right_z);

        // Round 7.
        let x_left = (0..block_size)
            .map(|i| (qx(i), ql(forward_shift(i, code.l, code.m, code.a_terms[2]))))
            .collect::<Vec<_>>();
        let mut tick = circuit.tick();
        tick.cx(&x_left).idle(1, &right_qubits);
        z_measurements.push(tick.mz(&z_qubits));

        // Round 8.
        let x_refs = circuit.tick().mx(&x_qubits);
        let tick = circuit.get_tick_mut(circuit.num_ticks() - 1).unwrap();
        tick.add_gate(pecos_core::Gate::pz(&z_qubits));
        tick.add_gate(pecos_core::Gate::idle(
            1.0,
            data_qubits
                .iter()
                .copied()
                .map(Into::into)
                .collect::<pecos_core::GateQubits>(),
        ));
        x_measurements.push(x_refs);
    }

    finish_css_memory_circuit(
        &mut circuit,
        CssMemoryCircuitFinish {
            data_qubits: &data_qubits,
            hx: code.hx(),
            hz: code.hz(),
            logical_x: code.logical_x(),
            logical_z: code.logical_z(),
            x_measurements: &x_measurements,
            z_measurements: &z_measurements,
            rounds,
            basis,
            circuit_type: "bivariate_bicycle_memory",
        },
    )
    .map_err(BivariateBicycleError::InvalidAnnotation)?;
    Ok(circuit)
}

fn validated_block_size(l: usize, m: usize) -> Result<usize, BivariateBicycleError> {
    if l == 0 || m == 0 {
        return Err(BivariateBicycleError::ZeroDimension { l, m });
    }
    l.checked_mul(m)
        .ok_or(BivariateBicycleError::SizeOverflow { l, m })
}

fn validate_terms(
    polynomial: &'static str,
    l: usize,
    m: usize,
    terms: &[BbMonomial],
) -> Result<[BbMonomial; 3], BivariateBicycleError> {
    if terms.len() != 3 {
        return Err(BivariateBicycleError::WrongTermCount {
            polynomial,
            actual: terms.len(),
        });
    }
    for (term_index, &(x_power, y_power)) in terms.iter().enumerate() {
        if x_power >= l || y_power >= m {
            return Err(BivariateBicycleError::ExponentOutOfRange {
                polynomial,
                term_index,
                x_power,
                y_power,
                l,
                m,
            });
        }
    }
    for first in 0..terms.len() {
        for second in first + 1..terms.len() {
            if terms[first] == terms[second] {
                return Err(BivariateBicycleError::DuplicateMonomial {
                    polynomial,
                    first,
                    second,
                });
            }
        }
    }
    Ok([terms[0], terms[1], terms[2]])
}

fn monomial_matrices(
    polynomial: &'static str,
    l: usize,
    m: usize,
    terms: &[BbMonomial; 3],
) -> Result<[F2Matrix; 3], BivariateBicycleError> {
    let matrices = terms.map(|term| monomial_matrix(l, m, term));
    for (term_index, matrix) in matrices.iter().enumerate() {
        if !is_permutation_matrix(matrix) {
            return Err(BivariateBicycleError::NonPermutationMonomial {
                polynomial,
                term_index,
            });
        }
    }
    Ok(matrices)
}

fn monomial_matrix(l: usize, m: usize, term: BbMonomial) -> F2Matrix {
    let size = l * m;
    let mut matrix = F2Matrix::zeros(size, size);
    for row in 0..size {
        matrix.set(row, forward_shift(row, l, m, term), 1);
    }
    matrix
}

fn is_permutation_matrix(matrix: &F2Matrix) -> bool {
    let rows = matrix.num_rows();
    rows == matrix.num_cols()
        && (0..rows).all(|row| {
            (0..rows)
                .filter(|&column| matrix.get(row, column) == 1)
                .count()
                == 1
        })
        && (0..rows).all(|column| {
            (0..rows)
                .filter(|&row| matrix.get(row, column) == 1)
                .count()
                == 1
        })
}

fn sum_matrices(matrices: &[F2Matrix; 3]) -> F2Matrix {
    let size = matrices[0].num_rows();
    let mut sum = F2Matrix::zeros(size, size);
    for matrix in matrices {
        for row in 0..size {
            for column in 0..size {
                sum.set(row, column, sum.get(row, column) ^ matrix.get(row, column));
            }
        }
    }
    sum
}

fn forward_shift(index: usize, l: usize, m: usize, term: BbMonomial) -> usize {
    let (x_power, y_power) = term;
    let x = index / m;
    let y = index % m;
    ((x + x_power) % l) * m + (y + y_power) % m
}

fn transpose_shift(index: usize, l: usize, m: usize, term: BbMonomial) -> usize {
    let (x_power, y_power) = term;
    let x = index / m;
    let y = index % m;
    ((x + l - x_power) % l) * m + (y + m - y_power) % m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connected_cluster_code_distance;
    use crate::fault_tolerance::dem_builder::{DemBuilder, DemSampler, NoiseConfig};
    use crate::fault_tolerance::{
        connected_cluster_fault_distance, graphlike_fault_distance, per_observable_fault_distances,
    };
    use pecos_quantum::{AnnotationKind, GateType};
    use pecos_random::PecosRng;
    use pecos_simulators::{CircuitExecutor, SparseStab};
    use std::collections::BTreeSet;
    use std::time::Instant;

    const A: [BbMonomial; 3] = [(3, 0), (0, 1), (0, 2)];
    const B: [BbMonomial; 3] = [(0, 3), (1, 0), (2, 0)];

    fn code_72() -> BivariateBicycleCode {
        BivariateBicycleCode::new(6, 6, &A, &B).expect("the paper's [[72,12,6]] code is valid")
    }

    fn circuit_72(rounds: usize, basis: BbMemoryBasis) -> TickCircuit {
        bb_memory_circuit(6, 6, &A, &B, rounds, basis)
            .expect("the paper's [[72,12,6]] memory circuit is valid")
    }

    fn sample_annotation_parities(circuit: &TickCircuit) -> (Vec<bool>, Vec<bool>) {
        let mut sim = SparseStab::new(144);
        let measurements = CircuitExecutor::new(circuit).run(&mut sim);
        let parity = |ids: &[pecos_core::MeasId]| {
            ids.iter()
                .fold(false, |value, id| value ^ measurements[id.index()].outcome)
        };
        let mut detectors = Vec::new();
        let mut observables = Vec::new();
        for annotation in circuit.annotations() {
            match &annotation.kind {
                AnnotationKind::Detector {
                    measurement_ids, ..
                } => detectors.push(parity(measurement_ids)),
                AnnotationKind::Observable { measurement_ids } => {
                    observables.push(parity(measurement_ids));
                }
                AnnotationKind::TrackedPauli => {}
            }
        }
        (detectors, observables)
    }

    #[test]
    fn code_72_has_the_reported_parameters_and_distance() {
        let code = code_72();
        assert_eq!(code.num_qubits(), 72);
        assert_eq!(code.num_logical_qubits(), 12);
        assert_eq!(code.logical_x().num_checks(), 12);
        assert_eq!(code.logical_z().num_checks(), 12);
        assert_eq!(
            connected_cluster_code_distance(code.hx(), code.logical_x(), 6)
                .expect("distance is at most six")
                .distance,
            6
        );
    }

    #[test]
    fn schedule_has_exact_round_shape_and_tanner_edges() {
        let code = code_72();
        let circuit = circuit_72(2, BbMemoryBasis::Z);
        let s = 36;
        assert_eq!(circuit.num_ticks(), 8 * 2 + 2);

        let expected_idle: [BTreeSet<usize>; 8] = [
            (s..2 * s).collect(),
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::new(),
            (2 * s..3 * s).collect(),
            (s..3 * s).collect(),
        ];
        let expected_counts = [
            vec![(GateType::PX, s), (GateType::CX, s), (GateType::Idle, s)],
            vec![(GateType::CX, 2 * s)],
            vec![(GateType::CX, 2 * s)],
            vec![(GateType::CX, 2 * s)],
            vec![(GateType::CX, 2 * s)],
            vec![(GateType::CX, 2 * s)],
            vec![(GateType::CX, s), (GateType::MZ, s), (GateType::Idle, s)],
            vec![
                (GateType::MX, s),
                (GateType::PZ, s),
                (GateType::Idle, 2 * s),
            ],
        ];

        for cycle in 0..2 {
            for round in 0..8 {
                let tick = circuit.get_tick(1 + 8 * cycle + round).unwrap();
                for &(gate_type, expected) in &expected_counts[round] {
                    let actual = tick
                        .iter_gate_instances()
                        .filter(|gate| gate.gate_type() == gate_type)
                        .count();
                    assert_eq!(actual, expected, "cycle {cycle}, round {}", round + 1);
                }
                let idle = tick
                    .iter_gate_instances()
                    .filter(|gate| gate.gate_type() == GateType::Idle)
                    .map(|gate| gate.qubits()[0].index())
                    .collect::<BTreeSet<_>>();
                assert_eq!(idle, expected_idle[round], "round {} idles", round + 1);

                for gate in tick
                    .iter_gate_instances()
                    .filter(|gate| gate.gate_type() == GateType::CX)
                {
                    let control = gate.qubits()[0].index();
                    let target = gate.qubits()[1].index();
                    assert!(target >= s, "q(X) is never a CNOT target");
                    assert!(control < 3 * s, "q(Z) is never a CNOT control");
                    if control < s {
                        assert!(target < 3 * s);
                        assert_eq!(code.hx().matrix().get(control, target - s), 1);
                    } else {
                        assert!(target >= 3 * s);
                        assert_eq!(code.hz().matrix().get(target - 3 * s, control - s), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn noiseless_memory_has_empty_syndrome_and_deterministic_observables() {
        for basis in [BbMemoryBasis::X, BbMemoryBasis::Z] {
            let circuit = circuit_72(2, basis);
            let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0)
                .expect("fault-free circuit has a DEM");
            assert_eq!(dem.num_detectors(), 144);
            assert_eq!(dem.num_observables(), 12);
            assert!(dem.to_mechanisms().0.is_empty());

            let zero_noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0);
            let sampler = DemSampler::from_tick_circuit(&circuit, &zero_noise)
                .expect("all annotated detector parities are deterministic");
            let mut rng = PecosRng::seed_from_u64(23);
            let (detectors, observables) = sampler.sample(&mut rng);
            assert_eq!(detectors, vec![false; 144]);
            assert_eq!(observables, vec![false; 12]);

            for _ in 0..8 {
                let (detectors, observables) = sample_annotation_parities(&circuit);
                assert_eq!(detectors, vec![false; 144]);
                assert_eq!(observables, vec![false; 12]);
            }
        }
    }

    #[test]
    fn swapping_rounds_three_and_six_breaks_fault_free_detectors() {
        let mut circuit = circuit_72(2, BbMemoryBasis::Z);
        for cycle in 0..2 {
            circuit
                .ticks_mut()
                .swap(1 + 8 * cycle + 2, 1 + 8 * cycle + 5);
        }
        let caught = (0..16).any(|_| {
            let (detectors, _) = sample_annotation_parities(&circuit);
            detectors.into_iter().any(|event| event)
        });
        assert!(
            caught,
            "the swapped schedule must not pass fault-free validation"
        );
    }

    #[test]
    fn builder_output_is_deterministic() {
        assert_eq!(
            format!("{:?}", circuit_72(2, BbMemoryBasis::Z)),
            format!("{:?}", circuit_72(2, BbMemoryBasis::Z))
        );
    }

    #[test]
    fn noncanonical_monomial_exponent_is_rejected() {
        let broken_a = [(6, 0), (0, 1), (0, 2)];
        assert!(matches!(
            BivariateBicycleCode::new(6, 6, &broken_a, &B),
            Err(BivariateBicycleError::ExponentOutOfRange {
                polynomial: "A",
                term_index: 0,
                ..
            })
        ));
    }

    #[test]
    fn every_observable_has_a_weight_six_circuit_fault_witness() {
        const WITNESSES: [[usize; 6]; 3] = [
            [4176, 4186, 4197, 4199, 4208, 4219],
            [3478, 3486, 3554, 3639, 3757, 3769],
            [4203, 4209, 4233, 4241, 4268, 4283],
        ];

        let circuit = circuit_72(2, BbMemoryBasis::Z);
        let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.001, 0.001, 0.001, 0.001)
            .expect("uniform circuit noise has a DEM");
        let (mechanisms, _) = dem.to_mechanisms();
        let mut covered_observables = BTreeSet::new();

        for witness in WITNESSES {
            let mut detectors = BTreeSet::new();
            let mut observables = BTreeSet::new();
            for index in witness {
                let (_, mechanism_detectors, mechanism_observables) = &mechanisms[index];
                for &detector in mechanism_detectors {
                    if !detectors.insert(detector) {
                        detectors.remove(&detector);
                    }
                }
                for &observable in mechanism_observables {
                    if !observables.insert(observable) {
                        observables.remove(&observable);
                    }
                }
            }
            assert!(
                detectors.is_empty(),
                "a stored weight-six circuit witness must be detector-free"
            );
            assert!(!observables.is_empty(), "a stored witness must be logical");
            covered_observables.extend(observables);
        }

        assert_eq!(
            covered_observables,
            (0_u32..12).collect(),
            "the stored witnesses must cover every encoded observable"
        );
    }

    #[test]
    #[ignore = "exact 4,284-mechanism [[72,12,6]] hypergraph search is too slow for normal tests"]
    fn circuit_distance_72_two_cycles() {
        let started = Instant::now();
        let circuit = circuit_72(2, BbMemoryBasis::Z);
        let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.001, 0.001, 0.001, 0.001)
            .expect("uniform circuit noise has a DEM");
        let build_elapsed = started.elapsed();
        let (mechanisms, _) = dem.to_mechanisms();
        println!(
            "BB [[72,12,6]] DEM: {} mechanisms, build {build_elapsed:?}",
            mechanisms.len()
        );

        let distance_started = Instant::now();
        let (method, overall) = match graphlike_fault_distance(&dem) {
            Ok(result) => ("graphlike", result),
            Err(error) => {
                println!("graphlike unavailable: {error}");
                (
                    "connected_cluster",
                    connected_cluster_fault_distance(&dem, 6),
                )
            }
        };
        let distance_elapsed = distance_started.elapsed();
        println!("{method} overall: {overall:?}, search {distance_elapsed:?}");

        let per_started = Instant::now();
        let per_observable = per_observable_fault_distances(&dem, 6);
        println!(
            "per-observable: {per_observable:?}, search {:?}",
            per_started.elapsed()
        );

        let overall = overall.expect("a logical fault exists through weight six");
        if overall.distance < 6 {
            println!("FULL SUB-DISTANCE WITNESS:");
            for &index in &overall.mechanism_indices {
                println!("mechanism[{index}] = {:?}", mechanisms[index]);
            }
        }
        assert_eq!(overall.distance, 6);
        assert!(
            per_observable
                .iter()
                .all(|result| result.as_ref().is_some_and(|result| result.distance == 6)),
            "every encoded observable must retain circuit fault distance six"
        );
    }
}
