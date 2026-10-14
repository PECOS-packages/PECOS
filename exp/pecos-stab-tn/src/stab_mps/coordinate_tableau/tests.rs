// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use nalgebra::{DMatrix, DVector};
use pecos_core::QubitId;
use pecos_random::PecosRng;
use pecos_simulators::CliffordGateable;

type Matrix = DMatrix<Complex64>;

fn row_matrix<S: IndexSet>(gens: &GensGeneric<S>, row: usize) -> Matrix {
    let n = gens.get_num_qubits();
    let mut matrix = Matrix::zeros(1 << n, 1 << n);
    for input in 0..1 << n {
        let mut output = input;
        let mut value = Complex64::new(1.0, 0.0);
        if gens.signs_minus.contains(row) {
            value = -value;
        }
        if gens.signs_i.contains(row) {
            value *= Complex64::new(0.0, 1.0);
        }
        for q in 0..n {
            let bit = input & (1 << q) != 0;
            match (gens.row_x[row].contains(q), gens.row_z[row].contains(q)) {
                (false, false) => {}
                (false, true) => {
                    if bit {
                        value = -value;
                    }
                }
                (true, false) => output ^= 1 << q,
                (true, true) => {
                    output ^= 1 << q;
                    value *= if bit {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::new(0.0, 1.0)
                    };
                }
            }
        }
        matrix[(output, input)] = value;
    }
    matrix
}

fn physical(n: usize, pauli: &[(usize, PauliKindForDecomp)], negative: bool) -> Matrix {
    let mut matrix = Matrix::zeros(1 << n, 1 << n);
    for input in 0..1 << n {
        let mut output = input;
        let mut value = Complex64::new(if negative { -1.0 } else { 1.0 }, 0.0);
        for &(q, kind) in pauli {
            let bit = input & (1 << q) != 0;
            match kind {
                PauliKindForDecomp::X => output ^= 1 << q,
                PauliKindForDecomp::Z => {
                    if bit {
                        value = -value;
                    }
                }
                PauliKindForDecomp::Y => {
                    output ^= 1 << q;
                    value *= if bit {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::new(0.0, 1.0)
                    };
                }
            }
        }
        matrix[(output, input)] = value;
    }
    matrix
}

fn invariants(tableau: &SparseStabY) {
    let n = tableau.num_qubits();
    for gens in [tableau.stabs(), tableau.destabs()] {
        for row in 0..n {
            assert!(
                !gens.signs_i.contains(row),
                "Hermitian Y rows have real signs"
            );
            for q in 0..n {
                assert_eq!(gens.row_x[row].contains(q), gens.col_x[q].contains(row));
                assert_eq!(gens.row_z[row].contains(q), gens.col_z[q].contains(row));
            }
        }
    }
    let rows: Vec<_> = [tableau.stabs(), tableau.destabs()]
        .into_iter()
        .flat_map(|gens| (0..n).map(move |row| row_matrix(gens, row)))
        .collect();
    let identity = Matrix::identity(1 << n, 1 << n);
    for (i, left) in rows.iter().enumerate() {
        assert!((left - left.adjoint()).norm() < 1e-12);
        assert!((left * left - &identity).norm() < 1e-12);
        for (j, right) in rows.iter().enumerate() {
            let anticommutes = (i < n) != (j < n) && i % n == j % n;
            let sign = Complex64::new(if anticommutes { -1.0 } else { 1.0 }, 0.0);
            assert!(
                (left * right - (right * left) * sign).norm() < 1e-12,
                "rows {i}, {j}"
            );
        }
    }
}

fn basis(tableau: &SparseStabY, active: &[usize]) -> Matrix {
    let n = tableau.num_qubits();
    let mut rng = PecosRng::seed_from_u64(179);
    let mut phi = DVector::from_fn(1 << n, |_, _| {
        Complex64::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5)
    });
    for row in 0..n {
        phi = (&phi + row_matrix(tableau.stabs(), row) * &phi) / Complex64::new(2.0, 0.0);
    }
    let norm = phi.norm();
    assert!(norm > 1e-12);
    phi /= Complex64::new(norm, 0.0);
    let mut columns = vec![phi];
    for x in 1usize..1 << active.len() {
        let bit = x.trailing_zeros() as usize;
        columns.push(row_matrix(tableau.destabs(), active[bit]) * &columns[x ^ (1 << bit)]);
    }
    Matrix::from_columns(&columns)
}

fn equal_up_to_phase(a: &Matrix, b: &Matrix) {
    let overlap = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.conj() * y)
        .sum::<Complex64>();
    let phase = overlap / overlap.norm();
    assert!((a * phase - b).norm() < 1e-10);
}

fn select_bit(matrix: &Matrix, bit: usize, value: bool) -> Matrix {
    let columns: Vec<_> = (0..matrix.ncols())
        .filter(|x| (x & (1 << bit) != 0) == value)
        .map(|x| matrix.column(x).into_owned())
        .collect();
    Matrix::from_columns(&columns)
}

fn right_inverse(matrix: &mut Matrix, gate: CoordinateGate) {
    match gate {
        CoordinateGate::H(bit) => {
            for x in 0..matrix.ncols() {
                if x & (1 << bit) == 0 {
                    let y = x ^ (1 << bit);
                    let a = matrix.column(x).into_owned();
                    let b = matrix.column(y).into_owned();
                    let factor = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
                    matrix.set_column(x, &((&a + &b) * factor));
                    matrix.set_column(y, &((&a - &b) * factor));
                }
            }
        }
        CoordinateGate::Sdg(bit) => {
            for x in 0..matrix.ncols() {
                if x & (1 << bit) != 0 {
                    for row in 0..matrix.nrows() {
                        matrix[(row, x)] *= Complex64::new(0.0, 1.0);
                    }
                }
            }
        }
        CoordinateGate::Cx(control, target) => {
            for x in 0..matrix.ncols() {
                if x & (1 << control) != 0 && x & (1 << target) == 0 {
                    matrix.swap_columns(x, x ^ (1 << target));
                }
            }
        }
    }
}

#[test]
fn random_coordinate_changes_preserve_dense_bases_and_canonical_relations() {
    let mut rng = PecosRng::seed_from_u64(748);
    let mut counts = [0; 3];
    for n in 1..=5 {
        for trial in 0..30 {
            let mut tableau = SparseStabY::with_seed(n, 15).with_destab_sign_tracking();
            for _ in 0..30 {
                let q = rng.next_u64() as usize % n;
                let r = (q + 1) % n;
                match rng.next_u64() % 5 {
                    0 => {
                        tableau.h(&[QubitId(q)]);
                    }
                    1 => {
                        tableau.sz(&[QubitId(q)]);
                    }
                    2 => {
                        tableau.y(&[QubitId(q)]);
                    }
                    3 if n > 1 => {
                        tableau.cx(&[(QubitId(q), QubitId(r))]);
                    }
                    _ => {
                        tableau.x(&[QubitId(q)]);
                    }
                }
            }
            let mut active: Vec<_> = (0..trial % (n + 1)).rev().collect();
            let pauli: Vec<_> = (0..n)
                .filter_map(|q| match rng.next_u64() % 4 {
                    0 => None,
                    1 => Some((q, PauliKindForDecomp::X)),
                    2 => Some((q, PauliKindForDecomp::Y)),
                    _ => Some((q, PauliKindForDecomp::Z)),
                })
                .collect();
            let negative = rng.next_bool_fast();
            invariants(&tableau);
            let parts = decompose(&tableau, &active, &pauli, negative);
            let p = physical(n, &pauli, negative);
            // Verify the D-then-S order, including odd F/G overlap and signs.
            let mut product = Matrix::identity(1 << n, 1 << n);
            for row in parts
                .active_flips
                .iter()
                .map(|&b| active[b])
                .chain(parts.dormant_flips.iter().copied())
            {
                product *= row_matrix(tableau.destabs(), row);
            }
            for row in parts
                .active_signs
                .iter()
                .map(|&b| active[b])
                .chain(parts.dormant_signs.iter().copied())
            {
                product *= row_matrix(tableau.stabs(), row);
            }
            assert!((&p - product * parts.phase).norm() < 1e-12);
            let old = basis(&tableau, &active);
            match parts.measurement_case() {
                MeasurementCase::Random => {
                    counts[0] += 1;
                    let mut promoted = tableau.clone();
                    let mut promoted_active = active.clone();
                    promote(&mut promoted, &mut promoted_active, &pauli, negative);
                    invariants(&promoted);
                    let expanded = basis(&promoted, &promoted_active);
                    equal_up_to_phase(&select_bit(&expanded, active.len(), false), &old);
                    equal_up_to_phase(&select_bit(&expanded, active.len(), true), &(&p * &old));
                    // Compare both halves together, so their relative phase is tested.
                    let columns: Vec<_> = old
                        .column_iter()
                        .map(nalgebra::Matrix::into_owned)
                        .chain((&p * &old).column_iter().map(nalgebra::Matrix::into_owned))
                        .collect();
                    equal_up_to_phase(&expanded, &Matrix::from_columns(&columns));
                    for outcome in [false, true] {
                        let mut measured = tableau.clone();
                        measure_random(&mut measured, &active, &pauli, negative, outcome);
                        invariants(&measured);
                        let signed = &p * Complex64::new(if outcome { -1.0 } else { 1.0 }, 0.0);
                        let projected = (&old + signed * &old)
                            * Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
                        equal_up_to_phase(&basis(&measured, &active), &projected);
                    }
                }
                MeasurementCase::Deterministic => {
                    counts[1] += 1;
                    assert!((&p * &old - &old * parts.phase).norm() < 1e-12);
                }
                MeasurementCase::Active => {
                    counts[2] += 1;
                    let plan = measurement_basis(&mut tableau, &active, &pauli, negative);
                    invariants(&tableau);
                    let mut transformed = old;
                    for gate in plan.gates {
                        right_inverse(&mut transformed, gate);
                    }
                    equal_up_to_phase(&basis(&tableau, &active), &transformed);
                    let signed_stab = row_matrix(tableau.stabs(), active[plan.pivot_bit])
                        * Complex64::new(if plan.negative { -1.0 } else { 1.0 }, 0.0);
                    assert!((&p - signed_stab).norm() < 1e-12);
                    let value = rng.next_bool_fast();
                    demote(&mut tableau, &mut active, plan.pivot_bit, value);
                    invariants(&tableau);
                    equal_up_to_phase(
                        &basis(&tableau, &active),
                        &select_bit(&transformed, plan.pivot_bit, value),
                    );
                }
            }
        }
    }
    assert!(counts.into_iter().all(|count| count > 10), "{counts:?}");
}
