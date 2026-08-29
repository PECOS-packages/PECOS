// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! Phase-exact agreement between the canonical named-gate table and the
//! independent state-vector/GPU single-qubit tables.

use num_complex::Complex64;
use pecos_core::gate_type::{GateType, NAMED_SINGLE_QUBIT_GATES};
use pecos_core::{Angle64, Clifford, QubitId};
use pecos_gpu_sims::gates as gpu_gates;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA, StateVecSoA32};

type Matrix2 = [[Complex64; 2]; 2];

// f64 products through T^8 accumulate a few ulps. This tolerance gives ample
// roundoff headroom without coming remotely close to accepting a phase change.
const F64_TOLERANCE: f64 = 1e-12;
// The f32 tables retain roughly seven decimal digits and include rounded
// 1/sqrt(2) entries, so they need a tolerance near f32 machine precision.
const F32_TOLERANCE: f64 = 1e-6;

fn canonical_matrix(gate: GateType) -> Matrix2 {
    let matrix = gate
        .canonical_1q_matrix()
        .expect("named single-qubit gate must have a canonical matrix");
    [
        [
            Complex64::new(matrix[0], matrix[1]),
            Complex64::new(matrix[2], matrix[3]),
        ],
        [
            Complex64::new(matrix[4], matrix[5]),
            Complex64::new(matrix[6], matrix[7]),
        ],
    ]
}

fn canonical_clifford_matrix(gate: Clifford) -> Matrix2 {
    let matrix = gate
        .canonical_1q_matrix()
        .expect("single-qubit Clifford must have a canonical matrix");
    [
        [
            Complex64::new(matrix[0], matrix[1]),
            Complex64::new(matrix[2], matrix[3]),
        ],
        [
            Complex64::new(matrix[4], matrix[5]),
            Complex64::new(matrix[6], matrix[7]),
        ],
    ]
}

fn max_entrywise_error(lhs: &Matrix2, rhs: &Matrix2) -> f64 {
    lhs.iter()
        .flatten()
        .zip(rhs.iter().flatten())
        .map(|(actual, expected)| (actual - expected).norm())
        .fold(0.0, f64::max)
}

fn matrix_times_state(matrix: &Matrix2, state: &[Complex64; 2]) -> [Complex64; 2] {
    [
        matrix[0][0] * state[0] + matrix[0][1] * state[1],
        matrix[1][0] * state[0] + matrix[1][1] * state[1],
    ]
}

fn max_state_error(lhs: &[Complex64; 2], rhs: &[Complex64; 2]) -> f64 {
    lhs.iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).norm())
        .fold(0.0, f64::max)
}

fn assert_phase_exact_state_eq(
    label: &str,
    actual: &[Complex64; 2],
    expected: &[Complex64; 2],
    tolerance: f64,
) {
    let error = max_state_error(actual, expected);
    assert!(
        error <= tolerance,
        "{label}: max state error {error:e} exceeds phase-exact tolerance {tolerance:e}; \
         actual={actual:?}, expected={expected:?}"
    );

    let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let phase_mutant = expected.map(|amplitude| amplitude * phase);
    assert!(
        max_state_error(actual, &phase_mutant) > tolerance,
        "{label}: exp(i*pi/4) matrix-phase mutant escaped the kernel guard"
    );
}

fn assert_phase_exact_matrix_eq(label: &str, actual: &Matrix2, expected: &Matrix2, tolerance: f64) {
    let error = max_entrywise_error(actual, expected);
    assert!(
        error <= tolerance,
        "{label}: max entrywise error {error:e} exceeds phase-exact tolerance {tolerance:e}; \
         actual={actual:?}, expected={expected:?}"
    );

    // Mutation check every comparison: a projectively equivalent matrix with
    // an exp(i*pi/4) scalar must be rejected by this phase-sensitive guard.
    let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let mut phase_mutant = *actual;
    for entry in phase_mutant.iter_mut().flatten() {
        *entry *= phase;
    }
    assert!(
        max_entrywise_error(&phase_mutant, expected) > tolerance,
        "{label}: exp(i*pi/4) global-phase mutant escaped the exactness guard"
    );
}

fn apply_f64(sim: &mut StateVecSoA, gate: GateType) {
    let qubit = [QubitId(0)];
    match gate {
        GateType::I => sim.identity(&qubit),
        GateType::X => sim.x(&qubit),
        GateType::Y => sim.y(&qubit),
        GateType::Z => sim.z(&qubit),
        GateType::H => sim.h(&qubit),
        GateType::F => sim.f(&qubit),
        GateType::Fdg => sim.fdg(&qubit),
        GateType::SX => sim.sx(&qubit),
        GateType::SXdg => sim.sxdg(&qubit),
        GateType::SY => sim.sy(&qubit),
        GateType::SYdg => sim.sydg(&qubit),
        GateType::SZ => sim.sz(&qubit),
        GateType::SZdg => sim.szdg(&qubit),
        GateType::T => sim.t(&qubit),
        GateType::Tdg => sim.tdg(&qubit),
        other => panic!("unsupported f64 single-qubit table gate {other:?}"),
    };
}

fn f64_simulator_matrix(gate: GateType) -> Matrix2 {
    let mut matrix = [[Complex64::new(0.0, 0.0); 2]; 2];
    for basis in 0..2 {
        let mut sim = StateVecSoA::new(1);
        sim.prepare_computational_basis(basis);
        apply_f64(&mut sim, gate);
        for (row, matrix_row) in matrix.iter_mut().enumerate() {
            matrix_row[basis] = sim.get_amplitude(row);
        }
    }
    matrix
}

fn apply_f32(sim: &mut StateVecSoA32, gate: GateType) {
    let qubit = [QubitId(0)];
    match gate {
        GateType::I => sim.identity(&qubit),
        GateType::X => sim.x(&qubit),
        GateType::Y => sim.y(&qubit),
        GateType::Z => sim.z(&qubit),
        GateType::H => sim.h(&qubit),
        GateType::F => sim.f(&qubit),
        GateType::Fdg => sim.fdg(&qubit),
        GateType::SX => sim.sx(&qubit),
        GateType::SXdg => sim.sxdg(&qubit),
        GateType::SY => sim.sy(&qubit),
        GateType::SYdg => sim.sydg(&qubit),
        GateType::SZ => sim.sz(&qubit),
        GateType::SZdg => sim.szdg(&qubit),
        GateType::T => sim.t(&qubit),
        GateType::Tdg => sim.tdg(&qubit),
        other => panic!("unsupported f32 single-qubit table gate {other:?}"),
    };
}

fn f32_simulator_matrix(gate: GateType) -> Matrix2 {
    let mut matrix = [[Complex64::new(0.0, 0.0); 2]; 2];
    for basis in 0..2 {
        let mut sim = StateVecSoA32::new(1);
        if basis == 1 {
            sim.x(&[QubitId(0)]);
            sim.flush();
        }
        apply_f32(&mut sim, gate);
        for (row, matrix_row) in matrix.iter_mut().enumerate() {
            let amplitude = sim.get_amplitude(row);
            matrix_row[basis] = Complex64::new(f64::from(amplitude.re), f64::from(amplitude.im));
        }
    }
    matrix
}

fn gpu_matrix(gate: GateType) -> Matrix2 {
    let entries = match gate {
        GateType::I => gpu_gates::I,
        GateType::X => gpu_gates::X,
        GateType::Y => gpu_gates::Y,
        GateType::Z => gpu_gates::Z,
        GateType::H => gpu_gates::H,
        GateType::SX => gpu_gates::SX,
        GateType::SXdg => gpu_gates::SXDG,
        GateType::SY => gpu_gates::SY,
        GateType::SYdg => gpu_gates::SYDG,
        GateType::SZ => gpu_gates::S,
        GateType::SZdg => gpu_gates::SDG,
        GateType::T => gpu_gates::T,
        GateType::Tdg => gpu_gates::TDG,
        GateType::F => gpu_gates::F,
        GateType::Fdg => gpu_gates::FDG,
        other => panic!("unsupported GPU single-qubit table gate {other:?}"),
    };
    [
        [
            Complex64::new(f64::from(entries[0]), f64::from(entries[1])),
            Complex64::new(f64::from(entries[2]), f64::from(entries[3])),
        ],
        [
            Complex64::new(f64::from(entries[4]), f64::from(entries[5])),
            Complex64::new(f64::from(entries[6]), f64::from(entries[7])),
        ],
    ]
}

fn apply_clifford_f64(sim: &mut StateVecSoA, gate: Clifford) {
    let qubit = [QubitId(0)];
    match gate {
        Clifford::H => sim.h(&qubit),
        Clifford::H2 => sim.h2(&qubit),
        Clifford::H3 => sim.h3(&qubit),
        Clifford::H4 => sim.h4(&qubit),
        Clifford::H5 => sim.h5(&qubit),
        Clifford::H6 => sim.h6(&qubit),
        Clifford::F => sim.f(&qubit),
        Clifford::Fdg => sim.fdg(&qubit),
        Clifford::F2 => sim.f2(&qubit),
        Clifford::F2dg => sim.f2dg(&qubit),
        Clifford::F3 => sim.f3(&qubit),
        Clifford::F3dg => sim.f3dg(&qubit),
        Clifford::F4 => sim.f4(&qubit),
        Clifford::F4dg => sim.f4dg(&qubit),
        _ => panic!("unsupported extended f64 Clifford {gate}"),
    };
}

fn apply_clifford_f32(sim: &mut StateVecSoA32, gate: Clifford) {
    let qubit = [QubitId(0)];
    match gate {
        Clifford::H => sim.h(&qubit),
        Clifford::H2 => sim.h2(&qubit),
        Clifford::H3 => sim.h3(&qubit),
        Clifford::H4 => sim.h4(&qubit),
        Clifford::H5 => sim.h5(&qubit),
        Clifford::H6 => sim.h6(&qubit),
        Clifford::F => sim.f(&qubit),
        Clifford::Fdg => sim.fdg(&qubit),
        Clifford::F2 => sim.f2(&qubit),
        Clifford::F2dg => sim.f2dg(&qubit),
        Clifford::F3 => sim.f3(&qubit),
        Clifford::F3dg => sim.f3dg(&qubit),
        Clifford::F4 => sim.f4(&qubit),
        Clifford::F4dg => sim.f4dg(&qubit),
        _ => panic!("unsupported extended f32 Clifford {gate}"),
    };
}

fn clifford_f64_matrix(gate: Clifford) -> Matrix2 {
    let mut matrix = [[Complex64::new(0.0, 0.0); 2]; 2];
    for basis in 0..2 {
        let mut sim = StateVecSoA::new(1);
        sim.prepare_computational_basis(basis);
        apply_clifford_f64(&mut sim, gate);
        for (row, matrix_row) in matrix.iter_mut().enumerate() {
            matrix_row[basis] = sim.get_amplitude(row);
        }
    }
    matrix
}

fn clifford_f32_matrix(gate: Clifford) -> Matrix2 {
    let mut matrix = [[Complex64::new(0.0, 0.0); 2]; 2];
    for basis in 0..2 {
        let mut sim = StateVecSoA32::new(1);
        if basis == 1 {
            sim.x(&[QubitId(0)]);
            sim.flush();
        }
        apply_clifford_f32(&mut sim, gate);
        for (row, matrix_row) in matrix.iter_mut().enumerate() {
            let amplitude = sim.get_amplitude(row);
            matrix_row[basis] = Complex64::new(f64::from(amplitude.re), f64::from(amplitude.im));
        }
    }
    matrix
}

fn gpu_clifford_matrix(gate: Clifford) -> Matrix2 {
    let entries = match gate {
        Clifford::H => gpu_gates::H,
        Clifford::H2 => gpu_gates::H2,
        Clifford::H3 => gpu_gates::H3,
        Clifford::H4 => gpu_gates::H4,
        Clifford::H5 => gpu_gates::H5,
        Clifford::H6 => gpu_gates::H6,
        Clifford::F => gpu_gates::F,
        Clifford::Fdg => gpu_gates::FDG,
        Clifford::F2 => gpu_gates::F2,
        Clifford::F2dg => gpu_gates::F2DG,
        Clifford::F3 => gpu_gates::F3,
        Clifford::F3dg => gpu_gates::F3DG,
        Clifford::F4 => gpu_gates::F4,
        Clifford::F4dg => gpu_gates::F4DG,
        _ => panic!("unsupported extended GPU Clifford {gate}"),
    };
    [
        [
            Complex64::new(f64::from(entries[0]), f64::from(entries[1])),
            Complex64::new(f64::from(entries[2]), f64::from(entries[3])),
        ],
        [
            Complex64::new(f64::from(entries[4]), f64::from(entries[5])),
            Complex64::new(f64::from(entries[6]), f64::from(entries[7])),
        ],
    ]
}

fn prepare_f64_kernel_input() -> (StateVecSoA, [Complex64; 2]) {
    let mut sim = StateVecSoA::new(1);
    let qubit = [QubitId(0)];
    sim.ry(Angle64::from_radians(0.731), &qubit)
        .rz(Angle64::from_radians(-0.417), &qubit);
    sim.flush();
    let input = [sim.get_amplitude(0), sim.get_amplitude(1)];
    sim.set_fusion(false);
    (sim, input)
}

fn prepare_f32_kernel_input() -> (StateVecSoA32, [Complex64; 2]) {
    let mut sim = StateVecSoA32::new(1);
    let qubit = [QubitId(0)];
    sim.ry(Angle64::from_radians(0.731), &qubit)
        .rz(Angle64::from_radians(-0.417), &qubit);
    sim.flush();
    let first = sim.get_amplitude(0);
    let second = sim.get_amplitude(1);
    let input = [
        Complex64::new(f64::from(first.re), f64::from(first.im)),
        Complex64::new(f64::from(second.re), f64::from(second.im)),
    ];
    sim.set_fusion(false);
    (sim, input)
}

#[test]
fn handwritten_single_qubit_kernels_match_canonical_phase_exactly() {
    // Both amplitudes of this fixed input are nonzero and complex, so every
    // matrix entry contributes. The f64 tolerance covers a few arithmetic ulps;
    // the f32 tolerance is near f32 machine precision. A pi/4 phase error is
    // roughly six orders of magnitude larger than even the f32 tolerance.
    for gate in [
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::H,
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
        GateType::T,
        GateType::Tdg,
    ] {
        let (mut sim, input) = prepare_f64_kernel_input();
        apply_f64(&mut sim, gate);
        let actual = [sim.get_amplitude(0), sim.get_amplitude(1)];
        let expected = matrix_times_state(&canonical_matrix(gate), &input);
        assert_phase_exact_state_eq(
            &format!("StateVecSoA arithmetic {gate:?}"),
            &actual,
            &expected,
            F64_TOLERANCE,
        );
    }

    for gate in [
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::H,
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
    ] {
        let (mut sim, input) = prepare_f32_kernel_input();
        apply_f32(&mut sim, gate);
        let first = sim.get_amplitude(0);
        let second = sim.get_amplitude(1);
        let actual = [
            Complex64::new(f64::from(first.re), f64::from(first.im)),
            Complex64::new(f64::from(second.re), f64::from(second.im)),
        ];
        let expected = matrix_times_state(&canonical_matrix(gate), &input);
        assert_phase_exact_state_eq(
            &format!("StateVecSoA32 arithmetic {gate:?}"),
            &actual,
            &expected,
            F32_TOLERANCE,
        );
    }

    for (label, gate) in [
        ("StateVecSoA arithmetic H3", Clifford::H3),
        ("StateVecSoA arithmetic H4", Clifford::H4),
    ] {
        let (mut sim, input) = prepare_f64_kernel_input();
        let qubit = [QubitId(0)];
        match gate {
            Clifford::H3 => sim.h3(&qubit),
            Clifford::H4 => sim.h4(&qubit),
            _ => unreachable!("only H3 and H4 have arithmetic kernels"),
        };
        let actual = [sim.get_amplitude(0), sim.get_amplitude(1)];
        let expected = matrix_times_state(&canonical_clifford_matrix(gate), &input);
        assert_phase_exact_state_eq(label, &actual, &expected, F64_TOLERANCE);
    }
}

#[test]
fn single_qubit_simulator_tables_match_canonical_phase_exactly() {
    for gate in NAMED_SINGLE_QUBIT_GATES {
        let canonical = canonical_matrix(gate);
        assert_phase_exact_matrix_eq(
            &format!("StateVecSoA::{gate:?}"),
            &f64_simulator_matrix(gate),
            &canonical,
            F64_TOLERANCE,
        );
        assert_phase_exact_matrix_eq(
            &format!("StateVecSoA32::{gate:?}"),
            &f32_simulator_matrix(gate),
            &canonical,
            F32_TOLERANCE,
        );
        let gpu = gpu_matrix(gate);
        assert_phase_exact_matrix_eq(
            &format!("gpu_gates::{gate:?}"),
            &gpu,
            &canonical,
            F32_TOLERANCE,
        );
    }

    // The additional H/F-family matrices are derived from the canonical
    // Clifford table in f64, f32 and GPU precision.
    for gate in [
        Clifford::H,
        Clifford::H2,
        Clifford::H3,
        Clifford::H4,
        Clifford::H5,
        Clifford::H6,
        Clifford::F,
        Clifford::Fdg,
        Clifford::F2,
        Clifford::F2dg,
        Clifford::F3,
        Clifford::F3dg,
        Clifford::F4,
        Clifford::F4dg,
    ] {
        let expected = canonical_clifford_matrix(gate);
        assert_phase_exact_matrix_eq(
            &format!("StateVecSoA::{gate}"),
            &clifford_f64_matrix(gate),
            &expected,
            F64_TOLERANCE,
        );
        assert_phase_exact_matrix_eq(
            &format!("StateVecSoA32::{gate}"),
            &clifford_f32_matrix(gate),
            &expected,
            F32_TOLERANCE,
        );
        assert_phase_exact_matrix_eq(
            &format!("gpu_gates::{gate}"),
            &gpu_clifford_matrix(gate),
            &expected,
            F32_TOLERANCE,
        );
    }
}
