// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at https://www.apache.org/licenses/LICENSE-2.0

//! Phase-exact named-gate checks for amplitude-carrying CPU simulators.

use num_complex::Complex64;
use pecos_core::gate_type::{
    GateType, NAMED_SINGLE_QUBIT_GATES, NAMED_TWO_QUBIT_ROOT_GATES, TwoQubitGateMatrix,
};
use pecos_core::{Angle64, Clifford, QubitId};
use pecos_simulators::state_vector_test_utils::StateVectorSimulator;
use pecos_simulators::{
    ArbitraryRotationGateable, CHForm, CliffordGateable, SparseStateVecAoS, SparseStateVecSoA,
    StateVecAoS, StateVecSoA, StateVecSoA32,
};

const F64_TOLERANCE: f64 = 1e-12;
const F32_TOLERANCE: f64 = 1e-6;

fn canonical_times_state(gate: GateType, state: &[Complex64; 2]) -> [Complex64; 2] {
    let matrix = gate
        .canonical_1q_matrix()
        .expect("named single-qubit gate must have a canonical matrix");
    let a = Complex64::new(matrix[0], matrix[1]);
    let b = Complex64::new(matrix[2], matrix[3]);
    let c = Complex64::new(matrix[4], matrix[5]);
    let d = Complex64::new(matrix[6], matrix[7]);
    [a * state[0] + b * state[1], c * state[0] + d * state[1]]
}

fn canonical_clifford_times_state(gate: Clifford, state: &[Complex64; 2]) -> [Complex64; 2] {
    let matrix = gate
        .canonical_1q_matrix()
        .expect("single-qubit Clifford must have a canonical matrix");
    let a = Complex64::new(matrix[0], matrix[1]);
    let b = Complex64::new(matrix[2], matrix[3]);
    let c = Complex64::new(matrix[4], matrix[5]);
    let d = Complex64::new(matrix[6], matrix[7]);
    [a * state[0] + b * state[1], c * state[0] + d * state[1]]
}

fn max_error(actual: &[Complex64; 2], expected: &[Complex64; 2]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).norm())
        .fold(0.0, f64::max)
}

fn max_state_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).norm())
        .fold(0.0, f64::max)
}

fn assert_phase_exact(
    simulator: &str,
    gate: &str,
    actual: &[Complex64; 2],
    expected: &[Complex64; 2],
    tolerance: f64,
) {
    let error = max_error(actual, expected);
    assert!(
        error <= tolerance,
        "{simulator}::{gate} differs from the canonical matrix by {error:e}; \
         actual={actual:?}, expected={expected:?}"
    );

    let root_half = std::f64::consts::FRAC_1_SQRT_2;
    let phase = Complex64::new(root_half, root_half);
    let mutant = expected.map(|amplitude| amplitude * phase);
    assert!(
        max_error(actual, &mutant) > tolerance,
        "{simulator}::{gate} accepted an exp(i*pi/4) phase mutation"
    );
}

fn assert_phase_exact_state(
    simulator: &str,
    gate: &str,
    actual: &[Complex64],
    expected: &[Complex64],
    tolerance: f64,
) {
    let error = max_state_error(actual, expected);
    assert!(
        error <= tolerance,
        "{simulator}::{gate} differs from the canonical matrix by {error:e}; \
         actual={actual:?}, expected={expected:?}"
    );

    let root_half = std::f64::consts::FRAC_1_SQRT_2;
    let phase = Complex64::new(root_half, root_half);
    let mutant: Vec<_> = expected.iter().map(|amplitude| amplitude * phase).collect();
    assert!(
        max_state_error(actual, &mutant) > tolerance,
        "{simulator}::{gate} accepted an exp(i*pi/4) phase mutation"
    );
}

fn apply_canonical_matrix(
    state: &mut [Complex64],
    matrix: pecos_core::gate_type::SingleQubitGateMatrix,
    qubit: QubitId,
) {
    let a = Complex64::new(matrix[0], matrix[1]);
    let b = Complex64::new(matrix[2], matrix[3]);
    let c = Complex64::new(matrix[4], matrix[5]);
    let d = Complex64::new(matrix[6], matrix[7]);
    let step = 1 << qubit.index();
    for block in state.chunks_exact_mut(step * 2) {
        let (zero, one) = block.split_at_mut(step);
        for (zero, one) in zero.iter_mut().zip(one) {
            let input_zero = *zero;
            let input_one = *one;
            *zero = a * input_zero + b * input_one;
            *one = c * input_zero + d * input_one;
        }
    }
}

fn apply_canonical_two_qubit_matrix(
    state: &mut [Complex64],
    matrix: TwoQubitGateMatrix,
    qubit1: QubitId,
    qubit2: QubitId,
) {
    let mask1 = 1 << qubit1.index();
    let mask2 = 1 << qubit2.index();
    for base in 0..state.len() {
        if base & (mask1 | mask2) != 0 {
            continue;
        }
        let indices = [base, base | mask2, base | mask1, base | mask1 | mask2];
        let input = indices.map(|index| state[index]);
        for row in 0..4 {
            let mut output = Complex64::new(0.0, 0.0);
            for (column, amplitude) in input.into_iter().enumerate() {
                let entry = 2 * (4 * row + column);
                output += Complex64::new(matrix[entry], matrix[entry + 1]) * amplitude;
            }
            state[indices[row]] = output;
        }
    }
}

fn apply_clifford_gate<S: CliffordGateable>(sim: &mut S, gate: Clifford, qubits: &[QubitId]) {
    match gate {
        Clifford::I => sim.identity(qubits),
        Clifford::X => sim.x(qubits),
        Clifford::Y => sim.y(qubits),
        Clifford::Z => sim.z(qubits),
        Clifford::H => sim.h(qubits),
        Clifford::H2 => sim.h2(qubits),
        Clifford::H3 => sim.h3(qubits),
        Clifford::H4 => sim.h4(qubits),
        Clifford::H5 => sim.h5(qubits),
        Clifford::H6 => sim.h6(qubits),
        Clifford::SX => sim.sx(qubits),
        Clifford::SXdg => sim.sxdg(qubits),
        Clifford::SY => sim.sy(qubits),
        Clifford::SYdg => sim.sydg(qubits),
        Clifford::SZ => sim.sz(qubits),
        Clifford::SZdg => sim.szdg(qubits),
        Clifford::F => sim.f(qubits),
        Clifford::Fdg => sim.fdg(qubits),
        Clifford::F2 => sim.f2(qubits),
        Clifford::F2dg => sim.f2dg(qubits),
        Clifford::F3 => sim.f3(qubits),
        Clifford::F3dg => sim.f3dg(qubits),
        Clifford::F4 => sim.f4(qubits),
        Clifford::F4dg => sim.f4dg(qubits),
        _ => panic!("unsupported Clifford gate {gate}"),
    };
}

fn apply_named_clifford_gate<S: CliffordGateable>(sim: &mut S, gate: GateType, qubits: &[QubitId]) {
    match gate {
        GateType::I => sim.identity(qubits),
        GateType::X => sim.x(qubits),
        GateType::Y => sim.y(qubits),
        GateType::Z => sim.z(qubits),
        GateType::H => sim.h(qubits),
        GateType::F => sim.f(qubits),
        GateType::Fdg => sim.fdg(qubits),
        GateType::SX => sim.sx(qubits),
        GateType::SXdg => sim.sxdg(qubits),
        GateType::SY => sim.sy(qubits),
        GateType::SYdg => sim.sydg(qubits),
        GateType::SZ => sim.sz(qubits),
        GateType::SZdg => sim.szdg(qubits),
        other => panic!("unsupported named single-qubit gate {other:?}"),
    };
}

fn apply_named_gate<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    gate: GateType,
    qubits: &[QubitId],
) {
    match gate {
        GateType::T => {
            sim.t(qubits);
        }
        GateType::Tdg => {
            sim.tdg(qubits);
        }
        _ => apply_named_clifford_gate(sim, gate, qubits),
    }
}

fn apply_named_two_qubit_gate<S: CliffordGateable>(
    sim: &mut S,
    gate: GateType,
    pairs: &[(QubitId, QubitId)],
) {
    match gate {
        GateType::SXX => sim.sxx(pairs),
        GateType::SXXdg => sim.sxxdg(pairs),
        GateType::SYY => sim.syy(pairs),
        GateType::SYYdg => sim.syydg(pairs),
        GateType::SZZ => sim.szz(pairs),
        GateType::SZZdg => sim.szzdg(pairs),
        other => panic!("unsupported named two-qubit root {other:?}"),
    };
}

fn check_state_vector<S>(simulator: &str, tolerance: f64)
where
    S: StateVectorSimulator + ArbitraryRotationGateable,
{
    for gate in NAMED_SINGLE_QUBIT_GATES {
        let mut sim = S::with_seed(1, 42);
        let qubit = [QubitId(0)];
        sim.ry(Angle64::from_radians(0.731), &qubit)
            .rz(Angle64::from_radians(-0.417), &qubit);
        let input = [sim.get_amplitude(0), sim.get_amplitude(1)];
        apply_named_gate(&mut sim, gate, &qubit);
        let actual = [sim.get_amplitude(0), sim.get_amplitude(1)];
        let expected = canonical_times_state(gate, &input);
        assert_phase_exact(
            simulator,
            &format!("{gate:?}"),
            &actual,
            &expected,
            tolerance,
        );
    }
}

fn check_state_vector_cliffords<S>(simulator: &str, tolerance: f64)
where
    S: StateVectorSimulator + ArbitraryRotationGateable,
{
    for &gate in Clifford::all_1q() {
        let mut sim = S::with_seed(1, 42);
        let qubit = [QubitId(0)];
        sim.ry(Angle64::from_radians(0.731), &qubit)
            .rz(Angle64::from_radians(-0.417), &qubit);
        let input = [sim.get_amplitude(0), sim.get_amplitude(1)];
        apply_clifford_gate(&mut sim, gate, &qubit);
        let actual = [sim.get_amplitude(0), sim.get_amplitude(1)];
        let expected = canonical_clifford_times_state(gate, &input);
        assert_phase_exact(simulator, &gate.to_string(), &actual, &expected, tolerance);
    }
}

fn prepare_three_qubit_state<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    sim.ry(Angle64::from_radians(0.731), &[QubitId(0)])
        .rz(Angle64::from_radians(-0.417), &[QubitId(0)])
        .rx(Angle64::from_radians(-0.293), &[QubitId(1)])
        .ry(Angle64::from_radians(0.619), &[QubitId(2)])
        .cx(&[(QubitId(0), QubitId(1))])
        .cz(&[(QubitId(1), QubitId(2))]);
}

fn state_vector<S: StateVectorSimulator>(sim: &mut S) -> Vec<Complex64> {
    (0..(1 << sim.num_qubits()))
        .map(|basis_state| sim.get_amplitude(basis_state))
        .collect()
}

fn prepare_four_qubit_state<S: StateVectorSimulator + ArbitraryRotationGateable>(sim: &mut S) {
    sim.ry(Angle64::from_radians(0.731), &[QubitId(0)])
        .rz(Angle64::from_radians(-0.417), &[QubitId(0)])
        .rx(Angle64::from_radians(-0.293), &[QubitId(1)])
        .ry(Angle64::from_radians(0.619), &[QubitId(2)])
        .rz(Angle64::from_radians(0.383), &[QubitId(3)])
        .cx(&[(QubitId(0), QubitId(2)), (QubitId(1), QubitId(3))])
        .cz(&[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))]);
}

fn check_two_qubit_root_kernels<S, F>(simulator: &str, tolerance: f64, configure: F)
where
    S: StateVectorSimulator + ArbitraryRotationGateable,
    F: Fn(&mut S) + Copy,
{
    for gate in NAMED_TWO_QUBIT_ROOT_GATES {
        let matrix = gate
            .canonical_2q_matrix()
            .expect("named two-qubit root must have a canonical matrix");
        for (case, pairs) in [
            ("scalar pair", vec![(QubitId(0), QubitId(1))]),
            ("SIMD pair", vec![(QubitId(2), QubitId(3))]),
            (
                "batched pairs",
                vec![(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))],
            ),
        ] {
            let mut sim = S::with_seed(4, 42);
            configure(&mut sim);
            prepare_four_qubit_state(&mut sim);
            let mut expected = state_vector(&mut sim);
            for &(q1, q2) in &pairs {
                apply_canonical_two_qubit_matrix(&mut expected, matrix, q1, q2);
            }

            apply_named_two_qubit_gate(&mut sim, gate, &pairs);
            let actual = state_vector(&mut sim);
            assert_phase_exact_state(
                simulator,
                &format!("{gate:?} {case}"),
                &actual,
                &expected,
                tolerance,
            );
        }
    }
}

fn check_three_qubit_multi_target_calls<S, F>(simulator: &str, tolerance: f64, configure: F)
where
    S: StateVectorSimulator + ArbitraryRotationGateable,
    F: Fn(&mut S) + Copy,
{
    let targets = [QubitId(0), QubitId(2)];
    for gate in NAMED_SINGLE_QUBIT_GATES {
        let mut sim = S::with_seed(3, 42);
        configure(&mut sim);
        prepare_three_qubit_state(&mut sim);
        let mut expected = state_vector(&mut sim);
        let matrix = gate
            .canonical_1q_matrix()
            .expect("named single-qubit gate must have a canonical matrix");
        for &target in &targets {
            apply_canonical_matrix(&mut expected, matrix, target);
        }
        apply_named_gate(&mut sim, gate, &targets);
        let actual = state_vector(&mut sim);
        assert_phase_exact_state(
            simulator,
            &format!("{gate:?} on q0,q2"),
            &actual,
            &expected,
            tolerance,
        );
    }

    for &gate in Clifford::all_1q() {
        let mut sim = S::with_seed(3, 42);
        configure(&mut sim);
        prepare_three_qubit_state(&mut sim);
        let mut expected = state_vector(&mut sim);
        let matrix = gate
            .canonical_1q_matrix()
            .expect("single-qubit Clifford must have a canonical matrix");
        for &target in &targets {
            apply_canonical_matrix(&mut expected, matrix, target);
        }
        apply_clifford_gate(&mut sim, gate, &targets);
        let actual = state_vector(&mut sim);
        assert_phase_exact_state(
            simulator,
            &format!("{gate} on q0,q2"),
            &actual,
            &expected,
            tolerance,
        );
    }
}

#[test]
fn phase_carrying_cpu_simulators_match_named_gate_conventions() {
    check_state_vector::<StateVecSoA>("StateVecSoA", F64_TOLERANCE);
    check_state_vector::<StateVecAoS>("StateVecAoS", F64_TOLERANCE);
    check_state_vector::<SparseStateVecSoA>("SparseStateVecSoA", F64_TOLERANCE);
    check_state_vector::<SparseStateVecAoS>("SparseStateVecAoS", F64_TOLERANCE);
    check_state_vector::<StateVecSoA32>("StateVecSoA32", F32_TOLERANCE);

    check_state_vector_cliffords::<StateVecSoA>("StateVecSoA", F64_TOLERANCE);
    check_state_vector_cliffords::<StateVecAoS>("StateVecAoS", F64_TOLERANCE);
    check_state_vector_cliffords::<SparseStateVecSoA>("SparseStateVecSoA", F64_TOLERANCE);
    check_state_vector_cliffords::<SparseStateVecAoS>("SparseStateVecAoS", F64_TOLERANCE);
    check_state_vector_cliffords::<StateVecSoA32>("StateVecSoA32", F32_TOLERANCE);
}

#[test]
fn phase_carrying_cpu_simulators_match_multi_target_conventions_on_three_qubits() {
    check_three_qubit_multi_target_calls::<StateVecSoA, _>("StateVecSoA", F64_TOLERANCE, |_| {});
    check_three_qubit_multi_target_calls::<StateVecAoS, _>("StateVecAoS", F64_TOLERANCE, |_| {});
    check_three_qubit_multi_target_calls::<SparseStateVecSoA, _>(
        "SparseStateVecSoA",
        F64_TOLERANCE,
        |_| {},
    );
    check_three_qubit_multi_target_calls::<SparseStateVecAoS, _>(
        "SparseStateVecAoS",
        F64_TOLERANCE,
        |_| {},
    );
    check_three_qubit_multi_target_calls::<StateVecSoA32, _>(
        "StateVecSoA32",
        F32_TOLERANCE,
        |_| {},
    );
    check_three_qubit_multi_target_calls::<StateVecSoA32, _>(
        "StateVecSoA32(fusion disabled)",
        F32_TOLERANCE,
        |sim| sim.set_fusion(false),
    );
}

#[test]
fn phase_carrying_cpu_simulators_match_two_qubit_root_conventions() {
    check_two_qubit_root_kernels::<StateVecSoA, _>("StateVecSoA", F64_TOLERANCE, |_| {});
    check_two_qubit_root_kernels::<StateVecAoS, _>("StateVecAoS", F64_TOLERANCE, |_| {});
    check_two_qubit_root_kernels::<SparseStateVecSoA, _>(
        "SparseStateVecSoA",
        F64_TOLERANCE,
        |_| {},
    );
    check_two_qubit_root_kernels::<SparseStateVecAoS, _>(
        "SparseStateVecAoS",
        F64_TOLERANCE,
        |_| {},
    );
    check_two_qubit_root_kernels::<StateVecSoA32, _>("StateVecSoA32", F32_TOLERANCE, |_| {});
    check_two_qubit_root_kernels::<StateVecSoA32, _>(
        "StateVecSoA32(fusion disabled)",
        F32_TOLERANCE,
        |sim| sim.set_fusion(false),
    );
}

#[test]
fn ch_form_matches_named_clifford_conventions() {
    let targets = [QubitId(0), QubitId(2)];
    for gate in NAMED_SINGLE_QUBIT_GATES {
        if matches!(gate, GateType::T | GateType::Tdg) {
            continue;
        }

        let mut sim = CHForm::new(3);
        sim.h(&[QubitId(0), QubitId(2)])
            .sz(&[QubitId(0)])
            .cx(&[(QubitId(0), QubitId(1))])
            .cz(&[(QubitId(1), QubitId(2))]);
        let input_state = sim.state_vector();
        let mut expected = input_state.clone();
        let matrix = gate
            .canonical_1q_matrix()
            .expect("named single-qubit gate must have a canonical matrix");
        for &target in &targets {
            apply_canonical_matrix(&mut expected, matrix, target);
        }

        apply_named_clifford_gate(&mut sim, gate, &targets);

        let actual = sim.state_vector();
        assert_phase_exact_state(
            "CHForm",
            &format!("{gate:?} on q0,q2"),
            &actual,
            &expected,
            F64_TOLERANCE,
        );
    }

    for &gate in Clifford::all_1q() {
        let mut sim = CHForm::new(3);
        sim.h(&[QubitId(0), QubitId(2)])
            .sz(&[QubitId(0)])
            .cx(&[(QubitId(0), QubitId(1))])
            .cz(&[(QubitId(1), QubitId(2))]);
        let input_state = sim.state_vector();
        let mut expected = input_state.clone();
        let matrix = gate
            .canonical_1q_matrix()
            .expect("single-qubit Clifford must have a canonical matrix");
        for &target in &targets {
            apply_canonical_matrix(&mut expected, matrix, target);
        }
        apply_clifford_gate(&mut sim, gate, &targets);
        let actual = sim.state_vector();
        assert_phase_exact_state(
            "CHForm",
            &format!("{gate} on q0,q2"),
            &actual,
            &expected,
            F64_TOLERANCE,
        );
    }
}
