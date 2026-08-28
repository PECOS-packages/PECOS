// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at https://www.apache.org/licenses/LICENSE-2.0

//! Phase-exact named-gate checks for amplitude-carrying CPU simulators.

use num_complex::Complex64;
use pecos_core::gate_type::{GateType, NAMED_SINGLE_QUBIT_GATES};
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

fn apply_clifford_gate<S: CliffordGateable>(sim: &mut S, gate: Clifford) {
    let qubit = [QubitId(0)];
    match gate {
        Clifford::I => sim.identity(&qubit),
        Clifford::X => sim.x(&qubit),
        Clifford::Y => sim.y(&qubit),
        Clifford::Z => sim.z(&qubit),
        Clifford::H => sim.h(&qubit),
        Clifford::H2 => sim.h2(&qubit),
        Clifford::H3 => sim.h3(&qubit),
        Clifford::H4 => sim.h4(&qubit),
        Clifford::H5 => sim.h5(&qubit),
        Clifford::H6 => sim.h6(&qubit),
        Clifford::SX => sim.sx(&qubit),
        Clifford::SXdg => sim.sxdg(&qubit),
        Clifford::SY => sim.sy(&qubit),
        Clifford::SYdg => sim.sydg(&qubit),
        Clifford::SZ => sim.sz(&qubit),
        Clifford::SZdg => sim.szdg(&qubit),
        Clifford::F => sim.f(&qubit),
        Clifford::Fdg => sim.fdg(&qubit),
        Clifford::F2 => sim.f2(&qubit),
        Clifford::F2dg => sim.f2dg(&qubit),
        Clifford::F3 => sim.f3(&qubit),
        Clifford::F3dg => sim.f3dg(&qubit),
        Clifford::F4 => sim.f4(&qubit),
        Clifford::F4dg => sim.f4dg(&qubit),
        _ => panic!("unsupported Clifford gate {gate}"),
    };
}

fn apply_named_gate<S: CliffordGateable + ArbitraryRotationGateable>(sim: &mut S, gate: GateType) {
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
        other => panic!("unsupported named single-qubit gate {other:?}"),
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
        apply_named_gate(&mut sim, gate);
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
        apply_clifford_gate(&mut sim, gate);
        let actual = [sim.get_amplitude(0), sim.get_amplitude(1)];
        let expected = canonical_clifford_times_state(gate, &input);
        assert_phase_exact(simulator, &gate.to_string(), &actual, &expected, tolerance);
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
fn ch_form_matches_named_clifford_conventions() {
    for gate in NAMED_SINGLE_QUBIT_GATES {
        if matches!(gate, GateType::T | GateType::Tdg) {
            continue;
        }

        let mut sim = CHForm::new(1);
        let qubit = [QubitId(0)];
        sim.h(&qubit).sz(&qubit);
        let input_state = sim.state_vector();
        let input = [input_state[0], input_state[1]];

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
            GateType::T | GateType::Tdg => unreachable!(),
            other => panic!("unsupported named single-qubit gate {other:?}"),
        };

        let state = sim.state_vector();
        let actual = [state[0], state[1]];
        let expected = canonical_times_state(gate, &input);
        assert_phase_exact(
            "CHForm",
            &format!("{gate:?}"),
            &actual,
            &expected,
            F64_TOLERANCE,
        );
    }

    for &gate in Clifford::all_1q() {
        let mut sim = CHForm::new(1);
        let qubit = [QubitId(0)];
        sim.h(&qubit).sz(&qubit);
        let input_state = sim.state_vector();
        let input = [input_state[0], input_state[1]];
        apply_clifford_gate(&mut sim, gate);
        let state = sim.state_vector();
        let actual = [state[0], state[1]];
        let expected = canonical_clifford_times_state(gate, &input);
        assert_phase_exact(
            "CHForm",
            &gate.to_string(),
            &actual,
            &expected,
            F64_TOLERANCE,
        );
    }
}
