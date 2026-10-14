// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Forward-time analytic and `StateVec` oracles for gate noise order (issue #1005).

use pecos_core::{Angle64, Gate, QubitId, gate_type::GateType};
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand::{GateIndex, make_gate};
use pecos_eeg::heisenberg::{
    build_noise_map, heisenberg_detection_probability, heisenberg_exact_from_circuit,
    heisenberg_sparse, heisenberg_with_noise_map,
};
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::{Bm, DepolarizingChannel, GateNoise, NoiseInjection, NoiseSpec};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};

struct IdentityNoise(GateNoise);

impl NoiseSpec for IdentityNoise {
    fn noise_after_gate(&self, _: usize, gate_type: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        if gate_type == GateType::I {
            self.0.injections.clone()
        } else {
            Vec::new()
        }
    }

    fn exact_noise_after_gate(&self, _: usize, gate_type: GateType, _: &[usize]) -> GateNoise {
        if gate_type == GateType::I {
            self.0.clone()
        } else {
            GateNoise::default()
        }
    }
}

fn rotation(label: Bm, rate: f64) -> NoiseInjection {
    NoiseInjection {
        eeg_type: EegType::H,
        label,
        label2: None,
        rate,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual {actual}, expected {expected}"
    );
}

fn assert_paths(
    preparation: &[Gate],
    detector: &Bm,
    readout: &[Gate],
    noise: &IdentityNoise,
    n: usize,
    expected: f64,
) {
    let gates = [make_gate(GateType::I, &(0..n).collect::<Vec<_>>())];
    let initial = StabilizerGroup::from_circuit(preparation, n);
    let index = GateIndex::build(&gates, n, noise, &[false]);
    let map = build_noise_map(&gates, noise, &[false]);
    let mut circuit = preparation.to_vec();
    circuit.extend_from_slice(&gates);
    circuit.extend_from_slice(readout);
    let results = [
        (
            "windowed",
            heisenberg_detection_probability(&gates, detector, noise, &initial, 0.0, &[false]),
        ),
        (
            "precomputed",
            heisenberg_with_noise_map(&gates, detector, &map, &initial, 0.0),
        ),
        (
            "sparse",
            heisenberg_sparse(&gates, detector, noise, &initial, 0.0, &index, None),
        ),
        (
            "sparse precomputed",
            heisenberg_sparse(&gates, detector, noise, &initial, 0.0, &index, Some(&map)),
        ),
        (
            "dense",
            heisenberg_exact_from_circuit(&circuit, &[0], noise, n).unwrap(),
        ),
    ];
    assert!(
        results
            .iter()
            .all(|(_, actual)| (actual - expected).abs() < 1e-12),
        "expected {expected}; paths: {results:?}"
    );
}

#[test]
fn noncommuting_rotations_follow_list_order() {
    let (a, b) = (0.23_f64, 0.37_f64);
    // RX(2a) sends +Z to (0,-sin(2a),cos(2a)); RZ(2b) then
    // gives <X>=sin(2a)sin(2b). Old order rotates Z first, giving <X>=0.
    let expected = (1.0 - (2.0 * a).sin() * (2.0 * b).sin()) / 2.0;
    let old = 0.5;
    assert!((expected - old).abs() > 1e-3);
    let mut state = StateVec::new(1);
    state.rx(Angle64::from_radians(2.0 * a), &[QubitId(0)]);
    state.rz(Angle64::from_radians(2.0 * b), &[QubitId(0)]);
    state.h(&[QubitId(0)]);
    assert_close(state.probability(1), expected);
    let noise = IdentityNoise(GateNoise {
        injections: vec![rotation(Bm::x(0), a), rotation(Bm::z(0), b)],
        depolarizing: Vec::new(),
    });
    assert_paths(
        &[Gate::pz(&[0])],
        &Bm::x(0),
        &[Gate::h(&[0]), Gate::mz(&[0])],
        &noise,
        1,
        expected,
    );
}

#[test]
fn overlapping_channel_follows_injection() {
    let (a, p) = (0.23_f64, 0.3);
    // On Phi+, the Y0X1 rotation gives <Z1>=-sin(2a):
    // i(Y0X1)Z1 = Y0Y1, whose Bell expectation is -1. A later q0 channel
    // cannot change Z1. In the old order it attenuates the Bell YY
    // correlation first, giving p_detect=(1+lambda*sin(2a))/2.
    let expected = (1.0 + (2.0 * a).sin()) / 2.0;
    let old = (1.0 + (1.0 - 4.0 * p / 3.0) * (2.0 * a).sin()) / 2.0;
    assert!((expected - old).abs() > 1e-3);
    let mut oracle = 0.0;
    for (branch, weight) in [(0, 1.0 - p), (1, p / 3.0), (2, p / 3.0), (3, p / 3.0)] {
        let mut state = StateVec::new(2);
        let q0 = [QubitId(0)];
        let pair = [(QubitId(0), QubitId(1))];
        state.h(&q0);
        state.cx(&pair);
        state.szdg(&q0);
        state.rxx(Angle64::from_radians(2.0 * a), &pair);
        state.sz(&q0);
        match branch {
            1 => {
                state.x(&q0);
            }
            2 => {
                state.y(&q0);
            }
            3 => {
                state.z(&q0);
            }
            _ => {}
        }
        oracle += weight * (state.probability(2) + state.probability(3));
    }
    assert_close(oracle, expected);
    let noise = IdentityNoise(GateNoise {
        injections: vec![rotation(Bm::y(0).multiply(&Bm::x(1)), a)],
        depolarizing: vec![DepolarizingChannel::OneQubit {
            qubit: 0,
            probability: p,
        }],
    });
    assert_paths(
        &[Gate::pz(&[0, 1]), Gate::h(&[0]), Gate::cx(&[(0, 1)])],
        &Bm::z(1),
        &[Gate::mz(&[1])],
        &noise,
        2,
        expected,
    );
}

#[test]
fn commuting_rotations_preserve_probability() {
    let (a, b) = (0.23_f64, 0.37_f64);
    // Two Z rotations on |+> combine to RZ(2(a+b)) in either order.
    // Both new and old orders give p_detect=(1-cos(2(a+b)))/2.
    let expected = (1.0 - (2.0 * (a + b)).cos()) / 2.0;
    let mut state = StateVec::new(1);
    state.h(&[QubitId(0)]);
    state.rz(Angle64::from_radians(2.0 * a), &[QubitId(0)]);
    state.rz(Angle64::from_radians(2.0 * b), &[QubitId(0)]);
    state.h(&[QubitId(0)]);
    assert_close(state.probability(1), expected);
    let noise = IdentityNoise(GateNoise {
        injections: vec![rotation(Bm::z(0), a), rotation(Bm::z(0), b)],
        depolarizing: Vec::new(),
    });
    assert_paths(
        &[Gate::pz(&[0]), Gate::h(&[0])],
        &Bm::x(0),
        &[Gate::h(&[0]), Gate::mz(&[0])],
        &noise,
        1,
        expected,
    );
}
