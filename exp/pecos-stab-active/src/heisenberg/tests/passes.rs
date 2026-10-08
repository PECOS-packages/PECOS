// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::joint_distribution::{assert_joint, with_noise};
use super::reordering::{dependencies, index, ready, reordered};
use super::width_profile;
use super::*;
use crate::heisenberg::passes::{
    FusionEvent, RemovalEvent, drop_measured_rotations_observed, fuse_rotations_observed,
};
use PauliKindForDecomp::{X, Y, Z};
use std::collections::BTreeMap;

#[derive(Debug, Default)]
struct Coverage {
    fusions: usize,
    zero_cancellations: usize,
    half_turn_removals: usize,
    constant_toggles: usize,
    quarter_turn_results: usize,
    non_clifford_results: usize,
    measurement_before: usize,
    measurement_after: usize,
    // (original, fused, dropped, fused then dropped, dropped then fused).
    peaks: BTreeMap<[usize; 5], usize>,
    widths: width_profile::Coverage,
}

impl Coverage {
    fn fuse(&mut self, program: HeisenbergProgram) -> HeisenbergProgram {
        fuse_rotations_observed(program, |event, toggles| {
            match event {
                FusionEvent::Fused => self.fusions += 1,
                FusionEvent::ZeroCancelled => self.zero_cancellations += 1,
                FusionEvent::HalfTurnRemoved => self.half_turn_removals += 1,
                FusionEvent::QuarterTurnResult => self.quarter_turn_results += 1,
                FusionEvent::NonCliffordResult => self.non_clifford_results += 1,
            }
            self.constant_toggles += toggles;
        })
    }

    fn drop(&mut self, program: HeisenbergProgram) -> HeisenbergProgram {
        drop_measured_rotations_observed(program, |event| match event {
            RemovalEvent::MeasurementBefore => self.measurement_before += 1,
            RemovalEvent::MeasurementAfter => self.measurement_after += 1,
        })
    }
}

fn compare(program: &HeisenbergProgram, coverage: &mut Coverage) {
    let expected = with_noise(program);
    let fused = coverage.fuse(program.clone());
    assert_eq!(fused.operations, fuse_from_zero(program.clone()).operations);
    let dropped = coverage.drop(program.clone());
    let fused_dropped = coverage.drop(fused.clone());
    let dropped_fused = coverage.fuse(dropped.clone());
    assert_eq!(
        dropped_fused.operations,
        fuse_from_zero(dropped.clone()).operations
    );
    let variants = [program, &fused, &dropped, &fused_dropped, &dropped_fused];
    let peaks = variants.map(|p| p.width_profile().into_iter().max().unwrap_or(0));
    *coverage.peaks.entry(peaks).or_default() += 1;
    for candidate in variants {
        assert_joint(&with_noise(candidate), &expected);
        width_profile::compare(candidate, &mut coverage.widths);
        assert_eq!(candidate.num_qubits, program.num_qubits);
        assert_eq!(candidate.num_noise_symbols, program.num_noise_symbols);
        assert_eq!(candidate.num_measurements, program.num_measurements);
        assert_eq!(candidate.num_records, program.num_records);
        assert_eq!(candidate.noise_channels.len(), program.noise_channels.len());
        for (actual, expected) in candidate.noise_channels.iter().zip(&program.noise_channels) {
            assert_eq!(actual.first_symbol, expected.first_symbol);
            assert_eq!(actual.qubits, expected.qubits);
            assert_eq!(actual.alternatives, expected.alternatives);
        }
        assert_eq!(candidate.detectors, program.detectors);
        assert_eq!(candidate.observables, program.observables);
    }
}

// Reference order from before the resume optimization. Keep the search and
// restart independent of the production cursor so skipping a pair is observable.
fn fuse_from_zero(mut program: HeisenbergProgram) -> HeisenbergProgram {
    fn body(operation: &HeisenbergOp) -> &VirtualPauli {
        match operation {
            HeisenbergOp::Rotation { pauli, .. } | HeisenbergOp::Measurement { pauli, .. } => pauli,
        }
    }

    fn first_pair(operations: &[HeisenbergOp]) -> Option<(usize, usize, Angle64)> {
        for (a, operation) in operations.iter().enumerate() {
            let HeisenbergOp::Rotation { pauli, angle, sign } = operation else {
                continue;
            };
            for (b, other) in operations.iter().enumerate().skip(a + 1) {
                if !pauli.bits().commutes_with(body(other).bits()) {
                    break;
                }
                if let HeisenbergOp::Rotation {
                    pauli: other_pauli,
                    angle: other_angle,
                    sign: other_sign,
                } = other
                    && pauli.bits() == other_pauli.bits()
                    && sign.noise == other_sign.noise
                    && sign.measurements == other_sign.measurements
                {
                    let fused = if sign.constant ^ other_sign.constant {
                        *angle - *other_angle
                    } else {
                        *angle + *other_angle
                    };
                    return Some((a, b, fused));
                }
            }
        }
        None
    }

    let mut operations = std::mem::take(&mut program.operations);
    while let Some((a, b, fused)) = first_pair(&operations) {
        operations.remove(b);
        if fused == Angle64::ZERO {
            operations.remove(a);
        } else if fused == Angle64::HALF_TURN {
            let removed = operations.remove(a);
            for later in &mut operations[a..] {
                if !body(&removed).bits().commutes_with(body(later).bits()) {
                    let (HeisenbergOp::Rotation { sign, .. }
                    | HeisenbergOp::Measurement { sign, .. }) = later;
                    sign.constant ^= true;
                }
            }
        } else {
            let HeisenbergOp::Rotation { angle, .. } = &mut operations[a] else {
                unreachable!("reference fusion starts at a rotation");
            };
            *angle = fused;
        }
    }
    program
        .with_operations(operations)
        .expect("restart-from-zero reference produced an invalid program: pass bug")
}

fn rotation(q: usize, axis: PauliKindForDecomp, angle: Angle64) -> HeisenbergOp {
    let frame = SparseStabY::with_seed(q + 1, 0).with_destab_sign_tracking();
    let (pauli, _) = builder::pullback(&frame, &[(q, axis)]);
    HeisenbergOp::Rotation {
        pauli,
        angle,
        sign: AffineSign::default(),
    }
}

fn measurement(q: usize, axis: PauliKindForDecomp, symbol: usize) -> HeisenbergOp {
    let HeisenbergOp::Rotation { pauli, sign, .. } = rotation(q, axis, Angle64::ZERO) else {
        unreachable!();
    };
    HeisenbergOp::Measurement {
        pauli,
        sign,
        symbol,
        record: Some(symbol),
    }
}

fn signed(
    mut operation: HeisenbergOp,
    constant: bool,
    noise: &[usize],
    measurements: &[usize],
) -> HeisenbergOp {
    let (HeisenbergOp::Rotation { sign, .. } | HeisenbergOp::Measurement { sign, .. }) =
        &mut operation;
    *sign = AffineSign {
        constant,
        noise: noise.to_vec(),
        measurements: measurements.to_vec(),
    };
    operation
}

fn program(operations: Vec<HeisenbergOp>) -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    // Supply nontrivial noise and annotation tables even for hand-built bodies.
    circuit
        .tick()
        .channel(channel::PauliChannel(0.07, 0.11, 0.13, 0));
    circuit.tick().h(&[1]);
    circuit.tick().h(&[1]);
    for operation in &operations {
        if matches!(operation, HeisenbergOp::Measurement { .. }) {
            let refs = circuit.tick().mz(&[0]);
            circuit.detector(&refs).unwrap();
            circuit.observable(&refs).unwrap();
        }
    }
    HeisenbergProgram::compile(&circuit)
        .unwrap()
        .with_operations(operations)
        .unwrap()
}

struct Fixture {
    name: &'static str,
    program: HeisenbergProgram,
    fused: Vec<HeisenbergOp>,
    dropped: Vec<HeisenbergOp>,
}

fn fixtures() -> Vec<Fixture> {
    let a = Angle64::QUARTER_TURN / 2u64;
    let generic = Angle64::from_radians(0.37);
    let x = rotation(0, X, a);
    let z = rotation(0, Z, generic);
    let mx = measurement(0, X, 0);
    let mz = measurement(0, Z, 0);
    let mut fixtures = Vec::new();
    let mut add = |name, operations: Vec<HeisenbergOp>, fused, dropped| {
        fixtures.push(Fixture {
            name,
            program: program(operations),
            fused,
            dropped,
        });
    };
    let quarter = vec![x.clone(), x.clone(), z.clone(), mx.clone()];
    add(
        "quarter then promotion",
        quarter.clone(),
        vec![rotation(0, X, Angle64::QUARTER_TURN), z.clone(), mx.clone()],
        quarter,
    );
    add(
        "consecutive removable rotations",
        vec![x.clone(), rotation(0, X, generic), mx.clone()],
        vec![rotation(0, X, a + generic), mx.clone()],
        vec![mx.clone()],
    );
    let cancel = vec![x.clone(), rotation(0, X, -a), mz.clone()];
    add(
        "zero cancellation",
        cancel.clone(),
        vec![mz.clone()],
        cancel,
    );
    let half = vec![x.clone(), x.clone(), x.clone(), x.clone(), mz.clone()];
    add(
        "four eighth turns",
        half.clone(),
        vec![signed(mz.clone(), true, &[], &[])],
        half,
    );
    let opposite = vec![
        rotation(0, X, a + a + a),
        signed(x.clone(), true, &[], &[]),
        mz.clone(),
    ];
    add(
        "opposite constants",
        opposite.clone(),
        vec![rotation(0, X, a + a), mz.clone()],
        opposite,
    );
    let unequal_noise = vec![
        signed(x.clone(), false, &[0], &[]),
        signed(x.clone(), false, &[1], &[]),
        mz.clone(),
    ];
    add(
        "unequal noise sets",
        unequal_noise.clone(),
        unequal_noise.clone(),
        unequal_noise,
    );
    let blocked = vec![x.clone(), z.clone(), x.clone(), mz.clone()];
    add(
        "anticommuting rotation blocks fusion",
        blocked.clone(),
        blocked.clone(),
        blocked,
    );
    let commuting = rotation(1, Y, generic);
    let across = vec![x.clone(), commuting.clone(), x.clone(), mz.clone()];
    add(
        "commuting rotation allows fusion",
        across.clone(),
        vec![rotation(0, X, a + a), commuting.clone(), mz.clone()],
        across,
    );
    let before = vec![mx.clone(), signed(x.clone(), true, &[0], &[0])];
    add(
        "measurement before",
        before.clone(),
        before,
        vec![mx.clone()],
    );
    let after = vec![signed(x.clone(), true, &[0], &[]), mx.clone()];
    add("measurement after", after.clone(), after, vec![mx.clone()]);
    let different = vec![x.clone(), mz.clone()];
    add(
        "different body kept",
        different.clone(),
        different.clone(),
        different,
    );
    let different_commuting = vec![x.clone(), measurement(1, X, 0)];
    add(
        "different commuting body kept",
        different_commuting.clone(),
        different_commuting.clone(),
        different_commuting,
    );
    let blocked_after = vec![x.clone(), z.clone(), mx.clone()];
    add(
        "anticommuting barrier before measurement",
        blocked_after.clone(),
        blocked_after.clone(),
        blocked_after,
    );
    let blocked_before = vec![mx.clone(), z.clone(), x.clone(), measurement(0, Z, 1)];
    add(
        "anticommuting barrier after measurement",
        blocked_before.clone(),
        blocked_before.clone(),
        blocked_before,
    );
    let producer = measurement(1, Z, 1);
    let intervening = vec![
        mx.clone(),
        producer.clone(),
        signed(x.clone(), false, &[], &[1]),
    ];
    add(
        "intervening producer permits deletion",
        intervening.clone(),
        intervening,
        vec![mx.clone(), producer],
    );
    let producer = measurement(1, X, 0);
    let equal = vec![
        producer.clone(),
        signed(x.clone(), false, &[0, 1], &[0]),
        signed(x.clone(), true, &[0, 1], &[0]),
        measurement(0, Z, 1),
    ];
    add(
        "equal nonempty sets",
        equal.clone(),
        vec![producer.clone(), measurement(0, Z, 1)],
        equal,
    );
    let unequal = vec![
        producer.clone(),
        signed(x.clone(), false, &[], &[0]),
        x.clone(),
        measurement(0, Z, 1),
    ];
    add(
        "unequal measurement sets",
        unequal.clone(),
        unequal.clone(),
        unequal,
    );
    let barrier = vec![x.clone(), mz.clone(), x.clone(), measurement(0, Z, 1)];
    add(
        "anticommuting measurement blocks fusion",
        barrier.clone(),
        barrier.clone(),
        barrier,
    );
    let between = vec![x.clone(), producer.clone(), x.clone(), measurement(0, Z, 1)];
    add(
        "commuting measurement allows fusion",
        between.clone(),
        vec![rotation(0, X, a + a), producer, measurement(0, Z, 1)],
        between,
    );
    let unblocking = vec![x.clone(), z.clone(), rotation(0, Z, -generic), mx.clone()];
    add(
        "fusion unblocks deletion",
        unblocking.clone(),
        vec![x.clone(), mx.clone()],
        unblocking,
    );
    let restart = vec![
        x.clone(),
        z.clone(),
        rotation(0, Z, -generic),
        x.clone(),
        mz.clone(),
    ];
    add(
        "fusion restarts",
        restart.clone(),
        vec![rotation(0, X, a + a), mz.clone()],
        restart,
    );
    let three_quarters = vec![
        rotation(0, X, a + a + a),
        rotation(0, X, a + a + a),
        mz.clone(),
    ];
    add(
        "three quarter result",
        three_quarters.clone(),
        vec![rotation(0, X, Angle64::THREE_QUARTERS_TURN), mz.clone()],
        three_quarters,
    );
    let wrap = vec![rotation(0, X, -a), x.clone(), mz.clone()];
    add("wraparound zero", wrap.clone(), vec![mz.clone()], wrap);
    let later = vec![
        x.clone(),
        rotation(0, X, Angle64::HALF_TURN - a),
        z.clone(),
        rotation(0, Z, generic),
        mx.clone(),
    ];
    add(
        "half turn toggles before subsequent fusion",
        later.clone(),
        vec![
            signed(rotation(0, Z, generic + generic), true, &[], &[]),
            mx,
        ],
        later,
    );
    // The kept quarter turn at A then fuses to a half turn and is removed, so
    // the scan must restart at zero to cancel the outer Z pair it was blocking.
    let restart_after_kept = vec![
        z.clone(),
        x.clone(),
        x.clone(),
        rotation(0, X, Angle64::HALF_TURN - a - a),
        z.clone(),
        measurement(0, X, 0),
    ];
    add(
        "kept fusion then removal at the same A restarts",
        restart_after_kept.clone(),
        vec![measurement(0, X, 0)],
        restart_after_kept,
    );
    // A same-body rotation with different sign sets commutes and does not stop
    // the scan; the outer pair fuses across it.
    let across_other_signs = vec![
        measurement(1, X, 0),
        x.clone(),
        signed(x.clone(), false, &[], &[0]),
        x.clone(),
        measurement(0, Z, 1),
    ];
    add(
        "fusion across same-body rotation with other sign sets",
        across_other_signs.clone(),
        vec![
            measurement(1, X, 0),
            rotation(0, X, Angle64::QUARTER_TURN),
            signed(x.clone(), false, &[], &[0]),
            measurement(0, Z, 1),
        ],
        across_other_signs,
    );
    add("empty", vec![], vec![], vec![]);
    fixtures
}

#[test]
fn exact_fixtures() {
    for fixture in fixtures() {
        assert_eq!(
            crate::fuse_rotations(fixture.program.clone()).operations,
            fixture.fused,
            "{}: fusion",
            fixture.name
        );
        assert_eq!(
            crate::drop_measured_rotations(fixture.program.clone()).operations,
            fixture.dropped,
            "{}: removal",
            fixture.name
        );
        compare(&fixture.program, &mut Coverage::default());
        if fixture.name == "quarter then promotion" {
            assert_eq!(
                &fuse_rotations(fixture.program).width_profile()[..2],
                &[0, 1]
            );
        }
    }
}

fn noisy_reset() -> HeisenbergProgram {
    let a = Angle64::from_radians(0.37);
    let mut circuit = TickCircuit::new();
    circuit.tick().ry(Angle64::from_radians(0.61), &[1]);
    circuit
        .tick()
        .channel(channel::PauliChannel(0.07, 0.11, 0.13, 0));
    circuit.tick().rx(a, &[0]);
    circuit.tick().rx(Angle64::HALF_TURN - a, &[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().pz(&[1]);
    circuit.tick().rz(Angle64::from_radians(0.43), &[1]);
    let refs = circuit.tick().mz(&[0, 1]);
    circuit.detector(&refs).unwrap();
    circuit.observable(&refs[..1]).unwrap();
    HeisenbergProgram::compile(&circuit).unwrap()
}

#[test]
fn half_turn_noise_and_hidden_reset() {
    let program = noisy_reset();
    for operation in &program.operations[1..3] {
        let HeisenbergOp::Rotation { sign, .. } = operation else {
            panic!()
        };
        assert_eq!(sign.noise, [1]);
    }
    assert!(matches!(
        &program.operations[3],
        HeisenbergOp::Measurement { record: None, .. }
    ));
    assert!(
        matches!(&program.operations[4], HeisenbergOp::Rotation { sign, .. } if sign.measurements == [0])
    );
    let mut expected = program.operations.clone();
    expected.drain(1..3);
    for operation in &mut expected[1..] {
        let (HeisenbergOp::Rotation { sign, .. } | HeisenbergOp::Measurement { sign, .. }) =
            operation;
        sign.constant ^= true;
    }
    let mut coverage = Coverage::default();
    let actual = coverage.fuse(program.clone());
    assert_eq!(actual.operations, expected);
    assert_eq!(coverage.fusions, 1);
    assert_eq!(coverage.half_turn_removals, 1);
    assert_eq!(coverage.constant_toggles, 4);
    compare(&program, &mut Coverage::default());
}

// Independent generator: existing generators and their RNG streams stay intact.
// Angle menu: 40% signed odd multiples of pi/4 (each of ±1, ±3 equally
// likely), 20% HALF_TURN - the last generic angle, 20% its negative,
// and 20% a fresh generic angle. Each block starts with a generic angle.
// A rotation repeats the previous qubit and axis with probability 75%.
fn random_circuit(rng: &mut PecosRng) -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().h(&[1]);
    circuit.tick().cx(&[(1, 2)]);
    circuit
        .tick()
        .channel(channel::PauliChannel(0.07, 0.11, 0.13, 0));
    let axes = [GateType::RX, GateType::RY, GateType::RZ];
    for _ in 0..3 {
        let mut q = index(rng, 3);
        let mut axis = axes[index(rng, axes.len())];
        let mut generic = Angle64::from_radians(0.2 + rng.next_f64());
        append(&mut circuit, make_gate(axis, &[q], &[generic]));
        for _ in 0..4 {
            if index(rng, 4) == 0 {
                q = index(rng, 3);
                axis = axes[index(rng, axes.len())];
            }
            let angle = match index(rng, 10) {
                0..=3 => {
                    let a = Angle64::QUARTER_TURN / 2u64;
                    [a, -a, a + a + a, -(a + a + a)][index(rng, 4)]
                }
                4..=5 => Angle64::HALF_TURN - generic,
                6..=7 => -generic,
                _ => {
                    generic = Angle64::from_radians(0.2 + rng.next_f64());
                    generic
                }
            };
            append(&mut circuit, make_gate(axis, &[q], &[angle]));
        }
        append(
            &mut circuit,
            make_gate(
                if rng.next_bool_fast() {
                    GateType::MX
                } else {
                    GateType::MZ
                },
                &[q],
                &[],
            ),
        );
    }
    circuit.tick().mz(&[0, 1, 2]);
    circuit
}

#[test]
fn distributions_widths_and_event_coverage() {
    let mut coverage = Coverage::default();
    let mut rng = PecosRng::seed_from_u64(71893);
    let mut changed_orders = 0;
    for _ in 0..80 {
        let original = HeisenbergProgram::compile(&random_circuit(&mut rng)).unwrap();
        compare(&original, &mut coverage);
        let predecessors = dependencies(&original, true);
        for _ in 0..2 {
            let mut order = Vec::new();
            while order.len() < original.operations.len() {
                let candidates = ready(&predecessors, &order);
                order.push(candidates[index(&mut rng, candidates.len())]);
            }
            changed_orders += usize::from(order.iter().copied().ne(0..original.operations.len()));
            let candidate = reordered(&original, &order).unwrap();
            assert_joint(&with_noise(&candidate), &with_noise(&original));
            compare(&candidate, &mut coverage);
        }
    }
    assert!(changed_orders > 0);
    assert!(coverage.fusions > 0);
    assert!(coverage.zero_cancellations > 0);
    assert!(coverage.half_turn_removals > 0);
    assert!(coverage.constant_toggles > 0);
    assert!(coverage.quarter_turn_results > 0);
    assert!(coverage.non_clifford_results > 0);
    assert!(coverage.measurement_before > 0);
    assert!(coverage.measurement_after > 0);
    println!("generated peephole coverage: {coverage:?}; changed orders: {changed_orders}");
    for fixture in fixtures() {
        compare(&fixture.program, &mut coverage);
    }
    compare(&noisy_reset(), &mut coverage);
    println!("peephole coverage: {coverage:?}; changed orders: {changed_orders}");
}
