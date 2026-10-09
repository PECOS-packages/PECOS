// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::reordering::{dependencies, index, random_circuit, ready, reordered};
use super::*;
use crate::MeasurementCase;
use crate::structure::ActiveStructure;
use std::cell::RefCell;

#[derive(Debug, Default)]
pub(super) struct Coverage {
    programs: usize,
    runs: usize,
    operations: usize,
    hidden_resets: usize,
    random: usize,
    deterministic: usize,
    active: usize,
    snapped: usize,
    projections: [usize; 2],
    basis_gates: usize,
    identity_rotations: usize,
    changed_orders: usize,
}

pub(super) fn compare(program: &HeisenbergProgram, coverage: &mut Coverage) {
    let expected = program.width_profile();
    assert_eq!(expected.len(), program.operations.len());
    coverage.programs += 1;
    let mut rng = PecosRng::seed_from_u64(731);
    for seed in [0, 1, 37, 981] {
        for noise_kind in 0..3 {
            let noise: Vec<_> = (0..program.num_noise_symbols)
                .map(|_| match noise_kind {
                    0 => false,
                    1 => true,
                    _ => rng.next_bool_fast(),
                })
                .collect();
            let mut actual = Vec::new();
            let pending: RefCell<Option<ActiveStructure>> = RefCell::new(None);
            let counts = RefCell::new(&mut *coverage);
            let (shot, _) = program.execute_observed(
                StabActive::with_seed(program.num_qubits, seed),
                &noise,
                |state, pauli, negative, _, record| {
                    let mut counts = counts.borrow_mut();
                    counts.hidden_resets += usize::from(record.is_none());
                    let mut structure = state.structure.clone();
                    let measurement = structure.measurement(pauli.factors(), negative);
                    match measurement.data().case() {
                        MeasurementCase::Random => counts.random += 1,
                        MeasurementCase::Deterministic => counts.deterministic += 1,
                        MeasurementCase::Active => {
                            counts.active += 1;
                            let expectation = StabActive::active_expectation(
                                &state.amplitudes,
                                measurement.data().parts(),
                            )
                            .re;
                            counts.snapped += usize::from(
                                1.0 - expectation.abs() <= crate::EXPECTATION_ENDPOINT_TOLERANCE,
                            );
                            *pending.borrow_mut() = Some(state.structure.clone());
                        }
                    }
                    None
                },
                |state, operation, measurements| {
                    actual.push(state.active_width());
                    let mut counts = counts.borrow_mut();
                    if let HeisenbergOp::Measurement {
                        pauli,
                        sign,
                        symbol,
                        ..
                    } = operation
                    {
                        if let Some(mut structure) = pending.borrow_mut().take() {
                            let measurement = structure
                                .measurement(pauli.factors(), sign.evaluate(&noise, measurements));
                            let active = measurement
                                .apply_measurement(measurements[*symbol])
                                .unwrap();
                            counts.basis_gates += usize::from(!active.gates().is_empty());
                            counts.projections[usize::from(active.projection().value())] += 1;
                        }
                    } else if let HeisenbergOp::Rotation { pauli, angle, sign } = operation {
                        let mut structure = state.structure.clone();
                        let rotation = structure.rotation(
                            *angle,
                            pauli.factors(),
                            sign.evaluate(&noise, measurements),
                        );
                        counts.identity_rotations += usize::from(rotation.is_identity());
                    }
                },
            );
            assert_eq!(
                expected.iter().copied().max().unwrap_or(0),
                shot.peak_active_width
            );
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual, expected, "seed {seed}, noise {noise:?}");
            coverage.runs += 1;
            coverage.operations += actual.len();
        }
    }
}

fn rotation(
    n: usize,
    factors: &[(usize, PauliKindForDecomp)],
    angle: Angle64,
    negative: bool,
) -> HeisenbergOp {
    let frame = SparseStabY::with_seed(n, 0).with_destab_sign_tracking();
    let (pauli, _) = builder::pullback(&frame, factors);
    HeisenbergOp::Rotation {
        pauli,
        angle,
        sign: AffineSign {
            constant: negative,
            ..AffineSign::default()
        },
    }
}

fn measurement(n: usize, q: usize, axis: PauliKindForDecomp, symbol: usize) -> HeisenbergOp {
    let frame = SparseStabY::with_seed(n, 0).with_destab_sign_tracking();
    let (pauli, _) = builder::pullback(&frame, &[(q, axis)]);
    HeisenbergOp::Measurement {
        pauli,
        sign: AffineSign::default(),
        symbol,
        record: Some(symbol),
    }
}

fn program(n: usize, operations: Vec<HeisenbergOp>) -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[n - 1]);
    circuit.tick().h(&[n - 1]);
    for op in &operations {
        if matches!(op, HeisenbergOp::Measurement { .. }) {
            circuit.tick().mz(&[0]);
        }
    }
    HeisenbergProgram::compile(&circuit)
        .unwrap()
        .with_operations(operations)
        .unwrap()
}

#[test]
fn random_programs_and_legal_orders_match_every_width() {
    let mut rng = PecosRng::seed_from_u64(19381);
    let mut coverage = Coverage::default();
    for _ in 0..120 {
        let original = HeisenbergProgram::compile(&random_circuit(&mut rng)).unwrap();
        compare(&original, &mut coverage);
        let predecessors = dependencies(&original, true);
        for _ in 0..3 {
            let mut order = Vec::new();
            while order.len() < original.operations.len() {
                let candidates = ready(&predecessors, &order);
                order.push(candidates[index(&mut rng, candidates.len())]);
            }
            coverage.changed_orders +=
                usize::from(order.iter().copied().ne(0..original.operations.len()));
            compare(&reordered(&original, &order).unwrap(), &mut coverage);
        }
    }
    println!("random width comparisons: {coverage:?}");
    assert!(coverage.changed_orders > 0);
    assert!(coverage.hidden_resets > 0);
    assert!(coverage.random > 0);
    assert!(coverage.deterministic > 0);
    assert!(coverage.active > 0);
    assert!(coverage.projections.iter().all(|&count| count > 0));
    assert!(coverage.basis_gates > 0);
}

#[test]
fn fused_quarter_turn_then_promotion() {
    use PauliKindForDecomp::{X, Z};
    for negative in [false, true] {
        let program = program(
            1,
            vec![
                rotation(1, &[(0, X)], Angle64::QUARTER_TURN, negative),
                rotation(1, &[(0, Z)], Angle64::from_radians(0.37), false),
            ],
        );
        assert_eq!(program.width_profile(), [0, 1]);
        compare(&program, &mut Coverage::default());
    }
}

#[test]
fn active_measurement_demotes() {
    use PauliKindForDecomp::{X, Z};
    let program = program(
        1,
        vec![
            rotation(1, &[(0, X)], Angle64::from_radians(0.37), false),
            measurement(1, 0, Z, 0),
        ],
    );
    assert_eq!(program.width_profile(), [1, 0]);
    compare(&program, &mut Coverage::default());
}

#[test]
fn random_measurement_updates_tableau() {
    use PauliKindForDecomp::{X, Z};
    let program = program(
        1,
        vec![
            measurement(1, 0, X, 0),
            rotation(1, &[(0, Z)], Angle64::from_radians(0.37), false),
        ],
    );
    assert_eq!(program.width_profile(), [0, 1]);
    compare(&program, &mut Coverage::default());
}

#[test]
fn identity_half_turn_and_snapped_active_measurements() {
    use PauliKindForDecomp::{X, Z};
    let mut coverage = Coverage::default();
    for negative in [false, true] {
        let a = Angle64::from_radians(0.79);
        let b = Angle64::from_radians(2.91);
        let program = program(
            1,
            vec![
                rotation(1, &[], a, negative),
                rotation(1, &[(0, X)], Angle64::ZERO, negative),
                rotation(1, &[(0, X)], Angle64::HALF_TURN, negative),
                rotation(1, &[(0, Z)], Angle64::from_radians(1.23), negative),
                rotation(1, &[(0, X)], a, negative),
                rotation(1, &[(0, X)], b, negative),
                rotation(1, &[(0, X)], -(a + b), negative),
                measurement(1, 0, Z, 0),
            ],
        );
        assert_eq!(program.width_profile(), [0, 0, 0, 0, 1, 1, 1, 0]);
        compare(&program, &mut coverage);
    }
    println!("hand-built width comparisons: {coverage:?}");
    assert!(coverage.identity_rotations > 0);
    assert!(coverage.snapped > 0);
}

#[test]
fn profile_exceeds_machine_word_then_demotes_every_coordinate() {
    use PauliKindForDecomp::{X, Z};
    let n = usize::try_from(usize::BITS).unwrap() + 3;
    let operations = (0..n)
        .map(|q| rotation(n, &[(q, X)], Angle64::from_radians(0.37), false))
        .chain((0..n).map(|q| measurement(n, q, Z, q)))
        .collect();
    let expected: Vec<_> = (1..=n).chain((0..n).rev()).collect();
    assert_eq!(program(n, operations).width_profile(), expected);
}

#[test]
fn overflowing_rotation_preserves_the_entire_state() {
    use PauliKindForDecomp::X;
    let angle = Angle64::from_radians(0.37);
    let program = program(
        3,
        (0..3)
            .map(|q| rotation(3, &[(q, X)], angle, false))
            .collect(),
    );
    assert_eq!(program.width_profile(), [1, 2, 3]);
    let mut widths = Vec::new();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        program.execute_observed(
            StabActive::with_seed(3, 0).with_max_active_width(1),
            &[],
            |_, _, _, _, _| None,
            |state, _, _| widths.push(state.active_width()),
        );
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<String>().unwrap(),
        "active width 2 exceeds limit 1"
    );
    assert_eq!(widths, [1]);

    let mut state = StabActive::with_seed(3, 0).with_max_active_width(1);
    state.rotate_pauli(angle, &[(0, X)], false);
    let before = state.clone();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.rotate_pauli(angle, &[(1, X), (2, X)], false);
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<String>().unwrap(),
        "active width 2 exceeds limit 1"
    );
    assert_eq!(state.active_width(), before.active_width());
    assert_eq!(state.peak_active_width(), before.peak_active_width());
    assert_eq!(state.structure.active(), before.structure.active());
    assert_eq!(
        format!("{:?}", state.structure.tableau()),
        format!("{:?}", before.structure.tableau())
    );
    assert_eq!(state.amplitudes, before.amplitudes);
}

#[test]
fn empty_program_has_zero_peak() {
    let program = HeisenbergProgram::compile(&TickCircuit::new()).unwrap();
    assert_eq!(program.width_profile(), [] as [usize; 0]);
    compare(&program, &mut Coverage::default());
}
