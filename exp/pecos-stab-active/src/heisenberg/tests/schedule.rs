// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::joint_distribution::{assert_joint, fixed_noise, noise_alternatives, with_noise};
use super::plan::{body, fixtures, manual, measurement, rotation};
use super::reordering::dependencies;
use super::*;
use crate::MeasurementCase;
use crate::heisenberg::schedule::{plan_cost, select_plan};
use PauliKindForDecomp::{X, Z};

fn rx_pair() -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().rx(Angle64::from_radians(0.37), &[0, 1]);
    circuit.tick().mz(&[0, 1]);
    HeisenbergProgram::compile(&circuit).unwrap()
}

fn t_layer() -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0, 1]);
    // T up to global phase, prepared on |+> so each rotation promotes.
    circuit.tick().rz(Angle64::QUARTER_TURN / 2, &[0, 1]);
    circuit.tick().mx(&[0, 1]);
    HeisenbergProgram::compile(&circuit).unwrap()
}

fn tie_fixture() -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().rx(Angle64::from_radians(0.37), &[0]);
    circuit.tick().mz(&[0]);
    circuit.tick().rz(Angle64::from_radians(0.29), &[1]);
    HeisenbergProgram::compile(&circuit).unwrap()
}

#[derive(Debug, Default)]
struct Coverage {
    programs: usize,
    reordered: usize,
    improved: usize,
    noise_alternatives: usize,
    shots: usize,
    quarter_turns: usize,
    symbolic_signs: usize,
}

fn compare(program: &HeisenbergProgram, coverage: &mut Coverage) {
    let candidate = schedule_for_width(program.clone());
    assert_eq!(candidate.validate(), Ok(()));
    // Independent test DAG, never shared with the production scheduler.
    let predecessors = dependencies(program, true);
    let mut order = Vec::new();
    for operation in candidate.operations() {
        let index = program
            .operations()
            .iter()
            .enumerate()
            .position(|(i, op)| op == operation && !order.contains(&i))
            .unwrap();
        assert!(
            predecessors[index].iter().all(|p| order.contains(p)),
            "illegal order at source operation {index}"
        );
        order.push(index);
    }
    coverage.programs += 1;
    coverage.reordered += usize::from(candidate.operations() != program.operations());
    let before = program.width_profile().into_iter().max().unwrap_or(0);
    let after = candidate.width_profile().into_iter().max().unwrap_or(0);
    assert!(
        after <= before,
        "greedy raised peak {before} -> {after}: {program:?}"
    );
    coverage.improved += usize::from(after < before);
    for op in program.operations() {
        let (HeisenbergOp::Rotation { sign, .. } | HeisenbergOp::Measurement { sign, .. }) = op;
        coverage.symbolic_signs +=
            usize::from(!sign.measurements.is_empty() || !sign.noise.is_empty());
        coverage.quarter_turns += usize::from(matches!(op, HeisenbergOp::Rotation { angle, .. }
            if *angle == Angle64::QUARTER_TURN || *angle == Angle64::THREE_QUARTERS_TURN));
    }
    for (_, noise) in noise_alternatives(program) {
        assert_joint(
            &fixed_noise(&candidate, &noise),
            &fixed_noise(program, &noise),
        );
        coverage.noise_alternatives += 1;
    }
    assert_joint(&with_noise(&candidate), &with_noise(program));
    let plan = candidate.plan(26).unwrap();
    assert_eq!(plan.width_profile(), candidate.width_profile());
    for seed in 0..24 {
        assert_eq!(
            plan.run(seed),
            candidate.run(seed),
            "seed {seed}: {candidate:?}"
        );
        coverage.shots += 1;
    }
}

#[test]
fn exactness_and_peak_corpus() {
    let mut coverage = Coverage::default();
    for program in fixtures()
        .into_iter()
        .chain([rx_pair(), t_layer(), tie_fixture()])
    {
        compare(&program, &mut coverage);
    }
    let mut rng = PecosRng::seed_from_u64(0x6006);
    for generator in [
        super::reordering::random_circuit,
        super::passes::random_circuit,
    ] {
        for _ in 0..40 {
            let program = HeisenbergProgram::compile(&generator(&mut rng)).unwrap();
            let fused = fuse_rotations(program.clone());
            let dropped = drop_measured_rotations(program.clone());
            for variant in [
                program,
                fused.clone(),
                dropped.clone(),
                drop_measured_rotations(fused),
                fuse_rotations(dropped),
            ] {
                compare(&variant, &mut coverage);
            }
        }
    }
    // Extend exact-angle fusion coverage without changing either existing RNG
    // stream. Kept quarter and three-quarter turns must participate in replay.
    let previous_quarters = coverage.quarter_turns;
    let mut fusion_rng = PecosRng::seed_from_u64(0x6007);
    for _ in 0..160 {
        let circuit = super::passes::random_circuit(&mut fusion_rng);
        let program = fuse_rotations(HeisenbergProgram::compile(&circuit).unwrap());
        compare(&program, &mut coverage);
    }
    assert!(coverage.quarter_turns > previous_quarters);
    println!("scheduler coverage: {coverage:?}");
    assert!(coverage.programs > 0 && coverage.reordered > 0 && coverage.improved > 0);
    assert!(coverage.quarter_turns > 0 && coverage.symbolic_signs > 0);
}

#[test]
fn strict_width_improvements_and_feasibility() {
    for program in [rx_pair(), t_layer()] {
        assert_eq!(program.width_profile(), [1, 2, 1, 0]);
        let scheduled = schedule_for_width(program.clone());
        assert_eq!(scheduled.width_profile(), [1, 0, 1, 0]);
        // Both candidates fit: selection must still prefer the lower peak.
        assert_eq!(
            plan_scheduled(&program, 26).unwrap().width_profile(),
            [1, 0, 1, 0]
        );
        assert!(matches!(
            program.plan(1),
            Err(PlanError::WidthExceeded { .. })
        ));
        assert_eq!(
            plan_scheduled(&program, 1).unwrap().width_profile(),
            [1, 0, 1, 0]
        );
        assert_eq!(
            plan_scheduled(&program, 0).unwrap_err(),
            PlanError::WidthExceeded {
                operation: 0,
                width: 1,
                limit: 0,
            }
        );
    }
}

#[test]
fn acceptance_prefers_lower_work_at_equal_peak() {
    let program = manual(
        1,
        vec![
            rotation(
                1,
                &[(0, X)],
                Angle64::from_radians(0.37),
                AffineSign::default(),
            ),
            rotation(
                1,
                &[(0, Z)],
                Angle64::from_radians(0.29),
                AffineSign::default(),
            ),
            measurement(1, &[(0, Z)], 0),
        ],
    );
    let scheduled = schedule_for_width(program.clone());
    assert_eq!(program.width_profile(), [1, 1, 0]);
    // The Active measurement takes priority over the non-promoting RZ.
    assert_eq!(scheduled.width_profile(), [1, 0, 0]);
    assert_eq!(plan_cost(&program.plan(26).unwrap()), (1, 6));
    assert_eq!(plan_cost(&scheduled.plan(26).unwrap()), (1, 5));
    assert_eq!(
        plan_scheduled(&program, 26).unwrap().width_profile(),
        [1, 0, 0]
    );
    compare(&program, &mut Coverage::default());
}

#[test]
fn classification_uses_live_replay_state() {
    let a = Angle64::from_radians(0.37);
    let program = manual(
        2,
        vec![
            measurement(2, &[(0, X)], 0),
            rotation(2, &[(1, X)], a, AffineSign::default()),
            rotation(2, &[(0, Z)], a, AffineSign::default()),
            measurement(2, &[(1, Z)], 1),
        ],
    );
    // MX installs X0 as a stabilizer, making RZ0 promoting. Replaying the
    // chosen prefix is essential to prefer MZ1 after RX1 over that promotion.
    assert_eq!(
        schedule_for_width(program.clone()).width_profile(),
        [0, 1, 0, 1]
    );
    compare(&program, &mut Coverage::default());
}

#[test]
fn acceptance_rejects_higher_work_at_equal_peak() {
    let a = Angle64::from_radians(0.37);
    let program = manual(
        2,
        vec![
            rotation(2, &[(0, X)], a, AffineSign::default()),
            rotation(
                2,
                &[(0, X), (1, X)],
                Angle64::from_radians(0.29),
                AffineSign::default(),
            ),
            measurement(2, &[(0, Z)], 0),
            measurement(2, &[(1, Z)], 1),
            rotation(
                2,
                &[(0, Z), (1, Z)],
                Angle64::from_radians(0.41),
                AffineSign::default(),
            ),
        ],
    );
    let scheduled = schedule_for_width(program.clone());
    let given = program.plan(26).unwrap();
    let greedy = scheduled.plan(26).unwrap();
    assert_eq!(given.width_profile(), [1, 2, 1, 0, 0]);
    assert_eq!(greedy.width_profile(), [1, 1, 2, 1, 0]);
    assert_eq!(plan_cost(&given), (2, 13));
    assert_eq!(plan_cost(&greedy), (2, 14));
    assert_eq!(
        plan_scheduled(&program, 26).unwrap().width_profile(),
        given.width_profile()
    );
    compare(&program, &mut Coverage::default());
}

#[test]
fn ties_keep_given_order() {
    let program = tie_fixture();
    let given = program.plan(26).unwrap();
    let scheduled = schedule_for_width(program.clone());
    assert_eq!(
        scheduled.operations(),
        &[
            program.operations[2].clone(),
            program.operations[0].clone(),
            program.operations[1].clone()
        ]
    );
    let greedy = scheduled.plan(26).unwrap();
    assert_eq!(given.width_profile(), [1, 0, 0]);
    assert_eq!(greedy.width_profile(), [0, 1, 0]);
    assert_eq!(plan_cost(&given), (1, 5));
    assert_eq!(plan_cost(&greedy), (1, 5));
    assert_eq!(
        plan_scheduled(&program, 26).unwrap().width_profile(),
        given.width_profile()
    );
}

#[test]
fn both_infeasible_return_given_error() {
    let program = tie_fixture();
    let given = program.plan(0).unwrap_err();
    let greedy = schedule_for_width(program.clone()).plan(0).unwrap_err();
    assert_eq!(
        given,
        PlanError::WidthExceeded {
            operation: 0,
            width: 1,
            limit: 0
        }
    );
    assert_eq!(
        greedy,
        PlanError::WidthExceeded {
            operation: 1,
            width: 1,
            limit: 0
        }
    );
    assert_eq!(plan_scheduled(&program, 0).unwrap_err(), given);
}

#[test]
fn r2_hidden_reset_precedes_active_reference() {
    let a = Angle64::from_radians(0.37);
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0, 1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().h(&[0, 1]);
    circuit.tick().rx(a, &[0]);
    circuit.tick().pz(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[1]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let mut hidden = measurement(2, &[(0, Z)], 1);
    if let HeisenbergOp::Measurement { record, .. } = &mut hidden {
        *record = None;
    }
    let reference = HeisenbergOp::Measurement {
        pauli: body(2, &[(0, Z), (1, Z)]),
        sign: AffineSign {
            measurements: vec![1],
            ..AffineSign::default()
        },
        symbol: 2,
        record: Some(1),
    };
    assert_eq!(
        program.operations(),
        &[
            measurement(2, &[(0, X), (1, X)], 0),
            rotation(2, &[(0, X)], a, AffineSign::default()),
            hidden,
            reference,
        ]
    );
    let mut structure = ActiveStructure::with_seed(2, 0);
    for op in &program.operations[..2] {
        replay_width_operation(&mut structure, op);
    }
    for (op, expected) in program.operations[2..]
        .iter()
        .zip([MeasurementCase::Random, MeasurementCase::Active])
    {
        let HeisenbergOp::Measurement { pauli, .. } = op else {
            unreachable!()
        };
        assert_eq!(
            structure.measurement(pauli.factors(), false).data().case(),
            expected
        );
    }
    // Dropping production R2 edges chooses the Active reference before the
    // Random hidden reset; with_operations then panics on causality. That is
    // an intended mutation failure, even before the order assertion below.
    let scheduled = schedule_for_width(program.clone());
    let position = |symbol| {
        scheduled
            .operations()
            .iter()
            .position(
                |op| matches!(op, HeisenbergOp::Measurement { symbol: s, .. } if *s == symbol),
            )
            .unwrap()
    };
    assert!(position(1) < position(2));
    compare(&program, &mut Coverage::default());
}

#[test]
fn non_width_errors_are_not_infeasibility() {
    let program = rx_pair();
    assert_eq!(
        plan_scheduled(&program, usize::MAX).unwrap_err(),
        PlanError::InvalidMaxWidth { limit: usize::MAX }
    );
    let invalid = PlanError::InvalidPlan {
        operation: 0,
        width: 0,
        reason: "test invalid plan",
    };
    assert_eq!(
        select_plan(Err(invalid.clone()), program.plan(26)).unwrap_err(),
        invalid
    );
    assert_eq!(
        select_plan(Err(invalid.clone()), Err(invalid.clone())).unwrap_err(),
        invalid
    );
}

#[test]
#[should_panic(expected = "scheduled candidate planning failed: pass bug")]
fn scheduled_only_non_width_error_is_pass_bug() {
    let _ = select_plan(
        rx_pair().plan(26),
        Err(PlanError::InvalidPlan {
            operation: 0,
            width: 0,
            reason: "test invalid scheduled plan",
        }),
    );
}

#[test]
fn empty_program_and_platform_width_work() {
    let empty = HeisenbergProgram::compile(&TickCircuit::new()).unwrap();
    assert_eq!(plan_cost(&plan_scheduled(&empty, 0).unwrap()), (0, 0));
    let limit = (isize::MAX.unsigned_abs() / size_of::<num_complex::Complex64>()).ilog2() as usize;
    let program = manual(
        limit,
        (0..limit)
            .map(|q| {
                rotation(
                    limit,
                    &[(q, X)],
                    Angle64::from_radians(0.37),
                    AffineSign::default(),
                )
            })
            .collect(),
    );
    // Planning and costing allocate no amplitudes, including at the platform limit.
    assert_eq!(
        plan_cost(&plan_scheduled(&program, limit).unwrap()),
        (limit, (1_u128 << (limit + 1)) - 2)
    );
}
