// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::joint_distribution::{
    assert_joint, conditional, fixed_noise, fixed_noise_by, with_noise, with_noise_by,
};
use super::reordering::{dependencies, index, random_circuit, ready, reordered};
use super::*;
use crate::MeasurementCase;
use crate::heisenberg::plan::{Corrections, Instruction};
use PauliKindForDecomp::{X, Y, Z};

#[derive(Default, Debug)]
struct Coverage {
    programs: usize,
    shots: usize,
    operations: usize,
    random: usize,
    active: usize,
    quarter: usize,
    promotion: usize,
    eta_one: usize,
    snapped: usize,
    hidden: usize,
    nonlast: usize,
    flipped_pivot: usize,
}

fn invariant(
    real: &ActiveStructure,
    planned: &ActiveStructure,
    corrections: &Corrections,
    noise: &[bool],
    outcomes: &[bool],
) {
    assert_eq!(real.active(), planned.active());
    for stabilizer in [true, false] {
        let r = if stabilizer {
            real.tableau().stabs()
        } else {
            real.tableau().destabs()
        };
        let p = if stabilizer {
            planned.tableau().stabs()
        } else {
            planned.tableau().destabs()
        };
        for index in 0..real.num_qubits() {
            let body = planned.row_bits(stabilizer, index);
            assert_eq!(real.row_bits(stabilizer, index), body, "row support");
            let negative = corrections.iter().fold(false, |s, (row, bit)| {
                s ^ (!row.commutes_with(&body) && bit.evaluate(noise, outcomes))
            });
            assert_eq!(
                r.signs_i.contains(index),
                p.signs_i.contains(index),
                "row imaginary sign"
            );
            assert_eq!(
                r.signs_minus.contains(index),
                p.signs_minus.contains(index) ^ negative,
                "row real sign, row={index}, stabilizer={stabilizer}"
            );
        }
    }
}

fn compare(program: &HeisenbergProgram, coverage: &mut Coverage) {
    let mut snapshots = Vec::new();
    let mut previous = 0;
    let mut width = 0;
    let plan = program
        .plan_observed(26, |structure, corrections, op, _| {
            for (_, bit) in &corrections[previous..] {
                if *bit == AffineSign::default() {
                    continue;
                }
                match &op.instruction {
                    Instruction::Clifford => coverage.quarter += 1,
                    Instruction::Rotation { .. } => coverage.promotion += 1,
                    Instruction::Measurement {
                        case: MeasurementCase::Random,
                        ..
                    } => coverage.random += 1,
                    Instruction::Measurement {
                        case: MeasurementCase::Active,
                        ..
                    } => coverage.active += 1,
                    Instruction::Measurement { .. } => panic!("unexpected correction source"),
                }
            }
            if let Instruction::Measurement {
                parts,
                projection: Some(p),
                ..
            } = &op.instruction
            {
                coverage.eta_one += usize::from(p.value.constant ^ op.sign.constant);
                coverage.nonlast += usize::from(p.pivot + 1 < width);
                coverage.flipped_pivot += usize::from(parts.flip & (1 << p.pivot) != 0);
            }
            previous = corrections.len();
            width = structure.width();
            snapshots.push((structure.clone(), corrections.clone()));
        })
        .unwrap();
    assert_eq!(plan.width_profile(), program.width_profile());
    coverage.programs += 1;
    let mut sampler = plan.sampler();
    for seed in 0..24 {
        assert_eq!(
            sampler.run(seed),
            program.run(seed),
            "seed={seed}, program={program:?}"
        );
        assert_eq!(plan.run(seed), program.run(seed));
        // Match production noise draws, then compare hidden symbols too.
        let mut state = StabActive::with_seed(program.num_qubits, seed);
        let mut noise = vec![false; program.num_noise_symbols];
        for channel in &program.noise_channels {
            channel.sample(&mut state.rng, &mut noise);
        }
        for forced in [false, true] {
            let mut real_ops = Vec::new();
            let mut probabilities = Vec::new();
            let (expected, outcomes) = program.execute_observed(
                state.clone(),
                &noise,
                |state, body, negative, symbol, record| {
                    let (p, case) = conditional(state, body, negative);
                    coverage.snapped +=
                        usize::from(case == MeasurementCase::Active && (p <= 0.0 || p >= 1.0));
                    coverage.hidden += usize::from(record.is_none());
                    probabilities.push((p.to_bits(), p <= 0.0 || p >= 1.0, symbol, record));
                    forced.then_some((seed + symbol as u64).is_multiple_of(2))
                },
                |state, _, measurements| {
                    let (reference, corrections) = &snapshots[real_ops.len()];
                    invariant(
                        &state.structure,
                        reference,
                        corrections,
                        &noise,
                        measurements,
                    );
                    real_ops.push((
                        state.amplitudes.clone(),
                        state.active_width(),
                        measurements.to_vec(),
                    ));
                },
            );
            let mut measurement_index = 0;
            let mut operation_index = 0;
            let actual = sampler.run_observed(
                seed,
                |p, symbol, record| {
                    assert_eq!(
                        (p.to_bits(), p <= 0.0 || p >= 1.0, symbol, record),
                        probabilities[measurement_index],
                        "probability/endpoint at measurement {measurement_index}"
                    );
                    measurement_index += 1;
                    forced.then_some((seed + symbol as u64).is_multiple_of(2))
                },
                |amplitudes, width, measurements| {
                    let (expected, expected_width, expected_measurements) =
                        &real_ops[operation_index];
                    assert_eq!(amplitudes, expected, "amplitudes at op {operation_index}");
                    assert_eq!(width, *expected_width);
                    assert_eq!(
                        measurements, expected_measurements,
                        "accepted outcomes at op {operation_index}"
                    );
                    operation_index += 1;
                },
            );
            assert_eq!(actual, expected);
            assert_eq!(measurement_index, probabilities.len());
            assert_eq!(operation_index, program.operations.len());
            if let Some((_, _, measurements)) = real_ops.last() {
                assert_eq!(*measurements, outcomes);
            }
            coverage.shots += 1;
            coverage.operations += program.operations.len();
        }
    }
}

fn body(n: usize, factors: &[(usize, PauliKindForDecomp)]) -> VirtualPauli {
    builder::pullback(
        &SparseStabY::with_seed(n, 0).with_destab_sign_tracking(),
        factors,
    )
    .0
}

fn rotation(
    n: usize,
    factors: &[(usize, PauliKindForDecomp)],
    angle: Angle64,
    sign: AffineSign,
) -> HeisenbergOp {
    HeisenbergOp::Rotation {
        pauli: body(n, factors),
        angle,
        sign,
    }
}

fn measurement(n: usize, factors: &[(usize, PauliKindForDecomp)], symbol: usize) -> HeisenbergOp {
    HeisenbergOp::Measurement {
        pauli: body(n, factors),
        sign: AffineSign::default(),
        symbol,
        record: Some(symbol),
    }
}

fn manual(n: usize, operations: Vec<HeisenbergOp>) -> HeisenbergProgram {
    let measurements = operations
        .iter()
        .filter(|op| matches!(op, HeisenbergOp::Measurement { .. }))
        .count();
    let program = HeisenbergProgram {
        num_qubits: n,
        operations: Vec::new(),
        noise_channels: Vec::new(),
        num_noise_symbols: 0,
        num_measurements: measurements,
        num_records: measurements,
        detectors: Vec::new(),
        observables: Vec::new(),
    };
    program.with_operations(operations).unwrap()
}

fn fixtures() -> Vec<HeisenbergProgram> {
    let theta = Angle64::from_radians(0.37);
    let mut programs = vec![
        manual(
            1,
            vec![
                measurement(1, &[(0, X)], 0),
                rotation(1, &[(0, Y)], Angle64::QUARTER_TURN, AffineSign::default()),
                measurement(1, &[(0, Z)], 1),
            ],
        ),
        manual(
            1,
            vec![
                rotation(1, &[(0, X)], theta, AffineSign::default()),
                rotation(1, &[(0, X)], Angle64::HALF_TURN, AffineSign::default()),
                measurement(1, &[(0, Z)], 0),
                measurement(1, &[(0, Z)], 1),
            ],
        ),
        manual(
            1,
            vec![measurement(1, &[(0, X)], 0), measurement(1, &[(0, X)], 1)],
        ),
        manual(
            1,
            vec![
                measurement(1, &[(0, X)], 0),
                rotation(1, &[(0, Y)], theta, AffineSign::default()),
                measurement(1, &[(0, Z)], 1),
            ],
        ),
        manual(
            1,
            vec![
                measurement(1, &[(0, X)], 0),
                rotation(1, &[(0, Y)], theta, AffineSign::default()),
                rotation(
                    1,
                    &[(0, Y)],
                    Angle64::from_radians(0.23),
                    AffineSign::default(),
                ),
                measurement(1, &[(0, Z)], 1),
            ],
        ),
        manual(
            2,
            vec![
                rotation(2, &[(0, X)], theta, AffineSign::default()),
                rotation(2, &[(1, X)], theta, AffineSign::default()),
                measurement(2, &[(0, Y)], 0),
                measurement(2, &[(1, Z)], 1),
            ],
        ),
        manual(
            1,
            vec![
                rotation(1, &[(0, X)], theta, AffineSign::default()),
                rotation(1, &[(0, X)], -theta, AffineSign::default()),
                measurement(1, &[(0, Z)], 0),
            ],
        ),
    ];
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    c.tick().mz(&[0]);
    c.tick().rzz(Angle64::QUARTER_TURN / 2, &[(0, 1)]);
    c.tick().rzz(Angle64::QUARTER_TURN / 2, &[(0, 1)]);
    c.tick().rx(theta, &[0]);
    c.tick().mz(&[0]);
    programs.push(fuse_rotations(HeisenbergProgram::compile(&c).unwrap()));
    let mut c = TickCircuit::new();
    c.tick().x(&[0]);
    c.tick().pz(&[0]);
    c.tick().rx(theta, &[1]);
    c.tick().cx(&[(0, 1)]);
    c.tick().mz(&[1]);
    programs.push(HeisenbergProgram::compile(&c).unwrap());
    // A multi-qubit Random correction, followed by a Y body.
    programs.push(manual(
        2,
        vec![
            rotation(
                2,
                &[(0, Y), (1, X)],
                Angle64::QUARTER_TURN,
                AffineSign::default(),
            ),
            measurement(2, &[(0, X)], 0),
            rotation(2, &[(0, Y), (1, Y)], theta, AffineSign::default()),
            measurement(2, &[(0, Z)], 1),
        ],
    ));
    programs
}

#[test]
fn exact_forced_and_row_invariant_corpus() {
    let mut coverage = Coverage::default();
    for program in fixtures() {
        compare(&program, &mut coverage);
    }
    let mut rng = PecosRng::seed_from_u64(0x5b);
    for _ in 0..40 {
        let c = random_circuit(&mut rng);
        oracle::compare(&c, 13);
        let program = HeisenbergProgram::compile(&c).unwrap();
        let predecessors = dependencies(&program, true);
        let mut order = Vec::new();
        while order.len() < program.operations.len() {
            let candidates = ready(&predecessors, &order);
            order.push(candidates[index(&mut rng, candidates.len())]);
        }
        let variants = [
            program.clone(),
            reordered(&program, &order).unwrap(),
            fuse_rotations(program.clone()),
            drop_measured_rotations(program.clone()),
            drop_measured_rotations(fuse_rotations(program)),
        ];
        for program in variants {
            compare(&program, &mut coverage);
        }
    }
    // Hand-built symbolic Clifford angles are deliberately retained as operations.
    for _ in 0..80 {
        let n = 3;
        let mut operations = vec![measurement(n, &[(0, X)], 0)];
        let mut symbols = 1;
        for _ in 0..16 {
            let q = index(&mut rng, n);
            let r = (q + 1 + index(&mut rng, 2)) % n;
            let kinds = [X, Y, Z];
            let factors = [
                (q, kinds[index(&mut rng, 3)]),
                (r, kinds[index(&mut rng, 3)]),
            ];
            let sign = AffineSign {
                constant: rng.next_bool_fast(),
                measurements: (0..symbols).filter(|_| rng.next_bool_fast()).collect(),
                noise: Vec::new(),
            };
            if index(&mut rng, 3) == 0 {
                let mut op = measurement(n, &factors, symbols);
                if let HeisenbergOp::Measurement { sign: s, .. } = &mut op {
                    *s = sign;
                }
                operations.push(op);
                symbols += 1;
            } else {
                let angles = [
                    Angle64::QUARTER_TURN,
                    Angle64::THREE_QUARTERS_TURN,
                    Angle64::HALF_TURN,
                    Angle64::from_radians(0.31),
                    Angle64::from_radians(-0.29),
                ];
                operations.push(rotation(
                    n,
                    &factors,
                    angles[index(&mut rng, angles.len())],
                    sign,
                ));
            }
        }
        compare(&manual(n, operations), &mut coverage);
    }
    println!("plan coverage (24 seeds, sampled and forced): {coverage:?}");
    assert!(
        coverage.random > 0
            && coverage.active > 0
            && coverage.quarter > 0
            && coverage.promotion > 0
    );
    assert!(
        coverage.eta_one > 0
            && coverage.snapped > 0
            && coverage.hidden > 0
            && coverage.nonlast > 0
            && coverage.flipped_pivot > 0
    );
}

#[test]
fn joint_distributions_share_enumerator() {
    let mut programs = fixtures();
    let mut rng = PecosRng::seed_from_u64(519);
    programs
        .extend((0..20).map(|_| HeisenbergProgram::compile(&random_circuit(&mut rng)).unwrap()));
    for program in programs {
        let plan = program.plan(26).unwrap();
        let mut sampler = plan.sampler();
        let settings = std::cell::RefCell::new(std::collections::BTreeSet::new());
        let mut run =
            |noise: &[bool], force: &mut dyn FnMut(f64, usize, Option<usize>) -> Option<bool>| {
                settings.borrow_mut().insert(noise.to_vec());
                let mut outcomes = vec![false; program.num_measurements];
                let shot = sampler
                    .fixed_noise_observed(0, noise, force, |_, _, m| outcomes.copy_from_slice(m));
                (shot, outcomes)
            };
        let noise = vec![false; program.num_noise_symbols];
        assert_joint(
            &fixed_noise_by(&program, &noise, &mut run),
            &fixed_noise(&program, &noise),
        );
        assert_joint(&with_noise_by(&program, &mut run), &with_noise(&program));
        // Every joint noise alternative visited by the shared enumerator,
        // including correlated components and combinations of channels.
        let alternatives = settings.borrow().clone();
        for noise in alternatives {
            assert_joint(
                &fixed_noise_by(&program, &noise, &mut run),
                &fixed_noise(&program, &noise),
            );
        }
    }
}

#[test]
fn exact_fixture_records_and_canonical_cancellation() {
    let programs = fixtures();
    for seed in 0..24 {
        let shot = programs[0].plan(1).unwrap().run(seed);
        assert_eq!(shot.records[1], !shot.records[0]);
        let shot = programs[1].plan(1).unwrap().run(seed);
        assert_eq!(shot.records[1], shot.records[0]);
        let shot = programs[2].plan(1).unwrap().run(seed);
        assert_eq!(shot.records[1], shot.records[0]);
        assert_eq!(programs[6].plan(1).unwrap().run(seed).records, [false]);
        assert_eq!(programs[8].plan(1).unwrap().run(seed).records.len(), 1);
    }
    let plan = programs[4].plan(1).unwrap();
    assert_eq!(plan.operations[2].sign, AffineSign::default());
}

#[test]
fn limits_scalar_buffers_and_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<SamplingPlan>();
    let empty = HeisenbergProgram::compile(&TickCircuit::new()).unwrap();
    let limit = (isize::MAX.unsigned_abs() / size_of::<num_complex::Complex64>()).ilog2() as usize;
    assert_eq!(
        empty.plan(limit + 1).unwrap_err(),
        PlanError::InvalidMaxWidth { limit: limit + 1 }
    );
    assert_eq!(
        empty.plan(usize::MAX).unwrap_err(),
        PlanError::InvalidMaxWidth { limit: usize::MAX }
    );
    let plan = empty.plan(limit).unwrap();
    assert_eq!(plan.width_profile(), &[] as &[usize]);
    assert_eq!(plan.run(7), empty.run(7));
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    c.tick().cx(&[(0, 129)]);
    c.tick().mz(&[0, 129]);
    let pure = HeisenbergProgram::compile(&c).unwrap();
    let plan = pure.plan(0).unwrap();
    let mut sampler = plan.sampler();
    for seed in 0..24 {
        assert_eq!(
            sampler.run_observed(
                seed,
                |_, _, _| None,
                |a, width, _| {
                    assert_eq!(a.len(), 1);
                    assert_eq!(width, 0);
                }
            ),
            pure.run(seed)
        );
    }
    let program = manual(
        2,
        vec![
            rotation(
                2,
                &[(0, X)],
                Angle64::from_radians(0.31),
                AffineSign::default(),
            ),
            rotation(
                2,
                &[(1, X)],
                Angle64::from_radians(0.31),
                AffineSign::default(),
            ),
        ],
    );
    assert_eq!(
        program.plan(1).unwrap_err(),
        PlanError::WidthExceeded {
            operation: 1,
            width: 2,
            limit: 1
        }
    );
    assert_eq!(
        program.plan(0).unwrap_err(),
        PlanError::WidthExceeded {
            operation: 0,
            width: 1,
            limit: 0
        }
    );
    assert_eq!(program.plan(2).unwrap().width_profile(), [1, 2]);
}

#[test]
fn validator_rejects_malformed_lowering() {
    let plan = fixtures()[5].plan(2).unwrap();
    let invalid = |p: SamplingPlan, reason| {
        let error = p.validate(2).unwrap_err();
        assert!(
            matches!(error, PlanError::InvalidPlan { reason: r, .. } if r == reason),
            "{error}"
        );
    };
    let mut p = plan.clone();
    p.operations[0].sign.measurements = vec![0];
    invalid(p, "measurement symbol not yet produced");
    let mut p = plan.clone();
    p.operations[0].sign.noise = vec![0];
    invalid(p, "noise symbol out of range");
    let mut p = plan.clone();
    p.operations[3].sign.measurements = vec![0, 0];
    invalid(p, "noncanonical affine sign");
    let mut p = plan.clone();
    if let Instruction::Rotation { parts, .. } = &mut p.operations[0].instruction {
        parts.flip = 2;
    }
    invalid(p, "amplitude mask outside active width");
    let mut p = plan.clone();
    if let Instruction::Measurement {
        projection: Some(projection),
        ..
    } = &mut p.operations[2].instruction
    {
        projection.pivot = 2;
    }
    invalid(p, "projection pivot outside active width");
    let mut p = plan.clone();
    if let Instruction::Measurement {
        projection: Some(projection),
        ..
    } = &mut p.operations[2].instruction
    {
        projection.value.measurements.push(1);
    }
    invalid(p, "measurement symbol not yet produced");
    let mut p = plan.clone();
    if let Instruction::Measurement {
        projection: Some(projection),
        ..
    } = &mut p.operations[2].instruction
    {
        projection
            .gates
            .push(pecos_stab_tn::stab_mps::coordinate_tableau::CoordinateGate::H(2));
    }
    invalid(p, "coordinate gate outside active width");
    let mut p = plan.clone();
    if let Instruction::Measurement { symbol, .. } = &mut p.operations[3].instruction {
        *symbol = 0;
    }
    invalid(p, "duplicate measurement symbol");
    let mut p = plan.clone();
    if let Instruction::Measurement { record, .. } = &mut p.operations[3].instruction {
        *record = Some(0);
    }
    invalid(p, "duplicate record ordinal");
    let mut p = plan.clone();
    if let Instruction::Measurement { symbol, .. } = &mut p.operations[3].instruction {
        *symbol = 2;
    }
    invalid(p, "measurement symbol out of range");
    let mut p = plan.clone();
    if let Instruction::Measurement { record, .. } = &mut p.operations[3].instruction {
        *record = Some(2);
    }
    invalid(p, "record ordinal out of range");
    let mut p = plan.clone();
    p.operations.pop();
    invalid(p, "width profile length mismatch");
    let mut p = plan.clone();
    p.operations[3].instruction = Instruction::Clifford;
    invalid(p, "width transition mismatch");
    let mut p = plan;
    if let Instruction::Measurement { record, .. } = &mut p.operations[3].instruction {
        *record = None;
    }
    invalid(p, "missing symbol or record producer");
}

#[test]
fn affine_xor_canonicalizes_both_lists() {
    let a = AffineSign {
        constant: true,
        noise: vec![3, 1, 3, 2],
        measurements: vec![2, 0, 2, 1],
    };
    let b = AffineSign {
        constant: true,
        noise: vec![2, 4],
        measurements: vec![1, 3],
    };
    assert_eq!(
        a.xor(&b),
        AffineSign {
            constant: false,
            noise: vec![1, 4],
            measurements: vec![0, 3]
        }
    );
    for noise in [[false; 5], [true, false, true, false, true]] {
        for m in [[false; 4], [true, false, true, false]] {
            assert_eq!(
                a.xor(&b).evaluate(&noise, &m),
                a.evaluate(&noise, &m) ^ b.evaluate(&noise, &m)
            );
        }
    }
}
