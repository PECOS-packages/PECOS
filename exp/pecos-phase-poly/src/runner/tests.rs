// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::tests::{assert_state, dense_gate, dense_projection, index};
use pecos_core::{Angle64, Gate, MeasId, PauliString, channel, unitary};

mod dense;
mod distillation;

fn append(circuit: &mut TickCircuit, mut gate: Gate) {
    if gate.gate_type.consumes_measurement_record() {
        let ordinal = circuit
            .iter_gate_batches()
            .map(|g| g.meas_ids.len())
            .sum::<usize>();
        gate.meas_ids = (0..gate.qubits.len())
            .map(|i| MeasId::from_raw(1000 - ordinal - i))
            .collect();
    }
    let tick = circuit.num_ticks();
    circuit.tick();
    circuit.get_tick_mut(tick).unwrap().add_gate(gate);
}

#[test]
fn compile_errors_have_locations() {
    for gate in [
        Gate::simple(GateType::H, vec![QubitId(1)]),
        Gate::rz(Angle64::from_turns(0.01), &[1]),
        Gate::rzz(
            Angle64::from_turns(0.125) + Angle64::from_turns(1e-12),
            &[(1, 2)],
        ),
        Gate::channel(channel::from_unitary(unitary::H(1))),
    ] {
        let mut circuit = TickCircuit::new();
        circuit.tick().x(&[0]);
        circuit.tick().z(&[0]);
        circuit.get_tick_mut(1).unwrap().add_gate(gate.clone());
        let error = compile(&circuit).unwrap_err();
        assert!(
            matches!(&error, CompileError::Gate { gate_type, qubits, tick: 1, batch: 1, .. }
            if *gate_type == gate.gate_type && *qubits == gate.qubits.iter().map(QubitId::index).collect::<Vec<_>>())
        );
        assert!(error.to_string().contains("tick 1, batch 1"));
    }
    let mut circuit = TickCircuit::new();
    let refs = circuit.tick().mz(&[0]);
    circuit.observable(&refs).unwrap();
    circuit.get_tick_mut(0).unwrap().remove_gate(0);
    let error = compile(&circuit).unwrap_err();
    assert_eq!(
        error,
        CompileError::Annotation {
            annotation: 0,
            measurement: refs[0].meas_id
        }
    );
    assert!(error.to_string().contains("annotation 0"));

    let mut missing = TickCircuit::new();
    missing.tick();
    missing.get_tick_mut(0).unwrap().add_gate(Gate::mz(&[0]));
    assert!(matches!(
        compile(&missing),
        Err(CompileError::Gate {
            tick: 0,
            batch: 0,
            ..
        })
    ));
    let mut duplicate = TickCircuit::new();
    duplicate.tick().mz(&[0]);
    let gate = duplicate.get_tick(0).unwrap().gate_batches()[0].clone();
    duplicate.tick();
    duplicate.get_tick_mut(1).unwrap().add_gate(gate);
    assert!(matches!(
        compile(&duplicate),
        Err(CompileError::Gate {
            tick: 1,
            batch: 0,
            ..
        })
    ));
}

#[test]
fn empty_noop_batches_compile_and_run() {
    let mut circuit = TickCircuit::new();
    append(&mut circuit, Gate::simple(GateType::I, vec![]));
    circuit.tick().idle(5, &[] as &[usize]);
    circuit.tracked_pauli(PauliString::identity());
    let program = compile(&circuit).unwrap();
    assert_eq!(program.num_qubits(), 0);
    assert_eq!(
        program.run_shots(1, 7).unwrap(),
        vec![ShotResult {
            records: vec![],
            detectors: vec![],
            observables: vec![],
        }]
    );
    for kind in [GateType::X, GateType::MZ] {
        let mut circuit = circuit.clone();
        append(&mut circuit, Gate::simple(kind, vec![]));
        assert!(matches!(compile(&circuit), Err(CompileError::Gate {
            gate_type, tick: 3, batch: 0, ..
        }) if gate_type == kind));
    }
}

#[test]
fn records_annotations_reset_and_seeds() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    append(&mut circuit, Gate::mz(&[0, 1]));
    circuit.tick().px(&[1]);
    circuit.tick().z(&[1]);
    append(&mut circuit, Gate::mx(&[1]));
    append(&mut circuit, Gate::mpz(&[0]));
    append(&mut circuit, Gate::mz(&[0]));
    let a = circuit.meas_ref(1, 0, QubitId(0)).unwrap();
    let b = circuit.meas_ref(1, 0, QubitId(1)).unwrap();
    let c = circuit.meas_ref(4, 0, QubitId(1)).unwrap();
    let d = circuit.meas_ref(5, 0, QubitId(0)).unwrap();
    let e = circuit.meas_ref(6, 0, QubitId(0)).unwrap();
    circuit.detector(&[d, b, c]).unwrap();
    circuit.observable(&[e, b, c, c]).unwrap();
    circuit.detector(&[b, a]).unwrap();
    circuit.observable(&[]).unwrap();
    circuit.tracked_pauli(PauliString::identity());
    let program = Program::compile(&circuit).unwrap();
    assert_eq!((program.num_qubits(), program.num_records()), (2, 5));
    assert_eq!(program.detectors(), &[vec![3, 1, 2], vec![1, 0]]);
    assert_eq!(program.observables(), &[vec![4, 1, 2, 2], vec![]]);
    let mut sim = PhasePoly::with_seed(2, 7);
    let mut rng = PecosRng::seed_from_u64(8);
    for _ in 0..10 {
        sim.x(&[QubitId(0)]);
        let shot = program.run_shot(&mut sim, &mut rng).unwrap();
        assert_eq!(shot.records, [true, false, true, true, false]);
        assert_eq!(shot.detectors, [false, true]);
        assert_eq!(shot.observables, [false, false]);
    }
    assert_eq!(program.run_shots(20, 17), program.run_shots(20, 17));
    assert_eq!(program.run_shots(0, 17).unwrap(), []);
    let mut small = PhasePoly::with_seed(1, 0);
    small.x(&[QubitId(0)]);
    assert_eq!(
        program.run_shot(&mut small, &mut rng),
        Err(RunError::QubitCount {
            expected: 2,
            actual: 1
        })
    );
    assert!(small.mz(&[QubitId(0)])[0].outcome);
    assert_eq!(
        compile(&TickCircuit::new())
            .unwrap()
            .run_shots(2, 0)
            .unwrap()[0]
            .records,
        Vec::<bool>::new()
    );
}

fn parity_phase(circuit: &mut TickCircuit, support: &[usize], dagger: bool) {
    let (&target, controls) = support.split_last().unwrap();
    for &control in controls {
        circuit.tick().cx(&[(control, target)]);
    }
    if dagger {
        circuit.tick().tdg(&[target]);
    } else {
        circuit.tick().t(&[target]);
    }
    for &control in controls.iter().rev() {
        circuit.tick().cx(&[(control, target)]);
    }
}

#[test]
fn incompatible_ccz_measurement_reports_original_location() {
    let mut circuit = TickCircuit::new();
    circuit.tick().px(&[0, 1, 2]);
    for support in [vec![0], vec![1], vec![2], vec![0, 1, 2]] {
        parity_phase(&mut circuit, &support, false);
    }
    for support in [vec![0, 1], vec![0, 2], vec![1, 2]] {
        parity_phase(&mut circuit, &support, true);
    }
    let mut sim = PhasePoly::with_seed(3, 7);
    let mut rng = PecosRng::seed_from_u64(9);
    compile(&circuit)
        .unwrap()
        .run_shot(&mut sim, &mut rng)
        .unwrap();
    let mut expected = vec![num_complex::Complex64::new(1.0 / 8.0_f64.sqrt(), 0.0); 8];
    expected[7] = -expected[7];
    assert_state(&sim, &expected);
    let tick = circuit.num_ticks();
    append(&mut circuit, Gate::simple(GateType::I, vec![QubitId(2)]));
    let mut gate = Gate::mx(&[0]);
    gate.meas_ids.push(MeasId::from_raw(42));
    circuit.get_tick_mut(tick).unwrap().add_gate(gate);
    let error = compile(&circuit)
        .unwrap()
        .run_shot(&mut sim, &mut rng)
        .unwrap_err();
    assert!(
        matches!(&error, RunError::Measurement { tick: t, batch: 1, qubit: QubitId(0), .. } if *t == tick)
    );
    assert!(std::error::Error::source(&error).is_some());
    assert!(error.to_string().contains(&format!("tick {tick}, batch 1")));
    assert_state(&sim, &expected);
}

#[test]
fn noise_correlations_and_independent_rng_streams() {
    use pecos_core::ChannelExpr;
    let mut circuit = TickCircuit::new();
    circuit.tick().channel(ChannelExpr::MixedUnitary(vec![
        (0.5, unitary::I(0) & unitary::I(129)),
        (0.5, unitary::X(0) & unitary::Y(129)),
    ]));
    circuit.tick().mz(&[0, 129]);
    let program = compile(&circuit).unwrap();
    let shots = program.run_shots(512, 2026).unwrap();
    assert_eq!(shots, program.run_shots(512, 2026).unwrap());
    for shot in &shots {
        assert_eq!(shot.records[0], shot.records[1]);
    }
    let count = shots.iter().filter(|s| s.records[0]).count();
    assert!((180..330).contains(&count));
    let mut a = PhasePoly::with_seed(130, 1);
    let mut b = PhasePoly::with_seed(130, 999);
    let mut ra = PecosRng::seed_from_u64(4);
    let mut rb = ra.clone();
    for _ in 0..40 {
        assert_eq!(
            program.run_shot(&mut a, &mut ra),
            program.run_shot(&mut b, &mut rb)
        );
    }
}
