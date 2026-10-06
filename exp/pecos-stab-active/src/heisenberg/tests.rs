// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

mod joint_distribution;
mod oracle;
mod reordering;
mod sampling;
mod validation;
mod width_profile;

use super::*;
use crate::{ArbitraryRotationGateable, CliffordGateable, PecosRng};
use pecos_core::{Gate, MeasId, QubitId, channel, gate_type::GateType};
use pecos_simulators::{DenseStateVec, ForcedMeasurement, SparseStabY};

fn append(circuit: &mut TickCircuit, mut gate: Gate) {
    if gate.gate_type.consumes_measurement_record() {
        let ordinal: usize = circuit.iter_gate_batches().map(|g| g.meas_ids.len()).sum();
        gate.meas_ids = (0..gate.qubits.len())
            .map(|i| MeasId::from_raw(7 + 3 * (ordinal + i)))
            .collect();
    }
    let tick = circuit.num_ticks();
    circuit.tick();
    circuit.get_tick_mut(tick).unwrap().add_gate(gate);
}

fn make_gate(kind: GateType, qubits: &[usize], angles: &[Angle64]) -> Gate {
    Gate::with_angles(
        kind,
        angles.to_vec(),
        qubits.iter().copied().map(QubitId).collect::<Vec<_>>(),
    )
}

#[test]
fn phase_normalization_and_word_symplectic() {
    use PauliKindForDecomp::{X, Y};
    let mut frame = SparseStabY::new(130).with_destab_sign_tracking();
    for physical in [vec![(0, Y)], vec![(0, Y), (129, Y)]] {
        let (pauli, sign) = builder::pullback(&frame, &physical);
        assert_eq!(pauli.factors(), physical);
        assert!(!sign);
    }
    frame.sz(&[QubitId(0)]);
    let (pauli, sign) = builder::pullback(&frame, &[(0, X)]);
    assert_eq!(pauli.factors(), &[(0, Y)]);
    assert!(sign); // S† X S = -Y.
    let (a, _) = builder::pullback(&frame, &[(129, X)]);
    let (b, _) = builder::pullback(&frame, &[(129, Y)]);
    assert!(!a.bits().commutes_with(b.bits()));
}

#[test]
fn repeated_mx_and_reset_fault_cancellation() {
    let mut circuit = TickCircuit::new();
    circuit.tick().mx(&[0]);
    circuit.tick().mx(&[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    for seed in 0..100 {
        let shot = program.run(seed);
        assert_eq!(shot.records[0], shot.records[1]);
    }
    let mut circuit = TickCircuit::new();
    circuit.tick().channel(channel::BitFlip(1.0, 0));
    circuit.tick().pz(&[0]);
    circuit.tick().mz(&[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let bits = vec![true, false];
    let shot = program.execute(
        StabActive::with_seed(1, 0),
        &bits,
        |_, _, _, symbol, record| {
            if symbol == 0 {
                assert!(record.is_none());
                Some(true)
            } else {
                Some(false)
            }
        },
    );
    assert_eq!(shot.records, [false]);
    assert_eq!(program.run(7).records, [false]);
    let HeisenbergOp::Measurement { sign, .. } = &program.operations[1] else {
        panic!()
    };
    assert_eq!(sign.noise, [0]);
    assert_eq!(sign.measurements, [0]);
}

#[test]
fn exact_quarter_turns_and_nearby_angles() {
    let axes = [
        GateType::RX,
        GateType::RY,
        GateType::RZ,
        GateType::RXX,
        GateType::RYY,
        GateType::RZZ,
    ];
    for axis in axes {
        for base in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::THREE_QUARTERS_TURN,
        ] {
            for delta in [
                Angle64::ZERO,
                Angle64::from_turns(1e-12),
                -Angle64::from_turns(1e-12),
            ] {
                let mut circuit = TickCircuit::new();
                let qs = &([0, 1])[..axis.quantum_arity()];
                append(&mut circuit, make_gate(axis, qs, &[base + delta]));
                let program = HeisenbergProgram::compile(&circuit).unwrap();
                assert_eq!(
                    program.operations.len(),
                    usize::from(delta != Angle64::ZERO)
                );
                for readout in [GateType::MZ, GateType::MX] {
                    let mut probe = TickCircuit::new();
                    probe.tick().rx(Angle64::from_radians(0.31), qs);
                    probe.tick().ry(Angle64::from_radians(0.37), qs);
                    append(&mut probe, make_gate(axis, qs, &[base + delta]));
                    append(&mut probe, make_gate(readout, qs, &[]));
                    oracle::compare(&probe, 53);
                }
            }
        }
    }
}

#[test]
fn rejected_kinds_have_locations_at_every_angle() {
    let rejected = [
        GateType::U,
        GateType::RXY1Q,
        GateType::RXYXY2Q,
        GateType::RXXRYYRZZ,
        GateType::U2q,
        GateType::T,
        GateType::Tdg,
        GateType::CH,
        GateType::CCX,
        GateType::MeasureLeaked,
        GateType::MeasureFree,
        GateType::Custom,
        GateType::QAlloc,
        GateType::QFree,
        GateType::MeasCrosstalkGlobalPayload,
        GateType::MeasCrosstalkLocalPayload,
    ];
    for kind in rejected {
        for angle in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::from_radians(0.123),
        ] {
            let mut circuit = TickCircuit::new();
            circuit.tick().h(&[4]);
            circuit.tick().h(&[4]);
            let qs: Vec<_> = (0..kind.quantum_arity().max(1)).collect();
            let gate = make_gate(kind, &qs, &vec![angle; kind.angle_arity()]);
            circuit.get_tick_mut(1).unwrap().add_gate(gate);
            assert!(
                matches!(HeisenbergProgram::compile(&circuit), Err(CompileError::Gate {
                gate_type, tick: 1, batch: 1, qubits, .. }) if gate_type == kind && qubits == qs)
            );
        }
    }
    for expression in [
        channel::Leakage(0.1, 0),
        channel::AmplitudeDamping(0.2, 0),
        channel::PhaseDamping(0.1, 0),
        channel::Erasure(0.1, 0),
    ] {
        let mut circuit = TickCircuit::new();
        circuit.tick().channel(expression);
        assert!(matches!(
            HeisenbergProgram::compile(&circuit),
            Err(CompileError::Gate {
                gate_type: GateType::Channel,
                tick: 0,
                batch: 0,
                ..
            })
        ));
    }
}

#[test]
fn named_clifford_frames_match_physical_trait_contract() {
    for kind in [
        GateType::H,
        GateType::F,
        GateType::Fdg,
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
        GateType::CX,
        GateType::CY,
        GateType::CZ,
        GateType::SXX,
        GateType::SXXdg,
        GateType::SYY,
        GateType::SYYdg,
        GateType::SZZ,
        GateType::SZZdg,
        GateType::SWAP,
    ] {
        let mut physical = StabActive::with_seed(2, 0);
        let mut frame = SparseStabY::new(2).with_destab_sign_tracking();
        let qs = &([QubitId(0), QubitId(1)])[..kind.quantum_arity()];
        dispatch::apply_clifford(&mut physical, kind, qs);
        dispatch::apply_clifford(&mut frame, kind, qs);
        for q in 0..2 {
            for axis in [
                PauliKindForDecomp::X,
                PauliKindForDecomp::Y,
                PauliKindForDecomp::Z,
            ] {
                assert_eq!(
                    builder::pullback(&frame, &[(q, axis)]),
                    builder::pullback(physical.structure.tableau(), &[(q, axis)]),
                    "{kind:?} {q} {axis:?}"
                );
            }
        }
    }
}

fn extended_clifford<S: CliffordGateable>(sim: &mut S, variant: usize, qs: &[QubitId]) {
    match variant {
        0 => {
            sim.h2(qs);
        }
        1 => {
            sim.h3(qs);
        }
        2 => {
            sim.h4(qs);
        }
        3 => {
            sim.h5(qs);
        }
        4 => {
            sim.h6(qs);
        }
        5 => {
            sim.f2(qs);
        }
        6 => {
            sim.f2dg(qs);
        }
        7 => {
            sim.f3(qs);
        }
        8 => {
            sim.f3dg(qs);
        }
        9 => {
            sim.f4(qs);
        }
        10 => {
            sim.f4dg(qs);
        }
        _ => panic!(),
    }
}

#[test]
fn signed_frames_on_entangled_prefixes() {
    let mut physical = StabActive::with_seed(3, 0);
    let mut frame = SparseStabY::new(3).with_destab_sign_tracking();
    let mut rng = PecosRng::seed_from_u64(772);
    let gates = [
        GateType::H,
        GateType::SX,
        GateType::SZdg,
        GateType::SXX,
        GateType::SYY,
        GateType::SZZ,
        GateType::SZZdg,
        GateType::SXXdg,
        GateType::SYYdg,
        GateType::F,
        GateType::Fdg,
        GateType::CX,
    ];
    for step in 0..500 {
        let q = usize::try_from(rng.next_u64() % 3).unwrap();
        let qs = [QubitId(q), QubitId((q + 1) % 3)];
        if step % 3 == 0 {
            let variant = usize::try_from(rng.next_u64() % 11).unwrap();
            extended_clifford(&mut physical, variant, &qs[..1]);
            extended_clifford(&mut frame, variant, &qs[..1]);
        } else {
            let kind = gates[usize::try_from(rng.next_u64() % 12).unwrap()];
            dispatch::apply_clifford(&mut physical, kind, &qs[..kind.quantum_arity()]);
            dispatch::apply_clifford(&mut frame, kind, &qs[..kind.quantum_arity()]);
        }
        for q in 0..3 {
            for axis in [
                PauliKindForDecomp::X,
                PauliKindForDecomp::Y,
                PauliKindForDecomp::Z,
            ] {
                assert_eq!(
                    builder::pullback(&frame, &[(q, axis)]),
                    builder::pullback(physical.structure.tableau(), &[(q, axis)]),
                    "step {step}"
                );
            }
        }
    }
}

#[test]
fn batched_rotations_measurements_and_post_measurement_noise() {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0, 1, 2, 3]);
    circuit
        .tick()
        .ry(Angle64::from_radians(0.28), &[0, 1, 2, 3]);
    circuit
        .tick()
        .rxx(Angle64::from_radians(0.57), &[(0, 1), (2, 3)]);
    circuit.tick().mx(&[3, 1]);
    circuit.tick().px(&[1, 3]);
    circuit.tick().mpz(&[0, 2]);
    circuit.tick().channel(channel::BitFlip(1.0, 0));
    circuit.tick().mz(&[0, 1, 2, 3]);
    oracle::compare(&circuit, 19);
    let mut witness = TickCircuit::new();
    witness.tick().mz(&[0]);
    witness.tick().channel(channel::BitFlip(1.0, 0));
    witness.tick().mz(&[0]);
    assert_eq!(
        HeisenbergProgram::compile(&witness).unwrap().run(0).records,
        [false, true]
    );
}

#[test]
fn signed_general_paulis_and_empty_support() {
    use PauliKindForDecomp::{X, Y, Z};
    for negative in [false, true] {
        for pauli in [
            vec![],
            vec![(0, Y)],
            vec![(0, Y), (1, Y)],
            vec![(0, X), (1, Z), (2, Y)],
        ] {
            let mut state = StabActive::with_seed(3, 81);
            state.rotate_pauli(Angle64::from_radians(0.23), &pauli, negative);
            let first = state.measure_pauli(&pauli, negative);
            let second = state.measure_pauli(&pauli, negative);
            assert_eq!(first.outcome, second.outcome);
            assert!(second.is_deterministic);
            if pauli.is_empty() {
                assert_eq!(first.outcome, negative);
            }
        }
    }
}

#[test]
fn malformed_channels_records_and_annotations_are_rejected() {
    let mut circuit = TickCircuit::new();
    circuit.tick().mz(&[0]);
    circuit.tick().mz(&[1]);
    let mut gate = circuit.get_tick(1).unwrap().gate_batches()[0].clone();
    gate.meas_ids[0] = MeasId::from_raw(0);
    circuit
        .get_tick_mut(1)
        .unwrap()
        .replace_gate_batch(0, gate)
        .unwrap();
    assert!(matches!(
        HeisenbergProgram::compile(&circuit),
        Err(CompileError::Gate { tick: 1, .. })
    ));
    let mut circuit = TickCircuit::new();
    let refs = circuit.tick().mz(&[0]);
    circuit.detector(&refs).unwrap();
    circuit.get_tick_mut(0).unwrap().remove_gate(0);
    assert!(matches!(
        HeisenbergProgram::compile(&circuit),
        Err(CompileError::Annotation { annotation: 0, .. })
    ));
    for p in [f64::NAN, -0.1, 1.1] {
        let expression = pecos_core::ChannelExpr::MixedUnitary(vec![(
            p,
            pecos_core::UnitaryRep::Pauli(pecos_core::PauliString::from_paulis(&[
                pecos_core::Pauli::X,
            ])),
        )]);
        let mut circuit = TickCircuit::new();
        circuit.tick().channel(expression);
        assert!(HeisenbergProgram::compile(&circuit).is_err());
    }
}

#[test]
fn explicit_decomposition_phases() {
    use pecos_stab_tn::stab_mps::pauli_decomp::decompose_pauli_string;
    let frame = SparseStabY::new(2).with_destab_sign_tracking();
    for (physical, expected) in [
        (
            vec![(0, PauliKindForDecomp::Y)],
            num_complex::Complex64::new(0.0, 1.0),
        ),
        (
            vec![(0, PauliKindForDecomp::Y), (1, PauliKindForDecomp::Y)],
            num_complex::Complex64::new(-1.0, 0.0),
        ),
    ] {
        assert_eq!(
            decompose_pauli_string(frame.stabs(), frame.destabs(), &physical).2,
            expected
        );
        assert!(!builder::pullback(&frame, &physical).1);
    }
}

#[test]
fn typed_with_noise_tiny_probabilities_and_composed_channels() {
    let mut circuit = TickCircuit::new();
    circuit.tick().mz(&[0]);
    circuit.tick().mz(&[0]);
    let noisy = circuit.with_noise(&|gate: &Gate| vec![channel::BitFlip(1.0, gate.qubits[0])]);
    assert_eq!(
        HeisenbergProgram::compile(&noisy).unwrap().run(7).records,
        [false, true]
    );
    let mut tiny = TickCircuit::new();
    tiny.tick().channel(channel::BitFlip(1e-15, 129));
    let program = HeisenbergProgram::compile(&tiny).unwrap();
    assert_eq!(program.num_qubits(), 130);
    assert_eq!(
        program.noise_channels[0].alternatives[1].0.to_bits(),
        1e-15_f64.to_bits()
    );
    let mut composed = TickCircuit::new();
    composed
        .tick()
        .channel(pecos_core::ChannelExpr::Compose(vec![
            channel::BitFlip(1.0, 0),
            channel::BitFlip(1.0, 0),
        ]));
    composed
        .tick()
        .channel(channel::BitFlip(1.0, 0) & channel::BitFlip(1.0, 1));
    composed.tick().mz(&[0, 1]);
    assert_eq!(
        HeisenbergProgram::compile(&composed)
            .unwrap()
            .run(7)
            .records,
        [true, true]
    );
    let mut non_pauli = TickCircuit::new();
    non_pauli
        .tick()
        .channel(channel::from_unitary(pecos_core::unitary::H(0)));
    assert!(matches!(
        HeisenbergProgram::compile(&non_pauli),
        Err(CompileError::Gate {
            gate_type: GateType::Channel,
            ..
        })
    ));
}

#[test]
fn stable_ids_can_execute_in_reverse_numeric_order() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().mz(&[0, 1]);
    let mut gate = circuit.get_tick(1).unwrap().gate_batches()[0].clone();
    gate.meas_ids = vec![MeasId::from_raw(91), MeasId::from_raw(12)].into();
    circuit
        .get_tick_mut(1)
        .unwrap()
        .replace_gate_batch(0, gate)
        .unwrap();
    let first = circuit.meas_ref(1, 0, QubitId(0)).unwrap();
    let second = circuit.meas_ref(1, 0, QubitId(1)).unwrap();
    circuit.detector(&[first, second, second]).unwrap();
    circuit.observable(&[second]).unwrap();
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    assert_eq!(program.run(1).detectors, [true]);
    assert_eq!(program.run(1).observables, [false]);
}

#[test]
fn exact_symbolic_pauli_channel_alternatives() {
    use pecos_core::{ChannelExpr, UnitaryRep, unitary};
    let product = UnitaryRep::Compose(vec![unitary::X(0), unitary::Y(0)]);
    let adjoint = UnitaryRep::Adjoint(Box::new(product));
    let half_turn = unitary::RX(Angle64::HALF_TURN, 1);
    let mut circuit = TickCircuit::new();
    circuit
        .tick()
        .channel(ChannelExpr::MixedUnitary(vec![(1.0, adjoint)]));
    circuit.tick().channel(channel::from_unitary(half_turn));
    circuit.tick().mz(&[0, 1]);
    assert_eq!(
        HeisenbergProgram::compile(&circuit).unwrap().run(1).records,
        [false, true]
    );
    let mut rejected = TickCircuit::new();
    rejected.tick().channel(channel::from_unitary(unitary::RX(
        Angle64::HALF_TURN + Angle64::from_turns(1e-12),
        0,
    )));
    assert!(HeisenbergProgram::compile(&rejected).is_err());
}
