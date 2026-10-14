// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::joint_distribution::{
    Distribution, assert_distribution, assert_joint, conditional, fixed_noise, with_noise,
};
use super::*;
use crate::MeasurementCase;

pub(super) fn reordered(
    program: &HeisenbergProgram,
    order: &[usize],
) -> Result<HeisenbergProgram, ProgramError> {
    let mut sorted = order.to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..program.operations.len()).collect::<Vec<_>>());
    program.clone().with_operations(
        order
            .iter()
            .map(|&i| program.operations[i].clone())
            .collect(),
    )
}

fn body_and_sign(operation: &HeisenbergOp) -> (&VirtualPauli, &AffineSign) {
    match operation {
        HeisenbergOp::Rotation { pauli, sign, .. }
        | HeisenbergOp::Measurement { pauli, sign, .. } => (pauli, sign),
    }
}

pub(super) fn dependencies(program: &HeisenbergProgram, include_r2: bool) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); program.operations.len()];
    for (later, b) in program.operations.iter().enumerate() {
        let (pb, sb) = body_and_sign(b);
        for (earlier, a) in program.operations[..later].iter().enumerate() {
            let (pa, _) = body_and_sign(a);
            let r1 = !pa.bits().commutes_with(pb.bits());
            let r2 = include_r2
                && matches!(a, HeisenbergOp::Measurement { symbol, .. }
                    if sb.measurements.contains(symbol));
            if r1 || r2 {
                predecessors[later].push(earlier);
            }
        }
    }
    predecessors
}

pub(super) fn ready(predecessors: &[Vec<usize>], order: &[usize]) -> Vec<usize> {
    (0..predecessors.len())
        .filter(|i| !order.contains(i) && predecessors[*i].iter().all(|p| order.contains(p)))
        .collect()
}

fn all_orders(predecessors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    fn visit(predecessors: &[Vec<usize>], order: &mut Vec<usize>, orders: &mut Vec<Vec<usize>>) {
        if order.len() == predecessors.len() {
            orders.push(order.clone());
            return;
        }
        for i in ready(predecessors, order) {
            order.push(i);
            visit(predecessors, order, orders);
            order.pop();
        }
    }
    let mut orders = Vec::new();
    visit(predecessors, &mut Vec::new(), &mut orders);
    orders
}

#[test]
fn hidden_reset_swap() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().pz(&[0]);
    circuit.tick().mz(&[1]);
    circuit.tick().mz(&[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let swapped = reordered(&program, &[1, 0, 2]).unwrap();
    let expected = fixed_noise(&program, &[]);
    let actual = fixed_noise(&swapped, &[]);
    assert_distribution(
        &actual.records,
        &Distribution::from([(vec![false, false], 1.0)]),
    );
    assert_distribution(
        &actual.symbols,
        &Distribution::from([(vec![true, false, false], 1.0)]),
    );
    assert_joint(&actual, &expected);
}

#[test]
fn records_follow_ordinals() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().mz(&[0]);
    circuit.tick().mz(&[1]);
    let mut program = HeisenbergProgram::compile(&circuit).unwrap();
    program.detectors = vec![vec![0], vec![0, 1]];
    program.observables = vec![vec![1], vec![0]];
    let swapped = reordered(&program, &[1, 0]).unwrap();
    assert_distribution(
        &fixed_noise(&swapped, &[]).records,
        &Distribution::from([(vec![true, false], 1.0)]),
    );
    let shot = swapped.run(0);
    assert_eq!(shot.records, [true, false]);
    assert_eq!(shot.detectors, [true, true]);
    assert_eq!(shot.observables, [false, true]);
}

fn conditional_trace(program: &HeisenbergProgram) -> Vec<(usize, f64, MeasurementCase)> {
    let mut trace = Vec::new();
    program.execute(
        StabActive::with_seed(program.num_qubits, 0),
        &vec![false; program.num_noise_symbols],
        |state, pauli, negative, symbol, _| {
            let (p, case) = conditional(state, pauli, negative);
            trace.push((symbol, p, case));
            Some(false)
        },
    );
    trace
}

#[test]
fn correlated_measurements_swap_changes_conditionals() {
    let mut circuit = TickCircuit::new();
    circuit.tick().ry(Angle64::from_radians(0.73), &[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[0, 1]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let swapped = reordered(&program, &[0, 2, 1]).unwrap();
    assert_joint(&fixed_noise(&swapped, &[]), &fixed_noise(&program, &[]));
    let before = conditional_trace(&program);
    let after = conditional_trace(&swapped);
    assert!(
        before
            .iter()
            .any(|(_, _, case)| *case == MeasurementCase::Active)
    );
    assert!(
        after
            .iter()
            .any(|(_, _, case)| *case == MeasurementCase::Active)
    );
    assert_eq!(before[0].0, 0);
    assert_eq!(after[1].0, 0);
    assert!((before[0].1 - after[1].1).abs() > 1e-10);
    let p = (0.73_f64 / 2.0).sin().powi(2);
    assert_distribution(
        &fixed_noise(&program, &[]).symbols,
        &Distribution::from([(vec![false, false], 1.0 - p), (vec![true, true], p)]),
    );
}

#[test]
fn r2_violation_is_loud() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().mz(&[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    assert_eq!(
        reordered(&program, &[1, 0]).unwrap_err(),
        ProgramError::Causality {
            operation: 0,
            symbol: 0
        },
    );
}

#[test]
fn absent_symbol_names_the_symbol() {
    let mut program = two_measurements();
    let HeisenbergOp::Measurement { sign, .. } = &mut program.operations[0] else {
        panic!("expected measurement");
    };
    sign.measurements = vec![7];
    assert_eq!(
        program.validate(),
        Err(ProgramError::MeasurementReferenceOutOfRange {
            operation: 0,
            symbol: 7,
            limit: 2,
        }),
    );
}

#[test]
fn known_stochastic_distributions() {
    use pecos_core::{ChannelExpr, Pauli, PauliString, UnitaryRep};
    let mut circuit = TickCircuit::new();
    circuit.tick().rx(Angle64::from_radians(0.61), &[0]);
    circuit.tick().mz(&[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    let p = (0.61_f64 / 2.0).sin().powi(2);
    let expected = Distribution::from([(vec![false], 1.0 - p), (vec![true], p)]);
    let actual = fixed_noise(&program, &[]);
    assert_distribution(&actual.records, &expected);
    assert_distribution(&actual.symbols, &expected);

    let mut noisy = TickCircuit::new();
    noisy.tick().channel(ChannelExpr::MixedUnitary(vec![
        (
            0.6,
            UnitaryRep::Pauli(PauliString::from_paulis(&[Pauli::I, Pauli::I])),
        ),
        (
            0.3,
            UnitaryRep::Pauli(PauliString::from_paulis(&[Pauli::X, Pauli::X])),
        ),
        (
            0.1,
            UnitaryRep::Pauli(PauliString::from_paulis(&[Pauli::X, Pauli::I])),
        ),
    ]));
    noisy.tick().channel(channel::BitFlip(0.2, 1));
    noisy.tick().mz(&[0, 1]);
    let program = HeisenbergProgram::compile(&noisy).unwrap();
    // II/XX/XI followed by an independent IX fault: 0.6*0.8,
    // 0.6*0.2, 0.3*0.2 + 0.1*0.8, 0.3*0.8 + 0.1*0.2.
    let expected = Distribution::from([
        (vec![false, false], 0.48),
        (vec![false, true], 0.12),
        (vec![true, false], 0.14),
        (vec![true, true], 0.26),
    ]);
    let actual = with_noise(&program);
    assert_distribution(&actual.records, &expected);
    assert_distribution(&actual.symbols, &expected);
    assert_distribution(
        &fixed_noise(&program, &[true, false, true, false, false, false]).records,
        &Distribution::from([(vec![true, true], 1.0)]),
    );
}

fn exhaustive_witness(program: &HeisenbergProgram) {
    let expected = with_noise(program);
    let predecessors = dependencies(program, true);
    let orders = all_orders(&predecessors);
    assert_ne!(orders.len(), 0);
    for order in &orders {
        let candidate = reordered(program, order)
            .unwrap_or_else(|error| panic!("dependency order {order:?} is invalid: {error}"));
        assert_joint(&with_noise(&candidate), &expected);
    }
    // Without (R2), every additional order must be rejected at construction.
    let without_r2 = all_orders(&dependencies(program, false));
    let mut rejected = 0;
    for order in &without_r2 {
        match reordered(program, order) {
            Ok(candidate) => assert_joint(&with_noise(&candidate), &expected),
            Err(error) => {
                assert!(matches!(error, ProgramError::Causality { .. }), "{error}");
                rejected += 1;
            }
        }
    }
    assert!(rejected > 0);
    assert_eq!(without_r2.len(), orders.len() + rejected);
}

#[test]
fn exhaustive_reset_orders() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().mz(&[0]);
    exhaustive_witness(&HeisenbergProgram::compile(&circuit).unwrap());
}

#[test]
fn exhaustive_reset_then_active_orders() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().pz(&[0]);
    circuit.tick().rx(Angle64::from_radians(0.43), &[1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[1]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    // The virtual readout Z0 Z1 commutes with reset Z0, but anticommutes
    // with its X0 correction. Qubit 1 makes the readout Active after reset.
    let (reset, _) = body_and_sign(&program.operations[0]);
    let (readout, sign) = body_and_sign(&program.operations[2]);
    assert!(reset.bits().commutes_with(readout.bits()));
    assert_eq!(sign.measurements, [0]);
    assert!(
        conditional_trace(&program)
            .iter()
            .any(|(symbol, _, case)| { *symbol == 1 && *case == MeasurementCase::Active })
    );
    exhaustive_witness(&program);
}

fn two_measurements() -> HeisenbergProgram {
    let mut circuit = TickCircuit::new();
    circuit.tick().mz(&[0, 1]);
    HeisenbergProgram::compile(&circuit).unwrap()
}

#[test]
fn duplicate_symbol() {
    let mut program = two_measurements();
    let HeisenbergOp::Measurement { symbol, .. } = &mut program.operations[1] else {
        panic!("expected measurement");
    };
    *symbol = 0;
    assert_eq!(
        program.validate(),
        Err(ProgramError::DuplicateSymbol {
            operation: 1,
            symbol: 0
        })
    );
}

#[test]
fn duplicate_ordinal() {
    let mut program = two_measurements();
    let HeisenbergOp::Measurement { record, .. } = &mut program.operations[1] else {
        panic!("expected measurement");
    };
    *record = Some(0);
    assert_eq!(
        program.validate(),
        Err(ProgramError::DuplicateOrdinal {
            operation: 1,
            ordinal: 0
        })
    );
}

#[test]
fn missing_ordinal() {
    let mut program = two_measurements();
    program.num_records += 1;
    assert_eq!(
        program.validate(),
        Err(ProgramError::MissingOrdinal { ordinal: 2 })
    );
}

#[test]
fn missing_symbol() {
    let mut program = two_measurements();
    program.num_measurements += 1;
    assert_eq!(
        program.validate(),
        Err(ProgramError::MissingSymbol { symbol: 2 })
    );
}

#[test]
fn deterministic_branches_and_wide_register() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[129]);
    circuit.tick().mz(&[129]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    assert_distribution(
        &with_noise(&program).symbols,
        &Distribution::from([(vec![true], 1.0)]),
    );
    let mut circuit = TickCircuit::new();
    circuit.tick().rx(Angle64::from_radians(0.31), &[0]);
    circuit.tick().rx(Angle64::from_radians(-0.31), &[0]);
    circuit.tick().mz(&[0]);
    let program = HeisenbergProgram::compile(&circuit).unwrap();
    assert_eq!(conditional_trace(&program)[0].2, MeasurementCase::Active);
    assert_distribution(
        &with_noise(&program).symbols,
        &Distribution::from([(vec![false], 1.0)]),
    );
}

#[test]
#[should_panic(expected = "joint oracle exceeds 4096 noise/measurement branches")]
fn joint_noise_measurement_branch_bound() {
    let mut circuit = TickCircuit::new();
    for _ in 0..3 {
        circuit.tick().channel(channel::BitFlip(0.2, 0));
    }
    // Eight noise alternatives times 1024 measurement leaves exceeds the bound;
    // neither source of branching exceeds it on its own.
    for _ in 0..10 {
        circuit.tick().h(&[0]);
        circuit.tick().mz(&[0]);
    }
    let _ = with_noise(&HeisenbergProgram::compile(&circuit).unwrap());
}

#[test]
#[should_panic(expected = "joint oracle exceeds 4096 noise/measurement branches")]
fn noise_alternative_branch_bound() {
    let mut circuit = TickCircuit::new();
    for _ in 0..13 {
        circuit.tick().channel(channel::BitFlip(0.2, 0));
    }
    let _ = with_noise(&HeisenbergProgram::compile(&circuit).unwrap());
}

pub(super) fn index(rng: &mut PecosRng, n: usize) -> usize {
    usize::try_from(rng.next_u64() % u64::try_from(n).unwrap()).unwrap()
}

pub(super) fn random_circuit(rng: &mut PecosRng) -> TickCircuit {
    let kinds = [
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
        GateType::RX,
        GateType::RY,
        GateType::RZ,
        GateType::RXX,
        GateType::RYY,
        GateType::RZZ,
        GateType::MZ,
        GateType::MX,
        GateType::MPZ,
        GateType::PZ,
        GateType::PX,
    ];
    let mut circuit = TickCircuit::new();
    append(&mut circuit, make_gate(GateType::PZ, &[0], &[]));
    for step in 0..8 {
        let q = index(rng, 3);
        let r = (q + 1 + index(rng, 2)) % 3;
        let gate = if step == 3 {
            if rng.next_bool_fast() {
                Gate::channel(channel::Depolarizing2(0.23, q, r))
            } else {
                Gate::channel(channel::PauliChannel(0.07, 0.11, 0.13, q))
            }
        } else {
            let kind = kinds[index(rng, kinds.len())];
            let qs = &([q, r])[..kind.quantum_arity()];
            let angles = if kind.angle_arity() == 0 {
                vec![]
            } else {
                vec![Angle64::from_radians(6.0 * rng.next_f64() - 3.0)]
            };
            make_gate(kind, qs, &angles)
        };
        append(&mut circuit, gate);
    }
    append(&mut circuit, make_gate(GateType::MZ, &[0, 1, 2], &[]));
    circuit
}

fn measurement_order(program: &HeisenbergProgram) -> Vec<usize> {
    program
        .operations
        .iter()
        .filter_map(|op| match op {
            HeisenbergOp::Measurement { symbol, .. } => Some(*symbol),
            HeisenbergOp::Rotation { .. } => None,
        })
        .collect()
}

#[test]
fn random_legal_reorders() {
    let mut rng = PecosRng::seed_from_u64(92041);
    let mut changed = 0;
    let mut measurements_changed = 0;
    let circuits = 60;
    let orders_per_circuit = 6;
    for _ in 0..circuits {
        let program = HeisenbergProgram::compile(&random_circuit(&mut rng)).unwrap();
        let expected = with_noise(&program);
        let predecessors = dependencies(&program, true);
        for _ in 0..orders_per_circuit {
            let mut order = Vec::new();
            while order.len() < program.operations.len() {
                let candidates = ready(&predecessors, &order);
                assert_ne!(candidates.len(), 0);
                order.push(candidates[index(&mut rng, candidates.len())]);
            }
            changed += usize::from(order.iter().copied().ne(0..program.operations.len()));
            let candidate = reordered(&program, &order).unwrap();
            measurements_changed +=
                usize::from(measurement_order(&candidate) != measurement_order(&program));
            assert_joint(&with_noise(&candidate), &expected);
        }
    }
    println!(
        "random reorders: {circuits} circuits, {} orders, {changed} changed, \
         {measurements_changed} changed measurement order",
        circuits * orders_per_circuit
    );
    assert!(measurements_changed > 0);
}

#[test]
fn compiled_random_programs_are_valid() {
    let mut rng = PecosRng::seed_from_u64(19381);
    for _ in 0..240 {
        let program = HeisenbergProgram::compile(&random_circuit(&mut rng)).unwrap();
        assert_eq!(program.validate(), Ok(()));
    }
}
