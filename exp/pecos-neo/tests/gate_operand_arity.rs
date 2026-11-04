use pecos_core::{Angle64, QubitId};
use pecos_neo::extensible::{
    Batch, BatchError, BatchExecutor, BatchedCircuit, DecompEntry, DecompOp, Decomposition,
    GateDefinitions, GateId, GateSpec, GateSupportSet, MeasBasis, OpBuilder, PrepBasis,
    ResolvedCircuit, ResolvedOp, ResultId, SimpleExecutor, gates,
};
use pecos_neo::runner::{CircuitRunner, ExecutionError};
use pecos_simulators::{CliffordGateable, SparseStab};
use std::sync::Arc;

fn qubits(count: usize) -> Vec<QubitId> {
    (0..count).map(QubitId).collect()
}

fn bits(state: &mut SparseStab, count: usize) -> Vec<bool> {
    state.mz(&qubits(count)).iter().map(|r| r.outcome).collect()
}

fn resolved_gate(gate_id: GateId, count: usize) -> ResolvedOp {
    ResolvedOp::Gate {
        gate_id,
        qubits: qubits(count).into_iter().collect(),
        angles: smallvec::smallvec![],
    }
}

#[test]
fn adapted_cx_rejects_partial_group_without_mutation_and_executes_every_pair() {
    let mut state = SparseStab::with_seed(4, 42);
    state.x(&[QubitId(0), QubitId(2)]);
    let mut runner = CircuitRunner::new();
    let bad = OpBuilder::new().gate(gates::CX, &qubits(3), &[]).build();
    let error = runner.apply_adapted_circuit(&mut state, &bad).unwrap_err();
    assert!(matches!(
        error,
        ExecutionError::QubitArity {
            arity: 2,
            got: 3,
            ..
        }
    ));
    assert!(error.to_string().contains("CX"));
    assert_eq!(bits(&mut state, 4), [true, false, true, false]);
    let good = OpBuilder::new().gate(gates::CX, &qubits(4), &[]).build();
    runner.apply_adapted_circuit(&mut state, &good).unwrap();
    assert_eq!(bits(&mut state, 4), [true, true, true, true]);
}

struct Executor {
    state: SparseStab,
    calls: Vec<(GateId, Vec<QubitId>, Vec<Angle64>)>,
}

impl SimpleExecutor for Executor {
    type MeasurementResults = Vec<bool>;
    fn execute_gate(&mut self, gate: GateId, qubits: &[QubitId], angles: &[Angle64]) {
        self.calls.push((gate, qubits.to_vec(), angles.to_vec()));
        match gate {
            gates::H => {
                self.state.h(qubits);
            }
            gates::CX => {
                self.state.cx(&[(qubits[0], qubits[1])]);
            }
            _ => {}
        }
    }
    fn execute_prep(&mut self, _: PrepBasis, _: QubitId) {}
    fn execute_measure(&mut self, _: MeasBasis, _: QubitId, _: ResultId, _: &mut Vec<bool>) {}
    fn get_result(&self, _: ResultId, _: &Vec<bool>) -> bool {
        true
    }
}

fn executor() -> Executor {
    Executor {
        state: SparseStab::with_seed(4, 42),
        calls: Vec::new(),
    }
}

#[test]
fn batching_groups_native_gates_including_extensions() {
    let resolved = ResolvedCircuit::new(vec![
        ResolvedOp::gate1(gates::H, QubitId(3)),
        resolved_gate(gates::H, 2),
        ResolvedOp::gate2(gates::CX, QubitId(2), QubitId(3)),
        resolved_gate(gates::CX, 4),
    ]);
    let batched = BatchedCircuit::from_resolved(&resolved, &GateDefinitions::new()).unwrap();
    assert!(matches!(&batched.batches[0], Batch::SingleQubit { qubits, .. } if qubits.len() == 3));
    assert!(matches!(&batched.batches[1], Batch::TwoQubit { pairs, .. } if pairs.len() == 3));
    let h = BatchedCircuit::from_resolved(
        &ResolvedCircuit::new(vec![resolved_gate(gates::H, 2)]),
        &GateDefinitions::new(),
    )
    .unwrap();
    let mut exec = executor();
    exec.execute_batched(&h);
    // H on both again must restore |00>; a dropped operand stays in |+>.
    exec.state.h(&[QubitId(0), QubitId(1)]);
    let measured = exec.state.mz(&[QubitId(0), QubitId(1)]);
    assert!(measured.iter().all(|r| r.is_deterministic && !r.outcome));
    let cx = BatchedCircuit::from_resolved(
        &ResolvedCircuit::new(vec![resolved_gate(gates::CX, 4)]),
        &GateDefinitions::new(),
    )
    .unwrap();
    exec.state.x(&[QubitId(0), QubitId(2)]);
    exec.execute_batched(&cx);
    assert_eq!(bits(&mut exec.state, 4), [true, true, true, true]);
}

#[test]
fn batching_rejects_partial_groups_even_in_extensions_and_conditionals() {
    for ops in [
        vec![resolved_gate(gates::CX, 3)],
        vec![resolved_gate(gates::CX, 2), resolved_gate(gates::CX, 3)],
        vec![ResolvedOp::Conditional {
            condition: ResultId(0),
            if_one: vec![],
            if_zero: vec![resolved_gate(gates::CX, 3)],
        }],
    ] {
        assert_eq!(
            BatchedCircuit::from_resolved(&ResolvedCircuit::new(ops), &GateDefinitions::new())
                .unwrap_err(),
            BatchError::QubitArity {
                gate_id: gates::CX,
                arity: 2,
                got: 3
            }
        );
    }
}

#[test]
fn registered_user_arity_controls_execution_and_batching() {
    let mut defs = GateDefinitions::new();
    let gate = defs.register(GateSpec::new("paired_flip").with_quantum_arity(2));
    defs.set_decomposition(
        gate,
        DecompEntry {
            requires: GateSupportSet::from_iter([gates::X]),
            decomposition: Decomposition::Dynamic(Arc::new(vec![DecompOp::gate1(gates::X, 1)])),
        },
    );
    let mut runner = CircuitRunner::with_definitions(defs.clone());
    let mut state = SparseStab::with_seed(4, 42);
    let bad = OpBuilder::new().gate(gate, &qubits(3), &[]).build();
    assert!(
        matches!(runner.apply_adapted_circuit(&mut state, &bad), Err(ExecutionError::DecompositionQubitArity { gate_id, arity: 2, got: 3 }) if gate_id == gate)
    );
    assert_eq!(bits(&mut state, 4), [false; 4]);
    let good = OpBuilder::new().gate(gate, &qubits(4), &[]).build();
    runner.apply_adapted_circuit(&mut state, &good).unwrap();
    assert_eq!(bits(&mut state, 4), [false, true, false, true]);
    assert!(matches!(
        BatchedCircuit::from_resolved(&ResolvedCircuit::new(vec![resolved_gate(gate, 3)]), &defs),
        Err(BatchError::QubitArity { .. })
    ));
    let batched =
        BatchedCircuit::from_resolved(&ResolvedCircuit::new(vec![resolved_gate(gate, 4)]), &defs)
            .unwrap();
    let mut exec = executor();
    exec.execute_batched(&batched);
    assert_eq!(
        exec.calls,
        vec![
            (gate, qubits(2), vec![]),
            (gate, vec![QubitId(2), QubitId(3)], vec![])
        ]
    );
}

#[test]
fn general_gates_and_conditional_groups_reach_executor() {
    let defs = GateDefinitions::new();
    let ops = vec![
        resolved_gate(gates::CCX, 6),
        ResolvedOp::Conditional {
            condition: ResultId(0),
            if_one: vec![resolved_gate(gates::H, 2)],
            if_zero: vec![],
        },
    ];
    let circuit = BatchedCircuit::from_resolved(&ResolvedCircuit::new(ops), &defs).unwrap();
    let mut exec = executor();
    exec.execute_batched(&circuit);
    assert_eq!(
        exec.calls
            .iter()
            .map(|(_, q, _)| q.len())
            .collect::<Vec<_>>(),
        [3, 3, 1, 1]
    );
    assert_eq!(
        BatchedCircuit::from_resolved(
            &ResolvedCircuit::new(vec![resolved_gate(GateId(1234), 2)]),
            &defs
        )
        .unwrap_err(),
        BatchError::UnknownArity {
            gate_id: GateId(1234)
        }
    );
}

#[test]
fn malformed_decomposition_child_reports_qubit_arity() {
    let mut defs = GateDefinitions::new();
    let gate = defs.register(GateSpec::new("invalid_child").with_quantum_arity(3));
    defs.set_decomposition(
        gate,
        DecompEntry {
            requires: GateSupportSet::from_iter([gates::CX]),
            decomposition: Decomposition::Dynamic(Arc::new(vec![DecompOp {
                gate: gates::CX,
                qubit_indices: smallvec::smallvec![0, 1, 2],
                angles: smallvec::smallvec![],
            }])),
        },
    );
    let mut state = SparseStab::with_seed(3, 42);
    state.x(&[QubitId(0)]);
    let circuit = OpBuilder::new().gate(gate, &qubits(3), &[]).build();
    assert!(matches!(
        CircuitRunner::with_definitions(defs).apply_adapted_circuit(&mut state, &circuit),
        Err(ExecutionError::QubitArity {
            arity: 2,
            got: 3,
            ..
        })
    ));
    assert_eq!(bits(&mut state, 3), [true, false, false]);
}

fn missing_angle_setup(negated: bool) -> (GateDefinitions, GateId, Vec<DecompOp>) {
    let mut defs = GateDefinitions::new();
    let id = defs.register(GateSpec::new("missing_angle"));
    let mut op = DecompOp::rotation(gates::RZ, 0, 0);
    if negated {
        op.angles[0] = pecos_neo::extensible::AngleSource::NegInput(0);
    }
    let ops = vec![op];
    defs.set_decomposition(
        id,
        DecompEntry {
            requires: GateSupportSet::from_iter([gates::RZ]),
            decomposition: Decomposition::Dynamic(Arc::new(ops.clone())),
        },
    );
    (defs, id, ops)
}

#[test]
fn runner_missing_decomposition_angle_returns_error() {
    for negated in [false, true] {
        let (defs, id, _) = missing_angle_setup(negated);
        let circuit = OpBuilder::new().gate(id, &[QubitId(0)], &[]).build();
        let result = CircuitRunner::with_definitions(defs)
            .apply_adapted_circuit(&mut SparseStab::with_seed(1, 42), &circuit);
        assert!(
            matches!(result, Err(ExecutionError::DecompositionAngleIndexOutOfRange { gate_id, index: 0, angle_count: 0 }) if gate_id == id)
        );
    }
}

#[test]
fn resolver_missing_decomposition_angle_returns_error() {
    use pecos_neo::extensible::{CircuitResolver, DecompositionRegistry, ResolutionError};
    for negated in [false, true] {
        let (defs, id, ops) = missing_angle_setup(negated);
        let mut registry = DecompositionRegistry::new();
        let support = GateSupportSet::from_iter([gates::RZ]);
        registry.register_dynamic(id, support.clone(), ops);
        let circuit = OpBuilder::new().gate(id, &[QubitId(0)], &[]).build();
        let result = CircuitResolver::new(&registry, &support)
            .with_definitions(&defs)
            .resolve(&circuit);
        assert!(
            matches!(result, Err(ResolutionError::AngleIndexOutOfRange { gate, index: 0, angle_count: 0 }) if gate == id)
        );
    }
}

#[test]
fn adapted_repeated_qubits_return_error() {
    let circuit = OpBuilder::new()
        .gate(gates::CX, &[QubitId(0), QubitId(0)], &[])
        .build();
    let result =
        CircuitRunner::new().apply_adapted_circuit(&mut SparseStab::with_seed(1, 42), &circuit);
    assert!(matches!(result, Err(ExecutionError::InvalidCommand(_))));
}

#[test]
fn definitions_match_every_builtin_quantum_arity() {
    let defs = GateDefinitions::new();
    for &gate in pecos_neo::command::GateType::ALL {
        assert_eq!(
            defs.quantum_arity(gate.to_gate_id()).map(usize::from),
            Some(gate.quantum_arity()),
            "{gate:?}"
        );
    }
    let registry = pecos_neo::extensible::GateRegistry::new();
    for id in [gates::PX, gates::PY] {
        assert_eq!(defs.spec(id), registry.get(id));
        assert_eq!(defs.quantum_arity(id), Some(1));
    }
}

#[test]
fn resolver_expands_qfree_for_both_qubits() {
    use pecos_neo::extensible::{CircuitResolver, DecompositionRegistry};
    let mut registry = DecompositionRegistry::new();
    let support = GateSupportSet::from_iter([gates::X]);
    registry.register_dynamic(
        gates::QFREE,
        support.clone(),
        vec![DecompOp::gate1(gates::X, 0)],
    );
    let circuit = OpBuilder::new().gate(gates::QFREE, &qubits(2), &[]).build();
    let resolved = CircuitResolver::new(&registry, &support)
        .resolve(&circuit)
        .unwrap();
    assert_eq!(resolved.ops.len(), 2);
    for (i, op) in resolved.ops.iter().enumerate() {
        assert!(
            matches!(op, ResolvedOp::Gate { gate_id, qubits, .. } if *gate_id == gates::X && qubits.as_slice() == [QubitId(i)])
        );
    }
}

fn angled_gate(gate_id: GateId, count: usize, angles: &[Angle64]) -> ResolvedOp {
    ResolvedOp::Gate {
        gate_id,
        qubits: qubits(count).into_iter().collect(),
        angles: angles.iter().copied().collect(),
    }
}

#[test]
fn batching_rejects_wrong_angle_counts_for_builtin_gates() {
    for (gate, count, expected, got) in [
        (gates::U, 1, 3, 2),
        (gates::CX, 2, 0, 1),
        (gates::RZ, 1, 1, 0),
    ] {
        let resolved =
            ResolvedCircuit::new(vec![angled_gate(gate, count, &vec![Angle64::ZERO; got])]);
        assert_eq!(
            BatchedCircuit::from_resolved(&resolved, &GateDefinitions::new()).unwrap_err(),
            BatchError::AngleArity {
                gate_id: gate,
                expected,
                got
            }
        );
    }
}

#[test]
fn batching_matches_execution_for_user_gate_angles() {
    let angle = Angle64::QUARTER_TURN;
    let mut defs = GateDefinitions::new();
    let undeclared = defs.register(GateSpec::new("undeclared"));
    let resolved = ResolvedCircuit::new(vec![angled_gate(undeclared, 1, &[angle])]);
    let circuit = BatchedCircuit::from_resolved(&resolved, &defs).unwrap();
    assert!(matches!(
        circuit.batches.as_slice(),
        [Batch::Rotation { gate_id, ops }] if *gate_id == undeclared && ops.as_slice() == [(QubitId(0), angle)]
    ));
}

#[test]
fn batching_preserves_angles_for_every_rotation_group() {
    let angle = Angle64::QUARTER_TURN;
    let resolved = ResolvedCircuit::new(vec![
        angled_gate(gates::RZ, 2, &[angle]),
        angled_gate(gates::RZZ, 4, &[angle]),
    ]);
    let circuit = BatchedCircuit::from_resolved(&resolved, &GateDefinitions::new()).unwrap();
    assert_eq!(circuit.batch_count(), 2);
    assert!(
        matches!(&circuit.batches[1], Batch::General { gate_id, ops } if *gate_id == gates::RZZ && ops.len() == 2)
    );
    let mut exec = executor();
    exec.execute_batched(&circuit);
    assert_eq!(
        exec.calls,
        vec![
            (gates::RZ, vec![QubitId(0)], vec![angle]),
            (gates::RZ, vec![QubitId(1)], vec![angle]),
            (gates::RZZ, qubits(2), vec![angle]),
            (gates::RZZ, vec![QubitId(2), QubitId(3)], vec![angle]),
        ]
    );
}

#[test]
fn batching_rejects_zero_arity_with_actionable_message() {
    let mut defs = GateDefinitions::new();
    let id = defs.register(GateSpec::new("zero").with_quantum_arity(0));
    let error =
        BatchedCircuit::from_resolved(&ResolvedCircuit::new(vec![resolved_gate(id, 1)]), &defs)
            .unwrap_err();
    assert_eq!(error, BatchError::UnknownArity { gate_id: id });
    assert!(
        error
            .to_string()
            .contains("register the gate with a declared")
    );
    assert!(error.to_string().contains("positive quantum arity"));
}

#[test]
fn batching_rejects_repeated_qubits_across_the_whole_command() {
    let mut defs = GateDefinitions::new();
    let custom = defs.register(GateSpec::new("paired").with_quantum_arity(2));
    for gate in [gates::CX, custom] {
        for qubits in [
            vec![QubitId(0), QubitId(0)],
            vec![QubitId(0), QubitId(1), QubitId(2), QubitId(0)],
        ] {
            let resolved = ResolvedCircuit::new(vec![ResolvedOp::Gate {
                gate_id: gate,
                qubits: qubits.into_iter().collect(),
                angles: smallvec::smallvec![],
            }]);
            assert_eq!(
                BatchedCircuit::from_resolved(&resolved, &defs).unwrap_err(),
                BatchError::DuplicateQubits {
                    gate_id: gate,
                    qubits: vec![0]
                }
            );
        }
    }
}

fn partial_cx_queue() -> pecos_neo::command::CommandQueue {
    [pecos_neo::command::GateCommand::new(
        pecos_neo::command::GateType::CX,
        qubits(3),
    )]
    .into_iter()
    .collect()
}

#[test]
#[should_panic(expected = "cannot execute invalid circuit")]
fn path_recording_rejects_partial_cx() {
    pecos_neo::sampling::PathExplorer::new(SparseStab::with_seed(3, 42))
        .run_and_record(&partial_cx_queue());
}

#[test]
#[should_panic(expected = "cannot execute invalid circuit")]
fn path_replay_rejects_partial_cx() {
    pecos_neo::sampling::PathExplorer::new(SparseStab::with_seed(3, 42)).run_with_path(
        &partial_cx_queue(),
        &pecos_neo::sampling::EnumeratedPath::new(0, 0),
    );
}

#[derive(Clone)]
struct BeforeGateObserver(Arc<std::sync::atomic::AtomicUsize>);
impl pecos_neo::noise::NoiseChannel for BeforeGateObserver {
    fn responds_to(&self, event: &pecos_neo::noise::NoiseEvent<'_>) -> bool {
        matches!(event, pecos_neo::noise::NoiseEvent::BeforeGate { .. })
    }
    fn apply(
        &self,
        _: &pecos_neo::noise::NoiseEvent<'_>,
        _: &mut pecos_neo::noise::NoiseContext,
        _: &mut pecos_random::PecosRng,
    ) -> pecos_neo::noise::NoiseResponse {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        pecos_neo::noise::NoiseResponse::None
    }
    fn name(&self) -> &'static str {
        "before gate observer"
    }
    fn clone_box(&self) -> Box<dyn pecos_neo::noise::NoiseChannel> {
        Box::new(self.clone())
    }
}

#[test]
fn rejected_gates_emit_no_before_gate_noise() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for operands in [qubits(3), vec![QubitId(0), QubitId(0)]] {
        let observed = Arc::new(AtomicUsize::new(0));
        let noise = pecos_neo::noise::ComposableNoiseModel::new()
            .add_channel(BeforeGateObserver(Arc::clone(&observed)));
        let mut runner = CircuitRunner::new().with_noise(noise);
        let mut state = SparseStab::with_seed(3, 42);
        let circuit = OpBuilder::new().gate(gates::CX, &operands, &[]).build();
        assert!(runner.apply_adapted_circuit(&mut state, &circuit).is_err());
        assert_eq!(observed.load(Ordering::Relaxed), 0);
        let valid = OpBuilder::new().gate(gates::CX, &qubits(2), &[]).build();
        runner.apply_adapted_circuit(&mut state, &valid).unwrap();
        assert_eq!(observed.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn overridden_qfree_receives_category_noise() {
    use pecos_neo::extensible::GateCategory;
    use pecos_neo::noise::{CategoryBasedChannel, ComposableNoiseModel, PauliWeights};
    use pecos_neo::runner::GateOverrides;
    let defs = GateDefinitions::new();
    let noise = ComposableNoiseModel::new()
        .with_gate_definitions(defs.clone())
        .add_channel(
            CategoryBasedChannel::new()
                .with_category_weights(GateCategory::QubitManagement, 1.0, PauliWeights::x_only())
                .unwrap(),
        );
    let overrides = GateOverrides::new().register(gates::QFREE, |_: &mut SparseStab, _, _| true);
    let mut runner = CircuitRunner::with_definitions(defs)
        .with_overrides(overrides)
        .with_noise(noise)
        .with_seed(42);
    let circuit = OpBuilder::new().gate(gates::QFREE, &qubits(1), &[]).build();
    let mut state = SparseStab::with_seed(1, 42);
    runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
    assert_eq!(bits(&mut state, 1), [true]);
}
