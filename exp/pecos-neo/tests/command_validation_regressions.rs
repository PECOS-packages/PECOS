use pecos_core::{Angle64, QubitId, TimeUnits};
use pecos_neo::adapter::command_queue_to_gates;
use pecos_neo::command::{CommandBuilder, CommandQueue, GateCommand, GateCommandError, GateType};
use pecos_neo::extensible::{
    AngleSnapper, CircuitValidator, CliffordValidator, CommandQueueValidation, ExactAngleValidator,
    GateRegistry, SnapPolicy, is_clifford_circuit, snap_command_queue,
};
use pecos_neo::runner::{CircuitRunner, ExecutionError};
use pecos_quantum::TickCircuit;
use pecos_simulators::SparseStab;

#[test]
fn fallible_shot_boundaries_report_raw_queue_errors() {
    use pecos_neo::adapter::QuantumEngineProgramRunner;
    use pecos_neo::program::{ProgramRunner, StaticProgram};
    use pecos_neo::sampling::ImportanceSamplingRunner;

    for command in [
        GateCommand::cx(QubitId(0), QubitId(0)),
        GateCommand::with_angles(GateType::H, vec![QubitId(0)], vec![Angle64::ZERO]),
    ] {
        let queue: CommandQueue = [command].into_iter().collect();
        assert!(queue.validate_for_execution().is_err());
        let mut importance = ImportanceSamplingRunner::new(SparseStab::with_seed(1, 42));
        assert!(importance.try_run_shot(&queue).is_err());
        assert!(importance.try_run_shot_biased(&queue).is_err());
        let mut program = StaticProgram::new(queue.clone(), 1);
        assert!(
            ProgramRunner::new(SparseStab::with_seed(1, 42))
                .try_run_shot(&mut program)
                .is_err()
        );
        let mut engine = QuantumEngineProgramRunner::new(Box::new(
            pecos_engines::quantum::SparseStabEngine::with_seed(1, 42),
        ));
        assert!(engine.try_run_shot(&mut program).is_err());
    }

    let duration = (1_u64 << 53) + 1;
    let queue: CommandQueue = [GateCommand::idle(QubitId(0), TimeUnits::new(duration))]
        .into_iter()
        .collect();
    let mut program = StaticProgram::new(queue.clone(), 1);
    let mut engine = QuantumEngineProgramRunner::new(Box::new(
        pecos_engines::quantum::SparseStabEngine::with_seed(1, 42),
    ));
    assert!(
        engine
            .try_run_shot(&mut program)
            .expect_err("lossless core conversion required")
            .to_string()
            .contains("cannot be represented exactly")
    );
    let mut importance = ImportanceSamplingRunner::new(SparseStab::with_seed(1, 42));
    for result in [
        importance.try_run_shot(&queue),
        importance.try_run_shot_biased(&queue),
    ] {
        assert!(matches!(
            result,
            Err(
                pecos_neo::sampling::ImportanceSamplingError::UnsupportedGate {
                    gate_type: GateType::Idle
                }
            )
        ));
    }
}

#[test]
fn snapping_preserves_signal_positions_and_values() {
    #[derive(Clone, Copy, Debug)]
    struct Marker(u8);
    pecos_core::impl_signal!(Marker);

    let queue = CommandBuilder::new()
        .signal(Marker(7))
        .h(&[0])
        .signal(Marker(9))
        .build();
    let snapped = snap_command_queue(&queue, &SnapPolicy::Exact, &AngleSnapper::clifford(1e-9))
        .expect("exact snapping");
    assert_eq!(
        snapped
            .iter_signals::<Marker>()
            .map(|(position, marker)| (position, marker.0))
            .collect::<Vec<_>>(),
        vec![(0, 7), (1, 9)]
    );
}

#[test]
fn builder_and_collect_preserve_input_and_boundaries_return_errors() {
    let queue = CommandBuilder::new().cx(&[(0usize, 0usize)]).build();
    assert_eq!(queue.len(), 1);
    assert!(matches!(
        command_queue_to_gates(&queue),
        Err(GateCommandError::InvalidGate { .. })
    ));
    assert!(TickCircuit::try_from(&queue).is_err());
    let error = CircuitRunner::<SparseStab>::new()
        .apply_circuit(&mut SparseStab::with_seed(1, 42), &queue)
        .expect_err("duplicate operands must be rejected before simulation");
    assert!(matches!(error, ExecutionError::InvalidCommand(_)));
    println!("duplicate CX: builder succeeds, execution returns {error}");

    let duration = (1_u64 << 53) + 1;
    let queue: CommandQueue = vec![GateCommand::idle(QubitId(0), TimeUnits::new(duration))]
        .into_iter()
        .collect();
    assert_eq!(
        queue.as_slice()[0].get_idle_duration(),
        Some(TimeUnits::new(duration))
    );
    assert_eq!(
        command_queue_to_gates(&queue),
        Err(GateCommandError::IdleDurationNotRepresentable { duration })
    );
    assert!(TickCircuit::try_from(&queue).is_err());
    CircuitRunner::<SparseStab>::new()
        .apply_circuit(&mut SparseStab::with_seed(1, 42), &queue)
        .expect("native command execution retains the integer duration");
    println!("Idle {duration}: collect and native execution succeed, core conversion returns Err");
}

#[test]
fn snapping_preserves_idle_duration_and_does_not_validate_collection() {
    let snapper = AngleSnapper::clifford(1e-9);
    for policy in [
        SnapPolicy::Exact,
        SnapPolicy::SnapOrKeep { tolerance: 1e-9 },
        SnapPolicy::SnapOrFail { tolerance: 1e-9 },
    ] {
        for duration in [0, 23, (1_u64 << 53) + 1, u64::MAX] {
            let queue: CommandQueue = [
                GateCommand::idle(QubitId(0), TimeUnits::new(duration)),
                GateCommand::cx(QubitId(0), QubitId(0)),
            ]
            .into_iter()
            .collect();
            let snapped =
                snap_command_queue(&queue, &policy, &snapper).expect("duration is not an angle");
            assert_eq!(snapped.as_slice(), queue.as_slice());
            println!("{policy:?}: Idle {duration} preserved");
        }
    }
}

#[test]
fn idle_circuits_remain_clifford_for_every_duration() {
    let registry = GateRegistry::new();
    let validator = ExactAngleValidator::new();
    for duration in [0, 23, (1_u64 << 53) + 1, u64::MAX] {
        let queue = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .gate(GateCommand::idle(QubitId(0), TimeUnits::new(duration)))
            .mz(&[0])
            .build();
        assert!(is_clifford_circuit(&queue));
        queue
            .validate(&validator, &registry)
            .expect("Idle has no rotation angle");
        println!("PZ -> H -> Idle({duration}) -> MZ: Clifford");
    }
}

#[test]
fn fixed_gates_with_surplus_angles_fail_all_clifford_checks() {
    use pecos_neo::extensible::{ExactAngleValidator, GateForValidation};
    let registry = GateRegistry::new();
    let gate_id = GateType::H.to_gate_id();
    let angles = vec![Angle64::ZERO];
    let validator = CliffordValidator::new();
    assert!(
        validator
            .validate(
                &[GateForValidation {
                    gate_id,
                    angles: angles.clone()
                }],
                &registry
            )
            .is_err()
    );
    assert!(!validator.is_gate_allowed(gate_id, &angles, &registry));
    assert!(
        ExactAngleValidator::new()
            .validate(
                &[GateForValidation {
                    gate_id,
                    angles: angles.clone()
                }],
                &registry
            )
            .is_err()
    );
    assert!(!ExactAngleValidator::new().is_gate_allowed(gate_id, &angles, &registry));
    let queue = CommandBuilder::new()
        .gate(GateCommand::with_angles(
            GateType::H,
            vec![QubitId(0)],
            angles,
        ))
        .build();
    assert!(!is_clifford_circuit(&queue));
}

#[test]
fn malformed_idle_payload_is_not_hidden_by_angle_classification() {
    let registry = GateRegistry::new();
    for payload in [vec![], vec![Angle64::ZERO; 2]] {
        let queue = CommandBuilder::new()
            .gate(GateCommand::with_angles(
                GateType::Idle,
                vec![QubitId(0)],
                payload,
            ))
            .build();
        assert!(!is_clifford_circuit(&queue));
        assert!(
            queue
                .validate(&ExactAngleValidator::new(), &registry)
                .is_err()
        );
        assert!(command_queue_to_gates(&queue).is_err());
    }
}
