use super::*;

fn synthetic() -> SeleneRuntime {
    let mut runtime = SeleneRuntime::new("synthetic-runtime.so");
    runtime.set_num_qubits(4);
    runtime.shot_start(17, Some(41)).unwrap();
    runtime.runtime_to_program_results.insert(901, 7);
    runtime.leakage_results.insert(7);
    runtime
}

fn batch() -> RuntimeOperationBatch {
    let mut batch = RuntimeOperationBatch::default();
    let payload = [17_u8, 29, 43];
    unsafe {
        runtime_batch_rxy((&raw mut batch).cast(), 0, 0.25, 0.5);
        runtime_batch_rzz((&raw mut batch).cast(), 1, 2, 0.75);
        runtime_batch_custom(
            (&raw mut batch).cast(),
            7301,
            payload.as_ptr().cast(),
            payload.len(),
        );
        runtime_batch_measure((&raw mut batch).cast(), 3, 901);
        runtime_batch_set_time((&raw mut batch).cast(), 20, 5);
    }
    batch
}

#[test]
fn native_callbacks_preserve_original_batch_and_result_namespaces() {
    let mut runtime = synthetic();
    runtime.set_custom_event_policy(RuntimeCustomEventPolicy::RejectUnhandled);
    runtime.set_custom_event_handler(|_| panic!("extraction must not call metadata handlers"));
    let extracted = runtime
        .collect_scheduled(|runtime| {
            runtime.retain_scheduled_batch(batch())?;
            runtime.retain_scheduled_batch(RuntimeOperationBatch {
                start_time_nanos: 40,
                ..Default::default()
            })?;
            Ok(vec![])
        })
        .unwrap();
    assert_eq!(extracted.len(), 2);
    assert_eq!(
        (extracted[0].runtime_shot_id, extracted[0].batch_index),
        (17, 0)
    );
    assert_eq!(
        (extracted[0].start_time_nanos, extracted[0].duration_nanos),
        (20, 5)
    );
    assert_eq!(extracted[0].operations, batch().operations);
    assert_eq!(
        extracted[0].measurements,
        [ScheduledMeasurement {
            operation_index: 3,
            runtime_result: 901,
            program_result: 7,
            leakage_aware: true
        }]
    );
    assert!(extracted[1].operations.is_empty());
    assert_eq!(extracted[1].batch_index, 1);
    assert!(runtime.custom_events.is_empty());
    assert!(runtime.last_gate_time_end_nanos.is_empty());
    assert!(runtime.scheduled_output.is_none());
    let next = runtime
        .collect_scheduled(|runtime| {
            runtime.retain_scheduled_batch(batch())?;
            Ok(vec![])
        })
        .unwrap();
    assert_eq!(next[0].batch_index, 2);
}

#[test]
fn invalid_batches_do_not_escape_and_failures_require_reset() {
    for case in 0..5 {
        let mut runtime = synthetic();
        let mut bad = batch();
        match case {
            0 => bad.start_time_nanos = u64::MAX,
            1 => {
                bad.operations[0] = RuntimeScheduledOp::Rz {
                    qubit_id: 0,
                    theta: f64::NAN,
                }
            }
            2 => bad.operations[0] = RuntimeScheduledOp::Reset { qubit_id: u64::MAX },
            3 => runtime.runtime_to_program_results.clear(),
            _ => bad.operations.push(RuntimeScheduledOp::Custom {
                tag: 7302,
                data: vec![0; MAX_PAYLOAD_BYTES],
            }),
        }
        let error = runtime
            .collect_scheduled(|runtime| {
                runtime.retain_scheduled_batch(RuntimeOperationBatch::default())?;
                runtime.retain_scheduled_batch(bad)?;
                Ok(vec![])
            })
            .unwrap_err();
        assert!(runtime.scheduled_output.is_none());
        assert_eq!(
            runtime.shot_end().unwrap_err().to_string(),
            error.to_string()
        );
        assert!(runtime.shot_start(18, None).is_err());
        runtime.reset().unwrap();
        runtime.shot_start(18, None).unwrap();
    }
}

#[test]
fn callback_budgets_reject_before_reading_oversize_payload_and_stop_appends() {
    let mut batch = RuntimeOperationBatch {
        extraction_budget: Some((1, 2)),
        ..Default::default()
    };
    unsafe {
        runtime_batch_custom(
            (&raw mut batch).cast(),
            7301,
            std::ptr::dangling::<u8>().cast(),
            3,
        );
        runtime_batch_reset((&raw mut batch).cast(), 0);
    }
    assert!(batch.operations.is_empty());
    assert_eq!(
        batch.callback_error,
        Some("scheduled payload budget exceeded")
    );
    let mut batch = RuntimeOperationBatch {
        extraction_budget: Some((1, 0)),
        ..Default::default()
    };
    unsafe {
        runtime_batch_reset((&raw mut batch).cast(), 0);
        runtime_batch_reset((&raw mut batch).cast(), 1);
        runtime_batch_reset((&raw mut batch).cast(), 2);
    }
    assert_eq!(batch.operations.len(), 1);
    assert_eq!(
        batch.callback_error,
        Some("scheduled operation budget exceeded")
    );
}

#[test]
fn operation_limit_applies_to_one_native_batch() {
    let mut runtime = synthetic();
    assert!(
        runtime
            .collect_scheduled(|runtime| {
                runtime.retain_scheduled_batch(RuntimeOperationBatch {
                    operations: vec![RuntimeScheduledOp::Reset { qubit_id: 0 }; MAX_OPERATIONS + 1],
                    ..Default::default()
                })?;
                Ok(vec![])
            })
            .is_err()
    );
}

#[test]
fn preflight_rejects_unsupported_inputs_without_loading_plugin() {
    for op in [
        QuantumOp::H(0).into(),
        QuantumOp::RZ(f64::NAN, 0).into(),
        Operation::AllocateResult { id: usize::MAX },
        Operation::TraceMetadata {
            metadata: TraceMetadata::new(),
            qubit: None,
        },
    ] {
        let mut runtime = synthetic();
        assert!(runtime.lower_scheduled_operations(&[op]).is_err());
        assert!(runtime.instance.is_none());
        assert!(runtime.batch_failure.is_none());
        assert!(runtime.scheduled_mode.is_none());
    }
}

#[test]
fn mode_and_clone_isolation_prevent_false_live_snapshots() {
    let mut runtime = synthetic();
    runtime.collect_scheduled(|_| Ok(vec![])).unwrap();
    assert!(runtime.lower_operations(&[]).is_err());
    assert!(runtime.lower_operations_with_metadata(&[]).is_err());
    let mut cloned = runtime.clone();
    assert!(
        cloned
            .lower_scheduled_operations(&[])
            .unwrap_err()
            .to_string()
            .contains("cloned scheduled runtime requires reset")
    );
    cloned.reset().unwrap();
    cloned.shot_start(90, None).unwrap();
    assert!(cloned.scheduled_mode.is_none());
    assert_eq!(runtime.pending_shot_start.unwrap().0, 17);
    runtime.drain_pending_scheduled_operations().unwrap();
    runtime.shot_start(18, None).unwrap();
    assert!(runtime.scheduled_mode.is_none());
    assert_eq!(runtime.runtime_batch_index, 0);
    runtime.select_output_mode(false).unwrap();
    assert!(runtime.lower_scheduled_operations(&[]).is_err());
}

#[test]
fn caught_unwind_drops_output_and_poison_survives_clone() {
    let mut runtime = synthetic();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.collect_scheduled(|runtime| {
            runtime.retain_scheduled_batch(batch())?;
            panic!("synthetic panic");
        })
    }));
    assert!(result.is_err());
    assert!(runtime.scheduled_output.is_none());
    assert!(runtime.shot_end().is_err());
    assert!(runtime.clone().shot_end().is_err());
    runtime.reset().unwrap();
    assert!(runtime.batch_failure.is_none());
}

#[test]
fn public_native_runtime_extracts_across_calls_and_terminal_flush() {
    let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(81, Some(7)).unwrap();
    let first = QisRuntime::lower_scheduled_operations(
        &mut runtime,
        &[
            Operation::AllocateQubit { id: 7 },
            Operation::AllocateQubit { id: 3 },
            QuantumOp::RXYXY2Q(-0.73, 0.41, 3, 7).into(),
        ],
    )
    .unwrap();
    assert!(first.iter().flat_map(|b| &b.operations).any(|op| *op
        == RuntimeScheduledOp::Rpp {
            qubit_id_1: 1,
            qubit_id_2: 0,
            theta: -0.73,
            phi: 0.41
        }));
    let second = QisRuntime::lower_scheduled_operations(
        &mut runtime,
        &[QuantumOp::MeasureLeaked(3, 123).into()],
    )
    .unwrap();
    let tail = QisRuntime::drain_pending_scheduled_operations(&mut runtime).unwrap();
    let all = first
        .into_iter()
        .chain(second)
        .chain(tail)
        .collect::<Vec<_>>();
    for (index, batch) in all.iter().enumerate() {
        assert_eq!(batch.runtime_shot_id, 81);
        assert_eq!(batch.batch_index, index);
    }
    assert!(
        all.iter()
            .flat_map(|b| &b.measurements)
            .any(|m| m.program_result == 123 && m.leakage_aware)
    );
    assert!(runtime.last_gate_time_end_nanos.is_empty());
    assert!(runtime.scheduled_output.is_none());
}

#[test]
fn capacity_changes_reject_before_recreating_native_state() {
    let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(20, None).unwrap();
    runtime
        .lower_scheduled_operations(&[QuantumOp::RZ(0.25, 0).into()])
        .unwrap();
    let instance = runtime.instance;
    runtime.set_num_qubits(5);
    assert!(
        runtime
            .lower_scheduled_operations(&[])
            .unwrap_err()
            .to_string()
            .contains("capacity changed")
    );
    assert_eq!(runtime.instance, instance);
    assert_eq!(runtime.initialized_num_qubits, Some(4));
    runtime.reset().unwrap();
    runtime.shot_start(21, None).unwrap();
    runtime.lower_scheduled_operations(&[]).unwrap();
    assert_eq!(runtime.initialized_num_qubits, Some(5));
}

#[test]
fn legacy_execution_must_not_bypass_scheduled_mode() {
    let mut runtime = synthetic();
    let mut interface = OperationCollector::default();
    interface.operations.push(QuantumOp::X(0).into());
    runtime.load_interface(interface).unwrap();
    runtime.collect_scheduled(|_| Ok(vec![])).unwrap();
    assert!(
        runtime.execute_until_quantum().is_err(),
        "flat legacy execution was allowed inside a scheduled shot"
    );
}

#[test]
fn scheduled_mode_must_not_follow_legacy_execution() {
    let mut runtime = synthetic();
    let mut interface = OperationCollector::default();
    interface.operations.push(QuantumOp::X(0).into());
    runtime.load_interface(interface).unwrap();
    assert_eq!(
        runtime.execute_until_quantum().unwrap(),
        Some(vec![QuantumOp::X(0)])
    );
    assert!(
        runtime.collect_scheduled(|_| Ok(vec![])).is_err(),
        "scheduled extraction was allowed after flat legacy execution"
    );
}

#[test]
fn legacy_terminal_drain_cannot_enter_a_scheduled_session() {
    let mut runtime = synthetic();
    runtime.collect_scheduled(|_| Ok(vec![])).unwrap();
    assert!(
        runtime
            .drain_pending_operations()
            .unwrap_err()
            .to_string()
            .contains("cannot mix")
    );
    assert!(runtime.drain_pending_scheduled_operations().is_ok());
}

#[test]
fn review_large_simple_schedule_is_not_limited_by_batch_count() {
    let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(31, None).unwrap();
    let operations = (0..200)
        .map(|i| QuantumOp::RXY(0.25, 0.5, i % 4).into())
        .collect::<Vec<_>>();
    let batches = runtime.lower_scheduled_operations(&operations).unwrap();
    assert_eq!(
        batches
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(op, RuntimeScheduledOp::Rxy { .. }))
            .count(),
        200
    );
}

#[test]
fn review_deferred_schedule_drains_without_inserting_barriers() {
    let mut runtime = crate::selene_runtimes::selene_soft_rz_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(32, None).unwrap();
    let mut batches = Vec::new();
    for i in 0..200 {
        batches.extend(
            runtime
                .lower_scheduled_operations(&[QuantumOp::RXY(0.25, 0.5, i % 4).into()])
                .unwrap(),
        );
    }
    assert!(
        batches.is_empty(),
        "expected native scheduling to defer until terminal barrier"
    );
    batches.extend(runtime.drain_pending_scheduled_operations().unwrap());
    assert_eq!(
        batches
            .iter()
            .flat_map(|b| &b.operations)
            .filter(|op| matches!(op, RuntimeScheduledOp::Rxy { .. }))
            .count(),
        200
    );
    runtime.shot_end().unwrap();
}

#[test]
fn review_shot_end_requires_a_successful_terminal_drain() {
    let mut runtime = crate::selene_runtimes::selene_soft_rz_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(33, None).unwrap();
    assert!(
        runtime
            .lower_scheduled_operations(&[QuantumOp::RXY(0.25, 0.5, 0).into()])
            .unwrap()
            .is_empty()
    );
    assert!(
        runtime.shot_end().is_err(),
        "shot_end discarded pending native work"
    );
    let batches = runtime.drain_pending_scheduled_operations().unwrap();
    assert!(batches.iter().any(|b| !b.operations.is_empty()));
    runtime.shot_end().unwrap();
}

#[test]
fn review_clone_guard_rejects_before_loading_a_valid_runtime() {
    let mut runtime = crate::selene_runtimes::selene_simple_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(34, None).unwrap();
    runtime
        .lower_scheduled_operations(&[QuantumOp::RZ(0.25, 0).into()])
        .unwrap();
    let mut cloned = runtime.clone();
    let error = cloned
        .lower_scheduled_operations(&[])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("cloned scheduled runtime requires reset"),
        "{error}"
    );
    assert!(cloned.instance.is_none());
    cloned.reset().unwrap();
    cloned.shot_start(35, None).unwrap();
    cloned
        .lower_scheduled_operations(&[QuantumOp::RZ(0.25, 0).into()])
        .unwrap();
    assert!(cloned.instance.is_some());
}

#[test]
fn new_submission_invalidates_terminal_drain_and_prevents_shot_replacement() {
    let mut runtime = crate::selene_runtimes::selene_soft_rz_runtime().unwrap();
    runtime.set_num_qubits(4);
    runtime.shot_start(40, None).unwrap();
    runtime
        .lower_scheduled_operations(&[QuantumOp::RXY(0.25, 0.5, 0).into()])
        .unwrap();
    runtime.drain_pending_scheduled_operations().unwrap();
    runtime
        .lower_scheduled_operations(&[QuantumOp::RXY(0.25, 0.5, 1).into()])
        .unwrap();
    assert!(runtime.shot_end().is_err());
    assert!(runtime.shot_start(41, None).is_err());
    assert!(
        !runtime
            .drain_pending_scheduled_operations()
            .unwrap()
            .is_empty()
    );
    runtime.shot_end().unwrap();
    runtime.shot_start(41, None).unwrap();
}

#[test]
fn non_finite_source_angles_have_a_specific_diagnostic() {
    let mut runtime = synthetic();
    assert!(
        runtime
            .lower_scheduled_operations(&[QuantumOp::RZ(f64::INFINITY, 0).into()])
            .unwrap_err()
            .to_string()
            .contains("non-finite scheduled source angle")
    );
}
