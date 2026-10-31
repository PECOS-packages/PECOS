//! Synthetic event protocol tests; no runtime-specific private schemas.
use super::*;
use crate::runtime_batch_adapter::MAX_ADAPTER_OPERATIONS;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Phase;
impl RuntimeBatchAdapter for Phase {
    fn normalize(&mut self, batch: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>> {
        batch
            .operations
            .iter()
            .map(|op| match op {
                RuntimeScheduledOp::Custom { tag: 7301, data } if data == &[1] => {
                    Ok(RuntimeScheduledOp::Rz {
                        qubit_id: 0,
                        theta: std::f64::consts::PI,
                    })
                }
                RuntimeScheduledOp::Custom { .. } => {
                    Err(RuntimeError::ExecutionError("unknown test event".into()))
                }
                gate => Ok(gate.clone()),
            })
            .collect()
    }
}

fn batch(start: u64) -> RuntimeOperationBatch {
    RuntimeOperationBatch {
        start_time_nanos: start,
        duration_nanos: 20,
        operations: vec![
            RuntimeScheduledOp::Custom {
                tag: 7301,
                data: vec![1],
            },
            RuntimeScheduledOp::Rxy {
                qubit_id: 0,
                theta: 0.5,
                phi: 0.0,
            },
        ],
        ..Default::default()
    }
}

fn runtime() -> SeleneRuntime {
    let mut runtime = SeleneRuntime::new("synthetic.so");
    runtime.set_num_qubits(2);
    runtime.set_batch_adapter_factory(|| Ok(Box::new(Phase)));
    runtime
}

#[test]
fn normalized_gates_share_original_batch_timing() {
    let mut runtime = runtime();
    assert_eq!(
        runtime.convert_runtime_batch(batch(100)).unwrap(),
        vec![
            QuantumOp::Idle(100e-9, 0),
            QuantumOp::RZ(std::f64::consts::PI, 0),
            QuantumOp::RXY(0.5, 0.0, 0),
        ]
    );
    assert_eq!(runtime.last_gate_time_end_nanos[0], 120);
    assert_eq!(runtime.custom_events()[0].operation_index, 0);
    assert_eq!(runtime.custom_events()[0].data, [1]);
    runtime.runtime_to_program_results.insert(9, (3, false));
    let ops = runtime
        .convert_runtime_batch(RuntimeOperationBatch {
            start_time_nanos: 170,
            duration_nanos: 10,
            operations: vec![RuntimeScheduledOp::Measure {
                qubit_id: 0,
                result_id: 9,
            }],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        ops,
        vec![QuantumOp::Idle(50e-9, 0), QuantumOp::Measure(0, 3)]
    );
}

#[test]
fn unknown_event_poisons_even_in_capture_mode() {
    let mut runtime = runtime();
    runtime.set_custom_event_policy(RuntimeCustomEventPolicy::Capture);
    let mut input = batch(0);
    input.operations[0] = RuntimeScheduledOp::Custom {
        tag: 7302,
        data: vec![],
    };
    assert!(
        runtime
            .convert_runtime_batch(input)
            .unwrap_err()
            .to_string()
            .contains("unknown test event")
    );
    assert!(runtime.convert_runtime_batch(batch(100)).is_err());
    runtime.reset().unwrap();
    assert!(runtime.convert_runtime_batch(batch(100)).is_ok());
}

#[test]
fn factories_are_fresh_for_each_shot_and_live_clones_reject() {
    let count = Arc::new(AtomicUsize::new(0));
    let calls = count.clone();
    let mut runtime = runtime();
    runtime.set_batch_adapter_factory(move || {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Phase))
    });
    for shot in 0..2 {
        runtime.shot_start(shot, None).unwrap();
        runtime.convert_runtime_batch(batch(100)).unwrap();
        runtime.convert_runtime_batch(batch(200)).unwrap();
    }
    assert_eq!(count.load(Ordering::SeqCst), 2);
    assert!(runtime.clone().convert_runtime_batch(batch(300)).is_err());
}

struct Panic;
impl RuntimeBatchAdapter for Panic {
    fn normalize(&mut self, _: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>> {
        panic!("adapter panic")
    }
}

#[test]
fn callback_panics_require_reset() {
    let mut runtime = runtime();
    runtime.set_batch_adapter_factory(|| Ok(Box::new(Panic)));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || runtime.convert_runtime_batch(batch(0))
        ))
        .is_err()
    );
    assert!(runtime.convert_runtime_batch(batch(100)).is_err());
}

#[test]
fn adapter_output_is_checked_before_lowering() {
    let source = RuntimeBatch {
        shot_id: 0,
        batch_index: 0,
        num_qubits: 2,
        start_time_nanos: 0,
        duration_nanos: 10,
        operations: vec![RuntimeScheduledOp::Measure {
            qubit_id: 0,
            result_id: 7,
        }],
    };
    for bad in [
        vec![],
        vec![RuntimeScheduledOp::Measure {
            qubit_id: 0,
            result_id: 8,
        }],
        vec![RuntimeScheduledOp::MeasureLeaked {
            qubit_id: 0,
            result_id: 7,
        }],
        vec![RuntimeScheduledOp::Measure {
            qubit_id: 1,
            result_id: 7,
        }],
        vec![RuntimeScheduledOp::Custom {
            tag: 1,
            data: vec![],
        }],
        vec![RuntimeScheduledOp::Rz {
            qubit_id: 2,
            theta: 0.0,
        }],
        vec![RuntimeScheduledOp::Rz {
            qubit_id: 0,
            theta: f64::NAN,
        }],
        vec![RuntimeScheduledOp::Reset { qubit_id: 0 }; MAX_ADAPTER_OPERATIONS + 1],
    ] {
        assert!(source.validate_output(&bad).is_err());
    }
    assert!(source.validate_output(&source.operations).is_ok());
}

#[test]
fn scheduled_transport_rejects_normalization_before_callbacks() {
    let mut runtime = runtime();
    runtime.set_batch_adapter_factory(|| panic!("must not create adapter"));
    runtime.shot_start(0, None).unwrap();
    let error = runtime
        .collect_scheduled(|_| panic!("must not collect native batches"))
        .unwrap_err();
    assert!(error.to_string().contains("ordinary QIS lowering"));
    assert!(runtime.scheduled_mode.is_none());
}

#[test]
fn draining_captured_events_cannot_certify_the_trace() {
    let mut runtime = SeleneRuntime::new("synthetic.so");
    runtime.set_custom_event_policy(RuntimeCustomEventPolicy::Capture);
    runtime.convert_runtime_batch(batch(0)).unwrap();
    assert!(!runtime.runtime_events_complete());
    runtime.take_custom_events();
    assert!(!runtime.runtime_events_complete());
    runtime.reset().unwrap();
    assert!(runtime.runtime_events_complete());
}

#[test]
fn adapter_keeps_shot_identity_after_native_end() {
    struct Identity(Arc<std::sync::Mutex<Vec<u64>>>);
    impl RuntimeBatchAdapter for Identity {
        fn normalize(&mut self, batch: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>> {
            self.0.lock().unwrap().push(batch.shot_id);
            Phase.normalize(batch)
        }
    }
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let records = seen.clone();
    let mut runtime = runtime();
    runtime.set_batch_adapter_factory(move || Ok(Box::new(Identity(records.clone()))));
    runtime.shot_start(19, None).unwrap();
    runtime.convert_runtime_batch(batch(0)).unwrap();
    // The native end hook clears these before delivering its final batches.
    runtime.active_shot = None;
    runtime.pending_shot_start = None;
    runtime.convert_runtime_batch(batch(100)).unwrap();
    runtime.reset().unwrap();
    runtime.shot_start(20, None).unwrap();
    runtime.convert_runtime_batch(batch(0)).unwrap();
    assert_eq!(*seen.lock().unwrap(), [19, 19, 20]);
}

#[test]
fn repeated_timed_targets_require_adapter_normalization() {
    for reset_first in [false, true] {
        for normalized in [false, true] {
            let mut runtime = SeleneRuntime::new("synthetic.so");
            runtime.set_num_qubits(1);
            if normalized {
                runtime.set_batch_adapter_factory(|| Ok(Box::new(Phase)));
            }
            let first = if reset_first {
                RuntimeScheduledOp::Reset { qubit_id: 0 }
            } else {
                RuntimeScheduledOp::Rxy {
                    qubit_id: 0,
                    theta: 0.5,
                    phi: 0.0,
                }
            };
            let result = runtime.convert_runtime_batch(RuntimeOperationBatch {
                start_time_nanos: 100,
                duration_nanos: 20,
                operations: vec![
                    first,
                    RuntimeScheduledOp::Rz {
                        qubit_id: 0,
                        theta: 0.5,
                    },
                ],
                ..Default::default()
            });
            if normalized {
                let ops = result.unwrap();
                assert_eq!(
                    ops.iter()
                        .filter(|op| matches!(op, QuantumOp::Idle(..)))
                        .count(),
                    usize::from(!reset_first)
                );
                assert_eq!(runtime.last_gate_time_end_nanos[0], 120);
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("starts before its previous operation ended")
                );
            }
        }
    }
}

#[test]
fn normalized_batches_still_reject_cross_batch_overlap() {
    let mut runtime = runtime();
    runtime.convert_runtime_batch(batch(100)).unwrap();
    let error = runtime.convert_runtime_batch(batch(110)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("starts before its previous operation ended")
    );
}

#[test]
fn decoder_batch_capture_retains_normalized_order_without_opaque_payloads() {
    let mut runtime = runtime();
    runtime.enable_decoder_batch_capture().unwrap();
    runtime.convert_runtime_batch(batch(100)).unwrap();
    let captured = runtime.take_decoder_batches().unwrap();
    let first = &captured.batches[0];
    assert!(first.normalization_applied);
    assert_eq!((first.start_time_nanos, first.duration_nanos), (100, 20));
    assert_eq!(
        first.operations,
        [
            QuantumOp::Idle(100e-9, 0),
            QuantumOp::RZ(std::f64::consts::PI, 0),
            QuantumOp::RXY(0.5, 0.0, 0),
        ]
    );
    assert!(!serde_json::to_string(&captured).unwrap().contains("7301"));
}
