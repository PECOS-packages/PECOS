// Work emitted only by the runtime end hook must execute before Complete.
use super::*;
use crate::runtime::{ClassicalState, Result as RuntimeResult, Shot};
use crate::scheduled::{RuntimeScheduledOp, ScheduledBatch};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    End,
    Drain,
    Panic,
    Measurement,
}
fn error() -> crate::runtime::RuntimeError {
    crate::runtime::RuntimeError::ExecutionError("synthetic terminal failure".into())
}

#[derive(Clone, Default)]
struct EndTail {
    state: ClassicalState,
    ends: Arc<AtomicUsize>,
    ended: bool,
    emitted: bool,
    fault: Fault,
}
impl QisRuntime for EndTail {
    fn load_interface(&mut self, _: OperationList) -> RuntimeResult<()> {
        Ok(())
    }
    fn execute_until_quantum(&mut self) -> RuntimeResult<Option<Vec<QuantumOp>>> {
        Ok(None)
    }
    fn provide_measurements(&mut self, _: BTreeMap<usize, bool>) -> RuntimeResult<()> {
        Ok(())
    }
    fn get_classical_state(&self) -> &ClassicalState {
        &self.state
    }
    fn get_classical_state_mut(&mut self) -> &mut ClassicalState {
        &mut self.state
    }
    fn is_complete(&self) -> bool {
        true
    }
    fn num_qubits(&self) -> usize {
        1
    }
    fn supports_operation_lowering(&self) -> bool {
        true
    }
    fn reset(&mut self) -> RuntimeResult<()> {
        self.ended = false;
        self.emitted = false;
        self.fault = Fault::None;
        Ok(())
    }

    fn shot_end(&mut self) -> RuntimeResult<Shot> {
        assert!(!self.ended, "end hook called twice");
        self.ends.fetch_add(1, Ordering::SeqCst);
        self.ended = true;
        match self.fault {
            Fault::End => return Err(error()),
            Fault::Panic => panic!("synthetic terminal panic"),
            _ => (),
        }
        Ok(Shot::default())
    }
    fn drain_pending_operations(&mut self) -> RuntimeResult<Vec<QuantumOp>> {
        if self.ended && !self.emitted {
            self.emitted = true;
            if matches!(self.fault, Fault::Drain) {
                return Err(error());
            }
            Ok(vec![if matches!(self.fault, Fault::Measurement) {
                QuantumOp::Measure(0, 99)
            } else {
                QuantumOp::X(0)
            }])
        } else {
            Ok(vec![])
        }
    }
    fn drain_pending_scheduled_operations(&mut self) -> RuntimeResult<Vec<ScheduledBatch>> {
        if self.ended && !self.emitted {
            self.emitted = true;
            if matches!(self.fault, Fault::Drain) {
                return Err(error());
            }
            let measure = matches!(self.fault, Fault::Measurement);
            Ok(vec![ScheduledBatch {
                runtime_shot_id: 0,
                batch_index: 0,
                start_time_nanos: 37,
                duration_nanos: 11,
                operations: vec![if measure {
                    RuntimeScheduledOp::Measure {
                        qubit_id: 0,
                        result_id: 99,
                    }
                } else {
                    RuntimeScheduledOp::Rxy {
                        qubit_id: 0,
                        theta: std::f64::consts::PI,
                        phi: 0.0,
                    }
                }],
                measurements: if measure {
                    vec![crate::scheduled::ScheduledMeasurement {
                        operation_index: 0,
                        runtime_result: 99,
                        program_result: 99,
                        leakage_aware: false,
                    }]
                } else {
                    vec![]
                },
            }])
        } else {
            Ok(vec![])
        }
    }
}

#[test]
fn empty_shot_end_preserves_completion_trace_shape() {
    let runtime = EndTail {
        emitted: true, // This runtime has no terminal work to release.
        ..Default::default()
    };
    let ends = Arc::clone(&runtime.ends);
    let mut engine = QisEngine::with_runtime(Box::new(runtime));
    let trace = Arc::new(Mutex::new(Vec::new()));
    engine.set_operation_trace_collector(Arc::clone(&trace));
    engine.dynamic_state = Some(DynamicExecutionState {
        execution_complete: true,
        terminal_error: None,
        finalized: false,
        terminal_lowering_flushed: true,
        sync_handle: None,
    });
    for _ in 0..2 {
        assert!(matches!(
            engine
                .continue_processing(ByteMessage::outcomes_builder().build())
                .unwrap(),
            EngineStage::Complete(_)
        ));
    }
    assert_eq!(ends.load(Ordering::SeqCst), 1);
    let trace = trace.lock().unwrap();
    let stages: Vec<_> = trace.iter().map(|chunk| chunk.stage.as_str()).collect();
    assert_eq!(stages, ["trace_complete"]);
}

#[test]
fn shot_end_tail_is_delivered_before_completion() {
    use pecos_engines::noise::IntoNoiseModel;
    use pecos_engines::runtime_frame::ShotContext;
    use pecos_engines::scheduled_frame::ScheduledIdleZ;
    use pecos_engines::{StateVecEngine, quantum_system::QuantumSystem};
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V6,
        ScheduledTransport::V7,
    ] {
        let runtime = EndTail::default();
        let ends = Arc::clone(&runtime.ends);
        let mut engine = QisEngine::with_runtime(Box::new(runtime));
        engine.scheduled_transport = mode;
        let trace = Arc::new(Mutex::new(Vec::new()));
        engine.set_operation_trace_collector(Arc::clone(&trace));
        engine.dynamic_state = Some(DynamicExecutionState {
            execution_complete: true,
            terminal_error: None,
            finalized: false,
            terminal_lowering_flushed: true,
            sync_handle: None,
        });
        let EngineStage::NeedsProcessing(commands) = engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .unwrap()
        else {
            panic!("shot_end tail was silently discarded");
        };
        assert_eq!(ends.load(Ordering::SeqCst), 1);
        assert!(!engine.dynamic_state.as_ref().unwrap().finalized);
        let noise = if mode == ScheduledTransport::V7 {
            use pecos_engines::scheduled_events::ScheduledEventIdleZ;
            ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), |_| {
                Ok(Box::new(Identity))
            })
            .into_noise_model()
            .unwrap()
        } else if mode.enabled() {
            ScheduledIdleZ::new(1, 0.0, 0.0, 0.0)
                .unwrap()
                .into_noise_model()
                .unwrap()
        } else {
            Box::new(pecos_engines::noise::PassThroughNoiseModel::builder().build())
        };
        let mut quantum = QuantumSystem::new(noise, Box::new(StateVecEngine::new(1)));
        quantum
            .begin_shot(ShotContext {
                run: 1,
                worker: 0,
                shot: 0,
            })
            .unwrap();
        let reply = quantum.process(commands).unwrap();
        for _ in 0..2 {
            assert!(matches!(
                engine.continue_processing(reply.clone()).unwrap(),
                EngineStage::Complete(_)
            ));
        }
        assert_eq!(ends.load(Ordering::SeqCst), 1);
        let mut measure = ByteMessage::quantum_operations_builder();
        measure.mz(&[0]);
        let values = quantum
            .quantum_engine_mut()
            .process(measure.build())
            .unwrap();
        assert_eq!(values.outcomes().unwrap(), vec![1]);
        if mode == ScheduledTransport::Off {
            let trace = trace.lock().unwrap();
            let stages: Vec<_> = trace.iter().map(|chunk| chunk.stage.as_str()).collect();
            assert_eq!(stages, ["shot_end", "trace_complete"]);
            assert_eq!(trace[0].lowered_quantum_ops.len(), 1);
            assert_eq!(trace[0].lowered_quantum_ops[0].gate_type, "X");
        }
    }
}

struct Identity;
impl pecos_engines::scheduled_events::ScheduledBatchAdapter for Identity {
    fn validate(
        &self,
        _: &pecos_engines::scheduled_events::ScheduledEventBatch,
    ) -> Result<(), PecosError> {
        Ok(())
    }
    fn translate(
        &mut self,
        batch: &pecos_engines::scheduled_events::ScheduledEventBatch,
        out: &mut pecos_engines::scheduled_events::ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError> {
        for op in &batch.operations {
            match op {
                pecos_engines::scheduled_events::ScheduledEventOp::Gate(gate) => {
                    out.push(gate.as_ref().clone())?;
                }
                pecos_engines::scheduled_events::ScheduledEventOp::Custom { .. } => {
                    return Err(PecosError::Input("unhandled event".into()));
                }
            }
        }
        Ok(())
    }
}

#[test]
fn terminal_failures_latch_and_require_reset() {
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V6,
        ScheduledTransport::V7,
    ] {
        for fault in [Fault::End, Fault::Drain, Fault::Panic, Fault::Measurement] {
            let runtime = EndTail {
                fault,
                ..Default::default()
            };
            let ends = Arc::clone(&runtime.ends);
            let mut engine = QisEngine::with_runtime(Box::new(runtime));
            engine.scheduled_transport = mode;
            let state = || DynamicExecutionState {
                execution_complete: true,
                terminal_error: None,
                finalized: false,
                terminal_lowering_flushed: true,
                sync_handle: None,
            };
            engine.dynamic_state = Some(state());
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                engine.continue_processing(ByteMessage::outcomes_builder().build())
            }));
            assert!(matches!(outcome, Err(_) | Ok(Err(_))));
            // The failure itself must latch, panics included; a retry that
            // fails for another reason would not prove the shot is poisoned.
            assert!(engine.terminal_failure_error().is_some());
            assert!(!engine.dynamic_state.as_ref().unwrap().finalized);
            assert!(
                engine
                    .continue_processing(ByteMessage::outcomes_builder().build())
                    .is_err()
            );
            let mut cloned = engine.clone();
            assert!(
                cloned
                    .continue_processing(ByteMessage::outcomes_builder().build())
                    .is_err()
            );
            assert_eq!(ends.load(Ordering::SeqCst), 1);
            // Manual completed-worker fixture: provide the reclaimed interface
            // that a real completed worker returns before whole-host reset.
            engine.interface = Some(Box::new(crate::QisHeliosInterface::new()));
            engine.reset_all().unwrap();
            engine.dynamic_state = Some(state());
            assert!(matches!(
                engine
                    .continue_processing(ByteMessage::outcomes_builder().build())
                    .unwrap(),
                EngineStage::NeedsProcessing(_)
            ));
            assert_eq!(ends.load(Ordering::SeqCst), 2);
        }
    }
}
