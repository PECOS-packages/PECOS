// Deterministic ownership and deadline tests; no global state or FFI library.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

struct SyncHandle {
    fail: Arc<AtomicBool>,
    calls: Arc<AtomicU64>,
}
impl DynamicSyncHandle for SyncHandle {
    fn wait_for_need_result(&self, _: u64) -> Option<u64> {
        None
    }
    fn set_measurement_result(&self, _: u64, _: bool) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn signal_result_ready(&self) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn get_pending_operations(&self) -> Result<Vec<Operation>, InterfaceError> {
        Ok(vec![])
    }
    fn abort_execution(&self) -> Result<(), InterfaceError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail.load(Ordering::SeqCst) {
            Err(InterfaceError::Other("abort failed".into()))
        } else {
            Ok(())
        }
    }
    fn get_named_results(
        &self,
    ) -> Result<BTreeMap<String, pecos_qis_ffi_types::NamedResult>, InterfaceError> {
        Ok(BTreeMap::new())
    }
    fn get_named_result_traces(&self) -> Result<Vec<NamedResultTrace>, InterfaceError> {
        Ok(vec![])
    }
}

fn running() -> (
    QisEngine,
    Sender<WorkerResult>,
    Arc<AtomicBool>,
    Arc<AtomicU64>,
) {
    let mut engine = QisEngine::with_runtime(Box::new(ShotEndAndResetFailRuntime::default()));
    let (work_tx, work_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let fail = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicU64::new(0));
    engine.persistent_worker = Some(PersistentDynamicWorker {
        work_tx,
        result_rx: Mutex::new(result_rx),
        handle: std::thread::spawn(move || while work_rx.recv().is_ok() {}),
    });
    engine.dynamic_state = Some(DynamicExecutionState {
        execution_complete: false,
        terminal_error: None,
        finalized: false,
        terminal_lowering_flushed: false,
        sync_handle: Some(Box::new(SyncHandle {
            fail: Arc::clone(&fail),
            calls: Arc::clone(&calls),
        })),
    });
    (engine, result_tx, fail, calls)
}

fn success() -> (OperationList, BoxedInterface) {
    (
        OperationList::new(),
        Box::new(FlakyResetInterface::default()),
    )
}
fn queue(sender: &Sender<WorkerResult>, result: WorkerResult) {
    assert!(sender.send(result).is_ok());
}

#[test]
fn abort_failure_retains_state_and_refuses_continue_before_input() {
    let (mut engine, sender, fail, _) = running();
    fail.store(true, Ordering::SeqCst);
    let error = engine.reset_all().unwrap_err();
    assert!(error.to_string().contains("abort failed"));
    assert!(engine.dynamic_state.is_some());
    assert!(engine.interface.is_none());
    assert!(engine.start(()).is_err());
    let Err(error) = engine.continue_processing(ByteMessage::builder().build()) else {
        panic!("latched continue succeeded")
    };
    assert!(error.to_string().contains("failed reset"));
    queue(&sender, Ok(success()));
    engine.reset_all().unwrap();
    assert!(engine.reset_failure.is_none());
}

#[test]
fn timeout_then_late_result_recovers_without_another_abort() {
    let (mut engine, sender, fail, calls) = running();
    let error = engine.reset_all_with_timeout(Duration::ZERO).unwrap_err();
    assert!(error.to_string().contains("timed out"));
    assert!(engine.dynamic_state.is_some());
    assert!(engine.persistent_worker.is_some());
    assert!(engine.reset_failure.is_some());
    fail.store(true, Ordering::SeqCst);
    queue(&sender, Ok(success()));
    engine.reset_all().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(engine.interface.is_some());
    engine.reset_all().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn queued_and_consumed_results_need_no_abort_or_second_receive() {
    for consumed in [false, true] {
        let (mut engine, sender, fail, calls) = running();
        fail.store(true, Ordering::SeqCst);
        queue(&sender, Ok(success()));
        if consumed {
            assert!(engine.check_worker_complete());
        }
        engine.reset_all().unwrap();
        assert!(engine.interface.is_some());
        assert!(engine.dynamic_state.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

struct CancelledInterface {
    teardown_fails: bool,
}
impl crate::QisInterface for CancelledInterface {
    fn load_program(&mut self, _: &[u8], _: ProgramFormat) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        Err(InterfaceError::ProgramError(
            pecos_qis_ffi_types::ProgramError::Cancelled,
        ))
    }
    fn execute_with_measurements(
        &mut self,
        _: BTreeMap<usize, bool>,
    ) -> Result<OperationList, InterfaceError> {
        self.collect_operations()
    }
    fn name(&self) -> &'static str {
        "cancelled"
    }
    fn reset(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        if self.teardown_fails {
            Err(InterfaceError::Other("exit failure".into()))
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_and_teardown_failure_remain_separate() {
    for teardown in [false, true] {
        let (mut engine, sender, _, _) = running();
        let worker = PersistentDynamicWorker::new();
        worker
            .execute(Box::new(CancelledInterface {
                teardown_fails: teardown,
            }))
            .unwrap();
        let result = worker
            .recv_result_until(Instant::now() + Duration::from_secs(2))
            .unwrap();
        queue(&sender, result);
        let reset = engine.reset_all();
        if teardown {
            let error = reset.unwrap_err().to_string();
            assert!(error.contains("cancelled"));
            assert!(error.contains("exit failure"));
            assert!(engine.reset_failure.is_some());
            assert!(engine.interface.is_some());
            engine.reset_all().unwrap();
        } else {
            reset.unwrap();
        }
        assert!(engine.interface.is_some());
    }
}

#[test]
fn disconnected_poisoned_and_missing_interface_stay_latched() {
    for mode in 0..3 {
        let (mut engine, sender, _, _) = running();
        if mode == 0 {
            drop(sender);
        } else if mode == 1 {
            let receiver = &engine.persistent_worker.as_ref().unwrap().result_rx;
            std::thread::scope(|scope| {
                assert!(
                    scope
                        .spawn(|| {
                            let _guard = receiver.lock().unwrap();
                            panic!("poison receiver");
                        })
                        .join()
                        .is_err()
                );
            });
        } else {
            engine.dynamic_state.as_mut().unwrap().execution_complete = true;
        }
        for _ in 0..2 {
            assert!(engine.reset_all().is_err());
            assert!(engine.reset_failure.is_some());
            assert!(engine.dynamic_state.is_some());
        }
    }
}

// An engine that never ran a dynamic shot has no worker interface to reclaim,
// so reset succeeds even without an interface.
#[test]
fn reset_without_interface_or_shot_succeeds() {
    let mut engine = QisEngine::with_runtime(Box::new(DummyRuntime::default()));
    ControlEngine::reset(&mut engine).unwrap();
    Engine::reset(&mut engine).unwrap();
    ClassicalEngine::reset(&mut engine).unwrap();
}
