//! Certification includes startup, execution, and repeatable completed reads.
use super::*;
use crate::RuntimeResult;
use crate::runtime::{ClassicalState, RuntimeError};
use std::sync::atomic::AtomicBool;

#[derive(Clone, Default)]
struct StartupRuntime {
    state: ClassicalState,
    fail: Arc<AtomicBool>,
}

impl QisRuntime for StartupRuntime {
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
        0
    }
    fn shot_start(&mut self, _: u64, _: Option<u64>) -> RuntimeResult<()> {
        if self.fail.load(Ordering::SeqCst) {
            Err(RuntimeError::ExecutionError(
                "shot_start injected failure".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Default)]
struct StartupInterface {
    fail_enable: Arc<AtomicBool>,
    worker_gate: Option<Arc<std::sync::Barrier>>,
}

impl crate::QisInterface for StartupInterface {
    fn load_program(&mut self, _: &[u8], _: ProgramFormat) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        if let Some(gate) = &self.worker_gate {
            gate.wait();
            return Err(InterfaceError::Other("worker injected failure".into()));
        }
        Ok(OperationList::new())
    }
    fn execute_with_measurements(
        &mut self,
        _: BTreeMap<usize, bool>,
    ) -> Result<OperationList, InterfaceError> {
        self.collect_operations()
    }
    fn name(&self) -> &'static str {
        "startup-test"
    }
    fn reset(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn supports_dynamic(&self) -> bool {
        true
    }
    fn enable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        if self.fail_enable.load(Ordering::SeqCst) {
            Err(InterfaceError::Other("enable injected failure".into()))
        } else {
            Ok(())
        }
    }
    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }
}

fn finish_empty_shot(engine: &mut QisEngine) {
    let mut stage = engine.start(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while matches!(stage, EngineStage::NeedsProcessing(_)) {
        assert!(Instant::now() < deadline);
        stage = engine
            .continue_processing(ByteMessage::builder().build())
            .unwrap();
    }
    assert_eq!(engine.shot_lifecycle, ShotLifecycle::Finalized);
}

fn assert_shot_start_failure(previous: bool) {
    let runtime = StartupRuntime::default();
    let fail = Arc::clone(&runtime.fail);
    let mut engine = QisEngine::new(Box::new(StartupInterface::default()), Box::new(runtime));
    if previous {
        finish_empty_shot(&mut engine);
    }
    fail.store(true, Ordering::SeqCst);
    assert!(
        engine
            .start(())
            .err()
            .expect("startup must fail")
            .to_string()
            .contains("shot_start injected failure")
    );
    assert!(
        engine
            .get_results()
            .unwrap_err()
            .to_string()
            .contains("shot_start injected failure")
    );
    assert!(matches!(engine.shot_lifecycle, ShotLifecycle::Failed(_)));
}

#[test]
fn first_shot_start_failure_is_latched() {
    assert_shot_start_failure(false);
}

#[test]
fn second_shot_start_failure_invalidates_certification() {
    assert_shot_start_failure(true);
}

fn assert_worker_startup_failure(submission: bool) {
    let interface = StartupInterface::default();
    let fail = Arc::clone(&interface.fail_enable);
    let mut engine = QisEngine::new(Box::new(interface), Box::new(StartupRuntime::default()));
    finish_empty_shot(&mut engine);
    let expected = if submission {
        engine.persistent_worker.as_mut().unwrap().work_tx.take();
        "Persistent worker channel closed"
    } else {
        fail.store(true, Ordering::SeqCst);
        "enable injected failure"
    };
    assert!(
        engine
            .start(())
            .err()
            .expect("startup must fail")
            .to_string()
            .contains(expected)
    );
    assert!(
        engine
            .get_results()
            .unwrap_err()
            .to_string()
            .contains(expected)
    );
}

#[test]
fn second_shot_enable_failure_invalidates_certification() {
    assert_worker_startup_failure(false);
}

#[test]
fn second_shot_submission_failure_invalidates_certification() {
    assert_worker_startup_failure(true);
}

fn assert_unfinished_shot(unreceived: bool) {
    let gate = Arc::new(std::sync::Barrier::new(2));
    let interface = StartupInterface {
        worker_gate: Some(Arc::clone(&gate)),
        ..Default::default()
    };
    let mut engine = QisEngine::new(Box::new(interface), Box::new(StartupRuntime::default()));
    assert!(engine.get_results().unwrap().data.is_empty());
    assert!(matches!(
        engine.start(()).unwrap(),
        EngineStage::NeedsProcessing(_)
    ));
    let midshot = engine.get_results();
    // Release the worker even if the mid-shot assertion will fail under mutation.
    gate.wait();
    let worker = engine.persistent_worker.as_mut().unwrap();
    worker.work_tx.take();
    worker.handle.take().unwrap().join().unwrap();
    let result = if unreceived {
        engine.get_results()
    } else {
        midshot
    };
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("shot not finished")
    );
    // Reading results did not consume the queued failure.
    assert!(engine.check_worker_complete());
    assert!(
        engine
            .get_results()
            .unwrap_err()
            .to_string()
            .contains("worker injected failure")
    );
}

#[test]
fn midshot_results_are_not_certified() {
    assert_unfinished_shot(false);
}

#[test]
fn unreceived_worker_failure_is_not_certified() {
    assert_unfinished_shot(true);
}

#[test]
fn certified_results_repeat_and_successful_reset_returns_idle() {
    let mut engine = QisEngine::new(
        Box::new(StartupInterface::default()),
        Box::new(StartupRuntime::default()),
    );
    finish_empty_shot(&mut engine);
    engine.measurement_results.insert(0, 1);
    for _ in 0..2 {
        assert_eq!(
            engine.get_results().unwrap().data.get("measurement_0"),
            Some(&Data::U32(1))
        );
    }
    engine.reset_all().unwrap();
    assert_eq!(engine.shot_lifecycle, ShotLifecycle::Idle);
    assert!(engine.get_results().unwrap().data.is_empty());
}

#[test]
fn clones_of_finalized_and_failed_engines_start_idle() {
    let runtime = StartupRuntime::default();
    let fail = Arc::clone(&runtime.fail);
    let mut engine = QisEngine::new(Box::new(StartupInterface::default()), Box::new(runtime));
    finish_empty_shot(&mut engine);
    engine.measurement_results.insert(7, 1);
    let finalized_clone = engine.clone();
    assert_eq!(finalized_clone.shot_lifecycle, ShotLifecycle::Idle);
    assert!(finalized_clone.get_results().unwrap().data.is_empty());

    fail.store(true, Ordering::SeqCst);
    assert!(engine.start(()).is_err());
    assert!(matches!(engine.shot_lifecycle, ShotLifecycle::Failed(_)));
    let failed_clone = engine.clone();
    assert_eq!(failed_clone.shot_lifecycle, ShotLifecycle::Idle);
    assert!(failed_clone.get_results().unwrap().data.is_empty());

    engine.reset_failure = Some("latched reset failure".into());
    let reset_failed_clone = engine.clone();
    assert_eq!(reset_failed_clone.shot_lifecycle, ShotLifecycle::Idle);
    assert!(
        reset_failed_clone
            .get_results()
            .unwrap_err()
            .to_string()
            .contains("latched reset failure")
    );
}
