//! Runtime IDs are per engine; RNG seeds retain the existing job derivation.
use super::*;
use crate::runtime::{ClassicalState, Result as RuntimeResult, RuntimeError};
use pecos_random::{derive_seed, nth_derived_seed};
use std::sync::atomic::AtomicBool;

type Starts = Arc<Mutex<Vec<(u64, u64)>>>;

#[derive(Clone, Default)]
struct RecordingRuntime {
    state: ClassicalState,
    starts: Starts,
    fail: Arc<AtomicBool>,
}
impl QisRuntime for RecordingRuntime {
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
    fn shot_start(&mut self, id: u64, seed: Option<u64>) -> RuntimeResult<()> {
        self.starts.lock().unwrap().push((id, seed.unwrap()));
        if self.fail.load(Ordering::SeqCst) {
            return Err(RuntimeError::ExecutionError(
                "injected start failure".into(),
            ));
        }
        self.state.shot_id = Some(id);
        Ok(())
    }
}

#[derive(Clone)]
struct EmptyInterface;
impl crate::QisInterface for EmptyInterface {
    fn load_program(&mut self, _: &[u8], _: ProgramFormat) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        Ok(OperationList::new())
    }
    fn execute_with_measurements(
        &mut self,
        _: BTreeMap<usize, bool>,
    ) -> Result<OperationList, InterfaceError> {
        self.collect_operations()
    }
    fn name(&self) -> &'static str {
        "shot-identity-test"
    }
    fn reset(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }
    fn supports_dynamic(&self) -> bool {
        true
    }
    fn enable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        Ok(())
    }
}
impl QisInterfaceBuilder for EmptyInterface {
    fn build_from_qis_program(&self, _: pecos_programs::Qis) -> Result<OperationList, PecosError> {
        Ok(OperationList::new())
    }
    fn build_from_interface(&self, ops: OperationList) -> Result<OperationList, PecosError> {
        Ok(ops)
    }
    fn name(&self) -> &'static str {
        "shot-identity-test"
    }
    fn create_dynamic_interface_from_qis(
        &self,
        _: pecos_programs::Qis,
    ) -> Result<BoxedInterface, PecosError> {
        Ok(Box::new(self.clone()))
    }
}
fn engine(runtime: RecordingRuntime) -> QisEngine {
    let mut engine = QisEngine::new(Box::new(EmptyInterface), Box::new(runtime));
    engine.set_dynamic_config(Box::new(EmptyInterface), "synthetic empty program");
    engine
}
fn finish(engine: &mut QisEngine) {
    let mut stage = engine.start(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while matches!(stage, EngineStage::NeedsProcessing(_)) {
        assert!(Instant::now() < deadline);
        stage = engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .unwrap();
    }
}

#[test]
fn runtime_ids_start_at_zero_without_changing_seeds_or_trace_numbers() {
    let runtime = RecordingRuntime::default();
    let starts = Arc::clone(&runtime.starts);
    let mut engine = engine(runtime);
    let trace = OperationTraceStore::default();
    engine.set_operation_trace_collector(Arc::clone(&trace));
    engine.set_seed(42);
    let mut rng = PecosRng::seed_from_u64(42);
    for _ in 0..3 {
        finish(&mut engine);
        assert_eq!(engine.current_shot_seed(), Some(rng.next_u64()));
        engine.reset_all().unwrap();
    }
    // Reseeding changes the stream but must not reuse a native identity.
    engine.set_seed(42);
    finish(&mut engine);
    let records = starts.lock().unwrap();
    assert_eq!(records[0].1, records[3].1);
    assert_eq!(
        records.iter().map(|r| r.0).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    let traces = trace.lock().unwrap();
    assert_eq!(
        traces
            .iter()
            .filter(|t| t.stage == "trace_complete")
            .map(|t| t.shot_index)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
}

#[test]
fn failed_start_consumes_identity_and_template_clone_restarts_at_zero() {
    let runtime = RecordingRuntime::default();
    let starts = Arc::clone(&runtime.starts);
    let fail = Arc::clone(&runtime.fail);
    let mut engine = engine(runtime);
    finish(&mut engine);
    fail.store(true, Ordering::SeqCst);
    assert!(engine.start(()).is_err());
    fail.store(false, Ordering::SeqCst);
    engine.reset_all().unwrap();
    finish(&mut engine);
    let mut cloned = engine.clone();
    finish(&mut cloned);
    assert_eq!(
        starts
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.0)
            .collect::<Vec<_>>(),
        [0, 1, 2, 0]
    );
}

#[test]
fn monte_carlo_local_ids_do_not_change_global_seed_derivation_or_replay() {
    use pecos_engines::{MonteCarloEngine, StateVecEngine};
    let mut first_seeds = None;
    for workers in [1, 2, 3] {
        let runtime = RecordingRuntime::default();
        let starts = Arc::clone(&runtime.starts);
        let mut mc = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(engine(runtime)))
            .with_quantum_engine(Box::new(StateVecEngine::new(1)))
            .with_seed(123)
            .build();
        let (_, report) = mc.run_with_workers_report_seeds(7, workers).unwrap();
        let mut expected = Vec::new();
        for worker in &report.workers {
            for shot in 0..worker.shots {
                let global = u64::try_from(worker.first_shot + shot).unwrap();
                let seed = derive_seed(
                    nth_derived_seed(report.base_seed, global),
                    "classical_engine",
                );
                expected.push((
                    u64::try_from(shot).unwrap(),
                    PecosRng::seed_from_u64(seed).next_u64(),
                ));
            }
        }
        // Runtime tasks can be stolen by Rayon threads, so use seed identity to
        // compare records; the local ID still pins each worker engine's sequence.
        let mut actual: Vec<_> = starts.lock().unwrap().clone();

        actual.sort_unstable();
        expected.sort_unstable();
        // Assert seeds first so the pre-fix failure identifies numbering only.
        let mut seeds: Vec<_> = actual.iter().map(|r| r.1).collect();
        seeds.sort_unstable();
        let mut expected_seeds: Vec<_> = expected.iter().map(|r| r.1).collect();
        expected_seeds.sort_unstable();
        assert_eq!(seeds, expected_seeds);
        if let Some(first) = &first_seeds {
            assert_eq!(&seeds, first);
        } else {
            first_seeds = Some(seeds);
        }
        assert_eq!(actual, expected);
        starts.lock().unwrap().clear();
        mc.run_with_workers_from_seed_report(&report).unwrap();
        let mut replay: Vec<_> = starts.lock().unwrap().clone();
        replay.sort_unstable();
        assert_eq!(replay, actual);
    }
}

#[test]
fn exhausted_runtime_ids_fail_without_wrapping_or_consuming_rng() {
    let runtime = RecordingRuntime::default();
    let starts = Arc::clone(&runtime.starts);
    let mut engine = engine(runtime);
    engine.last_runtime_shot_id = Some(u64::MAX - 1);
    finish(&mut engine);
    assert_eq!(starts.lock().unwrap()[0].0, u64::MAX);
    for _ in 0..2 {
        engine.reset_all().unwrap();
        let expected_draw = engine.rng.clone().next_u64();
        assert!(
            engine
                .start(())
                .err()
                .unwrap()
                .to_string()
                .contains("runtime shot IDs exhausted")
        );
        assert_eq!(engine.rng.clone().next_u64(), expected_draw);
        assert_eq!(starts.lock().unwrap().len(), 1);
    }
}

#[test]
fn exhausted_trace_index_fails_before_runtime_start_or_rng_draw() {
    let runtime = RecordingRuntime::default();
    let starts = Arc::clone(&runtime.starts);
    let mut engine = engine(runtime);
    engine.trace_shot_index = usize::MAX;
    let expected_draw = engine.rng.clone().next_u64();
    assert!(
        engine
            .start(())
            .err()
            .unwrap()
            .to_string()
            .contains("trace shot indices exhausted")
    );
    assert_eq!(engine.rng.clone().next_u64(), expected_draw);
    assert!(starts.lock().unwrap().is_empty());
}
