//! Public program fixtures through staged/scoped orchestration.
use super::*;
use pecos_engines::noise::{GeneralNoiseModel, IntoNoiseModel};
use pecos_engines::scheduled_frame::ScheduledIdleZ;
use std::sync::atomic::AtomicUsize;

#[derive(Clone)]
struct FeedbackNoise {
    pending: Option<ByteMessage>,
    exchanges: Arc<AtomicUsize>,
    rng: PecosRng,
}
impl ControlEngine for FeedbackNoise {
    type Input = ByteMessage;
    type Output = ByteMessage;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(
        &mut self,
        input: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        // This test profile explicitly treats staged empty waits as no-ops.
        if input.is_empty()? {
            return Ok(EngineStage::Complete(
                ByteMessage::outcomes_builder().build(),
            ));
        }
        self.pending = Some(input);
        Ok(EngineStage::NeedsProcessing(
            ByteMessage::quantum_operations_builder().mz(&[1]).build(),
        ))
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        if let Some(input) = self.pending.take() {
            assert_eq!(reply.outcomes()?, [0]);
            self.exchanges.fetch_add(1, Ordering::SeqCst);
            // Choose further simulator work from the noise-owned measurement.
            // Two H gates leave the ancilla unchanged for the next request.
            let mut commands = ByteMessage::quantum_operations_builder();
            commands.h(&[1]).h(&[1]);
            for gate in input.quantum_ops()? {
                commands.add_gate_command(&gate);
            }
            Ok(EngineStage::NeedsProcessing(commands.build()))
        } else {
            Ok(EngineStage::Complete(reply))
        }
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.pending = None;
        Ok(())
    }
}
impl pecos_core::RngManageable for FeedbackNoise {
    type Rng = PecosRng;
    fn set_rng(&mut self, rng: PecosRng) {
        self.rng = rng;
    }
    fn rng(&self) -> &PecosRng {
        &self.rng
    }
    fn rng_mut(&mut self) -> &mut PecosRng {
        &mut self.rng
    }
}
impl pecos_engines::noise::NoiseModel for FeedbackNoise {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[test]
fn scoped_monte_carlo_preserves_noise_owned_measurement_round_trips() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let source = super::program_shot_tests::program();
    let mut runs = Vec::new();
    for scoped in [false, true] {
        let exchanges = Arc::new(AtomicUsize::new(0));
        let mut mc = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(engine(&source, scoped, false, false)))
            .with_quantum_engine(Box::new(StateVecEngine::new(2)))
            .with_noise_model(Box::new(FeedbackNoise {
                pending: None,
                exchanges: exchanges.clone(),
                rng: PecosRng::seed_from_u64(0),
            }))
            .with_num_qubits(2)
            .with_seed(42)
            .build();
        let shots = mc.run_with_workers(7, 3).unwrap();
        assert!(exchanges.load(Ordering::SeqCst) >= 7);
        runs.push((shots, exchanges.load(Ordering::SeqCst)));
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn scoped_preserves_runtime_wire_batches_and_measurement_dependent_flow() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let source = super::program_shot_tests::program().replace(
        "    %read = call i1 @___read_future_bool(i64 0)",
        "    %read = call i1 @___read_future_bool(i64 0)\n    br i1 %read, label %yes, label %no\nyes:\n    call void @__quantum__qis__x__body(i64 0)\n    br label %join\nno:\n    br label %join\njoin:");
    for mode in [
        ScheduledTransport::Off,
        ScheduledTransport::V6,
        ScheduledTransport::V7,
    ] {
        for soft in [false, true] {
            let mut runs = Vec::new();
            for scoped in [false, true] {
                let mut engine = engine(&source, scoped, false, soft);
                engine.scheduled_transport = mode;
                let mut quantum =
                    super::terminal_lowering_tests::deferred_read_quantum_with_capacity(mode, 1);
                let mut wire = Vec::new();
                let empty_wait = ByteMessage::builder().build();
                let shot = engine
                    .run_with((), &mut |commands| {
                        // Exclude only legacy empty worker polls. Preserve every
                        // scheduled frame, including timed/custom-only frames.
                        if commands.as_bytes() != empty_wait.as_bytes() {
                            wire.push(commands.as_bytes().to_vec());
                        }
                        quantum.process(commands)
                    })
                    .unwrap();
                runs.push((shot, wire));
            }
            assert_eq!(runs[0], runs[1], "mode={mode:?}, soft={soft}");
        }
    }
}
use pecos_engines::{
    ClassicalControlEngineBuilder, MonteCarloEngine, QuantumSystem, StateVecEngine,
};

fn engine(source: &str, scoped: bool, scheduled: bool, soft: bool) -> QisEngine {
    let mut engine = crate::qis_engine()
        .interface(crate::helios_interface_builder())
        .program(pecos_programs::Qis::from_string(source))
        .runtime(if soft {
            crate::selene_soft_rz_runtime().unwrap()
        } else {
            crate::selene_simple_runtime().unwrap()
        })
        .scheduled_batches(scheduled)
        .build()
        .unwrap();
    engine.set_num_qubits_hint(1);
    engine.set_scoped_execution(scoped).unwrap();
    engine
}

#[test]
fn scoped_matches_staged_shot_rng_replay_and_worker_partitioning() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let source = super::program_shot_tests::program()
        .replace("@shot_label =", "@observed = private constant [8 x i8] c\"observed\"\ndeclare void @print_bool_selene(ptr, i64, i1)\n@shot_label =")
        .replace("    %read = call i1 @___read_future_bool(i64 0)", "    %read = call i1 @___read_future_bool(i64 0)\n    call void @print_bool_selene(ptr @observed, i64 8, i1 %read)");
    for scheduled in [false, true] {
        for soft in [false, true] {
            for workers in [1, 3] {
                let mut runs = Vec::new();
                for scoped in [false, true] {
                    let mut builder = MonteCarloEngine::builder()
                        .with_classical_engine(Box::new(engine(&source, scoped, scheduled, soft)))
                        .with_quantum_engine(Box::new(StateVecEngine::new(1)))
                        .with_num_qubits(1)
                        .with_seed(42);
                    if scheduled {
                        builder = builder.with_noise_model(
                            ScheduledIdleZ::new(1, 0.0, 0.0, 0.0)
                                .unwrap()
                                .into_noise_model()
                                .unwrap(),
                        );
                    } else {
                        builder = builder.with_noise_model(Box::new(
                            GeneralNoiseModel::builder().with_p1(0.2).build(),
                        ));
                    }
                    let mut mc = builder.build();
                    let (shots, report) = mc.run_with_workers_report_seeds(7, workers).unwrap();
                    assert_eq!(
                        shots,
                        mc.run_with_workers_from_seed_report(&report).unwrap()
                    );
                    for (i, shot) in shots.shots.iter().enumerate() {
                        assert!(shot.data.contains_key("observed"));
                        assert_eq!(shot.data["shot"], Data::U64(i as u64));
                        assert_eq!(shot.data["abi"], Data::U64(i as u64));
                    }
                    runs.push(shots);
                }
                assert_eq!(
                    runs[0], runs[1],
                    "scheduled={scheduled}, soft={soft}, workers={workers}"
                );
            }
        }
    }
}

#[test]
fn scoped_preserves_host_error_panic_and_requires_whole_host_reset() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let source = super::program_shot_tests::program();
    for panic in [false, true] {
        let mut engine = engine(&source, true, false, false);
        let caller = std::thread::current().id();
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.run_with((), &mut |_| {
                assert_eq!(std::thread::current().id(), caller);
                if panic {
                    std::panic::panic_any(73_u32);
                }
                Err(PecosError::RuntimeIndexOutOfBounds {
                    index: 17,
                    length: 2,
                })
            })
        }));
        if panic {
            assert_eq!(*attempt.unwrap_err().downcast::<u32>().unwrap(), 73);
        } else {
            assert!(matches!(
                attempt.unwrap(),
                Err(PecosError::RuntimeIndexOutOfBounds {
                    index: 17,
                    length: 2
                })
            ));
        }
        assert!(engine.interface.is_some());
        assert!(engine.persistent_worker.is_none());
        assert!(
            engine
                .run_with((), &mut |_| panic!("failed shot must not run"))
                .is_err()
        );
        let mut cloned = engine.clone();
        assert!(
            cloned
                .run_with((), &mut |_| panic!("clone must retain failure"))
                .is_err()
        );
        engine.set_scoped_execution(false).unwrap();
        assert!(engine.start(()).is_err());
        engine.set_scoped_execution(true).unwrap();
        ControlEngine::reset(&mut engine).unwrap();
        let mut quantum = QuantumSystem::new(
            Box::new(GeneralNoiseModel::builder().build()),
            Box::new(StateVecEngine::new(1)),
        );
        let shot = engine
            .run_with((), &mut |commands| quantum.process(commands))
            .unwrap();
        assert_eq!(shot.data["shot"], Data::U64(1));
        assert_eq!(shot.data["abi"], Data::U64(1));
    }
}

#[test]
fn scoped_caches_all_outcomes_without_extra_batches() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let source = r"
declare i32 @__quantum__qis__m__body(i64, i64)
declare i1 @___read_future_bool(i64)
declare void @__quantum__qis__x__body(i64)
define i64 @qmain(i64 %arg) {
  %a = call i32 @__quantum__qis__m__body(i64 0, i64 0)
  %b = call i32 @__quantum__qis__m__body(i64 0, i64 1)
  %ra = call i1 @___read_future_bool(i64 0)
  call void @__quantum__qis__x__body(i64 0)
  %rb = call i1 @___read_future_bool(i64 1)
  call void @__quantum__qis__x__body(i64 0)
  ret i64 0
}";
    let mut runs = Vec::new();
    for scoped in [false, true] {
        let mut engine = QisEngine::new(
            Box::new(crate::QisHeliosInterface::new()),
            Box::new(super::tests::DummyRuntime::default()),
        );
        engine
            .load_program(source.as_bytes(), ProgramFormat::LlvmIrText)
            .unwrap();
        engine.set_scoped_execution(scoped).unwrap();
        let mut batches = Vec::new();
        let mut quantum = StateVecEngine::new(1);
        engine
            .run_with((), &mut |commands| {
                let gates = commands.quantum_ops()?;
                // Staged orchestration can emit empty worker polls; compare actual
                // gate-batch boundaries, which affect arbitrary noise controllers.
                if !gates.is_empty() {
                    batches.push(gates);
                }
                quantum.process(commands)
            })
            .unwrap();
        runs.push(batches);
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn rejected_staged_restart_preserves_the_running_shot() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut engine = engine(&super::program_shot_tests::program(), false, false, false);
    let mut stage = engine.start(()).unwrap();
    assert!(engine.start(()).is_err());
    assert_eq!(engine.shot_lifecycle, ShotLifecycle::Running);
    let mut quantum = StateVecEngine::new(1);
    loop {
        match stage {
            EngineStage::Complete(_) => break,
            EngineStage::NeedsProcessing(commands) => {
                stage = engine
                    .continue_processing(quantum.process(commands).unwrap())
                    .unwrap();
            }
        }
    }
}

#[test]
fn scoped_rejects_staged_continuation_without_mutation() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut engine = engine(&super::program_shot_tests::program(), true, false, false);
    let mut quantum = StateVecEngine::new(1);
    let shot = engine
        .run_with((), &mut |commands| quantum.process(commands))
        .unwrap();
    assert!(
        engine
            .continue_processing(ByteMessage::outcomes_builder().build())
            .is_err()
    );
    assert!(
        engine
            .continue_processing(ByteMessage::outcomes_builder().add_outcomes(&[1]).build())
            .is_err()
    );
    assert_eq!(ClassicalEngine::get_results(&engine).unwrap(), shot);
}

#[test]
fn nested_interface_entry_cannot_mutate_outer_execution() {
    use crate::qis_interface::QisInterface;
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut nested = crate::QisHeliosInterface::new();
    nested
        .load_program(
            b"define i64 @qmain(i64 %arg) { ret i64 0 }",
            ProgramFormat::LlvmIrText,
        )
        .unwrap();
    let mut engine = engine(&super::program_shot_tests::program(), true, false, false);
    let mut quantum = StateVecEngine::new(1);
    let shot = engine
        .run_with((), &mut |commands| {
            assert!(nested.collect_operations().is_err());
            quantum.process(commands)
        })
        .unwrap();
    assert_eq!(shot.data["shot"], Data::U64(0));
    assert_eq!(shot.data["abi"], Data::U64(0));
    // Registration guard is released after the outer invocation.
    nested.collect_operations().unwrap();
}

struct PanicOnTeardown {
    inner: BoxedInterface,
    retry: bool,
}
impl crate::qis_interface::QisInterface for PanicOnTeardown {
    fn load_program(&mut self, bytes: &[u8], format: ProgramFormat) -> Result<(), InterfaceError> {
        self.inner.load_program(bytes, format)
    }
    fn collect_operations(&mut self) -> Result<OperationList, InterfaceError> {
        self.inner.collect_operations()
    }
    fn execute_with_measurements(
        &mut self,
        values: BTreeMap<usize, bool>,
    ) -> Result<OperationList, InterfaceError> {
        self.inner.execute_with_measurements(values)
    }
    fn name(&self) -> &'static str {
        "teardown-panic fixture"
    }
    fn reset(&mut self) -> Result<(), InterfaceError> {
        self.inner.reset()
    }
    fn supports_dynamic(&self) -> bool {
        self.inner.supports_dynamic()
    }
    fn supports_scoped_execution(&self) -> bool {
        self.inner.supports_scoped_execution()
    }
    fn enable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        self.inner.enable_dynamic_mode()
    }
    fn set_native_selene_handles(&mut self, enabled: bool) -> Result<(), InterfaceError> {
        self.inner.set_native_selene_handles(enabled)
    }
    fn get_sync_handle(&self) -> Option<Box<dyn DynamicSyncHandle>> {
        self.inner.get_sync_handle()
    }
    fn collect_operations_scoped(
        &mut self,
        request: &mut dyn FnMut(u64) -> Result<(), InterfaceError>,
    ) -> Result<OperationList, InterfaceError> {
        if !self.retry {
            return self.inner.collect_operations_scoped(request);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.collect_operations_scoped(request)
        }));
        if !matches!(result, Ok(Ok(_))) {
            // A misbehaving third-party interface converts the first failure
            // and retries. The engine must retain the original and refuse work.
            assert!(request(pecos_qis_ffi_types::RUNTIME_REQUEST_RESULT_ID as u64).is_err());
            return Err(InterfaceError::Other("converted host failure".into()));
        }
        result.unwrap()
    }
    fn disable_dynamic_mode(&mut self) -> Result<(), InterfaceError> {
        self.inner.disable_dynamic_mode()?;
        std::panic::panic_any(91_u32)
    }
}

#[test]
fn teardown_panic_does_not_replace_primary_host_failure() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut engine = engine(&super::program_shot_tests::program(), true, false, false);
    engine.interface = Some(Box::new(PanicOnTeardown {
        inner: engine.interface.take().unwrap(),
        retry: false,
    }));
    let result = engine.run_with((), &mut |_| {
        Err(PecosError::RuntimeIndexOutOfBounds {
            index: 17,
            length: 2,
        })
    });
    assert!(matches!(
        result,
        Err(PecosError::RuntimeIndexOutOfBounds {
            index: 17,
            length: 2
        })
    ));
    assert!(engine.reset_failure.as_ref().unwrap().contains("teardown"));
    assert!(engine.interface.is_some());
}

#[test]
fn scoped_both_entries_reuse_context_outputs_and_terminal_tail() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for void in [false, true] {
        let mut source = super::program_shot_tests::program();
        // A tail after the last native query must still be delivered/certified.
        source = source.replace(
            "    ret i64 0",
            "    call void @__quantum__qis__x__body(i64 0)\n    ret i64 0",
        );
        if void {
            source = source
                .replace("define i64 @qmain(i64 %arg)", "define void @main()")
                .replace("ret i64 0", "ret void");
        }
        let mut runs = Vec::new();
        for scoped in [false, true] {
            let mut engine = engine(&source, scoped, false, false);
            let mut shots = Vec::new();
            for _ in 0..3 {
                ControlEngine::reset(&mut engine).unwrap();
                let mut quantum = StateVecEngine::new(1);
                let mut gates = Vec::new();
                let shot = engine
                    .run_with((), &mut |commands| {
                        gates.extend(commands.quantum_ops()?);
                        quantum.process(commands)
                    })
                    .unwrap();
                assert_eq!(engine.shot_lifecycle, ShotLifecycle::Finalized);
                assert_eq!(engine.persistent_worker.is_none(), scoped);
                shots.push((shot, gates));
            }
            runs.push(shots);
        }
        assert_eq!(runs[0], runs[1]);
    }
}

#[test]
fn custom_interface_cannot_convert_or_retry_a_host_panic() {
    let _lock = crate::test_env::ENV_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut engine = engine(&super::program_shot_tests::program(), true, false, false);
    engine.interface = Some(Box::new(PanicOnTeardown {
        inner: engine.interface.take().unwrap(),
        retry: true,
    }));
    let mut calls = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.run_with((), &mut |_| {
            calls += 1;
            std::panic::panic_any(73_u32)
        })
    }));
    assert_eq!(*result.unwrap_err().downcast::<u32>().unwrap(), 73);
    assert_eq!(calls, 1);
    assert!(engine.interface.is_some());
    assert!(engine.reset_failure.as_ref().unwrap().contains("teardown"));
}
