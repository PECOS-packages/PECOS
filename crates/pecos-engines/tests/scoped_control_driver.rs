//! The scoped driver keeps classical control and all noise feedback on its caller.
use pecos_core::{RngManageable, errors::PecosError};
use pecos_engines::monte_carlo::{MonteCarloEngineBuilder, worker_scope::WorkerScope};
use pecos_engines::noise::NoiseModel;
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{
    ByteMessage, ClassicalEngine, ControlEngine, Engine, EngineStage, HybridEngine, QuantumSystem,
    StateVecEngine,
};
use pecos_random::PecosRng;
use std::any::Any;
use std::cell::Cell;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

thread_local! { static IN_SCOPE: Cell<bool> = const { Cell::new(false) }; }
struct Scope;
impl WorkerScope for Scope {
    fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                IN_SCOPE.set(self.0);
            }
        }
        let _restore = Restore(IN_SCOPE.replace(true));
        work();
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Behavior {
    Feedback,
    Error,
    Panic,
}
#[derive(Clone)]
struct FeedbackNoise {
    rng: PecosRng,
    pending: Option<ByteMessage>,
    calls: Arc<AtomicUsize>,
    replies: Arc<AtomicUsize>,
    behavior: Behavior,
    check_scope: bool,
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.check_scope {
            assert!(IN_SCOPE.get());
        }
        match self.behavior {
            Behavior::Error => {
                return Err(PecosError::RuntimeIndexOutOfBounds {
                    index: 17,
                    length: 2,
                });
            }
            Behavior::Panic => std::panic::panic_any(73_u32),
            Behavior::Feedback => {}
        }
        self.pending = Some(input);
        // A noise-generated measurement, not a measurement requested by the program.
        Ok(EngineStage::NeedsProcessing(
            ByteMessage::quantum_operations_builder().mz(&[1]).build(),
        ))
    }
    fn continue_processing(
        &mut self,
        reply: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, ByteMessage>, PecosError> {
        self.replies.fetch_add(1, Ordering::SeqCst);
        if let Some(input) = self.pending.take() {
            assert_eq!(reply.outcomes()?, vec![0]);
            let mut commands = ByteMessage::quantum_operations_builder();
            commands.add_gate_command(&pecos_engines::Gate::x(&[0]));
            for gate in input.quantum_ops()? {
                commands.add_gate_command(&gate);
            }
            Ok(EngineStage::NeedsProcessing(commands.build()))
        } else {
            Ok(EngineStage::Complete(reply))
        }
    }
    fn run_with(
        &mut self,
        _input: ByteMessage,
        _process: &mut dyn FnMut(ByteMessage) -> Result<ByteMessage, PecosError>,
    ) -> Result<ByteMessage, PecosError> {
        // Explicit boxed calls honor overrides; QuantumSystem's inner loop stays staged.
        Ok(ByteMessage::outcomes_builder().add_outcomes(&[99]).build())
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.pending = None;
        Ok(())
    }
}
impl RngManageable for FeedbackNoise {
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
impl NoiseModel for FeedbackNoise {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Clone, Default)]
struct InlineControl {
    input_override: Option<ByteMessage>,
    swallow: bool,
    shot: u64,
    check_scope: bool,
}
impl Engine for InlineControl {
    type Input = ();
    type Output = Shot;
    fn process(&mut self, (): ()) -> Result<Shot, PecosError> {
        panic!("requires controlled engine")
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}
impl ClassicalEngine for InlineControl {
    fn num_qubits(&self) -> usize {
        2
    }
    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        panic!("staged path used")
    }
    fn handle_measurements(&mut self, _: ByteMessage) -> Result<(), PecosError> {
        panic!("staged path used")
    }
    fn get_results(&self) -> Result<Shot, PecosError> {
        Ok(Shot::default())
    }
    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }
    fn set_program_shot(&mut self, shot: u64) {
        self.shot = shot;
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
impl ControlEngine for InlineControl {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;
    fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        panic!("override not dispatched")
    }
    fn continue_processing(
        &mut self,
        _: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        panic!("override not dispatched")
    }
    fn run_with(
        &mut self,
        (): (),
        process: &mut dyn FnMut(ByteMessage) -> Result<ByteMessage, PecosError>,
    ) -> Result<Shot, PecosError> {
        if self.check_scope {
            assert!(IN_SCOPE.get());
        }
        let caller = std::thread::current().id();
        let mut shot = Shot::default();
        shot.data.insert("shot".into(), Data::U64(self.shot));
        if let Some(input) = &self.input_override {
            if process(input.clone()).is_err() {
                assert!(process(input.clone()).is_err());
            }
            return Ok(shot);
        }
        if self.swallow {
            // A bad controller cannot hide a controlled-engine failure or retry it.
            assert!(process(ByteMessage::builder().build()).is_err());
            assert!(process(ByteMessage::builder().build()).is_err());
            return Ok(shot);
        }
        let first = process(ByteMessage::quantum_operations_builder().mz(&[0]).build())?;
        assert_eq!(first.outcomes()?, vec![1]);
        assert_eq!(std::thread::current().id(), caller);
        // Classical flow depends on a noise-generated measurement and state change.
        let second = process(ByteMessage::quantum_operations_builder().mz(&[0]).build())?;
        shot.data
            .insert("first".into(), Data::U32(first.outcomes()?[0]));
        shot.data
            .insert("second".into(), Data::U32(second.outcomes()?[0]));
        Ok(shot)
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}
fn host(
    behavior: Behavior,
    check_scope: bool,
) -> (HybridEngine, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let replies = Arc::new(AtomicUsize::new(0));
    let host = HybridEngine {
        classical_engine: Box::new(InlineControl {
            swallow: !matches!(behavior, Behavior::Feedback),
            check_scope,
            ..Default::default()
        }),
        quantum_system: QuantumSystem::new(
            Box::new(FeedbackNoise {
                rng: PecosRng::seed_from_u64(9),
                pending: None,
                calls: calls.clone(),
                replies: replies.clone(),
                behavior,
                check_scope,
            }),
            Box::new(StateVecEngine::new(2)),
        ),
    };
    (host, calls, replies)
}
#[test]
fn hybrid_dispatches_override_and_keeps_noise_measurement_feedback() {
    let (mut engine, calls, replies) = host(Behavior::Feedback, false);
    for _ in 0..2 {
        engine.reset().unwrap();
        let shot = engine.run_shot().unwrap();
        assert_eq!(shot.data["first"].as_u32(), Some(1));
        assert_eq!(shot.data["second"].as_u32(), Some(0));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert_eq!(replies.load(Ordering::SeqCst), 8);
}
#[test]
fn override_cannot_swallow_or_retry_controlled_error() {
    let (mut engine, calls, _) = host(Behavior::Error, false);
    assert!(matches!(
        engine.run_shot(),
        Err(PecosError::RuntimeIndexOutOfBounds {
            index: 17,
            length: 2
        })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
#[test]
fn override_cannot_swallow_or_retry_controlled_panic() {
    let (mut engine, calls, _) = host(Behavior::Panic, false);
    let panic =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.run_shot())).unwrap_err();
    assert_eq!(panic.downcast_ref::<u32>(), Some(&73));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
#[test]
fn monte_carlo_retains_scope_shot_identity_reset_and_feedback() {
    let (engine, calls, replies) = host(Behavior::Feedback, true);
    let mut mc = MonteCarloEngineBuilder::new()
        .with_hybrid_engine(engine)
        .with_seed(123)
        .build();
    let shots = mc.run_with_workers_in_scope(4, 2, &Scope).unwrap();
    assert_eq!(shots.shots.len(), 4);
    for (index, shot) in shots.shots.iter().enumerate() {
        assert!(matches!(shot.data["shot"], Data::U64(value) if value == index as u64));
        assert_eq!(shot.data["first"].as_u32(), Some(1));
        assert_eq!(shot.data["second"].as_u32(), Some(0));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 8);
    assert_eq!(replies.load(Ordering::SeqCst), 16);
}

#[derive(Clone)]
struct Staged {
    calls: usize,
}
impl ControlEngine for Staged {
    type Input = bool;
    type Output = usize;
    type EngineInput = Vec<u8>;
    type EngineOutput = usize;
    fn start(&mut self, complete: bool) -> Result<EngineStage<Vec<u8>, usize>, PecosError> {
        if complete {
            Ok(EngineStage::Complete(0))
        } else {
            Ok(EngineStage::NeedsProcessing(vec![]))
        }
    }
    fn continue_processing(
        &mut self,
        value: usize,
    ) -> Result<EngineStage<Vec<u8>, usize>, PecosError> {
        self.calls += 1;
        assert_eq!(value, 41 + self.calls);
        if self.calls == 2 {
            Ok(EngineStage::Complete(value))
        } else {
            Ok(EngineStage::NeedsProcessing(vec![1]))
        }
    }
    fn reset(&mut self) -> Result<(), PecosError> {
        self.calls = 0;
        Ok(())
    }
}
#[test]
fn default_driver_preserves_empty_requests_order_and_borrowed_callback() {
    let mut control = Staged { calls: 0 };
    let observed = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let output = control
        .run_with(false, &mut |input| {
            observed.borrow_mut().push(input);
            Ok(41 + observed.borrow().len())
        })
        .unwrap();
    assert_eq!(output, 43);
    assert_eq!(*observed.borrow(), vec![vec![], vec![1]]);
}
#[test]
fn default_driver_stops_at_sink_failure_and_immediate_completion() {
    let mut control = Staged { calls: 0 };
    assert_eq!(
        control
            .run_with(true, &mut |_| panic!("no callback for completion"))
            .unwrap(),
        0
    );
    let result = control.run_with(false, &mut |_| Err(PecosError::RuntimeDivisionByZero));
    assert!(matches!(result, Err(PecosError::RuntimeDivisionByZero)));
    assert_eq!(control.calls, 0);
}
#[test]
fn generic_box_forwards_override() {
    let mut control: Box<
        dyn ControlEngine<
                Input = (),
                Output = Shot,
                EngineInput = ByteMessage,
                EngineOutput = ByteMessage,
            >,
    > = Box::new(InlineControl::default());
    let mut calls = 0;
    control
        .run_with((), &mut |_| {
            calls += 1;
            Ok(ByteMessage::outcomes_builder()
                .add_outcomes(&[usize::from(calls == 1)])
                .build())
        })
        .unwrap();
    assert_eq!(calls, 2);
}
#[test]
fn classical_box_forwards_override() {
    let mut control: Box<dyn pecos_engines::ClassicalControlEngine> =
        Box::new(InlineControl::default());
    let mut calls = 0;
    ControlEngine::run_with(&mut control, (), &mut |_| {
        calls += 1;
        Ok(ByteMessage::outcomes_builder()
            .add_outcomes(&[usize::from(calls == 1)])
            .build())
    })
    .unwrap();
    assert_eq!(calls, 2);
}

#[test]
fn noise_box_forwards_override_without_changing_inner_noise_loop() {
    let mut noise: Box<dyn NoiseModel> = Box::new(FeedbackNoise {
        rng: PecosRng::seed_from_u64(0),
        pending: None,
        calls: Arc::new(AtomicUsize::new(0)),
        replies: Arc::new(AtomicUsize::new(0)),
        behavior: Behavior::Feedback,
        check_scope: false,
    });
    let result = ControlEngine::run_with(&mut noise, ByteMessage::builder().build(), &mut |_| {
        panic!("staged default used")
    })
    .unwrap();
    assert_eq!(result.outcomes().unwrap(), vec![99]);
}

#[test]
fn scoped_failure_requires_whole_host_reset() {
    use pecos_engines::noise::{GeneralNoiseModel, IntoNoiseModel};
    use pecos_engines::runtime_frame::{
        FrameLimits, FrameRecord, RuntimeNoise, ShotContext, encode_frame,
    };
    let cfg = RuntimeNoise::new(GeneralNoiseModel::builder(), 2, FrameLimits::default()).unwrap();
    let mut host = HybridEngine {
        classical_engine: Box::new(InlineControl {
            input_override: Some(ByteMessage::new(&[1, 2, 3])),
            ..Default::default()
        }),
        quantum_system: QuantumSystem::new(
            cfg.into_noise_model().unwrap(),
            Box::new(StateVecEngine::new(2)),
        ),
    };
    let context = |shot| ShotContext {
        run: 7,
        worker: 0,
        shot,
    };
    assert!(matches!(host.run_shot_with_context(context(0)),
        Err(PecosError::Input(message)) if message == "legacy frame byte limit"));
    host.quantum_system.reset().unwrap();
    assert!(host.quantum_system.begin_shot(context(1)).is_err());
    host.classical_engine
        .as_any_mut()
        .downcast_mut::<InlineControl>()
        .unwrap()
        .input_override =
        Some(encode_frame(&[FrameRecord::gate(pecos_engines::Gate::mz(&[0]))]).unwrap());
    host.reset().unwrap();
    host.run_shot_with_context(context(2)).unwrap();
}
