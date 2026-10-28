//! Embedded decoding through the real worker/runtime/MonteCarlo path.
use super::*;
use crate::runtime_decoder::{RuntimeDecoderFactory, RuntimeDecoderSession};
use pecos_engines::{ClassicalControlEngineBuilder, MonteCarloEngine, StateVecEngine};
use std::sync::atomic::{AtomicBool, AtomicUsize};

fn program() -> &'static str {
    include_str!("../tests/fixtures/runtime_decode.ll")
}

struct Session {
    next: u64,
    answer: u64,
    calls: Arc<AtomicUsize>,
    fail: Arc<AtomicBool>,
}
impl RuntimeDecoderSession for Session {
    fn decode(&mut self, window: u64, instance: u64, syndrome: &[u8]) -> Result<u64, PecosError> {
        assert_eq!((window, instance, syndrome), (7, self.next, &[1][..]));
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail.swap(false, Ordering::SeqCst) {
            return Err(PecosError::Input("synthetic decoder failure".into()));
        }
        self.next += 1;
        Ok(self.answer)
    }
}
fn factory(
    answer: u64,
    starts: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    fail: Arc<AtomicBool>,
) -> RuntimeDecoderFactory {
    Arc::new(move || {
        starts.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Session {
            next: 0,
            answer,
            calls: calls.clone(),
            fail: fail.clone(),
        }))
    })
}
fn engine(
    source: &str,
    factory: Option<RuntimeDecoderFactory>,
    soft: bool,
    scheduled: bool,
) -> QisEngine {
    let mut builder = crate::qis_engine()
        .interface(crate::helios_interface_builder())
        .program(pecos_programs::Qis::from_string(source))
        .runtime(if soft {
            crate::selene_soft_rz_runtime().unwrap()
        } else {
            crate::selene_simple_runtime().unwrap()
        })
        .scheduled_batches(scheduled);
    if let Some(factory) = factory {
        builder = builder.runtime_decoder(factory);
    }
    builder.build().unwrap()
}

#[test]
fn live_frame_changes_later_measurements_in_loops_and_isolates_shots_workers() {
    use pecos_engines::noise::IntoNoiseModel;
    use pecos_engines::scheduled_frame::ScheduledIdleZ;
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for (soft, scheduled, answer) in [
        (false, false, 0),
        (false, false, 1),
        (true, false, 1),
        (false, true, 1),
    ] {
        let starts = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let f = factory(
            answer,
            starts.clone(),
            calls.clone(),
            Arc::new(AtomicBool::new(false)),
        );
        let mut builder = MonteCarloEngine::builder()
            .with_classical_engine(Box::new(engine(program(), Some(f), soft, scheduled)))
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
        }
        let mut mc = builder.build();
        for workers in [1, 2] {
            let shots = mc.run_with_workers(4, workers).unwrap();
            for shot in shots.shots {
                assert_eq!(
                    shot.data["after"],
                    Data::Vec(vec![Data::U32(u32::from(answer == 0)); 3])
                );
            }
        }
        assert_eq!(starts.load(Ordering::SeqCst), 8);
        assert_eq!(calls.load(Ordering::SeqCst), 24);
    }
}

fn drive(engine: &mut QisEngine) -> Result<Shot, PecosError> {
    let mut quantum = StateVecEngine::new(1);
    let mut stage = engine.start(())?;
    for _ in 0..1000 {
        match stage {
            EngineStage::Complete(shot) => return Ok(shot),
            EngineStage::NeedsProcessing(commands) => {
                stage = engine.continue_processing(quantum.process(commands)?)?;
            }
        }
    }
    panic!("decoder program did not complete");
}

#[test]
fn failure_is_sticky_until_reset_and_reuse_has_a_fresh_session() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let starts = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let f = factory(
        1,
        starts.clone(),
        calls.clone(),
        Arc::new(AtomicBool::new(true)),
    );
    let mut engine = engine(program(), Some(f), false, false);
    engine.set_num_qubits_hint(1);
    assert!(
        drive(&mut engine)
            .unwrap_err()
            .to_string()
            .contains("synthetic decoder failure")
    );
    assert!(
        engine
            .get_results()
            .unwrap_err()
            .to_string()
            .contains("synthetic decoder failure")
    );
    assert!(
        engine
            .continue_processing(ByteMessage::builder().build())
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    ClassicalEngine::reset(&mut engine).unwrap();
    assert_eq!(
        drive(&mut engine).unwrap().data["after"],
        Data::Vec(vec![Data::U32(0); 3])
    );
    assert_eq!(starts.load(Ordering::SeqCst), 2);
}

#[test]
fn missing_provider_and_duplicate_instance_fail_without_a_response() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut absent = engine(program(), None, false, false);
    absent.set_num_qubits_hint(1);
    assert!(
        drive(&mut absent)
            .unwrap_err()
            .to_string()
            .contains("no runtime decoder configured")
    );
    ClassicalEngine::reset(&mut absent).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let f = factory(
        1,
        Arc::new(AtomicUsize::new(0)),
        calls.clone(),
        Arc::new(AtomicBool::new(false)),
    );
    let source = program().replace("i64 %instance, ptr %data", "i64 0, ptr %data");
    let mut duplicate = engine(&source, Some(f), false, false);
    duplicate.set_num_qubits_hint(1);
    assert!(
        drive(&mut duplicate)
            .unwrap_err()
            .to_string()
            .contains("instance sequence")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
