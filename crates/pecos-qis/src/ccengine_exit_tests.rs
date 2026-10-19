//! Exit diagnostics through a live worker and runtime, not named output markers.
use super::*;
use pecos_engines::shot_results::ProgramExit;
use pecos_engines::{ClassicalControlEngineBuilder, MonteCarloEngine, StateVecEngine};

fn engine(scheduled: bool, code: i32) -> QisEngine {
    let terminate = format!("call void @panic(i32 {code}, ptr @pascal)");
    let source = format!(
        r#"
@pascal = private constant [5 x i8] c"\04done"
@label = private constant [4 x i8] c"shot"
@after = private constant [5 x i8] c"after"
declare i64 @get_current_shot()
declare void @print_uint_selene(ptr, i64, i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i1 @___read_future_bool(i64)
declare void @panic(i32, ptr)
define i64 @qmain(i64 %arg) #0 {{
    %shot = call i64 @get_current_shot()
    call void @print_uint_selene(ptr @label, i64 4, i64 %shot)
    %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %read = call i1 @___read_future_bool(i64 0)
    %odd = trunc i64 %shot to i1
    br i1 %odd, label %terminate, label %normal
terminate:
    {terminate}
    unreachable
normal:
    call void @print_uint_selene(ptr @after, i64 5, i64 7)
    ret i64 0
}}
attributes #0 = {{ "EntryPoint" }}
"#
    );
    crate::qis_engine()
        .interface(crate::helios_interface_builder())
        .program(pecos_programs::Qis::from_string(source))
        .runtime(crate::selene_simple_runtime().unwrap())
        .scheduled_batches(scheduled)
        .build()
        .unwrap()
}

#[test]
fn exits_preserve_identity_partial_results_and_worker_isolation() {
    use pecos_engines::noise::IntoNoiseModel;
    use pecos_engines::scheduled_frame::ScheduledIdleZ;
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for scheduled in [false, true] {
        for code in [0, 1, 999] {
            let mut builder = MonteCarloEngine::builder()
                .with_classical_engine(Box::new(engine(scheduled, code)))
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
            for workers in [1, 3] {
                let (shots, report) = mc.run_with_workers_report_seeds(7, workers).unwrap();
                for (i, shot) in shots.shots.iter().enumerate() {
                    assert_eq!(shot.data["shot"], Data::U64(i as u64));
                    assert_eq!(shot.data.contains_key("after"), i % 2 == 0);
                    assert_eq!(
                        shot.program_exit,
                        (i % 2 == 1).then(|| ProgramExit {
                            code,
                            message: "done".into()
                        })
                    );
                }
                assert_eq!(
                    shots,
                    mc.run_with_workers_from_seed_report(&report).unwrap()
                );
            }
        }
    }
}

#[test]
fn exit_metadata_clears_on_reset_and_clone() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = engine(false, 1);
    engine.set_num_qubits_hint(1);
    engine.set_program_shot(1);
    assert!(finish(&mut engine).unwrap().program_exit.is_some());
    assert!(engine.get_results().unwrap().program_exit.is_some());
    let clone = engine.clone();
    assert_eq!(clone.get_results().unwrap().program_exit, None);
    engine.reset_all().unwrap();
    assert_eq!(engine.get_results().unwrap().program_exit, None);
    engine.set_program_shot(0);
    assert_eq!(finish(&mut engine).unwrap().program_exit, None);
}

#[test]
fn panic_still_fails_and_reset_recovers() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for code in [-1, 1000, 1001, i32::MIN, i32::MAX] {
        let mut engine = engine(false, code);
        engine.set_num_qubits_hint(1);
        engine.set_program_shot(1);
        assert!(
            finish(&mut engine)
                .unwrap_err()
                .to_string()
                .contains("done")
        );
        assert!(engine.get_results().is_err());
        engine.reset_all().unwrap();
        engine.set_program_shot(0);
        assert_eq!(finish(&mut engine).unwrap().program_exit, None);
    }
}

fn finish(engine: &mut QisEngine) -> Result<Shot, PecosError> {
    let mut quantum = StateVecEngine::new(1);
    let mut stage = engine.start(())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(Instant::now() < deadline);
        match stage {
            EngineStage::Complete(shot) => return Ok(shot),
            EngineStage::NeedsProcessing(commands) => {
                stage = engine.continue_processing(quantum.process(commands)?)?;
            }
        }
    }
}
