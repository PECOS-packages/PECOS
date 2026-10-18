//! Job-wide program identity and RNG through a real program worker.
use super::*;
use pecos_engines::{ClassicalControlEngineBuilder, MonteCarloEngine, StateVecEngine};
use pecos_random::{PCG32_INIT_STATE, PCGRandom};

const PROGRAM: &str = r#"
@shot_label = private constant [4 x i8] c"shot"
@abi_label = private constant [3 x i8] c"abi"
@rng_label = private constant [3 x i8] c"rng"
declare i64 @get_current_shot()
declare { i32, i64 } @selene_get_current_shot(ptr)
declare void @print_uint_selene(ptr, i64, i64)
declare void @random_seed(i64)
declare i32 @random_int()
declare void @__quantum__qis__x__body(i64)
declare i32 @__quantum__qis__m__body(i64, i64)
declare i1 @___read_future_bool(i64)
define i64 @qmain(i64 %arg) #0 {
    %shot = call i64 @get_current_shot()
    call void @print_uint_selene(ptr @shot_label, i64 4, i64 %shot)
    call void @random_seed(i64 %shot)
    %random = call i32 @random_int()
    %wide = zext i32 %random to i64
    call void @print_uint_selene(ptr @rng_label, i64 3, i64 %wide)
    ; Suspend on a measurement: the program index must survive feedback.
    call void @__quantum__qis__x__body(i64 0)
    %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
    %read = call i1 @___read_future_bool(i64 0)
    %abi = call { i32, i64 } @selene_get_current_shot(ptr null)
    %id = extractvalue { i32, i64 } %abi, 1
    call void @print_uint_selene(ptr @abi_label, i64 3, i64 %id)
    ret i64 0
}
attributes #0 = { "EntryPoint" }
"#;

fn engine(scheduled: bool) -> QisEngine {
    crate::qis_engine()
        .interface(crate::helios_interface_builder())
        .program(pecos_programs::Qis::from_string(PROGRAM))
        .runtime(crate::selene_simple_runtime().unwrap())
        .scheduled_batches(scheduled)
        .build()
        .unwrap()
}
fn assert_shot(shot: &Shot, index: u64) {
    assert_eq!(shot.data["shot"], Data::U64(index));
    assert_eq!(shot.data["abi"], Data::U64(index));
    let mut rng = PCGRandom::from_seed_and_stream(PCG32_INIT_STATE, index);
    assert_eq!(shot.data["rng"], Data::U64(u64::from(rng.next_u32())));
}

#[test]
fn program_shot_and_rng_are_global_across_workers_replay_and_offsets() {
    use pecos_engines::noise::IntoNoiseModel;
    use pecos_engines::scheduled_frame::ScheduledIdleZ;
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    for scheduled in [false, true] {
        for workers in [1, 2, 3] {
            let mut builder = MonteCarloEngine::builder()
                .with_classical_engine(Box::new(engine(scheduled)))
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
            let (shots, mut report) = mc.run_with_workers_report_seeds(7, workers).unwrap();
            for (i, shot) in shots.shots.iter().enumerate() {
                assert_shot(shot, u64::try_from(i).unwrap());
            }
            assert_eq!(
                shots,
                mc.run_with_workers_from_seed_report(&report).unwrap()
            );
            // A new run is a new job: program indices restart, though simulation
            // RNG seeds advance. Explicit program seeds still give the same data.
            assert_eq!(shots, mc.run_with_workers(7, workers).unwrap());
            for worker in &mut report.workers {
                worker.first_shot += 17;
            }
            let offset = mc.run_with_workers_from_seed_report(&report).unwrap();
            for (i, shot) in offset.shots.iter().enumerate() {
                assert_shot(shot, 17 + u64::try_from(i).unwrap());
            }
        }
    }
}

fn finish(engine: &mut QisEngine) -> Shot {
    let mut quantum = StateVecEngine::new(1);
    let mut stage = engine.start(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(Instant::now() < deadline);
        match stage {
            EngineStage::Complete(shot) => return shot,
            EngineStage::NeedsProcessing(commands) => {
                stage = engine
                    .continue_processing(quantum.process(commands).unwrap())
                    .unwrap();
            }
        }
    }
}

#[test]
fn direct_program_identity_reset_override_and_clone_lifecycle() {
    let _lock = crate::test_env::ENV_MUTEX.lock().unwrap();
    let mut engine = engine(false);
    engine.set_num_qubits_hint(1);
    assert_shot(&finish(&mut engine), 0);
    engine.reset_all().unwrap();
    assert_shot(&finish(&mut engine), 1);
    engine.set_program_shot(u64::MAX);
    assert_shot(&finish(&mut engine), u64::MAX);
    // The host override applies to only one invocation.
    assert_shot(&finish(&mut engine), 3);
    engine.set_program_shot(99);
    let mut clone = engine.clone();
    assert_shot(&finish(&mut clone), 0);
    engine.reset_all().unwrap();
    assert_shot(&finish(&mut engine), 4);
}
