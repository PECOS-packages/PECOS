#![cfg(unix)]
use pecos_core::Angle64;
use pecos_engines::{
    Engine, Gate, StateVecEngine,
    noise::IntoNoiseModel,
    quantum_system::QuantumSystem,
    runtime_frame::ShotContext,
    scheduled_events::{
        ScheduledEventBatch, ScheduledEventOp, ScheduledResult, encode_event_batches,
    },
    scheduled_frame::ScheduledIdleZ,
};
use pecos_foreign::scheduled_adapter::{MAX_CONFIG_BYTES, NativeScheduledAdapter};
use std::{path::PathBuf, process::Command};

struct Fixture {
    path: PathBuf,
    monitor: libloading::Library,
    _dir: tempfile::TempDir,
}
impl Fixture {
    fn new(bad: u32) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(if cfg!(target_os = "macos") {
            "adapter.dylib"
        } else {
            "adapter.so"
        });
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let result = Command::new("cc")
            .args([
                "-std=c11", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
            ])
            .arg(format!("-DBAD_DESCRIPTOR={bad}"))
            .arg("-I")
            .arg(root.join("include"))
            .arg(root.join("tests/fixtures/scheduled_adapter.c"))
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let monitor = unsafe { libloading::Library::new(&path) }.unwrap();
        Self {
            path,
            monitor,
            _dir: dir,
        }
    }
    fn factory(&self, mode: u8) -> NativeScheduledAdapter {
        unsafe { NativeScheduledAdapter::load(&self.path, &[mode]) }.unwrap()
    }
    fn count(&self, i: u32) -> u64 {
        unsafe {
            self.monitor
                .get::<unsafe extern "C" fn(u32) -> u64>(b"probe_count\0")
                .unwrap()(i)
        }
    }
}
fn context() -> ShotContext {
    ShotContext {
        run: 11,
        worker: 3,
        shot: 5,
    }
}
fn batch(flip: bool) -> ScheduledEventBatch {
    ScheduledEventBatch {
        runtime_shot_id: 77,
        batch_index: 0,
        start_nanos: 0,
        duration_nanos: 0,
        operations: vec![
            ScheduledEventOp::Gate(Box::new(Gate::pz(&[0]))),
            ScheduledEventOp::Custom {
                tag: 4242,
                payload: vec![u8::from(flip)],
            },
            ScheduledEventOp::Gate(Box::new(Gate::mz(&[0]))),
        ],
        measurements: vec![ScheduledResult {
            operation_index: 2,
            runtime_result: 900,
            program_result: 17,
        }],
    }
}
fn system(factory: NativeScheduledAdapter) -> QuantumSystem {
    let mut system = QuantumSystem::new(
        factory
            .with_noise(ScheduledIdleZ::new(2, 0.0, 0.0, 0.0).unwrap())
            .into_noise_model()
            .unwrap(),
        Box::new(StateVecEngine::new(2)),
    );
    system.set_seed(42);
    system.begin_shot(context()).unwrap();
    system
}
#[test]
fn c_plugin_runs_through_owner_and_preserves_result_namespaces() {
    let f = Fixture::new(0);
    for mode in [0, 13, 14] {
        for flip in [false, true] {
            let mut s = system(f.factory(mode));
            let result = s
                .process(encode_event_batches(&[batch(flip)]).unwrap())
                .unwrap()
                .outcomes()
                .unwrap();
            assert_eq!(result, vec![u32::from(flip)]);
        }
    }
    assert_eq!(f.count(0), 6);
    assert_eq!(f.count(1), 6);
}
#[test]
fn rejects_incompatible_descriptors_before_create() {
    for bad in 1..=9 {
        let f = Fixture::new(bad);
        assert!(unsafe { NativeScheduledAdapter::load(&f.path, &[0]) }.is_err());
        assert_eq!(f.count(0), 0);
    }
    let f = Fixture::new(0);
    assert!(
        unsafe { NativeScheduledAdapter::load(&f.path, &vec![0; MAX_CONFIG_BYTES + 1]) }.is_err()
    );
}
#[test]
fn creation_failure_cleans_nonnull_handles_once() {
    let f = Fixture::new(0);
    for mode in [4, 5, 6] {
        assert!(f.factory(mode).create(context()).is_err());
    }
    assert_eq!(f.count(0), 1);
    assert_eq!(f.count(1), 1);
}
#[test]
fn ignored_callback_errors_and_translation_errors_poison_until_reset() {
    for mode in [1, 2, 3, 7, 8, 9, 10, 11, 12] {
        let f = Fixture::new(0);
        let mut s = system(f.factory(mode));
        let input = encode_event_batches(&[batch(true)]).unwrap();
        assert!(s.process(input.clone()).is_err(), "mode {mode}");
        let calls = (f.count(2), f.count(3));
        assert!(s.process(input.clone()).is_err());
        assert_eq!(calls, (f.count(2), f.count(3)));
        s.reset().unwrap();
        assert_eq!(f.count(1), 1);
        s.begin_shot(context()).unwrap();
        assert!(s.process(input).is_err());
        assert_eq!(f.count(0), 2);
        drop(s);
        assert_eq!(f.count(1), 2);
    }
}
#[test]
fn whole_message_validates_before_translation_and_recovers_with_fresh_session() {
    let f = Fixture::new(0);
    let mut s = system(f.factory(0));
    let a = batch(true);
    let mut b = batch(false);
    b.batch_index = 1;
    b.measurements[0].runtime_result = 901;
    b.operations[1] = ScheduledEventOp::Custom {
        tag: 9999,
        payload: vec![],
    };
    assert!(
        s.process(encode_event_batches(&[a.clone(), b]).unwrap())
            .is_err()
    );
    assert_eq!(f.count(2), 2);
    assert_eq!(f.count(3), 0);
    s.reset().unwrap();
    s.begin_shot(context()).unwrap();
    assert_eq!(
        s.process(encode_event_batches(&[a]).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![1]
    );
}
#[test]
fn multiple_inputs_and_clone_rules_remain_owned_by_quantum_system() {
    let f = Fixture::new(0);
    let mut s = system(f.factory(0));
    let mut a = batch(false);
    a.operations.truncate(2);
    a.measurements.clear();
    assert_eq!(
        s.process(encode_event_batches(&[a]).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        Vec::<u32>::new()
    );
    let mut cloned = s.clone();
    assert!(
        cloned
            .process(encode_event_batches(&[batch(false)]).unwrap())
            .is_err()
    );
    let mut b = batch(true);
    b.batch_index = 1;
    b.operations.remove(0);
    b.measurements[0].operation_index = 1;
    assert_eq!(
        s.process(encode_event_batches(&[b]).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![1]
    );
    assert_eq!(f.count(0), 1);
    cloned.reset().unwrap();
    cloned.begin_shot(context()).unwrap();
    assert_eq!(
        cloned
            .process(encode_event_batches(&[batch(false)]).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![0]
    );
    assert_eq!(f.count(0), 2);
}
#[test]
fn independent_sessions_can_move_between_threads() {
    let f = Fixture::new(0);
    let factory = f.factory(0);
    let jobs: Vec<_> = (0..4)
        .map(|_| {
            let factory = factory.clone();
            let mut s = system(factory);
            std::thread::spawn(move || {
                s.process(encode_event_batches(&[batch(true)]).unwrap())
                    .unwrap()
                    .outcomes()
                    .unwrap()
            })
        })
        .collect();
    for j in jobs {
        assert_eq!(j.join().unwrap(), vec![1]);
    }
    assert_eq!(f.count(0), 4);
    assert_eq!(f.count(1), 4);
}
#[test]
fn direct_adapter_errors_latch_and_session_retains_library_after_factory_drop() {
    let f = Fixture::new(0);
    let factory = f.factory(2);
    let adapter = factory.create(context()).unwrap();
    drop(factory);
    assert!(adapter.validate(&batch(false)).is_err());
    let n = f.count(2);
    assert!(adapter.validate(&batch(false)).is_err());
    assert_eq!(f.count(2), n);
    drop(adapter);
    assert_eq!(f.count(1), 1);
    // Remove every other loader reference before invoking the surviving session.
    let factory = f.factory(0);
    let adapter = factory.create(context()).unwrap();
    drop(factory);
    drop(f);
    adapter.validate(&batch(false)).unwrap();
    drop(adapter);
}
#[test]
fn all_admitted_gate_kinds_agree_with_source_copy() {
    let f = Fixture::new(0);
    let gates = vec![
        Gate::pz(&[0]),
        Gate::pz(&[1]),
        Gate::rxy1q(Angle64::new(1), Angle64::new(u64::MAX), &[0]),
        Gate::rz(Angle64::new(123), &[1]),
        Gate::rzz(Angle64::new(u64::MAX), &[(0, 1)]),
        Gate::rxyxy2q(Angle64::new(456), Angle64::new(789), &[(0, 1)]),
        Gate::mz(&[0]),
        Gate::measure_z_leaked(&[1]),
    ];
    let b = ScheduledEventBatch {
        runtime_shot_id: 77,
        batch_index: 0,
        start_nanos: 0,
        duration_nanos: 0,
        operations: gates
            .into_iter()
            .map(|g| ScheduledEventOp::Gate(Box::new(g)))
            .collect(),
        measurements: vec![
            ScheduledResult {
                operation_index: 6,
                runtime_result: 0,
                program_result: 0,
            },
            ScheduledResult {
                operation_index: 7,
                runtime_result: 1,
                program_result: 1,
            },
        ],
    };
    let input = encode_event_batches(&[b]).unwrap();
    let a = system(f.factory(0)).process(input.clone()).unwrap();
    let b = system(f.factory(13)).process(input).unwrap();
    assert_eq!(a.outcomes().unwrap(), b.outcomes().unwrap());
}

#[test]
fn crosstalk_uses_existing_noise_quantum_measurement_exchange() {
    use pecos_engines::scheduled_frame::ScheduledNoise;
    use std::collections::BTreeMap;
    let f = Fixture::new(0);
    for mode in [15, 16] {
        let profile = ScheduledNoise::new(2)
            .unwrap()
            .with_measurement_crosstalk(
                1.0,
                1.0,
                BTreeMap::from([("0->1".into(), 1.0), ("1->0".into(), 1.0)]),
            )
            .unwrap();
        let mut s = QuantumSystem::new(
            f.factory(mode)
                .with_noise(profile)
                .into_noise_model()
                .unwrap(),
            Box::new(StateVecEngine::new(2)),
        );
        s.set_seed(42);
        s.begin_shot(context()).unwrap();
        assert_eq!(
            s.process(encode_event_batches(&[batch(false)]).unwrap())
                .unwrap()
                .outcomes()
                .unwrap(),
            vec![1]
        );
    }
}
#[test]
fn already_created_session_can_move_and_drop_on_another_thread() {
    let f = Fixture::new(0);
    let adapter = f.factory(0).create(context()).unwrap();
    assert_eq!(f.count(0), 1);
    std::thread::spawn(move || {
        adapter.validate(&batch(false)).unwrap();
        drop(adapter);
    })
    .join()
    .unwrap();
    assert_eq!(f.count(1), 1);
}
#[test]
fn metadata_preserves_seeded_idle_noise_results() {
    let f = Fixture::new(0);
    for seed in 0..16 {
        let mut a = batch(false);
        a.operations[0] = ScheduledEventOp::Gate(Box::new(Gate::rxy1q(
            Angle64::QUARTER_TURN,
            Angle64::ZERO,
            &[0],
        )));
        a.operations.pop();
        a.measurements.clear();
        let mut b = batch(false);
        b.batch_index = 1;
        b.start_nanos = 1_000_000_000;
        b.operations[0] = ScheduledEventOp::Gate(Box::new(Gate::rxy1q(
            -Angle64::QUARTER_TURN,
            Angle64::ZERO,
            &[0],
        )));
        let mut plain = vec![a.clone(), b.clone()];
        for x in &mut plain {
            x.operations.remove(1);
            for m in &mut x.measurements {
                m.operation_index -= 1;
            }
        }
        let mut results = Vec::new();
        for input in [vec![a, b], plain] {
            let mut s = QuantumSystem::new(
                f.factory(0)
                    .with_noise(ScheduledIdleZ::new(2, 0.3, 0.1, 0.0).unwrap())
                    .into_noise_model()
                    .unwrap(),
                Box::new(StateVecEngine::new(2)),
            );
            s.set_seed(seed);
            s.begin_shot(context()).unwrap();
            results.push(
                s.process(encode_event_batches(&input).unwrap())
                    .unwrap()
                    .outcomes()
                    .unwrap(),
            );
        }
        assert_eq!(results[0], results[1]);
    }
}

#[test]
fn callback_failure_preserves_plugin_phase_and_status() {
    let f = Fixture::new(0);
    let adapter = f.factory(17).create(context()).unwrap();
    let error = adapter.validate(&batch(false)).unwrap_err().to_string();
    assert!(error.contains("validate status 61"), "{error}");
    assert!(error.contains("operation index out of bounds"), "{error}");
}
