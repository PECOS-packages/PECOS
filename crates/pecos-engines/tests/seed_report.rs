use pecos_core::errors::PecosError;
use pecos_engines::ByteMessage;
use pecos_engines::monte_carlo::engine::{
    ExternalClassicalEngine, MonteCarloEngine, SeedReport, WorkerSeedRecord,
};
use pecos_engines::shot_results::ShotVec;

// Measuring |+> produces seed-sensitive outcomes; depolarizing noise also
// exercises propagation of worker seeds into the noise model.
fn make_test_monte_carlo_engine(seed: u64) -> MonteCarloEngine {
    let circuit = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .h(&[0])
        .mz(&[0])
        .build();
    let mut engine = MonteCarloEngine::new_with_depolarizing_noise(
        Box::new(ExternalClassicalEngine::new_with_circuit(circuit)),
        0.1,
    );
    engine.set_seed(seed);
    engine.default_workers = 2;
    engine
}

fn example_report() -> SeedReport {
    SeedReport {
        root_seed: 42,
        base_seed: 123_456_789,
        num_shots: 10,
        num_workers: 2,
        workers: vec![
            WorkerSeedRecord {
                worker_idx: 0,
                shots: 5,
                seed: 111,
            },
            WorkerSeedRecord {
                worker_idx: 1,
                shots: 5,
                seed: 222,
            },
        ],
    }
}

#[test]
fn seed_report_from_json_file_reads_valid_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seed_report.json");
    std::fs::write(&path, serde_json::to_string(&example_report()).unwrap()).unwrap();

    let report = SeedReport::from_json_file(&path).unwrap();

    assert_eq!(report.root_seed, 42);
    assert_eq!(report.num_workers, 2);
    assert_eq!(report.workers.len(), 2);
}

#[test]
fn seed_report_from_json_file_returns_error_for_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let err = SeedReport::from_json_file(dir.path().join("missing.json")).unwrap_err();
    assert!(matches!(err, PecosError::Input(_)));
    assert!(err.to_string().contains("Failed to read seed report JSON"));
}

#[test]
fn seed_report_from_json_str_parses_valid_report() {
    let json = r#"
    {
        "root_seed": 42,
        "base_seed": 123456789,
        "num_shots": 10,
        "num_workers": 2,
        "workers": [
            { "worker_idx": 0, "shots": 5, "seed": 6 },
            { "worker_idx": 1, "shots": 5, "seed": 435 }
        ]
    }
    "#;
    let report = SeedReport::from_json_str(json).unwrap();
    assert_eq!(report.root_seed, 42);
    assert_eq!(report.base_seed, 123_456_789);
    assert_eq!(report.num_shots, 10);
    assert_eq!(report.num_workers, 2);
    assert_eq!(report.workers.len(), 2);
    assert_eq!(report.workers[0].worker_idx, 0);
    assert_eq!(report.workers[0].shots, 5);
    assert_eq!(report.workers[0].seed, 6);
    assert_eq!(report.workers[1].worker_idx, 1);
    assert_eq!(report.workers[1].shots, 5);
    assert_eq!(report.workers[1].seed, 435);
}

#[test]
fn run_with_seed_report_returns_expected_worker_metadata() {
    let mut engine = make_test_monte_carlo_engine(42);
    let (shots, report) = engine.run_with_workers_report_seeds(10, 2).unwrap();

    assert_eq!(shots.len(), 10);
    assert_eq!(report.root_seed, 42);
    assert_eq!(report.num_shots, 10);
    assert_eq!(report.num_workers, 2);
    assert_eq!(report.workers.len(), 2);
    assert_eq!(report.workers.iter().map(|w| w.shots).sum::<usize>(), 10);
    assert_eq!(report.workers[0].worker_idx, 0);
    assert_eq!(report.workers[1].worker_idx, 1);
}

#[test]
fn run_with_seed_report_is_deterministic_for_same_seed_workers_and_shots() {
    let (shots_a, report_a) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let (shots_b, report_b) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let (shots_c, report_c) = make_test_monte_carlo_engine(43)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();

    assert_eq!(shots_a, shots_b);
    assert_ne!(shots_a, shots_c, "fixture must expose changes in RNG seeds");
    assert_eq!(
        serde_json::to_value(&report_a).unwrap(),
        serde_json::to_value(&report_b).unwrap()
    );
    let seeds_a: Vec<_> = report_a.workers.iter().map(|w| w.seed).collect();
    let seeds_c: Vec<_> = report_c.workers.iter().map(|w| w.seed).collect();
    assert_ne!(seeds_a, seeds_c);
}

#[test]
fn rerun_from_seed_report_reproduces_original_results() {
    let (original_results, report) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let replay_engine = make_test_monte_carlo_engine(999_999);
    let replayed_results = replay_engine
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(replayed_results, original_results);

    // Alter only the recorded worker seeds. Root/base metadata cannot replace
    // those seeds, and dropping worker seeding must make this assertion fail.
    let (_, other_report) = make_test_monte_carlo_engine(43)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let mut altered_report = report;
    for (record, other) in altered_report.workers.iter_mut().zip(other_report.workers) {
        record.seed = other.seed;
    }
    let altered_results = replay_engine
        .run_with_workers_from_seed_report(&altered_report)
        .unwrap();
    assert_ne!(altered_results, original_results);
}

#[test]
fn rerun_from_seed_report_loaded_from_json_reproduces_original_results() {
    let (original_results, report) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let loaded_report = SeedReport::from_json_str(&json).unwrap();
    let replay_engine = make_test_monte_carlo_engine(999_999);
    let replayed_results = replay_engine
        .run_with_workers_from_seed_report(&loaded_report)
        .unwrap();
    assert_eq!(replayed_results, original_results);
}

#[test]
fn consecutive_runs_on_one_engine_advance_seeds_and_shots() {
    let mut engine = make_test_monte_carlo_engine(42);
    let (first, first_report) = engine.run_with_workers_report_seeds(128, 2).unwrap();
    let (second, second_report) = engine.run_with_workers_report_seeds(128, 2).unwrap();

    assert_ne!(first_report.base_seed, second_report.base_seed);
    assert_ne!(first_report.workers[0].seed, second_report.workers[0].seed);
    assert_ne!(first, second);
    assert_eq!(second_report.root_seed, 42);

    // Explicitly resetting the seed still restarts the original sequence.
    engine.set_seed(42);
    let (restarted, restarted_report) = engine.run_with_workers_report_seeds(128, 2).unwrap();
    assert_eq!(restarted, first);
    assert_eq!(restarted_report.base_seed, first_report.base_seed);
}

#[test]
fn consecutive_runs_through_public_wrappers_advance_shots() {
    let mut engine = make_test_monte_carlo_engine(42);
    let first = engine.run(128).unwrap();
    let second = engine.run(128).unwrap();
    assert_ne!(first, second, "run must advance the RNG between batches");

    engine.set_seed(42);
    let first = engine.run_with_workers(128, 2).unwrap();
    let second = engine.run_with_workers(128, 2).unwrap();
    assert_ne!(first, second, "run_with_workers must advance the RNG");
}

#[test]
fn replay_does_not_mutate_caller_engine_stream() {
    let (archived_shots, archived_report) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let mut control = make_test_monte_carlo_engine(999);
    let mut replay_engine = make_test_monte_carlo_engine(999);

    assert_eq!(
        control.run_with_workers(128, 2).unwrap(),
        replay_engine.run_with_workers(128, 2).unwrap()
    );
    assert_eq!(
        replay_engine
            .run_with_workers_from_seed_report(&archived_report)
            .unwrap(),
        archived_shots
    );
    assert_eq!(replay_engine.seed, 999);

    let (expected_shots, expected_report) = control.run_with_workers_report_seeds(128, 2).unwrap();
    let (actual_shots, actual_report) =
        replay_engine.run_with_workers_report_seeds(128, 2).unwrap();
    assert_eq!(actual_report.base_seed, expected_report.base_seed);
    assert_eq!(actual_report.root_seed, expected_report.root_seed);
    assert_eq!(actual_shots, expected_shots);
}

#[test]
fn replay_uses_worker_seeds_after_multiple_recorded_runs() {
    let mut original = make_test_monte_carlo_engine(42);
    original.run_with_workers(128, 2).unwrap();
    let (shots, mut report) = original.run_with_workers_report_seeds(128, 2).unwrap();

    // These fields describe provenance; the individual worker seeds drive replay.
    report.root_seed = 1;
    report.base_seed = 2;
    let actual = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(actual, shots);
}

#[test]
fn replay_accepts_reordered_workers_and_preserves_shot_order() {
    let (expected, mut report) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 3)
        .unwrap();
    report.workers.reverse();
    let actual = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(actual, expected);
}

// Run the hybrid engine directly to obtain an independent expectation for a
// custom allocation; this bypasses Monte Carlo replay and its worker splitting.
fn run_worker_shots(seed: u64, shots: usize) -> ShotVec {
    let mut engine = make_test_monte_carlo_engine(999).hybrid_engine_template;
    engine.set_seed(seed);
    let mut results = Vec::new();
    for _ in 0..shots {
        engine.reset().unwrap();
        results.push(engine.run_shot().unwrap());
    }
    ShotVec { shots: results }
}

#[test]
fn replay_honors_custom_worker_shot_counts_and_indices() {
    let mut report = example_report();
    report.workers[0].worker_idx = 9;
    report.workers[0].shots = 7;
    report.workers[1].worker_idx = 2;
    report.workers[1].shots = 3;

    let mut expected = run_worker_shots(222, 3);
    expected.shots.extend(run_worker_shots(111, 7).shots);
    let actual = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn replay_can_reproduce_a_subset_of_one_recorded_worker() {
    let (original, mut report) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 4)
        .unwrap();
    let mut worker = report.workers[2].clone();
    worker.shots = 17;
    report.num_workers = 1;
    report.num_shots = worker.shots;
    report.workers = vec![worker];

    let replay = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(replay.shots, original.shots[64..81]);
}

#[test]
fn replay_accepts_zero_shot_worker_records() {
    let mut report = example_report();
    report.workers[0].shots = 0;
    report.workers[1].shots = 10;
    let actual = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(actual, run_worker_shots(222, 10));
}

#[test]
fn replay_rejects_invalid_reports_without_panicking() {
    type InvalidCase = (&'static str, fn(&mut SeedReport));
    let invalid_cases: [InvalidCase; 10] = [
        ("zero shots", |r| {
            r.num_shots = 0;
            r.workers.iter_mut().for_each(|w| w.shots = 0);
        }),
        ("zero workers", |r| r.num_workers = 0),
        ("empty records", |r| r.workers.clear()),
        ("missing record", |r| {
            r.workers.pop();
        }),
        ("extra record", |r| {
            r.workers.push(WorkerSeedRecord {
                worker_idx: 2,
                shots: 0,
                seed: 333,
            });
        }),
        ("shot total mismatch", |r| r.workers[0].shots = 7),
        ("duplicate worker index", |r| r.workers[1].worker_idx = 0),
        ("shot total overflow", |r| r.workers[0].shots = usize::MAX),
        ("oversized shot metadata", |r| r.num_shots = usize::MAX),
        ("oversized worker metadata", |r| r.num_workers = usize::MAX),
    ];

    for (name, invalidate) in invalid_cases {
        let mut report = example_report();
        invalidate(&mut report);
        let json = serde_json::to_string(&report).unwrap();
        let parsed = SeedReport::from_json_str(&json);
        assert!(
            matches!(parsed, Err(PecosError::Input(_))),
            "{name}: {parsed:?}"
        );
        let result = make_test_monte_carlo_engine(999).run_with_workers_from_seed_report(&report);
        assert!(
            matches!(result, Err(PecosError::Input(_))),
            "{name}: {result:?}"
        );
    }
}

#[test]
fn seed_report_from_json_str_rejects_malformed_input() {
    for json in ["not json", "{}", r#"{"root_seed":"not a number"}"#] {
        assert!(matches!(
            SeedReport::from_json_str(json),
            Err(PecosError::Input(_))
        ));
    }
}

#[test]
fn seed_report_to_json_file_round_trips_metadata_and_replay() {
    let (expected, report) = make_test_monte_carlo_engine(42)
        .run_with_workers_report_seeds(128, 3)
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chosen-report.json");
    report.to_json_file(&path).unwrap();
    let loaded = SeedReport::from_json_file(&path).unwrap();
    assert_eq!(
        serde_json::to_value(&loaded).unwrap(),
        serde_json::to_value(&report).unwrap()
    );
    let actual = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&loaded)
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn seed_report_to_json_file_returns_input_error_for_bad_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing-parent").join("seed_report.json");
    let err = example_report().to_json_file(path).unwrap_err();
    assert!(matches!(err, PecosError::Input(_)));
    assert!(err.to_string().contains("Failed to write seed report JSON"));
}
