use pecos_core::errors::PecosError;
use pecos_engines::ByteMessage;
use pecos_engines::monte_carlo::engine::{
    ExternalClassicalEngine, MonteCarloEngine, SeedReport, WorkerSeedRecord,
};
use pecos_engines::shot_results::{Data, ShotVec};
use pecos_random::nth_derived_seed;

// Measuring |+> produces seed-sensitive outcomes; depolarizing noise also
// exercises propagation of shot seeds into the noise model.
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

#[test]
fn depolarizing_constructor_applies_requested_noise_probability() {
    // Preparing and measuring |0> produces ones only when noise is applied.
    let circuit = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .mz(&[0])
        .build();

    for (probability, expect_ones) in [(0.0, false), (0.5, true)] {
        let controller = ExternalClassicalEngine::new_with_circuit(circuit.clone());
        let mut engine =
            MonteCarloEngine::new_with_depolarizing_noise(Box::new(controller), probability);
        engine.set_seed(42);
        let results = engine.run_with_workers(128, 1).unwrap();
        assert_eq!(results.len(), 128);

        let mut ones = 0;
        for shot in results.shots {
            match shot.data.get("result") {
                Some(Data::U32(0)) => {}
                Some(Data::U32(1)) => ones += 1,
                result => panic!("expected a measured bit, got {result:?}"),
            }
        }
        assert_eq!(
            ones > 0,
            expect_ones,
            "p={probability}: measured {ones} ones"
        );
    }
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
                first_shot: 0,
            },
            WorkerSeedRecord {
                worker_idx: 1,
                shots: 5,
                first_shot: 5,
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
            { "worker_idx": 0, "shots": 5, "first_shot": 0 },
            { "worker_idx": 1, "shots": 5, "first_shot": 5 }
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
    assert_eq!(report.workers[0].first_shot, 0);
    assert_eq!(report.workers[1].worker_idx, 1);
    assert_eq!(report.workers[1].shots, 5);
    assert_eq!(report.workers[1].first_shot, 5);
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
    assert_eq!(report.workers[0].first_shot, 0);
    assert_eq!(report.workers[1].first_shot, 5);
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
    assert_ne!(report_a.base_seed, report_c.base_seed);
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

    // The recorded base seed drives every shot during replay.
    let (_, other_report) = make_test_monte_carlo_engine(43)
        .run_with_workers_report_seeds(128, 2)
        .unwrap();
    let mut altered_report = report;
    altered_report.base_seed = other_report.base_seed;
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
fn replay_uses_base_seed_and_root_seed_is_only_provenance() {
    let mut original = make_test_monte_carlo_engine(42);
    original.run_with_workers(128, 2).unwrap();
    let (shots, mut report) = original.run_with_workers_report_seeds(128, 2).unwrap();

    // Only root_seed is provenance, even after earlier runs advanced the RNG.
    report.root_seed = 1;
    let actual = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_eq!(actual, shots);
    report.base_seed = 2;
    let altered = make_test_monte_carlo_engine(999)
        .run_with_workers_from_seed_report(&report)
        .unwrap();
    assert_ne!(altered, shots);
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
fn run_worker_shots(base_seed: u64, first_shot: usize, shots: usize) -> ShotVec {
    let mut engine = make_test_monte_carlo_engine(999).hybrid_engine_template;
    let mut results = Vec::new();
    for i in 0..shots {
        engine.set_seed(nth_derived_seed(base_seed, (first_shot + i) as u64));
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

    report.workers[0].first_shot = 40;
    report.workers[1].first_shot = 12;
    let mut expected = run_worker_shots(report.base_seed, 12, 3);
    expected
        .shots
        .extend(run_worker_shots(report.base_seed, 40, 7).shots);
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
    assert_eq!(worker.first_shot, 64);
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
    assert_eq!(actual, run_worker_shots(report.base_seed, 5, 10));
}

#[test]
fn replay_rejects_invalid_reports_without_panicking() {
    type InvalidCase = (&'static str, fn(&mut SeedReport));
    let invalid_cases: [InvalidCase; 11] = [
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
                first_shot: 10,
            });
        }),
        ("shot total mismatch", |r| r.workers[0].shots = 7),
        ("duplicate worker index", |r| r.workers[1].worker_idx = 0),
        ("shot total overflow", |r| r.workers[0].shots = usize::MAX),
        ("shot range overflow", |r| {
            r.workers[0].first_shot = usize::MAX;
        }),
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

#[test]
fn fresh_reports_record_prefix_sums_including_empty_workers() {
    for (shots, workers) in [(10, 3), (2, 4)] {
        let (_, report) = make_test_monte_carlo_engine(42)
            .run_with_workers_report_seeds(shots, workers)
            .unwrap();
        let mut first_shot = 0;
        for (index, record) in report.workers.iter().enumerate() {
            assert_eq!(record.worker_idx, index);
            assert_eq!(record.first_shot, first_shot);
            first_shot += record.shots;
        }
        assert_eq!(first_shot, shots);
    }
}
