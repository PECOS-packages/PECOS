use pecos_engines::monte_carlo::engine::{MonteCarloEngine,SeedReport};
use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;

/// Tests that importing a valid SeedReport from JSON file works correctly.
#[test]
fn seed_report_from_json_file_reads_valid_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("seed_report.json");

    std::fs::write(
        &path,
        r#"
        {
            "root_seed": 42,
            "base_seed": 123456789,
            "num_shots": 10,
            "num_workers": 2,
            "workers": [
                { "worker_idx": 0, "shots": 5, "seed": 111 },
                { "worker_idx": 1, "shots": 5, "seed": 222 }
            ]
        }
        "#,
    )
    .unwrap();

    let report = SeedReport::from_json_file(&path).unwrap();

    assert_eq!(report.root_seed, 42);
    assert_eq!(report.num_workers, 2);
    assert_eq!(report.workers.len(), 2);
}

/// Tests that importing a missing SeedReport JSON file fails as expected.
#[test]
fn seed_report_from_json_file_returns_error_for_missing_file() {
    let err = SeedReport::from_json_file("does-not-exist-seed-report.json").unwrap_err();

    let msg = format!("{err}");
    assert!(msg.contains("Failed to read seed report JSON"));
}

/// Verifies that a valid JSON seed report is deserialized with all worker data intact.
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
            { "worker_idx": 1, "shots": 7, "seed": 435 }
        ]
    }
    "#;
    let report = SeedReport::from_json_str(json).unwrap();
    assert_eq!(report.root_seed, 42);
    assert_eq!(report.base_seed, 123456789);
    assert_eq!(report.num_shots, 10);
    assert_eq!(report.num_workers, 2);
    assert_eq!(report.workers.len(), 2);
    assert_eq!(report.workers[0].worker_idx, 0);
    assert_eq!(report.workers[0].shots, 5);
    assert_eq!(report.workers[0].seed, 6);
    assert_eq!(report.workers[1].worker_idx, 1);
    assert_eq!(report.workers[1].shots, 7);
    assert_eq!(report.workers[1].seed, 435);
}

/// Tests run_with_workers_seed_report method.
/// Ensures the method kicks the job off correctly and creates a 
/// SeedReport with the right properties.
#[test]
fn run_with_seed_report_returns_expected_worker_metadata() {

    fn make_test_monte_carlo_engine() -> MonteCarloEngine {
        MonteCarloEngine::new_with_defaults(Box::new(
            ExternalClassicalEngine::new(),
        ))
    }

    let mut engine = make_test_monte_carlo_engine();
    engine.set_seed(42);

    let num_shots = 10;
    let num_workers = 2;

    let (_shots, report) = engine
        .run_with_workers_seed_report(num_shots, num_workers, false)
        .unwrap();

    assert_eq!(report.root_seed, 42);
    assert_eq!(report.num_shots, 10);
    assert_eq!(report.num_workers, 2);
    assert_eq!(report.workers.len(), 2);

    let total_worker_shots: usize = report.workers.iter().map(|w| w.shots).sum();
    assert_eq!(total_worker_shots, num_shots);

    assert_eq!(report.workers[0].worker_idx, 0);
    assert_eq!(report.workers[1].worker_idx, 1);
}

/// Tests seed determinism.
/// The two runs with the same seed ('a' and 'b') should agree.
/// The run with a different seed ('c') should disagree with the others.
#[test]
fn run_with_seed_report_is_deterministic_for_same_seed_workers_and_shots() {

    fn make_test_monte_carlo_engine() -> MonteCarloEngine {
        MonteCarloEngine::new_with_defaults(Box::new(
            ExternalClassicalEngine::new(),
        ))
    }

    let mut engine_a = make_test_monte_carlo_engine();
    let mut engine_b = make_test_monte_carlo_engine();
    let mut engine_c = make_test_monte_carlo_engine();

    engine_a.set_seed(42);
    engine_b.set_seed(42);
    engine_c.set_seed(43);

    let (_shots_a, report_a) = engine_a
        .run_with_workers_seed_report(10, 2, false)
        .unwrap();

    let (_shots_b, report_b) = engine_b
        .run_with_workers_seed_report(10, 2, false)
        .unwrap();

    let (_shots_c, report_c) = engine_c
        .run_with_workers_seed_report(10, 2, false)
        .unwrap();

    assert_eq!(report_a.root_seed, report_b.root_seed);
    assert_eq!(report_a.base_seed, report_b.base_seed);
    assert_eq!(report_a.workers.len(), report_b.workers.len());

    let seeds_a: Vec<u64> = report_a.workers.iter().map(|w| w.seed).collect();
    let seeds_c: Vec<u64> = report_c.workers.iter().map(|w| w.seed).collect();

    assert_ne!(seeds_a,seeds_c); 

    for (worker_a, worker_b) in report_a.workers.iter().zip(report_b.workers.iter()) {
        assert_eq!(worker_a.worker_idx, worker_b.worker_idx);
        assert_eq!(worker_a.shots, worker_b.shots);
        assert_eq!(worker_a.seed, worker_b.seed);
    }
}

/// Tests that rerunning a job from the seed report produces
/// the same results as the original job.
#[test]
fn rerun_from_seed_report_reproduces_original_results() {

    fn make_test_monte_carlo_engine() -> MonteCarloEngine {
        MonteCarloEngine::new_with_defaults(Box::new(
            ExternalClassicalEngine::new(),
        ))
    }

    let mut original_engine = make_test_monte_carlo_engine();
    original_engine.set_seed(42);

    let (original_results, report) = original_engine
        .run_with_workers_seed_report(20, 2, false)
        .unwrap();

    let mut replay_engine = make_test_monte_carlo_engine();

    let replayed_results = replay_engine
        .rerun_from_seed_report(&report)
        .unwrap();

    assert_eq!(replayed_results, original_results);
}

/// Tests that rerunning a job from a saved string seed report
/// produces the same results as the original job.
#[test]
fn rerun_from_seed_report_loaded_from_json_reproduces_original_results() {

    fn make_test_monte_carlo_engine() -> MonteCarloEngine {
        MonteCarloEngine::new_with_defaults(Box::new(
            ExternalClassicalEngine::new(),
        ))
    }

    let mut original_engine = make_test_monte_carlo_engine();
    original_engine.set_seed(42);

    let (original_results, report) = original_engine
        .run_with_workers_seed_report(20, 2, false)
        .unwrap();

    let json = serde_json::to_string(&report).unwrap();
    let loaded_report = SeedReport::from_json_str(&json).unwrap();

    let mut replay_engine = make_test_monte_carlo_engine();

    let replayed_results = replay_engine
        .rerun_from_seed_report(&loaded_report)
        .unwrap();

    assert_eq!(replayed_results, original_results);
}