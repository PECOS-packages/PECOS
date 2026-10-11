use pecos_engines::ByteMessage;
use pecos_engines::monte_carlo::engine::{ExternalClassicalEngine, MonteCarloEngine};
use pecos_engines::quantum::StabVecEngine;

fn engine(extra_gates: bool, stab_vec: bool) -> MonteCarloEngine {
    let mut circuit = ByteMessage::quantum_operations_builder();
    circuit.pz(&[0, 1]).h(&[0]).mz(&[0]);
    if extra_gates {
        circuit.h(&[1]).x(&[1]).mz(&[1]);
    }
    let mut builder = MonteCarloEngine::builder()
        .with_num_qubits(2)
        .with_classical_engine(Box::new(ExternalClassicalEngine::new_with_circuit(
            circuit.build(),
        )))
        .with_depolarizing_noise(0.1)
        .with_seed(42);
    if stab_vec {
        builder = builder.with_quantum_engine(Box::new(StabVecEngine::new(2)));
    }
    builder.build()
}

#[test]
fn per_shot_results_do_not_depend_on_worker_count() {
    let expected = engine(true, false).run_with_workers(128, 1).unwrap();
    for workers in [2, 3] {
        let actual = engine(true, false).run_with_workers(128, workers).unwrap();
        assert_eq!(actual, expected, "workers={workers}");
    }
}

fn assert_earlier_shots_do_not_change_draws(stab_vec: bool) {
    let a = engine(false, stab_vec).run_with_workers(128, 1).unwrap();
    let b = engine(true, stab_vec).run_with_workers(128, 1).unwrap();
    for (index, (a, b)) in a.shots.iter().zip(&b.shots).enumerate() {
        assert_eq!(a.data["result"], b.data["result"], "shot={index}");
    }
}

#[test]
fn shot_draws_do_not_depend_on_earlier_shots_state_vec() {
    assert_earlier_shots_do_not_change_draws(false);
}

#[test]
fn shot_draws_do_not_depend_on_earlier_shots_stab_vec() {
    assert_earlier_shots_do_not_change_draws(true);
}

fn assert_prefix_runs_match(workers: usize) {
    let short = engine(true, false).run_with_workers(10, workers).unwrap();
    let long = engine(true, false).run_with_workers(24, workers).unwrap();
    assert_eq!(short.shots, long.shots[..10], "workers={workers}");
}

#[test]
fn prefix_runs_match_one_worker() {
    assert_prefix_runs_match(1);
}

#[test]
fn prefix_runs_match_two_workers() {
    assert_prefix_runs_match(2);
}

#[test]
fn prefix_runs_match_three_workers() {
    assert_prefix_runs_match(3);
}
