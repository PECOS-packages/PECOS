use pecos_core::errors::PecosError;
use pecos_engines::ByteMessage;
use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
use pecos_engines::monte_carlo::{MonteCarloEngine, WorkerScope};
use std::sync::atomic::{AtomicUsize, Ordering};

fn engine() -> MonteCarloEngine {
    let circuit = ByteMessage::quantum_operations_builder()
        .pz(&[0])
        .h(&[0])
        .mz(&[0])
        .build();
    let mut engine = MonteCarloEngine::new_with_depolarizing_noise(
        Box::new(ExternalClassicalEngine::new_with_circuit(circuit)),
        0.1,
    )
    .unwrap();
    engine.set_seed(42);
    engine
}

struct CountingScope(AtomicUsize);
impl WorkerScope for CountingScope {
    fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        work();
        Ok(())
    }
}

#[test]
fn scope_preserves_seeded_results_replay_and_clones() {
    let mut plain = engine();
    let mut scoped = plain.clone();
    let scope = CountingScope(AtomicUsize::new(0));
    let (expected, report) = plain.run_with_workers_report_seeds(41, 4).unwrap();
    let (actual, actual_report) = scoped
        .run_with_workers_report_seeds_in_scope(41, 4, &scope)
        .unwrap();
    assert_eq!(expected.shots, actual.shots);
    assert_eq!(
        serde_json::to_value(&report).unwrap(),
        serde_json::to_value(&actual_report).unwrap()
    );
    assert_eq!(scope.0.load(Ordering::SeqCst), 4);
    let replay = scoped
        .run_with_workers_from_seed_report_in_scope(&report, &scope)
        .unwrap();
    assert_eq!(expected.shots, replay.shots);
    assert_eq!(scope.0.load(Ordering::SeqCst), 8);
    assert_eq!(
        plain.run_with_workers(17, 2).unwrap().shots,
        scoped
            .run_with_workers_in_scope(17, 2, &scope)
            .unwrap()
            .shots
    );
}

#[test]
fn empty_workers_do_not_enter_context_and_zero_shots_remain_invalid() {
    let scope = CountingScope(AtomicUsize::new(0));
    let mut engine = engine();
    assert_eq!(
        engine
            .run_with_workers_in_scope(1, 4, &scope)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(scope.0.load(Ordering::SeqCst), 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.run_with_workers_in_scope(0, 4, &scope)
        }))
        .is_err()
    );
    assert_eq!(scope.0.load(Ordering::SeqCst), 1);
}

struct Skip;
impl WorkerScope for Skip {
    fn execute(&self, _work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        Ok(())
    }
}
struct Move;
impl WorkerScope for Move {
    fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        std::thread::scope(|scope| scope.spawn(work).join().unwrap());
        Ok(())
    }
}
struct Panic(AtomicUsize);
impl WorkerScope for Panic {
    fn execute(&self, _work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        panic!("scope panic")
    }
}
#[test]
fn invalid_scopes_fail_and_panics_join_all_jobs_without_poisoning_template() {
    let mut engine = engine();
    assert!(
        engine
            .run_with_workers_in_scope(4, 1, &Skip)
            .unwrap_err()
            .to_string()
            .contains("did not execute")
    );
    assert!(
        engine
            .run_with_workers_in_scope(4, 1, &Move)
            .unwrap_err()
            .to_string()
            .contains("another thread")
    );
    let scope = Panic(AtomicUsize::new(0));
    assert!(
        engine
            .run_with_workers_in_scope(4, 4, &scope)
            .unwrap_err()
            .to_string()
            .contains("scope panic")
    );
    assert_eq!(scope.0.load(Ordering::SeqCst), 4);
    assert_eq!(engine.run(1).unwrap().len(), 1);
}
