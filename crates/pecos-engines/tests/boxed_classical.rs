use pecos_core::errors::PecosError;
use pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
use pecos_engines::{ByteMessage, ClassicalControlEngine, Engine};

#[test]
fn boxed_external_rejects_malformed_circuit() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(
        ExternalClassicalEngine::new_with_circuit(ByteMessage::new(&[1, 2, 3])),
    );
    let result = engine.process(());
    let Err(PecosError::Input(message)) = result else {
        panic!("expected malformed batch error, got {result:?}");
    };
    assert_eq!(message, "Message too small for batch header");
}

#[test]
fn boxed_external_rejects_quantum_circuit() {
    let mut engine: Box<dyn ClassicalControlEngine> =
        Box::new(ExternalClassicalEngine::new_with_circuit(
            ByteMessage::quantum_operations_builder().x(&[0]).build(),
        ));
    let result = engine.process(());
    let Err(PecosError::Processing(message)) = result else {
        panic!("expected quantum processing error, got {result:?}");
    };
    assert_eq!(
        message,
        "Box<dyn ClassicalControlEngine>::process(()) cannot execute quantum commands; use start()/continue_processing() with a quantum engine, or the simulation builder."
    );
}

// `ExternalClassicalEngine::start` does not clear results; callers reset between shots,
// as Monte Carlo does. Boxed `process` must not add a reset of its own, so seeded
// results survive until an explicit `reset`.
#[test]
fn boxed_external_empty_circuit_preserves_results_until_reset() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(ExternalClassicalEngine::new());
    engine
        .handle_measurements(
            ByteMessage::outcomes_builder()
                .add_outcomes(&[1, 1])
                .build(),
        )
        .unwrap();
    let shot = engine.process(()).unwrap();
    assert_eq!(
        ["result", "result_1"].map(|key| shot.data[key].as_u32()),
        [Some(1), Some(1)]
    );

    Engine::reset(&mut engine).unwrap();
    let shot = engine.process(()).unwrap();
    assert_eq!(
        ["result", "result_1"].map(|key| shot.data[key].as_u32()),
        [Some(0), Some(0)]
    );
}
