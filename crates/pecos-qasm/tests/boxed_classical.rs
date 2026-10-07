use pecos_core::errors::PecosError;
use pecos_engines::{ClassicalControlEngine, Engine};
use pecos_qasm::QASMEngine;
use std::str::FromStr;

#[test]
fn boxed_qasm_rejects_batch_before_trailing_assignment() {
    let mut engine: Box<dyn ClassicalControlEngine> = Box::new(QASMEngine::from_str(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; measure q[0] -> c[0]; c = 1;",
    ).unwrap());
    let result = engine.process(());
    let Err(PecosError::Processing(message)) = result else {
        panic!("expected quantum processing error before trailing assignment, got {result:?}");
    };
    assert_eq!(
        message,
        "Box<dyn ClassicalControlEngine>::process(()) cannot execute quantum commands; use start()/continue_processing() with a quantum engine, or the simulation builder."
    );
}

#[test]
fn boxed_qasm_classical_assignment_runs_after_reset() {
    let mut engine: Box<dyn ClassicalControlEngine> =
        Box::new(QASMEngine::from_str("OPENQASM 2.0; creg c[1]; c = 1;").unwrap());
    for _ in 0..2 {
        Engine::reset(&mut engine).unwrap();
        assert_eq!(engine.process(()).unwrap().data["c"].as_u32(), Some(1));
    }
}
