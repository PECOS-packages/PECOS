//! Regression tests for state-preserving allocation across command batches.

use pecos_core::RngManageable;
use pecos_engines::quantum::{StateVectorEngine, StateVectorSimulator};
use pecos_engines::{ByteMessage, Engine};
use pecos_simulators::{StateVec, StateVecAoS, StateVecSoA};

fn check_growth<S: StateVectorSimulator + 'static>()
where
    <S as RngManageable>::Rng: Clone,
{
    for mut engine in [
        StateVectorEngine::<S>::new(1),
        StateVectorEngine::<S>::with_seed(1, 42),
    ] {
        let mut first = ByteMessage::quantum_operations_builder();
        first.x(&[0]);
        engine.process(first.build()).unwrap();

        let mut second = ByteMessage::quantum_operations_builder();
        second.z(&[2]);
        second.mz(&[0]);
        assert_eq!(
            engine.process(second.build()).unwrap().outcomes().unwrap(),
            vec![1]
        );

        // Intermediate and newly allocated qubits start in |0>.
        let mut third = ByteMessage::quantum_operations_builder();
        third.mz(&[1, 2]);
        assert_eq!(
            engine.process(third.build()).unwrap().outcomes().unwrap(),
            vec![0, 0]
        );
    }
}

#[test]
fn sparse_soa_growth_preserves_state() {
    check_growth::<StateVec>();
}

#[test]
fn aos_growth_preserves_state() {
    check_growth::<StateVecAoS>();
}

#[test]
fn dense_soa_growth_preserves_state() {
    check_growth::<StateVecSoA>();
}
