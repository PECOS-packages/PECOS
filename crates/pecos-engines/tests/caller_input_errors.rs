use pecos_engines::quantum::{StateVectorEngine, StateVectorSimulator};
use pecos_simulators::{StateVec, StateVecAoS, StateVecSoA};
#[test]
fn engine_growth_returns_error() {
    let mut engine = StateVectorEngine::<StateVec>::new(1);
    assert!(engine.ensure_qubit_count(usize::MAX).is_err());
}
#[test]
fn trait_statevec_growth_returns_error() {
    assert!(StateVectorSimulator::ensure_qubit_count(&mut StateVec::new(1), usize::MAX).is_err());
}

#[test]
fn trait_statevecaos_growth_returns_error() {
    assert!(
        StateVectorSimulator::ensure_qubit_count(&mut StateVecAoS::new(1), usize::MAX).is_err()
    );
}

#[test]
fn trait_statevecsoa_growth_returns_error() {
    assert!(
        StateVectorSimulator::ensure_qubit_count(&mut StateVecSoA::new(1), usize::MAX).is_err()
    );
}
