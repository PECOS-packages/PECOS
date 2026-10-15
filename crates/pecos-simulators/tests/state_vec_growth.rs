//! Growth preserves the quantum state, deferred gates, and random stream.

use num_complex::Complex64;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::{PecosRng, RngExt};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, StateVec, StateVecAoS, StateVecSoA,
    StateVectorSimulator,
};

fn check_growth<S>(grow: fn(&mut S, usize))
where
    S: StateVectorSimulator + ArbitraryRotationGateable + RngManageable<Rng = PecosRng> + Clone,
{
    let mut sim = S::with_seed(2, 42);
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.rz(Angle64::from_radians(0.7), &[QubitId(0)]);
    sim.sx(&[QubitId(1)]);
    sim.sz(&[QubitId(0)]);

    // Read a clone so that growth itself must handle pending gates/frames.
    let mut snapshot = sim.clone();
    let old: Vec<_> = (0..4).map(|i| snapshot.get_amplitude(i)).collect();
    // Advance beyond the seeded RNG's initial state before testing preservation.
    let _ = sim.rng_mut().random::<u64>();
    let mut expected_rng = sim.rng().clone();
    grow(&mut sim, 4);
    grow(&mut sim, 4);
    grow(&mut sim, 1);
    assert_eq!(sim.num_qubits(), 4);
    assert_eq!(sim.rng_mut().random::<u64>(), expected_rng.random::<u64>());

    // All implementations map qubit k to bit k: new high bits must be zero.
    let expected: Vec<_> = (0..16)
        .map(|i| {
            if i >> 2 == 0 {
                old[i & 0b11]
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    let actual: Vec<_> = (0..16).map(|i| sim.get_amplitude(i)).collect();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual - expected).norm() < 1e-12);
    }
    assert!((actual.iter().map(Complex64::norm_sqr).sum::<f64>() - 1.0).abs() < 1e-12);

    // Exercise the new qubits, cross-boundary gates, and a second growth.
    sim.x(&[QubitId(2)]);
    sim.cx(&[(QubitId(0), QubitId(3))]);
    grow(&mut sim, 5);
    for i in 0..32 {
        let expected = if i & 0b10100 == 0b00100 && ((i >> 3) & 1) == (i & 1) {
            old[i & 0b11]
        } else {
            Complex64::new(0.0, 0.0)
        };
        assert!((sim.get_amplitude(i) - expected).norm() < 1e-12);
    }
    sim.reset();
    assert_eq!(sim.num_qubits(), 5);
    assert_eq!(sim.get_amplitude(0), Complex64::new(1.0, 0.0));
    for i in 1..32 {
        assert_eq!(sim.get_amplitude(i), Complex64::new(0.0, 0.0));
    }

    let mut empty = S::with_seed(0, 42);
    grow(&mut empty, 0);
    grow(&mut empty, 2);
    empty.x(&[QubitId(1)]);
    assert_eq!(empty.get_amplitude(2), Complex64::new(1.0, 0.0));
}

#[test]
fn sparse_soa_growth_preserves_amplitudes() {
    check_growth(StateVec::ensure_qubit_count);
}

#[test]
fn aos_growth_preserves_amplitudes() {
    check_growth(StateVecAoS::ensure_qubit_count);
}

#[test]
fn dense_soa_growth_preserves_amplitudes() {
    check_growth(StateVecSoA::ensure_qubit_count);
}

#[test]
fn dense_soa_growth_resizes_reused_unitary_scratch() {
    let mut sim = StateVecSoA::new(2);
    sim.h(&[QubitId(0)]);
    let identity = std::array::from_fn(|row| {
        std::array::from_fn(|col| Complex64::new(f64::from(row == col), 0.0))
    });
    sim.two_qubit_unitary(0, 1, identity);
    let before = sim.state();
    sim.ensure_qubit_count(4);
    sim.x(&[QubitId(3)]);
    sim.two_qubit_unitary(0, 3, identity);
    for i in 0..16 {
        let expected = if i >> 2 == 2 {
            before[i & 0b11]
        } else {
            Complex64::new(0.0, 0.0)
        };
        assert!((sim.get_amplitude(i) - expected).norm() < 1e-12);
    }
}
