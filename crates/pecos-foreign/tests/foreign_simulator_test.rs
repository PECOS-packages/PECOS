//! Integration test for `ForeignSimulator`.
//!
//! Implements a toy computational-basis simulator as C-ABI functions, wraps it
//! into a `ForeignSimulator`, and verifies it works through `CliffordGateable`.
//!
//! The toy simulator only tracks classical bit flips -- it's not a real quantum
//! simulator, but it exercises the full vtable -> trait pipeline.

use pecos_core::QubitId;
use pecos_foreign::{ForeignMeasurementResult, ForeignSimulator, ForeignSimulatorVTable};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};

// -- "Foreign" simulator state --

struct ToySimState {
    bits: Vec<bool>, // true = |1>
}

// -- C-ABI callbacks --

unsafe extern "C" fn toy_sz(_handle: *mut (), _qubits: *const usize, _num_qubits: usize) {
    // S gate is a phase -- doesn't change computational basis
}

unsafe extern "C" fn toy_h(handle: *mut (), qubits: *const usize, num_qubits: usize) {
    let state = unsafe { &mut *handle.cast::<ToySimState>() };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    // Toy: H flips the bit (not physically correct, but tests the interface)
    for &q in qs {
        state.bits[q] = !state.bits[q];
    }
}

unsafe extern "C" fn toy_cx(handle: *mut (), pairs: *const usize, num_pairs: usize) {
    let state = unsafe { &mut *handle.cast::<ToySimState>() };
    let flat = unsafe { std::slice::from_raw_parts(pairs, num_pairs * 2) };
    for chunk in flat.chunks_exact(2) {
        let (control, target) = (chunk[0], chunk[1]);
        if state.bits[control] {
            state.bits[target] = !state.bits[target];
        }
    }
}

unsafe extern "C" fn toy_mz(
    handle: *mut (),
    qubits: *const usize,
    num_qubits: usize,
    results_out: *mut ForeignMeasurementResult,
) {
    let state = unsafe { &*handle.cast::<ToySimState>() };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    let out = unsafe { std::slice::from_raw_parts_mut(results_out, num_qubits) };
    for (i, &q) in qs.iter().enumerate() {
        out[i] = ForeignMeasurementResult {
            outcome: u8::from(state.bits[q]),
            is_deterministic: 1,
        };
    }
}

unsafe extern "C" fn toy_reset(handle: *mut ()) {
    let state = unsafe { &mut *handle.cast::<ToySimState>() };
    state.bits.fill(false);
}

unsafe extern "C" fn toy_destroy(handle: *mut ()) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle.cast::<ToySimState>());
        }
    }
}

fn make_toy_sim(num_qubits: usize) -> ForeignSimulator {
    let state = Box::new(ToySimState {
        bits: vec![false; num_qubits],
    });
    let handle = Box::into_raw(state).cast::<()>();

    let vtable = ForeignSimulatorVTable {
        version: pecos_foreign::version::SIMULATOR_VTABLE_VERSION,
        sz: toy_sz,
        h: toy_h,
        cx: toy_cx,
        mz: toy_mz,
        rx: None,
        rz: None,
        rzz: None,
        reset: toy_reset,
        set_seed: None,
        destroy: toy_destroy,
    };

    unsafe { ForeignSimulator::new(handle, vtable) }.expect("vtable version should match")
}

#[test]
fn test_foreign_simulator_h_and_measure() {
    let mut sim = make_toy_sim(3);

    // H on qubit 0 (toy: flips to |1>)
    sim.h(&[QubitId(0)]);

    let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
    assert!(results[0].outcome, "qubit 0 should be |1> after H");
    assert!(!results[1].outcome, "qubit 1 should be |0>");
    assert!(!results[2].outcome, "qubit 2 should be |0>");
}

#[test]
fn test_foreign_simulator_cx() {
    let mut sim = make_toy_sim(2);

    // Set qubit 0 to |1> via H, then CNOT(0, 1)
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);

    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    assert!(results[0].outcome, "control should be |1>");
    assert!(results[1].outcome, "target should be |1> (flipped by CNOT)");
}

#[test]
fn test_foreign_simulator_reset() {
    let mut sim = make_toy_sim(2);

    sim.h(&[QubitId(0), QubitId(1)]);
    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    assert!(results[0].outcome && results[1].outcome);

    sim.reset();
    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    assert!(!results[0].outcome, "qubit 0 should be |0> after reset");
    assert!(!results[1].outcome, "qubit 1 should be |0> after reset");
}

#[test]
fn test_foreign_simulator_derived_gates() {
    // The whole point: X, Z, Y etc. are decomposed into H, SZ, CX automatically.
    // Since our toy H just flips bits and SZ is a no-op, X = H Z H = H SZ SZ H
    // should flip twice via H, with SZ as no-ops. Net: no change.
    // But the important thing is that the derived methods compile and dispatch correctly.
    let mut sim = make_toy_sim(2);

    // X gate uses the default impl: h -> z -> h (where z = sz -> sz)
    // In our toy: H flips, SZ no-ops, H flips back. Net: no change.
    sim.x(&[QubitId(0)]);
    let results = sim.mz(&[QubitId(0)]);
    assert!(
        !results[0].outcome,
        "X on toy sim should be H(SZ(SZ(H(|0>))))"
    );

    // Z gate = SZ SZ (no-ops in toy) -- should not change state
    sim.h(&[QubitId(1)]); // flip to |1>
    sim.z(&[QubitId(1)]); // no-op (SZ SZ)
    let results = sim.mz(&[QubitId(1)]);
    assert!(
        results[0].outcome,
        "Z should not change computational basis"
    );
}

#[test]
fn test_foreign_simulator_no_rotations() {
    let sim = make_toy_sim(1);
    assert!(
        !sim.supports_rotations(),
        "toy sim should not support rotations"
    );
}

#[test]
#[should_panic(expected = "does not support rotation gates")]
fn test_foreign_simulator_rotation_panic() {
    let mut sim = make_toy_sim(1);
    // Should panic because rx is None
    sim.rx(std::f64::consts::FRAC_PI_2.into(), &[QubitId(0)]);
}

#[test]
fn test_foreign_simulator_batch_gates() {
    let mut sim = make_toy_sim(4);

    // Batch H on multiple qubits at once
    sim.h(&[QubitId(0), QubitId(1), QubitId(2)]);

    let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    assert!(results[0].outcome);
    assert!(results[1].outcome);
    assert!(results[2].outcome);
    assert!(!results[3].outcome, "qubit 3 was not H'd");

    // Batch CX: two pairs at once
    sim.reset();
    sim.h(&[QubitId(0), QubitId(2)]); // set controls to |1>
    sim.cx(&[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))]);

    let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    assert!(results[0].outcome && results[1].outcome);
    assert!(results[2].outcome && results[3].outcome);
}

#[test]
fn test_foreign_simulator_version_mismatch() {
    let state = Box::new(ToySimState {
        bits: vec![false; 1],
    });
    let handle = Box::into_raw(state).cast::<()>();

    let vtable = ForeignSimulatorVTable {
        version: 9999, // wrong version
        sz: toy_sz,
        h: toy_h,
        cx: toy_cx,
        mz: toy_mz,
        rx: None,
        rz: None,
        rzz: None,
        reset: toy_reset,
        set_seed: None,
        destroy: toy_destroy,
    };

    // Should return None on version mismatch
    let result = unsafe { ForeignSimulator::new(handle, vtable) };
    assert!(result.is_none(), "wrong version should return None");

    unsafe {
        let _ = Box::from_raw(handle.cast::<ToySimState>());
    }
}
