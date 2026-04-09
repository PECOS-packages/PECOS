//! Integration test: `ForeignSimulator` with pecos-neo's `CircuitRunner`.
//!
//! Proves that a foreign simulator (C-ABI vtable) can be used with
//! pecos-neo's typed command system -- no `ByteMessage` serialization needed.

use pecos_core::QubitId;
use pecos_foreign::{ForeignMeasurementResult, ForeignSimulator, ForeignSimulatorVTable};
use pecos_neo::prelude::*;
use pecos_simulators::QuantumSimulator;

// -- Reuse the toy simulator from foreign_simulator_test, but with set_seed --

struct ToySimState {
    bits: Vec<bool>,
}

unsafe extern "C" fn toy_sz(_handle: *mut (), _qubits: *const usize, _num_qubits: usize) {}

unsafe extern "C" fn toy_h(handle: *mut (), qubits: *const usize, num_qubits: usize) {
    let state = unsafe { &mut *handle.cast::<ToySimState>() };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
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

unsafe extern "C" fn toy_set_seed(_handle: *mut (), _seed: u64) {
    // Toy sim is deterministic, no RNG to seed
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
        set_seed: Some(toy_set_seed),
        destroy: toy_destroy,
    };

    unsafe { ForeignSimulator::new(handle, vtable) }.expect("vtable version should match")
}

#[test]
fn test_foreign_sim_with_circuit_runner() {
    // Build a circuit using pecos-neo's typed commands
    let mut commands = CommandQueue::new();
    commands.push(GateCommand::h(QubitId(0)));
    commands.push(GateCommand::cx(QubitId(0), QubitId(1)));
    commands.push(GateCommand::mz(QubitId(0)));
    commands.push(GateCommand::mz(QubitId(1)));

    // Create the foreign simulator
    let mut sim = make_toy_sim(2);

    // Create a pecos-neo CircuitRunner generic over ForeignSimulator
    let mut runner = CircuitRunner::<ForeignSimulator>::new();

    // Run the circuit
    let outcomes = runner.apply_circuit(&mut sim, &commands).unwrap();

    // In our toy: H flips qubit 0, CX propagates to qubit 1
    assert_eq!(outcomes.len(), 2);
    let q0 = outcomes.get(QubitId(0)).unwrap();
    let q1 = outcomes.get(QubitId(1)).unwrap();
    assert!(q0.outcome, "qubit 0 should be |1> after H");
    assert!(
        q1.outcome,
        "qubit 1 should be |1> after CX with control=|1>"
    );
}

#[test]
fn test_foreign_sim_reset_and_rerun() {
    let mut commands = CommandQueue::new();
    commands.push(GateCommand::h(QubitId(0)));
    commands.push(GateCommand::mz(QubitId(0)));

    let mut sim = make_toy_sim(1);
    let mut runner = CircuitRunner::<ForeignSimulator>::new();

    // First shot
    let outcomes = runner.apply_circuit(&mut sim, &commands).unwrap();
    assert!(outcomes.get(QubitId(0)).unwrap().outcome);

    // Reset and second shot
    sim.reset();
    let outcomes = runner.apply_circuit(&mut sim, &commands).unwrap();
    assert!(outcomes.get(QubitId(0)).unwrap().outcome);
}

#[test]
fn test_foreign_sim_derived_gates_via_neo() {
    // X gate is NOT in our vtable -- pecos-neo decomposes it into H + Z + H
    // (where Z = SZ + SZ). In our toy, H flips and SZ is no-op.
    // So X = H(SZ(SZ(H(|0>)))) = H(H(|0>)) = |0> (double flip).
    let mut commands = CommandQueue::new();
    commands.push(GateCommand::x(QubitId(0)));
    commands.push(GateCommand::mz(QubitId(0)));

    let mut sim = make_toy_sim(1);
    let mut runner = CircuitRunner::<ForeignSimulator>::new();

    let outcomes = runner.apply_circuit(&mut sim, &commands).unwrap();
    assert!(
        !outcomes.get(QubitId(0)).unwrap().outcome,
        "X on toy sim decomposes to H(SZ(SZ(H))) = double flip = |0>"
    );
}
