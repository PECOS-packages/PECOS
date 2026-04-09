//! Test the conformance suite itself by running it against a real simulator
//! wrapped as a `ForeignSimulator`.
//!
//! We wrap PECOS's `SparseStab` in C-ABI callbacks and run the conformance tests.
//! Since `SparseStab` is a correct Clifford simulator, all tests must pass.

use pecos_core::QubitId;
use pecos_foreign::conformance::run_conformance_tests;
use pecos_foreign::{ForeignMeasurementResult, ForeignSimulator, ForeignSimulatorVTable};
use pecos_simulators::{CliffordGateable, SparseStab};

use std::cell::UnsafeCell;

/// Wrap `SparseStab` behind C-ABI function pointers.
/// This is a "real" foreign simulator -- same interface a Go/C author would use.
struct SimHolder {
    sim: UnsafeCell<SparseStab>,
}

impl SimHolder {
    fn new(n: usize) -> Box<Self> {
        Box::new(Self {
            sim: UnsafeCell::new(SparseStab::new(n)),
        })
    }
}

unsafe extern "C" fn real_sz(handle: *mut (), qubits: *const usize, num_qubits: usize) {
    let holder = unsafe { &*handle.cast::<SimHolder>() };
    let sim = unsafe { &mut *holder.sim.get() };
    let qs: Vec<QubitId> = unsafe { std::slice::from_raw_parts(qubits, num_qubits) }
        .iter()
        .map(|&q| QubitId(q))
        .collect();
    sim.sz(&qs);
}

unsafe extern "C" fn real_h(handle: *mut (), qubits: *const usize, num_qubits: usize) {
    let holder = unsafe { &*handle.cast::<SimHolder>() };
    let sim = unsafe { &mut *holder.sim.get() };
    let qs: Vec<QubitId> = unsafe { std::slice::from_raw_parts(qubits, num_qubits) }
        .iter()
        .map(|&q| QubitId(q))
        .collect();
    sim.h(&qs);
}

unsafe extern "C" fn real_cx(handle: *mut (), pairs: *const usize, num_pairs: usize) {
    let holder = unsafe { &*handle.cast::<SimHolder>() };
    let sim = unsafe { &mut *holder.sim.get() };
    let flat = unsafe { std::slice::from_raw_parts(pairs, num_pairs * 2) };
    let pair_vec: Vec<(QubitId, QubitId)> = flat
        .chunks_exact(2)
        .map(|p| (QubitId(p[0]), QubitId(p[1])))
        .collect();
    sim.cx(&pair_vec);
}

unsafe extern "C" fn real_mz(
    handle: *mut (),
    qubits: *const usize,
    num_qubits: usize,
    results_out: *mut ForeignMeasurementResult,
) {
    let holder = unsafe { &*handle.cast::<SimHolder>() };
    let sim = unsafe { &mut *holder.sim.get() };
    let qs: Vec<QubitId> = unsafe { std::slice::from_raw_parts(qubits, num_qubits) }
        .iter()
        .map(|&q| QubitId(q))
        .collect();
    let results = sim.mz(&qs);
    let out = unsafe { std::slice::from_raw_parts_mut(results_out, num_qubits) };
    for (i, r) in results.iter().enumerate() {
        out[i] = ForeignMeasurementResult {
            outcome: u8::from(r.outcome),
            is_deterministic: u8::from(r.is_deterministic),
        };
    }
}

unsafe extern "C" fn real_reset(handle: *mut ()) {
    let holder = unsafe { &*handle.cast::<SimHolder>() };
    let sim = unsafe { &mut *holder.sim.get() };
    sim.reset();
}

unsafe extern "C" fn real_set_seed(_handle: *mut (), _seed: u64) {}

unsafe extern "C" fn real_destroy(handle: *mut ()) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle.cast::<SimHolder>());
        }
    }
}

fn make_real_sim(n: usize) -> ForeignSimulator {
    let holder = SimHolder::new(n);
    let handle = Box::into_raw(holder).cast::<()>();

    let vtable = ForeignSimulatorVTable {
        version: pecos_foreign::version::SIMULATOR_VTABLE_VERSION,
        sz: real_sz,
        h: real_h,
        cx: real_cx,
        mz: real_mz,
        rx: None,
        rz: None,
        rzz: None,
        reset: real_reset,
        set_seed: Some(real_set_seed),
        destroy: real_destroy,
    };

    unsafe { ForeignSimulator::new(handle, vtable) }.expect("vtable version should match")
}

#[test]
fn test_conformance_passes_with_real_simulator() {
    let mut sim = make_real_sim(4);
    let report = run_conformance_tests(&mut sim);

    assert!(
        report.all_passed(),
        "SparseStab wrapped as ForeignSimulator should pass all conformance tests. \
         Passed {}/{}, first failure: {:?}",
        report.tests_passed,
        report.tests_run,
        if report.first_failure.is_null() {
            "none"
        } else {
            unsafe { std::ffi::CStr::from_ptr(report.first_failure) }
                .to_str()
                .unwrap_or("?")
        }
    );

    assert_eq!(report.tests_run, 6, "expected 6 conformance tests");
}

// -- Broken simulator: H is a no-op (should fail conformance) --

unsafe extern "C" fn broken_h(_handle: *mut (), _qubits: *const usize, _num_qubits: usize) {
    // Deliberately broken: H does nothing
}

fn make_broken_sim(n: usize) -> ForeignSimulator {
    let holder = SimHolder::new(n);
    let handle = Box::into_raw(holder).cast::<()>();

    let vtable = ForeignSimulatorVTable {
        version: pecos_foreign::version::SIMULATOR_VTABLE_VERSION,
        sz: real_sz,
        h: broken_h, // broken!
        cx: real_cx,
        mz: real_mz,
        rx: None,
        rz: None,
        rzz: None,
        reset: real_reset,
        set_seed: Some(real_set_seed),
        destroy: real_destroy,
    };

    unsafe { ForeignSimulator::new(handle, vtable) }.expect("vtable version should match")
}

#[test]
fn test_conformance_fails_with_broken_simulator() {
    let mut sim = make_broken_sim(4);
    let report = run_conformance_tests(&mut sim);

    assert!(
        !report.all_passed(),
        "broken simulator (no-op H) should fail conformance tests"
    );
    assert!(
        report.tests_passed < report.tests_run,
        "at least one test should fail"
    );
    assert!(
        !report.first_failure.is_null(),
        "first_failure should name the failing test"
    );
}
