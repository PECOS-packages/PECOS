//! Conformance test suite for foreign simulators.
//!
//! Runs a battery of quantum circuits against a foreign simulator and checks
//! that the results are correct. Any correct Clifford simulator must pass all tests.
//!
//! # Usage from C
//!
//! ```c
//! PecosConformanceReport report;
//! int ok = pecos_run_conformance_tests(handle, &vtable, 2, &report);
//! if (!ok) {
//!     printf("FAILED: %s\n", report.first_failure);
//! }
//! ```
//!
//! # Usage from Rust
//!
//! ```rust,ignore
//! let results = run_conformance_tests(&mut foreign_sim);
//! assert!(results.all_passed());
//! ```

use crate::simulator::{ForeignSimulator, ForeignSimulatorVTable};
use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, QuantumSimulator};
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::os::raw::c_char;

/// Result of the conformance test suite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConformanceReport {
    /// Total number of tests run.
    pub tests_run: u32,
    /// Number of tests passed.
    pub tests_passed: u32,
    /// Null-terminated name of the first failing test, or null if all passed.
    /// Points to a static string -- do not free.
    pub first_failure: *const c_char,
}

impl ConformanceReport {
    fn new() -> Self {
        Self {
            tests_run: 0,
            tests_passed: 0,
            first_failure: std::ptr::null(),
        }
    }

    fn record_pass(&mut self) {
        self.tests_run += 1;
        self.tests_passed += 1;
    }

    fn record_fail(&mut self, name: &'static CStr) {
        self.tests_run += 1;
        if self.first_failure.is_null() {
            self.first_failure = name.as_ptr();
        }
    }

    /// Whether all tests passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.tests_run == self.tests_passed
    }
}

/// Run conformance tests against a `ForeignSimulator` (Rust API).
pub fn run_conformance_tests(sim: &mut ForeignSimulator) -> ConformanceReport {
    let mut report = ConformanceReport::new();

    test_x_gate_determinism(sim, &mut report);
    test_h_self_inverse(sim, &mut report);
    test_bell_state_correlation(sim, &mut report);
    test_z_preserves_basis(sim, &mut report);
    test_reset(sim, &mut report);
    test_batch_h(sim, &mut report);

    report
}

// --- Individual tests ---

fn test_x_gate_determinism(sim: &mut ForeignSimulator, report: &mut ConformanceReport) {
    // X|0> must always give |1>
    sim.reset();
    sim.x(&[QubitId(0)]);
    let results = sim.mz(&[QubitId(0)]);
    if results.len() == 1 && results[0].outcome {
        report.record_pass();
    } else {
        report.record_fail(c"x_gate_determinism: X|0> should measure |1>");
    }
}

fn test_h_self_inverse(sim: &mut ForeignSimulator, report: &mut ConformanceReport) {
    // HH|0> must give |0> (deterministic)
    sim.reset();
    sim.h(&[QubitId(0)]);
    sim.h(&[QubitId(0)]);
    let results = sim.mz(&[QubitId(0)]);
    if results.len() == 1 && !results[0].outcome && results[0].is_deterministic {
        report.record_pass();
    } else {
        report.record_fail(c"h_self_inverse: HH|0> should deterministically measure |0>");
    }
}

fn test_bell_state_correlation(sim: &mut ForeignSimulator, report: &mut ConformanceReport) {
    // H(q0) then CX(q0, q1) -- both qubits must agree on every shot.
    // Run 20 trials to be sure.
    let mut all_correlated = true;
    for _ in 0..20 {
        sim.reset();
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        if results.len() != 2 || results[0].outcome != results[1].outcome {
            all_correlated = false;
            break;
        }
    }
    if all_correlated {
        report.record_pass();
    } else {
        report.record_fail(c"bell_state_correlation: qubits must always agree in Bell state");
    }
}

fn test_z_preserves_basis(sim: &mut ForeignSimulator, report: &mut ConformanceReport) {
    // Z|0> = |0> (Z is diagonal, doesn't change computational basis)
    sim.reset();
    sim.z(&[QubitId(0)]);
    let r0 = sim.mz(&[QubitId(0)]);

    // Z(X|0>) = Z|1> = -|1>, which still measures |1>
    sim.reset();
    sim.x(&[QubitId(0)]);
    sim.z(&[QubitId(0)]);
    let r1 = sim.mz(&[QubitId(0)]);

    if r0.len() == 1 && !r0[0].outcome && r1.len() == 1 && r1[0].outcome {
        report.record_pass();
    } else {
        report.record_fail(c"z_preserves_basis: Z should not change computational basis");
    }
}

fn test_reset(sim: &mut ForeignSimulator, report: &mut ConformanceReport) {
    // After X then reset, qubit should measure |0>
    sim.reset();
    sim.x(&[QubitId(0)]);
    sim.reset();
    let results = sim.mz(&[QubitId(0)]);
    if results.len() == 1 && !results[0].outcome {
        report.record_pass();
    } else {
        report.record_fail(c"reset: after reset, qubit should measure |0>");
    }
}

fn test_batch_h(sim: &mut ForeignSimulator, report: &mut ConformanceReport) {
    // HH on qubit 0 and H on qubit 1:
    // qubit 0 should deterministically be |0> (HH = I)
    // qubit 1 is random -- we just check it returns a result
    sim.reset();
    sim.h(&[QubitId(0), QubitId(1)]);
    sim.h(&[QubitId(0)]); // second H only on qubit 0
    let results = sim.mz(&[QubitId(0), QubitId(1)]);
    if results.len() == 2 && !results[0].outcome && results[0].is_deterministic {
        report.record_pass();
    } else {
        report.record_fail(c"batch_h: HH on qubit 0 should deterministically give |0>");
    }
}

// ============================================================================
// C-ABI entry point
// ============================================================================

/// Run conformance tests against a foreign simulator.
///
/// Creates a `ForeignSimulator` from the given handle + vtable, runs all tests,
/// and writes results to `report_out`.
///
/// Returns 1 if all tests passed, 0 if any failed.
///
/// # Arguments
/// - `handle`: opaque simulator handle
/// - `vtable`: pointer to a `ForeignSimulatorVTable`
/// - `num_qubits`: number of qubits the simulator was created with (must be >= 2)
/// - `report_out`: pointer to a `ConformanceReport` to fill
///
/// # Safety
/// All pointers must be valid. The simulator must have been created with at least 2 qubits.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_run_conformance_tests(
    handle: *mut (),
    vtable: *const ForeignSimulatorVTable,
    _num_qubits: usize,
    report_out: *mut ConformanceReport,
) -> i32 {
    let vtable_copy = unsafe { *vtable };

    let Some(sim) = (unsafe { ForeignSimulator::new(handle, vtable_copy) }) else {
        // Version mismatch
        unsafe { *report_out = ConformanceReport::new() };
        return 0;
    };
    // ManuallyDrop prevents Drop from calling vtable.destroy -- caller owns the handle.
    let mut sim = ManuallyDrop::new(sim);
    let report = run_conformance_tests(&mut sim);

    unsafe { *report_out = report };

    i32::from(report.all_passed())
}
