//! C-ABI bridge for foreign simulators.
//!
//! Exposes functions that Go (or any C-ABI language) calls to register a simulator
//! with PECOS and then use it.

use pecos_foreign::{ForeignMeasurementResult, ForeignSimulator, ForeignSimulatorVTable};

/// C-compatible vtable passed from Go. Must match the Go `PecosSimulatorVTable` struct layout.
#[repr(C)]
pub struct CSimulatorVTable {
    pub version: u32,
    pub sz: unsafe extern "C" fn(handle: *mut (), qubits: *const usize, num_qubits: usize),
    pub h: unsafe extern "C" fn(handle: *mut (), qubits: *const usize, num_qubits: usize),
    pub cx: unsafe extern "C" fn(handle: *mut (), pairs: *const usize, num_pairs: usize),
    pub mz: unsafe extern "C" fn(
        handle: *mut (),
        qubits: *const usize,
        num_qubits: usize,
        results_out: *mut ForeignMeasurementResult,
    ),
    pub rx: Option<
        unsafe extern "C" fn(handle: *mut (), theta: f64, qubits: *const usize, num_qubits: usize),
    >,
    pub rz: Option<
        unsafe extern "C" fn(handle: *mut (), theta: f64, qubits: *const usize, num_qubits: usize),
    >,
    pub rzz: Option<
        unsafe extern "C" fn(handle: *mut (), theta: f64, pairs: *const usize, num_pairs: usize),
    >,
    pub reset: unsafe extern "C" fn(handle: *mut ()),
    pub set_seed: Option<unsafe extern "C" fn(handle: *mut (), seed: u64)>,
    pub destroy: unsafe extern "C" fn(handle: *mut ()),
}

/// Create a `ForeignSimulator` from a Go-provided handle and vtable.
///
/// Returns an opaque pointer to a boxed `ForeignSimulator`.
/// Caller must call `pecos_foreign_simulator_free` to destroy it.
///
/// # Safety
///
/// - `handle` must be a valid simulator handle from Go's registry
/// - `vtable` must point to a valid, fully-populated `CSimulatorVTable`
/// - All non-Option function pointers must remain valid until `destroy` is called
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_simulator_create(
    handle: *mut (),
    vtable: *const CSimulatorVTable,
) -> *mut ForeignSimulator {
    let vt = unsafe { &*vtable };

    let foreign_vtable = ForeignSimulatorVTable {
        version: vt.version,
        sz: vt.sz,
        h: vt.h,
        cx: vt.cx,
        mz: vt.mz,
        rx: vt.rx,
        rz: vt.rz,
        rzz: vt.rzz,
        reset: vt.reset,
        set_seed: vt.set_seed,
        destroy: vt.destroy,
    };

    let sim = unsafe { ForeignSimulator::new(handle, foreign_vtable) };
    Box::into_raw(Box::new(sim))
}

/// Check whether a foreign simulator supports rotation gates.
///
/// # Safety
///
/// `sim` must be a valid pointer from `pecos_foreign_simulator_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_simulator_supports_rotations(
    sim: *const ForeignSimulator,
) -> bool {
    let s = unsafe { &*sim };
    s.supports_rotations()
}

/// Destroy a foreign simulator created by `pecos_foreign_simulator_create`.
///
/// # Safety
///
/// `sim` must be a valid pointer from `pecos_foreign_simulator_create`.
/// Must not be called more than once for the same pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_foreign_simulator_free(sim: *mut ForeignSimulator) {
    if !sim.is_null() {
        unsafe {
            let _ = Box::from_raw(sim);
        }
    }
}
