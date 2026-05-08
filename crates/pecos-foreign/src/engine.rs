//! Outbound engine interface for foreign languages.
//!
//! This module lets foreign languages (Go, Julia, C, etc.) create and use
//! PECOS quantum engines. The flow is:
//!
//! 1. Create an engine: [`pecos_engine_create`]
//! 2. Build a circuit with [`PecosCircuitBuilder`] functions
//! 3. Run the circuit: [`pecos_engine_process`]
//! 4. Read measurement results from the output
//! 5. Reset for next shot: [`pecos_engine_reset`]
//! 6. Destroy when done: [`pecos_engine_free`]
//!
//! This is the "PECOS → Foreign" direction: foreign code uses PECOS engines,
//! as opposed to the decoder/simulator modules where foreign code provides
//! implementations for PECOS to use.

use pecos_core::Angle64;
use pecos_core::errors::PecosError;
use pecos_engines::Engine;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use pecos_engines::byte_message::message::ByteMessage;
use pecos_engines::quantum::{
    CoinTossEngine, DensityMatrixEngine, SparseStabEngine, StabVecEngine, StabilizerEngine,
    StateVecEngine,
};
use std::ffi::CStr;
use std::os::raw::c_char;

/// Opaque engine handle. Wraps a `Box<dyn QuantumEngine>` equivalent.
///
/// We use a concrete enum rather than a trait object so we avoid needing
/// `dyn` dispatch for the common case and keep the C ABI simple.
enum EngineInner {
    StateVec(StateVecEngine),
    SparseStab(SparseStabEngine),
    Stabilizer(StabilizerEngine),
    StabVec(StabVecEngine),
    DensityMatrix(DensityMatrixEngine),
    CoinToss(CoinTossEngine),
}

impl EngineInner {
    fn process(&mut self, input: ByteMessage) -> Result<ByteMessage, PecosError> {
        match self {
            Self::StateVec(e) => e.process(input),
            Self::SparseStab(e) => e.process(input),
            Self::Stabilizer(e) => e.process(input),
            Self::StabVec(e) => e.process(input),
            Self::DensityMatrix(e) => e.process(input),
            Self::CoinToss(e) => e.process(input),
        }
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        match self {
            Self::StateVec(e) => e.reset(),
            Self::SparseStab(e) => e.reset(),
            Self::Stabilizer(e) => e.reset(),
            Self::StabVec(e) => e.reset(),
            Self::DensityMatrix(e) => e.reset(),
            Self::CoinToss(e) => e.reset(),
        }
    }
}

/// Opaque engine handle exposed to C.
pub struct PecosEngine {
    inner: EngineInner,
}

/// Opaque circuit builder handle exposed to C.
pub struct PecosCircuitBuilder {
    builder: ByteMessageBuilder,
}

// ============================================================================
// Engine lifecycle
// ============================================================================

/// Create a new PECOS quantum engine.
///
/// # Arguments
/// - `engine_type`: null-terminated C string, one of:
///   `"state_vec"`, `"sparse_stab"`, `"stabilizer"`, `"stab_vec"`,
///   `"density_matrix"`, `"coin_toss"`
/// - `num_qubits`: number of qubits
/// - `seed`: RNG seed (0 means use default/random seed)
///
/// # Returns
/// Opaque engine pointer, or null on error.
///
/// # Safety
/// `engine_type` must be a valid null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_engine_create(
    engine_type: *const c_char,
    num_qubits: usize,
    seed: u64,
) -> *mut PecosEngine {
    let c_str = unsafe { CStr::from_ptr(engine_type) };
    let Ok(type_str) = c_str.to_str() else {
        return std::ptr::null_mut();
    };

    let inner = if seed == 0 {
        match type_str {
            "state_vec" => EngineInner::StateVec(StateVecEngine::new(num_qubits)),
            "sparse_stab" => EngineInner::SparseStab(SparseStabEngine::new(num_qubits)),
            "stabilizer" => EngineInner::Stabilizer(StabilizerEngine::new(num_qubits)),
            "stab_vec" => EngineInner::StabVec(StabVecEngine::new(num_qubits)),
            "density_matrix" => EngineInner::DensityMatrix(DensityMatrixEngine::new(num_qubits)),
            "coin_toss" => EngineInner::CoinToss(CoinTossEngine::new(num_qubits)),
            _ => return std::ptr::null_mut(),
        }
    } else {
        match type_str {
            "state_vec" => EngineInner::StateVec(StateVecEngine::with_seed(num_qubits, seed)),
            "sparse_stab" => EngineInner::SparseStab(SparseStabEngine::with_seed(num_qubits, seed)),
            "stabilizer" => EngineInner::Stabilizer(StabilizerEngine::with_seed(num_qubits, seed)),
            "stab_vec" => EngineInner::StabVec(StabVecEngine::with_seed(num_qubits, seed)),
            "density_matrix" => {
                EngineInner::DensityMatrix(DensityMatrixEngine::with_seed(num_qubits, seed))
            }
            "coin_toss" => EngineInner::CoinToss(CoinTossEngine::with_seed(num_qubits, seed)),
            _ => return std::ptr::null_mut(),
        }
    };

    Box::into_raw(Box::new(PecosEngine { inner }))
}

/// Process a circuit (`ByteMessage`) through an engine.
///
/// Takes raw bytes of a built circuit, runs it, and returns measurement outcomes.
///
/// # Arguments
/// - `engine`: engine handle
/// - `input_ptr`: pointer to circuit bytes (from `pecos_circuit_build`)
/// - `input_len`: length of circuit bytes
/// - `output_ptr`: pointer to write output measurement bytes (caller frees with `pecos_free_bytes`)
/// - `output_len`: pointer to write output length
///
/// # Returns
/// 0 on success, -1 on error.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_engine_process(
    engine: *mut PecosEngine,
    input_ptr: *const u8,
    input_len: usize,
    output_ptr: *mut *mut u8,
    output_len: *mut usize,
) -> i32 {
    let eng = unsafe { &mut *engine };
    let input_bytes = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    let input = ByteMessage::new(input_bytes);

    if let Ok(output) = eng.inner.process(input) {
        let bytes = output.into_bytes();
        let len = bytes.len();
        let boxed = bytes.into_boxed_slice();
        unsafe {
            *output_len = len;
            *output_ptr = if len == 0 {
                std::ptr::null_mut()
            } else {
                Box::into_raw(boxed).cast::<u8>()
            };
        }
        0
    } else {
        unsafe {
            *output_ptr = std::ptr::null_mut();
            *output_len = 0;
        }
        -1
    }
}

/// Reset an engine for the next shot.
///
/// # Safety
/// `engine` must be a valid pointer from `pecos_engine_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_engine_reset(engine: *mut PecosEngine) -> i32 {
    let eng = unsafe { &mut *engine };
    match eng.inner.reset() {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Destroy an engine.
///
/// # Safety
/// `engine` must be a valid pointer from `pecos_engine_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_engine_free(engine: *mut PecosEngine) {
    if !engine.is_null() {
        unsafe {
            let _ = Box::from_raw(engine);
        }
    }
}

/// Free bytes returned by `pecos_engine_process`.
///
/// # Safety
/// `ptr` must be from `pecos_engine_process`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_free_bytes(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

// ============================================================================
// Circuit builder
// ============================================================================

/// Create a new circuit builder.
#[unsafe(no_mangle)]
pub extern "C" fn pecos_circuit_new() -> *mut PecosCircuitBuilder {
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    Box::into_raw(Box::new(PecosCircuitBuilder { builder }))
}

/// Add H gate(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_h(
    circuit: *mut PecosCircuitBuilder,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.h(qs);
}

/// Add X gate(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_x(
    circuit: *mut PecosCircuitBuilder,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.x(qs);
}

/// Add Z gate(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_z(
    circuit: *mut PecosCircuitBuilder,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.z(qs);
}

/// Add S (SZ) gate(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_sz(
    circuit: *mut PecosCircuitBuilder,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.sz(qs);
}

/// Add CNOT (CX) gate(s) to the circuit.
///
/// `pairs` is interleaved `[c0, t0, c1, t1, ...]`, `num_pairs` is pair count.
///
/// # Panics
///
/// Panics if `num_pairs * 2` overflows `usize`.
///
/// # Safety
/// `circuit` must be valid. `pairs` must point to `2 * num_pairs` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_cx(
    circuit: *mut PecosCircuitBuilder,
    pairs: *const usize,
    num_pairs: usize,
) {
    let c = unsafe { &mut *circuit };
    let flat_len = num_pairs.checked_mul(2).expect("num_pairs overflow");
    let flat = unsafe { std::slice::from_raw_parts(pairs, flat_len) };
    let pair_vec: Vec<(usize, usize)> = flat.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    c.builder.cx(&pair_vec);
}

/// Add RX rotation gate(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_rx(
    circuit: *mut PecosCircuitBuilder,
    theta_radians: f64,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.rx(Angle64::from_radians(theta_radians), qs);
}

/// Add RZ rotation gate(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_rz(
    circuit: *mut PecosCircuitBuilder,
    theta_radians: f64,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.rz(Angle64::from_radians(theta_radians), qs);
}

/// Add RZZ rotation gate(s) to the circuit.
///
/// # Panics
///
/// Panics if `num_pairs * 2` overflows `usize`.
///
/// # Safety
/// `circuit` must be valid. `pairs` must point to `2 * num_pairs` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_rzz(
    circuit: *mut PecosCircuitBuilder,
    theta_radians: f64,
    pairs: *const usize,
    num_pairs: usize,
) {
    let c = unsafe { &mut *circuit };
    let flat_len = num_pairs.checked_mul(2).expect("num_pairs overflow");
    let flat = unsafe { std::slice::from_raw_parts(pairs, flat_len) };
    let pair_vec: Vec<(usize, usize)> = flat.chunks_exact(2).map(|p| (p[0], p[1])).collect();
    c.builder
        .rzz(Angle64::from_radians(theta_radians), &pair_vec);
}

/// Add Z-basis measurement(s) to the circuit.
///
/// # Safety
/// `circuit` must be valid. `qubits` must point to `num_qubits` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_mz(
    circuit: *mut PecosCircuitBuilder,
    qubits: *const usize,
    num_qubits: usize,
) {
    let c = unsafe { &mut *circuit };
    let qs = unsafe { std::slice::from_raw_parts(qubits, num_qubits) };
    c.builder.mz(qs);
}

/// Build the circuit into bytes that can be passed to `pecos_engine_process`.
///
/// Returns the byte length. Caller must free with `pecos_free_bytes`.
///
/// # Safety
/// `circuit` must be valid. `output_ptr` and `output_len` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_build(
    circuit: *mut PecosCircuitBuilder,
    output_ptr: *mut *mut u8,
    output_len: *mut usize,
) {
    let c = unsafe { &mut *circuit };
    let msg = c.builder.build();
    let bytes = msg.into_bytes();
    let len = bytes.len();
    let boxed = bytes.into_boxed_slice();
    unsafe {
        *output_len = len;
        *output_ptr = if len == 0 {
            std::ptr::null_mut()
        } else {
            Box::into_raw(boxed).cast::<u8>()
        };
    }
}

/// Reset the circuit builder for reuse (preserves allocated memory).
///
/// # Safety
/// `circuit` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_reset(circuit: *mut PecosCircuitBuilder) {
    let c = unsafe { &mut *circuit };
    c.builder.reset();
    let _ = c.builder.for_quantum_operations();
}

/// Destroy a circuit builder.
///
/// # Safety
/// `circuit` must be a valid pointer from `pecos_circuit_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_circuit_free(circuit: *mut PecosCircuitBuilder) {
    if !circuit.is_null() {
        unsafe {
            let _ = Box::from_raw(circuit);
        }
    }
}

// ============================================================================
// Result parsing helpers
// ============================================================================

/// Parse measurement outcomes from engine output bytes.
///
/// Returns the number of outcomes. Caller must free `outcomes_out` with `pecos_free_outcomes`.
///
/// # Safety
/// `output_ptr` must point to valid engine output bytes of length `output_len`.
/// `outcomes_out` and `num_outcomes` must be valid pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_parse_outcomes(
    output_ptr: *const u8,
    output_len: usize,
    outcomes_out: *mut *mut u32,
    num_outcomes: *mut usize,
) -> i32 {
    let bytes = unsafe { std::slice::from_raw_parts(output_ptr, output_len) };
    let msg = ByteMessage::new(bytes);

    if let Ok(outcomes) = msg.outcomes() {
        let len = outcomes.len();
        let boxed = outcomes.into_boxed_slice();
        unsafe {
            *num_outcomes = len;
            *outcomes_out = if len == 0 {
                std::ptr::null_mut()
            } else {
                Box::into_raw(boxed).cast::<u32>()
            };
        }
        0
    } else {
        unsafe {
            *outcomes_out = std::ptr::null_mut();
            *num_outcomes = 0;
        }
        -1
    }
}

/// Free outcomes array from `pecos_parse_outcomes`.
///
/// # Safety
/// `ptr` must be from `pecos_parse_outcomes`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pecos_free_outcomes(ptr: *mut u32, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}
