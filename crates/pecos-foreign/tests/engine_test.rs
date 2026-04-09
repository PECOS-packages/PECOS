//! Integration test for the outbound engine C ABI.
//!
//! Exercises the full round trip through the C API:
//! create engine -> build circuit -> process -> parse outcomes -> reset -> destroy

use pecos_foreign::engine::*;
use std::ffi::CString;

/// Helper: create an engine, returning a non-null pointer.
fn create_engine(name: &str, num_qubits: usize, seed: u64) -> *mut PecosEngine {
    let c_name = CString::new(name).unwrap();
    let engine = unsafe { pecos_engine_create(c_name.as_ptr(), num_qubits, seed) };
    assert!(!engine.is_null(), "failed to create engine '{name}'");
    engine
}

/// Helper: build a circuit, process it, and return parsed outcomes.
///
/// # Safety
/// `engine` must be a valid engine pointer.
unsafe fn run_circuit(
    engine: *mut PecosEngine,
    build_fn: impl FnOnce(*mut PecosCircuitBuilder),
) -> Vec<u32> {
    // Build circuit
    let circuit = pecos_circuit_new();
    build_fn(circuit);

    let mut circuit_bytes: *mut u8 = std::ptr::null_mut();
    let mut circuit_len: usize = 0;
    unsafe { pecos_circuit_build(circuit, &raw mut circuit_bytes, &raw mut circuit_len) };
    unsafe { pecos_circuit_free(circuit) };

    // Process
    let mut output_bytes: *mut u8 = std::ptr::null_mut();
    let mut output_len: usize = 0;
    let rc = unsafe {
        pecos_engine_process(
            engine,
            circuit_bytes,
            circuit_len,
            &raw mut output_bytes,
            &raw mut output_len,
        )
    };
    unsafe { pecos_free_bytes(circuit_bytes, circuit_len) };
    assert_eq!(rc, 0, "engine process failed");

    // Parse outcomes
    let mut outcomes_ptr: *mut u32 = std::ptr::null_mut();
    let mut num_outcomes: usize = 0;
    let rc = unsafe {
        pecos_parse_outcomes(
            output_bytes,
            output_len,
            &raw mut outcomes_ptr,
            &raw mut num_outcomes,
        )
    };
    unsafe { pecos_free_bytes(output_bytes, output_len) };
    assert_eq!(rc, 0, "parse outcomes failed");

    if num_outcomes > 0 && !outcomes_ptr.is_null() {
        let slice = unsafe { std::slice::from_raw_parts(outcomes_ptr, num_outcomes) };
        let v = slice.to_vec();
        unsafe { pecos_free_outcomes(outcomes_ptr, num_outcomes) };
        v
    } else {
        vec![]
    }
}

#[test]
fn test_engine_create_all_types() {
    for name in &[
        "state_vec",
        "sparse_stab",
        "stabilizer",
        "clifford_rz",
        "density_matrix",
        "coin_toss",
    ] {
        let engine = create_engine(name, 2, 42);
        unsafe { pecos_engine_free(engine) };
    }
}

#[test]
fn test_engine_create_invalid() {
    let c_name = CString::new("nonexistent").unwrap();
    let engine = unsafe { pecos_engine_create(c_name.as_ptr(), 2, 0) };
    assert!(engine.is_null());
}

#[test]
fn test_bell_state_correlation() {
    // Use a seeded engine so results are reproducible
    let engine = create_engine("state_vec", 2, 12345);

    for _ in 0..20 {
        let outcomes = unsafe {
            run_circuit(engine, |c| {
                let q0: usize = 0;
                let q1: usize = 1;
                let pair = [q0, q1];
                pecos_circuit_h(c, &raw const q0, 1);
                pecos_circuit_cx(c, pair.as_ptr(), 1);
                pecos_circuit_mz(c, [q0, q1].as_ptr(), 2);
            })
        };

        assert_eq!(outcomes.len(), 2, "expected 2 measurement results");
        assert_eq!(
            outcomes[0], outcomes[1],
            "Bell state qubits must be correlated"
        );

        unsafe { pecos_engine_reset(engine) };
    }

    unsafe { pecos_engine_free(engine) };
}

#[test]
fn test_deterministic_x_gate() {
    // X|0> = |1>, should always measure 1
    let engine = create_engine("stabilizer", 1, 42);

    let outcomes = unsafe {
        run_circuit(engine, |c| {
            let q: usize = 0;
            pecos_circuit_x(c, &raw const q, 1);
            pecos_circuit_mz(c, &raw const q, 1);
        })
    };

    assert_eq!(outcomes, vec![1], "X|0> should always give |1>");
    unsafe { pecos_engine_free(engine) };
}

#[test]
fn test_circuit_reuse() {
    let engine = create_engine("sparse_stab", 2, 99);

    // First shot
    let outcomes1 = unsafe {
        run_circuit(engine, |c| {
            let q: usize = 0;
            pecos_circuit_h(c, &raw const q, 1);
            pecos_circuit_mz(c, &raw const q, 1);
        })
    };
    assert_eq!(outcomes1.len(), 1);

    // Reset and second shot
    unsafe { pecos_engine_reset(engine) };

    let outcomes2 = unsafe {
        run_circuit(engine, |c| {
            let qs = [0usize, 1];
            pecos_circuit_x(c, qs.as_ptr(), 2);
            pecos_circuit_mz(c, qs.as_ptr(), 2);
        })
    };
    assert_eq!(outcomes2, vec![1, 1], "X on both qubits");

    unsafe { pecos_engine_free(engine) };
}

#[test]
fn test_rotation_gate() {
    // RX(pi)|0> = |1> (up to global phase)
    let engine = create_engine("state_vec", 1, 42);

    let outcomes = unsafe {
        run_circuit(engine, |c| {
            let q: usize = 0;
            pecos_circuit_rx(c, std::f64::consts::PI, &raw const q, 1);
            pecos_circuit_mz(c, &raw const q, 1);
        })
    };

    assert_eq!(outcomes, vec![1], "RX(pi)|0> should give |1>");
    unsafe { pecos_engine_free(engine) };
}

#[test]
fn test_circuit_builder_reset() {
    // Test that circuit builder reset works
    let circuit = pecos_circuit_new();

    // Build first circuit
    unsafe {
        let q: usize = 0;
        pecos_circuit_h(circuit, &raw const q, 1);
    }
    let mut bytes1: *mut u8 = std::ptr::null_mut();
    let mut len1: usize = 0;
    unsafe { pecos_circuit_build(circuit, &raw mut bytes1, &raw mut len1) };

    // Reset and build second circuit
    unsafe { pecos_circuit_reset(circuit) };
    unsafe {
        let q: usize = 0;
        pecos_circuit_x(circuit, &raw const q, 1);
    }
    let mut bytes2: *mut u8 = std::ptr::null_mut();
    let mut len2: usize = 0;
    unsafe { pecos_circuit_build(circuit, &raw mut bytes2, &raw mut len2) };

    // Both should produce valid circuits
    assert!(len1 > 0);
    assert!(len2 > 0);

    unsafe {
        pecos_free_bytes(bytes1, len1);
        pecos_free_bytes(bytes2, len2);
        pecos_circuit_free(circuit);
    }
}
