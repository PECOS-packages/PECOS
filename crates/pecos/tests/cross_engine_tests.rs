//! Cross-engine comparison tests.
//!
//! Verify that QASM, PHIR-JSON, and QASM->PHIR->PhirEngine produce identical
//! results for the same circuits when run with the same seed.

#![cfg(feature = "runtime")]

use pecos::prelude::*;

// ---------------------------------------------------------------------------
// QASM programs
// ---------------------------------------------------------------------------

/// Measure |0> -- deterministic, always 0.
const MEASURE_ZERO_QASM: &str = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
"#;

/// X gate then measure -- deterministic, always 1.
const X_MEASURE_QASM: &str = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
x q[0];
measure q[0] -> c[0];
"#;

/// Bell state -- non-deterministic but correlated.
const BELL_QASM: &str = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"#;

// ---------------------------------------------------------------------------
// PHIR-JSON programs (equivalent circuits)
// ---------------------------------------------------------------------------

const MEASURE_ZERO_PHIR_JSON: &str = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "c", "size": 1},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"cop": "Result", "args": ["m"], "returns": ["c"]}
    ]
}"#;

const X_MEASURE_PHIR_JSON: &str = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "c", "size": 1},
        {"qop": "X", "args": [["q", 0]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"cop": "Result", "args": ["m"], "returns": ["c"]}
    ]
}"#;

const BELL_PHIR_JSON: &str = r#"{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
        {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
        {"data": "cvar_define", "data_type": "i64", "variable": "c", "size": 2},
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [["q", 0], ["q", 1]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        {"cop": "Result", "args": ["m"], "returns": ["c"]}
    ]
}"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run a QASM program through `QasmEngine` and return c-register values.
fn run_qasm(qasm: &str, seed: u64, shots: usize) -> Vec<u32> {
    let results = sim_builder()
        .classical(qasm_engine().qasm(qasm))
        .seed(seed)
        .run(shots)
        .expect("QasmEngine run should succeed");

    results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(Data::as_u32))
        .collect()
}

/// Run a PHIR-JSON program through `PhirJsonEngine` and return c-register values.
fn run_phir_json(json: &str, seed: u64, shots: usize) -> Vec<u32> {
    let results = sim_builder()
        .classical(phir_json_engine().json(json).expect("JSON parse"))
        .seed(seed)
        .run(shots)
        .expect("PhirJsonEngine run should succeed");

    results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(Data::as_u32))
        .collect()
}

/// Run a QASM program through QASM->PHIR->RON->PhirEngine path.
fn run_qasm_via_phir(qasm: &str, seed: u64, shots: usize) -> Vec<u32> {
    let ron_string = pecos_qasm::qasm_to_ron(qasm).expect("QASM to RON conversion");
    let builder = pecos_phir::phir_engine()
        .from_ron(&ron_string)
        .expect("RON to PhirEngineBuilder");

    let results = sim_builder()
        .classical(builder)
        .seed(seed)
        .run(shots)
        .expect("PhirEngine run should succeed");

    results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(Data::as_u32))
        .collect()
}

// ---------------------------------------------------------------------------
// Cross-engine comparison: QASM vs PHIR-JSON
// ---------------------------------------------------------------------------

#[test]
fn cross_engine_measure_zero_qasm_vs_phir_json() {
    let qasm_results = run_qasm(MEASURE_ZERO_QASM, 42, 20);
    let phir_results = run_phir_json(MEASURE_ZERO_PHIR_JSON, 42, 20);

    assert!(
        qasm_results.iter().all(|&v| v == 0),
        "QASM: measuring |0> should give 0"
    );
    assert!(
        phir_results.iter().all(|&v| v == 0),
        "PHIR-JSON: measuring |0> should give 0"
    );
    assert_eq!(qasm_results, phir_results);
}

#[test]
fn cross_engine_x_measure_qasm_vs_phir_json() {
    let qasm_results = run_qasm(X_MEASURE_QASM, 42, 20);
    let phir_results = run_phir_json(X_MEASURE_PHIR_JSON, 42, 20);

    assert!(
        qasm_results.iter().all(|&v| v == 1),
        "QASM: X|0> then measure should give 1"
    );
    assert!(
        phir_results.iter().all(|&v| v == 1),
        "PHIR-JSON: X|0> then measure should give 1"
    );
    assert_eq!(qasm_results, phir_results);
}

#[test]
fn cross_engine_bell_qasm_vs_phir_json() {
    let qasm_results = run_qasm(BELL_QASM, 42, 100);
    let phir_results = run_phir_json(BELL_PHIR_JSON, 42, 100);

    assert_eq!(qasm_results.len(), 100);
    assert_eq!(phir_results.len(), 100);
    assert_eq!(qasm_results, phir_results, "Bell state: QASM != PHIR-JSON");

    // Verify Bell state correlations: only 00 and 11 outcomes
    for &v in &qasm_results {
        assert!(
            v == 0 || v == 3,
            "Bell state should only produce 00 or 11, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cross-engine comparison: QASM vs QASM->PHIR->PhirEngine
// ---------------------------------------------------------------------------

#[test]
fn cross_engine_measure_zero_qasm_vs_phir_engine() {
    let qasm_results = run_qasm(MEASURE_ZERO_QASM, 42, 20);
    let phir_results = run_qasm_via_phir(MEASURE_ZERO_QASM, 42, 20);

    assert!(qasm_results.iter().all(|&v| v == 0));
    assert!(phir_results.iter().all(|&v| v == 0));
}

#[test]
fn cross_engine_x_measure_qasm_vs_phir_engine() {
    let qasm_results = run_qasm(X_MEASURE_QASM, 42, 20);
    let phir_results = run_qasm_via_phir(X_MEASURE_QASM, 42, 20);

    assert!(qasm_results.iter().all(|&v| v == 1));
    assert!(phir_results.iter().all(|&v| v == 1));
}

#[test]
fn cross_engine_bell_qasm_vs_phir_engine() {
    let qasm_results = run_qasm(BELL_QASM, 42, 100);
    let phir_results = run_qasm_via_phir(BELL_QASM, 42, 100);

    assert_eq!(qasm_results.len(), 100);
    assert_eq!(phir_results.len(), 100);

    for &v in &qasm_results {
        assert!(v == 0 || v == 3, "QASM Bell state: only 00 or 11, got {v}");
    }
    for &v in &phir_results {
        assert!(
            v == 0 || v == 3,
            "PhirEngine Bell state: only 00 or 11, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// Three-way comparison
// ---------------------------------------------------------------------------

#[test]
fn three_way_measure_zero() {
    let qasm = run_qasm(MEASURE_ZERO_QASM, 99, 10);
    let phir_json = run_phir_json(MEASURE_ZERO_PHIR_JSON, 99, 10);
    let phir_engine = run_qasm_via_phir(MEASURE_ZERO_QASM, 99, 10);

    assert!(qasm.iter().all(|&v| v == 0));
    assert!(phir_json.iter().all(|&v| v == 0));
    assert!(phir_engine.iter().all(|&v| v == 0));
}

#[test]
fn three_way_x_measure() {
    let qasm = run_qasm(X_MEASURE_QASM, 99, 10);
    let phir_json = run_phir_json(X_MEASURE_PHIR_JSON, 99, 10);
    let phir_engine = run_qasm_via_phir(X_MEASURE_QASM, 99, 10);

    assert!(qasm.iter().all(|&v| v == 1));
    assert!(phir_json.iter().all(|&v| v == 1));
    assert!(phir_engine.iter().all(|&v| v == 1));
}
