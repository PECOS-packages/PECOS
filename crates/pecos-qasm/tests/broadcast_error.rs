// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.
use pecos_qasm::QASMParser;

#[test]
fn odd_two_qubit_broadcast_names_even_count_requirement() {
    let error = QASMParser::parse_str("OPENQASM 2.0; qreg q[3]; CX q;")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("3 qubits")
            && error.contains("must be even number")
            && error.contains("complete qubit pairs"),
        "{error}"
    );
}
