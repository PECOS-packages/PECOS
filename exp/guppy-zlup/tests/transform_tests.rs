//! Integration tests for Guppy IR to Zlup transformation.

use guppy_zlup::{CompileError, compile};

#[test]
fn test_compile_empty_program() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": []
    }"#;

    let result = compile(ir);
    assert!(result.is_ok());
}

#[test]
fn test_compile_simple_function() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "main",
                "params": [],
                "body": []
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("fn main()"));
}

#[test]
fn test_compile_qalloc() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "bell",
                "params": [],
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 2}
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("qalloc"));
    assert!(result.contains("q"));
}

#[test]
fn test_compile_gate() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "test_gate",
                "params": [],
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 2}
                    },
                    {
                        "kind": "gate",
                        "gate": "h",
                        "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
                        ]
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("h q[0]"));
}

#[test]
fn test_compile_two_qubit_gate() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "bell",
                "params": [],
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 2}
                    },
                    {
                        "kind": "gate",
                        "gate": "h",
                        "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
                        ]
                    },
                    {
                        "kind": "gate",
                        "gate": "cx",
                        "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
                        ]
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("h q[0]"));
    assert!(result.contains("cx"));
}

#[test]
fn test_compile_for_loop() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "test_loop",
                "params": [],
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 4}
                    },
                    {
                        "kind": "for",
                        "var": "i",
                        "range": {
                            "start": {"kind": "literal", "value": 0},
                            "end": {"kind": "literal", "value": 4}
                        },
                        "body": [
                            {
                                "kind": "gate",
                                "gate": "h",
                                "targets": [
                                    {"kind": "index", "array": "q", "index": {"kind": "ident", "name": "i"}}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("for"));
    assert!(result.contains("0..4"));
}

#[test]
fn test_compile_if_statement() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "test_if",
                "params": [
                    {"name": "cond", "type": {"kind": "primitive", "name": "bool"}}
                ],
                "body": [
                    {
                        "kind": "if",
                        "condition": {"kind": "ident", "name": "cond"},
                        "then_body": [
                            {"kind": "return", "return_value": {"kind": "literal", "value": 1}}
                        ],
                        "else_body": [
                            {"kind": "return", "return_value": {"kind": "literal", "value": 0}}
                        ]
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    // Simple identifier conditions are converted to != 0 for u1/bool compatibility
    assert!(result.contains("if (cond != 0)"));
    assert!(result.contains("else"));
}

#[test]
fn test_compile_function_with_params() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "add",
                "params": [
                    {"name": "a", "type": {"kind": "primitive", "name": "int"}},
                    {"name": "b", "type": {"kind": "primitive", "name": "int"}}
                ],
                "return_type": {"kind": "primitive", "name": "int"},
                "body": [
                    {
                        "kind": "return",
                        "return_value": {
                            "kind": "binary",
                            "op": "add",
                            "left": {"kind": "ident", "name": "a"},
                            "right": {"kind": "ident", "name": "b"}
                        }
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("fn add(a: i64, b: i64)"));
    assert!(result.contains("-> i64"));
}

#[test]
fn test_compile_invalid_json() {
    let ir = "not valid json";
    let result = compile(ir);
    assert!(matches!(result, Err(CompileError::Parse(_))));
}

#[test]
fn test_compile_result_statement() {
    // Test with proper measure statement (new IR format)
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "bell",
                "params": [],
                "return_type": {"kind": "primitive", "name": "None"},
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 2}
                    },
                    {
                        "kind": "gate",
                        "gate": "h",
                        "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
                        ]
                    },
                    {
                        "kind": "gate",
                        "gate": "cx",
                        "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
                        ]
                    },
                    {
                        "kind": "measure",
                        "targets": [{"kind": "ident", "name": "q"}],
                        "results": ["m"]
                    },
                    {
                        "kind": "result",
                        "tag": "measurements",
                        "value": {"kind": "ident", "name": "m"}
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("fn bell()"));
    assert!(result.contains("-> unit"));
    assert!(result.contains("qalloc(2)"));
    assert!(result.contains("h q[0]"));
    assert!(result.contains("cx"));
    // Measure entire register: mz([2]u1) q
    assert!(result.contains("mz([2]u1) q"));
    assert!(result.contains(r#"result("measurements", m)"#));
    // Should have implicit return (return; is equivalent to return unit; in Zlup)
    assert!(result.contains("return;"));
}

#[test]
fn test_compile_single_qubit_measure() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "test_single",
                "params": [],
                "return_type": {"kind": "primitive", "name": "None"},
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 4}
                    },
                    {
                        "kind": "measure",
                        "targets": [{"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}],
                        "results": ["m"]
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    // Single qubit measurement: mz(u1) q[0]
    assert!(result.contains("mz(u1) q[0]"));
}

#[test]
fn test_compile_multiple_qubit_measure() {
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "test_multi",
                "params": [],
                "return_type": {"kind": "primitive", "name": "None"},
                "body": [
                    {
                        "kind": "qalloc",
                        "name": "q",
                        "size": {"kind": "literal", "value": 4}
                    },
                    {
                        "kind": "measure",
                        "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
                        ],
                        "results": ["m"]
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    // Multiple explicit qubits: mz([2]u1) [q[0], q[1]]
    assert!(result.contains("mz([2]u1) [q[0], q[1]]"));
}

#[test]
fn test_compile_binding_statement() {
    // Test binding (from annotated assignment like x: int = 5)
    let ir = r#"{
        "version": "0.1.0",
        "functions": [
            {
                "name": "test_binding",
                "params": [],
                "return_type": {"kind": "primitive", "name": "int"},
                "body": [
                    {
                        "kind": "binding",
                        "name": "x",
                        "type": {"kind": "primitive", "name": "int"},
                        "value": {"kind": "literal", "value": 5},
                        "is_mutable": true
                    },
                    {
                        "kind": "return",
                        "return_value": {"kind": "ident", "name": "x"}
                    }
                ]
            }
        ]
    }"#;

    let result = compile(ir).unwrap();
    assert!(result.contains("fn test_binding()"));
    assert!(result.contains("mut x: i64 = 5"));
    assert!(result.contains("return x"));
}
