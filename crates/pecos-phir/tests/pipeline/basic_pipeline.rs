//! Test for the PHIR (PECOS High-level Intermediate Representation) compilation pipeline

use pecos_phir::{InputFormat, PhirConfig, Pipeline};

#[test]
fn test_simple_hadamard_measure() {
    // Sample HUGR JSON (new format with modules array)
    let hugr_json = r#"{
        "modules": [{
            "version": "live",
            "metadata": {"name": "hadamard_test"},
            "nodes": [
                {"parent": 0, "op": "Module"},
                {"parent": 0, "op": "FuncDefn", "name": "main"},
                {"parent": 1, "op": "Input"},
                {"parent": 1, "op": "Output"},
                {"parent": 1, "op": "Extension", "name": "QAlloc"},
                {"parent": 1, "op": "Extension", "name": "H"},
                {"parent": 1, "op": "Extension", "name": "MeasureFree"}
            ],
            "edges": [
                [[2, 0], [4, 0]],
                [[4, 0], [5, 0]],
                [[5, 0], [6, 0]],
                [[6, 0], [3, 0]]
            ]
        }],
        "extensions": []
    }"#;

    let config = PhirConfig {
        debug: true,
        ..Default::default()
    };

    let pipeline = Pipeline::new(config);
    let result: Result<(), _> = pipeline.compile_and_execute(hugr_json, InputFormat::HUGR);

    match result {
        Ok(()) => {
            // Currently just testing that pipeline doesn't crash
            // TODO: Add actual execution and verification
            println!("Pipeline execution completed successfully");
        }
        Err(e) => {
            eprintln!("Compilation failed: {e:?}");
            // For now, expect this to fail since parsers aren't implemented
            assert!(e.to_string().contains("not yet implemented"));
        }
    }
}

#[test]
fn test_bell_state_circuit() {
    let hugr_json = r#"{
        "modules": [{
            "version": "live",
            "metadata": {"name": "bell_state"},
            "nodes": [
                {"parent": 0, "op": "Module"},
                {"parent": 0, "op": "FuncDefn", "name": "main"},
                {"parent": 1, "op": "Input"},
                {"parent": 1, "op": "Output"},
                {"parent": 1, "op": "Extension", "name": "QAlloc"},
                {"parent": 1, "op": "Extension", "name": "QAlloc"},
                {"parent": 1, "op": "Extension", "name": "H"},
                {"parent": 1, "op": "Extension", "name": "CX"},
                {"parent": 1, "op": "Extension", "name": "MeasureFree"},
                {"parent": 1, "op": "Extension", "name": "MeasureFree"}
            ],
            "edges": [
                [[2, 0], [4, 0]],
                [[2, 0], [5, 0]],
                [[4, 0], [6, 0]],
                [[6, 0], [7, 0]],
                [[5, 0], [7, 1]],
                [[7, 0], [8, 0]],
                [[7, 1], [9, 0]],
                [[8, 0], [3, 0]],
                [[9, 0], [3, 1]]
            ]
        }],
        "extensions": []
    }"#;

    let config = PhirConfig::default();
    let pipeline = Pipeline::new(config);
    let result: Result<(), _> = pipeline.compile_and_execute(hugr_json, InputFormat::HUGR);

    match result {
        Ok(()) => {
            println!("Bell state pipeline execution completed successfully");
        }
        Err(e) => {
            eprintln!("Bell state compilation failed: {e:?}");
            // For now, expect this to fail since parsers aren't implemented
            assert!(e.to_string().contains("not yet implemented"));
        }
    }
}
