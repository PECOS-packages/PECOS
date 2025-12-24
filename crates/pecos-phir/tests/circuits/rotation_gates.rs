//! Test for understanding how HUGR represents rotation gates with angles

use pecos_phir::{InputFormat, PhirConfig, Pipeline};

#[test]
fn test_hugr_rotation_gate_structure() {
    // This is how HUGR represents a rotation gate with an angle
    // The angle is passed through the dataflow as a constant input
    let hugr_json = r#"{
        "modules": [{
            "version": "live",
            "metadata": {"name": "rotation_test"},
            "nodes": [
                {"parent": 0, "op": "Module"},
                {"parent": 0, "op": "FuncDefn", "name": "main"},
                {"parent": 1, "op": "Input"},
                {"parent": 1, "op": "Output"},
                {"parent": 1, "op": "Extension", "name": "QAlloc"},
                {"parent": 1, "op": {"op": "Const", "value": 1.5708}},
                {"parent": 1, "op": "Extension", "name": "Rx"},
                {"parent": 1, "op": "Extension", "name": "MeasureFree"}
            ],
            "edges": [
                [[2, 0], [4, 0]],
                [[4, 0], [6, 0]],
                [[5, 0], [6, 1]],
                [[6, 0], [7, 0]],
                [[7, 0], [3, 0]]
            ]
        }],
        "extensions": []
    }"#;

    // Test with new pipeline API
    let config = PhirConfig::default();
    let pipeline = Pipeline::new(config);
    let result: Result<(), _> = pipeline.compile_and_execute(hugr_json, InputFormat::HUGR);

    match result {
        Ok(()) => {
            println!("Rotation gate pipeline execution completed successfully");
        }
        Err(e) => {
            eprintln!("Rotation test failed: {e:?}");
            // For now, expect this to fail since parsers aren't implemented
            assert!(e.to_string().contains("not yet implemented"));
        }
    }
}

#[test]
fn test_edge_based_angle_passing() {
    // Test to understand how angles flow through edges
    let hugr_json = r#"{
        "modules": [{
            "version": "live",
            "metadata": {"name": "edge_test"},
            "nodes": [
                {"parent": 0, "op": "Module"},
                {"parent": 0, "op": "FuncDefn", "name": "main"},
                {"parent": 1, "op": "Input"},
                {"parent": 1, "op": "Output"},
                {"parent": 1, "op": {"op": "Const", "value": 3.14159}},
                {"parent": 1, "op": "Extension", "name": "QAlloc"},
                {"parent": 1, "op": "Extension", "name": "Rz"},
                {"parent": 1, "op": "Extension", "name": "MeasureFree"}
            ],
            "edges": [
                [[2, 0], [5, 0]],
                [[4, 0], [6, 1]],
                [[5, 0], [6, 0]],
                [[6, 0], [7, 0]],
                [[7, 0], [3, 0]]
            ]
        }],
        "extensions": []
    }"#;

    // Test edge-based angle passing with new pipeline API
    let config = PhirConfig::default();
    let pipeline = Pipeline::new(config);
    let result: Result<(), _> = pipeline.compile_and_execute(hugr_json, InputFormat::HUGR);

    match result {
        Ok(()) => {
            println!("Edge-based angle passing pipeline execution completed successfully");
        }
        Err(e) => {
            eprintln!("Edge test failed: {e:?}");
            // For now, expect this to fail since parsers aren't implemented
            assert!(e.to_string().contains("not yet implemented"));
        }
    }
}
