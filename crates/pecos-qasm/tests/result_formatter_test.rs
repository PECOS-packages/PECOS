// Tests for the result_formatter module

use pecos_engines::shot_results::{Data, Shot, ShotVec};
use pecos_engines::sim_builder;
use pecos_qasm::QASMEngine;
use pecos_qasm::result_formatter::{
    QASMResultFormatter, format_as_binary_strings, format_as_decimal_arrays,
};
use std::collections::BTreeMap;
use std::str::FromStr;

#[test]
fn test_format_as_binary_strings() {
    // Create test data with different register sizes
    let mut shot1 = Shot::default();
    shot1.data.insert("c".to_string(), Data::U32(0)); // "00"
    shot1.data.insert("d".to_string(), Data::U32(5)); // "101"
    shot1.data.insert("e".to_string(), Data::U32(1)); // "0001"

    let mut shot2 = Shot::default();
    shot2.data.insert("c".to_string(), Data::U32(3)); // "11"
    shot2.data.insert("d".to_string(), Data::U32(2)); // "010"
    shot2.data.insert("e".to_string(), Data::U32(15)); // "1111"

    let mut shot3 = Shot::default();
    shot3.data.insert("c".to_string(), Data::U32(1)); // "01"
    shot3.data.insert("d".to_string(), Data::U32(7)); // "111"
    // e is missing in this shot

    let results = ShotVec {
        shots: vec![shot1, shot2, shot3],
    };

    // Define register sizes
    let mut register_sizes = BTreeMap::new();
    register_sizes.insert("c".to_string(), 2); // 2-bit register
    register_sizes.insert("d".to_string(), 3); // 3-bit register
    register_sizes.insert("e".to_string(), 4); // 4-bit register

    // Format as binary strings
    let formatted = format_as_binary_strings(&results, &register_sizes);

    // Verify the result
    let json_str = serde_json::to_string(&formatted).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Check register c
    assert_eq!(
        parsed["c"].as_array().unwrap(),
        &vec![
            serde_json::Value::String("00".to_string()),
            serde_json::Value::String("11".to_string()),
            serde_json::Value::String("01".to_string()),
        ]
    );

    // Check register d
    assert_eq!(
        parsed["d"].as_array().unwrap(),
        &vec![
            serde_json::Value::String("101".to_string()),
            serde_json::Value::String("010".to_string()),
            serde_json::Value::String("111".to_string()),
        ]
    );

    // Check register e (with missing value filled as zeros)
    assert_eq!(
        parsed["e"].as_array().unwrap(),
        &vec![
            serde_json::Value::String("0001".to_string()),
            serde_json::Value::String("1111".to_string()),
            serde_json::Value::String("0000".to_string()), // Missing value becomes "0000"
        ]
    );
}

#[test]
fn test_format_as_decimal_arrays() {
    // Create test data
    let mut shot1 = Shot::default();
    shot1.data.insert("c".to_string(), Data::U32(0));
    shot1.data.insert("d".to_string(), Data::I32(-5));

    let mut shot2 = Shot::default();
    shot2.data.insert("c".to_string(), Data::U32(3));
    shot2.data.insert("d".to_string(), Data::I32(10));

    let results = ShotVec {
        shots: vec![shot1, shot2],
    };

    // Format all registers
    let formatted_all = format_as_decimal_arrays(&results, None);
    let json_str = serde_json::to_string(&formatted_all).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Check that both registers are present
    assert!(parsed.as_object().unwrap().contains_key("c"));
    assert!(parsed.as_object().unwrap().contains_key("d"));

    // Format only specific registers
    let formatted_specific = format_as_decimal_arrays(&results, Some(&["c"]));
    let json_str = serde_json::to_string(&formatted_specific).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Check that only "c" is present
    assert!(parsed.as_object().unwrap().contains_key("c"));
    assert!(!parsed.as_object().unwrap().contains_key("d"));

    // Verify values
    assert_eq!(
        parsed["c"].as_array().unwrap(),
        &vec![
            serde_json::Value::Number(0.into()),
            serde_json::Value::Number(3.into()),
        ]
    );
}

#[test]
fn test_empty_results() {
    let results = ShotVec { shots: vec![] };
    let register_sizes = BTreeMap::new();

    // Test empty results formatting
    let formatted = format_as_binary_strings(&results, &register_sizes);
    assert_eq!(formatted, serde_json::json!({}));

    let formatted_decimal = format_as_decimal_arrays(&results, None);
    assert_eq!(formatted_decimal, serde_json::json!({}));
}

#[test]
fn test_qasm_result_formatter_trait() {
    // Create a simple QASM program
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        creg d[1];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Create engine
    let engine = QASMEngine::from_str(qasm).unwrap();

    // Create mock results
    let mut shot1 = Shot::default();
    shot1.data.insert("c".to_string(), Data::U32(0));
    shot1.data.insert("d".to_string(), Data::U32(0));

    let mut shot2 = Shot::default();
    shot2.data.insert("c".to_string(), Data::U32(3));
    shot2.data.insert("d".to_string(), Data::U32(0));

    let results = ShotVec {
        shots: vec![shot1, shot2],
    };

    // Use the trait method
    let formatted = engine.get_binary_string_results(&results).unwrap();
    let json_str = serde_json::to_string(&formatted).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Verify the format
    assert_eq!(
        parsed["c"].as_array().unwrap(),
        &vec![
            serde_json::Value::String("00".to_string()),
            serde_json::Value::String("11".to_string()),
        ]
    );
    assert_eq!(
        parsed["d"].as_array().unwrap(),
        &vec![
            serde_json::Value::String("0".to_string()),
            serde_json::Value::String("0".to_string()),
        ]
    );
}

#[test]
fn test_large_register_values() {
    // Test with larger bit widths
    let mut shot = Shot::default();
    shot.data.insert("reg8".to_string(), Data::U32(255)); // 8-bit max
    shot.data.insert("reg16".to_string(), Data::U32(65535)); // 16-bit max
    shot.data
        .insert("reg32".to_string(), Data::U32(4_294_967_295)); // 32-bit max

    let results = ShotVec { shots: vec![shot] };

    let mut register_sizes = BTreeMap::new();
    register_sizes.insert("reg8".to_string(), 8);
    register_sizes.insert("reg16".to_string(), 16);
    register_sizes.insert("reg32".to_string(), 32);

    let formatted = format_as_binary_strings(&results, &register_sizes);
    let json_str = serde_json::to_string(&formatted).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Check proper formatting
    assert_eq!(
        parsed["reg8"].as_array().unwrap()[0].as_str().unwrap(),
        "11111111" // 8 ones
    );
    assert_eq!(
        parsed["reg16"].as_array().unwrap()[0].as_str().unwrap(),
        "1111111111111111" // 16 ones
    );
    assert_eq!(
        parsed["reg32"].as_array().unwrap()[0].as_str().unwrap(),
        "11111111111111111111111111111111" // 32 ones
    );
}

#[test]
fn test_integration_with_actual_simulation() {
    use pecos_programs::Qasm;
    use pecos_qasm::qasm_engine;

    // Run an actual QASM simulation
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg a[2];
        creg b[1];

        // Create known values
        x q[0]; // Flip first qubit
        x q[2]; // Flip third qubit

        measure q[0] -> a[0];
        measure q[1] -> a[1];
        measure q[2] -> b[0];
    "#;

    // Create engine to get register sizes
    let engine = QASMEngine::from_str(qasm).unwrap();
    let _register_sizes = engine.classical_register_sizes().unwrap();

    // Run simulation
    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .run(5)
        .unwrap();

    // Convert to ShotMap for analysis
    let shot_map = shot_vec.try_as_shot_map().unwrap();

    // Get binary strings for each register
    let a_binary = shot_map.try_bits_as_binary("a").unwrap();
    let b_binary = shot_map.try_bits_as_binary("b").unwrap();

    assert_eq!(a_binary.len(), 5);
    assert_eq!(b_binary.len(), 5);

    // Check that all values are consistent (deterministic circuit)
    for i in 0..5 {
        assert_eq!(a_binary[i], "01"); // a[0]=1, a[1]=0
        assert_eq!(b_binary[i], "1"); // b[0]=1
    }
}

#[test]
fn test_mixed_data_types() {
    // Test with different data types in the Data enum
    let mut shot = Shot::default();
    shot.data.insert("u8_reg".to_string(), Data::U8(255));
    shot.data.insert("u16_reg".to_string(), Data::U16(1000));
    shot.data.insert("i32_reg".to_string(), Data::I32(-42)); // This will be handled in decimal format
    shot.data.insert("bool_reg".to_string(), Data::Bool(true)); // This won't appear in binary format

    let results = ShotVec { shots: vec![shot] };

    let mut register_sizes = BTreeMap::new();
    register_sizes.insert("u8_reg".to_string(), 8);
    register_sizes.insert("u16_reg".to_string(), 10);
    register_sizes.insert("i32_reg".to_string(), 8);
    register_sizes.insert("bool_reg".to_string(), 1);

    // Test binary format (only handles u32 currently)
    let formatted_binary = format_as_binary_strings(&results, &register_sizes);
    let json_str = serde_json::to_string(&formatted_binary).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Only u8 and u16 should be converted (through as_u32())
    assert_eq!(
        parsed["u8_reg"].as_array().unwrap()[0].as_str().unwrap(),
        "11111111"
    );
    assert_eq!(
        parsed["u16_reg"].as_array().unwrap()[0].as_str().unwrap(),
        "1111101000"
    );

    // Test decimal format (handles all numeric types)
    let formatted_decimal = format_as_decimal_arrays(&results, None);
    let json_str = serde_json::to_string(&formatted_decimal).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(
        parsed["u8_reg"].as_array().unwrap()[0].as_u64().unwrap(),
        255
    );
    assert_eq!(
        parsed["u16_reg"].as_array().unwrap()[0].as_u64().unwrap(),
        1000
    );
    assert_eq!(
        parsed["i32_reg"].as_array().unwrap()[0].as_i64().unwrap(),
        -42
    );
}

#[test]
fn test_zero_width_registers() {
    // Test edge case with zero-width registers (shouldn't normally happen, but let's be safe)
    let mut shot = Shot::default();
    shot.data.insert("c".to_string(), Data::U32(0));

    let results = ShotVec { shots: vec![shot] };

    let mut register_sizes = BTreeMap::new();
    register_sizes.insert("c".to_string(), 0); // Zero-width register

    let formatted = format_as_binary_strings(&results, &register_sizes);
    let json_str = serde_json::to_string(&formatted).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Should produce empty strings for zero-width registers
    assert_eq!(
        parsed["c"].as_array().unwrap()[0].as_str().unwrap(),
        "" // Empty string for zero-width
    );
}

#[test]
fn test_bell_state_formatting() {
    // Test a real Bell state scenario
    use pecos_programs::Qasm;
    use pecos_qasm::qasm_engine;

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        h q[0];
        cx q[0], q[1];

        measure q -> c;
    "#;

    let engine = QASMEngine::from_str(qasm).unwrap();
    let _register_sizes = engine.classical_register_sizes().unwrap();

    // Run with enough shots to likely see both outcomes
    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .run(20)
        .unwrap();

    // Convert to ShotMap for analysis
    let shot_map = shot_vec.try_as_shot_map().unwrap();

    // Get binary strings for the register
    let c_binary = shot_map.try_bits_as_binary("c").unwrap();

    // Verify all values are either "00" or "11" (Bell state)
    for binary_str in &c_binary {
        assert!(
            binary_str == "00" || binary_str == "11",
            "Bell state should only produce '00' or '11', got '{binary_str}'"
        );
    }
}
