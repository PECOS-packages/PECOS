use pecos_qasm::{ParseConfig, QASMEngine, QASMParser};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_custom_include_paths() {
    // Create multiple temp directories to test path searching
    let temp_dir1 = TempDir::new().unwrap();
    let temp_dir2 = TempDir::new().unwrap();
    let temp_dir3 = TempDir::new().unwrap();

    // Create include files in different directories
    let file1_path = temp_dir1.path().join("gates1.inc");
    let file2_path = temp_dir2.path().join("gates2.inc");
    let file3_path = temp_dir3.path().join("gates3.inc");

    fs::write(&file1_path, "gate g1 a { u1(pi/2) a; }").unwrap();
    fs::write(&file2_path, "gate g2 a { u2(0,pi) a; }").unwrap();
    fs::write(&file3_path, "gate g3 a { u3(pi,0,pi) a; }").unwrap();

    // QASM program that uses all includes
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        include "gates1.inc";
        include "gates2.inc";
        include "gates3.inc";
        qreg q[2];
        g1 q[0];
        g2 q[1];
        g3 q[0];
    "#;

    // Parse with custom include paths
    let custom_paths = vec![
        temp_dir1.path().to_path_buf(),
        temp_dir2.path().to_path_buf(),
        temp_dir3.path().to_path_buf(),
    ];

    let config = ParseConfig {
        search_paths: custom_paths,
        ..Default::default()
    };
    let program = QASMParser::parse_with_config(qasm, &config).unwrap();

    // Verify the program parsed successfully and has gate definitions
    assert!(program.gate_definitions.contains_key("g1"));
    assert!(program.gate_definitions.contains_key("g2"));
    assert!(program.gate_definitions.contains_key("g3"));
}

#[test]
fn test_include_path_priority() {
    // Test that custom paths are searched before standard locations
    let temp_dir1 = TempDir::new().unwrap();
    let temp_dir2 = TempDir::new().unwrap();

    // Create same file in both locations with different content
    let file1_path = temp_dir1.path().join("common.inc");
    let file2_path = temp_dir2.path().join("common.inc");

    fs::write(&file1_path, "gate priority1 a { X a; }").unwrap();
    fs::write(&file2_path, "gate priority2 a { Y a; }").unwrap();

    let qasm = r#"
        OPENQASM 2.0;
        include "common.inc";
        qreg q[1];
    "#;

    // Test with first directory in path - should get priority1
    let config = ParseConfig {
        search_paths: vec![temp_dir1.path().into()],
        ..Default::default()
    };
    let program1 = QASMParser::parse_with_config(qasm, &config).unwrap();
    assert!(program1.gate_definitions.contains_key("priority1"));
    assert!(!program1.gate_definitions.contains_key("priority2"));

    // Test with second directory in path - should get priority2
    let config = ParseConfig {
        search_paths: vec![temp_dir2.path().into()],
        ..Default::default()
    };
    let program2 = QASMParser::parse_with_config(qasm, &config).unwrap();
    assert!(!program2.gate_definitions.contains_key("priority1"));
    assert!(program2.gate_definitions.contains_key("priority2"));

    // Test with both paths - first should take priority
    let config = ParseConfig {
        search_paths: vec![temp_dir1.path().into(), temp_dir2.path().into()],
        ..Default::default()
    };
    let program3 = QASMParser::parse_with_config(qasm, &config).unwrap();
    assert!(program3.gate_definitions.contains_key("priority1"));
    assert!(!program3.gate_definitions.contains_key("priority2"));
}

#[test]
fn test_engine_with_custom_include_paths() {
    let temp_dir = TempDir::new().unwrap();
    let include_path = temp_dir.path().join("custom.inc");

    fs::write(&include_path, "gate custom a { H a; }").unwrap();

    let qasm = r#"
        OPENQASM 2.0;
        include "custom.inc";
        qreg q[1];
        custom q[0];
    "#;

    let engine = QASMEngine::builder()
        .with_include_paths(&[temp_dir.path().to_str().unwrap()])
        .build_from_str(qasm)
        .unwrap();

    // Verify the gate was loaded
    assert!(engine.gate_definitions().unwrap().contains_key("custom"));
}

#[test]
fn test_paths_with_virtual_includes() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("file.inc");

    fs::write(&file_path, "gate file_gate a { Z a; }").unwrap();

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        include "file.inc";
        include "virtual.inc";
        qreg q[1];
        file_gate q[0];
        virtual_gate q[0];
    "#;

    let virtual_includes = vec![(
        "virtual.inc".to_string(),
        "gate virtual_gate a { s a; }".to_string(),
    )];

    let config = ParseConfig {
        search_paths: vec![temp_dir.path().into()],
        includes: virtual_includes.into_iter().collect(),
        ..Default::default()
    };
    let program = QASMParser::parse_with_config(qasm, &config).unwrap();

    // Both gates should be available
    assert!(program.gate_definitions.contains_key("file_gate"));
    assert!(program.gate_definitions.contains_key("virtual_gate"));
}

#[test]
fn test_include_not_found_with_custom_paths() {
    let temp_dir = TempDir::new().unwrap();

    let qasm = r#"
        OPENQASM 2.0;
        include "nonexistent.inc";
    "#;

    // Even with custom paths, missing file should error
    let config = ParseConfig {
        search_paths: vec![temp_dir.path().into()],
        ..Default::default()
    };
    let result = QASMParser::parse_with_config(qasm, &config);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_path_collection_types() {
    // Test that various collection types work as include paths
    let temp_dir = TempDir::new().unwrap();
    let include_path = temp_dir.path().join("test.inc");
    fs::write(&include_path, "gate test a { H a; }").unwrap();

    let qasm = r#"
        OPENQASM 2.0;
        include "test.inc";
        qreg q[1];
    "#;

    // Test with Vec
    let config = ParseConfig {
        search_paths: vec![temp_dir.path().into()],
        ..Default::default()
    };
    let _program1 = QASMParser::parse_with_config(qasm, &config).unwrap();

    // Test with slice
    let paths = [temp_dir.path().into()];
    let config = ParseConfig {
        search_paths: paths.to_vec(),
        ..Default::default()
    };
    let _program2 = QASMParser::parse_with_config(qasm, &config).unwrap();

    // Test with iterator
    let config = ParseConfig {
        search_paths: std::iter::once(temp_dir.path().into()).collect(),
        ..Default::default()
    };
    let _program3 = QASMParser::parse_with_config(qasm, &config).unwrap();

    // Test with PathBuf vector
    let path_vec: Vec<PathBuf> = vec![temp_dir.path().to_path_buf()];
    let config = ParseConfig {
        search_paths: path_vec.into_iter().collect(),
        ..Default::default()
    };
    let _program4 = QASMParser::parse_with_config(qasm, &config).unwrap();
}
