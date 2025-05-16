use pecos_qasm::Preprocessor;
use pecos_qasm::parser::QASMParser;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_simple_include() {
    let temp_dir = TempDir::new().unwrap();
    let include_path = temp_dir.path().join("gates.inc");
    let main_path = temp_dir.path().join("main.qasm");

    // Write the include file
    fs::write(
        &include_path,
        r#"
        include "qelib1.inc";
        gate hadamard a {
            u2(0,pi) a;
        }
    "#,
    )
    .unwrap();

    // Write the main file
    fs::write(
        &main_path,
        r#"
        OPENQASM 2.0;
        include "gates.inc";
        qreg q[2];
        hadamard q[0];
        hadamard q[1];
    "#,
    )
    .unwrap();

    // Parse with preprocessing
    let program = QASMParser::parse_file(&main_path).unwrap();

    // Check that the gate definition was loaded
    assert!(program.gate_definitions.contains_key("hadamard"));
    // After expansion, we'll have more than 2 operations due to gate expansion
    assert!(program.operations.len() > 2);
}

#[test]
fn test_nested_includes() {
    let temp_dir = TempDir::new().unwrap();
    let base_inc = temp_dir.path().join("base.inc");
    let gates_inc = temp_dir.path().join("gates.inc");
    let main_path = temp_dir.path().join("main.qasm");

    // Write the base include file
    fs::write(
        &base_inc,
        r"
        gate u2(phi,lambda) q {
            H q;
            RZ(lambda) q;
            H q;
            RZ(phi) q;
        }
    ",
    )
    .unwrap();

    // Write the gates include file that includes base
    fs::write(
        &gates_inc,
        r#"
        include "base.inc";
        gate hadamard a {
            u2(0,pi) a;
        }
    "#,
    )
    .unwrap();

    // Write the main file
    fs::write(
        &main_path,
        r#"
        OPENQASM 2.0;
        include "gates.inc";
        qreg q[1];
        hadamard q[0];
    "#,
    )
    .unwrap();

    // Parse with preprocessing
    let program = QASMParser::parse_file(&main_path).unwrap();

    // Check that both gate definitions were loaded
    assert!(program.gate_definitions.contains_key("u2"));
    assert!(program.gate_definitions.contains_key("hadamard"));
}

#[test]
fn test_preprocessor_direct() {
    let temp_dir = TempDir::new().unwrap();
    let include_path = temp_dir.path().join("gates.inc");

    // Write the include file
    fs::write(
        &include_path,
        r"
        gate H a {
            u2(0,pi) a;
        }
    ",
    )
    .unwrap();

    // Create QASM with include
    let qasm = format!(
        r#"
        OPENQASM 2.0;
        include "{}";
        qreg q[1];
        H q[0];
    "#,
        include_path.display()
    );

    // Preprocess with the temp directory in include path
    let mut preprocessor = Preprocessor::new();
    if let Some(path_str) = temp_dir.path().to_str() {
        preprocessor.add_path(path_str);
    } else {
        panic!("Invalid path");
    }
    let preprocessed = preprocessor.preprocess_str(&qasm).unwrap();

    // Check that include was replaced
    assert!(!preprocessed.contains("include"));
    assert!(preprocessed.contains("gate H a"));
    assert!(preprocessed.contains("qreg q[1]"));
}

#[test]
fn test_qelib1_include() {
    // Test that qelib1.inc can be loaded
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        H q[0];
    "#;

    // Parse with preprocessing
    let program = QASMParser::parse_str(qasm).unwrap();

    // Check that gate definitions from qelib1 were loaded
    assert!(program.gate_definitions.contains_key("h"));
    assert!(program.gate_definitions.contains_key("x"));
    assert!(program.gate_definitions.contains_key("z"));
}

#[test]
fn test_circular_include_detection() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("file1.inc");
    let file2 = temp_dir.path().join("file2.inc");

    // Create circular includes
    fs::write(&file1, format!(r#"include "{}";"#, file2.display())).unwrap();
    fs::write(&file2, format!(r#"include "{}";"#, file1.display())).unwrap();

    let qasm = format!(
        r#"
        OPENQASM 2.0;
        include "{}";
        qreg q[1];
    "#,
        file1.display()
    );

    // This should fail with circular dependency error
    let result = QASMParser::parse_str(&qasm);
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("Circular dependency"));
    }
}

#[test]
fn test_include_relative_paths() {
    let temp_dir = TempDir::new().unwrap();
    let includes_dir = temp_dir.path().join("includes");
    fs::create_dir(&includes_dir).unwrap();

    let gates_inc = includes_dir.join("gates.inc");
    let main_path = temp_dir.path().join("main.qasm");

    // Write the include file in includes directory
    fs::write(
        &gates_inc,
        r"
        gate my_gate a {
            X a;
        }
    ",
    )
    .unwrap();

    // Write the main file that includes from includes dir
    fs::write(
        &main_path,
        r#"
        OPENQASM 2.0;
        include "gates.inc";
        qreg q[1];
        my_gate q[0];
    "#,
    )
    .unwrap();

    // Parse with preprocessing - should find gates.inc in includes/ directory
    let program = QASMParser::parse_file(&main_path).unwrap();

    // Check that the gate definition was loaded
    assert!(program.gate_definitions.contains_key("my_gate"));
}
