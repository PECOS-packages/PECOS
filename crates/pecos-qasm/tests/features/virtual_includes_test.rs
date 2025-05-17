use pecos_qasm::parser::{ParseConfig, QASMParser};
use pecos_qasm::{Preprocessor, QASMEngine};

#[test]
fn test_virtual_include_single() {
    // Create a virtual include
    let virtual_includes = vec![(
        "my_gates.inc".to_string(),
        r#"
            include "qelib1.inc";
            gate my_h a {
                u2(0,pi) a;
            }
        "#
        .to_string(),
    )];

    let qasm = r#"
        OPENQASM 2.0;
        include "my_gates.inc";
        qreg q[1];
        my_h q[0];
    "#;

    // Parse with virtual includes
    let program = {
        let config = ParseConfig {
            includes: virtual_includes,
            ..Default::default()
        };
        QASMParser::parse_with_config(qasm, &config)
    }
    .unwrap();

    // Verify the gate was loaded
    assert!(program.gate_definitions.contains_key("my_h"));
    // After expansion, my_h expands to u2, which expands to more operations
    assert!(program.operations.len() > 1);
}

#[test]
fn test_virtual_include_multiple() {
    // Create multiple virtual includes
    let virtual_includes = vec![
        (
            "basics.inc".to_string(),
            r"
            gate prep q {
                H q;
            }
        "
            .to_string(),
        ),
        (
            "advanced.inc".to_string(),
            r"
            gate bell a,b {
                H a;
                CX a,b;
            }
        "
            .to_string(),
        ),
    ];

    let qasm = r#"
        OPENQASM 2.0;
        include "basics.inc";
        include "advanced.inc";
        qreg q[2];
        prep q[0];
        bell q[0],q[1];
    "#;

    // Parse with virtual includes
    let program = {
        let config = ParseConfig {
            includes: virtual_includes,
            ..Default::default()
        };
        QASMParser::parse_with_config(qasm, &config)
    }
    .unwrap();

    // Verify both gates were loaded
    assert!(program.gate_definitions.contains_key("prep"));
    assert!(program.gate_definitions.contains_key("bell"));
    // After gate expansion, we have 3 operations: h (from prep), H and cx (from bell)
    assert_eq!(program.operations.len(), 3);
}

#[test]
fn test_virtual_include_nested() {
    // Create virtual includes with nesting
    let virtual_includes = vec![
        (
            "base.inc".to_string(),
            r"
            gate u2(phi,lambda) q {
                RZ(phi+lambda) q;
            }
        "
            .to_string(),
        ),
        (
            "derived.inc".to_string(),
            r#"
            include "base.inc";
            gate h q {
                H q;
            }
        "#
            .to_string(),
        ),
    ];

    let qasm = r#"
        OPENQASM 2.0;
        include "derived.inc";
        qreg q[1];
        h q[0];
    "#;

    // Parse with virtual includes
    let program = {
        let config = ParseConfig {
            includes: virtual_includes,
            ..Default::default()
        };
        QASMParser::parse_with_config(qasm, &config)
    }
    .unwrap();

    // Verify both gates were loaded from nested includes
    assert!(program.gate_definitions.contains_key("u2"));
    assert!(program.gate_definitions.contains_key("h"));
}

#[test]
fn test_virtual_include_circular_dependency() {
    // Create circular virtual includes
    let virtual_includes = vec![
        ("a.inc".to_string(), r#"include "b.inc";"#.to_string()),
        ("b.inc".to_string(), r#"include "a.inc";"#.to_string()),
    ];

    let qasm = r#"
        OPENQASM 2.0;
        include "a.inc";
        qreg q[1];
    "#;

    // This should fail with circular dependency error
    let result = {
        let config = ParseConfig {
            includes: virtual_includes,
            ..Default::default()
        };
        QASMParser::parse_with_config(qasm, &config)
    };
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("Circular dependency"));
    }
}

#[test]
fn test_virtual_include_with_engine() {
    // Test using virtual includes with the engine
    let _virtual_includes = [(
        "custom.inc".to_string(),
        r#"
            include "qelib1.inc";
            gate sqrt_x a {
                sx a;
            }
        "#
        .to_string(),
    )];

    let qasm = r#"
        OPENQASM 2.0;
        include "custom.inc";
        qreg q[1];
        sqrt_x q[0];
    "#;

    // Create engine and load with virtual includes
    let _engine = QASMEngine::builder()
        .with_virtual_include(
            "custom.inc",
            r#"
            include "qelib1.inc";
            gate sqrt_x a {
                sx a;
            }
        "#,
        )
        .build_from_str(qasm)
        .unwrap();
}

#[test]
fn test_virtual_include_overrides_file() {
    // Virtual includes should take precedence over file system includes
    let virtual_includes = vec![(
        "qelib1.inc".to_string(),
        r"
            gate h a {
                // Custom implementation with native gates only
                H a;
            }
        "
        .to_string(),
    )];

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
    "#;

    // Parse with virtual includes
    let program = {
        let config = ParseConfig {
            includes: virtual_includes,
            ..Default::default()
        };
        QASMParser::parse_with_config(qasm, &config)
    }
    .unwrap();

    // Should use our custom h gate, not the standard one
    assert!(program.gate_definitions.contains_key("h"));
    // Our custom version should not have other standard gates
    assert!(!program.gate_definitions.contains_key("x"));
    assert!(!program.gate_definitions.contains_key("cx"));
}

#[test]
fn test_preprocessor_direct_usage() {
    // Test using the preprocessor directly
    let mut preprocessor = Preprocessor::new();
    preprocessor.add_include("test.inc", "gate id a { U(0,0,0) a; }");

    let qasm = r#"
        OPENQASM 2.0;
        include "test.inc";
        qreg q[1];
        id q[0];
    "#;

    let preprocessed = preprocessor.preprocess_str(qasm).unwrap();

    // The include should be replaced with the content
    assert!(!preprocessed.contains("include"));
    assert!(preprocessed.contains("gate id a"));
}

#[test]
fn test_mixed_virtual_and_file_includes() {
    use std::fs;
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let file_inc = temp_dir.path().join("file.inc");

    // Create a file include
    fs::write(&file_inc, "gate from_file a { X a; }").unwrap();

    // Create a virtual include
    let virtual_includes = vec![(
        "virtual.inc".to_string(),
        "gate from_virtual a { Y a; }".to_string(),
    )];

    let qasm = format!(
        r#"
        OPENQASM 2.0;
        include "virtual.inc";
        include "{}";
        qreg q[1];
        from_virtual q[0];
        from_file q[0];
    "#,
        file_inc.display()
    );

    // Parse with virtual includes
    let config = ParseConfig {
        includes: virtual_includes,
        ..Default::default()
    };
    let program = QASMParser::parse_with_config(&qasm, &config).unwrap();

    // Both gates should be loaded
    assert!(program.gate_definitions.contains_key("from_virtual"));
    assert!(program.gate_definitions.contains_key("from_file"));
}
