use pecos_qasm::parser::QASMParser;

#[test]
fn test_qelib1_include_parsing() {
    // Test parsing a simple QASM program with qelib1.inc
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
    "#;

    match QASMParser::parse_str(qasm) {
        Ok(program) => {
            println!(
                "Successfully parsed with {} gate definitions",
                program.gate_definitions.len()
            );
            for name in program.gate_definitions.keys() {
                println!("  - {name}");
            }
        }
        Err(e) => {
            println!("Parse error: {e:?}");
            panic!("Failed to parse qelib1.inc");
        }
    }
}

#[test]
fn test_inline_gate_def() {
    // Test parsing gate definitions inline
    let qasm = r"
        OPENQASM 2.0;
        gate H a { id a; }
        gate id a { RZ(0) a; }
        qreg q[1];
        H q[0];
    ";

    match QASMParser::parse_str_raw(qasm) {
        Ok(program) => {
            println!(
                "Successfully parsed {} operations",
                program.operations.len()
            );
        }
        Err(e) => {
            println!("Parse error: {e:?}");
            panic!("Failed to parse inline gates");
        }
    }
}
