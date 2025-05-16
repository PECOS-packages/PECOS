use pecos_qasm::parser::QASMParser;
use std::io::Write;

#[test]
fn test_parse_simple_program() -> Result<(), Box<dyn std::error::Error>> {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        H q[0];
        CX q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    let program = QASMParser::parse_str(qasm)?;

    assert_eq!(program.version, "2.0");
    assert_eq!(
        program.quantum_registers.get("q").map(std::vec::Vec::len),
        Some(2)
    );
    assert_eq!(program.classical_registers.get("c"), Some(&2));
    assert_eq!(program.operations.len(), 4);

    Ok(())
}

#[test]
fn test_parse_conditional_program() -> Result<(), Box<dyn std::error::Error>> {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        H q[0];
        measure q[0] -> c[0];
    "#;

    let mut file = tempfile::NamedTempFile::new()?;
    write!(&mut file, "{qasm}")?;

    let program = QASMParser::parse_file(file.path())?;

    // Print debug information
    println!("Quantum registers: {:?}", program.quantum_registers);
    println!("Classical registers: {:?}", program.classical_registers);
    println!("Number of operations: {}", program.operations.len());
    for (i, op) in program.operations.iter().enumerate() {
        println!("Operation {i}: {op:?}");
    }

    // Verify the program was parsed correctly
    assert_eq!(program.quantum_registers.len(), 1);
    assert_eq!(program.classical_registers.len(), 1);
    assert_eq!(program.operations.len(), 2); // h, measure

    // Check if the operations are correct
    match &program.operations[0] {
        pecos_qasm::Operation::Gate { name, .. } => {
            assert_eq!(name, "H");
        }
        _ => panic!("First operation should be a gate"),
    }

    match &program.operations[1] {
        pecos_qasm::Operation::Measure { .. } => {
            // Measurement parsed correctly
        }
        _ => panic!("Second operation should be a measure"),
    }

    Ok(())
}
