use crate::parser::QASMParser;
use pecos_core::errors::PecosError;
use std::path::Path;

/// Quickly parse a QASM file to extract just the number of qubits.
///
/// # Arguments
///
/// * `path` - Path to the QASM file
///
/// # Returns
///
/// * `Result<usize, PecosError>` - The total number of qubits on success, or a parsing error
pub fn count_qubits_in_file<P: AsRef<Path>>(path: P) -> Result<usize, PecosError> {
    // Parse the file using the existing parser
    let program = QASMParser::parse_file(path)?;

    // Use the total_qubits from the program
    Ok(program.total_qubits)
}

/// Quickly parse a QASM string to extract just the number of qubits.
///
/// # Arguments
///
/// * `qasm` - QASM program as a string
///
/// # Returns
///
/// * `Result<usize, PecosError>` - The total number of qubits on success, or a parsing error
pub fn count_qubits_in_str(qasm: &str) -> Result<usize, PecosError> {
    // Parse the string using the existing parser
    let program = QASMParser::parse_str(qasm)?;

    // Use the total_qubits from the program
    Ok(program.total_qubits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_count_qubits_in_str() -> Result<(), Box<dyn std::error::Error>> {
        // Test with a simple program that has one register with 2 qubits
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
        "#;

        let qubit_count = count_qubits_in_str(qasm)?;
        assert_eq!(qubit_count, 2);

        // Test with multiple registers
        let qasm_multiple = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q1[3];
            qreg q2[4];
            creg c[2];
        "#;

        let qubit_count = count_qubits_in_str(qasm_multiple)?;
        assert_eq!(qubit_count, 7); // 3 + 4 = 7

        Ok(())
    }

    #[test]
    fn test_count_qubits_in_file() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary file with a simple QASM program
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[5];
            creg c[1];
            x q[0];
        "#;

        let mut file = NamedTempFile::new()?;
        write!(file.as_file_mut(), "{qasm}")?;

        let qubit_count = count_qubits_in_file(file.path())?;
        assert_eq!(qubit_count, 5);

        Ok(())
    }
}
