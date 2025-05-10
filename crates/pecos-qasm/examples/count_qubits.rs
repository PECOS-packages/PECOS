use pecos_qasm::{count_qubits_in_file, count_qubits_in_str};
use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        // If a file path is provided, count qubits in the file
        let path = Path::new(&args[1]);
        if path.exists() {
            match count_qubits_in_file(path) {
                Ok(count) => {
                    println!("File: {}", path.display());
                    println!("Total qubits: {count}");
                }
                Err(e) => {
                    eprintln!("Error parsing file: {e}");
                }
            }
        } else {
            // Treat the argument as a QASM string
            match count_qubits_in_str(&args[1]) {
                Ok(count) => {
                    println!("String input");
                    println!("Total qubits: {count}");
                }
                Err(e) => {
                    eprintln!("Error parsing string: {e}");
                }
            }
        }
    } else {
        // If no arguments are provided, use an example string
        println!("No input provided. Using example QASM program...");

        // Create an example QASM program
        let example_qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";

            // Define quantum registers
            qreg q1[2];
            qreg q2[3];

            // Define classical registers
            creg c[5];

            // Apply some gates
            h q1[0];
            cx q1[0], q1[1];
            x q2[0];

            // Measure qubits
            measure q1[0] -> c[0];
            measure q1[1] -> c[1];
            measure q2[0] -> c[2];
        "#;

        // Count qubits in the example program
        match count_qubits_in_str(example_qasm) {
            Ok(count) => {
                println!("Example QASM program:");
                println!("Total qubits: {count}");
            }
            Err(e) => {
                eprintln!("Error parsing example: {e}");
            }
        }

        // Demo creating a temporary file for the file-based function
        println!("\nCreating a temporary QASM file...");
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("example.qasm");

        fs::write(&file_path, example_qasm)?;
        println!("Wrote example QASM to: {}", file_path.display());

        // Count qubits using the file function
        match count_qubits_in_file(&file_path) {
            Ok(count) => {
                println!("Total qubits from file: {count}");
            }
            Err(e) => {
                eprintln!("Error parsing file: {e}");
            }
        }
    }

    Ok(())
}
