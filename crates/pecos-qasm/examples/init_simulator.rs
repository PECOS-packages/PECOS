use pecos_engines::engines::Engine;
use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::{QASMEngine, count_qubits_in_file};
use std::env;
use std::path::Path;

fn main() {
    // Get the QASM file path from command-line args or use a default
    let args: Vec<String> = env::args().collect();
    let qasm_path = if args.len() >= 2 {
        args[1].clone()
    } else {
        "../../examples/qasm/bell.qasm".to_string()
    };

    let path = Path::new(&qasm_path);

    // First use our utility function to get the qubit count statically
    match count_qubits_in_file(path) {
        Ok(qubit_count) => {
            println!(
                "Static analysis: QASM file '{}' requires {} qubits",
                path.display(),
                qubit_count
            );

            // Now we know how many qubits to allocate for the simulator
            println!("Initializing simulator with {qubit_count} qubits");

            // This is how you would initialize a simulator with the qubit count
            // Here we're using the QASMEngine directly, but you could use any simulator
            let engine_result = QASMEngine::with_seed(path, 42);

            match engine_result {
                Ok(mut engine) => {
                    println!("Successfully initialized simulator from file");

                    // The num_qubits method initially returns 0 because no qubits have been allocated yet
                    println!(
                        "Before execution: Simulator has {} qubits (via num_qubits method)",
                        engine.num_qubits()
                    );

                    // Run the simulation to allocate qubits
                    println!("Running simulation...");
                    match engine.process(()) {
                        Ok(result) => {
                            println!("Simulation completed successfully");
                            // Use registers field instead of deprecated measurements field
                            println!("Measurement results: {:?}", result.registers);

                            // Now num_qubits should match our static count
                            println!(
                                "After execution: Simulator has {} qubits (via num_qubits method)",
                                engine.num_qubits()
                            );
                        }
                        Err(e) => {
                            println!("Simulation failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    println!("Failed to initialize simulator: {e}");
                }
            }
        }
        Err(e) => {
            eprintln!("Error counting qubits: {e}");
        }
    }

    // End of main function
}
