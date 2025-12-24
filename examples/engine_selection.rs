//! Example demonstrating different ways to use PECOS engines
//!
//! This example shows:
//! 1. Static engine selection (compile-time) - best performance
//! 2. Dynamic engine selection (runtime) - flexible but slightly slower
//! 3. Using the new sim() API vs the traditional .to_sim() API

use pecos::prelude::*;
use pecos::{EngineType, DynamicEngineBuilder, sim_dynamic};
use pecos_engines::{sim, SimBuilder, DepolarizingNoise};
use pecos_qasm::qasm_engine;
use pecos_qis_sim::llvm_engine;
use pecos_selene_engine::selene_executable;
use pecos_programs::QasmProgram;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example quantum circuit in OpenQASM
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    println!("=== PECOS Engine Selection Examples ===\n");

    // =========================================================================
    // 1. Static Engine Selection (Compile-time)
    // =========================================================================
    println!("1. Static Engine Selection (best performance):");

    // Traditional .to_sim() pattern
    let results_traditional = qasm_engine()
        .program(QasmProgram::from_string(qasm_code))
        .to_sim()
        .seed(42)
        .noise(DepolarizingNoise { p: 0.01 })
        .run(1000)?;

    println!("   Traditional pattern: {} shots completed", results_traditional.len());

    // New sim() pattern - functionally equivalent
    let results_functional = sim(qasm_engine().program(QasmProgram::from_string(qasm_code)))
        .seed(42)
        .noise(DepolarizingNoise { p: 0.01 })
        .run(1000)?;

    println!("   Functional pattern: {} shots completed", results_functional.len());

    // Using From trait explicitly
    let results_from = SimBuilder::from(qasm_engine().program(QasmProgram::from_string(qasm_code)))
        .seed(42)
        .noise(DepolarizingNoise { p: 0.01 })
        .run(1000)?;

    println!("   From trait pattern: {} shots completed\n", results_from.len());

    // =========================================================================
    // 2. Dynamic Engine Selection (Runtime)
    // =========================================================================
    println!("2. Dynamic Engine Selection (runtime flexibility):");

    // Simulate getting engine type from user input or config
    let user_choice = "qasm"; // Could come from CLI args, config file, etc.

    // Create engine based on runtime selection
    let dynamic_builder = match user_choice {
        "qasm" => {
            println!("   User selected QASM engine");
            DynamicEngineBuilder::new(qasm_engine().program(QasmProgram::from_string(qasm_code)))
        }
        "llvm" => {
            println!("   User selected LLVM engine");
            // In real code, you'd have LLVM IR here
            use pecos_programs::QisProgram;
            DynamicEngineBuilder::new(qis_engine().program(QisProgram::from_string("define void @main() { ret void }")))
        }
        "selene" => {
            println!("   User selected Selene engine");
            // In real code, you'd have HUGR here
            use pecos_programs::HugrProgram;
            DynamicEngineBuilder::new(selene_executable().program(HugrProgram::from_bytes(vec![])).qubits(2))
        }
        _ => panic!("Unknown engine type: {}", user_choice),
    };

    // Use the dynamically selected engine
    let results_dynamic = sim_dynamic(dynamic_builder)
        .seed(42)
        .noise(DepolarizingNoise { p: 0.01 })
        .run(1000)?;

    println!("   Dynamic selection: {} shots completed\n", results_dynamic.len());

    // =========================================================================
    // 3. Advanced: Storing Multiple Engines
    // =========================================================================
    println!("3. Advanced: Managing multiple engines:");

    use std::collections::BTreeMap;

    // Create a collection of engines (useful for benchmarking, A/B testing, etc.)
    let mut engines: BTreeMap<&str, DynamicEngineBuilder> = BTreeMap::new();

    // Add different engine configurations
    engines.insert("qasm_basic", DynamicEngineBuilder::new(
        qasm_engine().program(QasmProgram::from_string(qasm_code))
    ));

    engines.insert("qasm_with_includes", DynamicEngineBuilder::new(
        qasm_engine()
            .program(QasmProgram::from_string(qasm_code))
            .with_virtual_includes(vec![
                ("custom.inc".to_string(), "// Custom gates".to_string())
            ])
    ));

    // Run simulations with different engines
    for (name, engine) in engines {
        let results = sim_dynamic(engine)
            .seed(42)
            .run(100)?;
        println!("   Engine '{}': {} shots completed", name, results.len());
    }

    println!("\n=== Example Complete ===");

    Ok(())
}

// =========================================================================
// Helper Functions
// =========================================================================

/// Example function showing how to create an engine based on file extension
#[allow(dead_code)]
fn create_engine_from_file(path: &str) -> Result<DynamicEngineBuilder, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;

    let builder = if path.ends_with(".qasm") {
        DynamicEngineBuilder::new(qasm_engine().program(QasmProgram::from_string(&content)))
    } else if path.ends_with(".ll") {
        use pecos_programs::QisProgram;
        DynamicEngineBuilder::new(qis_engine().program(QisProgram::from_string(&content)))
    } else if path.ends_with(".hugr") {
        // In real code, you'd parse HUGR here
        use pecos_programs::HugrProgram;
        let hugr_bytes = std::fs::read(path)?;
        DynamicEngineBuilder::new(selene_executable().program(HugrProgram::from_bytes(hugr_bytes)).qubits(2))
    } else {
        return Err("Unknown file type".into());
    };

    Ok(builder)
}

/// Example function showing engine selection from enum
#[allow(dead_code)]
fn create_engine_from_type(
    engine_type: EngineType,
    source: &str,
) -> DynamicEngineBuilder {
    match engine_type {
        EngineType::Qasm => DynamicEngineBuilder::new(qasm_engine().program(QasmProgram::from_string(source))),
        EngineType::Llvm => {
            use pecos_programs::QisProgram;
            DynamicEngineBuilder::new(qis_engine().program(QisProgram::from_string(source)))
        },
        EngineType::Selene => {
            // In real code, you'd parse HUGR from source
            use pecos_programs::HugrProgram;
            DynamicEngineBuilder::new(selene_executable().program(HugrProgram::from_bytes(vec![])).qubits(2))
        }
    }
}