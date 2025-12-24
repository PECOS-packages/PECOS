//! Example demonstrating HUGR to QIS compilation

use pecos_hugr_qis::prelude::*;
use std::path::PathBuf;

fn main() {
    // Example 1: Basic compilation with defaults
    println!("Example 1: Basic compilation");
    let hugr_bytes = b"{}"; // Invalid HUGR for demo
    match compile_hugr_bytes_to_string(hugr_bytes) {
        Ok(_) => println!("Compilation succeeded"),
        Err(e) => println!("Expected error for invalid HUGR: {e}"),
    }

    // Example 2: Advanced compilation with custom configuration
    println!("\nExample 2: Advanced compilation with custom config");
    let config = HugrCompilerConfig {
        name: Some("my_quantum_program".to_string()),
        opt_level: Some(OptimizationLevel::Aggressive),
        target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
        save_hugr: Some(PathBuf::from("debug_output.hugr")),
        ..Default::default()
    };

    let compiler = HugrCompiler::with_config(config);
    match compiler.compile_hugr_bytes_to_string(hugr_bytes) {
        Ok(_) => println!("Compilation succeeded"),
        Err(e) => println!("Expected error for invalid HUGR: {e}"),
    }

    // Example 3: Using CompileArgs directly
    println!("\nExample 3: Direct CompileArgs usage");
    let args = CompileArgs {
        opt_level: OptimizationLevel::None, // Fast compilation
        target_triple: Some("aarch64-apple-darwin".to_string()),
        entry: Some("my_entry_point".to_string()),
        ..Default::default()
    };

    match compile_hugr_bytes_to_string_with_options(hugr_bytes, &args) {
        Ok(_) => println!("Compilation succeeded"),
        Err(e) => println!("Expected error for invalid HUGR: {e}"),
    }

    // Example 4: Compile to bitcode
    println!("\nExample 4: Bitcode compilation");
    match compile_hugr_bytes_to_bitcode(hugr_bytes) {
        Ok(_) => println!("Bitcode compilation succeeded"),
        Err(e) => println!("Expected error for invalid HUGR: {e}"),
    }

    // Example 5: Check HUGR validity
    println!("\nExample 5: HUGR validation");
    match check_hugr(hugr_bytes) {
        Ok(()) => println!("HUGR is valid"),
        Err(e) => println!("HUGR validation failed: {e}"),
    }

    // Example 6: Target machine information
    println!("\nExample 6: Target machine info");
    match get_native_target_machine(OptimizationLevel::Default) {
        Ok(tm) => {
            println!("Native target triple: {}", tm.get_triple());
            println!("CPU: {}", tm.get_cpu());
            println!("Features: {:?}", tm.get_feature_string());
        }
        Err(e) => println!("Failed to get target machine: {e}"),
    }
}
