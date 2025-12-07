//! Example demonstrating the Quest quantum simulator API with CPU and GPU support
//!
//! This example shows how to use the Quest state vector and density matrix simulators
//! with the PECOS `sim()` API, including CPU and GPU mode selection.

use pecos::prelude::*;
use pecos::{quest_state_vec, sim};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple QASM program that creates a Bell state
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    println!("==== Quest State Vector Simulation (CPU) ====");
    // Use Quest state vector simulator with CPU mode (default)
    let results = sim(program.clone())
        .quantum(quest_state_vec().with_cpu())
        .seed(42)
        .run(100)?;

    println!("Ran 100 shots with Quest state vector (CPU)");
    let shot_map = results.try_as_shot_map()?;
    let measurements = shot_map.try_bits_as_u64("c")?;
    let zeros = measurements.iter().filter(|&&x| x == 0).count();
    let ones = measurements.iter().filter(|&&x| x == 3).count();
    println!("Results: |00⟩: {zeros}, |11⟩: {ones}");

    // Demonstrate GPU mode (only works if compiled with --features gpu)
    #[cfg(feature = "gpu")]
    {
        println!("\n==== Quest State Vector Simulation (GPU) ====");
        let results_gpu = sim(program.clone())
            .quantum(quest_state_vec().with_gpu())
            .seed(42)
            .run(100)?;

        println!("Ran 100 shots with Quest state vector (GPU)");
        let shot_map_gpu = results_gpu.try_as_shot_map()?;
        let measurements_gpu = shot_map_gpu.try_bits_as_u64("c")?;
        let zeros_gpu = measurements_gpu.iter().filter(|&&x| x == 0).count();
        let ones_gpu = measurements_gpu.iter().filter(|&&x| x == 3).count();
        println!("Results: |00⟩: {zeros_gpu}, |11⟩: {ones_gpu}");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!(
            "\nNote: GPU mode not available. Compile with --features gpu to enable GPU acceleration"
        );
    }

    Ok(())
}
