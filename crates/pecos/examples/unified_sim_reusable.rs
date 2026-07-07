//! Demonstration of the reusable simulation pattern with unified builder
//!
//! This example shows how to build a simulation once and run it multiple times
//! with different shot counts or seeds.

use pecos::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Build once, run multiple times with sim_builder()
    println!("Example 1: Reusable simulation with sim_builder()");

    let qasm = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#,
    );

    // Build the simulation once
    let built_sim = sim_builder()
        .classical(qasm_engine().program(qasm))
        .quantum(sparse_stab())
        .noise(DepolarizingNoise { p: 0.001 })
        .seed(42)
        .build()?;

    // Run multiple times with different shot counts
    println!("  Running with 100 shots...");
    let mut built_sim = built_sim;
    let results1 = built_sim.run(100)?;
    println!("    Got {} results", results1.len());

    println!("  Running with 500 shots...");
    let results2 = built_sim.run(500)?;
    println!("    Got {} results", results2.len());

    println!("  Running with 1000 shots and different seed...");
    // Note: MonteCarloEngine doesn't support changing seed after creation
    let results3 = built_sim.run(1000)?;
    println!("    Got {} results", results3.len());

    // Example 2: Build once, run multiple times with sim()
    println!("\nExample 2: Reusable simulation with sim() auto-selection");

    let qasm2 = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        h q[1];
        h q[2];
        measure q -> c;
    "#,
    );

    // Build with auto-selected engine
    let mut sim2 = sim(qasm2).quantum(sparse_stab()).seed(42).build()?;

    // Run parameter sweep
    println!("  Running parameter sweep:");
    for shots in [10, 100, 1000, 10000] {
        let results = sim2.run(shots)?;
        println!("    {} shots -> {} results", shots, results.len());
    }

    // Example 3: Compare direct run vs build-then-run
    println!("\nExample 3: Performance comparison");

    let qasm3 = Qasm::from_string(
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q -> c;
    "#,
    );

    // Direct run (builds each time)
    let start = Instant::now();
    for _ in 0..5 {
        let _ = sim(qasm3.clone()).shots(100).run()?;
    }
    let direct_time = start.elapsed();
    println!("  Direct run 5 times: {direct_time:?}");

    // Build once, run multiple times
    let start = Instant::now();
    let mut sim3 = sim(qasm3).build()?;
    for _ in 0..5 {
        let _ = sim3.run(100)?;
    }
    let reuse_time = start.elapsed();
    println!("  Build once, run 5 times: {reuse_time:?}");
    println!(
        "  Speedup: {:.2}x",
        direct_time.as_secs_f64() / reuse_time.as_secs_f64()
    );

    Ok(())
}
