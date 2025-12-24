//! Example: Reusable simulations with the unified API
//!
//! This example demonstrates how to build a simulation once and run it
//! multiple times with different parameters, which is much more efficient
//! than rebuilding for each run.

use std::time::Instant;

// For demonstration, we'll use conceptual examples.
// In real usage, you would use actual engine builders:
// - pecos_qasm::unified_engine_builder::qasm_engine()
// - pecos_qis_sim::engine_builder::qis_engine()
// - pecos_selene_engine::selene_executable()

fn main() {
    println!("This example demonstrates the reusable simulation pattern.\n");

    println!("In real usage, you would create simulations like this:");
    println!("```rust");
    println!("use pecos_qasm::unified_engine_builder::qasm_engine;");
    println!("use pecos_engines::{{DepolarizingNoise, sim_builder}};");
    println!();
    println!("// Build a reusable simulation");
    println!("let sim = sim_builder()");
    println!("    .classical(qasm_engine().qasm(qasm_code))");
    println!("    .seed(42)");
    println!("    .noise(DepolarizingNoise {{{{ p: 0.01 }}}});");
    println!("    .build()?;");
    println!();
    println!("// Run multiple times with different shot counts");
    println!("let results_100 = sim.run(100)?;");
    println!("let results_1000 = sim.run(1000)?;");
    println!("```\n");

    // Example 1: Statistical analysis pattern
    println!("=== Pattern 1: Statistical Analysis ===");
    println!("With a fixed seed, each run produces identical results.");
    println!("This is useful for:");
    println!("- Debugging quantum algorithms");
    println!("- Reproducible research");
    println!("- Regression testing\n");

    // Example 2: Production pattern
    println!("=== Pattern 2: Production Use ===");
    println!("Without a seed, each run produces different results.");
    println!("```rust");
    println!("let sim = sim_builder()");
    println!("    .classical(engine)");
    println!("    .auto_workers()  // Use all CPU cores");
    println!("    .build()?;       // No seed = random");
    println!();
    println!("// Each API request gets different results");
    println!("for request in requests {{{{");
    println!("    let results = sim.run(request.shots)?;");
    println!("}}");
    println!("```\n");

    // Example 3: Controlled variation
    println!("=== Pattern 3: Controlled Variation ===");
    println!("Use run_with_seed() for different but reproducible results:");
    println!("```rust");
    println!("let sim = sim_builder().classical(engine).build()?;");
    println!();
    println!("// Different seed for each experiment");
    println!("for experiment_id in 0..10 {{{{");
    println!("    let results = sim.run_with_seed(1000, Some(42 + experiment_id))?;");
    println!("}}");
    println!("```\n");

    // Example 4: Parameter sweeps
    println!("=== Pattern 4: Parameter Sweeps ===");
    println!("Build multiple simulations with different parameters:");
    println!("```rust");
    println!("let noise_levels = [0.001, 0.005, 0.01, 0.02];");
    println!();
    println!("let simulations: Vec<_> = noise_levels.iter()");
    println!("    .map(|&p| {{{{");
    println!("        sim_builder()");
    println!("            .classical(engine.clone())");
    println!("            .seed(42)  // Same seed for fair comparison");
    println!("            .noise(DepolarizingNoise {{{{ p }}}});");
    println!("            .build()");
    println!("    }})");
    println!("    .collect::<Result<_, _>>()?;");
    println!("```\n");

    // Performance considerations
    println!("=== Performance Tips ===");
    println!("1. Build once, run many times - parsing/compilation happens once");
    println!("2. Use auto_workers() for CPU-bound simulations");
    println!("3. For benchmarking, warm up with a few runs first");
    println!("4. Consider memory usage when storing many simulation results\n");

    // Timing demonstration
    println!("=== Timing Example ===");
    let start = Instant::now();
    println!("Building simulation... (would compile QASM/LLVM here)");
    std::thread::sleep(std::time::Duration::from_millis(10));
    let build_time = start.elapsed();

    println!("Build time: {build_time:?}");
    println!("Now running multiple times without rebuilding:");

    for shots in [100, 1000, 10000] {
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let run_time = start.elapsed();
        println!("  {shots} shots: {run_time:?} (simulated)");
    }
    println!("\nTotal time saved by reusing the built simulation!");
}
