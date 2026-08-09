//! Example of using noise models with the unified API
//!
//! This example demonstrates:
//! 1. Direct builder usage with the unified simulation API
//! 2. Different types of noise models
//! 3. Complex noise model configurations

use pecos_engines::noise::{
    BiasedDepolarizingNoiseModel, DepolarizingNoiseModel, GeneralNoiseModel,
};
use pecos_engines::sim_builder;
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Example 1: General noise model with detailed configuration
    println!("Example 1: GeneralNoiseModelBuilder with unified API");
    let idle_model = BTreeMap::from([
        ("X".to_string(), 1.0 / 3.0),
        ("Y".to_string(), 1.0 / 3.0),
        ("Z".to_string(), 1.0 / 3.0),
    ]);
    let general_noise = GeneralNoiseModel::builder()
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_prep(0.001)
        .with_p_meas_0(0.001)
        .with_p_meas_1(0.001)
        .with_p_idle_linear(0.0001, &idle_model)
        .with_idle_after_2q(1.0)
        .with_seed(42);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(general_noise)
        .seed(42)
        .run(1000)?;

    println!("Got {} shots with general noise", results.shots.len());

    // Example 2: Simple depolarizing noise
    println!("\nExample 2: Simple depolarizing noise");
    let depolarizing = DepolarizingNoiseModel::builder().with_uniform_probability(0.001);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(depolarizing)
        .seed(42)
        .run(1000)?;

    println!("Got {} shots with depolarizing noise", results.shots.len());

    // Example 3: Custom depolarizing noise with different rates
    println!("\nExample 3: Custom depolarizing noise");
    let custom_depolarizing = DepolarizingNoiseModel::builder()
        .with_p_prep(0.001)
        .with_p_meas(0.002)
        .with_p1(0.001)
        .with_p2(0.01);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(custom_depolarizing)
        .seed(42)
        .run(1000)?;

    println!(
        "Got {} shots with custom depolarizing noise",
        results.shots.len()
    );

    // Example 4: Biased depolarizing noise
    println!("\nExample 4: Biased depolarizing noise");
    let biased = BiasedDepolarizingNoiseModel::builder().with_uniform_probability(0.001);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(biased)
        .seed(42)
        .workers(4) // Use multiple workers
        .run(1000)?;

    println!(
        "Got {} shots with biased depolarizing noise",
        results.shots.len()
    );

    // Example 5: No noise (ideal simulation)
    println!("\nExample 5: Ideal simulation (no noise)");
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .run(1000)?;

    println!("Got {} shots with no noise", results.shots.len());

    Ok(())
}
