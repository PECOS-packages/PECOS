use pecos_engines::Engine;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::noise::DepolarizingNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{EngineSystem, QuantumSystem};
use std::env;

fn main() {
    // Parse seed from command line if provided
    let args: Vec<String> = env::args().collect();
    let mut seed_option = None;

    for i in 1..args.len() {
        if args[i] == "--seed"
            && i + 1 < args.len()
            && let Ok(seed) = args[i + 1].parse::<u64>()
        {
            seed_option = Some(seed);
            break;
        }
    }

    let circ = ByteMessage::quantum_operations_builder()
        .add_h(&[0])
        .add_cx(&[0], &[1])
        .add_measurements(&[0])
        .add_measurements(&[1])
        .build();

    let quantum = Box::new(StateVecEngine::new(2));

    // Create noise with or without seed
    let mut noise_builder = DepolarizingNoiseModel::builder().with_uniform_probability(0.1);

    if let Some(seed) = seed_option {
        noise_builder = noise_builder.with_seed(seed);
    }

    let noise = Box::new(noise_builder.build());
    let mut system = QuantumSystem::new(noise, quantum);

    // Also set seed on the system if provided
    if let Some(seed) = seed_option {
        system.set_seed(seed);
    }

    print!("[");
    for _ in 0..20 {
        system.reset().expect("failed to reset");
        let results = system
            .process_as_system(circ.clone())
            .expect("failed to process circ");
        let meas = results.outcomes().expect("failed to parse measurements");

        print!("\"");
        for &value in &meas {
            print!("{value}");
        }
        print!("\", ");
    }
    print!("]");

    println!();
}
