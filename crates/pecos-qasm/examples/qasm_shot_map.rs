use pecos_engines::{ShotMap, ShotMapDisplayExt};
use pecos_qasm::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run a simple QASM circuit
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];
        creg ancilla[1];

        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];

        measure q -> c;
        measure q[0] -> ancilla[0];
    "#;

    // Run simulation - run_qasm returns ShotVec directly
    let shot_vec = run_qasm(
        qasm,
        20,
        PassThroughNoiseModel::builder(),
        None,
        None,
        Some(42),
    )?;

    // Convert to ShotMap for display and columnar access
    let shot_map: ShotMap = shot_vec.try_as_shot_map()?;

    println!("=== QASM Results ===");
    println!("{}", shot_map.display()); // Default decimal
    println!("\n=== Binary format ===");
    println!("{}", shot_map.display().bitvec_binary());

    println!("\n=== Using ShotMap ===");
    println!("Shots: {}", shot_map.num_shots());
    println!("Registers: {:?}", shot_map.register_names());

    // Extract specific register values using try_* methods
    match shot_map.try_bitvecs("c") {
        Ok(c_values) => {
            println!("\nRegister 'c' has {} measurements", c_values.len());

            // Count unique outcomes
            let mut counts = std::collections::BTreeMap::new();
            for bitvec in &c_values {
                // Convert BitVec to string for counting
                let mut key = String::new();
                for bit in bitvec {
                    key.push(if *bit { '1' } else { '0' });
                }
                *counts.entry(key).or_insert(0) += 1;
            }

            println!("Outcome counts:");
            let mut sorted_outcomes: Vec<_> = counts.iter().collect();
            sorted_outcomes.sort_by_key(|(k, _)| k.as_str());
            for (outcome, count) in sorted_outcomes {
                println!("  {outcome}: {count}");
            }
        }
        Err(e) => println!("Error accessing 'c' register: {e}"),
    }

    // Also demonstrate accessing as decimal values
    match shot_map.try_bits_as_decimal("c") {
        Ok(decimal_values) => {
            println!("\nDecimal values for 'c': {decimal_values:?}");
        }
        Err(e) => println!("Error converting to decimal: {e}"),
    }

    Ok(())
}
