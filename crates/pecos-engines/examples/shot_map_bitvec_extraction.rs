//! Example demonstrating `BitVec` extraction methods in `ShotMap`
//!
//! This example focuses on the various ways to extract `BitVec` data:
//! - `try_bits_as_u64()` - Extract as integers (up to 64 bits)
//! - `try_bits_as_binary()` - Extract as binary strings
//! - `try_bits_as_decimal()` - Extract as decimal strings (arbitrary precision)
//! - `try_bits_as_hex()` - Extract as hexadecimal strings
//!
//! These methods are useful for analyzing quantum measurement results,
//! creating histograms, and converting to different representations.

use pecos_core::errors::PecosError;
use pecos_engines::DataVec;
use pecos_engines::prelude::*;

fn main() -> Result<(), PecosError> {
    // Create a ShotVec simulating quantum measurements
    let mut shot_vec = ShotVec::new();

    // Simulate 8 shots with different measurement patterns
    for i in 0..8 {
        let mut shot = Shot::default();

        // 3-qubit measurement results
        shot.add_register("qubits", i, 3);

        // 8-bit classical register
        shot.add_register("creg", i * 13 + 5, 8);

        // 16-bit ancilla register
        shot.add_register("ancilla", (i * i + 7) * 11, 16);

        shot_vec.shots.push(shot);
    }

    // Convert to ShotMap for columnar analysis
    let shot_map = shot_vec.try_as_shot_map()?;

    println!("=== BitVec Extract Methods Demo ===\n");

    // 1. Extract as u64 values
    println!("1. Extract as u64 integers:");
    println!("---------------------------");
    let qubit_ints = shot_map.try_bits_as_u64("qubits")?;
    let creg_ints = shot_map.try_bits_as_u64("creg")?;
    let ancilla_ints = shot_map.try_bits_as_u64("ancilla")?;

    println!("  Qubits (3-bit):  {qubit_ints:?}");
    println!("  CReg (8-bit):    {creg_ints:?}");
    println!("  Ancilla (16-bit): {ancilla_ints:?}\n");

    // 2. Extract as binary strings
    println!("2. Extract as binary strings:");
    println!("-----------------------------");
    let qubit_binary = shot_map.try_bits_as_binary("qubits")?;
    let creg_binary = shot_map.try_bits_as_binary("creg")?;

    println!("  Qubits:");
    for (i, state) in qubit_binary.iter().enumerate() {
        println!("    Shot {i}: |{state}⟩");
    }

    println!("\n  Classical Register:");
    for (i, value) in creg_binary.iter().enumerate() {
        println!("    Shot {i}: {value}");
    }
    println!();

    // 3. Extract as decimal strings
    println!("3. Extract as decimal strings:");
    println!("------------------------------");
    let qubit_decimal = shot_map.try_bits_as_decimal("qubits")?;
    let ancilla_decimal = shot_map.try_bits_as_decimal("ancilla")?;

    println!("  Qubits:  {qubit_decimal:?}");
    println!("  Ancilla: {ancilla_decimal:?}\n");

    // 4. Extract as hexadecimal strings
    println!("4. Extract as hexadecimal strings:");
    println!("----------------------------------");
    let creg_hex = shot_map.try_bits_as_hex("creg")?;
    let ancilla_hex = shot_map.try_bits_as_hex("ancilla")?;

    println!("  Classical Register (hex):");
    for (i, hex) in creg_hex.iter().enumerate() {
        println!("    Shot {i}: 0x{hex}");
    }

    println!("\n  Ancilla Register (hex):");
    for (i, hex) in ancilla_hex.iter().enumerate() {
        println!("    Shot {i}: 0x{hex}");
    }
    println!();

    // 5. Practical use case: Histogram of measurement outcomes
    println!("5. Measurement outcome histogram:");
    println!("---------------------------------");
    let measurements = shot_map.try_bits_as_u64("qubits")?;
    let mut histogram = std::collections::BTreeMap::new();

    for value in &measurements {
        *histogram.entry(*value).or_insert(0) += 1;
    }

    let mut sorted_outcomes: Vec<_> = histogram.into_iter().collect();
    sorted_outcomes.sort_by_key(|&(value, _)| value);

    for (value, count) in sorted_outcomes {
        let binary = format!("{value:03b}");
        // Precision loss is acceptable for percentage calculation
        #[allow(clippy::cast_precision_loss)]
        let percentage = (f64::from(count) / (measurements.len() as f64)) * 100.0;
        println!("  |{binary}⟩ ({value}): {count} times ({percentage:.1}%)");
    }

    // 6. Show format comparisons for the same data
    println!("\n6. Format comparison for Shot 0:");
    println!("---------------------------------");
    if let Some(DataVec::BitVec(vecs)) = shot_map.get("creg")
        && let Some(bv) = vecs.first()
    {
        println!("  Original BitVec: {bv:?}");
    }
    println!("  As u64:     {}", creg_ints[0]);
    println!("  As binary:  {}", creg_binary[0]);
    println!("  As decimal: {}", shot_map.try_bits_as_decimal("creg")?[0]);
    println!("  As hex:     0x{}", creg_hex[0]);

    Ok(())
}
