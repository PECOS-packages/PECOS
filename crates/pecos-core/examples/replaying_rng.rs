// This example demonstrates how to use RecordingRng with ReplayingRng
// for deterministic record and replay of random operations

use pecos_core::rng::{RecordingRng, ReplayingRng};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

// Epsilon value for floating-point comparisons
const EPSILON: f64 = 1e-10;

// Helper function for approximate float comparison
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

fn main() {
    println!("=== RECORDING PHASE ===");

    // Create a RecordingRng with a Xoshiro256PlusPlus as the underlying RNG
    let xoshiro_rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut recording_rng = RecordingRng::new(xoshiro_rng);

    // Generate various types of random values
    println!("Generating random values:");

    // Booleans
    let bool_val = recording_rng.random::<bool>();
    println!("Boolean: {bool_val}");

    // Integer in range
    let int_val = recording_rng.random_range(-10..10);
    println!("Integer (-10..10): {int_val}");

    // Floating point
    let float_val = recording_rng.random::<f64>();
    println!("Float: {float_val}");

    // Character
    let char_val = recording_rng.random::<char>();
    println!("Character: {char_val}");

    // Get the recorded values
    let recorded_values = recording_rng.recorded_values().to_vec();
    println!("\nRecorded {} raw values", recorded_values.len());

    // === REPLAY PHASE ===
    println!("\n=== REPLAY PHASE ===");

    // Now use ReplayingRng to replay the sequence
    let mut replaying_rng = ReplayingRng::from_values(recorded_values);

    // Replay the exact same sequence
    println!("Replaying random values:");

    // Replay booleans
    let replay_bool = replaying_rng.random::<bool>();
    println!(
        "Boolean: {} (matches: {})",
        replay_bool,
        replay_bool == bool_val
    );

    // Replay integer
    let replay_int = replaying_rng.random_range(-10..10);
    println!(
        "Integer (-10..10): {} (matches: {})",
        replay_int,
        replay_int == int_val
    );

    // Replay float
    let replay_float = replaying_rng.random::<f64>();
    println!(
        "Float: {} (matches: {})",
        replay_float,
        approx_eq(replay_float, float_val)
    );

    // Replay char
    let replay_char = replaying_rng.random::<char>();
    println!(
        "Character: {} (matches: {})",
        replay_char,
        replay_char == char_val
    );

    // Verify all matches
    println!("\n=== VERIFICATION ===");
    assert_eq!(bool_val, replay_bool);
    assert_eq!(int_val, replay_int);
    assert!(approx_eq(float_val, replay_float));
    assert_eq!(char_val, replay_char);

    println!("All values matched successfully!");
    println!("\nReplayingRng pairs well with RecordingRng for creating");
    println!("deterministic tests with reproducible random sequences.");
}
