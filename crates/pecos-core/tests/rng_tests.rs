// Tests for RNG functionality in pecos-core

use pecos_core::rng::{RecordingRng, ReplayingRng};
use rand::{RngExt, SeedableRng};
use rand_core::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

// Epsilon value for floating-point comparisons
const EPSILON: f64 = 1e-10;

// Helper function for approximate float comparison
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

#[test]
fn test_replaying_rng_basic_values() {
    // Create a recording RNG with Xoshiro256PlusPlus
    let xoshiro_rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut recording_rng = RecordingRng::new(xoshiro_rng);

    // Generate various types of random values
    let bool_value = recording_rng.random::<bool>();
    let int_value = recording_rng.random_range(1..100);
    let float_value = recording_rng.random::<f64>();

    // Get the recorded values
    let recorded_values = recording_rng.recorded_values().to_vec();

    // Create a ReplayingRng with the recorded values
    let mut replaying_rng = ReplayingRng::from_values(recorded_values);

    // Verify replayed values match
    assert_eq!(bool_value, replaying_rng.random::<bool>());
    assert_eq!(int_value, replaying_rng.random_range(1..100));
    assert!(approx_eq(float_value, replaying_rng.random::<f64>()));
}

#[test]
fn test_replaying_rng_char_generation() {
    // This test specifically targets the char generation issue we identified
    let xoshiro_rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut recording_rng = RecordingRng::new(xoshiro_rng);

    // Generate a random character
    let random_char = recording_rng.random::<char>();

    // Get the recorded values
    let recorded_values = recording_rng.recorded_values().to_vec();

    // Create a ReplayingRng with the recorded values
    let mut replaying_rng = ReplayingRng::from_values(recorded_values);

    // Verify the replayed char matches
    let replay_char = replaying_rng.random::<char>();
    assert_eq!(random_char, replay_char);
}

#[test]
fn test_complex_rng_sequence() {
    // Create a recording RNG with Xoshiro256PlusPlus
    let xoshiro_rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut recording_rng = RecordingRng::new(xoshiro_rng);

    // Generate a complex sequence of random values
    let bool_value1 = recording_rng.random::<bool>();
    let bool_value2 = recording_rng.random::<bool>();
    let int_value = recording_rng.random_range(-50..50);
    let float_value = recording_rng.random::<f64>();
    let float_range_value = recording_rng.random_range(0.0..10.0);
    let char_value = recording_rng.random::<char>();

    // Get the recorded values
    let recorded_values = recording_rng.recorded_values().to_vec();

    // Create a ReplayingRng with the recorded values
    let mut replaying_rng = ReplayingRng::from_values(recorded_values);

    // Verify replayed values match
    assert_eq!(bool_value1, replaying_rng.random::<bool>());
    assert_eq!(bool_value2, replaying_rng.random::<bool>());
    assert_eq!(int_value, replaying_rng.random_range(-50..50));
    assert!(approx_eq(float_value, replaying_rng.random::<f64>()));
    assert!(approx_eq(
        float_range_value,
        replaying_rng.random_range(0.0..10.0)
    ));
    assert_eq!(char_value, replaying_rng.random::<char>());
}

#[test]
fn test_replaying_rng_custom_function() {
    // Function that uses multiple random operations internally
    fn generate_random_data<R: Rng + ?Sized>(rng: &mut R, seed: u32) -> f64 {
        let base = rng.random_range(1.0..10.0);
        let factor = if rng.random::<bool>() { 1.0 } else { -1.0 };
        let noise = rng.random::<f64>() * 0.1;

        base * factor + noise + (f64::from(seed) * 0.1)
    }

    // Create recording RNG and original results
    let mut recording_rng = RecordingRng::new(Xoshiro256PlusPlus::seed_from_u64(123));
    let mut original_results = Vec::new();
    for i in 0..5 {
        original_results.push(generate_random_data(&mut recording_rng, i));
    }

    // Replay with ReplayingRng
    let mut replaying_rng = ReplayingRng::from_values(recording_rng.recorded_values().to_vec());
    let mut replayed_results = Vec::new();
    for i in 0..5 {
        replayed_results.push(generate_random_data(&mut replaying_rng, i));
    }

    // Verify results match
    assert_eq!(original_results, replayed_results);
}

#[test]
fn test_replaying_rng_from_seed() {
    // Create two ReplayingRng instances with the same seed
    let mut rng1 = ReplayingRng::seed_from_u64(42);
    let mut rng2 = ReplayingRng::seed_from_u64(42);

    // Verify both RNGs produce the same sequence
    for _ in 0..10 {
        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }

    for _ in 0..10 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
}

#[test]
fn test_replaying_rng_fill_bytes() {
    // Create a simple ReplayingRng
    let values = vec![1, 2, 3, 4, 5];
    let mut rng = ReplayingRng::from_values(values);

    // Create two buffers and fill them with random bytes
    let mut buffer1 = [0u8; 16];
    let mut buffer2 = [0u8; 16];

    // Fill the first buffer
    rng.fill_bytes(&mut buffer1);

    // Reset position to beginning
    let values = vec![1, 2, 3, 4, 5];
    let mut rng = ReplayingRng::from_values(values);

    // Fill the second buffer
    rng.fill_bytes(&mut buffer2);

    // Verify both buffers are the same since we reset the RNG
    assert_eq!(buffer1, buffer2);
}
