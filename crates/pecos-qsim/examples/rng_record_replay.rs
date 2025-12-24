// Example demonstrating how to use RecordingRng and ReplayingRng with StateVec
//
// This example shows how to:
// 1. Create a StateVec simulator with RecordingRng to record random outcomes
// 2. Run a quantum circuit with measurements
// 3. Create a new StateVec with ReplayingRng to deterministically replay the same circuit

use pecos_core::RngManageable;
use pecos_core::rng::{RecordingRng, ReplayingRng};
use pecos_qsim::{CliffordGateable, StateVec};
use pecos_rng::PecosRng;

fn main() {
    println!("=== RNG Recording and Replay Example ===\n");

    // Step 1: Create a simulator with RecordingRng
    let base_rng = PecosRng::seed_from_u64(42);
    let recording_rng = RecordingRng::new(base_rng);
    let mut sim_recording = StateVec::with_rng(2, recording_rng);

    println!("Running circuit with recording RNG:");

    // Create a Bell state
    sim_recording.h(0).cx(0, 1);
    println!("Created Bell state |00⟩ + |11⟩ / √2");

    // Measure both qubits - in a Bell state, results should match
    let result1 = sim_recording.mz(0);
    let result2 = sim_recording.mz(1);

    println!("Measurement outcomes:");
    println!("  Qubit 0: {}", if result1.outcome { "1" } else { "0" });
    println!("  Qubit 1: {}", if result2.outcome { "1" } else { "0" });

    // Get recorded values
    let recording_rng = sim_recording.rng();
    let recorded_values = recording_rng.recorded_values();
    let recorded_bytes = recording_rng.recorded_bytes();

    println!("\nRecorded random values: {recorded_values:?}");
    println!("Recorded random bytes: {recorded_bytes:?}");

    // Step 2: Create a new simulator with ReplayingRng
    println!("\nReplaying the same circuit with ReplayingRng:");

    let replaying_rng =
        ReplayingRng::from_values_and_bytes(recorded_values.to_vec(), recorded_bytes.to_vec());
    let mut sim_replaying = StateVec::with_rng(2, replaying_rng);

    // Run the same circuit
    sim_replaying.h(0).cx(0, 1);
    println!("Created Bell state |00⟩ + |11⟩ / √2");

    // Measure both qubits - should get the same results as before
    let replay_result1 = sim_replaying.mz(0);
    let replay_result2 = sim_replaying.mz(1);

    println!("Replayed measurement outcomes:");
    println!(
        "  Qubit 0: {}",
        if replay_result1.outcome { "1" } else { "0" }
    );
    println!(
        "  Qubit 1: {}",
        if replay_result2.outcome { "1" } else { "0" }
    );

    // Verify results match
    if result1.outcome == replay_result1.outcome && result2.outcome == replay_result2.outcome {
        println!("\nReplay successful! Measurement outcomes match the original run.");
    } else {
        println!("\nReplay failed! Measurement outcomes do not match.");
    }

    println!("\nThis demonstrates how to make quantum simulations with random measurements");
    println!("fully deterministic by recording and replaying the random number sequences.");
}
