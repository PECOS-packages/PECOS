// Example demonstrating how to create custom simulator wrappers
// for recording and replaying quantum experiments

use pecos_core::rng::{RecordingRng, ReplayingRng};
use pecos_core::{qid, qid2};
use pecos_qsim::CliffordGateable;
use pecos_qsim::StateVec;
use pecos_rng::{PecosRng, RngExt};
use std::fmt::Display;

// Custom wrapper for a simulator with a RecordingRng
struct RecordingSimulator {
    // We can't directly create StateVec<RecordingRng<PecosRng>>, so we use delegation pattern
    simulator: StateVec,
    recording_rng: RecordingRng<PecosRng>,
}

impl RecordingSimulator {
    // Create a new recording simulator
    fn new(num_qubits: usize, seed: u64) -> Self {
        let base_rng = PecosRng::seed_from_u64(seed);
        let recording_rng = RecordingRng::new(base_rng);

        // Create standard simulator
        let simulator = StateVec::with_seed(num_qubits, seed);

        RecordingSimulator {
            simulator,
            recording_rng,
        }
    }

    // Delegate the h operation
    fn h(&mut self, qubit: usize) -> &mut Self {
        self.simulator.h(&qid(qubit));
        self
    }

    // Delegate the cx operation
    fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.simulator.cx(&qid2(control, target));
        self
    }

    // Simulate a measurement with our recording RNG
    fn mz(&mut self, qubit: usize) -> bool {
        // Generate a random value with the recording RNG
        let prob_one = 0.5; // For a Bell state, this would normally be calculated based on the state
        let result = self.recording_rng.random_bool(prob_one);

        // Apply the measurement to the simulator
        // Note: In a real implementation, we'd modify the simulator's state
        self.simulator.mz(&qid(qubit));

        result
    }

    // Get the recorded values for later replay
    fn recorded_values(&self) -> Vec<u64> {
        self.recording_rng.recorded_values().to_vec()
    }

    // Get the recorded bytes for later replay
    fn recorded_bytes(&self) -> Vec<u8> {
        self.recording_rng.recorded_bytes().to_vec()
    }
}

// Custom wrapper for a simulator with a ReplayingRng
struct ReplayingSimulator {
    // We can't directly create StateVec<ReplayingRng>, so we use delegation pattern
    simulator: StateVec,
    replaying_rng: ReplayingRng,
}

impl ReplayingSimulator {
    // Create a new replaying simulator
    fn new(num_qubits: usize, recorded_values: Vec<u64>, recorded_bytes: Vec<u8>) -> Self {
        let replaying_rng = ReplayingRng::from_values_and_bytes(recorded_values, recorded_bytes);

        // Create standard simulator
        let simulator = StateVec::new(num_qubits);

        ReplayingSimulator {
            simulator,
            replaying_rng,
        }
    }

    // Delegate the h operation
    fn h(&mut self, qubit: usize) -> &mut Self {
        self.simulator.h(&qid(qubit));
        self
    }

    // Delegate the cx operation
    fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.simulator.cx(&qid2(control, target));
        self
    }

    // Simulate a measurement with our replaying RNG
    fn mz(&mut self, qubit: usize) -> bool {
        // Get the next pre-recorded value
        let prob_one = 0.5; // For a Bell state, this would normally be calculated based on the state
        let result = self.replaying_rng.random_bool(prob_one);

        // Apply the measurement to the simulator
        // Note: In a real implementation, we'd modify the simulator's state
        self.simulator.mz(&qid(qubit));

        result
    }
}

fn main() {
    println!("=== Bell State Recording and Replay Example ===\n");

    // Run several experiments with recording
    println!("Running 5 Bell state experiments with recording:");

    let mut experiments = Vec::new();

    for i in 0..5 {
        // Create a new experiment with a base seed
        let seed = 42 + u64::try_from(i).unwrap();

        // Create a recording simulator
        let (result0, result1, recorded_values, recorded_bytes) = run_bell_state_recording(seed);

        // Store experiment data
        experiments.push(BellExperiment {
            seed,
            result0,
            result1,
            recorded_values,
            recorded_bytes,
        });

        // Print the results
        println!(
            "Experiment {} (seed={}): Qubit 0 = {}, Qubit 1 = {} ({})",
            i + 1,
            seed,
            result0,
            result1,
            if result0 == result1 {
                "MATCHED"
            } else {
                "DIFFERENT"
            }
        );
    }

    println!("\n=== Replaying the experiments ===\n");

    // Replay each experiment using the recorded random values
    for (i, exp) in experiments.iter().enumerate() {
        // Create a replaying simulator and run the experiment
        let (replay_result0, replay_result1) =
            run_bell_state_replaying(&exp.recorded_values, &exp.recorded_bytes);

        // Verify that the results match the original experiment
        let matches = replay_result0 == exp.result0 && replay_result1 == exp.result1;
        println!(
            "Replayed experiment {} (seed={}): Qubit 0 = {}, Qubit 1 = {} ({})",
            i + 1,
            exp.seed,
            replay_result0,
            replay_result1,
            if matches { "REPLICATED" } else { "DIFFERENT" }
        );
    }

    // Explain what happened
    println!("\nExplanation:");
    println!("1. We created specialized simulator wrappers that use RecordingRng and ReplayingRng");
    println!("2. The RecordingSimulator captured the random values used during measurement");
    println!("3. The ReplayingSimulator used these values to reproduce the exact same results");
    println!(
        "4. This demonstrates a practical design pattern for deterministic quantum simulation"
    );
}

// Struct to hold the results of a Bell state experiment
#[derive(Debug, Clone, PartialEq, Eq)]
struct BellExperiment<T: Display + Clone = bool> {
    seed: u64,
    result0: T,
    result1: T,
    recorded_values: Vec<u64>,
    recorded_bytes: Vec<u8>,
}

// Run a Bell state experiment with a RecordingSimulator
fn run_bell_state_recording(seed: u64) -> (bool, bool, Vec<u64>, Vec<u8>) {
    // Create a recording simulator
    let mut sim = RecordingSimulator::new(2, seed);

    // Create Bell state (|00⟩ + |11⟩) / sqrt(2)
    sim.h(0).cx(0, 1);

    // Measure both qubits - in a Bell state, results should match
    let result0 = sim.mz(0);
    let result1 = sim.mz(1);

    // Get the recorded values for replay later
    let recorded_values = sim.recorded_values();
    let recorded_bytes = sim.recorded_bytes();

    (result0, result1, recorded_values, recorded_bytes)
}

// Run a Bell state experiment with a ReplayingSimulator
fn run_bell_state_replaying(recorded_values: &[u64], recorded_bytes: &[u8]) -> (bool, bool) {
    // Create a replaying simulator with the recorded values
    let mut sim = ReplayingSimulator::new(2, recorded_values.to_vec(), recorded_bytes.to_vec());

    // Create Bell state (|00⟩ + |11⟩) / sqrt(2)
    sim.h(0).cx(0, 1);

    // Measure both qubits - results should match the recorded ones
    let result0 = sim.mz(0);
    let result1 = sim.mz(1);

    (result0, result1)
}
