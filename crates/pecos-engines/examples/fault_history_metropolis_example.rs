// Demonstration of how to build a Metropolis stepper on top of a fault history
use pecos_core::errors::PecosError;
use pecos_random::PecosRng;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::engine_system::{ControlEngine, EngineStage};
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::quantum::StabilizerEngine;
use pecos_engines::shot_results::Shot;
use pecos_engines::{ClassicalControlEngine, ClassicalEngine, Engine};
use std::any::Any;

// If I did everything right, this should be a d=5 surface code syndrome extraction circuit
// If not, it is at least a good example
fn sec_circuit() -> ByteMessage {
    let mut builder = ByteMessage::quantum_operations_builder();

    let distance = 5;
    let num_ancilla_qubits = distance * distance - 1;
    let num_data_qubits = distance * distance;
    let num_qubits = num_ancilla_qubits + num_data_qubits;
    let num_x_stabilizers = num_ancilla_qubits / 2;
    let num_z_stabilizers = num_ancilla_qubits / 2;

    // Prepare the data qubits in the |0> state
    for qubit in 0..num_data_qubits {
        builder.pz(&[qubit]);
    }

    // Prepare the ancilla qubits in the |0> state
    for qubit in num_data_qubits..num_qubits {
        builder.pz(&[qubit]);
    }
    // Apply Hadamard gates for the X stabilizer ancilla qubits
    for qubit in num_data_qubits..num_data_qubits + num_x_stabilizers {
        builder.h(&[qubit]);
    }
    // Add all the CNOTs (done by hand for now...)
    for pair in [
        (27, 1),
        (28, 3),
        (29, 5),
        (30, 7),
        (31, 11),
        (32, 13),
        (33, 15),
        (34, 17),
        (35, 21),
        (36, 23),
        (0, 37),
        (2, 38),
        (4, 39),
        (6, 41),
        (8, 42),
        (10, 43),
        (12, 44),
        (14, 45),
        (16, 47),
        (18, 48),
        (27, 2),
        (28, 4),
        (29, 6),
        (30, 8),
        (31, 12),
        (32, 14),
        (33, 16),
        (34, 18),
        (35, 22),
        (36, 24),
        (1, 37),
        (3, 38),
        (5, 40),
        (7, 41),
        (9, 42),
        (11, 43),
        (13, 44),
        (15, 46),
        (17, 47),
        (19, 48),
        (25, 0),
        (26, 2),
        (27, 6),
        (28, 8),
        (29, 10),
        (30, 12),
        (31, 16),
        (32, 18),
        (33, 20),
        (34, 22),
        (5, 37),
        (7, 38),
        (9, 39),
        (11, 41),
        (13, 42),
        (15, 43),
        (17, 44),
        (19, 45),
        (21, 47),
        (23, 48),
        (25, 1),
        (26, 3),
        (27, 7),
        (28, 9),
        (29, 11),
        (30, 13),
        (31, 17),
        (32, 19),
        (33, 21),
        (34, 23),
        (6, 37),
        (8, 38),
        (10, 40),
        (12, 41),
        (14, 42),
        (16, 43),
        (18, 44),
        (20, 46),
        (22, 47),
        (24, 48),
    ] {
        builder.cx(&[pair]);
    }
    for qubit in num_data_qubits..num_data_qubits + num_x_stabilizers {
        builder.h(&[qubit]);
    }
    // Measure the ancilla qubits
    for qubit in num_data_qubits..num_qubits {
        builder.mz(&[qubit]);
    }
    // Measure the data qubits
    for qubit in 0..num_data_qubits {
        builder.mz(&[qubit]);
    }
    builder.build()
}

/// A classical control engine that replays one fixed `ByteMessage` circuit.
///
/// `MonteCarloEngine` requires a full measurement round trip: it sends the
/// circuit out via `start`, and the quantum engine's measurement results come
/// back through `continue_processing`, which is where they must be captured
/// into the returned `Shot` (otherwise every shot reports empty results).
#[derive(Clone)]
struct FixedCircuitEngine {
    circuit: ByteMessage,
    num_qubits: usize,
}

impl FixedCircuitEngine {
    fn new(circuit: ByteMessage, num_qubits: usize) -> Self {
        Self {
            circuit,
            num_qubits,
        }
    }
}

impl Engine for FixedCircuitEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, (): Self::Input) -> Result<Self::Output, PecosError> {
        Ok(Shot::default())
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}

impl ClassicalEngine for FixedCircuitEngine {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn generate_commands(&mut self) -> Result<ByteMessage, PecosError> {
        Ok(self.circuit.clone())
    }

    fn handle_measurements(&mut self, _message: ByteMessage) -> Result<(), PecosError> {
        Ok(())
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        Ok(Shot::default())
    }

    fn compile(&self) -> Result<(), PecosError> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl ControlEngine for FixedCircuitEngine {
    type Input = ();
    type Output = Shot;
    type EngineInput = ByteMessage;
    type EngineOutput = ByteMessage;

    fn start(&mut self, (): ()) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        Ok(EngineStage::NeedsProcessing(self.circuit.clone()))
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        // Pack the measured bits into a single register so outcomes are
        // actually captured instead of discarded.
        let outcomes = measurements.outcomes()?;
        let mut shot = Shot::default();
        let value = outcomes
            .iter()
            .enumerate()
            .fold(0u64, |acc, (i, &bit)| acc | (u64::from(bit) << i));
        shot.add_register_u64("m", value, outcomes.len());
        Ok(EngineStage::Complete(shot))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Ok(())
    }
}

fn main() -> Result<(), PecosError> {
    // Create a d=5 surface code syndrome extraction circuit
    let num_qubits = 49;
    let p = 0.1; // Depolarizing noise probability
    let circ = sec_circuit();

    let classical_engine: Box<dyn ClassicalControlEngine> =
        Box::new(FixedCircuitEngine::new(circ, num_qubits));

    // Set up the Monte Carlo engine with a stabilizer quantum engine and depolarizing noise
    let mut mc = MonteCarloEngine::builder()
        .with_classical_engine(classical_engine.clone())
        .with_quantum_engine(Box::new(StabilizerEngine::new(num_qubits)))
        .with_depolarizing_noise(p)
        .fault_history_enabled()
        .build();
    mc.set_seed(0);

    // Catalog all of the available faults in the circuit
    let mut fault_catalog = mc.return_fault_catalog()?;
    println!("Fault catalog: {} fault sites", fault_catalog.sites.len());
    println!("Fault catalog details:");
    for fault in &fault_catalog.sites {
        println!(
            "\tFault at gate {} ({}) on qubits {:?} with fault id {}",
            fault.gate_index, fault.gate_type, fault.qubits, fault.uid
        );
    }

    // Run the original sampling
    let run = mc.run(10)?;
    let shots = run.results;
    let histories = run.fault_histories;

    // Print out the fault histories for each shot
    println!("------------------------------------------------------");
    println!("Collected {} shots:", shots.len());
    for per_shot in histories.clone() {
        let probability = fault_catalog.fault_history_probability(&per_shot);
        println!("\tNew shot: probability {:e}, with {} faults", probability, per_shot.len());
        for fault in per_shot {
            // Print out all of the faults that happened
            println!(
                "\t\tFault at site {}: outcome {} ({})",
                fault.site_uid, fault.outcome_index, fault.outcome_label
            );
        }
    }

    struct MetropolisStepper {rng: PecosRng }

    struct MetropolisStep<State> {
        state: State,
        accepted: bool,
        acceptance_probability: f64,
    }

    // metropolis stepper object
    // implements `step` method which takes two states and performs a metropolis step,
    // returning a metropolis step that has been accepted with probability `min(1,acceptance_ratio)`,
    // and updating the state of the metropolis step based on if the step was accepted or not.
    impl MetropolisStepper {
        #[must_use]
        pub fn step<State>(
            &mut self,
            current: State,
            proposal: State,
            acceptance_ratio: f64)
            -> MetropolisStep<State> {
            assert!(
                acceptance_ratio >= 0.0,
                "Acceptance ratio must be a non-negative value! got {}", acceptance_ratio
            );

            // Metropolis-hastings acceptance probability
            // assuming that the hastings correction comes in `acceptance_ratio` if needed.
            let acceptance_probability: f64 = acceptance_ratio.min(1.0);
            // accept the new state with probability `acceptance_probability`
            let accepted: bool = acceptance_probability == 1.0 || self.rng.next_f64() < acceptance_probability;
            // return the updated state
            MetropolisStep {
                state: if accepted {proposal} else {current},
                accepted,
                acceptance_probability,
            }
        }

    }

    // Lets create a metropolis stepper object
    let mut stepper = MetropolisStepper {
        rng: PecosRng::seed_from_u64(0),
    };

    // pick the optimal seed... 42.
    fault_catalog.set_seed(42);

    // Starting with `histories[0]` as our `current` state, generate 1,000,000 `proposal` states
    // and perform metropolis steps to determine if we should step from `current` to `proposal`.
    let mut current = histories[0].clone();

    for _i in 1..1_000 {
        let (proposal, correction) = fault_catalog.random_flip_hastings_correction(&current);
        let ratio: f64 = correction * fault_catalog.fault_histories_probability_ratio(&proposal, &current);
        let result = stepper.step(current.clone(), proposal.clone(), ratio);
        current = result.state;
        if result.accepted {
            let probability = fault_catalog.fault_history_probability(&current);
            println!("Proposal ratio: {}", ratio);
            println!("Hastings correction: {}", correction);
            println!("new state probability: {:e}", probability);
        }
    }

    Ok(())
}
