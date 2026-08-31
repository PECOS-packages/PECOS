// Demonstration of how to:
// 1. Turn on fault history tracking in a simulation
// 2. Rerun a circuit with a specified fault history
// 3. Perturb the fault history and see how the outcome changes
use pecos_core::errors::PecosError;
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

    // Print out relative probabilities for each fault history
    println!("------------------------------------------------------");
    println!("Relative Probabilities of fault histories:");
    for (ind1, history1) in histories.clone().into_iter().enumerate() {
        for (ind2, history2) in histories.clone().into_iter().enumerate() {
            let relative_prob = fault_catalog.fault_histories_probability_ratio(&history1, &history2);
            println!(
                "\tP(history {}) / P(history {}): {:e}",
                ind1, ind2, relative_prob
            );
        }
    }

    // Print out the relative probabilities of each history if the model changed
    println!("------------------------------------------------------");
    println!("Relative Probabilities if error prob changes from 0.1 -> 0.01:");
    let mut mc2 = MonteCarloEngine::builder()
        .with_classical_engine(classical_engine.clone())
        .with_quantum_engine(Box::new(StabilizerEngine::new(num_qubits)))
        .with_depolarizing_noise(p/10.0) // Change the error probability to 0.01
        .fault_history_enabled()
        .build();
    mc2.set_seed(0);
    let fault_catalog2 = mc2.return_fault_catalog()?;
    for (ind1, history1) in histories.clone().into_iter().enumerate() {
        let relative_prob = fault_catalog.fault_catalog_probability_ratio(&fault_catalog2, &history1);
        println!(
            "\tP(history {} | p=0.1) / P(history {} | p=0.01) = {:e}",
            ind1, ind1, relative_prob
        );
    }

    // Now, we can pick one of the samples and rerun it with the same fault history.
    // This is to demonstrate that we have the capability of running a specific seed.
    // Just for completeness, we will change the seed and observe how the outcome changes.
    println!("------------------------------------------------------");
    println!("Rerun with the same fault history but different seed:");
    let mut mc = MonteCarloEngine::builder()
        .with_classical_engine(classical_engine.clone())
        .with_quantum_engine(Box::new(StabilizerEngine::new(num_qubits)))
        .with_depolarizing_noise(p)
        .fault_history_enabled()
        .build();
    mc.set_seed(0); // Changed the seed
    let run2 = mc.run_with_fault_history(&histories[0])?;
    let shots2 = run2.results;
    let histories2 = run2.fault_histories;

    // Print out the fault histories for each shot
    println!("Collected {} shots", shots2.len());
    for per_shot in histories2.clone() {
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

    // Finally, we can perturb the fault history and see how the relative
    // probabilities change. To do this, we will need to set the random number
    // generator up for our fault catalog so that it can propose random flips
    fault_catalog.set_seed(1);
    let new_history = fault_catalog.random_flip(&histories[0]);
    let relative_prob = fault_catalog.fault_histories_probability_ratio(&histories[0], &new_history);
    println!("------------------------------------------------------");
    println!("Flipped the fault history from:");
    for fault in &histories[0] {
        println!(
            "\tFault at site {}: outcome {} ({})",
            fault.site_uid, fault.outcome_index, fault.outcome_label
        );
    }
    println!("to:");
    for fault in &new_history {
        println!(
            "\tFault at site {}: outcome {} ({})",
            fault.site_uid, fault.outcome_index, fault.outcome_label
        );
    }
    println!(
        "P(history 0) / P(perturbed history 0) = {}",
        relative_prob
    );

    Ok(())
}
