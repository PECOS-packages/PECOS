//! Circuit Monte Carlo with logical-error estimation and failing fault histories.
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::engine_system::{ControlEngine, EngineStage};
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::quantum::StabilizerEngine;
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{ClassicalEngine, Engine};
use pecos_fusion_blossom::FusionBlossomDecoder;
use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, record_offset_to_absolute_index};
use pecos_qec::geometry::StabilizerCheck;
use pecos_qec::{
    MemoryBasis, ParityCheckMatrix, ParityCheckMatrixError, SurfaceCode, coloration_memory_circuit,
};
use std::any::Any;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Clone)]
struct FixedCircuitEngine {
    circuit: ByteMessage,
    num_qubits: usize,
    shot: Shot,
}

impl Engine for FixedCircuitEngine {
    type Input = ();
    type Output = Shot;

    fn process(&mut self, (): ()) -> Result<Shot, PecosError> {
        self.get_results()
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        self.shot = Shot::default();
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

    fn handle_measurements(&mut self, message: ByteMessage) -> Result<(), PecosError> {
        let outcomes = message.outcomes()?;
        let mut bits = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            bits.push(outcome as u8);
        }
        self.shot.data.insert("m".into(), Data::Bytes(bits));
        Ok(())
    }

    fn get_results(&self) -> Result<Shot, PecosError> {
        Ok(self.shot.clone())
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
        let commands = self.generate_commands()?;
        Ok(EngineStage::NeedsProcessing(commands))
    }

    fn continue_processing(
        &mut self,
        measurements: ByteMessage,
    ) -> Result<EngineStage<ByteMessage, Shot>, PecosError> {
        self.handle_measurements(measurements)?;
        let completed_shot = std::mem::take(&mut self.shot);
        Ok(EngineStage::Complete(completed_shot))
    }

    fn reset(&mut self) -> Result<(), PecosError> {
        Engine::reset(self)
    }
}

// builds a check matrix out of a list of checks containing qubit indices.
fn build_check_matrix(
    checks: &[StabilizerCheck],
    num_data_qubits: usize,
) -> Result<ParityCheckMatrix, ParityCheckMatrixError> {
    let mut rows = Vec::with_capacity(checks.len());
    for check in checks {
        let mut row = vec![0_u8; num_data_qubits];
        for qubit in check.qubits() {
            row[qubit] = 1;
        }
        rows.push(row);
    }
    ParityCheckMatrix::from_dense(rows)
}

// converts a list of DEM records to the corresponding measurement indices in the circuit.
fn measurement_indices(records: &[i32], num_measurements: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(records.len());
    for &record in records {
        let index = record_offset_to_absolute_index(num_measurements, record)
            .expect("DEM record must reference a circuit measurement");
        indices.push(index);
    }
    indices
}

// Runs a Monte Carlo Simulation of the code specified. (Right now it is geared for surface codes.)
fn sample(
    n_shots: usize,
    p_ref: f64,
    mut histories: impl Write,
) -> Result<usize, Box<dyn std::error::Error>> {
    let distance = 5;
    let rounds = 5;
    let seed = 42;
    let code = SurfaceCode::rotated(distance)?;
    let num_data_qubits = code.num_data_qubits();

    // Convert the stabilizers to check matrices and construct the memory circuit.
    let hx = build_check_matrix(code.x_stabilizers(), num_data_qubits)?;
    let hz = build_check_matrix(code.z_stabilizers(), num_data_qubits)?;
    let memory = coloration_memory_circuit(&hx, &hz, rounds, MemoryBasis::Z)?;
    // construct a dem from the memory circuit
    let dem = DemBuilder::try_from_tick_circuit(&memory, p_ref, p_ref, p_ref, p_ref)?;
    // Construct the Fusion Blossom decoder from the decomposed DEM.
    let decomposed_dem = dem.to_string_decomposed();
    let mut decoder = FusionBlossomDecoder::from_dem(&decomposed_dem)?;

    // One message preserves the complete history; expand batches into individual fault sites.
    let mut circuit = ByteMessage::quantum_operations_builder();
    for gate in memory.iter_gate_instances() {
        let gate_command = gate.to_gate();
        circuit.add_gate_command(&gate_command);
    }
    let num_qubits = num_data_qubits + hx.num_checks() + hz.num_checks();
    let classical_engine = FixedCircuitEngine {
        circuit: circuit.build(),
        num_qubits,
        shot: Shot::default(),
    };
    let quantum_engine = StabilizerEngine::new(num_qubits);

    // Set up the Monte Carlo engine with fault history tracking.
    let mut monte_carlo = MonteCarloEngine::builder()
        .with_classical_engine(Box::new(classical_engine))
        .with_quantum_engine(Box::new(quantum_engine))
        .with_depolarizing_noise(p_ref)
        .fault_history_enabled()
        .with_seed(seed)
        .build();

    // Coloration assigns measurement IDs in execution order; DEM records count from the end.
    let num_measurements = memory.num_measurements();
    let mut detectors = Vec::with_capacity(dem.detectors.len());
    for detector in &dem.detectors {
        let indices = measurement_indices(&detector.records, num_measurements);
        detectors.push((detector.id as usize, indices));
    }
    let mut observables = Vec::with_capacity(dem.observables.len());
    for observable in &dem.observables {
        let indices = measurement_indices(&observable.records, num_measurements);
        observables.push((observable.id, indices));
    }

    // default syndrome vector is all zeroes, it'll be overwritten each shot.
    let mut syndrome = vec![0_u8; dem.num_detectors()];
    // failures starts with an 0-integer and counds logical fails as we go.
    let mut failures = 0;
    writeln!(
        histories,
        "# distance={distance} rounds={rounds} p={p_ref} shots={n_shots} seed={seed}"
    )?;
    writeln!(
        histories,
        "# shot\tpredicted\tobserved\tfaults (site_uid:outcome_index:label)"
    )?;

    let batch_size = 1_000;
    for batch_start in (0..n_shots).step_by(batch_size) {
        // The last batch may contain fewer than 1,000 shots.
        let remaining_shots = n_shots - batch_start;
        let shots_in_batch = remaining_shots.min(batch_size);
        let run = monte_carlo.run_with_fault_tracking(shots_in_batch)?;

        // Each shot and its fault history have the same index in the batch.
        for shot_offset in 0..run.results.shots.len() {
            let shot = &run.results.shots[shot_offset];
            let history = &run.fault_histories[shot_offset];
            let shot_index = batch_start + shot_offset;
            let Data::Bytes(bits) = &shot.data["m"] else {
                return Err("missing measurement record".into());
            };
            // calculate parities of all detectors and observables from the measurement bits.
            for (id, records) in &detectors {
                let mut detector_parity = 0_u8;
                for &measurement_index in records {
                    detector_parity ^= bits[measurement_index];
                }
                syndrome[*id] = detector_parity;
            }
            // package actual measured observable into a u64 for comparison with decoder's prediction.
            let mut observed = 0_u64;
            for (id, records) in &observables {
                let mut observable_parity = 0_u64;
                for &measurement_index in records {
                    observable_parity ^= u64::from(bits[measurement_index]);
                }
                let observable_bit = observable_parity << id;
                observed |= observable_bit;
            }
            // use decoder to get predicted observable from the syndromes.
            let predicted = decoder.decode_to_obs_mask(&syndrome)?;
            // logical failure if predicted and observed logical don't match.
            if predicted != observed {
                failures += 1;
                // write failure history information to the output file.
                write!(histories, "{shot_index}\t{predicted}\t{observed}")?;
                for fault in history.iter() {
                    write!(
                        histories,
                        "\t{}:{}:{}",
                        fault.site_uid(),
                        fault.outcome_index(),
                        fault.outcome_label()
                    )?;
                }
                writeln!(histories)?;
            }
        }
    }
    histories.flush()?;
    Ok(failures)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional arguments: shot count, output path.
    let mut args = std::env::args().skip(1);
    let n_shots = match args.next() {
        Some(argument) => argument.parse::<usize>()?,
        None => 10_000,
    };
    if n_shots == 0 {
        return Err("shot count must be positive".into());
    }
    let p_phys: f64 = match args.next() {
        Some(argument) => argument.parse::<f64>()?,
        None => 0.001,
    };
    let output = match args.next() {
        Some(path) => path,
        None => String::from("failing_fault_histories.txt"),
    };
    let output_file = File::create(&output)?;
    let history_writer = BufWriter::new(output_file);
    let failures = sample(n_shots, p_phys, history_writer)?;
    let logical_error_rate = failures as f64 / n_shots as f64;
    let variance = logical_error_rate * (1.0 - logical_error_rate) / n_shots as f64;
    let standard_error = variance.sqrt();
    println!(
        "Logical error rate: {logical_error_rate:.2e} +/- {standard_error:.2e} ({failures}/{n_shots})"
    );
    println!("Saved {failures} failing fault histories to {output}");
    Ok(())
}

// Original observable u64-packing code, kept for reference:
// let observed = observables.iter().fold(0_u64, |mask, (id, records)| {
//     mask | (u64::from(parity(records)) << id)
// });
