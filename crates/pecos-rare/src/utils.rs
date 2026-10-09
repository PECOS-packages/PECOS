/// This provides some helpful utilities all rare event sampling workflows.
///
/// This is a work-in-progress and only supports a small set of parameters
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::engine_system::{ControlEngine, EngineStage};
use pecos_engines::shot_results::{Data, Shot};
use pecos_engines::{ClassicalEngine, Engine};
use std::any::Any;
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DetectorErrorModel, record_offset_to_absolute_index,
};
use pecos_decoder_core::ObservableDecoder;
use pecos_fusion_blossom::FusionBlossomDecoder;
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::quantum::StabilizerEngine;
use pecos_engines::FaultCatalog;
use pecos_quantum::TickCircuit;
use pecos_random::{PecosRng, time_seed, RngExt};
use indicatif::{MultiProgress,ProgressBar,ProgressStyle};

use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Some tools to build engines for simulations
#[derive(Clone)]
pub(crate) struct FixedCircuitEngine {
    pub(crate) circuit: ByteMessage,
    pub(crate) num_qubits: usize,
    pub(crate) shot: Shot,
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

/// Returns a fixed circuit classical and quantum (stabilizer) engine
/// for a provided tick circuit
fn build_engines(
    tcircuit: &TickCircuit,
    num_qubits: usize,
) -> (FixedCircuitEngine, StabilizerEngine) {
    let mut circuit = ByteMessage::quantum_operations_builder();
    for gate in tcircuit.iter_gate_instances() {
        circuit.add_gate_command(&gate.to_gate());
    }
    let classical_engine = FixedCircuitEngine {
        circuit: circuit.build(),
        num_qubits,
        shot: Shot::default(),
    };
    let quantum_engine = StabilizerEngine::new(num_qubits);
    (classical_engine, quantum_engine)
}

fn measurement_indices(records: &[i32], num_measurements: usize) -> Vec<usize> {
    records
        .iter()
        .map(|&record| {
            record_offset_to_absolute_index(num_measurements, record)
                .expect("DEM record must reference a circuit measurement")
        })
        .collect()
}

fn get_detectors(dem: &DetectorErrorModel, num_measurements: usize) -> Vec<(usize, Vec<usize>)> {
    dem.detectors
        .iter()
        .map(|detector| {
            (
                detector.id as usize,
                measurement_indices(&detector.records, num_measurements),
            )
        })
        .collect()
}

fn get_observables(dem: &DetectorErrorModel, num_measurements: usize) -> Vec<(usize, Vec<usize>)> {
    dem.observables
        .iter()
        .map(|observable| {
            (
                observable.id as usize,
                measurement_indices(&observable.records, num_measurements),
            )
        })
        .collect()
}

// Sets everything up for a simulation
pub fn setup_rare_event_simulation(
    circuit: &TickCircuit,
    p_phys: f64, // Physical Error Rate
    seed: u64, // Random seed for reproducibility
) -> Result<
    (
        FaultCatalog,
        MonteCarloEngine,
        Box<dyn ObservableDecoder>,
        Vec<(usize, Vec<usize>)>,
        Vec<(usize, Vec<usize>)>,
    ),
    PecosError,
> {

    let num_qubits =  circuit.all_qubits().len();
    let dem = DemBuilder::try_from_tick_circuit(circuit, p_phys, p_phys, p_phys, p_phys)
        .map_err(|error| PecosError::Processing(format!("DEM construction failed: {error}")))?;
    let detectors = get_detectors(&dem, circuit.num_measurements());
    let observables = get_observables(&dem, circuit.num_measurements());

    let (classical_engine, quantum_engine) = build_engines(circuit, num_qubits);

    let mut monte_carlo = MonteCarloEngine::builder()
        .with_classical_engine(Box::new(classical_engine))
        .with_quantum_engine(Box::new(quantum_engine))
        .with_depolarizing_noise(p_phys)
        .fault_history_enabled()
        .with_seed(seed)
        .build();

    // Get the fault catalog
    let fault_catalog = monte_carlo.return_fault_catalog()?;

    // Set up the fusion blossom decoder
    let decoder: Box<dyn ObservableDecoder> = Box::new(
        FusionBlossomDecoder::from_dem(&dem.to_string_decomposed())
            .map_err(|error| PecosError::with_context(error, "Decoder construction failed"))?,
    );

    // Return a bunch of stuff
    Ok((fault_catalog, monte_carlo, decoder, detectors, observables))
}

// Simple function to check if a logical failure has ocurred
pub fn is_logical_failure(
    shot: &Shot,
    decoder: &mut dyn ObservableDecoder,
    detectors: &[(usize, Vec<usize>)],
    observables: &[(usize, Vec<usize>)],
) -> Result<bool, PecosError> {

    // Collect the data from the shot
    let Some(Data::Bytes(bits)) = shot.data.get("m") else {
        return Err(PecosError::Processing("missing measurement record".into()));
    };
    
    // Get the syndrome
    let mut syndrome = vec![0_u8; detectors.len()];
    syndrome.fill(0);
    for (id, records) in detectors {
        syndrome[*id] = records
            .iter()
            .fold(0, |parity, &index| parity ^ bits[index]);
    }

    // Figure out the observed value from the measurement outcomes
    let mut observed = 0_u64;
    for (id, records) in observables {
        let parity = records
            .iter()
            .fold(0, |parity, &index| parity ^ bits[index]);
        observed |= u64::from(parity) << id;
    }

    // Check if the decoded syndrome matches the observed value
    let predicted = decoder
        .decode_to_observables(&syndrome)
        .map_err(|error| PecosError::with_context(error, "Decoding failed"))?;
    Ok(predicted != observed)
}

// Tools for loading bar

// runs the rock paper scissors game in the loading bar
fn run_rps(loading: ProgressBar, game_pb: ProgressBar) {
    const SPIN_FRAMES: usize = 15;
    const RESULT_FRAMES: usize = 13;
    const FRAME_TIME: Duration = Duration::from_millis(70);

    let mut rng = PecosRng::seed_from_u64(time_seed());
    let moves = ["🪨 ", "📄", "✂️ "];
    let (mut left_wins, mut right_wins) = (0_u64, 0_u64);

    'rounds: while !loading.is_finished() {
        // Shuffle both players' moves, then lock in the final frame.
        for frame in 0..SPIN_FRAMES {
            if loading.is_finished() {
                break 'rounds;
            }

            let left = rng.random_range(0..3);
            let right = rng.random_range(0..3);

            let status = if frame == SPIN_FRAMES - 1 {
                // Only the final choices count toward the score.
                match (left, right) {
                    (a, b) if a == b => "🤝 Tie!",
                    (0, 2) | (1, 0) | (2, 1) => {
                        left_wins += 1;
                        "⬅️ Left wins!"
                    }
                    _ => {
                        right_wins += 1;
                        "➡️ Right wins!"
                    }
                }
            } else {
                "Choosing..."
            };

            game_pb.set_message(format!(
                "Left {left_wins:>3} - {right_wins:>3} Right | {} vs {} | {status}\n",
                moves[left], moves[right]
            ));
            game_pb.tick();

            thread::sleep(FRAME_TIME);
        }

        // Freeze the matchup briefly so the result is readable.
        for _ in 0..RESULT_FRAMES {
            if loading.is_finished() {
                break 'rounds;
            }
            thread::sleep(FRAME_TIME);
        }
    }

    let winner = match left_wins.cmp(&right_wins) {
        std::cmp::Ordering::Greater => "🏆 Left Player Wins the Tournament!",
        std::cmp::Ordering::Less => "🏆 Right Player Wins the Tournament!",
        std::cmp::Ordering::Equal => "🤝 It's a Dramatic Tie!",
    };

    game_pb.finish_with_message(format!(
        "{winner} Final: {left_wins}-{right_wins}"
    ));
}

// start rock paper scissors loading bar
pub fn start_rps_loading(total_steps: u64) -> (ProgressBar, JoinHandle<()>) {
    let mp = MultiProgress::new();

    let main_pb = mp.add(ProgressBar::new(total_steps));
    main_pb.set_style(
        ProgressStyle::with_template(
            "Loading [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%)",
        )
        .unwrap()
        .progress_chars("=]."),
    );

    let game_pb = mp.add(ProgressBar::new_spinner());
    game_pb.set_style(
        ProgressStyle::with_template("{spinner} {msg}").unwrap(),
    );

    let loading = main_pb.clone();
    let game_thread = thread::spawn(move || run_rps(loading, game_pb));

    (main_pb, game_thread)
}