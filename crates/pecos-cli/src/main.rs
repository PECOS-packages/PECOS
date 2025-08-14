use clap::{Args, Parser, Subcommand};
use env_logger::Env;
use log::debug;
use pecos::prelude::*;

mod engine_setup;
use engine_setup::setup_cli_engine;

#[derive(Parser)]
#[command(
    name = "pecos",
    version = env!("CARGO_PKG_VERSION"),
    about = "A quantum error correction simulator",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile QIR program to native code
    Compile(CompileArgs),
    /// Run quantum program (supports QIR, PHIR/JSON, and QASM formats)
    Run(RunArgs),
}

#[derive(Args)]
struct CompileArgs {
    /// Path to the quantum program (LLVM IR or QASM)
    program: String,
}

/// Type of quantum noise model to use for simulation
#[derive(PartialEq, Eq, Clone, Debug, Default)]
enum NoiseModelType {
    /// Simple depolarizing noise model with uniform error probabilities
    ///
    /// This model applies the same error probability to all operations
    #[default]
    Depolarizing,
    /// General noise model with configurable error probabilities
    ///
    /// This model allows setting different error probabilities for:
    /// - state preparation
    /// - measurement of |0⟩ state
    /// - measurement of |1⟩ state
    /// - single-qubit gates
    /// - two-qubit gates
    General,
}

/// Type of quantum simulator to use for simulation
#[derive(PartialEq, Eq, Clone, Debug, Default)]
enum SimulatorType {
    /// State vector simulator (full quantum state representation)
    ///
    /// This simulator can handle all quantum gates including arbitrary rotations.
    /// Best for small to medium circuits with non-Clifford gates.
    #[default]
    StateVector,
    /// Stabilizer simulator (Clifford circuit optimization)
    ///
    /// This simulator is optimized for Clifford circuits and can efficiently
    /// simulate larger qubit counts for circuits limited to Clifford gates
    /// (H, S, CNOT, Pauli gates, etc.)
    Stabilizer,
}

impl std::str::FromStr for NoiseModelType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "depolarizing" | "dep" => Ok(NoiseModelType::Depolarizing),
            "general" | "gen" => Ok(NoiseModelType::General),
            _ => Err(format!(
                "Unknown noise model type: {s}. Valid options are 'depolarizing' (dep) or 'general' (gen)"
            )),
        }
    }
}

impl std::str::FromStr for SimulatorType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "statevector" | "state" | "sv" | "full" => Ok(SimulatorType::StateVector),
            "stabilizer" | "stab" | "clifford" => Ok(SimulatorType::Stabilizer),
            _ => Err(format!(
                "Unknown simulator type: {s}. Valid options are 'statevector' (sv, state, full) or 'stabilizer' (stab, clifford)"
            )),
        }
    }
}

#[derive(Args, Clone)]
struct RunArgs {
    /// Path to the quantum program (LLVM IR, JSON, or QASM)
    program: String,

    /// Number of shots for parallel execution
    #[arg(short, long, default_value_t = 1)]
    shots: usize,

    /// Number of parallel workers
    #[arg(short, long, default_value_t = 1)]
    workers: usize,

    /// Type of noise model to use (depolarizing or general)
    #[arg(
        short = 'm',
        long = "model",
        value_parser,
        default_value = "depolarizing"
    )]
    noise_model: NoiseModelType,

    /// Type of quantum simulator to use (statevector or stabilizer)
    /// - statevector: Full quantum state simulator (handles all gates, default)
    /// - stabilizer: Clifford circuit simulator (faster for Clifford circuits)
    #[arg(short = 'S', long = "sim", value_parser, default_value = "statevector")]
    simulator: SimulatorType,

    /// Noise probability (between 0 and 1)
    /// For depolarizing model: uniform error probability
    /// For general model: comma-separated probabilities in order:
    /// `prep,meas_0,meas_1,single_qubit,two_qubit`
    /// Example: --noise 0.01,0.02,0.02,0.05,0.1
    #[arg(short = 'p', long = "noise", value_parser = parse_noise_probability)]
    noise_probability: Option<String>,

    /// Seed for random number generation (for reproducible results)
    #[arg(short = 'd', long)]
    seed: Option<u64>,

    /// Output file path to write results to
    /// If not specified, results will be printed to stdout
    #[arg(short = 'o', long = "output")]
    output_file: Option<String>,

    /// Format for displaying `BitVec` results (decimal, binary, hex)
    /// - decimal: Display as decimal numbers (default)
    /// - binary: Display as binary strings
    /// - hex: Display as hexadecimal strings
    #[arg(short = 'f', long = "format", default_value = "decimal")]
    display_format: String,
}

/// Parse noise probability specification from command line argument
///
/// For a depolarizing model, a single probability is expected: "0.01"
/// For a general model, five probabilities are expected: "0.01,0.02,0.02,0.05,0.1"
/// representing [prep, `meas_0`, `meas_1`, `single_qubit`, `two_qubit`]
fn parse_noise_probability(arg: &str) -> Result<String, String> {
    // Split string into values (either a single value or comma-separated list)
    let values: Vec<&str> = if arg.contains(',') {
        arg.split(',').collect()
    } else {
        vec![arg]
    };

    // Check number of values
    if values.len() != 1 && values.len() != 5 {
        return Err(format!(
            "Expected 1 or 5 probabilities, got {}",
            values.len()
        ));
    }

    // Validate each probability value
    for s in &values {
        // Parse and validate numeric value
        let prob = s
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("Invalid value '{s}': not a valid number"))?;

        // Check value range
        if !(0.0..=1.0).contains(&prob) {
            return Err(format!("Probability {prob} must be between 0 and 1"));
        }
    }

    Ok(arg.to_string())
}

/// Extract probability values from noise specification string
///
/// Handles both single value and comma-separated formats, with safe defaults
fn parse_noise_values(noise_str_opt: Option<&String>) -> Vec<f64> {
    // Default to 0.0 if no string provided
    let Some(noise_str) = noise_str_opt else {
        return vec![0.0];
    };

    // Parse either comma-separated or single value
    if noise_str.contains(',') {
        noise_str
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect()
    } else {
        vec![noise_str.parse::<f64>().unwrap_or(0.0)]
    }
}

/// Parse a single probability value for depolarizing noise model
///
/// Takes the first probability value if multiple are provided
fn parse_depolarizing_noise_probability(noise_str_opt: Option<&String>) -> f64 {
    parse_noise_values(noise_str_opt)[0] // Always has at least one value
}

/// Parse five probability values for general noise model
///
/// Returns a tuple of five probabilities: (prep, `meas_0`, `meas_1`, `single_qubit`, `two_qubit`)
/// If a single value is provided, it's used for all five parameters
fn parse_general_noise_probabilities(noise_str_opt: Option<&String>) -> (f64, f64, f64, f64, f64) {
    let probs = parse_noise_values(noise_str_opt);

    if probs.len() == 5 {
        (probs[0], probs[1], probs[2], probs[3], probs[4])
    } else {
        // Use the first value for all parameters
        let p = probs[0];
        (p, p, p, p, p)
    }
}

fn run_program(args: &RunArgs) -> Result<(), PecosError> {
    // get_program_path now includes proper context in its errors
    let program_path = get_program_path(&args.program)?;

    // Detect the program type (for informational purposes)
    let program_type = detect_program_type(&program_path)?;
    debug!("Detected program type: {program_type:?}");

    // Set up the engine
    let classical_engine =
        setup_cli_engine(&program_path, Some(args.shots.div_ceil(args.workers)))?;

    // Create the appropriate noise model based on user selection
    let noise_model: Box<dyn NoiseModel> = match args.noise_model {
        NoiseModelType::Depolarizing => {
            // Create a depolarizing noise model with single probability
            let prob = parse_depolarizing_noise_probability(args.noise_probability.as_ref());
            let mut model = DepolarizingNoiseModel::new_uniform(prob);

            // Set seed if provided
            if let Some(s) = args.seed {
                let noise_seed = derive_seed(s, "noise_model");
                model.set_seed(noise_seed)?;
            }

            Box::new(model)
        }
        NoiseModelType::General => {
            // Create a general noise model with five probabilities
            let (prep, meas_0, meas_1, single_qubit, two_qubit) =
                parse_general_noise_probabilities(args.noise_probability.as_ref());
            let mut builder = GeneralNoiseModel::builder()
                .with_prep_probability(prep)
                .with_meas_0_probability(meas_0)
                .with_meas_1_probability(meas_1)
                .with_p1_probability(single_qubit)
                .with_p2_probability(two_qubit);

            // Set seed if provided
            if let Some(s) = args.seed {
                let noise_seed = derive_seed(s, "noise_model");
                builder = builder.with_seed(noise_seed);
            }

            Box::new(builder.build())
        }
    };

    // Create the appropriate quantum engine based on user selection
    let quantum_engine: Option<Box<dyn QuantumEngine>> = match args.simulator {
        SimulatorType::StateVector => {
            // Use StateVecEngine - full quantum state simulator
            let num_qubits = classical_engine.num_qubits();
            let engine = if let Some(seed) = args.seed {
                let engine_seed = derive_seed(seed, "quantum_engine");
                Box::new(StateVecEngine::with_seed(num_qubits, engine_seed))
            } else {
                Box::new(StateVecEngine::new(num_qubits))
            };
            Some(engine)
        }
        SimulatorType::Stabilizer => {
            // Use SparseStabEngine - Clifford circuit optimizer
            let num_qubits = classical_engine.num_qubits();
            let engine = if let Some(seed) = args.seed {
                let engine_seed = derive_seed(seed, "quantum_engine");
                Box::new(SparseStabEngine::with_seed(num_qubits, engine_seed))
            } else {
                Box::new(SparseStabEngine::new(num_qubits))
            };
            Some(engine)
        }
    };

    // Run the simulation with the selected engine and noise model
    let results = run_sim(
        classical_engine,
        args.shots,
        args.seed,
        Some(args.workers),
        Some(noise_model),
        quantum_engine,
    )?;

    // Convert to ShotMap for better display formatting
    let shot_map = results.try_as_shot_map()?;

    // Format the results using the new display system with the selected format
    let results_str = match args.display_format.to_lowercase().as_str() {
        "binary" | "bin" => format!("{}", shot_map.display().bitvec_binary()),
        "hexadecimal" | "hex" => format!("{}", shot_map.display().bitvec_hex()),
        "decimal" | "dec" => format!("{}", shot_map.display().bitvec_decimal()),
        _ => {
            eprintln!(
                "Warning: Unknown display format '{}', using decimal",
                args.display_format
            );
            format!("{}", shot_map.display().bitvec_decimal())
        }
    };

    // Either write to the specified output file or print to stdout
    match &args.output_file {
        Some(file_path) => {
            // Ensure parent directory exists
            if let Some(parent) = std::path::Path::new(file_path).parent()
                && !parent.exists()
            {
                std::fs::create_dir_all(parent).map_err(|e| {
                    PecosError::Resource(format!("Failed to create directory: {e}"))
                })?;
            }

            // Write results to file
            std::fs::write(file_path, results_str)
                .map_err(|e| PecosError::Resource(format!("Failed to write output file: {e}")))?;
            println!("Results written to {file_path}");
        }
        None => {
            // Print results to stdout
            println!("{results_str}");
        }
    }

    Ok(())
}

fn main() -> Result<(), PecosError> {
    // Initialize logger with default "info" level if not specified
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Compile(args) => {
            // get_program_path and detect_program_type now include proper error context
            let program_path = get_program_path(&args.program)?;

            let program_type = detect_program_type(&program_path)?;

            match program_type {
                ProgramType::QIR => {
                    let engine = setup_cli_engine(&program_path, None)?;
                    // The compile method should already return a properly formatted PecosError::Compilation
                    engine.compile()?;
                }
                ProgramType::PHIR => {
                    println!("PHIR/JSON programs don't require compilation");
                }
                ProgramType::QASM => {
                    println!("QASM programs don't require compilation");
                }
            }
        }
        Commands::Run(args) => run_program(args)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cli_seed_argument() {
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.json",
            "-d",
            "42",
            "-s",
            "100",
            "-w",
            "2",
        ]);

        match cmd.command {
            Commands::Run(args) => {
                assert_eq!(args.seed, Some(42));
                assert_eq!(args.shots, 100);
                assert_eq!(args.workers, 2);
                assert_eq!(args.noise_model, NoiseModelType::Depolarizing); // Default
                assert_eq!(args.simulator, SimulatorType::StateVector); // Default
                assert_eq!(args.output_file, None); // Default
                assert_eq!(args.display_format, "decimal".to_string()); // Default
            }
            Commands::Compile(_) => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_no_seed_argument() {
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "-s", "100", "-w", "2"]);

        match cmd.command {
            Commands::Run(args) => {
                assert_eq!(args.seed, None);
                assert_eq!(args.shots, 100);
                assert_eq!(args.workers, 2);
                assert_eq!(args.noise_model, NoiseModelType::Depolarizing); // Default
                assert_eq!(args.simulator, SimulatorType::StateVector); // Default
                assert_eq!(args.output_file, None); // Default
                assert_eq!(args.display_format, "decimal".to_string()); // Default
            }
            Commands::Compile(_) => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_general_noise_model() {
        // Test with long option
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.json",
            "--model",
            "general",
            "-p",
            "0.01,0.02,0.03,0.04,0.05",
            "-d",
            "42",
        ]);

        match cmd.command {
            Commands::Run(args) => {
                assert_eq!(args.seed, Some(42));
                assert_eq!(args.noise_model, NoiseModelType::General);
                assert_eq!(
                    args.noise_probability,
                    Some("0.01,0.02,0.03,0.04,0.05".to_string())
                );
                assert_eq!(args.output_file, None); // Default
            }
            Commands::Compile(_) => panic!("Expected Run command"),
        }

        // Test with short option
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.json",
            "-m",
            "general",
            "-p",
            "0.01,0.02,0.03,0.04,0.05",
            "-d",
            "42",
        ]);

        match cmd.command {
            Commands::Run(args) => {
                assert_eq!(args.seed, Some(42));
                assert_eq!(args.noise_model, NoiseModelType::General);
                assert_eq!(
                    args.noise_probability,
                    Some("0.01,0.02,0.03,0.04,0.05".to_string())
                );
            }
            Commands::Compile(_) => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_output_file_option() {
        // Test with output file specified using short flag
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "-o", "results.json"]);

        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.output_file, Some("results.json".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // Test with output file specified using long flag
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.json",
            "--output",
            "path/to/results.json",
        ]);

        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.output_file, Some("path/to/results.json".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn verify_cli_simulator_options() {
        // Test with statevector simulator (explicitly specified)
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "-S", "statevector"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.simulator, SimulatorType::StateVector);
        } else {
            panic!("Expected Run command");
        }

        // Test with stabilizer simulator
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "-S", "stabilizer"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.simulator, SimulatorType::Stabilizer);
        } else {
            panic!("Expected Run command");
        }

        // Test with aliases
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "--sim", "stab"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.simulator, SimulatorType::Stabilizer);
        } else {
            panic!("Expected Run command");
        }

        let cmd = Cli::parse_from(["pecos", "run", "program.json", "--sim", "sv"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.simulator, SimulatorType::StateVector);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn verify_cli_display_format_options() {
        // Test with binary format
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "-f", "binary"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.display_format, "binary");
        } else {
            panic!("Expected Run command");
        }

        // Test with hex format
        let cmd = Cli::parse_from(["pecos", "run", "program.json", "--format", "hex"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.display_format, "hex");
        } else {
            panic!("Expected Run command");
        }

        // Test default format
        let cmd = Cli::parse_from(["pecos", "run", "program.json"]);
        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.display_format, "decimal");
        } else {
            panic!("Expected Run command");
        }
    }
}
