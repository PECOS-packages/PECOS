use clap::{Args, Parser, Subcommand};
use env_logger::Env;
use pecos::prelude::*;
use std::error::Error;

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
    /// Run quantum program (supports QIR and PHIR/JSON formats)
    Run(RunArgs),
}

#[derive(Args)]
struct CompileArgs {
    /// Path to the quantum program (LLVM IR)
    program: String,
}

#[derive(PartialEq, Eq, Clone, Debug, Default)]
enum NoiseModelType {
    /// Simple depolarizing noise model with uniform error probabilities
    #[default]
    Depolarizing,
    /// General noise model with configurable error probabilities
    General,
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

#[derive(Args, Debug)]
struct RunArgs {
    /// Path to the quantum program (LLVM IR or JSON)
    program: String,

    /// Number of shots for parallel execution
    #[arg(short, long, default_value_t = 1)]
    shots: usize,

    /// Number of parallel workers
    #[arg(short, long, default_value_t = 1)]
    workers: usize,

    /// Type of noise model to use (depolarizing or general)
    #[arg(long = "model", value_parser, default_value = "depolarizing")]
    noise_model: NoiseModelType,

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
}

fn parse_noise_probability(arg: &str) -> Result<String, String> {
    // Check if it's a comma-separated list
    if arg.contains(',') {
        // Split by comma and parse each value
        let probs: Result<Vec<f64>, _> = arg
            .split(',')
            .map(|s| {
                s.trim().parse::<f64>().map_err(|_| {
                    format!(
                        "Invalid probability value '{s}': must be a valid floating point number"
                    )
                })
            })
            .collect();

        // Check if all values are valid probabilities
        let probs = probs?;
        for prob in &probs {
            if !(0.0..=1.0).contains(prob) {
                return Err(format!("Noise probability {prob} must be between 0 and 1"));
            }
        }

        // For general noise model, we expect 5 probabilities
        if probs.len() != 5 && probs.len() != 1 {
            return Err(format!(
                "Expected either 1 probability for depolarizing model or 5 probabilities for general model, got {}",
                probs.len()
            ));
        }

        // Return the original string since it's valid
        Ok(arg.to_string())
    } else {
        // Single probability value
        let prob: f64 = arg
            .parse()
            .map_err(|_| "Must be a valid floating point number")?;

        if !(0.0..=1.0).contains(&prob) {
            return Err("Noise probability must be between 0 and 1".into());
        }

        Ok(arg.to_string())
    }
}

fn run_program(args: &RunArgs) -> Result<(), Box<dyn Error>> {
    let program_path = get_program_path(&args.program)?;
    let classical_engine = setup_engine(&program_path, Some(args.shots.div_ceil(args.workers)))?;

    // Process based on the selected noise model
    match args.noise_model {
        NoiseModelType::Depolarizing => {
            // Single noise probability for depolarizing model
            let prob = if let Some(noise_str) = &args.noise_probability {
                // If it contains commas, take the first value
                if noise_str.contains(',') {
                    noise_str
                        .split(',')
                        .next()
                        .unwrap()
                        .trim()
                        .parse::<f64>()
                        .unwrap_or(0.0)
                } else {
                    noise_str.parse::<f64>().unwrap_or(0.0)
                }
            } else {
                0.0
            };

            // Create a depolarizing noise model
            let mut noise_model = DepolarizingNoiseModel::new_uniform(prob);

            // If a seed is provided, set it on the noise model
            if let Some(s) = args.seed {
                let noise_seed = derive_seed(s, "noise_model");
                noise_model.set_seed(noise_seed)?;
            }

            // Use the generic approach with noise model
            let results = MonteCarloEngine::run_with_noise_model(
                classical_engine,
                Box::new(noise_model),
                args.shots,
                args.workers,
                args.seed,
            )?;

            results.print();
        }
        NoiseModelType::General => {
            // For general model, we need to parse the comma-separated probabilities
            let (prep, meas_0, meas_1, single_qubit, two_qubit) =
                if let Some(noise_str) = &args.noise_probability {
                    if noise_str.contains(',') {
                        // Parse the comma-separated values
                        let probs: Vec<f64> = noise_str
                            .split(',')
                            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                            .collect();

                        // We should already have validated the length in the parser
                        if probs.len() == 5 {
                            (probs[0], probs[1], probs[2], probs[3], probs[4])
                        } else {
                            // Use the first value for all if only one value is provided
                            let p = probs[0];
                            (p, p, p, p, p)
                        }
                    } else {
                        // Single probability value - use for all parameters
                        let p = noise_str.parse::<f64>().unwrap_or(0.0);
                        (p, p, p, p, p)
                    }
                } else {
                    // Default: no noise
                    (0.0, 0.0, 0.0, 0.0, 0.0)
                };

            // Create the general noise model
            let mut noise_model =
                GeneralNoiseModel::new(prep, meas_0, meas_1, single_qubit, two_qubit);

            // If a seed is provided, set it on the noise model
            if let Some(s) = args.seed {
                let noise_seed = derive_seed(s, "noise_model");
                // We can now silence the non-deterministic warning since we've fixed that issue
                noise_model.reset_with_seed(noise_seed).map_err(|e| {
                    Box::<dyn Error>::from(format!("Failed to set noise model seed: {e}"))
                })?;
            }

            // Use the generic function with the general noise model
            let results = MonteCarloEngine::run_with_noise_model(
                classical_engine,
                Box::new(noise_model),
                args.shots,
                args.workers,
                args.seed,
            )?;

            results.print();
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logger with default "info" level if not specified
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Compile(args) => {
            let program_path = get_program_path(&args.program)?;
            match detect_program_type(&program_path)? {
                ProgramType::QIR => {
                    let engine = setup_engine(&program_path, None)?;
                    engine.compile()?;
                }
                ProgramType::PHIR => {
                    println!("PHIR/JSON programs don't require compilation");
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
            }
            Commands::Compile(_) => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_general_noise_model() {
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
            }
            Commands::Compile(_) => panic!("Expected Run command"),
        }
    }
}
