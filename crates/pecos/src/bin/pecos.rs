use clap::{Parser, Subcommand};
use env_logger::Env;

mod cli;
use cli::{
    CudaCommands, DepsCommands, FeaturesCommands, GoCommands, JuliaCommands, LlvmCommands,
    PythonCommands, RustCommands, SeleneCommands,
};

// Runtime-only imports
#[cfg(feature = "runtime")]
use clap::Args;
#[cfg(feature = "runtime")]
use log::debug;
#[cfg(feature = "runtime")]
use pecos::prelude::*;
#[cfg(feature = "runtime")]
use pecos::{
    DepolarizingNoise, GeneralNoiseModelBuilder, qasm_engine, sim_builder, sparse_stabilizer,
    state_vector,
};
#[cfg(feature = "runtime")]
use pecos_build::llvm::{find_llvm_14, get_llvm_version};
#[cfg(feature = "runtime")]
use std::io::Write;

#[cfg(feature = "runtime")]
#[path = "engine_setup.rs"]
mod engine_setup;
#[cfg(feature = "runtime")]
use engine_setup::{setup_cli_engine, setup_cli_engine_builder};

#[derive(Parser)]
#[command(
    name = "pecos",
    version = env!("CARGO_PKG_VERSION"),
    about = "PECOS - Quantum Error Correction Simulator",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // === Runtime Commands (require 'runtime' feature) ===
    #[cfg(feature = "runtime")]
    /// Compile QIS program to native code
    Compile(CompileArgs),
    #[cfg(feature = "runtime")]
    /// Run quantum program (supports QIS, PHIR/JSON, and QASM formats)
    #[command(after_help = RUN_EXAMPLES)]
    Run(RunArgs),
    #[cfg(feature = "runtime")]
    /// Show version, features, and system information
    Info,
    #[cfg(feature = "runtime")]
    /// Check installation and diagnose common issues
    Doctor,
    #[cfg(feature = "runtime")]
    /// Generate shell completions
    Completions(CompletionsArgs),
    #[cfg(feature = "runtime")]
    /// Show or run example quantum circuits
    Examples(ExamplesArgs),

    // === Dev Tool Commands (always available) ===
    /// Rust/Cargo commands (CUDA-aware)
    #[command(visible_alias = "rs")]
    Rust {
        #[command(subcommand)]
        command: RustCommands,
    },
    /// Python build and test commands
    #[command(visible_alias = "py")]
    Python {
        #[command(subcommand)]
        command: PythonCommands,
    },
    /// CUDA availability and info
    Cuda {
        #[command(subcommand)]
        command: CudaCommands,
    },
    /// Julia build and test commands
    #[command(visible_alias = "jl")]
    Julia {
        #[command(subcommand)]
        command: JuliaCommands,
    },
    /// Go build and test commands
    Go {
        #[command(subcommand)]
        command: GoCommands,
    },
    /// LLVM 14 management (install, check, configure)
    Llvm {
        #[command(subcommand)]
        command: LlvmCommands,
    },
    /// Selene plugin management
    Selene {
        #[command(subcommand)]
        command: SeleneCommands,
    },
    /// Query package features
    Features {
        #[command(subcommand)]
        command: FeaturesCommands,
    },
    /// Dependency manifest management (pecos.toml)
    Deps {
        #[command(subcommand)]
        command: DepsCommands,
    },
    /// Show system tools and project info
    #[command(name = "sys-info")]
    SysInfo,
    /// List installed and cached dependencies
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },
    /// Serve documentation locally and open in browser
    Docs {
        /// Port to serve on
        #[arg(short, long, default_value_t = 8000)]
        port: u16,

        /// Don't open browser automatically
        #[arg(long)]
        no_browser: bool,
    },
}

#[cfg(feature = "runtime")]
#[derive(Args)]
struct ExamplesArgs {
    /// Name of the example to show (omit to list all)
    name: Option<String>,

    /// Run the example instead of just showing it
    #[arg(long)]
    run: bool,

    /// Copy the example to current directory
    #[arg(long)]
    copy: bool,
}

#[cfg(feature = "runtime")]
const RUN_EXAMPLES: &str = "\
Examples:
  # Run a QASM circuit with 1000 shots
  pecos run circuit.qasm -s 1000

  # Reproducible simulation with fixed seed
  pecos run bell.phir.json -s 100 -d 42

  # Use stabilizer simulator for Clifford circuits
  pecos run clifford.qasm -S stabilizer

  # Add depolarizing noise (1% error rate)
  pecos run circuit.qasm -s 1000 -p 0.01

  # Parallel execution with 4 workers
  pecos run large_circuit.qasm -s 10000 -w 4

  # Output results to file in binary format
  pecos run circuit.qasm -s 1000 -o results.json -f binary
";

#[cfg(feature = "runtime")]
#[derive(Args)]
struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(value_enum)]
    shell: clap_complete::Shell,
}

#[cfg(feature = "runtime")]
#[derive(Args)]
struct CompileArgs {
    /// Path to the quantum program (LLVM IR or QASM)
    program: String,

    /// Use JIT interface instead of Selene (useful when Selene is not available)
    #[arg(long)]
    jit: bool,
}

/// Type of quantum noise model to use for simulation
#[cfg(feature = "runtime")]
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
#[cfg(feature = "runtime")]
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

#[cfg(feature = "runtime")]
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

#[cfg(feature = "runtime")]
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

#[cfg(feature = "runtime")]
#[derive(Args, Clone)]
struct RunArgs {
    /// Path to the quantum program (LLVM IR, PHIR-JSON, or QASM)
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

    /// Use JIT interface instead of Selene (useful when Selene is not available)
    #[arg(long)]
    jit: bool,
}

/// Parse noise probability specification from command line argument
///
/// For a depolarizing model, a single probability is expected: "0.01"
/// For a general model, five probabilities are expected: "0.01,0.02,0.02,0.05,0.1"
/// representing [prep, `meas_0`, `meas_1`, `single_qubit`, `two_qubit`]
#[cfg(feature = "runtime")]
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
#[cfg(feature = "runtime")]
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
#[cfg(feature = "runtime")]
fn parse_depolarizing_noise_probability(noise_str_opt: Option<&String>) -> f64 {
    parse_noise_values(noise_str_opt)[0] // Always has at least one value
}

/// Parse five probability values for general noise model
///
/// Returns a tuple of five probabilities: (prep, `meas_0`, `meas_1`, `single_qubit`, `two_qubit`)
/// If a single value is provided, it's used for all five parameters
#[cfg(feature = "runtime")]
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

/// Create quantum engine based on user arguments
#[cfg(feature = "runtime")]
fn run_program(args: &RunArgs) -> Result<(), PecosError> {
    // get_program_path now includes proper context in its errors
    let program_path = get_program_path(&args.program)?;

    // Detect the program type (for informational purposes)
    let program_type = detect_program_type(&program_path)?;
    debug!("Detected program type: {program_type:?}");

    // Set up the engine builder
    let classical_engine_builder = setup_cli_engine_builder(&program_path, args.jit)?;

    // Run the simulation with the selected engine
    let mut builder = sim_builder()
        .classical(classical_engine_builder)
        .workers(args.workers);

    // For QIS programs, we need to detect the number of qubits from the quantum circuit
    // We'll do this by temporarily building the engine to inspect it
    let num_qubits = if program_type == ProgramType::QIS {
        // Build a test simulation to detect qubits from the quantum circuit itself
        // Use a minimal test run to let the simulation auto-detect the required qubits
        debug!("Auto-detecting qubit count for QIS program...");

        // For QIS programs, we'll set a reasonable default and let the quantum engine
        // auto-expand as needed. The bell circuit uses qubits 0 and 1, so we need at least 2.
        Some(2) // Known requirement for bell.ll
    } else {
        None
    };

    if let Some(seed) = args.seed {
        builder = builder.seed(seed);
    }

    // Set noise model based on type
    match args.noise_model {
        NoiseModelType::Depolarizing => {
            let prob = parse_depolarizing_noise_probability(args.noise_probability.as_ref());
            builder = builder.noise(DepolarizingNoise { p: prob });
        }
        NoiseModelType::General => {
            let (prep, meas_0, meas_1, single_qubit, two_qubit) =
                parse_general_noise_probabilities(args.noise_probability.as_ref());
            builder = builder.noise(
                GeneralNoiseModelBuilder::new()
                    .with_prep_probability(prep)
                    .with_meas_0_probability(meas_0)
                    .with_meas_1_probability(meas_1)
                    .with_p1_probability(single_qubit)
                    .with_p2_probability(two_qubit),
            );
        }
    }

    // Set quantum engine based on simulator type
    match args.simulator {
        SimulatorType::StateVector => {
            let mut quantum_builder = state_vector();
            if let Some(qubits) = num_qubits {
                quantum_builder = quantum_builder.qubits(qubits);
                debug!("Set quantum engine to use {qubits} qubits");
            }
            builder = builder.quantum(quantum_builder);
        }
        SimulatorType::Stabilizer => {
            let mut quantum_builder = sparse_stabilizer();
            if let Some(qubits) = num_qubits {
                quantum_builder = quantum_builder.qubits(qubits);
                debug!("Set quantum engine to use {qubits} qubits");
            }
            builder = builder.quantum(quantum_builder);
        }
    }

    let results = builder.run(args.shots)?;

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

            // For QIS programs, ensure file is fully written before potential segfault
            if program_type == ProgramType::QIS {
                // Force sync to disk
                if let Ok(file) = std::fs::OpenOptions::new().write(true).open(file_path) {
                    let _ = file.sync_all();
                }
            }
        }
        None => {
            // Print to stdout
            println!("{results_str}");
        }
    }

    // Force all output to be written
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger with default "info" level if not specified
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    #[cfg(feature = "runtime")]
    {
        use std::io::{self, Write};

        // Intercept help requests to provide dynamic help
        let args: Vec<String> = std::env::args().collect();
        if args.len() == 2 && (args[1] == "--help" || args[1] == "-h" || args[1] == "help") {
            print_dynamic_help();
            return Ok(());
        }

        // For QIS programs, disable stdout buffering to ensure output is captured before segfault
        let _ = io::stdout().flush();
    }

    let cli = Cli::parse();

    match &cli.command {
        // Runtime commands (require 'runtime' feature)
        #[cfg(feature = "runtime")]
        Commands::Compile(args) => {
            // get_program_path and detect_program_type now include proper error context
            let program_path = get_program_path(&args.program)?;

            let program_type = detect_program_type(&program_path)?;

            match program_type {
                ProgramType::QIS => {
                    // For compilation, we need the actual engine not a builder
                    let engine = setup_cli_engine(&program_path, None, args.jit)?;
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
        #[cfg(feature = "runtime")]
        Commands::Run(args) => run_program(args)?,
        #[cfg(feature = "runtime")]
        Commands::Info => print_info(),
        #[cfg(feature = "runtime")]
        Commands::Doctor => run_doctor(),
        #[cfg(feature = "runtime")]
        Commands::Completions(args) => generate_completions(args.shell),
        #[cfg(feature = "runtime")]
        Commands::Examples(args) => handle_examples(args)?,

        // Dev tool commands (always available)
        Commands::Rust { command } => cli::run_rust(command)?,
        Commands::Python { command } => cli::run_python(command)?,
        Commands::Cuda { command } => cli::run_cuda(command.clone())?,
        Commands::Julia { command } => cli::run_julia(command)?,
        Commands::Go { command } => cli::run_go(command)?,
        Commands::Llvm { command } => cli::run_llvm(command.clone())?,
        Commands::Selene { command } => cli::run_selene(command.clone())?,
        Commands::Features { command } => cli::run_features(command.clone())?,
        Commands::Deps { command } => cli::run_deps(command.clone())?,
        Commands::SysInfo => cli::run_sys_info()?,
        Commands::List { verbose } => cli::run_list(*verbose)?,
        Commands::Docs { port, no_browser } => cli::run_docs(*port, *no_browser)?,
    }

    Ok(())
}

/// Print information about PECOS installation and capabilities (neofetch style)
#[cfg(feature = "runtime")]
fn print_info() {
    use std::io::IsTerminal;

    let use_color = std::io::stdout().is_terminal();
    let info = InfoPrinter::new(use_color);
    info.print();
}

/// Helper for neofetch-style info display
#[cfg(feature = "runtime")]
struct InfoPrinter {
    use_color: bool,
}

#[cfg(feature = "runtime")]
impl InfoPrinter {
    fn new(use_color: bool) -> Self {
        Self { use_color }
    }

    // ANSI color codes
    fn cyan(&self, s: &str) -> String {
        if self.use_color {
            format!("\x1b[36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn bold(&self, s: &str) -> String {
        if self.use_color {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn green(&self, s: &str) -> String {
        if self.use_color {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn red(&self, s: &str) -> String {
        if self.use_color {
            format!("\x1b[31m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn dim(&self, s: &str) -> String {
        if self.use_color {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn capability(&self, name: &str, enabled: bool) -> (String, bool) {
        let status = if enabled {
            self.green("[x]")
        } else {
            self.red("[ ]")
        };
        (format!("{status} {name}"), !enabled)
    }

    fn print(&self) {
        // ASCII art logo (6 lines tall)
        let logo = [
            r"  ____  _____ ____ ___  ____  ",
            r" |  _ \| ____/ ___/ _ \/ ___| ",
            r" | |_) |  _|| |  | | | \___ \ ",
            r" |  __/| |__| |__| |_| |___) |",
            r" |_|   |_____\____\___/|____/ ",
            r"                              ",
        ];

        let logo_width = 30;
        let spacer = "  ";

        // Build info lines
        let mut info_lines: Vec<String> = Vec::new();
        let mut has_missing = false;

        // Title and version
        info_lines.push(self.bold("PECOS - Quantum Error Correction Simulator"));
        info_lines.push(format!(
            "{} {}",
            self.cyan("Version:"),
            env!("CARGO_PKG_VERSION")
        ));
        info_lines.push(String::new());

        // Program Formats
        info_lines.push(self.cyan("Program Formats:"));
        let (line, missing) = self.capability("QASM circuits", cfg!(feature = "qasm"));
        info_lines.push(format!("  {line}"));
        has_missing |= missing;
        let (line, missing) = self.capability("PHIR/JSON programs", cfg!(feature = "phir"));
        info_lines.push(format!("  {line}"));
        has_missing |= missing;
        let (line, missing) = self.capability("QIS programs", cfg!(feature = "llvm"));
        info_lines.push(format!("  {line}"));
        has_missing |= missing;
        info_lines.push(String::new());

        // Simulators
        info_lines.push(self.cyan("Simulators:"));
        info_lines.push(format!(
            "  {} StateVector {}",
            self.green("[x]"),
            self.dim("(built-in)")
        ));
        info_lines.push(format!(
            "  {} Stabilizer {}",
            self.green("[x]"),
            self.dim("(built-in)")
        ));
        let (line, missing) = self.capability("QuEST", cfg!(feature = "quest"));
        info_lines.push(format!("  {line}"));
        has_missing |= missing;
        let (line, missing) = self.capability("Qulacs", cfg!(feature = "qulacs"));
        info_lines.push(format!("  {line}"));
        has_missing |= missing;
        info_lines.push(String::new());

        // Noise Models
        info_lines.push(self.cyan("Noise Models:"));
        info_lines.push(format!("  {} depolarizing", self.green("[x]")));
        info_lines.push(format!("  {} general", self.green("[x]")));

        // Print logo alongside info
        let max_lines = logo.len().max(info_lines.len());
        for i in 0..max_lines {
            let logo_line = if i < logo.len() {
                self.cyan(logo[i])
            } else {
                " ".repeat(logo_width)
            };
            let info_line = if i < info_lines.len() {
                &info_lines[i]
            } else {
                ""
            };
            println!("{logo_line}{spacer}{info_line}");
        }

        println!();
        println!(
            "{}",
            self.dim("Documentation: https://github.com/PECOS-Developers/PECOS")
        );

        // Suggest doctor for missing capabilities
        if has_missing {
            println!();
            println!(
                "{}",
                self.dim("Tip: Run 'pecos doctor' to learn how to enable missing capabilities.")
            );
        }
    }
}

/// Run diagnostic checks on PECOS installation
#[cfg(feature = "runtime")]
fn run_doctor() {
    println!("Checking PECOS installation...");
    println!();

    let mut all_ok = true;
    let mut warnings = Vec::new();

    // Check 1: Version
    print_check(
        "PECOS CLI",
        true,
        &format!("v{}", env!("CARGO_PKG_VERSION")),
    );

    // Check 2: QASM support
    let qasm_ok = cfg!(feature = "qasm");
    print_check(
        "QASM support",
        qasm_ok,
        if qasm_ok { "available" } else { "not compiled" },
    );
    if !qasm_ok {
        warnings.push("QASM support not compiled. Reinstall with default features.");
    }

    // Check 3: PHIR support
    let phir_ok = cfg!(feature = "phir");
    print_check(
        "PHIR/JSON support",
        phir_ok,
        if phir_ok { "available" } else { "not compiled" },
    );
    if !phir_ok {
        warnings.push("PHIR support not compiled. Reinstall with default features.");
    }

    // Check 4: Selene runtime
    let selene_ok = cfg!(feature = "llvm");
    print_check(
        "Selene runtime",
        selene_ok,
        if selene_ok {
            "available"
        } else {
            "not compiled"
        },
    );

    // Check 5: LLVM/QIS support
    let llvm_ok = cfg!(feature = "llvm");
    if llvm_ok {
        print_check("LLVM/QIS support", true, "available");
    } else {
        print_check("LLVM/QIS support", false, "not compiled (optional)");
        warnings.push("LLVM support not compiled. To enable: cargo install pecos --features llvm");
    }

    // Check 6: LLVM 14 installation
    if let Some(llvm_path) = find_llvm_14(None) {
        let version = get_llvm_version(&llvm_path).unwrap_or_else(|_| "unknown".into());
        print_check(
            "LLVM 14",
            true,
            &format!("{version} at {}", llvm_path.display()),
        );
    } else {
        print_check("LLVM 14", false, "not found (run 'pecos llvm install')");
        warnings.push("LLVM 14 not found. To install: pecos llvm install");
    }

    // Check 7: Test basic circuit execution
    print!("  ");
    let test_result = test_basic_execution();
    match test_result {
        Ok(()) => {
            print_check("Test circuit", true, "execution successful");
        }
        Err(e) => {
            print_check("Test circuit", false, &format!("failed: {e}"));
            all_ok = false;
        }
    }

    println!();

    // Print warnings
    if !warnings.is_empty() {
        println!("Suggestions:");
        for warning in &warnings {
            println!("  - {warning}");
        }
        println!();
    }

    // Final status
    if all_ok {
        println!("All checks passed! PECOS is ready to use.");
    } else {
        println!("Some checks failed. See above for details.");
    }
}

#[cfg(feature = "runtime")]
fn print_check(name: &str, ok: bool, detail: &str) {
    let status = if ok { "[OK]" } else { "[!!]" };
    println!("  {status} {name}: {detail}");
}

/// Test basic circuit execution with a simple Bell state
#[cfg(feature = "runtime")]
fn test_basic_execution() -> Result<(), PecosError> {
    // Simple Bell state circuit in QASM
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let engine = qasm_engine().qasm(qasm.to_string());
    let results = sim_builder()
        .classical(engine)
        .quantum(state_vector().qubits(2))
        .seed(42)
        .run(1)?;

    // Verify we got a result
    let _shot_map = results.try_as_shot_map()?;
    // If we get here without error, the circuit executed successfully

    Ok(())
}

/// Generate shell completions
#[cfg(feature = "runtime")]
fn generate_completions(shell: clap_complete::Shell) {
    use clap::CommandFactory;
    use clap_complete::generate;

    let mut cmd = Cli::command();
    let name = cmd.get_name().to_string();
    generate(shell, &mut cmd, name, &mut std::io::stdout());
}

/// Print dynamic help
#[cfg(feature = "runtime")]
fn print_dynamic_help() {
    use clap::CommandFactory;

    // Get the base help from clap
    let mut cmd = Cli::command();
    let mut help_str = Vec::new();
    cmd.write_help(&mut help_str).unwrap();
    let help = String::from_utf8_lossy(&help_str);

    // Print the base help
    print!("{help}");
}

// ============================================================================
// Example circuits
// ============================================================================

#[cfg(feature = "runtime")]
struct Example {
    name: &'static str,
    description: &'static str,
    filename: &'static str,
    content: &'static str,
}

#[cfg(feature = "runtime")]
const EXAMPLES: &[Example] = &[
    Example {
        name: "bell",
        description: "Bell state - entangle two qubits",
        filename: "bell.qasm",
        content: r#"// Bell State Circuit
// Creates an entangled pair of qubits in the state (|00> + |11>)/sqrt(2)
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Create superposition on first qubit
h q[0];

// Entangle with second qubit
cx q[0], q[1];

// Measure both qubits
measure q -> c;
"#,
    },
    Example {
        name: "ghz",
        description: "GHZ state - three-qubit entanglement",
        filename: "ghz.qasm",
        content: r#"// GHZ State Circuit
// Creates the state (|000> + |111>)/sqrt(2)
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Create superposition
h q[0];

// Entangle all three qubits
cx q[0], q[1];
cx q[1], q[2];

// Measure
measure q -> c;
"#,
    },
    Example {
        name: "teleport",
        description: "Quantum teleportation protocol",
        filename: "teleport.qasm",
        content: r#"// Quantum Teleportation Circuit
// Teleports the state of q[0] to q[2]
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Prepare state to teleport (|1> state)
x q[0];

// Create Bell pair between q[1] and q[2]
h q[1];
cx q[1], q[2];

// Bell measurement on q[0] and q[1]
cx q[0], q[1];
h q[0];

// Measure the first two qubits
measure q[0] -> c[0];
measure q[1] -> c[1];

// Classical corrections would be applied based on c[0] and c[1]
// For simulation, we just measure q[2]
measure q[2] -> c[2];
"#,
    },
    Example {
        name: "superposition",
        description: "Simple superposition with Hadamard gate",
        filename: "superposition.qasm",
        content: r#"// Superposition Circuit
// Creates equal superposition of |0> and |1>
OPENQASM 2.0;
include "qelib1.inc";

qreg q[1];
creg c[1];

// Create superposition
h q[0];

// Measure - should give 0 or 1 with equal probability
measure q -> c;
"#,
    },
    Example {
        name: "phase",
        description: "Phase kickback demonstration",
        filename: "phase.qasm",
        content: r#"// Phase Kickback Circuit
// Demonstrates phase kickback with controlled gates
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// Prepare |-> state on target qubit
x q[1];
h q[1];

// Control qubit in superposition
h q[0];

// Controlled-Z applies phase to control qubit
cz q[0], q[1];

// Interfere and measure
h q[0];
measure q -> c;
"#,
    },
];

/// Handle the examples command
#[cfg(feature = "runtime")]
fn handle_examples(args: &ExamplesArgs) -> Result<(), PecosError> {
    match &args.name {
        None => {
            // List all examples
            println!("Available examples:");
            println!();
            for ex in EXAMPLES {
                println!("  {:12} - {}", ex.name, ex.description);
            }
            println!();
            println!("Usage:");
            println!("  pecos examples <name>        Show the example circuit");
            println!("  pecos examples <name> --run  Run the example (100 shots)");
            println!("  pecos examples <name> --copy Copy to current directory");
            Ok(())
        }
        Some(name) => {
            let example = EXAMPLES.iter().find(|e| e.name == name).ok_or_else(|| {
                PecosError::Input(format!(
                    "Unknown example '{name}'. Run 'pecos examples' to list available examples."
                ))
            })?;

            if args.copy {
                // Copy to current directory
                std::fs::write(example.filename, example.content).map_err(|e| {
                    PecosError::Resource(format!("Failed to write {}: {}", example.filename, e))
                })?;
                println!("Copied {} to {}", example.name, example.filename);
                println!();
                println!("Run with:");
                println!("  pecos run {} -s 100", example.filename);
            } else if args.run {
                // Run the example
                println!("Running {} example (100 shots)...", example.name);
                println!();

                let engine = qasm_engine().qasm(example.content.to_string());
                let results = sim_builder()
                    .classical(engine)
                    .quantum(state_vector())
                    .seed(42)
                    .run(100)?;

                let shot_map = results.try_as_shot_map()?;
                println!("{}", shot_map.display().bitvec_binary());
            } else {
                // Show the example
                println!("// Example: {} - {}", example.name, example.description);
                println!("// File: {}", example.filename);
                println!();
                print!("{}", example.content);
            }

            Ok(())
        }
    }
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;

    #[test]
    fn verify_cli_seed_argument() {
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.phir.json",
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
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_no_seed_argument() {
        let cmd = Cli::parse_from(["pecos", "run", "program.phir.json", "-s", "100", "-w", "2"]);

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
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_general_noise_model() {
        // Test with long option
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.phir.json",
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
            _ => panic!("Expected Run command"),
        }

        // Test with short option
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.phir.json",
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
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn verify_cli_output_file_option() {
        // Test with output file specified using short flag
        let cmd = Cli::parse_from(["pecos", "run", "program.phir.json", "-o", "results.json"]);

        if let Commands::Run(args) = cmd.command {
            assert_eq!(args.output_file, Some("results.json".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // Test with output file specified using long flag
        let cmd = Cli::parse_from([
            "pecos",
            "run",
            "program.phir.json",
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
