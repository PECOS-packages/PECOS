//! Zluppy CLI - A Zig/SLR/NASA Power of 10 quantum programming language.
//!
//! ## Usage
//!
//! ```bash
//! # Initialize a new project
//! zlup init my-project
//!
//! # Build project using zlup.toml
//! zlup build
//!
//! # Compile to SLR-AST JSON (Python/PECOS bridge)
//! zlup compile program.zlp --target slr -o output.json
//!
//! # Compile to OpenQASM 2.0 (simulators/hardware)
//! zlup compile program.zlp --target qasm -o output.qasm
//!
//! # Compile to HUGR (requires --features hugr)
//! zlup compile program.zlp --target hugr -o output.hugr
//!
//! # Check without compiling (semantic validation)
//! zlup check program.zlp
//!
//! # Check with strict mode (NASA Power of 10)
//! zlup check program.zlp --strict
//!
//! # Parse and dump AST (for debugging)
//! zlup parse program.zlp
//!
//! # Analyze parallelism opportunities
//! zlup analyze program.zlp
//! zlup analyze program.zlp --format json --verbose
//! ```

// Experimental - suppress warnings during development
// The unused_assignments warning is triggered by miette's derive macro
// for the #[source_code], #[label], and #[help] fields
#![allow(dead_code, unused_assignments)]

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use miette::{Diagnostic, NamedSource, SourceSpan};

use zlup::codegen::SlrCodegen;
use zlup::config::{CONFIG_FILE_NAME, Config, TargetConfig};
use zlup::semantic::SemanticAnalyzer;

// =============================================================================
// CLI Structure
// =============================================================================

/// Zluppy - A Zig/SLR/NASA Power of 10 quantum programming language.
///
/// Zluppy solves the same problems as Guppy but through a different lens:
/// explicit over implicit, low-level but safe, simple and obvious.
#[derive(Parser)]
#[command(name = "zlup")]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Zlup project with zlup.toml.
    Init {
        /// Project name (also creates directory with this name)
        #[arg(value_name = "NAME")]
        name: String,

        /// Create project in current directory instead of new directory
        #[arg(long)]
        here: bool,
    },

    /// Build project using zlup.toml configuration.
    Build {
        /// Override strict mode setting
        #[arg(long)]
        strict: Option<bool>,

        /// Override execution target
        #[arg(short, long, value_enum)]
        target: Option<Target>,

        /// Override output format
        #[arg(short, long, value_enum)]
        format: Option<Format>,

        /// Build mode (debug/release)
        #[arg(short, long, value_enum, default_value = "debug")]
        mode: Mode,

        /// Emit compact output (no pretty-printing)
        #[arg(long)]
        compact: bool,
    },

    /// Compile a Zluppy source file to a target format.
    Compile {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file (use - for stdout, default: derived from input)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Execution target (what you're compiling for)
        #[arg(short, long, value_enum, default_value = "simulator")]
        target: Target,

        /// Output format (how to serialize)
        #[arg(short, long, value_enum, default_value = "slr")]
        format: Format,

        /// Build mode (debug/release)
        #[arg(short, long, value_enum, default_value = "debug")]
        mode: Mode,

        /// Emit compact output (no pretty-printing)
        #[arg(long)]
        compact: bool,

        /// Override strict mode from target+mode defaults
        #[arg(long)]
        strict: Option<bool>,

        /// Override log level (elide logs below this level)
        /// Values: trace=0, debug=100, info=200, warn=300, error=400
        #[arg(long, value_name = "LEVEL")]
        log_level: Option<u32>,

        /// Completely elide sim.* commands (no barrier).
        /// By default, hardware targets emit barriers for sim commands to
        /// preserve ordering. This flag removes them entirely for max optimization.
        #[arg(long)]
        elide_sim: bool,
    },

    /// Check a Zluppy source file for errors without compiling.
    Check {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Enable strict mode (NASA Power of 10 checks)
        #[arg(long)]
        strict: bool,
    },

    /// Parse a Zluppy source file and dump the AST (for debugging).
    Parse {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output format
        #[arg(short, long, value_enum, default_value = "debug")]
        format: AstFormat,
    },

    /// Format a Zluppy source file.
    #[command(name = "fmt")]
    Format {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Write output to file instead of stdout
        #[arg(short, long)]
        write: bool,

        /// Check if file is formatted (exit 1 if not)
        #[arg(long)]
        check: bool,
    },

    /// Lint a Zluppy source file for style and best practices.
    Lint {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Lint configuration level (default is strict for safety)
        #[arg(short, long, value_enum, default_value = "strict")]
        level: LintLevel,

        /// Treat warnings as errors
        #[arg(long)]
        deny_warnings: bool,

        /// Output format
        #[arg(short, long, value_enum, default_value = "pretty")]
        format: LintFormat,

        /// Apply safe fixes automatically
        #[arg(long)]
        fix: bool,

        /// Also apply unsafe fixes (requires --fix)
        #[arg(long, requires = "fix")]
        unsafe_fixes: bool,

        /// Show diff of fixes without applying them
        #[arg(long)]
        diff: bool,

        /// Show fix statistics only (no diagnostics output)
        #[arg(long)]
        statistics: bool,
    },

    /// Evaluate a Zluppy expression or small program (playground mode).
    ///
    /// Examples:
    ///   zlup eval "2 + 3"
    ///   zlup eval "std.pi * 2"
    ///   echo "x := 5; x * 2" | zlup eval -
    Eval {
        /// Expression to evaluate (or - for stdin)
        #[arg(value_name = "EXPR")]
        expr: String,

        /// Show AST and intermediate steps
        #[arg(long)]
        verbose: bool,
    },

    /// Analyze a Zluppy source file for parallelism opportunities.
    ///
    /// This command performs static analysis to identify:
    /// - Qubit allocator lifetimes and scopes
    /// - Operation dependencies (qubit and data dependencies)
    /// - Parallel execution layers (operations that can run simultaneously)
    ///
    /// Examples:
    ///   zlup analyze program.zlp
    ///   zlup analyze program.zlp --format json
    ///   zlup analyze program.zlp --verbose
    Analyze {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: AnalyzeFormat,

        /// Show detailed dependency graph
        #[arg(long, short)]
        verbose: bool,
    },

    /// Generate documentation from doc comments in a Zluppy source file.
    ///
    /// Examples:
    ///   zlup doc program.zlp
    ///   zlup doc program.zlp -o docs.md
    ///   zlup doc program.zlp --all
    Doc {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Include private (non-pub) items
        #[arg(long)]
        all: bool,
    },

    /// Run tests defined in a Zluppy source file.
    ///
    /// Tests are defined with `test "name" { ... }` blocks.
    ///
    /// Examples:
    ///   zlup test program.zlp
    ///   zlup test program.zlp --filter "addition"
    ///   zlup test program.zlp --verbose
    Test {
        /// Input file (use - for stdin)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Only run tests matching this pattern
        #[arg(long, value_name = "PATTERN")]
        filter: Option<String>,

        /// Enable strict mode (NASA Power of 10)
        #[arg(long)]
        strict: bool,

        /// Print verbose output
        #[arg(long, short)]
        verbose: bool,
    },
}

/// Lint configuration levels.
#[derive(Clone, Copy, ValueEnum)]
enum LintLevel {
    /// Minimal checks (only unused variable/measurement)
    Minimal,
    /// Relaxed checks (warnings instead of errors)
    Relaxed,
    /// Strict checks (NASA Power of 10 enforcement) - DEFAULT
    Strict,
}

/// Lint output formats.
#[derive(Clone, Copy, ValueEnum)]
enum LintFormat {
    /// Human-readable format with colors
    Pretty,
    /// JSON format for tooling integration
    Json,
    /// Compact one-line-per-diagnostic format
    Compact,
}

/// Execution target - what you're compiling for (affects semantics and passes).
#[derive(Clone, Copy, ValueEnum, Default, Debug)]
enum Target {
    /// Simulator: full debug info, relaxed constraints, keep simulation constructs
    #[default]
    Simulator,
    /// Hardware: strict constraints, drop simulation artifacts, enforce gate sets
    Hardware,
    /// Emulator: hardware-like constraints but with simulation visibility
    Emulator,
}

impl Target {
    /// Whether this target implies strict mode by default.
    fn default_strict(&self) -> bool {
        match self {
            Target::Simulator => false,
            Target::Hardware => true,
            Target::Emulator => true,
        }
    }

    /// Default log elision level for this target.
    fn default_log_elision(&self) -> Option<u32> {
        match self {
            Target::Simulator => None,     // Keep all logs
            Target::Hardware => Some(300), // Warn and above only
            Target::Emulator => Some(200), // Info and above
        }
    }

    /// How sim.* commands should be handled for this target.
    ///
    /// Simulator commands (noise control, etc.) are only meaningful for
    /// the simulator target. For hardware and emulator, they emit barriers
    /// by default to preserve ordering semantics.
    fn sim_mode(&self) -> zlup::codegen::slr::SimMode {
        use zlup::codegen::slr::SimMode;
        match self {
            Target::Simulator => SimMode::Emit, // Output actual sim commands
            Target::Hardware => SimMode::Barrier, // Emit barrier to preserve ordering
            Target::Emulator => SimMode::Barrier, // Emit barrier to preserve ordering
        }
    }
}

/// Output format - how to serialize the compiled output.
#[derive(Clone, Copy, ValueEnum, Default, Debug)]
enum Format {
    /// SLR-AST JSON (Python/PECOS bridge)
    #[default]
    Slr,
    /// PHIR-JSON format (PECOS simulator targeting) - see pecos-phir-json spec v0.1.0
    PhirJson,
    /// OpenQASM 2.0 (simulators and hardware)
    Qasm,
    /// HUGR (hardware/experiments) - requires --features hugr
    #[cfg(feature = "hugr")]
    Hugr,
}

/// AST output formats for parse command.
#[derive(Clone, Copy, ValueEnum)]
enum AstFormat {
    /// Rust Debug format
    Debug,
    /// JSON format
    Json,
}

/// Output formats for analyze command.
#[derive(Clone, Copy, ValueEnum, Default)]
enum AnalyzeFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for tooling integration
    Json,
}

/// Build mode - optimization and debug level.
///
/// Combined with Target, this controls the full compilation behavior.
/// Target controls *what* you're building for, Mode controls *how* optimized.
#[derive(Clone, Copy, ValueEnum, Default, Debug)]
enum Mode {
    /// Debug (default): all logs kept, no optimizations, permissive
    #[default]
    Debug,

    /// Release: elide debug/trace logs, enable optimizations, stricter checks
    Release,
}

impl Mode {
    /// Log elision adjustment for this mode (added to target's default).
    fn log_elision_adjustment(&self) -> Option<u32> {
        match self {
            Mode::Debug => None,        // Don't elide beyond target default
            Mode::Release => Some(100), // Bump elision by one level
        }
    }

    /// Whether this mode implies stricter checks.
    fn strict_bias(&self) -> bool {
        match self {
            Mode::Debug => false,
            Mode::Release => true,
        }
    }
}

/// Compute effective settings from target + mode combination.
fn effective_settings(target: Target, mode: Mode) -> (bool, Option<u32>) {
    // Strict: either target or mode can enable it
    let strict = target.default_strict() || mode.strict_bias();

    // Log elision: start with target default, then apply mode adjustment
    let log_elision = match (target.default_log_elision(), mode.log_elision_adjustment()) {
        (Some(t), Some(m)) => Some(t.max(m)), // Take the higher (more elision)
        (Some(t), None) => Some(t),
        (None, Some(m)) => Some(m),
        (None, None) => None,
    };

    (strict, log_elision)
}

// =============================================================================
// Error Handling
// =============================================================================

/// CLI error with source context.
/// Fields like `src`, `span`, and `help` are used by miette's Diagnostic derive,
/// not directly in our code.
#[derive(Debug, Diagnostic, thiserror::Error)]
enum CliError {
    #[error("failed to read file: {path}")]
    #[diagnostic(code(zlup::io::read))]
    ReadError {
        path: String,
        #[source]
        source: io::Error,
    },

    #[error("failed to write file: {path}")]
    #[diagnostic(code(zlup::io::write))]
    WriteError {
        path: String,
        #[source]
        source: io::Error,
    },

    #[error("parse error")]
    #[diagnostic(code(zlup::parse))]
    ParseError {
        #[source_code]
        src: NamedSource<String>,
        #[label("error here")]
        span: SourceSpan,
        #[help]
        help: String,
    },

    #[error("semantic error: {message}")]
    #[diagnostic(code(zlup::semantic))]
    SemanticError {
        message: String,
        #[source_code]
        src: NamedSource<String>,
        #[label("{message}")]
        span: SourceSpan,
    },

    #[error("codegen error: {message}")]
    #[diagnostic(code(zlup::codegen))]
    CodegenError { message: String },

    #[error("file needs formatting: {path}")]
    #[diagnostic(code(zlup::fmt::check))]
    FormatterCheckFailed { path: String },

    #[error("lint failed: {path} has {error_count} error(s) and {warning_count} warning(s)")]
    #[diagnostic(code(zlup::lint))]
    LintFailed {
        path: String,
        error_count: usize,
        warning_count: usize,
    },

    #[error("HUGR codegen requires --features hugr")]
    #[diagnostic(code(zlup::feature))]
    HugrNotEnabled,

    #[error("config error: {message}")]
    #[diagnostic(code(zlup::config))]
    ConfigError { message: String },

    #[error("project directory '{path}' already exists")]
    #[diagnostic(code(zlup::init))]
    ProjectExists { path: String },

    #[error("failed to create directory '{path}': {source}")]
    #[diagnostic(code(zlup::io::mkdir))]
    CreateDirError {
        path: String,
        #[source]
        source: io::Error,
    },
}

// =============================================================================
// Input/Output Helpers
// =============================================================================

/// Read source from file or stdin.
fn read_source(path: &PathBuf) -> Result<(String, String), CliError> {
    if path.as_os_str() == "-" {
        let mut source = String::new();
        io::stdin()
            .read_to_string(&mut source)
            .map_err(|e| CliError::ReadError {
                path: "<stdin>".to_string(),
                source: e,
            })?;
        Ok((source, "<stdin>".to_string()))
    } else {
        let source = fs::read_to_string(path).map_err(|e| CliError::ReadError {
            path: path.display().to_string(),
            source: e,
        })?;
        Ok((source, path.display().to_string()))
    }
}

/// Compute the byte offset for a source location.
fn location_to_offset(source: &str, location: &zlup::ast::SourceLocation) -> usize {
    source
        .lines()
        .take(location.line.saturating_sub(1) as usize)
        .map(|l| l.len() + 1)
        .sum::<usize>()
        + location.column.saturating_sub(1) as usize
}

/// Convert a pest parse error message to a more user-friendly message.
fn friendly_parse_error(pest_message: &str) -> String {
    // Check for common patterns and provide better messages with suggestions
    if pest_message.contains("expected identifier") {
        "expected an identifier (variable, function, or type name)".to_string()
    } else if pest_message.contains("expected type_expr") {
        "expected a type (e.g., u32, bool, []u8, ?T)".to_string()
    } else if pest_message.contains("expected expr") {
        "expected an expression".to_string()
    } else if pest_message.contains("expected statement") {
        "expected a statement (binding, assignment, if, for, etc.)".to_string()
    } else if pest_message.contains("expected \"(\"") {
        "expected '(' - check for missing parentheses".to_string()
    } else if pest_message.contains("expected \")\"") {
        "expected ')' - check for unmatched parentheses".to_string()
    } else if pest_message.contains("expected \"{\"") {
        "expected '{' - blocks require braces".to_string()
    } else if pest_message.contains("expected \"}\"") {
        "expected '}' - check for unmatched braces".to_string()
    } else if pest_message.contains("expected \"[\"") {
        "expected '[' - arrays use square brackets".to_string()
    } else if pest_message.contains("expected \"]\"") {
        "expected ']' - check for unmatched brackets".to_string()
    } else if pest_message.contains("expected \";\"") {
        "expected ';' - statements must end with semicolon".to_string()
    } else if pest_message.contains("expected \":=\"") || pest_message.contains("expected \"=\"") {
        "expected ':=' for binding or '=' for assignment".to_string()
    } else if pest_message.contains("expected assign_op") {
        "unexpected token - expected assignment (=, +=, -=, etc.)".to_string()
    } else if pest_message.contains("expected top_level_decl") {
        "expected a declaration (fn, struct, enum, or binding)".to_string()
    } else if pest_message.contains("expected return_type") {
        "expected '-> T' return type after function parameters".to_string()
    } else if pest_message.contains("expected param") {
        "expected function parameter (name: Type)".to_string()
    } else if pest_message.contains("expected block") {
        "expected a block { ... }".to_string()
    } else if pest_message.contains("expected EOI") {
        "unexpected content after end of file".to_string()
    } else if pest_message.contains("expected string_literal") {
        "expected a string (\"...\", r\"...\", or \"\"\"...\"\"\")".to_string()
    } else if pest_message.contains("expected number_literal") {
        "expected a number (42, 0xFF, 3.14, etc.)".to_string()
    } else if pest_message.contains("expected bool_literal") {
        "expected 'true' or 'false'".to_string()
    } else {
        // Fall back to the original message but clean it up
        pest_message
            .lines()
            .next()
            .unwrap_or(pest_message)
            .to_string()
    }
}

/// Write output to file or stdout.
fn write_output(path: Option<&PathBuf>, content: &str) -> Result<(), CliError> {
    match path {
        Some(p) if p.as_os_str() != "-" => {
            fs::write(p, content).map_err(|e| CliError::WriteError {
                path: p.display().to_string(),
                source: e,
            })
        }
        _ => io::stdout()
            .write_all(content.as_bytes())
            .map_err(|e| CliError::WriteError {
                path: "<stdout>".to_string(),
                source: e,
            }),
    }
}

/// Derive output path from input path and format.
fn derive_output_path(input: &Path, format: Format) -> PathBuf {
    if input.as_os_str() == "-" {
        return PathBuf::from("-");
    }

    let stem = input.file_stem().unwrap_or_default();
    let ext = match format {
        Format::Slr => "slr.json",
        Format::PhirJson => "phir.json",
        Format::Qasm => "qasm",
        #[cfg(feature = "hugr")]
        Format::Hugr => "hugr",
    };

    input.with_file_name(format!("{}.{}", stem.to_string_lossy(), ext))
}

// =============================================================================
// Commands
// =============================================================================

/// Execute the init command - create a new project.
fn cmd_init(name: String, here: bool) -> Result<(), CliError> {
    let project_dir = if here {
        std::env::current_dir().map_err(|e| CliError::ReadError {
            path: ".".to_string(),
            source: e,
        })?
    } else {
        let dir = PathBuf::from(&name);
        if dir.exists() {
            return Err(CliError::ProjectExists {
                path: dir.display().to_string(),
            });
        }
        fs::create_dir_all(&dir).map_err(|e| CliError::CreateDirError {
            path: dir.display().to_string(),
            source: e,
        })?;
        dir
    };

    // Create zlup.toml
    let config = Config::new(&name);
    let config_path = project_dir.join(CONFIG_FILE_NAME);
    let config_content = config.to_toml().map_err(|e| CliError::ConfigError {
        message: e.to_string(),
    })?;
    fs::write(&config_path, config_content).map_err(|e| CliError::WriteError {
        path: config_path.display().to_string(),
        source: e,
    })?;

    // Create main.zlp with example content
    let main_path = project_dir.join("main.zlp");
    let main_content = r#"//! Main entry point for the quantum program.

/// Main function - program entry point.
pub fn main() -> unit {
    // Allocate qubits (no mut needed - just applying gates)
    q := qalloc(2);
    pz q;

    // Create Bell state
    h q[0];
    cx (q[0], q[1]);

    // Measure
    result := mz(u1) q[0];

    return unit;
}
"#;
    fs::write(&main_path, main_content).map_err(|e| CliError::WriteError {
        path: main_path.display().to_string(),
        source: e,
    })?;

    eprintln!(
        "Created new project '{}' at {}",
        name,
        project_dir.display()
    );
    eprintln!("  {} - project configuration", CONFIG_FILE_NAME);
    eprintln!("  main.zlp - main source file");
    eprintln!();
    eprintln!("To build: cd {} && zlup build", project_dir.display());

    Ok(())
}

/// Execute the build command - build using zlup.toml.
fn cmd_build(
    strict_override: Option<bool>,
    target_override: Option<Target>,
    format_override: Option<Format>,
    mode: Mode,
    compact: bool,
) -> Result<(), CliError> {
    // Find zlup.toml
    let current_dir = std::env::current_dir().map_err(|e| CliError::ReadError {
        path: ".".to_string(),
        source: e,
    })?;

    let (config, config_path) =
        Config::find_and_load(&current_dir).map_err(|e| CliError::ConfigError {
            message: e.to_string(),
        })?;

    let project_root = Config::project_root(&config_path);

    // Determine format (CLI overrides config)
    let format = format_override.unwrap_or(match config.build.target {
        TargetConfig::Slr => Format::Slr,
        #[cfg(feature = "hugr")]
        TargetConfig::Hugr => Format::Hugr,
        #[cfg(not(feature = "hugr"))]
        TargetConfig::Hugr => {
            eprintln!("Warning: HUGR format specified but not enabled, using SLR");
            Format::Slr
        }
    });

    // Target defaults to simulator unless overridden
    let target = target_override.unwrap_or(Target::Simulator);

    // Get entry file
    let entry_path = config.entry_path(&config_path);
    if !entry_path.exists() {
        return Err(CliError::ReadError {
            path: entry_path.display().to_string(),
            source: io::Error::new(io::ErrorKind::NotFound, "entry file not found"),
        });
    }

    // Create output directory
    let output_dir = config.output_path(&config_path);
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir).map_err(|e| CliError::CreateDirError {
            path: output_dir.display().to_string(),
            source: e,
        })?;
    }

    // Derive output file name
    let output_ext = match format {
        Format::Slr => "slr.json",
        Format::PhirJson => "phir.json",
        Format::Qasm => "qasm",
        #[cfg(feature = "hugr")]
        Format::Hugr => "hugr",
    };
    let output_name = entry_path.file_stem().unwrap_or_default().to_string_lossy();
    let output_path = output_dir.join(format!("{}.{}", output_name, output_ext));

    // Compute effective settings
    let (default_strict, _) = effective_settings(target, mode);
    let strict = strict_override.unwrap_or(config.build.strict || default_strict);

    eprintln!(
        "Building {} ({}) [{:?} -> {:?}, {}]",
        config.package.name,
        config.package.version,
        target,
        format,
        if strict { "strict" } else { "normal" }
    );

    // Compile
    cmd_compile(CompileOptions {
        input: entry_path,
        output: Some(output_path.clone()),
        target,
        format,
        mode,
        compact,
        strict_override: Some(strict),
        log_level_override: None,
        elide_sim: false,
    })?;

    eprintln!(
        "Built {} -> {}",
        project_root.join(&config.package.entry).display(),
        output_path.display()
    );

    Ok(())
}

/// Options for the compile command.
struct CompileOptions {
    input: PathBuf,
    output: Option<PathBuf>,
    target: Target,
    format: Format,
    mode: Mode,
    compact: bool,
    strict_override: Option<bool>,
    log_level_override: Option<u32>,
    elide_sim: bool,
}

/// Execute the compile command.
fn cmd_compile(opts: CompileOptions) -> Result<(), CliError> {
    let CompileOptions {
        input,
        output,
        target,
        format,
        mode,
        compact,
        strict_override,
        log_level_override,
        elide_sim,
    } = opts;

    // Resolve settings from target + mode with overrides
    let (default_strict, default_log_level) = effective_settings(target, mode);
    let strict = strict_override.unwrap_or(default_strict);
    let log_level = log_level_override.or(default_log_level);
    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    // Semantic analysis
    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(|e| {
        let (span, message) = if let Some(loc) = e.location() {
            let start = location_to_offset(&source, loc);
            (SourceSpan::from(start..start + 1), e.to_string())
        } else {
            (SourceSpan::from(0..1), e.to_string())
        };
        CliError::SemanticError {
            message,
            src: NamedSource::new(&filename, source.clone()),
            span,
        }
    })?;

    // Derive module name from filename for log namespacing
    let module_name = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "main".to_string());

    // Code generation
    let output_content = match format {
        Format::Slr => {
            use zlup::codegen::slr::LogElisionLevel;

            // Create codegen with release settings if we have any log elision
            let mut codegen = if log_level.is_some() {
                SlrCodegen::new_release()
            } else {
                SlrCodegen::new()
            };

            // Set module for automatic log namespacing
            codegen.set_module(&module_name);

            // Apply log elision level from profile (possibly overridden)
            if let Some(level) = log_level {
                codegen.set_log_elision(LogElisionLevel(Some(level)));
            }

            // Apply sim mode based on target (with optional full elision override)
            let sim_mode = if elide_sim {
                zlup::codegen::slr::SimMode::Elide
            } else {
                target.sim_mode()
            };
            codegen.set_sim_mode(sim_mode);

            let slr_program =
                codegen
                    .compile(&program)
                    .map_err(|e: zlup::codegen::slr::SlrError| CliError::CodegenError {
                        message: e.to_string(),
                    })?;

            if compact {
                codegen.to_json_compact(&slr_program).map_err(
                    |e: zlup::codegen::slr::SlrError| CliError::CodegenError {
                        message: e.to_string(),
                    },
                )?
            } else {
                codegen
                    .to_json(&slr_program)
                    .map_err(|e: zlup::codegen::slr::SlrError| CliError::CodegenError {
                        message: e.to_string(),
                    })?
            }
        }
        Format::PhirJson => {
            use zlup::codegen::PhirJsonCodegen;

            let mut codegen = PhirJsonCodegen::new();
            let phir_json_program =
                codegen
                    .compile(&program)
                    .map_err(|e| CliError::CodegenError {
                        message: e.to_string(),
                    })?;

            if compact {
                codegen
                    .to_json_compact(&phir_json_program)
                    .map_err(|e| CliError::CodegenError {
                        message: e.to_string(),
                    })?
            } else {
                codegen
                    .to_json(&phir_json_program)
                    .map_err(|e| CliError::CodegenError {
                        message: e.to_string(),
                    })?
            }
        }
        Format::Qasm => {
            use zlup::codegen::QasmCodegen;

            let mut codegen = QasmCodegen::new();
            codegen
                .compile(&program)
                .map_err(|e| CliError::CodegenError {
                    message: e.to_string(),
                })?
        }
        #[cfg(feature = "hugr")]
        Format::Hugr => {
            use zlup::codegen::HugrCodegen;

            let mut codegen = HugrCodegen::new();
            let hugr = codegen
                .compile(&program)
                .map_err(|e| CliError::CodegenError {
                    message: e.to_string(),
                })?;

            // Serialize HUGR to text envelope format (compatible with PECOS hugr_engine)
            codegen
                .to_string(&hugr)
                .map_err(|e| CliError::CodegenError {
                    message: e.to_string(),
                })?
        }
    };

    // Write output
    let output_path = output.unwrap_or_else(|| derive_output_path(&input, format));
    write_output(Some(&output_path), &output_content)?;

    eprintln!(
        "Compiled {} -> {}",
        filename,
        if output_path.as_os_str() == "-" {
            "<stdout>".to_string()
        } else {
            output_path.display().to_string()
        }
    );

    Ok(())
}

/// Execute the check command.
fn cmd_check(input: PathBuf, strict: bool) -> Result<(), CliError> {
    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    // Semantic analysis
    let mut analyzer = if strict {
        SemanticAnalyzer::new()
    } else {
        SemanticAnalyzer::new_permissive()
    };
    analyzer.analyze(&program).map_err(|e| {
        let (span, message) = if let Some(loc) = e.location() {
            let start = location_to_offset(&source, loc);
            (SourceSpan::from(start..start + 1), e.to_string())
        } else {
            (SourceSpan::from(0..1), e.to_string())
        };
        CliError::SemanticError {
            message,
            src: NamedSource::new(&filename, source.clone()),
            span,
        }
    })?;

    eprintln!("OK: {}", filename);
    Ok(())
}

/// Execute the parse command.
fn cmd_parse(input: PathBuf, format: AstFormat) -> Result<(), CliError> {
    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    // Output
    let output = match format {
        AstFormat::Debug => format!("{:#?}", program),
        AstFormat::Json => serde_json::to_string_pretty(&program)
            .unwrap_or_else(|e| format!("JSON serialization error: {}", e)),
    };

    println!("{}", output);
    Ok(())
}

/// Execute the format command.
fn cmd_format(input: PathBuf, write: bool, check: bool) -> Result<(), CliError> {
    use zlup::formatter::{FormatOptions, format};

    let (source, filename) = read_source(&input)?;
    let options = FormatOptions::default();
    let formatted = format(&source, &options);

    if check {
        // Check mode: exit 1 if file needs formatting
        if source != formatted {
            eprintln!("Would reformat: {}", filename);
            return Err(CliError::FormatterCheckFailed { path: filename });
        }
        eprintln!("OK: {}", filename);
        return Ok(());
    }

    if write {
        // Write mode: write back to file
        if input.as_os_str() == "-" {
            // Can't write back to stdin
            return Err(CliError::WriteError {
                path: "<stdin>".to_string(),
                source: io::Error::new(io::ErrorKind::InvalidInput, "cannot write to stdin"),
            });
        }
        if source != formatted {
            fs::write(&input, &formatted).map_err(|e| CliError::WriteError {
                path: input.display().to_string(),
                source: e,
            })?;
            eprintln!("Formatted: {}", filename);
        } else {
            eprintln!("Already formatted: {}", filename);
        }
    } else {
        // Default: print to stdout
        print!("{}", formatted);
    }

    Ok(())
}

// Arguments come from CLI parsing - grouping them wouldn't improve readability
#[allow(clippy::too_many_arguments)]
fn cmd_lint(
    input: PathBuf,
    level: LintLevel,
    deny_warnings: bool,
    format: LintFormat,
    fix: bool,
    unsafe_fixes: bool,
    show_diff: bool,
    statistics_only: bool,
) -> Result<(), CliError> {
    use zlup::linter::{FixSafety, LintConfig, Linter, Severity, apply_fixes};

    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = e.location.line.saturating_sub(1) as usize * 80 + e.location.column as usize;
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    // Configure linter
    let config = match level {
        LintLevel::Minimal => LintConfig::minimal(),
        LintLevel::Relaxed => LintConfig::relaxed(),
        LintLevel::Strict => LintConfig::strict(),
    };

    // Run linter (with source for fix computation)
    let diagnostics = Linter::new(config).with_source(&source).lint(&program);

    // Statistics-only mode
    if statistics_only {
        let errors = diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Error | Severity::Deny))
            .count();
        let warnings = diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Warning))
            .count();
        let hints = diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Hint))
            .count();
        let fixable_safe = diagnostics.iter().filter(|d| d.has_safe_fix()).count();
        let fixable_unsafe = diagnostics
            .iter()
            .filter(|d| {
                d.fix
                    .as_ref()
                    .is_some_and(|f| f.safety == FixSafety::Unsafe)
            })
            .count();

        println!("File: {}", filename);
        println!("Errors: {}", errors);
        println!("Warnings: {}", warnings);
        println!("Hints: {}", hints);
        println!("Total: {}", diagnostics.len());
        println!("Fixable (safe): {}", fixable_safe);
        println!("Fixable (unsafe): {}", fixable_unsafe);

        if errors > 0 || (deny_warnings && warnings > 0) {
            return Err(CliError::LintFailed {
                path: filename,
                error_count: errors,
                warning_count: warnings,
            });
        }
        return Ok(());
    }

    // Diff mode - show what would change without applying
    if show_diff {
        let fix_result = apply_fixes(&source, &diagnostics, unsafe_fixes);

        if fix_result.safe_fixes_applied > 0 || fix_result.unsafe_fixes_applied > 0 {
            eprintln!(
                "Would apply {} safe fix(es){} to {}",
                fix_result.safe_fixes_applied,
                if fix_result.unsafe_fixes_applied > 0 {
                    format!(" and {} unsafe fix(es)", fix_result.unsafe_fixes_applied)
                } else {
                    String::new()
                },
                filename
            );

            // Generate unified diff
            print_unified_diff(&source, &fix_result.source, &filename);

            if fix_result.fixes_skipped > 0 {
                eprintln!(
                    "Would skip {} fix(es) (conflicts or safety level)",
                    fix_result.fixes_skipped
                );
            }
        } else {
            eprintln!("No fixes available");
        }

        // Still return error if there are issues
        let has_errors = diagnostics
            .iter()
            .any(|d| matches!(d.severity, Severity::Error | Severity::Deny));
        let has_warnings = diagnostics
            .iter()
            .any(|d| matches!(d.severity, Severity::Warning));
        if has_errors || (deny_warnings && has_warnings) {
            let errors = diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Error | Severity::Deny))
                .count();
            let warnings = diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Warning))
                .count();
            return Err(CliError::LintFailed {
                path: filename,
                error_count: errors,
                warning_count: warnings,
            });
        }
        return Ok(());
    }

    // Apply fixes if requested
    if fix {
        let fix_result = apply_fixes(&source, &diagnostics, unsafe_fixes);

        if fix_result.safe_fixes_applied > 0 || fix_result.unsafe_fixes_applied > 0 {
            // Write fixed source back to file
            if input.as_os_str() == "-" {
                // Can't write back to stdin
                return Err(CliError::WriteError {
                    path: "<stdin>".to_string(),
                    source: io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "cannot apply fixes to stdin",
                    ),
                });
            }

            fs::write(&input, &fix_result.source).map_err(|e| CliError::WriteError {
                path: input.display().to_string(),
                source: e,
            })?;

            eprintln!(
                "Applied {} safe fix(es){} to {}",
                fix_result.safe_fixes_applied,
                if fix_result.unsafe_fixes_applied > 0 {
                    format!(" and {} unsafe fix(es)", fix_result.unsafe_fixes_applied)
                } else {
                    String::new()
                },
                filename
            );

            if fix_result.fixes_skipped > 0 {
                eprintln!(
                    "Skipped {} fix(es) (conflicts or safety level)",
                    fix_result.fixes_skipped
                );
            }

            // Re-lint to show remaining issues
            let program = zlup::parse_file(&fix_result.source, &filename).map_err(|e| {
                let start =
                    e.location.line.saturating_sub(1) as usize * 80 + e.location.column as usize;
                CliError::ParseError {
                    src: NamedSource::new(&filename, fix_result.source.clone()),
                    span: SourceSpan::from(start..start + 1),
                    help: friendly_parse_error(&e.message),
                }
            })?;

            let config = match level {
                LintLevel::Minimal => LintConfig::minimal(),
                LintLevel::Relaxed => LintConfig::relaxed(),
                LintLevel::Strict => LintConfig::strict(),
            };

            let remaining_diagnostics = Linter::new(config)
                .with_source(&fix_result.source)
                .lint(&program);

            if remaining_diagnostics.is_empty() {
                eprintln!("All issues fixed!");
                return Ok(());
            }

            // Output remaining diagnostics
            match format {
                LintFormat::Pretty => print_diagnostics_pretty(&remaining_diagnostics, &filename),
                LintFormat::Json => print_diagnostics_json(&remaining_diagnostics),
                LintFormat::Compact => print_diagnostics_compact(&remaining_diagnostics, &filename),
            }

            let errors = remaining_diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Error | Severity::Deny))
                .count();
            let warnings = remaining_diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Warning))
                .count();
            let hints = remaining_diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Hint))
                .count();

            eprintln!();
            eprintln!(
                "Remaining: {} error(s), {} warning(s), {} hint(s) in {}",
                errors, warnings, hints, filename
            );

            let has_errors = remaining_diagnostics
                .iter()
                .any(|d| matches!(d.severity, Severity::Error | Severity::Deny));
            let has_warnings = remaining_diagnostics
                .iter()
                .any(|d| matches!(d.severity, Severity::Warning));

            if has_errors || (deny_warnings && has_warnings) {
                return Err(CliError::LintFailed {
                    path: filename,
                    error_count: errors,
                    warning_count: warnings,
                });
            }

            return Ok(());
        } else {
            eprintln!("No fixes available to apply");
        }
    }

    // Check for errors
    let has_errors = diagnostics
        .iter()
        .any(|d| matches!(d.severity, Severity::Error | Severity::Deny));
    let has_warnings = diagnostics
        .iter()
        .any(|d| matches!(d.severity, Severity::Warning));

    // Output diagnostics
    match format {
        LintFormat::Pretty => print_diagnostics_pretty(&diagnostics, &filename),
        LintFormat::Json => print_diagnostics_json(&diagnostics),
        LintFormat::Compact => print_diagnostics_compact(&diagnostics, &filename),
    }

    // Summary
    if !diagnostics.is_empty() {
        let errors = diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Error | Severity::Deny))
            .count();
        let warnings = diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Warning))
            .count();
        let hints = diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Hint))
            .count();
        let fixable_safe = diagnostics.iter().filter(|d| d.has_safe_fix()).count();
        let fixable_total = diagnostics.iter().filter(|d| d.has_fix()).count();

        eprintln!();
        eprintln!(
            "Found {} error(s), {} warning(s), {} hint(s) in {}",
            errors, warnings, hints, filename
        );

        if fixable_safe > 0 {
            eprintln!(
                "{} issue(s) can be fixed automatically (run with --fix)",
                fixable_safe
            );
        }
        if fixable_total > fixable_safe {
            eprintln!(
                "{} additional issue(s) can be fixed with --fix --unsafe-fixes",
                fixable_total - fixable_safe
            );
        }
    } else {
        eprintln!("No issues found in {}", filename);
    }

    // Return error if there are errors, or warnings in deny mode
    if has_errors || (deny_warnings && has_warnings) {
        return Err(CliError::LintFailed {
            path: filename,
            error_count: diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Error | Severity::Deny))
                .count(),
            warning_count: diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Warning))
                .count(),
        });
    }

    Ok(())
}

fn print_diagnostics_pretty(diagnostics: &[zlup::linter::LintDiagnostic], filename: &str) {
    use zlup::linter::Severity;

    for diag in diagnostics {
        let (severity_str, color) = match diag.severity {
            Severity::Hint => ("hint", "\x1b[36m"),       // Cyan
            Severity::Warning => ("warning", "\x1b[33m"), // Yellow
            Severity::Error => ("error", "\x1b[31m"),     // Red
            Severity::Deny => ("deny", "\x1b[91m"),       // Bright Red
        };
        let reset = "\x1b[0m";
        let bold = "\x1b[1m";

        if let Some(ref loc) = diag.location {
            eprintln!(
                "{}{}{}:{}{}: {}{}{}: {}",
                bold,
                filename,
                reset,
                loc.line,
                loc.column,
                color,
                severity_str,
                reset,
                diag.message
            );
        } else {
            eprintln!(
                "{}{}{}: {}{}{}: {}",
                bold, filename, reset, color, severity_str, reset, diag.message
            );
        }

        eprintln!("   [{}]", diag.rule);

        if let Some(ref suggestion) = diag.suggestion {
            eprintln!("  \x1b[32mhelp{}: {}", reset, suggestion);
        }
        eprintln!();
    }
}

fn print_diagnostics_json(diagnostics: &[zlup::linter::LintDiagnostic]) {
    use serde_json::json;

    let json_diags: Vec<_> = diagnostics
        .iter()
        .map(|d| {
            json!({
                "rule": d.rule,
                "message": d.message,
                "severity": format!("{:?}", d.severity).to_lowercase(),
                "location": d.location.as_ref().map(|l| json!({
                    "line": l.line,
                    "column": l.column
                })),
                "suggestion": d.suggestion
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&json_diags).unwrap());
}

fn print_diagnostics_compact(diagnostics: &[zlup::linter::LintDiagnostic], filename: &str) {
    use zlup::linter::Severity;

    for diag in diagnostics {
        let severity = match diag.severity {
            Severity::Hint => "H",
            Severity::Warning => "W",
            Severity::Error => "E",
            Severity::Deny => "D",
        };

        if let Some(ref loc) = diag.location {
            println!(
                "{}:{}:{}: {} [{}] {}",
                filename, loc.line, loc.column, severity, diag.rule, diag.message
            );
        } else {
            println!(
                "{}: {} [{}] {}",
                filename, severity, diag.rule, diag.message
            );
        }
    }
}

/// Print a unified diff between original and fixed source code.
fn print_unified_diff(original: &str, fixed: &str, filename: &str) {
    let orig_lines: Vec<&str> = original.lines().collect();
    let fixed_lines: Vec<&str> = fixed.lines().collect();

    // ANSI colors
    let red = "\x1b[31m";
    let green = "\x1b[32m";
    let cyan = "\x1b[36m";
    let reset = "\x1b[0m";

    println!("{}--- a/{}{}", cyan, filename, reset);
    println!("{}+++ b/{}{}", cyan, filename, reset);

    // Simple diff: find changed lines
    let max_len = orig_lines.len().max(fixed_lines.len());
    let mut i = 0;

    while i < max_len {
        // Find a hunk of changes
        let hunk_start = i;
        let mut orig_hunk = Vec::new();
        let mut fixed_hunk = Vec::new();
        let mut has_changes = false;

        // Collect context before changes (up to 3 lines)
        let context_start = hunk_start.saturating_sub(3);

        // Find changes
        while i < max_len {
            let orig_line = orig_lines.get(i).copied();
            let fixed_line = fixed_lines.get(i).copied();

            if orig_line != fixed_line {
                has_changes = true;
                if let Some(line) = orig_line {
                    orig_hunk.push((i, line));
                }
                if let Some(line) = fixed_line {
                    fixed_hunk.push((i, line));
                }
                i += 1;
            } else if has_changes {
                // Add trailing context
                let mut context_count = 0;
                while i < max_len && context_count < 3 {
                    let ol = orig_lines.get(i).copied();
                    let fl = fixed_lines.get(i).copied();
                    if ol == fl {
                        context_count += 1;
                        i += 1;
                    } else {
                        break;
                    }
                }
                break;
            } else {
                i += 1;
            }
        }

        if has_changes {
            // Print hunk header
            let orig_start = orig_hunk
                .first()
                .map(|(n, _)| *n + 1)
                .unwrap_or(hunk_start + 1);
            let fixed_start = fixed_hunk
                .first()
                .map(|(n, _)| *n + 1)
                .unwrap_or(hunk_start + 1);
            println!(
                "{}@@ -{},{} +{},{} @@{}",
                cyan,
                orig_start,
                orig_hunk.len(),
                fixed_start,
                fixed_hunk.len(),
                reset
            );

            // Print context before
            for j in context_start..hunk_start {
                if let Some(line) = orig_lines.get(j) {
                    println!(" {}", line);
                }
            }

            // Print removed lines
            for (_, line) in &orig_hunk {
                println!("{}-{}{}", red, line, reset);
            }

            // Print added lines
            for (_, line) in &fixed_hunk {
                println!("{}+{}{}", green, line, reset);
            }
        }
    }
}

// =============================================================================
// Analyze Command
// =============================================================================

fn cmd_analyze(input: PathBuf, format: AnalyzeFormat, verbose: bool) -> Result<(), CliError> {
    use zlup::analysis::{
        AllocatorAnalysis, DependencyGraph, OperationTagger, analyze_parallelism,
    };

    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    // Run analysis passes
    let allocator_analysis = AllocatorAnalysis::analyze(&program);
    let tagger = OperationTagger::tag(&program);
    let dep_graph = DependencyGraph::build(tagger.operations);
    let summaries = analyze_parallelism(&program);

    match format {
        AnalyzeFormat::Text => {
            println!("=== Parallelism Analysis: {} ===\n", filename);

            // Allocator summary
            println!("Allocators:");
            if allocator_analysis.allocators.is_empty() {
                println!("  (none)");
            } else {
                for (name, info) in &allocator_analysis.allocators {
                    let size_str = info
                        .size
                        .map(|s| format!("[{}]", s))
                        .unwrap_or_else(|| "[?]".to_string());
                    println!(
                        "  {} {}qubit (scope depth: {}, line: {})",
                        name, size_str, info.scope_depth, info.defined_at_line
                    );
                }
            }
            println!();

            // Function summaries
            println!("Function Analysis:");
            for summary in &summaries {
                println!("  {}:", summary.function_name);
                println!("    Total operations: {}", summary.total_ops);
                println!("    Quantum operations: {}", summary.quantum_ops);
                println!("    Classical operations: {}", summary.classical_ops);
                println!("    Parallel layers: {}", summary.num_layers);
                println!("    Max parallelism: {} ops/layer", summary.max_parallelism);
                println!();
            }

            // Detailed output if verbose
            if verbose {
                println!("=== Dependency Graph ===\n");
                dep_graph.debug_print();
            }
        }
        AnalyzeFormat::Json => {
            use serde_json::json;

            let allocators: Vec<_> = allocator_analysis
                .allocators
                .iter()
                .map(|(name, info)| {
                    json!({
                        "name": name,
                        "size": info.size,
                        "scope_depth": info.scope_depth,
                        "defined_at_line": info.defined_at_line,
                    })
                })
                .collect();

            let functions: Vec<_> = summaries
                .iter()
                .map(|s| {
                    json!({
                        "name": s.function_name,
                        "total_ops": s.total_ops,
                        "quantum_ops": s.quantum_ops,
                        "classical_ops": s.classical_ops,
                        "num_layers": s.num_layers,
                        "max_parallelism": s.max_parallelism,
                    })
                })
                .collect();

            let layers = dep_graph.parallel_layers();
            let layer_details: Vec<_> = layers
                .iter()
                .enumerate()
                .map(|(i, ops)| {
                    let op_details: Vec<_> = ops
                        .iter()
                        .map(|&id| {
                            let op = &dep_graph.operations[id];
                            json!({
                                "id": op.id,
                                "description": op.description,
                                "line": op.line,
                                "is_quantum": op.touches_qubits(),
                            })
                        })
                        .collect();
                    json!({
                        "layer": i,
                        "operations": op_details,
                    })
                })
                .collect();

            let output = json!({
                "file": filename,
                "allocators": allocators,
                "functions": functions,
                "parallel_layers": layer_details,
                "total_operations": dep_graph.operations.len(),
                "total_dependencies": dep_graph.edges.len(),
            });

            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
    }

    Ok(())
}

// =============================================================================
// Eval Command
// =============================================================================

fn cmd_eval(expr: String, verbose: bool) -> Result<(), CliError> {
    use zlup::comptime::ComptimeEvaluator;

    // Read expression from stdin if -
    let input = if expr == "-" {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| CliError::ReadError {
                path: "<stdin>".to_string(),
                source: e,
            })?;
        buf
    } else {
        expr
    };

    let input = input.trim();

    // Wrap in a minimal program context for parsing
    // Try as expression first, then as statements
    let source = if input.contains(';') || input.contains(":=") || input.starts_with("fn ") {
        // Looks like statements - wrap in a function
        format!("fn __eval__() -> type {{ {} }}", input)
    } else {
        // Simple expression - wrap to evaluate
        format!("__result__ := {};", input)
    };

    if verbose {
        eprintln!("--- Source ---");
        eprintln!("{}", source);
    }

    // Parse
    let program = zlup::parse(&source).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new("<eval>", source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    if verbose {
        eprintln!("--- AST ---");
        eprintln!("{:#?}", program);
    }

    // Try comptime evaluation
    let mut evaluator = ComptimeEvaluator::new();

    // Evaluate declarations
    for decl in &program.declarations {
        match decl {
            zlup::ast::TopLevelDecl::Binding(binding) => {
                if let Some(ref value) = binding.value {
                    match evaluator.eval_expr(value) {
                        Ok(value) => {
                            if binding.name == "__result__" {
                                // This is our wrapped expression result
                                println!("{}", value);
                            } else if verbose {
                                println!("{} = {}", binding.name, value);
                            }
                        }
                        Err(e) => {
                            if verbose {
                                eprintln!("Comptime eval error: {}", e);
                            }
                            // Fall back to showing the expression was parsed
                            println!("(parsed: {})", binding.name);
                        }
                    }
                }
            }
            zlup::ast::TopLevelDecl::Fn(func) if verbose => {
                println!("fn {} defined", func.name);
            }
            _ => {}
        }
    }

    Ok(())
}

// =============================================================================
// Doc Command
// =============================================================================

fn cmd_doc(input: PathBuf, output: Option<PathBuf>, all: bool) -> Result<(), CliError> {
    use zlup::docgen::{DocConfig, extract_doc_items, generate_markdown};

    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    let config = DocConfig {
        include_private: all,
        ..Default::default()
    };

    let items = extract_doc_items(&program, &config);
    let module_name = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "module".to_string());
    let markdown = generate_markdown(&items, &module_name);

    write_output(output.as_ref(), &markdown)?;
    Ok(())
}

// =============================================================================
// Test Command
// =============================================================================

fn cmd_test(
    input: PathBuf,
    filter: Option<String>,
    strict: bool,
    verbose: bool,
) -> Result<(), CliError> {
    use zlup::test_runner::{TestOutcome, TestRunConfig, TestRunner, format_results};

    let (source, filename) = read_source(&input)?;

    // Parse
    let program = zlup::parse_file(&source, &filename).map_err(|e| {
        let start = location_to_offset(&source, &e.location);
        CliError::ParseError {
            src: NamedSource::new(&filename, source.clone()),
            span: SourceSpan::from(start..start + 1),
            help: friendly_parse_error(&e.message),
        }
    })?;

    let config = TestRunConfig {
        filter,
        strict,
        verbose,
    };

    let runner = TestRunner::new(config);
    let results = runner.run(&program);
    let output = format_results(&results);
    print!("{}", output);

    // Exit with failure if any tests failed
    let has_failures = results
        .iter()
        .any(|r| matches!(r.outcome, TestOutcome::Fail(_)));
    if has_failures {
        return Err(CliError::CodegenError {
            message: "some tests failed".to_string(),
        });
    }

    Ok(())
}

// =============================================================================
// Main
// =============================================================================

fn main() -> ExitCode {
    // Initialize logging from ZLUP_LOG environment variable
    zlup::logging::init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Init { name, here } => cmd_init(name, here),

        Commands::Build {
            strict,
            target,
            format,
            mode,
            compact,
        } => cmd_build(strict, target, format, mode, compact),

        Commands::Compile {
            input,
            output,
            target,
            format,
            mode,
            compact,
            strict,
            log_level,
            elide_sim,
        } => cmd_compile(CompileOptions {
            input,
            output,
            target,
            format,
            mode,
            compact,
            strict_override: strict,
            log_level_override: log_level,
            elide_sim,
        }),

        Commands::Check { input, strict } => cmd_check(input, strict),

        Commands::Parse { input, format } => cmd_parse(input, format),

        Commands::Format {
            input,
            write,
            check,
        } => cmd_format(input, write, check),

        Commands::Lint {
            input,
            level,
            deny_warnings,
            format,
            fix,
            unsafe_fixes,
            diff,
            statistics,
        } => cmd_lint(
            input,
            level,
            deny_warnings,
            format,
            fix,
            unsafe_fixes,
            diff,
            statistics,
        ),

        Commands::Eval { expr, verbose } => cmd_eval(expr, verbose),

        Commands::Analyze {
            input,
            format,
            verbose,
        } => cmd_analyze(input, format, verbose),

        Commands::Doc { input, output, all } => cmd_doc(input, output, all),

        Commands::Test {
            input,
            filter,
            strict,
            verbose,
        } => cmd_test(input, filter, strict, verbose),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{:?}", miette::Report::new(e));
            ExitCode::FAILURE
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Target defaults
    // -------------------------------------------------------------------------

    #[test]
    fn target_simulator_is_permissive() {
        assert!(!Target::Simulator.default_strict());
        assert_eq!(Target::Simulator.default_log_elision(), None);
    }

    #[test]
    fn target_hardware_is_strict() {
        assert!(Target::Hardware.default_strict());
        assert_eq!(Target::Hardware.default_log_elision(), Some(300)); // Warn+
    }

    #[test]
    fn target_emulator_is_strict_but_more_logs() {
        assert!(Target::Emulator.default_strict());
        assert_eq!(Target::Emulator.default_log_elision(), Some(200)); // Info+
    }

    // -------------------------------------------------------------------------
    // Mode behavior
    // -------------------------------------------------------------------------

    #[test]
    fn mode_debug_is_permissive() {
        assert!(!Mode::Debug.strict_bias());
        assert_eq!(Mode::Debug.log_elision_adjustment(), None);
    }

    #[test]
    fn mode_release_adds_strictness() {
        assert!(Mode::Release.strict_bias());
        assert_eq!(Mode::Release.log_elision_adjustment(), Some(100)); // Debug+
    }

    // -------------------------------------------------------------------------
    // Effective settings (target + mode combinations)
    // -------------------------------------------------------------------------

    #[test]
    fn simulator_debug_is_fully_permissive() {
        let (strict, log_elision) = effective_settings(Target::Simulator, Mode::Debug);
        assert!(!strict);
        assert_eq!(log_elision, None); // All logs
    }

    #[test]
    fn simulator_release_enables_strict_and_elides_trace() {
        let (strict, log_elision) = effective_settings(Target::Simulator, Mode::Release);
        assert!(strict); // Release adds strict
        assert_eq!(log_elision, Some(100)); // Debug+ (elide trace)
    }

    #[test]
    fn hardware_debug_is_strict_but_keeps_warn_logs() {
        let (strict, log_elision) = effective_settings(Target::Hardware, Mode::Debug);
        assert!(strict); // Hardware is always strict
        assert_eq!(log_elision, Some(300)); // Warn+ from target
    }

    #[test]
    fn hardware_release_is_strict_with_warn_logs() {
        let (strict, log_elision) = effective_settings(Target::Hardware, Mode::Release);
        assert!(strict);
        // Hardware default (300) > Release adjustment (100), so 300 wins
        assert_eq!(log_elision, Some(300));
    }

    #[test]
    fn emulator_debug_is_strict_with_info_logs() {
        let (strict, log_elision) = effective_settings(Target::Emulator, Mode::Debug);
        assert!(strict);
        assert_eq!(log_elision, Some(200)); // Info+
    }

    #[test]
    fn emulator_release_is_strict_with_info_logs() {
        let (strict, log_elision) = effective_settings(Target::Emulator, Mode::Release);
        assert!(strict);
        // Emulator default (200) > Release adjustment (100), so 200 wins
        assert_eq!(log_elision, Some(200));
    }

    // -------------------------------------------------------------------------
    // Output path derivation
    // -------------------------------------------------------------------------

    #[test]
    fn derive_output_path_slr() {
        let input = Path::new("/path/to/program.zlp");
        let output = derive_output_path(input, Format::Slr);
        assert_eq!(output, Path::new("/path/to/program.slr.json"));
    }

    #[test]
    fn derive_output_path_phir_json() {
        let input = Path::new("/path/to/program.zlp");
        let output = derive_output_path(input, Format::PhirJson);
        assert_eq!(output, Path::new("/path/to/program.phir.json"));
    }

    #[test]
    fn derive_output_path_qasm() {
        let input = Path::new("/path/to/program.zlp");
        let output = derive_output_path(input, Format::Qasm);
        assert_eq!(output, Path::new("/path/to/program.qasm"));
    }

    #[test]
    fn derive_output_path_stdin_returns_stdout() {
        let input = Path::new("-");
        let output = derive_output_path(input, Format::Slr);
        assert_eq!(output, Path::new("-"));
    }

    // -------------------------------------------------------------------------
    // Log level constants (for reference in tests)
    // -------------------------------------------------------------------------

    #[test]
    fn log_levels_are_spaced_by_100() {
        // Trace = 0, Debug = 100, Info = 200, Warn = 300, Error = 400
        // This documents the expected spacing
        assert_eq!(Target::Emulator.default_log_elision(), Some(200)); // Info
        assert_eq!(Target::Hardware.default_log_elision(), Some(300)); // Warn
        assert_eq!(Mode::Release.log_elision_adjustment(), Some(100)); // Debug
    }
}
