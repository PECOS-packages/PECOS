//! CLI for guppy-zlup - Guppy linter and Zlup compiler

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::mpsc::channel;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use guppy_zlup::{
    CompileError, Config, LintResult, Linter, OutputFormat, Severity, compile_file, ir, lint_source,
};
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode};
use rayon::prelude::*;

#[derive(Parser)]
#[command(name = "guppy-zlup")]
#[command(author, version, about = "Guppy linter and Zlup compiler")]
#[command(
    long_about = "Validate Guppy quantum programs against NASA Power of 10 rules and compile to Zlup"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Output format for lint results.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum CliOutputFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for machine parsing
    Json,
    /// SARIF format for GitHub Actions integration
    Sarif,
}

impl From<CliOutputFormat> for OutputFormat {
    fn from(f: CliOutputFormat) -> Self {
        match f {
            CliOutputFormat::Text => OutputFormat::Text,
            CliOutputFormat::Json => OutputFormat::Json,
            CliOutputFormat::Sarif => OutputFormat::Sarif,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Validate Guppy files using guppylang (requires Python with guppylang installed)
    Validate {
        /// Files to validate
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Check files for lint violations
    Check {
        /// Files or directories to check
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Treat warnings as errors
        #[arg(short = 'W', long)]
        warnings_as_errors: bool,

        /// Path to config file (pyproject.toml)
        #[arg(long)]
        config: Option<PathBuf>,

        /// Disable specific rules (can be used multiple times)
        #[arg(long = "disable", short = 'D')]
        disabled_rules: Vec<String>,

        /// Maximum complexity for ZLUP007
        #[arg(long)]
        max_complexity: Option<u32>,

        /// Output format
        #[arg(long, short = 'f', value_enum, default_value = "text")]
        format: CliOutputFormat,

        /// Watch for file changes and re-lint
        #[arg(long, short = 'w')]
        watch: bool,
    },

    /// Emit validated IR as JSON
    Emit {
        /// File to emit
        file: PathBuf,

        /// Output file (default: ir.json)
        #[arg(short, long, default_value = "ir.json")]
        output: PathBuf,

        /// Skip lint check
        #[arg(long)]
        skip_lint: bool,

        /// Print to stdout instead of writing to file
        #[arg(long)]
        stdout: bool,
    },

    /// Compile Guppy source or IR to Zlup
    Compile {
        /// Input file (.py for source, .json for IR)
        input: PathBuf,

        /// Output file (default: <input>.zlp)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Print to stdout instead of writing to file
        #[arg(long)]
        stdout: bool,

        /// Input is IR JSON (skip linting)
        #[arg(long)]
        ir: bool,

        /// Validate with guppylang before compiling (requires Python)
        #[arg(long)]
        validate: bool,

        /// Run parallelism analysis on generated Zlup
        #[arg(long)]
        analyze: bool,
    },

    /// Analyze Guppy source for parallelism opportunities (compiles to Zlup first)
    Analyze {
        /// Input file (.py for source, .json for IR)
        input: PathBuf,

        /// Input is IR JSON (skip linting)
        #[arg(long)]
        ir: bool,

        /// Output format (text or json)
        #[arg(short, long, value_enum, default_value = "text")]
        format: AnalyzeFormat,

        /// Show detailed dependency graph
        #[arg(long, short)]
        verbose: bool,
    },
}

/// Output format for analysis results.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum AnalyzeFormat {
    /// Human-readable text output
    #[default]
    Text,
    /// JSON output for tooling integration
    Json,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Commands::Validate { files, json } => cmd_validate(&files, json),

        Commands::Check {
            files,
            warnings_as_errors,
            config,
            disabled_rules,
            max_complexity,
            format,
            watch,
        } => {
            if watch {
                cmd_watch(
                    &files,
                    warnings_as_errors,
                    config.as_deref(),
                    &disabled_rules,
                    max_complexity,
                    format.into(),
                )
            } else {
                cmd_check(
                    &files,
                    warnings_as_errors,
                    config.as_deref(),
                    &disabled_rules,
                    max_complexity,
                    format.into(),
                )
            }
        }

        Commands::Emit {
            file,
            output,
            skip_lint,
            stdout,
        } => cmd_emit(&file, &output, skip_lint, stdout),

        Commands::Compile {
            input,
            output,
            stdout,
            ir,
            validate,
            analyze,
        } => {
            if ir {
                cmd_compile_ir(&input, output.as_deref(), stdout, analyze)
            } else {
                cmd_compile_source(&input, output.as_deref(), stdout, validate, analyze)
            }
        }

        Commands::Analyze {
            input,
            ir,
            format,
            verbose,
        } => cmd_analyze(&input, ir, format, verbose),
    }
}

/// Collect all Python files from paths (files or directories).
fn collect_python_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();

    for path in paths {
        if path.is_file() {
            files.push(path.clone());
        } else if path.is_dir() {
            // Recursively find all .py files
            if let Ok(entries) = walkdir(path) {
                files.extend(entries);
            }
        }
    }

    files
}

/// Walk directory recursively to find Python files.
fn walkdir(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Skip hidden directories and common non-source dirs
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.')
                && name != "__pycache__"
                && name != "node_modules"
                && name != "venv"
            {
                files.extend(walkdir(&path)?);
            }
        } else if path.extension().is_some_and(|ext| ext == "py") {
            files.push(path);
        }
    }

    Ok(files)
}

fn cmd_check(
    paths: &[PathBuf],
    warnings_as_errors: bool,
    config_path: Option<&Path>,
    disabled_rules: &[String],
    max_complexity: Option<u32>,
    format: OutputFormat,
) -> ExitCode {
    // Collect all Python files
    let files = collect_python_files(paths);

    if files.is_empty() {
        eprintln!("No Python files found.");
        return ExitCode::from(1);
    }

    // Load base config
    let mut base_config = if let Some(path) = config_path {
        match Config::from_pyproject(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error loading config from {}: {}", path.display(), e);
                return ExitCode::from(1);
            }
        }
    } else if let Some(first_file) = files.first() {
        if let Some(pyproject) = Config::find_pyproject(first_file) {
            match Config::from_pyproject(&pyproject) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Warning: Error loading {}: {}", pyproject.display(), e);
                    Config::default()
                }
            }
        } else {
            Config::default()
        }
    } else {
        Config::default()
    };

    // Apply CLI overrides
    if warnings_as_errors {
        base_config.treat_warnings_as_errors = true;
    }

    for rule in disabled_rules {
        base_config.disable_rule(rule);
    }

    if let Some(complexity) = max_complexity {
        base_config.max_complexity = complexity;
    }

    // Lint all files in parallel
    let results: Vec<(PathBuf, Result<LintResult, std::io::Error>)> = files
        .par_iter()
        .map(|file| {
            let result = std::fs::read_to_string(file).map(|source| {
                let linter = Linter::new(base_config.clone());
                linter.lint_source(&source, file.to_str().unwrap_or("<stdin>"))
            });
            (file.clone(), result)
        })
        .collect();

    // Merge all results
    let mut combined = LintResult::default();
    let mut io_errors = Vec::new();

    for (file, result) in results {
        match result {
            Ok(lint_result) => combined.merge(lint_result),
            Err(e) => io_errors.push((file, e)),
        }
    }

    // Report IO errors
    for (file, e) in &io_errors {
        eprintln!("Error reading {}: {}", file.display(), e);
    }

    // Output results in the requested format
    let output = combined.format(format);
    println!("{}", output);

    // Summary for text format
    if format == OutputFormat::Text && !combined.diagnostics.is_empty() {
        let error_count = combined
            .diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Error))
            .count();
        let warning_count = combined
            .diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Warning))
            .count();
        println!(
            "\nFound {} error(s) and {} warning(s) in {} file(s).",
            error_count,
            warning_count,
            files.len()
        );
    }

    if !io_errors.is_empty() || !combined.is_ok(base_config.treat_warnings_as_errors) {
        return ExitCode::from(1);
    }

    if format == OutputFormat::Text {
        println!("All checks passed!");
    }
    ExitCode::SUCCESS
}

fn cmd_watch(
    paths: &[PathBuf],
    warnings_as_errors: bool,
    config_path: Option<&Path>,
    disabled_rules: &[String],
    max_complexity: Option<u32>,
    format: OutputFormat,
) -> ExitCode {
    println!("Watching for changes... (press Ctrl+C to stop)");

    // Run initial check
    let _ = cmd_check(
        paths,
        warnings_as_errors,
        config_path,
        disabled_rules,
        max_complexity,
        format,
    );
    println!("\n---\n");

    // Set up file watcher
    let (tx, rx) = channel();

    let mut debouncer = match new_debouncer(Duration::from_millis(500), tx) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error creating file watcher: {}", e);
            return ExitCode::from(1);
        }
    };

    // Watch all paths
    for path in paths {
        let watch_path = if path.is_file() {
            path.parent().unwrap_or(path)
        } else {
            path.as_path()
        };

        if let Err(e) = debouncer
            .watcher()
            .watch(watch_path, RecursiveMode::Recursive)
        {
            eprintln!("Error watching {}: {}", watch_path.display(), e);
        }
    }

    // Convert config_path to owned PathBuf for the loop
    let config_path_owned = config_path.map(|p| p.to_path_buf());
    let disabled_rules_owned: Vec<String> = disabled_rules.to_vec();

    // Watch loop
    loop {
        match rx.recv() {
            Ok(Ok(events)) => {
                // Check if any Python files changed
                let has_py_changes = events
                    .iter()
                    .any(|e| e.path.extension().is_some_and(|ext| ext == "py"));

                if has_py_changes {
                    // Clear screen (optional, works on most terminals)
                    print!("\x1B[2J\x1B[1;1H");
                    println!("File changed, re-checking...\n");

                    let _ = cmd_check(
                        paths,
                        warnings_as_errors,
                        config_path_owned.as_deref(),
                        &disabled_rules_owned,
                        max_complexity,
                        format,
                    );
                    println!("\n---\nWatching for changes... (press Ctrl+C to stop)");
                }
            }
            Ok(Err(errors)) => {
                eprintln!("Watch error: {:?}", errors);
            }
            Err(e) => {
                eprintln!("Channel error: {}", e);
                return ExitCode::from(1);
            }
        }
    }
}

fn cmd_emit(file: &PathBuf, output: &PathBuf, skip_lint: bool, to_stdout: bool) -> ExitCode {
    let source = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::from(1);
        }
    };

    if !skip_lint {
        let config = Config::default();
        let linter = Linter::new(config);
        let result = linter.lint_source(&source, file.to_str().unwrap_or("<stdin>"));

        if result.has_errors {
            println!("{}", result);
            println!("\nCannot emit IR due to lint errors.");
            return ExitCode::from(1);
        }

        if result.has_warnings {
            println!("{}", result);
            println!();
        }
    }

    // Emit IR
    match ir::emit_ir(&source, file.to_str()) {
        Ok(ir_data) => {
            let json = serde_json::to_string_pretty(&ir_data).unwrap();

            if to_stdout {
                println!("{}", json);
                return ExitCode::SUCCESS;
            }

            if let Err(e) = std::fs::write(output, json) {
                eprintln!("Error writing IR: {}", e);
                return ExitCode::from(1);
            }
            println!("Wrote IR to {}", output.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error emitting IR: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_compile_ir(input: &Path, output: Option<&Path>, to_stdout: bool, analyze: bool) -> ExitCode {
    let result = match compile_file(input.to_str().unwrap_or("")) {
        Ok(zlup) => zlup,
        Err(e) => {
            eprintln!("Error: {}", format_error(&e));
            return ExitCode::from(1);
        }
    };

    if to_stdout {
        println!("{}", result);
        if analyze {
            eprintln!();
            run_analysis(&result, AnalyzeFormat::Text, false);
        }
        return ExitCode::SUCCESS;
    }

    let output_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("zlp"));

    if let Err(e) = std::fs::write(&output_path, &result) {
        eprintln!("Error writing output: {}", e);
        return ExitCode::from(1);
    }

    println!("Compiled {} -> {}", input.display(), output_path.display());

    if analyze {
        println!();
        run_analysis(&result, AnalyzeFormat::Text, false);
    }

    ExitCode::SUCCESS
}

fn cmd_compile_source(
    input: &PathBuf,
    output: Option<&Path>,
    to_stdout: bool,
    validate: bool,
    analyze: bool,
) -> ExitCode {
    // Run guppylang validation if requested
    if validate {
        let exit = cmd_validate(std::slice::from_ref(input), false);
        if exit != ExitCode::SUCCESS {
            eprintln!("\nCannot compile due to validation errors.");
            return exit;
        }
    }

    let source = match std::fs::read_to_string(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return ExitCode::from(1);
        }
    };

    // Run linter
    let result = lint_source(&source, input.to_str());
    if result.has_errors {
        println!("{}", result);
        let error_count = result
            .diagnostics
            .iter()
            .filter(|d| matches!(d.severity, Severity::Error))
            .count();
        println!("\nCannot compile due to {} lint error(s).", error_count);
        return ExitCode::from(1);
    }

    if result.has_warnings {
        println!("{}", result);
        println!();
    }

    // Emit IR
    let ir_data = match ir::emit_ir(&source, input.to_str()) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("Error emitting IR: {}", e);
            return ExitCode::from(1);
        }
    };

    // Compile IR to Zlup
    let ir_json = serde_json::to_string(&ir_data).unwrap();
    let zlup_source = match guppy_zlup::compile(&ir_json) {
        Ok(src) => src,
        Err(e) => {
            eprintln!("Error compiling to Zlup: {}", e);
            return ExitCode::from(1);
        }
    };

    if to_stdout {
        println!("{}", zlup_source);
        if analyze {
            eprintln!();
            run_analysis(&zlup_source, AnalyzeFormat::Text, false);
        }
        return ExitCode::SUCCESS;
    }

    let output_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| input.with_extension("zlp"));

    if let Err(e) = std::fs::write(&output_path, &zlup_source) {
        eprintln!("Error writing output: {}", e);
        return ExitCode::from(1);
    }

    println!("Compiled {} -> {}", input.display(), output_path.display());

    if analyze {
        println!();
        run_analysis(&zlup_source, AnalyzeFormat::Text, false);
    }

    ExitCode::SUCCESS
}

fn format_error(e: &CompileError) -> String {
    match e {
        CompileError::Io(io) => format!("IO error: {}", io),
        CompileError::Parse(parse) => format!("Parse error: {}", parse),
        CompileError::Transform(transform) => format!("Transform error: {}", transform),
    }
}

fn cmd_analyze(input: &PathBuf, is_ir: bool, format: AnalyzeFormat, verbose: bool) -> ExitCode {
    // Get Zlup source - either compile from Guppy/IR or read directly
    let zlup_source = if is_ir {
        match compile_file(input.to_str().unwrap_or("")) {
            Ok(zlup) => zlup,
            Err(e) => {
                eprintln!("Error compiling IR: {}", format_error(&e));
                return ExitCode::from(1);
            }
        }
    } else {
        let source = match std::fs::read_to_string(input) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading file: {}", e);
                return ExitCode::from(1);
            }
        };

        // Run linter first
        let result = lint_source(&source, input.to_str());
        if result.has_errors {
            println!("{}", result);
            let error_count = result
                .diagnostics
                .iter()
                .filter(|d| matches!(d.severity, Severity::Error))
                .count();
            println!("\nCannot analyze due to {} lint error(s).", error_count);
            return ExitCode::from(1);
        }

        // Emit IR and compile to Zlup
        let ir_data = match ir::emit_ir(&source, input.to_str()) {
            Ok(ir) => ir,
            Err(e) => {
                eprintln!("Error emitting IR: {}", e);
                return ExitCode::from(1);
            }
        };

        let ir_json = serde_json::to_string(&ir_data).unwrap();
        match guppy_zlup::compile(&ir_json) {
            Ok(src) => src,
            Err(e) => {
                eprintln!("Error compiling to Zlup: {}", e);
                return ExitCode::from(1);
            }
        }
    };

    run_analysis(&zlup_source, format, verbose);
    ExitCode::SUCCESS
}

fn run_analysis(zlup_source: &str, format: AnalyzeFormat, verbose: bool) {
    use zlup::analysis::{
        AllocatorAnalysis, DependencyGraph, OperationTagger, analyze_parallelism,
    };

    // Parse the Zlup source
    let program = match zlup::parse(zlup_source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error parsing generated Zlup: {}", e);
            return;
        }
    };

    // Run analysis passes
    let allocator_analysis = AllocatorAnalysis::analyze(&program);
    let tagger = OperationTagger::tag(&program);
    let dep_graph = DependencyGraph::build(tagger.operations);
    let summaries = analyze_parallelism(&program);

    match format {
        AnalyzeFormat::Text => {
            println!("=== Parallelism Analysis ===\n");

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
                "allocators": allocators,
                "functions": functions,
                "parallel_layers": layer_details,
                "total_operations": dep_graph.operations.len(),
                "total_dependencies": dep_graph.edges.len(),
            });

            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
    }
}

fn cmd_validate(files: &[PathBuf], json_output: bool) -> ExitCode {
    // Find the validation script relative to the executable or in known locations
    let script_locations = [
        // Relative to executable
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("scripts/validate_guppy.py"))),
        // In the source tree
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/validate_guppy.py")),
    ];

    let script_path = script_locations
        .iter()
        .filter_map(|p| p.as_ref())
        .find(|p| p.exists());

    let script_path = match script_path {
        Some(p) => p.clone(),
        None => {
            eprintln!("Error: Could not find validate_guppy.py script");
            eprintln!("Make sure guppy-zlup is properly installed");
            return ExitCode::from(1);
        }
    };

    let python_files = collect_python_files(files);
    if python_files.is_empty() {
        eprintln!("No Python files found.");
        return ExitCode::from(1);
    }

    let mut all_valid = true;
    let mut results: Vec<serde_json::Value> = Vec::new();

    for file in &python_files {
        // Try uv first, then fall back to python3
        let output = std::process::Command::new("uv")
            .args(["run", "python"])
            .arg(&script_path)
            .arg(file)
            .arg(if json_output { "--json" } else { "" })
            .output()
            .or_else(|_| {
                std::process::Command::new("python3")
                    .arg(&script_path)
                    .arg(file)
                    .arg(if json_output { "--json" } else { "" })
                    .output()
            });

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                if json_output {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                        results.push(json);
                    }
                } else {
                    if !stdout.is_empty() {
                        print!("{}", stdout);
                    }
                    if !stderr.is_empty() {
                        eprint!("{}", stderr);
                    }
                }

                if !out.status.success() {
                    all_valid = false;
                }
            }
            Err(e) => {
                eprintln!("Error running validation for {}: {}", file.display(), e);
                eprintln!(
                    "Make sure Python with guppylang is available (uv run python or python3)"
                );
                all_valid = false;
            }
        }
    }

    if json_output {
        let combined = serde_json::json!({
            "files": results,
            "all_valid": all_valid,
        });
        println!("{}", serde_json::to_string_pretty(&combined).unwrap());
    } else if all_valid {
        println!(
            "\nAll {} file(s) validated successfully!",
            python_files.len()
        );
    } else {
        println!("\nValidation failed for some files.");
    }

    if all_valid {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
