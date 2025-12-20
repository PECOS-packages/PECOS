//! Implementation of the `rust` subcommand (CUDA-aware cargo commands)

use cargo_metadata::MetadataCommand;
use pecos_build::Result;
use pecos_build::errors::Error;
use std::collections::BTreeSet;
use std::process::Command;

/// FFI crates that should be excluded from workspace-wide cargo commands
const FFI_CRATES: &[&str] = &["pecos-rslib", "pecos-julia-ffi", "pecos-go-ffi"];

/// Run the rust subcommand
pub fn run(command: &super::RustCommands) -> Result<()> {
    match command {
        super::RustCommands::Check { include_ffi } => run_check(*include_ffi),
        super::RustCommands::Clippy { include_ffi, fix } => run_clippy(*include_ffi, *fix),
        super::RustCommands::Test {
            release,
            include_ffi,
        } => run_test(*release, *include_ffi),
        super::RustCommands::Fmt { check } => run_fmt(*check),
    }
}

/// Check if CUDA is available (local ~/.pecos/cuda/ or system)
fn is_cuda_available() -> bool {
    pecos_build::cuda::is_cuda_available()
}

/// Check if a tool is available
fn is_tool_available(tool: &str) -> bool {
    Command::new(tool)
        .args(["--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Get features for a package excluding certain features
///
/// Uses `cargo_metadata` directly instead of spawning a subprocess to avoid
/// Windows file locking issues (running executable can't be replaced).
fn get_features_excluding(package: &str, exclude: &str) -> Result<String> {
    let metadata = MetadataCommand::new()
        .no_deps()
        .exec()
        .map_err(|e| Error::Config(format!("Failed to get cargo metadata: {e}")))?;

    let pkg = metadata
        .packages
        .iter()
        .find(|p| p.name == package)
        .ok_or_else(|| Error::Config(format!("Package '{package}' not found in workspace")))?;

    let exclusions: BTreeSet<&str> = exclude.split(',').map(str::trim).collect();

    let features: Vec<&str> = pkg
        .features
        .keys()
        .map(String::as_str)
        .filter(|f| !exclusions.contains(f))
        .collect();

    Ok(features.join(","))
}

/// Run a cargo command and return success status
fn run_cargo_command(args: &[&str]) -> bool {
    let status = Command::new("cargo").args(args).status();
    matches!(status, Ok(s) if s.success())
}

/// Run cargo check with CUDA-aware feature handling
#[allow(clippy::too_many_lines)]
fn run_check(include_ffi: bool) -> Result<()> {
    let cuda_available = is_cuda_available();

    if cuda_available {
        println!("CUDA detected - checking with all features");

        let mut args: Vec<&str> = vec!["check", "--workspace", "--all-targets", "--all-features"];

        let exclude_flags: Vec<String>;
        if !include_ffi {
            exclude_flags = FFI_CRATES
                .iter()
                .map(|c| format!("--exclude={c}"))
                .collect();
            for flag in &exclude_flags {
                args.push(flag);
            }
        }

        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo check failed".to_string()));
        }
    } else {
        println!("CUDA not detected - checking all features except GPU");

        println!(
            "Checking workspace packages (excluding FFI crates and those with GPU features)..."
        );
        let mut args: Vec<&str> = vec![
            "check",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--exclude=pecos",
            "--exclude=pecos-quest",
            // pecos-selene-quest has cuda feature that enables pecos-quest/cuda
            "--exclude=pecos-selene-quest",
            // benchmarks depends on pecos, and --all-features enables pecos/cuda
            "--exclude=benchmarks",
        ];

        let exclude_flags: Vec<String> = FFI_CRATES
            .iter()
            .map(|c| format!("--exclude={c}"))
            .collect();
        for flag in &exclude_flags {
            args.push(flag);
        }

        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo check (workspace) failed".to_string()));
        }

        println!("Checking pecos with all features except cuda...");
        let pecos_features = get_features_excluding("pecos", "cuda")?;
        let features_arg = format!("--features={pecos_features}");
        if !run_cargo_command(&["check", "-p", "pecos", "--all-targets", &features_arg]) {
            return Err(Error::Config("cargo check (pecos) failed".to_string()));
        }

        println!("Checking pecos-quest with all features except cuda...");
        let quest_features = get_features_excluding("pecos-quest", "cuda")?;
        let features_arg = format!("--features={quest_features}");
        if !run_cargo_command(&["check", "-p", "pecos-quest", "--all-targets", &features_arg]) {
            return Err(Error::Config(
                "cargo check (pecos-quest) failed".to_string(),
            ));
        }

        println!("Checking pecos-selene-quest without cuda...");
        let selene_quest_features = get_features_excluding("pecos-selene-quest", "cuda")?;
        let features_arg = format!("--features={selene_quest_features}");
        if !run_cargo_command(&[
            "check",
            "-p",
            "pecos-selene-quest",
            "--all-targets",
            &features_arg,
        ]) {
            return Err(Error::Config(
                "cargo check (pecos-selene-quest) failed".to_string(),
            ));
        }
    }

    if include_ffi {
        println!("Checking pecos-rslib...");
        // Only use --all-features if CUDA is available, otherwise exclude cuda
        if cuda_available {
            if !run_cargo_command(&[
                "check",
                "-p",
                "pecos-rslib",
                "--all-targets",
                "--all-features",
            ]) {
                return Err(Error::Config(
                    "cargo check (pecos-rslib) failed".to_string(),
                ));
            }
        } else {
            let rslib_features = get_features_excluding("pecos-rslib", "cuda")?;
            let features_arg = format!("--features={rslib_features}");
            if !run_cargo_command(&["check", "-p", "pecos-rslib", "--all-targets", &features_arg]) {
                return Err(Error::Config(
                    "cargo check (pecos-rslib) failed".to_string(),
                ));
            }
        }

        if is_tool_available("julia") {
            println!("Checking pecos-julia-ffi...");
            if !run_cargo_command(&[
                "check",
                "-p",
                "pecos-julia-ffi",
                "--all-targets",
                "--all-features",
            ]) {
                return Err(Error::Config(
                    "cargo check (pecos-julia-ffi) failed".to_string(),
                ));
            }
        }

        if is_tool_available("go") {
            println!("Checking pecos-go-ffi...");
            if !run_cargo_command(&[
                "check",
                "-p",
                "pecos-go-ffi",
                "--all-targets",
                "--all-features",
            ]) {
                return Err(Error::Config(
                    "cargo check (pecos-go-ffi) failed".to_string(),
                ));
            }
        }
    }

    println!();
    println!("cargo check completed successfully");
    Ok(())
}

/// Run cargo clippy with CUDA-aware feature handling
#[allow(clippy::too_many_lines)]
fn run_clippy(include_ffi: bool, fix: bool) -> Result<()> {
    let cuda_available = is_cuda_available();

    let fix_args: Vec<&str> = if fix {
        vec!["--fix", "--allow-staged", "--allow-dirty"]
    } else {
        vec![]
    };

    if cuda_available {
        println!("CUDA detected - running clippy with all features");

        let mut args: Vec<&str> = vec!["clippy", "--workspace", "--all-targets", "--all-features"];
        args.extend(&fix_args);

        let exclude_flags: Vec<String>;
        if !include_ffi {
            exclude_flags = FFI_CRATES
                .iter()
                .map(|c| format!("--exclude={c}"))
                .collect();
            for flag in &exclude_flags {
                args.push(flag);
            }
        }

        args.extend(&["--", "-D", "warnings"]);

        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo clippy failed".to_string()));
        }
    } else {
        println!("CUDA not detected - running clippy on all features except CUDA");

        println!(
            "Running clippy on workspace packages (excluding FFI crates and those with CUDA features)..."
        );
        let mut args: Vec<&str> = vec![
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--exclude=pecos",
            "--exclude=pecos-quest",
            // pecos-selene-quest has cuda feature that enables pecos-quest/cuda
            "--exclude=pecos-selene-quest",
            // benchmarks depends on pecos, and --all-features enables pecos/cuda
            "--exclude=benchmarks",
        ];
        args.extend(&fix_args);

        let exclude_flags: Vec<String> = FFI_CRATES
            .iter()
            .map(|c| format!("--exclude={c}"))
            .collect();
        for flag in &exclude_flags {
            args.push(flag);
        }

        args.extend(&["--", "-D", "warnings"]);

        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo clippy (workspace) failed".to_string()));
        }

        println!("Running clippy on pecos with all features except cuda...");
        let pecos_features = get_features_excluding("pecos", "cuda")?;
        let features_arg = format!("--features={pecos_features}");
        let mut args: Vec<&str> = vec!["clippy", "-p", "pecos", "--all-targets", &features_arg];
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo clippy (pecos) failed".to_string()));
        }

        println!("Running clippy on pecos-quest with all features except cuda...");
        let quest_features = get_features_excluding("pecos-quest", "cuda")?;
        let features_arg = format!("--features={quest_features}");
        let mut args: Vec<&str> = vec![
            "clippy",
            "-p",
            "pecos-quest",
            "--all-targets",
            &features_arg,
        ];
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo clippy (pecos-quest) failed".to_string(),
            ));
        }

        println!("Running clippy on pecos-selene-quest without cuda...");
        let selene_quest_features = get_features_excluding("pecos-selene-quest", "cuda")?;
        let features_arg = format!("--features={selene_quest_features}");
        let mut args: Vec<&str> = vec![
            "clippy",
            "-p",
            "pecos-selene-quest",
            "--all-targets",
            &features_arg,
        ];
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo clippy (pecos-selene-quest) failed".to_string(),
            ));
        }
    }

    if include_ffi {
        println!("Running clippy on pecos-rslib...");
        let mut args: Vec<&str> = vec!["clippy", "-p", "pecos-rslib", "--all-targets"];
        // Only use --all-features if CUDA is available, otherwise exclude cuda
        if cuda_available {
            args.push("--all-features");
        } else {
            let rslib_features = get_features_excluding("pecos-rslib", "cuda")?;
            let features_arg_owned = format!("--features={rslib_features}");
            // Need to leak the string to get a &'static str for the args vec
            let features_arg: &'static str = Box::leak(features_arg_owned.into_boxed_str());
            args.push(features_arg);
        }
        args.extend(&fix_args);
        args.extend(&["--", "-D", "warnings"]);
        if !run_cargo_command(&args) {
            return Err(Error::Config(
                "cargo clippy (pecos-rslib) failed".to_string(),
            ));
        }

        if is_tool_available("julia") {
            println!("Running clippy on pecos-julia-ffi...");
            let mut args: Vec<&str> = vec![
                "clippy",
                "-p",
                "pecos-julia-ffi",
                "--all-targets",
                "--all-features",
            ];
            args.extend(&fix_args);
            args.extend(&["--", "-D", "warnings"]);
            if !run_cargo_command(&args) {
                return Err(Error::Config(
                    "cargo clippy (pecos-julia-ffi) failed".to_string(),
                ));
            }
        }

        if is_tool_available("go") {
            println!("Running clippy on pecos-go-ffi...");
            let mut args: Vec<&str> = vec![
                "clippy",
                "-p",
                "pecos-go-ffi",
                "--all-targets",
                "--all-features",
            ];
            args.extend(&fix_args);
            args.extend(&["--", "-D", "warnings"]);
            if !run_cargo_command(&args) {
                return Err(Error::Config(
                    "cargo clippy (pecos-go-ffi) failed".to_string(),
                ));
            }
        }
    }

    println!();
    println!("cargo clippy completed successfully");
    Ok(())
}

/// Run cargo test with CUDA-aware feature handling
fn run_test(release: bool, include_ffi: bool) -> Result<()> {
    let cuda_available = is_cuda_available();
    let release_flag = if release { "--release" } else { "" };

    println!("Testing workspace packages...");
    // runtime = sim + qasm + phir (format parsers)
    // hugr = qis (includes llvm) + hugr compilation
    let mut args: Vec<&str> = vec!["test", "--workspace", "--features=runtime,hugr"];

    for crate_name in FFI_CRATES {
        args.push("--exclude");
        args.push(crate_name);
    }

    args.extend(&["--exclude", "pecos-quest", "--exclude", "pecos-decoders"]);

    if !release_flag.is_empty() {
        args.push(release_flag);
    }

    if !run_cargo_command(&args) {
        return Err(Error::Config("cargo test (workspace) failed".to_string()));
    }

    if cuda_available {
        println!("CUDA detected - testing pecos-quest with all features");
        let mut args = vec!["test", "-p", "pecos-quest", "--all-features"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo test (pecos-quest) failed".to_string()));
        }
    } else {
        println!("CUDA not detected - testing pecos-quest with cpu features only");
        let mut args = vec!["test", "-p", "pecos-quest", "--features=cpu"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo test (pecos-quest) failed".to_string()));
        }
    }

    println!("Testing pecos-decoders...");
    let mut args = vec!["test", "-p", "pecos-decoders", "--all-features"];
    if !release_flag.is_empty() {
        args.push(release_flag);
    }
    if !run_cargo_command(&args) {
        return Err(Error::Config(
            "cargo test (pecos-decoders) failed".to_string(),
        ));
    }

    if include_ffi {
        println!("Testing pecos-rslib...");
        let mut args = vec!["test", "-p", "pecos-rslib", "--all-features"];
        if !release_flag.is_empty() {
            args.push(release_flag);
        }
        if !run_cargo_command(&args) {
            return Err(Error::Config("cargo test (pecos-rslib) failed".to_string()));
        }
    }

    println!();
    println!("cargo test completed successfully");
    Ok(())
}

/// Run cargo fmt
fn run_fmt(check: bool) -> Result<()> {
    let mut args = vec!["fmt", "--all"];
    if check {
        args.extend(&["--", "--check"]);
    }

    if !run_cargo_command(&args) {
        if check {
            return Err(Error::Config(
                "cargo fmt check failed - formatting issues found".to_string(),
            ));
        }
        return Err(Error::Config("cargo fmt failed".to_string()));
    }

    if check {
        println!("All Rust code is properly formatted");
    } else {
        println!("Rust code formatted successfully");
    }
    Ok(())
}
