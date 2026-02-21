//! Implementation of the `julia` subcommand

use pecos_build::Result;
use pecos_build::errors::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Run the julia subcommand
pub fn run(command: &super::JuliaCommands) -> Result<()> {
    match command {
        super::JuliaCommands::Check { quiet } => run_check(*quiet),
        super::JuliaCommands::Build { profile, rustflags } => {
            run_build(profile, rustflags.as_deref())
        }
        super::JuliaCommands::Test => run_test(),
        super::JuliaCommands::Fmt { check } => run_fmt(*check),
        super::JuliaCommands::Lint => run_lint(),
    }
}

/// Check if Julia is available
fn run_check(quiet: bool) -> Result<()> {
    match Command::new("julia").args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            if !quiet {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("julia: {}", version.trim());
            }
            Ok(())
        }
        _ => {
            if !quiet {
                eprintln!("julia: not found");
            }
            Err(Error::Config("Julia not available".to_string()))
        }
    }
}

/// Get the repository root
fn get_repo_root() -> Result<PathBuf> {
    let mut current = std::env::current_dir()?;

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            let content = fs::read_to_string(&cargo_toml)?;
            if content.contains("[workspace]") {
                return Ok(current);
            }
        }

        if !current.pop() {
            return Err(Error::Config(
                "Could not find PECOS repository root".to_string(),
            ));
        }
    }
}

/// Build Julia FFI library
fn run_build(profile: &str, rustflags: Option<&str>) -> Result<()> {
    // Check Julia is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Julia is not installed. Please install Julia to build the Julia FFI library."
                .to_string(),
        ));
    }

    let repo_root = get_repo_root()?;
    let julia_ffi_dir = repo_root.join("julia/pecos-julia-ffi");

    if !julia_ffi_dir.exists() {
        return Err(Error::Config(format!(
            "Julia FFI directory not found: {}",
            julia_ffi_dir.display()
        )));
    }

    // Determine cargo profile flag
    // Note: "dev" and "debug" are equivalent - Cargo calls it "dev" but outputs to target/debug/
    let cargo_profile_flag: Vec<&str> = match profile {
        "native" => vec!["--profile", "native"],
        "release" => vec!["--release"],
        "dev" | "debug" => vec![],
        _ => {
            return Err(Error::Config(format!(
                "Unknown profile: {profile}. Use dev/debug, release, or native."
            )));
        }
    };

    println!("Building Julia FFI library ({profile})...");

    let mut cmd = Command::new("cargo");
    cmd.arg("build").args(&cargo_profile_flag);
    cmd.current_dir(&julia_ffi_dir);

    // Set RUSTFLAGS if provided
    if let Some(flags) = rustflags {
        let existing = std::env::var("RUSTFLAGS").unwrap_or_default();
        let new_flags = if existing.is_empty() {
            flags.to_string()
        } else {
            format!("{existing} {flags}")
        };
        cmd.env("RUSTFLAGS", new_flags);
    }

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => {
            println!("Julia FFI library built successfully");
            Ok(())
        }
        Ok(_) => Err(Error::Config("Julia FFI build failed".to_string())),
        Err(e) => Err(Error::Config(format!(
            "Failed to run cargo for Julia FFI: {e}"
        ))),
    }
}

/// Run Julia tests
fn run_test() -> Result<()> {
    // Check Julia is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Julia is not installed. Please install Julia to run tests.".to_string(),
        ));
    }

    // Build FFI library first
    println!("Building Julia FFI library...");
    run_build("release", None)?;

    let repo_root = get_repo_root()?;
    let julia_pkg = repo_root.join("julia/PECOS.jl");

    if !julia_pkg.exists() {
        return Err(Error::Config(format!(
            "Julia package not found: {}",
            julia_pkg.display()
        )));
    }

    println!("Running Julia tests...");

    let status = Command::new("julia")
        .args([
            "--project=.",
            "-e",
            "using Pkg; Pkg.instantiate(); include(\"test/runtests.jl\")",
        ])
        .current_dir(&julia_pkg)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Julia tests passed");
            Ok(())
        }
        Ok(_) => Err(Error::Config("Julia tests failed".to_string())),
        Err(e) => Err(Error::Config(format!("Failed to run Julia tests: {e}"))),
    }
}

/// Format Julia code
fn run_fmt(check: bool) -> Result<()> {
    // Check Julia is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Julia is not installed. Please install Julia to format code.".to_string(),
        ));
    }

    let repo_root = get_repo_root()?;
    let julia_pkg = repo_root.join("julia/PECOS.jl");

    if !julia_pkg.exists() {
        return Err(Error::Config(format!(
            "Julia package not found: {}",
            julia_pkg.display()
        )));
    }

    if check {
        println!("Checking Julia code formatting...");
    } else {
        println!("Formatting Julia code...");
    }

    // First, ensure JuliaFormatter is installed in the default environment
    // (not the project environment, to avoid modifying Project.toml)
    let install_formatter = r#"
        using Pkg
        # Install to default environment, not project
        Pkg.activate()
        if !haskey(Pkg.project().dependencies, "JuliaFormatter")
            Pkg.add("JuliaFormatter")
        end
        "#;

    let install_status = Command::new("julia")
        .args(["-e", install_formatter])
        .current_dir(&julia_pkg)
        .status();

    if !matches!(install_status, Ok(s) if s.success()) {
        return Err(Error::Config(
            "Failed to install JuliaFormatter".to_string(),
        ));
    }

    // Now run the formatter using JuliaFormatter from default env
    // but operating on the project directory
    let julia_code = if check {
        r#"
        using JuliaFormatter
        if !format("."; verbose=false, overwrite=false)
            println("Formatting issues found. Run 'pecos julia fmt' to fix.")
            exit(1)
        else
            println("All Julia code is properly formatted.")
        end
        "#
    } else {
        r#"
        using JuliaFormatter
        format("."; verbose=true)
        "#
    };

    let status = Command::new("julia")
        .args(["-e", julia_code])
        .current_dir(&julia_pkg)
        .status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(_) => Err(Error::Config("Julia formatting check failed".to_string())),
        Err(e) => Err(Error::Config(format!("Failed to run Julia formatter: {e}"))),
    }
}

/// Run Julia linting with Aqua.jl
fn run_lint() -> Result<()> {
    // Check Julia is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Julia is not installed. Please install Julia to run linting.".to_string(),
        ));
    }

    // Build FFI library first
    println!("Building Julia FFI library...");
    run_build("release", None)?;

    let repo_root = get_repo_root()?;
    let julia_pkg = repo_root.join("julia/PECOS.jl");

    if !julia_pkg.exists() {
        return Err(Error::Config(format!(
            "Julia package not found: {}",
            julia_pkg.display()
        )));
    }

    println!("Running Julia code quality checks with Aqua.jl...");

    let status = Command::new("julia")
        .args(["--project=.", "test/aqua_tests.jl"])
        .current_dir(&julia_pkg)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Julia linting passed");
            Ok(())
        }
        Ok(_) => Err(Error::Config("Julia linting failed".to_string())),
        Err(e) => Err(Error::Config(format!("Failed to run Julia linting: {e}"))),
    }
}
