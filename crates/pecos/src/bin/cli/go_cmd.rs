//! Implementation of the `go` subcommand

use pecos_build::Result;
use pecos_build::errors::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Run the go subcommand
pub fn run(command: &super::GoCommands) -> Result<()> {
    match command {
        super::GoCommands::Check { quiet } => run_check(*quiet),
        super::GoCommands::Build { profile, rustflags } => run_build(profile, rustflags.as_deref()),
        super::GoCommands::Test => run_test(),
        super::GoCommands::Fmt { check } => run_fmt(*check),
        super::GoCommands::Lint => run_lint(),
    }
}

/// Check if Go is available
fn run_check(quiet: bool) -> Result<()> {
    match Command::new("go").args(["version"]).output() {
        Ok(output) if output.status.success() => {
            if !quiet {
                let version = String::from_utf8_lossy(&output.stdout);
                // Parse "go version go1.21.0 linux/amd64" to "go1.21.0 linux/amd64"
                let version = version
                    .trim()
                    .strip_prefix("go version ")
                    .unwrap_or(version.trim());
                println!("go: {version}");
            }
            Ok(())
        }
        _ => {
            if !quiet {
                eprintln!("go: not found");
            }
            Err(Error::Config("Go not available".to_string()))
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

/// Build Go FFI library
fn run_build(profile: &str, rustflags: Option<&str>) -> Result<()> {
    // Check Go is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Go is not installed. Please install Go to build the Go FFI library.".to_string(),
        ));
    }

    let repo_root = get_repo_root()?;
    let go_ffi_dir = repo_root.join("go/pecos-go-ffi");

    if !go_ffi_dir.exists() {
        return Err(Error::Config(format!(
            "Go FFI directory not found: {}",
            go_ffi_dir.display()
        )));
    }

    // Determine cargo profile flag
    let cargo_profile_flag: Vec<&str> = match profile {
        "native" => vec!["--profile", "native"],
        "release" => vec!["--release"],
        "debug" => vec![],
        _ => {
            return Err(Error::Config(format!(
                "Unknown profile: {profile}. Use debug, release, or native."
            )));
        }
    };

    println!("Building Go FFI library ({profile})...");

    let mut cmd = Command::new("cargo");
    cmd.arg("build").args(&cargo_profile_flag);
    cmd.current_dir(&go_ffi_dir);

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
            println!("Go FFI library built successfully");
            Ok(())
        }
        Ok(_) => Err(Error::Config("Go FFI build failed".to_string())),
        Err(e) => Err(Error::Config(format!(
            "Failed to run cargo for Go FFI: {e}"
        ))),
    }
}

/// Run Go tests
fn run_test() -> Result<()> {
    // Check Go is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Go is not installed. Please install Go to run tests.".to_string(),
        ));
    }

    // Build FFI library first
    println!("Building Go FFI library...");
    run_build("release", None)?;

    let repo_root = get_repo_root()?;
    let go_pkg = repo_root.join("go/pecos");

    if !go_pkg.exists() {
        return Err(Error::Config(format!(
            "Go package not found: {}",
            go_pkg.display()
        )));
    }

    println!("Running Go tests...");

    // Set LD_LIBRARY_PATH to include the release directory
    let lib_path = repo_root.join("target/release");
    let existing_lib_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let new_lib_path = if existing_lib_path.is_empty() {
        lib_path.display().to_string()
    } else {
        format!("{}:{existing_lib_path}", lib_path.display())
    };

    let status = Command::new("go")
        .args(["test", "-v"])
        .current_dir(&go_pkg)
        .env("LD_LIBRARY_PATH", &new_lib_path)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Go tests passed");
            Ok(())
        }
        Ok(_) => Err(Error::Config("Go tests failed".to_string())),
        Err(e) => Err(Error::Config(format!("Failed to run Go tests: {e}"))),
    }
}

/// Format Go code
fn run_fmt(check: bool) -> Result<()> {
    // Check Go is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Go is not installed. Please install Go to format code.".to_string(),
        ));
    }

    let repo_root = get_repo_root()?;
    let go_pkg = repo_root.join("go/pecos");

    if !go_pkg.exists() {
        return Err(Error::Config(format!(
            "Go package not found: {}",
            go_pkg.display()
        )));
    }

    if check {
        println!("Checking Go code formatting...");

        // gofmt -l returns list of files that need formatting
        let output = Command::new("gofmt")
            .args(["-l", "."])
            .current_dir(&go_pkg)
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if stdout.trim().is_empty() {
                    println!("All Go code is properly formatted.");
                    Ok(())
                } else {
                    eprintln!("Formatting issues found in:");
                    for line in stdout.lines() {
                        eprintln!("  {line}");
                    }
                    eprintln!("Run 'pecos go fmt' to fix.");
                    Err(Error::Config("Go formatting check failed".to_string()))
                }
            }
            Ok(_) => Err(Error::Config("gofmt failed".to_string())),
            Err(e) => Err(Error::Config(format!("Failed to run gofmt: {e}"))),
        }
    } else {
        println!("Formatting Go code...");

        let status = Command::new("gofmt")
            .args(["-w", "."])
            .current_dir(&go_pkg)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("Go code formatted successfully");
                Ok(())
            }
            Ok(_) => Err(Error::Config("gofmt failed".to_string())),
            Err(e) => Err(Error::Config(format!("Failed to run gofmt: {e}"))),
        }
    }
}

/// Run Go linting with go vet
fn run_lint() -> Result<()> {
    // Check Go is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Go is not installed. Please install Go to run linting.".to_string(),
        ));
    }

    let repo_root = get_repo_root()?;
    let go_pkg = repo_root.join("go/pecos");

    if !go_pkg.exists() {
        return Err(Error::Config(format!(
            "Go package not found: {}",
            go_pkg.display()
        )));
    }

    println!("Running Go linting...");

    let status = Command::new("go")
        .args(["vet", "./..."])
        .current_dir(&go_pkg)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Go linting passed");
            Ok(())
        }
        Ok(_) => Err(Error::Config("Go linting failed".to_string())),
        Err(e) => Err(Error::Config(format!("Failed to run go vet: {e}"))),
    }
}
