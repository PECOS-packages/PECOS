//! Implementation of the `python` subcommand

use pecos_build::Result;
use pecos_build::errors::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Run the python subcommand
pub fn run(command: &super::PythonCommands) -> Result<()> {
    match command {
        super::PythonCommands::Check { quiet } => run_check(*quiet),
        super::PythonCommands::Build {
            profile,
            rustflags,
            cuda,
        } => run_build(profile, rustflags.as_deref(), *cuda),
        super::PythonCommands::Test {
            markers,
            verbose,
            selene,
            numpy,
        } => run_test(markers.as_deref(), *verbose, *selene, *numpy),
    }
}

/// Check if Python and uv are available
fn run_check(quiet: bool) -> Result<()> {
    // Check uv first
    let uv_ok = match Command::new("uv").args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            if !quiet {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("uv: {}", version.trim());
            }
            true
        }
        _ => {
            if !quiet {
                eprintln!("uv: not found");
            }
            false
        }
    };

    // Check Python via uv
    let python_ok = match Command::new("uv")
        .args(["run", "python", "--version"])
        .output()
    {
        Ok(output) if output.status.success() => {
            if !quiet {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("python: {}", version.trim());
            }
            true
        }
        _ => {
            if !quiet {
                eprintln!("python: not found (via uv)");
            }
            false
        }
    };

    if uv_ok && python_ok {
        Ok(())
    } else {
        Err(Error::Config("Python/uv not available".to_string()))
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

/// Build all pecos rslib crates via maturin
fn run_build(profile: &str, rustflags: Option<&str>, cuda: bool) -> Result<()> {
    // Check Python/uv is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Python/uv is not available. Please install uv and set up a Python environment."
                .to_string(),
        ));
    }

    let repo_root = get_repo_root()?;

    // Determine maturin release flag
    let maturin_release = matches!(profile, "release" | "native");

    // Set RUSTFLAGS if provided or for native profile
    let mut flags = std::env::var("RUSTFLAGS").unwrap_or_default();
    if profile == "native" {
        if !flags.is_empty() {
            flags.push(' ');
        }
        flags.push_str("-C target-cpu=native");
    }
    if let Some(extra) = rustflags {
        if !flags.is_empty() {
            flags.push(' ');
        }
        flags.push_str(extra);
    }

    let venv_bin = repo_root.join(".venv/bin");
    let path_with_venv = format!(
        "{}:{}",
        venv_bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    // Build all rslib crates via maturin (incremental — cargo inside maturin
    // handles change detection, skips recompilation when nothing changed)
    let crates = ["pecos-rslib", "pecos-rslib-llvm"];
    for crate_name in crates {
        let crate_dir = repo_root.join(format!("python/{crate_name}"));
        if !crate_dir.exists() {
            continue;
        }

        println!(
            "Building {crate_name} ({}{})...",
            profile,
            if cuda && crate_name == "pecos-rslib" {
                " +cuda"
            } else {
                ""
            }
        );

        let maturin = venv_bin.join("maturin");
        let mut cmd = Command::new(&maturin);
        cmd.args(["develop", "--uv"]);
        if maturin_release {
            cmd.arg("--release");
        }
        cmd.current_dir(&crate_dir);
        if !flags.is_empty() {
            cmd.env("RUSTFLAGS", &flags);
        }
        cmd.env("PATH", &path_with_venv);
        cmd.env_remove("CONDA_PREFIX");

        let status = cmd.status();
        match status {
            Ok(s) if s.success() => {}
            Ok(_) => {
                return Err(Error::Config(format!(
                    "maturin develop failed for {crate_name}"
                )));
            }
            Err(e) => {
                return Err(Error::Config(format!(
                    "Failed to run maturin develop for {crate_name}: {e}"
                )));
            }
        }
    }

    // Install quantum-pecos in editable mode (--no-deps since rslib crates
    // are already installed by maturin develop above)
    println!("Installing quantum-pecos...");
    let mut pip_cmd = Command::new("uv");
    pip_cmd.args(["pip", "install", "--no-deps", "-e"]);

    if cuda {
        pip_cmd.arg("./python/quantum-pecos[all,cuda]");
    } else {
        pip_cmd.arg("./python/quantum-pecos[all]");
    }

    pip_cmd.current_dir(&repo_root);
    pip_cmd.env_remove("CONDA_PREFIX");

    let status = pip_cmd.status();
    match status {
        Ok(s) if s.success() => {
            println!("Python build completed successfully");
            Ok(())
        }
        Ok(_) => Err(Error::Config("quantum-pecos install failed".to_string())),
        Err(e) => Err(Error::Config(format!(
            "Failed to install quantum-pecos: {e}"
        ))),
    }
}

/// Run pytest
fn run_test(markers: Option<&str>, verbose: u8, selene: bool, numpy: bool) -> Result<()> {
    // Check Python/uv is available first
    if run_check(true).is_err() {
        return Err(Error::Config(
            "Python/uv is not available. Please install uv and set up a Python environment."
                .to_string(),
        ));
    }

    let repo_root = get_repo_root()?;

    // Determine which tests to run
    if selene {
        println!("Running Selene plugin tests...");
        run_pytest_dir(&repo_root, "python/selene-plugins", None, verbose, false)?;
    } else if numpy {
        println!("Running NumPy/SciPy compatibility tests...");
        run_pytest_dir(
            &repo_root,
            "python/pecos-rslib/tests",
            Some("numpy and not performance"),
            verbose,
            true,
        )?;
    } else {
        // Default: run core tests
        println!("Running pecos-rslib tests...");
        run_pytest_dir(
            &repo_root,
            "python/pecos-rslib/tests",
            markers.or(Some("not performance and not numpy")),
            verbose,
            false,
        )?;

        println!("Running quantum-pecos tests...");
        run_pytest_dir(
            &repo_root,
            "python/quantum-pecos/tests",
            markers.or(Some("not optional_dependency and not numpy")),
            verbose,
            false,
        )?;
    }

    println!("Python tests completed");
    Ok(())
}

/// Run pytest on a directory
fn run_pytest_dir(
    repo_root: &PathBuf,
    test_dir: &str,
    markers: Option<&str>,
    verbose: u8,
    numpy_compat: bool,
) -> Result<()> {
    let mut cmd = Command::new("uv");
    cmd.arg("run");

    if numpy_compat {
        cmd.args(["--group", "numpy-compat"]);
    }

    cmd.arg("pytest").arg(test_dir);

    if let Some(m) = markers {
        cmd.args(["-m", m]);
    }

    // Add verbosity flags
    for _ in 0..verbose {
        cmd.arg("-v");
    }

    cmd.current_dir(repo_root);

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(_) => Err(Error::Config(format!("pytest failed for {test_dir}"))),
        Err(e) => Err(Error::Config(format!("Failed to run pytest: {e}"))),
    }
}
