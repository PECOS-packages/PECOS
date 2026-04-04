//! Implementation of the `python` subcommand

use pecos_build::Result;
use pecos_build::errors::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Run the python subcommand
pub fn run(command: &super::PythonCommands) -> Result<()> {
    match command {
        super::PythonCommands::Build {
            profile,
            rustflags,
            cuda,
        } => run_build(profile, rustflags.as_deref(), *cuda),
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

/// Check if Python and uv are available (internal helper)
fn check_python_available() -> Result<()> {
    let uv_ok = Command::new("uv")
        .args(["--version"])
        .output()
        .is_ok_and(|o| o.status.success());

    let python_ok = Command::new("uv")
        .args(["run", "python", "--version"])
        .output()
        .is_ok_and(|o| o.status.success());

    if uv_ok && python_ok {
        Ok(())
    } else {
        Err(Error::Config("Python/uv not available".to_string()))
    }
}

/// Build all pecos rslib crates via maturin
fn run_build(profile: &str, rustflags: Option<&str>, cuda: bool) -> Result<()> {
    if check_python_available().is_err() {
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

    let venv_bin = if cfg!(windows) {
        repo_root.join(".venv/Scripts")
    } else {
        repo_root.join(".venv/bin")
    };
    let path_sep = if cfg!(windows) { ";" } else { ":" };
    let path_with_venv = format!(
        "{}{}{}",
        venv_bin.display(),
        path_sep,
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
        // On macOS, add rpath for system libc++ and clean Homebrew paths
        // (cdylibs linking inkwell/LLVM reference @rpath/libc++.1.dylib)
        #[cfg(target_os = "macos")]
        {
            if !flags.contains("-rpath") {
                let rpath_flag = " -C link-arg=-Wl,-rpath,/usr/lib";
                flags.push_str(rpath_flag);
            }
        }

        if !flags.is_empty() {
            cmd.env("RUSTFLAGS", &flags);
        }
        cmd.env("PATH", &path_with_venv);
        cmd.env_remove("CONDA_PREFIX");
        #[cfg(target_os = "macos")]
        {
            cmd.env_remove("LIBRARY_PATH");
            cmd.env_remove("LD_LIBRARY_PATH");
            cmd.env_remove("DYLD_LIBRARY_PATH");
            cmd.env_remove("DYLD_FALLBACK_LIBRARY_PATH");
            cmd.env("LIBRARY_PATH", "/usr/lib");
        }

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
