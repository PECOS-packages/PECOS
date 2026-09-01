//! Implementation of the `python` subcommand

use pecos_build::Result;
use pecos_build::errors::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Run the python subcommand
pub fn run(command: &super::PythonCommands) -> Result<()> {
    match command {
        super::PythonCommands::Build {
            profile,
            rustflags,
            cuda,
            no_cuda,
        } => {
            let cuda_resolved = resolve_cuda_choice(*cuda, *no_cuda);
            run_build(profile.as_str(), rustflags.as_deref(), cuda_resolved)
        }
    }
}

/// Decide whether to install CUDA Python packages for this build.
///
/// Decide whether `pecos python build` builds the CUDA (Rust) backend crate
/// (`pecos-rslib-cuda`).
///
/// - `--cuda`    -> build it. The caller opted in and is responsible for CUDA
///   setup first (e.g. `just build-cuda` runs `setup-quiet`, which installs the
///   cuQuantum SDK that crate's build needs).
/// - `--no-cuda` -> skip it.
/// - neither     -> do NOT build it. The auto-detect path (e.g. `just build-lite`)
///   does no CUDA setup, so building the backend here could fail or be slow. When a
///   toolkit + GPU are present, just print a notice on how to enable CUDA.
fn resolve_cuda_choice(cuda: bool, no_cuda: bool) -> bool {
    if cuda {
        return true;
    }
    if no_cuda {
        return false;
    }
    if super::cuda_cmd::should_install_cuda_python() {
        println!(
            "CUDA toolkit + NVIDIA GPU detected, but `pecos python build` only builds the \
             CUDA (Rust) backend when you pass --cuda. To enable CUDA: run \
             `pecos python build --cuda` (or `just build-cuda`) for the `pecos-rslib-cuda` \
             backend, and `uv sync --group cuda12|cuda13` (or `pecos cuda setup-python`) for \
             the CUDA Python packages."
        );
    }
    false
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

    // Map our profile name to maturin's cargo-profile flag. `dev`/`debug` use
    // cargo's default dev profile (no flag), `release` uses --release, `native`
    // uses --profile native so artifacts land in target/native/. Routing native
    // through --profile native (rather than --release with target-cpu RUSTFLAGS)
    // also lets the C++ build.rs files in pecos-pymatching/-chromobius/-tesseract
    // detect "native" via OUT_DIR and add -march=native to their C++ compilation.
    let cargo_profile_flag: &[&str] = match profile {
        "release" => &["--release"],
        "native" => &["--profile", "native"],
        "dev" | "debug" => &[],
        other => {
            return Err(Error::Config(format!(
                "Unknown profile: {other} (expected dev, debug, release, or native)"
            )));
        }
    };

    // Build up RUSTFLAGS. For native we inject -C target-cpu=native because
    // profile.native.rustflags in Cargo.toml is still gated on nightly; other
    // callers (Justfile julia-build/build-selene/bench) inject the
    // same flag so the resulting artifacts are consistent regardless of entry
    // point.
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

    // The mwpf decoder builds pure-Rust (its LP solver is the workspace
    // `highs` shim, patched in via the root Cargo.toml), so it is enabled by
    // default; PECOS_BUILD_MWPF=0 opts out.
    let mwpf_override = std::env::var("PECOS_BUILD_MWPF").ok();
    let mwpf_enabled = !matches!(mwpf_override.as_deref(), Some("0" | "false" | "no"));
    if !mwpf_enabled {
        println!("  (mwpf decoder disabled via PECOS_BUILD_MWPF)");
    }

    // Build all rslib crates via maturin (incremental — cargo inside maturin
    // handles change detection, skips recompilation when nothing changed).
    // The CUDA (Rust) backend is its own crate, built only on an explicit --cuda
    // (`cuda` is true only then -- see resolve_cuda_choice); the auto-detect path
    // does no CUDA setup, so it must not pull in pecos-rslib-cuda.
    let mut crates = vec!["pecos-rslib", "pecos-rslib-llvm"];
    if cuda {
        crates.push("pecos-rslib-cuda");
    }
    for crate_name in crates {
        let crate_dir = repo_root.join(format!("python/{crate_name}"));
        if !crate_dir.exists() {
            continue;
        }

        println!("Building {crate_name} ({profile})...");

        remove_stale_extension_artifacts(&repo_root, profile, crate_name)?;

        let maturin = venv_bin.join("maturin");
        let mut cmd = Command::new(&maturin);
        cmd.args(["develop", "--uv", "--locked"]);
        cmd.args(cargo_profile_flag);
        // Maturin's CLI --features REPLACES (not merges with) the features list
        // in pyproject.toml's [tool.maturin], so any time we pass extra features
        // we must also pass `extension-module` -- otherwise the cdylib loses
        // pyo3's extension-module + abi3 settings and the resulting wheel either
        // links libpython directly (wrong) or fails entirely on machines without
        // a linkable libpython. The same applies to CI's MATURIN_PEP517_ARGS.
        if mwpf_enabled && crate_name == "pecos-rslib" {
            cmd.args(["--features", "extension-module,mwpf"]);
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

        // Apply PECOS build environment (SDKROOT, LLVM, CUDA, etc.)
        // This is the single source of truth — same logic as `pecos env`.
        let build_env = super::env_cmd::collect_env();
        for (key, value) in &build_env {
            // Don't override PATH — we already set it above with venv
            if key != "PATH" {
                cmd.env(key, value);
            }
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

    // `--no-deps` (above) means this editable install pulls no dependencies, so
    // naming a CUDA extra here would be inert: the CUDA Python stack
    // (cupy/cuquantum/pytket-cutensornet) is installed separately via
    // `uv sync --group cuda12|cuda13` (`just build`'s sync-deps, `pecos setup`, or
    // `pecos cuda setup-python`), not by this command. Request the dependency-free
    // `[all]` extra, which exists regardless of CUDA toolkit major and avoids an
    // unknown-extra warning.
    pip_cmd.arg("./python/quantum-pecos[all]");

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

fn cargo_profile_dir(profile: &str) -> &'static str {
    match profile {
        "release" => "release",
        "native" => "native",
        _ => "debug",
    }
}

fn extension_library_filename(crate_name: &str) -> String {
    let module_name = crate_name.replace('-', "_");

    #[cfg(target_os = "windows")]
    {
        format!("{module_name}.dll")
    }

    #[cfg(target_os = "macos")]
    {
        format!("lib{module_name}.dylib")
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        format!("lib{module_name}.so")
    }
}

fn extension_artifact_candidates(
    repo_root: &Path,
    profile: &str,
    crate_name: &str,
) -> [PathBuf; 3] {
    let filename = extension_library_filename(crate_name);
    let target_dir = repo_root.join("target");
    let profile_dir = target_dir.join(cargo_profile_dir(profile));

    [
        profile_dir.join(&filename),
        profile_dir.join("deps").join(&filename),
        target_dir.join("maturin").join(&filename),
    ]
}

fn remove_stale_extension_artifacts(
    repo_root: &Path,
    profile: &str,
    crate_name: &str,
) -> Result<()> {
    let [profile_artifact, deps_artifact, maturin_staging_artifact] =
        extension_artifact_candidates(repo_root, profile, crate_name);

    for path in [profile_artifact, deps_artifact] {
        match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() && metadata.len() == 0 => {
                println!(
                    "Removing zero-byte extension artifact before rebuild: {}",
                    path.display()
                );
                fs::remove_file(path)?;
            }
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }

    match fs::metadata(&maturin_staging_artifact) {
        Ok(metadata) if metadata.is_file() => {
            println!(
                "Removing stale maturin staging artifact before rebuild: {}",
                maturin_staging_artifact.display()
            );
            fs::remove_file(maturin_staging_artifact)?;
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cargo_profile_dir_matches_cargos_target_subdir() {
        assert_eq!(cargo_profile_dir("dev"), "debug");
        assert_eq!(cargo_profile_dir("debug"), "debug");
        assert_eq!(cargo_profile_dir("release"), "release");
        assert_eq!(cargo_profile_dir("native"), "native");
    }

    #[test]
    fn extension_artifact_candidates_use_python_module_name() {
        let repo = PathBuf::from("/repo");
        let candidates = extension_artifact_candidates(&repo, "debug", "pecos-rslib-llvm");
        let filename = extension_library_filename("pecos-rslib-llvm");

        assert!(filename.contains("pecos_rslib_llvm"));
        assert_eq!(candidates[0], repo.join("target/debug").join(&filename));
        assert_eq!(
            candidates[1],
            repo.join("target/debug/deps").join(&filename)
        );
        assert_eq!(candidates[2], repo.join("target/maturin").join(&filename));
    }
}
