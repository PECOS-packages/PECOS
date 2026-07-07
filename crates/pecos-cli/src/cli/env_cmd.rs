// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Implementation of the `env` subcommand.
//!
//! Prints the build environment variables for the current platform. This is
//! the single source of truth for platform-specific build configuration.
//! CI workflows, Justfile recipes, and `pecos python build` should all derive
//! their environment from this command.
//!
//! Usage:
//!   eval $(pecos env)           # bash/zsh — set variables in current shell
//!   pecos env --format json     # machine-readable output
//!   pecos env --format show     # human-readable display
//!   pecos env --github-actions  # write GitHub Actions env/path files

use std::collections::BTreeMap;
use std::fmt::Write;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

#[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "macos"))]
use std::ffi::OsString;

use pecos_build::Result;
use pecos_build::errors::Error;
use pecos_build::llvm::LLVM_SYS_PREFIX_ENV;

/// Collect the build environment for the current platform.
///
/// Returns a map of environment variable names to values. Only includes
/// variables that PECOS needs to set — does not duplicate the entire shell
/// environment.
pub fn collect_env() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();

    // LLVM
    if let Some(llvm_path) = pecos_build::llvm::find_configured_or_detected_llvm(None) {
        let llvm_str = pecos_build::llvm::path_to_env_string(&llvm_path);
        env.insert("PECOS_LLVM".into(), llvm_str.clone());
        env.insert(LLVM_SYS_PREFIX_ENV.into(), llvm_str);

        // Add LLVM bin to PATH
        let bin_path = llvm_path.join("bin");
        if bin_path.exists() {
            let current_path = std::env::var_os("PATH").unwrap_or_default();
            let path_entries =
                std::iter::once(bin_path).chain(std::env::split_paths(&current_path));
            if let Ok(path) = std::env::join_paths(path_entries) {
                env.insert("PATH".into(), path.to_string_lossy().into_owned());
            }
        }

        if let Ok(libdir) = pecos_build::llvm::get_llvm_libdir(&llvm_path) {
            if pecos_build::llvm::get_llvm_shared_mode(&llvm_path)
                .is_ok_and(|mode| mode.trim().eq_ignore_ascii_case("shared"))
            {
                add_llvm_runtime_library_path(&mut env, &libdir);
            }

            if let Some(libclang_dir) = find_libclang_dir(&llvm_path, &libdir) {
                env.insert(
                    "LIBCLANG_PATH".into(),
                    pecos_build::llvm::path_to_env_string(&libclang_dir),
                );
            }
        }
    }

    // macOS-specific
    #[cfg(target_os = "macos")]
    {
        // SDKROOT — needed for bindgen/clang to find system headers
        if std::env::var("SDKROOT").is_err()
            && let Ok(output) = std::process::Command::new("xcrun")
                .args(["--show-sdk-path"])
                .output()
            && output.status.success()
        {
            let sdk = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !sdk.is_empty() {
                env.insert("SDKROOT".into(), sdk);
            }
        }

        // Deployment target
        env.insert("MACOSX_DEPLOYMENT_TARGET".into(), "13.2".into());
    }

    // CUDA
    if let Some(cuda_path) = pecos_build::cuda::find_cuda() {
        env.insert("CUDA_PATH".into(), cuda_path.display().to_string());
    }

    // cuQuantum
    if let Some(cuquantum_path) = pecos_build::cuquantum::find_cuquantum() {
        env.insert(
            "CUQUANTUM_ROOT".into(),
            cuquantum_path.display().to_string(),
        );
    }

    // cmake — set CMAKE explicitly so cmake-rs (used by highs-sys via the
    // MWPF decoder) finds the binary without depending on PATH. When the user
    // has system cmake, this is redundant but harmless; when they're using the
    // PECOS-managed install, this is what lets `cargo test` / `cargo check`
    // build the mwpf dep tree without further plumbing.
    if let Some(cmake_bin) = pecos_build::cmake::find_cmake() {
        env.insert("CMAKE".into(), cmake_bin.display().to_string());
    }

    // PYO3_PYTHON — point pyo3's build script at a Python that ships libpython
    // so `cargo test` on pecos-rslib* (which depend on pyo3) can link. macOS's
    // Apple-shipped python3 (the one CommandLineTools provides) has no
    // libpython, so the default PATH lookup fails to link. The repo's .venv
    // (created by uv) does ship libpython, so prefer that. Respect an existing
    // PYO3_PYTHON if the caller already set one.
    if std::env::var_os("PYO3_PYTHON").is_none()
        && let Some(repo_root) = pecos_build::llvm::find_cargo_project_root()
    {
        let venv_python = if cfg!(windows) {
            repo_root.join(".venv").join("Scripts").join("python.exe")
        } else {
            repo_root.join(".venv").join("bin").join("python")
        };
        if venv_python.exists() {
            env.insert("PYO3_PYTHON".into(), venv_python.display().to_string());
        }
    }

    env
}

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
fn add_llvm_runtime_library_path(env: &mut BTreeMap<String, String>, libdir: &Path) {
    prepend_path_env(env, "LD_LIBRARY_PATH", libdir);
}

#[cfg(target_os = "macos")]
fn add_llvm_runtime_library_path(env: &mut BTreeMap<String, String>, libdir: &Path) {
    prepend_path_env(env, "DYLD_LIBRARY_PATH", libdir);
}

#[cfg(target_os = "windows")]
fn add_llvm_runtime_library_path(_env: &mut BTreeMap<String, String>, _libdir: &Path) {
    // Windows LLVM DLLs are expected in bin/, which is already prepended to PATH.
}

#[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "macos"))]
fn prepend_path_env(env: &mut BTreeMap<String, String>, key: &str, first: &Path) {
    let current = env
        .get(key)
        .map(OsString::from)
        .or_else(|| std::env::var_os(key));
    let mut entries = vec![first.to_path_buf()];
    if let Some(current) = current {
        entries.extend(std::env::split_paths(&current));
    }

    if let Ok(joined) = std::env::join_paths(entries) {
        env.insert(key.to_string(), joined.to_string_lossy().into_owned());
    }
}

fn find_libclang_dir(llvm_path: &Path, libdir: &Path) -> Option<PathBuf> {
    let mut candidates = vec![libdir.to_path_buf()];
    if cfg!(windows) {
        candidates.insert(0, llvm_path.join("bin"));
    }

    candidates
        .into_iter()
        .find(|candidate| contains_libclang(candidate))
}

fn contains_libclang(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };

    entries
        .filter_map(std::result::Result::ok)
        .any(|entry| entry.file_name().to_str().is_some_and(is_libclang_filename))
}

fn is_libclang_filename(name: &str) -> bool {
    if cfg!(windows) {
        return name.eq_ignore_ascii_case("libclang.dll");
    }

    if cfg!(target_os = "macos") {
        return name == "libclang.dylib"
            || (name.starts_with("libclang.")
                && Path::new(name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("dylib")));
    }

    name == "libclang.so"
        || name.starts_with("libclang.so.")
        || name.starts_with("libclang-") && name.contains(".so")
}

/// Print environment in shell-eval format: `export KEY="VALUE"`
pub fn print_shell(env: &BTreeMap<String, String>) {
    for (key, value) in env {
        println!("export {key}=\"{value}\"");
    }
}

/// Print environment in JSON format.
pub fn print_json(env: &BTreeMap<String, String>) {
    let mut out = String::from("{\n");
    for (i, (key, value)) in env.iter().enumerate() {
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        let _ = write!(out, "  \"{key}\": \"{escaped}\"");
        if i + 1 < env.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push('}');
    println!("{out}");
}

/// Print environment in human-readable format.
pub fn print_show(env: &BTreeMap<String, String>) {
    if env.is_empty() {
        println!("No PECOS-specific environment variables needed.");
        return;
    }
    println!("PECOS build environment:");
    for (key, value) in env {
        println!("  {key}={value}");
    }
}

/// Write environment variables to GitHub Actions environment files.
pub fn write_github_actions(env: &BTreeMap<String, String>) -> Result<()> {
    let github_env = std::env::var("GITHUB_ENV").map_err(|_| {
        Error::Config(
            "GITHUB_ENV is not set; --github-actions must run inside GitHub Actions".into(),
        )
    })?;
    let github_path = std::env::var("GITHUB_PATH").map_err(|_| {
        Error::Config(
            "GITHUB_PATH is not set; --github-actions must run inside GitHub Actions".into(),
        )
    })?;

    write_github_actions_files(env, Path::new(&github_env), Path::new(&github_path))
}

fn write_github_actions_files(
    env: &BTreeMap<String, String>,
    github_env: &Path,
    github_path: &Path,
) -> Result<()> {
    let mut env_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(github_env)?;
    for (key, value) in env {
        if key != "PATH" {
            writeln!(env_file, "{key}={value}")?;
        }
    }

    let mut path_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(github_path)?;
    if let Some(llvm_path) = env.get(LLVM_SYS_PREFIX_ENV) {
        writeln!(
            path_file,
            "{}",
            pecos_build::llvm::path_to_env_string(&Path::new(llvm_path).join("bin"))
        )?;
    }

    Ok(())
}

/// Run the env subcommand.
pub fn run(format: &str, github_actions: bool) -> Result<()> {
    let env = collect_env();
    if github_actions {
        return write_github_actions(&env);
    }

    match format {
        "json" => print_json(&env),
        "show" => print_show(&env),
        _ => print_shell(&env),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_actions_writer_uses_env_file_and_path_file() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let env_path = std::env::temp_dir().join(format!("pecos-gh-env-{unique}"));
        let path_path = std::env::temp_dir().join(format!("pecos-gh-path-{unique}"));

        let llvm_prefix = Path::new("/opt/pecos/llvm-21.1");
        let llvm_prefix_str = llvm_prefix.display().to_string();
        let llvm_bin_str = pecos_build::llvm::path_to_env_string(&llvm_prefix.join("bin"));

        let mut env = BTreeMap::new();
        env.insert(LLVM_SYS_PREFIX_ENV.to_string(), llvm_prefix_str.clone());
        env.insert(
            "PATH".to_string(),
            "/opt/pecos/llvm-21.1/bin:/usr/bin".to_string(),
        );
        env.insert("PECOS_LLVM".to_string(), llvm_prefix_str.clone());

        write_github_actions_files(&env, &env_path, &path_path).unwrap();

        let env_file = std::fs::read_to_string(&env_path).unwrap();
        let path_file = std::fs::read_to_string(&path_path).unwrap();

        assert!(env_file.contains(&format!("{LLVM_SYS_PREFIX_ENV}={llvm_prefix_str}")));
        assert!(env_file.contains(&format!("PECOS_LLVM={llvm_prefix_str}")));
        assert!(!env_file.contains("PATH="));
        assert_eq!(path_file.trim(), llvm_bin_str);

        let _ = std::fs::remove_file(env_path);
        let _ = std::fs::remove_file(path_path);
    }
}
