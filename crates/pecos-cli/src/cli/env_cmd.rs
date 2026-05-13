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
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::path::Path;

use pecos_build::Result;
use pecos_build::errors::Error;

/// Collect the build environment for the current platform.
///
/// Returns a map of environment variable names to values. Only includes
/// variables that PECOS needs to set — does not duplicate the entire shell
/// environment.
pub fn collect_env() -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();

    // LLVM
    if let Some(llvm_path) = pecos_build::llvm::find_llvm_14(None) {
        let llvm_str = llvm_path.display().to_string();
        env.insert("PECOS_LLVM".into(), llvm_str.clone());
        env.insert("LLVM_SYS_140_PREFIX".into(), llvm_str);

        // Add LLVM bin to PATH
        let bin_path = llvm_path.join("bin");
        if bin_path.exists() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            env.insert(
                "PATH".into(),
                format!("{}:{current_path}", bin_path.display()),
            );
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

    env
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
    if let Some(llvm_path) = env.get("LLVM_SYS_140_PREFIX") {
        writeln!(path_file, "{}", Path::new(llvm_path).join("bin").display())?;
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

        let mut env = BTreeMap::new();
        env.insert(
            "LLVM_SYS_140_PREFIX".to_string(),
            "/opt/pecos/llvm-14".to_string(),
        );
        env.insert(
            "PATH".to_string(),
            "/opt/pecos/llvm-14/bin:/usr/bin".to_string(),
        );
        env.insert("PECOS_LLVM".to_string(), "/opt/pecos/llvm-14".to_string());

        write_github_actions_files(&env, &env_path, &path_path).unwrap();

        let env_file = std::fs::read_to_string(&env_path).unwrap();
        let path_file = std::fs::read_to_string(&path_path).unwrap();

        assert!(env_file.contains("LLVM_SYS_140_PREFIX=/opt/pecos/llvm-14"));
        assert!(env_file.contains("PECOS_LLVM=/opt/pecos/llvm-14"));
        assert!(!env_file.contains("PATH="));
        assert_eq!(path_file.trim(), "/opt/pecos/llvm-14/bin");

        let _ = std::fs::remove_file(env_path);
        let _ = std::fs::remove_file(path_path);
    }
}
