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
//!   pecos env --show            # human-readable display

use std::collections::BTreeMap;
use std::fmt::Write;

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

/// Run the env subcommand.
pub fn run(format: &str) {
    let env = collect_env();
    match format {
        "json" => print_json(&env),
        "show" => print_show(&env),
        _ => print_shell(&env),
    }
}
