//! Implementation of the `gpu` subcommand
//!
//! Provides GPU (wgpu adapter/device) detection by running the gpu-check binary
//! from pecos-gpu-sims.

use pecos_build::Result;
use pecos_build::errors::Error;
use std::process::Command;

/// Run the gpu subcommand
pub fn run(command: &super::GpuCommands) -> Result<()> {
    match command {
        super::GpuCommands::Check { quiet, json } => run_check(*quiet, *json),
    }
}

/// Check if a GPU (wgpu adapter/device) is available
fn run_check(quiet: bool, json: bool) -> Result<()> {
    // Build and run the gpu-check binary from pecos-gpu-sims
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "pecos-gpu-sims", "--bin", "gpu-check", "-q"]);
    cmd.arg("--");
    if quiet && !json {
        cmd.arg("-q");
    }
    if json {
        cmd.arg("--json");
    }

    let output = cmd
        .output()
        .map_err(|e| Error::Config(format!("Failed to run gpu-check: {e}")))?;

    if json {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.is_empty() {
            print!("{stdout}");
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() && !quiet {
            eprint!("{stderr}");
        }
    } else if !quiet {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.is_empty() {
            print!("{stdout}");
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            eprint!("{stderr}");
        }
    }

    if output.status.success() {
        Ok(())
    } else {
        if !quiet && !json {
            match output.status.code() {
                Some(1) => {
                    eprintln!();
                    eprintln!(
                        "GPU not available. pecos-gpu-sims requires a GPU with Vulkan, Metal, or DX12 support."
                    );
                }
                Some(2) => {
                    eprintln!();
                    eprintln!("GPU adapter detected, but device creation failed.");
                    eprintln!(
                        "pecos-gpu-sims will likely fail until the wgpu/driver issue is resolved."
                    );
                }
                Some(3) => {
                    eprintln!();
                    eprintln!(
                        "GPU adapter and device were available, but the simulator smoke test failed."
                    );
                    eprintln!(
                        "pecos-gpu-sims will likely fail until the startup/runtime issue is resolved."
                    );
                }
                Some(code) => {
                    eprintln!();
                    eprintln!("gpu-check failed with exit code {code}.");
                }
                None => {
                    eprintln!();
                    eprintln!("gpu-check terminated without an exit code.");
                }
            }
        }
        Err(Error::Config(format!(
            "gpu-check failed{}",
            output
                .status
                .code()
                .map_or_else(String::new, |code| format!(" (exit code {code})"))
        )))
    }
}
