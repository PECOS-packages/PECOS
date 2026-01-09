//! Implementation of the `gpu` subcommand
//!
//! Provides GPU (wgpu adapter) detection by running the gpu-check binary
//! from pecos-gpu-sims.

use pecos_build::Result;
use pecos_build::errors::Error;
use std::process::Command;

/// Run the gpu subcommand
pub fn run(command: &super::GpuCommands) -> Result<()> {
    match command {
        super::GpuCommands::Check { quiet } => run_check(*quiet),
    }
}

/// Check if a GPU (wgpu adapter) is available
fn run_check(quiet: bool) -> Result<()> {
    // Build and run the gpu-check binary from pecos-gpu-sims
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "-p", "pecos-gpu-sims", "--bin", "gpu-check", "-q"]);

    if quiet {
        cmd.args(["--", "-q"]);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
    } else {
        cmd.arg("--");
    }

    let status = cmd
        .status()
        .map_err(|e| Error::Config(format!("Failed to run gpu-check: {e}")))?;

    if status.success() {
        Ok(())
    } else {
        if !quiet {
            eprintln!();
            eprintln!(
                "GPU not available. pecos-gpu-sims requires a GPU with Vulkan, Metal, or DX12 support."
            );
        }
        Err(Error::Config("GPU not available".to_string()))
    }
}
