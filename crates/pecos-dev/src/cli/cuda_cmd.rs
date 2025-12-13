//! Implementation of the `cuda` subcommand

use crate::Result;
use crate::errors::Error;
use std::process::Command;

/// Run the cuda subcommand
pub fn run(command: super::CudaCommands) -> Result<()> {
    match command {
        super::CudaCommands::Check { quiet } => run_check(quiet),
    }
}

/// Check if CUDA is available
#[allow(clippy::collapsible_if)]
fn run_check(quiet: bool) -> Result<()> {
    // Check for nvcc first
    if let Ok(output) = Command::new("nvcc").args(["--version"]).output()
        && output.status.success()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse CUDA version from nvcc output
        let version = stdout
            .lines()
            .find(|l| l.contains("release"))
            .and_then(|l| l.split("release ").nth(1))
            .map(|s| s.split(',').next().unwrap_or(s).to_string());

        if !quiet {
            if let Some(ver) = version {
                println!("cuda: {ver}");
            } else {
                println!("cuda: available (nvcc found)");
            }
        }
        return Ok(());
    }

    // Check CUDA_PATH environment variable
    if let Ok(path) = std::env::var("CUDA_PATH") {
        if !quiet {
            println!("cuda: CUDA_PATH={path}");
        }
        return Ok(());
    }

    if !quiet {
        eprintln!("cuda: not found");
    }
    Err(Error::Config("CUDA not available".to_string()))
}
