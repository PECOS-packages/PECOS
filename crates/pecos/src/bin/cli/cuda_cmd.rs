//! Implementation of the `cuda` subcommand

use pecos_build::Result;
use pecos_build::cuda::{
    find_cuda, get_cuda_version, get_pecos_cuda_dir, is_valid_cuda_installation,
};
use pecos_build::errors::Error;

/// Run the cuda subcommand
pub fn run(command: super::CudaCommands) -> Result<()> {
    match command {
        super::CudaCommands::Check { quiet } => run_check(quiet),
        super::CudaCommands::Find { export } => run_find(export),
        super::CudaCommands::Version => run_version(),
        super::CudaCommands::Validate { path } => run_validate(path),
        super::CudaCommands::SetupPython => run_setup_python(),
    }
}

/// Check if CUDA is available
fn run_check(quiet: bool) -> Result<()> {
    if let Some(cuda_path) = find_cuda() {
        if !quiet {
            // Determine if it's a local or system installation
            let is_local = get_pecos_cuda_dir().is_ok_and(|p| cuda_path.starts_with(&p));

            let location = if is_local { "local" } else { "system" };

            if let Ok(version) = get_cuda_version(&cuda_path) {
                println!("cuda: {version} ({location})");
            } else {
                println!("cuda: available at {} ({location})", cuda_path.display());
            }
            println!("path: {}", cuda_path.display());
        }
        Ok(())
    } else {
        if !quiet {
            eprintln!("cuda: not found");
            eprintln!();
            eprintln!("Install with: pecos install cuda");
            eprintln!("Or set CUDA_PATH to your system CUDA installation");
        }
        Err(Error::Cuda("CUDA not available".to_string()))
    }
}

/// Find CUDA installation path
fn run_find(export: bool) -> Result<()> {
    if let Some(cuda_path) = find_cuda() {
        if export {
            println!("export CUDA_PATH=\"{}\"", cuda_path.display());
            println!("export PATH=\"{}/bin:$PATH\"", cuda_path.display());
        } else {
            println!("{}", cuda_path.display());
        }
        Ok(())
    } else {
        eprintln!("CUDA not found");
        eprintln!();
        eprintln!("Install with: pecos install cuda");
        Err(Error::Cuda("CUDA not found".to_string()))
    }
}

/// Show CUDA version
fn run_version() -> Result<()> {
    if let Some(cuda_path) = find_cuda() {
        let version = get_cuda_version(&cuda_path)?;
        println!("CUDA version: {version}");
        println!("Location: {}", cuda_path.display());

        // Check if local or system
        let is_local = get_pecos_cuda_dir().is_ok_and(|p| cuda_path.starts_with(&p));
        println!(
            "Type: {}",
            if is_local {
                "local (~/.pecos/deps/cuda/)"
            } else {
                "system"
            }
        );

        Ok(())
    } else {
        eprintln!("CUDA not found");
        Err(Error::Cuda("CUDA not found".to_string()))
    }
}

/// Validate CUDA installation
fn run_validate(path: Option<String>) -> Result<()> {
    let cuda_path = if let Some(p) = path {
        std::path::PathBuf::from(p)
    } else {
        find_cuda()
            .ok_or_else(|| Error::Cuda("CUDA not found. Specify a path or install first.".into()))?
    };

    println!("Validating CUDA installation at: {}", cuda_path.display());
    println!();

    let exe_ext = if cfg!(windows) { ".exe" } else { "" };

    // Check required files
    let required_files = [
        (format!("bin/nvcc{exe_ext}"), "NVCC compiler"),
        ("include/cuda_runtime.h".to_string(), "CUDA runtime header"),
        ("include/cuda.h".to_string(), "CUDA driver header"),
    ];

    let mut all_present = true;
    for (file, description) in &required_files {
        let file_path = cuda_path.join(file);
        if file_path.exists() {
            println!("  [OK] {description} ({file})");
        } else {
            println!("  [MISSING] {description} ({file})");
            all_present = false;
        }
    }

    // Check libraries
    let lib_dir = if cfg!(windows) { "lib/x64" } else { "lib64" };

    let lib_ext = if cfg!(windows) { "lib" } else { "so" };
    let lib_prefix = if cfg!(windows) { "" } else { "lib" };

    let required_libs = ["cudart", "cublas"];

    println!();
    println!("Libraries ({lib_dir}/):");
    for lib in &required_libs {
        let lib_name = format!("{lib_prefix}{lib}.{lib_ext}");
        let lib_path = cuda_path.join(lib_dir).join(&lib_name);

        // Also check lib/ on Linux
        let alt_lib_path = cuda_path.join("lib").join(&lib_name);

        if lib_path.exists() || alt_lib_path.exists() {
            println!("  [OK] {lib_name}");
        } else {
            println!("  [MISSING] {lib_name}");
            all_present = false;
        }
    }

    // Check version
    println!();
    if let Ok(version) = get_cuda_version(&cuda_path) {
        println!("Version: {version} [OK]");
    } else {
        println!("Version: could not determine [WARNING]");
    }

    println!();
    if all_present {
        println!("Validation: PASSED");
        if is_valid_cuda_installation(&cuda_path) {
            println!("Installation is valid and ready for use.");
        }
        Ok(())
    } else {
        println!("Validation: FAILED");
        println!("Some required components are missing.");
        Err(Error::Cuda(
            "CUDA validation failed - some components are missing".to_string(),
        ))
    }
}

/// Install CUDA Python packages
fn run_setup_python() -> Result<()> {
    use std::process::Command;

    // First check if CUDA toolkit is available
    if find_cuda().is_none() {
        eprintln!("Error: CUDA toolkit not found.");
        eprintln!();
        eprintln!("Install CUDA toolkit first with:");
        eprintln!("  pecos install cuda");
        eprintln!();
        eprintln!("Or set CUDA_PATH to your system CUDA installation.");
        return Err(Error::Cuda(
            "CUDA toolkit required before installing Python packages".to_string(),
        ));
    }

    println!("Installing CUDA Python packages (cupy, cuquantum, pytket-cutensornet)...");
    println!();

    // Run uv sync --group cuda to install CUDA packages via dependency group
    let status = Command::new("uv")
        .args(["sync", "--group", "cuda"])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!();
            println!("CUDA Python packages installed successfully.");
            println!();
            println!("Verify with:");
            println!("  python -c \"import cupy; print('cupy:', cupy.cuda.is_available())\"");
            Ok(())
        }
        Ok(_) => {
            eprintln!();
            eprintln!("Failed to install CUDA Python packages.");
            eprintln!();
            eprintln!("You may need to install manually:");
            eprintln!("  uv sync --group cuda");
            Err(Error::Cuda(
                "Failed to install CUDA Python packages".to_string(),
            ))
        }
        Err(e) => {
            eprintln!("Error running uv: {e}");
            eprintln!();
            eprintln!("Make sure uv is installed and in your PATH.");
            Err(Error::Cuda(format!("Failed to run uv: {e}")))
        }
    }
}
