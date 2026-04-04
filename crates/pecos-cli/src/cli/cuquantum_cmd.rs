//! Implementation of the `cuquantum` subcommand

use pecos_build::Result;
use pecos_build::cuquantum::config::auto_configure_cuquantum;
use pecos_build::cuquantum::{
    find_cuquantum, get_cuquantum_version, get_lib_dir, get_pecos_cuquantum_dir,
    is_valid_cuquantum_installation,
};
use pecos_build::errors::Error;

/// Run the cuquantum subcommand
pub fn run(command: super::CuQuantumCommands) -> Result<()> {
    match command {
        super::CuQuantumCommands::Check { quiet } => run_check(quiet),
        super::CuQuantumCommands::Find { export } => run_find(export),
        super::CuQuantumCommands::Version => run_version(),
        super::CuQuantumCommands::Validate { path } => run_validate(path),
        super::CuQuantumCommands::Configure => run_configure(),
    }
}

/// Check if cuQuantum is available
fn run_check(quiet: bool) -> Result<()> {
    if let Some(cuquantum_path) = find_cuquantum() {
        if !quiet {
            // Determine if it's a local or system installation
            let is_local = get_pecos_cuquantum_dir().is_ok_and(|p| cuquantum_path.starts_with(&p));

            let location = if is_local { "local" } else { "system" };

            if let Ok(version) = get_cuquantum_version(&cuquantum_path) {
                println!("cuquantum: {version} ({location})");
            } else {
                println!(
                    "cuquantum: available at {} ({location})",
                    cuquantum_path.display()
                );
            }
            println!("path: {}", cuquantum_path.display());
        }
        Ok(())
    } else {
        if !quiet {
            eprintln!("cuquantum: not found");
            eprintln!();
            eprintln!("Install with: pecos install cuquantum");
            eprintln!("Or set CUQUANTUM_ROOT to your system cuQuantum installation");
        }
        Err(Error::CuQuantum("cuQuantum not available".to_string()))
    }
}

/// Find cuQuantum installation path
fn run_find(export: bool) -> Result<()> {
    if let Some(cuquantum_path) = find_cuquantum() {
        if export {
            println!("export CUQUANTUM_ROOT=\"{}\"", cuquantum_path.display());
        } else {
            println!("{}", cuquantum_path.display());
        }
        Ok(())
    } else {
        eprintln!("cuQuantum not found");
        eprintln!();
        eprintln!("Install with: pecos install cuquantum");
        Err(Error::CuQuantum("cuQuantum not found".to_string()))
    }
}

/// Show cuQuantum version
fn run_version() -> Result<()> {
    if let Some(cuquantum_path) = find_cuquantum() {
        let version = get_cuquantum_version(&cuquantum_path)?;
        println!("cuQuantum version: {version}");
        println!("Location: {}", cuquantum_path.display());

        // Check if local or system
        let is_local = get_pecos_cuquantum_dir().is_ok_and(|p| cuquantum_path.starts_with(&p));
        println!(
            "Type: {}",
            if is_local {
                "local (~/.pecos/deps/cuquantum/)"
            } else {
                "system"
            }
        );

        Ok(())
    } else {
        eprintln!("cuQuantum not found");
        Err(Error::CuQuantum("cuQuantum not found".to_string()))
    }
}

/// Configure .cargo/config.toml with cuQuantum path
fn run_configure() -> Result<()> {
    let cuquantum_path = auto_configure_cuquantum(None)?;
    println!("Configured cuQuantum path: {}", cuquantum_path.display());
    println!("Updated .cargo/config.toml with CUQUANTUM_ROOT");

    // Print runtime library path hint
    if let Some(lib_dir) = get_lib_dir(&cuquantum_path) {
        println!();
        println!("For runtime library loading, add to your shell profile:");
        println!(
            "  export LD_LIBRARY_PATH=\"{}:$LD_LIBRARY_PATH\"",
            lib_dir.display()
        );
    }

    Ok(())
}

/// Validate cuQuantum installation
fn run_validate(path: Option<String>) -> Result<()> {
    let cuquantum_path = if let Some(p) = path {
        std::path::PathBuf::from(p)
    } else {
        find_cuquantum().ok_or_else(|| {
            Error::CuQuantum("cuQuantum not found. Specify a path or install first.".into())
        })?
    };

    println!(
        "Validating cuQuantum installation at: {}",
        cuquantum_path.display()
    );
    println!();

    // Check required directories
    let required_dirs = [
        ("include", "Include directory"),
        ("lib", "Library directory (lib)"),
    ];

    let mut all_present = true;

    println!("Directories:");
    for (dir, description) in &required_dirs {
        let dir_path = cuquantum_path.join(dir);
        if dir_path.exists() {
            println!("  [OK] {description}");
        } else {
            // lib64 is also acceptable on Linux
            let alt_path = if *dir == "lib" {
                cuquantum_path.join("lib64")
            } else {
                dir_path.clone()
            };
            if alt_path.exists() {
                println!("  [OK] {description} (lib64)");
            } else {
                println!("  [MISSING] {description}");
                all_present = false;
            }
        }
    }

    // Check for key headers
    println!();
    println!("Headers:");
    let headers = [
        "include/custatevec.h",
        "include/cutensornet.h",
        "include/cudensitymat.h",
    ];

    for header in &headers {
        let header_path = cuquantum_path.join(header);
        if header_path.exists() {
            println!("  [OK] {header}");
        } else {
            println!("  [MISSING] {header}");
            // Headers are optional - some versions may not have all components
        }
    }

    // Check for libraries
    println!();
    println!("Libraries:");
    let lib_dir = if cuquantum_path.join("lib64").exists() {
        "lib64"
    } else {
        "lib"
    };

    let lib_ext = if cfg!(windows) { "lib" } else { "so" };
    let lib_prefix = if cfg!(windows) { "" } else { "lib" };

    let libraries = ["custatevec", "cutensornet", "cudensitymat"];

    for lib in &libraries {
        let lib_name = format!("{lib_prefix}{lib}.{lib_ext}");
        let lib_path = cuquantum_path.join(lib_dir).join(&lib_name);

        if lib_path.exists() {
            println!("  [OK] {lib_name}");
        } else {
            // Check for versioned library on Linux (e.g., libcustatevec.so.1)
            let versioned_exists = if cfg!(unix) {
                let pattern = format!("{lib_prefix}{lib}.{lib_ext}.");
                cuquantum_path
                    .join(lib_dir)
                    .read_dir()
                    .ok()
                    .is_some_and(|entries| {
                        entries.flatten().any(|e| {
                            e.file_name()
                                .to_str()
                                .is_some_and(|n| n.starts_with(&pattern))
                        })
                    })
            } else {
                false
            };

            if versioned_exists {
                println!("  [OK] {lib_name} (versioned)");
            } else {
                println!("  [MISSING] {lib_name}");
                all_present = false;
            }
        }
    }

    // Check version
    println!();
    if let Ok(version) = get_cuquantum_version(&cuquantum_path) {
        println!("Version: {version} [OK]");
    } else {
        println!("Version: could not determine [WARNING]");
    }

    println!();
    if all_present {
        println!("Validation: PASSED");
        if is_valid_cuquantum_installation(&cuquantum_path) {
            println!("Installation is valid and ready for use.");
        }
        Ok(())
    } else {
        println!("Validation: FAILED");
        println!("Some required components are missing.");
        Err(Error::CuQuantum(
            "cuQuantum validation failed - some components are missing".to_string(),
        ))
    }
}
