//! Simple GPU availability check
//!
//! Exit codes:
//! - 0: GPU adapter found, device creation succeeded, and simulator smoke test passed
//! - 1: no compatible GPU adapter found
//! - 2: adapter found but device creation failed
//! - 3: simulator startup or smoke test failed
//!
//! Used by build tools to conditionally enable GPU-dependent features.

use pecos_core::{QubitId, qid};
use pecos_gpu_sims::GpuStabMulti;
use pecos_gpu_sims::gpu_probe::{GpuAdapterInfo, GpuStartupError, request_default_gpu_device};
use pecos_random::PecosRng;
use serde_json::json;
use std::process::ExitCode;

fn run_simulator_smoke_test() -> Result<(), String> {
    let num_shots = 16;
    let mut sim = GpuStabMulti::<PecosRng>::with_seed(2, num_shots, 42)?;

    // Create a Bell state and verify perfect Z-basis parity correlation.
    sim.h(&qid(0));
    sim.cx(&[(QubitId(0), QubitId(1))]);
    let results = sim.mz(&[QubitId(0), QubitId(1)]);

    if results.len() != num_shots {
        return Err(format!(
            "Expected {num_shots} shots from smoke test, got {}",
            results.len()
        ));
    }

    for (shot_idx, shot) in results.iter().enumerate() {
        if shot.len() != 2 {
            return Err(format!(
                "Smoke test shot {shot_idx} returned {} measurement results",
                shot.len()
            ));
        }
        if shot[0] != shot[1] {
            return Err(format!(
                "Smoke test Bell parity mismatch on shot {shot_idx}: {shot:?}"
            ));
        }
    }

    Ok(())
}

fn print_json(
    exit_code: u8,
    status: &str,
    info: Option<&GpuAdapterInfo>,
    message: Option<&str>,
) -> ExitCode {
    let payload = json!({
        "exit_code": exit_code,
        "status": status,
        "name": info.map(|info| info.name.clone()),
        "backend": info.map(|info| format!("{:?}", info.backend)),
        "device_type": info.map(|info| format!("{:?}", info.device_type)),
        "message": message,
    });
    println!("{payload}");
    ExitCode::from(exit_code)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let quiet = args.iter().any(|arg| arg == "-q" || arg == "--quiet");
    let json_output = args.iter().any(|arg| arg == "-j" || arg == "--json");

    match request_default_gpu_device("gpu-check device") {
        Ok(gpu) => match run_simulator_smoke_test() {
            Ok(()) => {
                if json_output {
                    print_json(0, "available", Some(&gpu.info), None)
                } else {
                    if !quiet {
                        println!(
                            "gpu: available - {} ({:?}, {:?}), simulator smoke test passed",
                            gpu.info.name, gpu.info.backend, gpu.info.device_type
                        );
                    }
                    ExitCode::SUCCESS
                }
            }
            Err(error) => {
                if json_output {
                    print_json(3, "smoke_test_failed", Some(&gpu.info), Some(&error))
                } else {
                    if !quiet {
                        eprintln!(
                            "gpu: device created, but simulator smoke test failed - {} ({:?}, {:?})",
                            gpu.info.name, gpu.info.backend, gpu.info.device_type
                        );
                        eprintln!("smoke test error: {error}");
                    }
                    ExitCode::from(3)
                }
            }
        },
        Err(GpuStartupError::NoAdapter) => {
            if json_output {
                print_json(
                    1,
                    "unavailable",
                    None,
                    Some("No compatible GPU adapter found (Vulkan/Metal/DX12)"),
                )
            } else {
                if !quiet {
                    eprintln!("gpu: not available");
                    eprintln!("No compatible GPU adapter found (Vulkan/Metal/DX12)");
                }
                ExitCode::from(1)
            }
        }
        Err(GpuStartupError::DeviceCreation { info, error }) => {
            if json_output {
                print_json(2, "device_creation_failed", Some(&info), Some(&error))
            } else {
                if !quiet {
                    eprintln!(
                        "gpu: adapter found but device creation failed - {} ({:?}, {:?})",
                        info.name, info.backend, info.device_type
                    );
                    eprintln!("device error: {error}");
                }
                ExitCode::from(2)
            }
        }
    }
}
