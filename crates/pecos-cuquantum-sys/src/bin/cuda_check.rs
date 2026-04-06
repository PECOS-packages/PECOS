//! CUDA/cuQuantum runtime availability check
//!
//! Exit codes:
//! - 0: cuQuantum libraries loaded and CUDA device synchronized
//! - 1: cuQuantum libraries not found
//!
//! Used by `pecos rust test` to decide whether to run cuQuantum tests.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|arg| arg == "-j" || arg == "--json");
    let quiet = args.iter().any(|arg| arg == "-q" || arg == "--quiet");

    match pecos_cuquantum_sys::try_load() {
        Ok(backend) => {
            // Libraries loaded -- try a real CUDA call to verify GPU is present
            let rc = unsafe { (backend.cudaDeviceSynchronize)() };
            if rc == 0 {
                if json_output {
                    println!(
                        r#"{{"status":"available","message":"cuQuantum loaded, CUDA device present"}}"#
                    );
                } else if !quiet {
                    println!("cuda: available - cuQuantum loaded, CUDA device present");
                }
                ExitCode::SUCCESS
            } else {
                if json_output {
                    println!(
                        r#"{{"status":"unavailable","message":"cuQuantum loaded but no CUDA device (cudaDeviceSynchronize returned {rc})"}}"#
                    );
                } else if !quiet {
                    eprintln!("cuda: cuQuantum loaded but no CUDA device (error {rc})");
                }
                ExitCode::from(1)
            }
        }
        Err(e) => {
            if json_output {
                println!(r#"{{"status":"unavailable","message":"{e}"}}"#);
            } else if !quiet {
                eprintln!("cuda: not available - {e}");
            }
            ExitCode::from(1)
        }
    }
}
