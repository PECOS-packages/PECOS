//! Simple GPU availability check
//!
//! Exits with 0 if a GPU adapter is available, 1 otherwise.
//! Used by build tools to conditionally enable GPU-dependent features.

use std::process::ExitCode;

fn main() -> ExitCode {
    let quiet = std::env::args().any(|arg| arg == "-q" || arg == "--quiet");

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Ok(adapter) = adapter {
        if !quiet {
            let info = adapter.get_info();
            println!("gpu: {} ({:?})", info.name, info.backend);
        }
        ExitCode::SUCCESS
    } else {
        if !quiet {
            eprintln!("gpu: not available");
            eprintln!("No compatible GPU adapter found (Vulkan/Metal/DX12)");
        }
        ExitCode::FAILURE
    }
}
