//! Shared GPU startup probe utilities.

/// Adapter/device information for the selected default GPU backend.
#[derive(Clone, Debug)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
}

/// Device context returned by the default GPU startup probe.
#[derive(Debug)]
pub struct GpuDeviceContext {
    pub info: GpuAdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Errors that can occur while creating the default GPU device.
#[derive(Clone, Debug)]
pub enum GpuStartupError {
    NoAdapter,
    DeviceCreation { info: GpuAdapterInfo, error: String },
}

impl std::fmt::Display for GpuStartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "No GPU adapter found"),
            Self::DeviceCreation { info, error } => write!(
                f,
                "Failed to create device for {} ({:?}, {:?}): {error}",
                info.name, info.backend, info.device_type
            ),
        }
    }
}

impl std::error::Error for GpuStartupError {}

/// Request the default high-performance GPU adapter and device used by PECOS GPU sims.
pub fn request_default_gpu_device(
    label: &'static str,
) -> Result<GpuDeviceContext, GpuStartupError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|_| GpuStartupError::NoAdapter)?;

    let info = adapter.get_info();
    let info = GpuAdapterInfo {
        name: info.name,
        backend: info.backend,
        device_type: info.device_type,
    };

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .map_err(|error| GpuStartupError::DeviceCreation {
        info: info.clone(),
        error: error.to_string(),
    })?;

    Ok(GpuDeviceContext {
        info,
        device,
        queue,
    })
}
