//! Shared process-wide GPU context.
//!
//! All PECOS GPU simulators share a single `wgpu::Instance`, `wgpu::Adapter`,
//! `wgpu::Device`, and `wgpu::Queue`. Creating one device per simulator used to
//! trigger driver-level races when simulators ran in parallel (e.g. per-shot
//! rayon parallelism or cargo's default parallel tests) and could SIGSEGV the
//! Vulkan/wgpu stack.
//!
//! The shared context is initialized lazily on first access via `OnceLock`,
//! requesting the superset of optional features we care about
//! (`SHADER_F64`, `SUBGROUP`). Simulators that need a particular feature check
//! the corresponding `supports_*` flag on the context.

use std::sync::OnceLock;

/// Adapter/device information for the selected default GPU backend.
#[derive(Clone, Debug)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
}

/// Shared GPU device context.
///
/// `wgpu::Device` and `wgpu::Queue` are internally reference-counted handles,
/// so returning a cloned `GpuDeviceContext` from the process-wide singleton is
/// cheap and all clones point at the same underlying device.
#[derive(Clone, Debug)]
pub struct GpuDeviceContext {
    pub info: GpuAdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// True iff the device was created with `wgpu::Features::SHADER_F64`.
    pub supports_f64: bool,
    /// True iff the device was created with `wgpu::Features::SUBGROUP`.
    pub supports_subgroup: bool,
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

static GPU_CONTEXT: OnceLock<Result<GpuDeviceContext, GpuStartupError>> = OnceLock::new();

/// Return a handle to the shared process-wide GPU context.
///
/// On first call, initializes the wgpu instance/adapter/device/queue. Later
/// calls return cheap clones pointing at the same underlying device.
///
/// # Errors
/// Returns `GpuStartupError` if no suitable GPU adapter is found or device
/// creation fails. The error is memoized: once initialization fails, every
/// subsequent call returns a clone of the same error.
pub fn gpu_context() -> Result<GpuDeviceContext, GpuStartupError> {
    match GPU_CONTEXT.get_or_init(init_gpu_context) {
        Ok(ctx) => Ok(ctx.clone()),
        Err(err) => Err(err.clone()),
    }
}

fn init_gpu_context() -> Result<GpuDeviceContext, GpuStartupError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|_| GpuStartupError::NoAdapter)?;

    let adapter_raw_info = adapter.get_info();
    let info = GpuAdapterInfo {
        name: adapter_raw_info.name,
        backend: adapter_raw_info.backend,
        device_type: adapter_raw_info.device_type,
    };

    // Reject software renderers and unknown device types -- they technically
    // "work" but OOM on real workloads (common on CI runners without real GPUs)
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu => {}
        other => {
            return Err(GpuStartupError::DeviceCreation {
                info,
                error: format!("Device type {other:?} is not a hardware GPU"),
            });
        }
    }

    // Check buffer limits -- real GPUs support at least 128MB storage buffers.
    // Software renderers or broken drivers may report very low limits.
    let limits = adapter.limits();
    let min_buffer_mb = 128;
    if limits.max_storage_buffer_binding_size < min_buffer_mb * 1024 * 1024 {
        return Err(GpuStartupError::DeviceCreation {
            info,
            error: format!(
                "GPU storage buffer limit too small ({} MB, need at least {min_buffer_mb} MB)",
                limits.max_storage_buffer_binding_size / 1024 / 1024
            ),
        });
    }

    // Opportunistically request optional features that individual simulators
    // want. Intersecting with adapter.features() makes each optional: if the
    // adapter cannot provide SHADER_F64, we still get a device without it and
    // the f64 simulator will bail out with a clear error.
    let optional = wgpu::Features::SHADER_F64 | wgpu::Features::SUBGROUP;
    let adapter_features = adapter.features();
    let required_features = optional & adapter_features;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("PECOS shared GPU device"),
        required_features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .map_err(|error| GpuStartupError::DeviceCreation {
        info: info.clone(),
        error: error.to_string(),
    })?;

    let device_features = device.features();
    Ok(GpuDeviceContext {
        info,
        device,
        queue,
        supports_f64: device_features.contains(wgpu::Features::SHADER_F64),
        supports_subgroup: device_features.contains(wgpu::Features::SUBGROUP),
    })
}
