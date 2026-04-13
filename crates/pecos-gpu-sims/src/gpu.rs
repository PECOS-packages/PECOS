//! wgpu-based state vector simulator implementation

use bytemuck::{Pod, Zeroable};
use pecos_random::PecosRng;
use rand::RngExt;
use std::borrow::Cow;

use crate::gates;
use crate::gpu_probe::{GpuStartupError, gpu_context};

/// Alignment for uniform buffer offsets (wgpu minimum is typically 256 bytes)
const UNIFORM_ALIGNMENT: usize = 256;

/// Maximum number of gates that can be batched in a single submission
const MAX_BATCH_SIZE: usize = 256;

/// Size of `GateParams` struct (padded to alignment)
const ALIGNED_GATE_PARAMS_SIZE: usize = UNIFORM_ALIGNMENT;

/// A wgpu feature that a simulator may require.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequiredFeature {
    /// Double-precision shaders (Vulkan `shaderFloat64`). Required by
    /// [`crate::GpuStateVec64`]. Not available on Metal / Apple Silicon.
    ShaderF64,
}

impl std::fmt::Display for RequiredFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequiredFeature::ShaderF64 => write!(f, "SHADER_F64"),
        }
    }
}

/// Error type for GPU operations
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter found
    NoAdapter,
    /// Shared GPU startup failed (adapter or device creation via `gpu_context`)
    Startup(GpuStartupError),
    /// Buffer mapping failed
    BufferMap(wgpu::BufferAsyncError),
    /// Too many qubits for available memory
    TooManyQubits { requested: u32, max: u32 },
    /// Required GPU feature unavailable on this adapter
    UnsupportedFeature(RequiredFeature),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(
                f,
                "No GPU adapter found. GpuStateVec32 requires a GPU with Vulkan, Metal, or DX12 support. \
                 Check GPU availability with `gpu-check` or use a CPU-based simulator instead (e.g., StateVec)."
            ),
            GpuError::Startup(e) => write!(f, "GPU startup failed: {e}"),
            GpuError::BufferMap(e) => write!(f, "Buffer mapping failed: {e}"),
            GpuError::TooManyQubits { requested, max } => {
                write!(f, "Too many qubits: {requested} requested, max {max}")
            }
            GpuError::UnsupportedFeature(feat) => {
                write!(f, "GPU does not support required feature: {feat}")
            }
        }
    }
}

impl std::error::Error for GpuError {}

impl From<GpuStartupError> for GpuError {
    fn from(err: GpuStartupError) -> Self {
        match err {
            GpuStartupError::NoAdapter => GpuError::NoAdapter,
            GpuStartupError::DeviceCreation { .. } => GpuError::Startup(err),
        }
    }
}

/// Parameters for single-qubit gate (matches WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GateParams {
    target_qubit: u32,
    control_qubit: u32,
    num_qubits: u32,
    _padding: u32,
    // Matrix stored as two vec4s for WGSL uniform alignment
    // matrix_row0 = [a_re, a_im, b_re, b_im]
    // matrix_row1 = [c_re, c_im, d_re, d_im]
    matrix_row0: [f32; 4],
    matrix_row1: [f32; 4],
}

/// Parameters for measurement collapse (matches WGSL struct)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeasureParams {
    target_qubit: u32,
    outcome: u32,
    norm_factor: f32,
    _padding: u32,
}

/// Which compute pipeline a queued gate should use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GatePipeline {
    Single,
    Diagonal,
    CX,
    CY,
    CZ,
    Swap,
    Rxx,
    Ryy,
    Rzz,
}

/// A gate waiting in the CPU-side queue until the next flush.
#[derive(Clone)]
struct QueuedGate {
    pipeline: GatePipeline,
    params: GateParams,
}

/// Cross-platform GPU state vector quantum simulator
pub struct GpuStateVec32 {
    device: wgpu::Device,
    queue: wgpu::Queue,

    num_qubits: u32,
    num_amplitudes: usize,

    // GPU buffers
    state_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    measure_params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    // Compute pipelines
    single_gate_pipeline: wgpu::ComputePipeline,
    diagonal_gate_pipeline: wgpu::ComputePipeline,
    cx_pipeline: wgpu::ComputePipeline,
    cy_pipeline: wgpu::ComputePipeline,
    cz_pipeline: wgpu::ComputePipeline,
    swap_pipeline: wgpu::ComputePipeline,
    rxx_pipeline: wgpu::ComputePipeline,
    ryy_pipeline: wgpu::ComputePipeline,
    rzz_pipeline: wgpu::ComputePipeline,
    collapse_pipeline: wgpu::ComputePipeline,

    // Bind group layouts: held to outlive the bind groups built from them
    // (wgpu may keep only weak references). Underscore-prefixed = intentionally
    // unread; their job is RAII lifetime, not direct use.
    _gate_bind_group_layout: wgpu::BindGroupLayout,
    _collapse_bind_group_layout: wgpu::BindGroupLayout,

    // Persistent bind groups
    gate_bind_group: wgpu::BindGroup,
    collapse_bind_group: wgpu::BindGroup,
    marginal_bind_group: wgpu::BindGroup,

    // GPU-side marginal probability reduction
    partial_sums_buffer: wgpu::Buffer,
    _marginal_bind_group_layout: wgpu::BindGroupLayout,
    marginal_pipeline: wgpu::ComputePipeline,
    num_partial_sums: u64,

    // Persistent kernel: for small states that fit in workgroup shared memory
    persistent_pipeline: wgpu::ComputePipeline,
    _persistent_bind_group_layout: wgpu::BindGroupLayout,
    persistent_bind_group: wgpu::BindGroup,
    gate_queue_buffer: wgpu::Buffer,
    /// Max qubits where the state fits in workgroup shared memory (0 if unavailable)
    persistent_max_qubits: u32,

    // Gate queue: gates accumulate here and are flushed in a single GPU submission
    gate_queue: Vec<QueuedGate>,
    params_staging: Vec<u8>,

    // RNG for measurements (Send + Sync for parallel Monte Carlo)
    rng: PecosRng,
}

/// Maximum workgroups per dimension (wgpu limit is 65535)
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

impl GpuStateVec32 {
    /// Compute the number of workgroups needed for a given number of elements.
    /// Uses 256 threads per workgroup (standard for GPU compute).
    /// Returns (x, y) dimensions for dispatch, using 2D dispatch when count exceeds limit.
    fn compute_workgroups(num_elements: usize) -> (u32, u32) {
        // Safe truncation: with max 30 qubits, max elements is 2^30 = ~1B
        // div_ceil(2^30, 256) = ~4M, well within u32 range
        #[allow(clippy::cast_possible_truncation)]
        let total_workgroups = num_elements.div_ceil(256) as u32;

        if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
            (total_workgroups, 1)
        } else {
            // Split into 2D dispatch with balanced dimensions to minimize wasted threads.
            // Find smallest y such that ceil(total/y) <= MAX_WORKGROUPS_PER_DIM
            let y = total_workgroups.div_ceil(MAX_WORKGROUPS_PER_DIM);
            let x = total_workgroups.div_ceil(y);
            (x, y)
        }
    }

    /// Create a new GPU state vector simulator
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits (state vector will have `2^num_qubits` amplitudes)
    ///
    /// # Errors
    /// Returns an error if no GPU is available or if the requested size exceeds GPU memory
    ///
    /// # Panics
    /// Panics if the operating system fails to provide entropy for RNG initialization.
    // GPU initialization requires setting up multiple interdependent resources (buffers,
    // bind group layouts, pipeline layouts, compute pipelines) in a specific order.
    // Extracting these into separate functions would complicate ownership without
    // improving readability, so we allow the longer function.
    // similar_names: cx_pipeline/cz_pipeline are standard quantum gate names (CNOT vs CZ).
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn new(num_qubits: u32) -> Result<Self, GpuError> {
        // Limit to reasonable size (30 qubits = 16 GB for f32 complex)
        if num_qubits > 30 {
            return Err(GpuError::TooManyQubits {
                requested: num_qubits,
                max: 30,
            });
        }

        let num_amplitudes = 1usize << num_qubits;

        let ctx = gpu_context()?;
        let device = ctx.device;
        let queue = ctx.queue;

        // Determine max qubits for persistent kernel based on available shared memory.
        // Each amplitude is vec2<f32> = 8 bytes. State of n qubits = 2^n * 8 bytes.
        let shared_mem_bytes = device.limits().max_compute_workgroup_storage_size;
        let persistent_max_qubits = if shared_mem_bytes >= 8 {
            (shared_mem_bytes / 8).ilog2()
        } else {
            0
        };

        // Create shader module
        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Quantum simulation shaders"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
            });

        // Create buffers
        let state_buffer_size = (num_amplitudes * 8) as u64; // 2 * f32 per amplitude
        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("State vector"),
            size: state_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gate parameters"),
            size: (ALIGNED_GATE_PARAMS_SIZE * MAX_BATCH_SIZE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let measure_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Measure parameters"),
            size: std::mem::size_of::<MeasureParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            size: (num_amplitudes * 8) as u64, // For reading state vector (2 * f32 per amplitude)
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layouts
        // Gate bind group uses dynamic offset for uniform buffer to avoid per-gate bind group creation
        let gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gate bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true, // Dynamic offset to select gate params
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                                GateParams,
                            >(
                            )
                                as u64),
                        },
                        count: None,
                    },
                ],
            });

        let collapse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Collapse bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create pipeline layouts
        let gate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gate pipeline layout"),
            bind_group_layouts: &[Some(&gate_bind_group_layout)],
            immediate_size: 0,
        });

        let collapse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collapse pipeline layout"),
                bind_group_layouts: &[Some(&collapse_bind_group_layout)],
                immediate_size: 0,
            });

        // Create compute pipelines
        let single_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Single gate pipeline"),
                layout: Some(&gate_pipeline_layout),
                module: &shader,
                entry_point: Some("apply_single_gate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let diagonal_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Diagonal gate pipeline"),
                layout: Some(&gate_pipeline_layout),
                module: &shader,
                entry_point: Some("apply_diagonal_gate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let cx_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CX pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_cx"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CY pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_cy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CZ pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_cz"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let swap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SWAP pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_swap"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let rxx_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RXX pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_rxx"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let ryy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RYY pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_ryy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let rzz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RZZ pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_rzz"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let collapse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Collapse pipeline"),
            layout: Some(&collapse_pipeline_layout),
            module: &shader,
            entry_point: Some("collapse_state"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create persistent bind group for gate operations with dynamic offset
        let gate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gate bind group (persistent)"),
            layout: &gate_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(std::mem::size_of::<GateParams>() as u64),
                    }),
                },
            ],
        });

        // Persistent collapse bind group (same buffers every time)
        let collapse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collapse bind group (persistent)"),
            layout: &collapse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: measure_params_buffer.as_entire_binding(),
                },
            ],
        });

        // GPU-side marginal probability reduction: workgroup partial sums
        let (meas_wg_x, meas_wg_y) = Self::compute_workgroups(num_amplitudes);
        let num_partial_sums = u64::from(meas_wg_x) * u64::from(meas_wg_y);

        let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Marginal partial sums"),
            size: num_partial_sums * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let marginal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Marginal probability bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let marginal_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Marginal probability pipeline layout"),
                bind_group_layouts: &[Some(&marginal_bind_group_layout)],
                immediate_size: 0,
            });

        let marginal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Marginal probability pipeline"),
            layout: Some(&marginal_pipeline_layout),
            module: &shader,
            entry_point: Some("reduce_marginal_probability"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let marginal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Marginal probability bind group (persistent)"),
            layout: &marginal_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: partial_sums_buffer.as_entire_binding(),
                },
            ],
        });

        // Persistent kernel: gate queue in a storage buffer
        // Max gate queue: 256 gates * 12 u32 per gate + 2 u32 header = 3074 u32 = ~12KB
        let gate_queue_buffer_size = (2 + MAX_BATCH_SIZE * 12) * 4;
        let gate_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent gate queue"),
            size: gate_queue_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let persistent_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Persistent kernel bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let persistent_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Persistent kernel pipeline layout"),
                bind_group_layouts: &[Some(&persistent_bind_group_layout)],
                immediate_size: 0,
            });

        // Compile persistent kernel shader with dynamic shared memory size
        let shared_size = 1u32 << persistent_max_qubits;
        let persistent_shader_src = include_str!("persistent_kernel_f32.wgsl")
            .replace("{SHARED_SIZE}", &shared_size.to_string());
        let persistent_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Persistent kernel shader (f32)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(persistent_shader_src)),
        });

        let persistent_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Persistent kernel pipeline"),
                layout: Some(&persistent_pipeline_layout),
                module: &persistent_shader,
                entry_point: Some("apply_gate_queue_persistent"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let persistent_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Persistent kernel bind group"),
            layout: &persistent_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gate_queue_buffer.as_entire_binding(),
                },
            ],
        });

        let mut sim = Self {
            device,
            queue,
            num_qubits,
            num_amplitudes,
            state_buffer,
            params_buffer,
            measure_params_buffer,
            staging_buffer,
            single_gate_pipeline,
            diagonal_gate_pipeline,
            cx_pipeline,
            cy_pipeline,
            cz_pipeline,
            swap_pipeline,
            rxx_pipeline,
            ryy_pipeline,
            rzz_pipeline,
            collapse_pipeline,
            _gate_bind_group_layout: gate_bind_group_layout,
            _collapse_bind_group_layout: collapse_bind_group_layout,
            gate_bind_group,
            collapse_bind_group,
            marginal_bind_group,
            partial_sums_buffer,
            _marginal_bind_group_layout: marginal_bind_group_layout,
            marginal_pipeline,
            num_partial_sums,
            persistent_pipeline,
            _persistent_bind_group_layout: persistent_bind_group_layout,
            persistent_bind_group,
            gate_queue_buffer,
            persistent_max_qubits,
            gate_queue: Vec::with_capacity(256),
            params_staging: vec![0u8; ALIGNED_GATE_PARAMS_SIZE * MAX_BATCH_SIZE],
            rng: rand::make_rng(),
        };

        // Initialize to |0...0> state
        sim.reset();

        Ok(sim)
    }

    /// Create a new GPU state vector simulator with a specific RNG seed.
    ///
    /// This is useful for reproducible Monte Carlo simulations.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits (state vector will have `2^num_qubits` amplitudes)
    /// * `seed` - Seed for the random number generator
    ///
    /// # Errors
    /// Returns an error if no GPU is available or if the requested size exceeds GPU memory
    pub fn with_seed(num_qubits: u32, seed: u64) -> Result<Self, GpuError> {
        let mut sim = Self::new(num_qubits)?;
        sim.rng = PecosRng::seed_from_u64(seed);
        Ok(sim)
    }

    /// Reset state to |0...0>
    pub fn reset(&mut self) {
        self.gate_queue.clear();

        // Create initial state: |0...0> = [1+0i, 0+0i, 0+0i, ...]
        let mut initial_state = vec![[0.0f32, 0.0f32]; self.num_amplitudes];
        initial_state[0] = [1.0, 0.0];

        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&initial_state));
    }

    /// Multiply two 2x2 complex matrices in [`a_re`, `a_im`, `b_re`, `b_im`, `c_re`, `c_im`, `d_re`, `d_im`] format.
    fn matrix_mul_f32(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        #[inline]
        fn cmul(xr: f32, xi: f32, yr: f32, yi: f32) -> (f32, f32) {
            (xr * yr - xi * yi, xr * yi + xi * yr)
        }

        let (c0r, c0i) = {
            let (t1r, t1i) = cmul(a[0], a[1], b[0], b[1]);
            let (t2r, t2i) = cmul(a[2], a[3], b[4], b[5]);
            (t1r + t2r, t1i + t2i)
        };
        let (c1r, c1i) = {
            let (t1r, t1i) = cmul(a[0], a[1], b[2], b[3]);
            let (t2r, t2i) = cmul(a[2], a[3], b[6], b[7]);
            (t1r + t2r, t1i + t2i)
        };
        let (c2r, c2i) = {
            let (t1r, t1i) = cmul(a[4], a[5], b[0], b[1]);
            let (t2r, t2i) = cmul(a[6], a[7], b[4], b[5]);
            (t1r + t2r, t1i + t2i)
        };
        let (c3r, c3i) = {
            let (t1r, t1i) = cmul(a[4], a[5], b[2], b[3]);
            let (t2r, t2i) = cmul(a[6], a[7], b[6], b[7]);
            (t1r + t2r, t1i + t2i)
        };

        [c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i]
    }

    /// Reorder single-qubit gates to group same-qubit gates together for fusion.
    ///
    /// Single-qubit gates on different qubits commute, so they can be freely
    /// reordered. Two-qubit gates act as barriers and are not moved.
    fn reorder_for_fusion(queue: &mut [QueuedGate]) {
        let mut start = 0;
        while start < queue.len() {
            if !matches!(
                queue[start].pipeline,
                GatePipeline::Single | GatePipeline::Diagonal
            ) {
                start += 1;
                continue;
            }

            let mut end = start + 1;
            while end < queue.len()
                && matches!(
                    queue[end].pipeline,
                    GatePipeline::Single | GatePipeline::Diagonal
                )
            {
                end += 1;
            }

            queue[start..end].sort_by_key(|g| g.params.target_qubit);
            start = end;
        }
    }

    /// Fuse consecutive single-qubit gates on the same qubit by multiplying matrices.
    fn fuse_gate_queue(queue: &mut [QueuedGate]) -> Vec<QueuedGate> {
        Self::reorder_for_fusion(queue);
        if queue.len() <= 1 {
            return queue.to_vec();
        }

        let mut fused = Vec::with_capacity(queue.len());
        let mut i = 0;

        while i < queue.len() {
            let gate = &queue[i];
            let is_1q = matches!(gate.pipeline, GatePipeline::Single | GatePipeline::Diagonal);
            if !is_1q {
                fused.push(queue[i].clone());
                i += 1;
                continue;
            }

            let target = gate.params.target_qubit;
            let mut matrix = [
                gate.params.matrix_row0[0],
                gate.params.matrix_row0[1],
                gate.params.matrix_row0[2],
                gate.params.matrix_row0[3],
                gate.params.matrix_row1[0],
                gate.params.matrix_row1[1],
                gate.params.matrix_row1[2],
                gate.params.matrix_row1[3],
            ];
            let mut j = i + 1;

            while j < queue.len() {
                let next = &queue[j];
                let next_is_1q =
                    matches!(next.pipeline, GatePipeline::Single | GatePipeline::Diagonal);
                if !next_is_1q || next.params.target_qubit != target {
                    break;
                }
                let next_matrix = [
                    next.params.matrix_row0[0],
                    next.params.matrix_row0[1],
                    next.params.matrix_row0[2],
                    next.params.matrix_row0[3],
                    next.params.matrix_row1[0],
                    next.params.matrix_row1[1],
                    next.params.matrix_row1[2],
                    next.params.matrix_row1[3],
                ];
                matrix = Self::matrix_mul_f32(&next_matrix, &matrix);
                j += 1;
            }

            let is_diagonal =
                matrix[2] == 0.0 && matrix[3] == 0.0 && matrix[4] == 0.0 && matrix[5] == 0.0;

            fused.push(QueuedGate {
                pipeline: if is_diagonal {
                    GatePipeline::Diagonal
                } else {
                    GatePipeline::Single
                },
                params: GateParams {
                    target_qubit: target,
                    control_qubit: 0,
                    num_qubits: gate.params.num_qubits,
                    _padding: 0,
                    matrix_row0: [matrix[0], matrix[1], matrix[2], matrix[3]],
                    matrix_row1: [matrix[4], matrix[5], matrix[6], matrix[7]],
                },
            });

            i = j;
        }

        fused
    }

    /// Flush all queued gates to the GPU in a single command buffer submission.
    ///
    /// Gates are accumulated by trait methods (h, cx, rz, etc.) and dispatched
    /// together here. This amortizes encoder creation and `queue.submit()` overhead
    /// across all queued gates.
    #[allow(clippy::cast_possible_truncation)]
    /// Encode the fused gate queue into the persistent kernel's storage buffer format.
    /// Returns the byte slice to write.
    fn encode_persistent_queue(
        fused: &[QueuedGate],
        num_qubits: u32,
        staging: &mut Vec<u8>,
    ) -> usize {
        // Header: [num_gates, num_qubits]
        // Each gate: 12 x u32 [type, target, control, pad, matrix(8 x f32 as u32)]
        let num_gates = fused.len();
        let total_u32 = 2 + num_gates * 12;
        let total_bytes = total_u32 * 4;

        if staging.len() < total_bytes {
            staging.resize(total_bytes, 0);
        }

        let buf: &mut [u32] = bytemuck::cast_slice_mut(&mut staging[..total_bytes]);

        buf[0] = num_gates as u32;
        buf[1] = num_qubits;

        for (i, gate) in fused.iter().enumerate() {
            let base = 2 + i * 12;
            buf[base] = match gate.pipeline {
                GatePipeline::Single => 0,
                GatePipeline::Diagonal => 1,
                GatePipeline::CX => 2,
                GatePipeline::CY => 3,
                GatePipeline::CZ => 4,
                GatePipeline::Swap => 5,
                GatePipeline::Rxx => 6,
                GatePipeline::Ryy => 7,
                GatePipeline::Rzz => 8,
            };
            buf[base + 1] = gate.params.target_qubit;
            buf[base + 2] = gate.params.control_qubit;
            buf[base + 3] = 0;
            // Matrix: f32 -> u32 bitcast
            buf[base + 4] = gate.params.matrix_row0[0].to_bits();
            buf[base + 5] = gate.params.matrix_row0[1].to_bits();
            buf[base + 6] = gate.params.matrix_row0[2].to_bits();
            buf[base + 7] = gate.params.matrix_row0[3].to_bits();
            buf[base + 8] = gate.params.matrix_row1[0].to_bits();
            buf[base + 9] = gate.params.matrix_row1[1].to_bits();
            buf[base + 10] = gate.params.matrix_row1[2].to_bits();
            buf[base + 11] = gate.params.matrix_row1[3].to_bits();
        }

        total_bytes
    }

    fn flush_gates(&mut self) {
        if self.gate_queue.is_empty() {
            return;
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Flush gates encoder"),
            });
        self.record_flush_gates(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Record queued gate dispatches into `encoder` without submitting.
    /// Callers that follow up with a readback can chain the copy into the
    /// same encoder, saving a submit round trip.
    fn record_flush_gates(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.gate_queue.is_empty() {
            return;
        }

        // Fuse consecutive single-qubit gates on the same qubit
        let fused = Self::fuse_gate_queue(&mut self.gate_queue);

        // Use persistent kernel if state fits in shared memory
        if self.num_qubits <= self.persistent_max_qubits {
            let total_bytes =
                Self::encode_persistent_queue(&fused, self.num_qubits, &mut self.params_staging);
            self.queue.write_buffer(
                &self.gate_queue_buffer,
                0,
                &self.params_staging[..total_bytes],
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Persistent kernel pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.persistent_pipeline);
                pass.set_bind_group(0, &self.persistent_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1); // Single workgroup
            }

            self.gate_queue.clear();
            return;
        }

        // Regular path: N dispatches into this encoder
        let aligned = ALIGNED_GATE_PARAMS_SIZE;
        let total_size = fused.len() * aligned;
        for (i, gate) in fused.iter().enumerate() {
            let offset = i * aligned;
            let bytes = bytemuck::bytes_of(&gate.params);
            self.params_staging[offset..offset + bytes.len()].copy_from_slice(bytes);
        }
        self.queue
            .write_buffer(&self.params_buffer, 0, &self.params_staging[..total_size]);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched gate pass"),
                timestamp_writes: None,
            });

            let num_pairs = self.num_amplitudes / 2;
            let (pair_wg_x, pair_wg_y) = Self::compute_workgroups(num_pairs);
            let (amp_wg_x, amp_wg_y) = Self::compute_workgroups(self.num_amplitudes);

            let mut current_pipeline = None;

            for (i, gate) in fused.iter().enumerate() {
                // Only switch pipeline when the gate type changes
                if current_pipeline != Some(gate.pipeline) {
                    let pipeline = match gate.pipeline {
                        GatePipeline::Single => &self.single_gate_pipeline,
                        GatePipeline::Diagonal => &self.diagonal_gate_pipeline,
                        GatePipeline::CX => &self.cx_pipeline,
                        GatePipeline::CY => &self.cy_pipeline,
                        GatePipeline::CZ => &self.cz_pipeline,
                        GatePipeline::Swap => &self.swap_pipeline,
                        GatePipeline::Rxx => &self.rxx_pipeline,
                        GatePipeline::Ryy => &self.ryy_pipeline,
                        GatePipeline::Rzz => &self.rzz_pipeline,
                    };
                    pass.set_pipeline(pipeline);
                    current_pipeline = Some(gate.pipeline);
                }

                let offset = u32::try_from(i * ALIGNED_GATE_PARAMS_SIZE)
                    .expect("batch offset always fits in u32 (i < MAX_BATCH_SIZE)");
                pass.set_bind_group(0, &self.gate_bind_group, &[offset]);

                let (wg_x, wg_y) = match gate.pipeline {
                    GatePipeline::Single => (pair_wg_x, pair_wg_y),
                    _ => (amp_wg_x, amp_wg_y),
                };
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        self.gate_queue.clear();
    }

    /// Wait for all submitted GPU work to complete.
    ///
    /// Flushes any queued gates first, then waits for the GPU to finish.
    /// Call this before timing measurements to ensure all asynchronous GPU
    /// operations have finished.
    pub fn sync(&mut self) {
        self.flush_gates();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Queue an arbitrary single-qubit gate for batched dispatch.
    fn queue_single_gate(&mut self, qubit: u32, matrix: [f32; 8]) {
        // Diagonal gates have zero off-diagonal elements (b=0, c=0).
        // Use the specialized diagonal shader: half the arithmetic, fully coalesced.
        let is_diagonal =
            matrix[2] == 0.0 && matrix[3] == 0.0 && matrix[4] == 0.0 && matrix[5] == 0.0;
        let pipeline = if is_diagonal {
            GatePipeline::Diagonal
        } else {
            GatePipeline::Single
        };
        self.gate_queue.push(QueuedGate {
            pipeline,
            params: GateParams {
                target_qubit: qubit,
                control_qubit: 0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [matrix[0], matrix[1], matrix[2], matrix[3]],
                matrix_row1: [matrix[4], matrix[5], matrix[6], matrix[7]],
            },
        });

        // Flush when we hit the buffer capacity
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    /// Queue a CX gate for batched dispatch.
    fn queue_cx(&mut self, control: u32, target: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::CX,
            params: GateParams {
                target_qubit: target,
                control_qubit: control,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [0.0; 4],
                matrix_row1: [0.0; 4],
            },
        });

        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    /// Queue a CZ gate for batched dispatch.
    fn queue_cz(&mut self, control: u32, target: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::CZ,
            params: GateParams {
                target_qubit: target,
                control_qubit: control,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [0.0; 4],
                matrix_row1: [0.0; 4],
            },
        });

        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_cy(&mut self, control: u32, target: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::CY,
            params: GateParams {
                target_qubit: target,
                control_qubit: control,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [0.0; 4],
                matrix_row1: [0.0; 4],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_swap(&mut self, qubit0: u32, qubit1: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Swap,
            params: GateParams {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [0.0; 4],
                matrix_row1: [0.0; 4],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_rxx(&mut self, qubit0: u32, qubit1: u32, theta: f32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Rxx,
            params: GateParams {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [theta, 0.0, 0.0, 0.0],
                matrix_row1: [0.0; 4],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_ryy(&mut self, qubit0: u32, qubit1: u32, theta: f32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Ryy,
            params: GateParams {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [theta, 0.0, 0.0, 0.0],
                matrix_row1: [0.0; 4],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    /// Queue an RZZ gate for batched dispatch.
    fn queue_rzz(&mut self, qubit0: u32, qubit1: u32, theta: f32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Rzz,
            params: GateParams {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [theta, 0.0, 0.0, 0.0],
                matrix_row1: [0.0; 4],
            },
        });

        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    /// Apply an arbitrary single-qubit gate
    pub fn apply_single_gate(&mut self, qubit: u32, matrix: [f32; 8]) {
        let params = GateParams {
            target_qubit: qubit,
            control_qubit: 0,
            num_qubits: self.num_qubits,
            _padding: 0,
            matrix_row0: [matrix[0], matrix[1], matrix[2], matrix[3]],
            matrix_row1: [matrix[4], matrix[5], matrix[6], matrix[7]],
        };

        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gate encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Single gate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.single_gate_pipeline);
            // Use persistent bind group with dynamic offset (offset 0 for single gate)
            pass.set_bind_group(0, &self.gate_bind_group, &[0]);

            // Dispatch: one thread per pair of amplitudes
            let num_pairs = self.num_amplitudes / 2;
            let (wg_x, wg_y) = Self::compute_workgroups(num_pairs);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Apply a single CX (CNOT) gate directly.
    ///
    /// This bypasses the trait layer and dispatches directly to the GPU.
    pub fn apply_cx(&mut self, control: u32, target: u32) {
        let params = GateParams {
            target_qubit: target,
            control_qubit: control,
            num_qubits: self.num_qubits,
            _padding: 0,
            matrix_row0: [0.0; 4],
            matrix_row1: [0.0; 4],
        };

        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("CX encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CX pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cx_pipeline);
            pass.set_bind_group(0, &self.gate_bind_group, &[0]);

            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Measure a qubit, collapsing the state
    ///
    /// Returns 0 or 1
    ///
    /// # Panics
    ///
    /// Panics if the GPU device poll fails (indicates a driver or hardware failure).
    // Measurement involves computing probabilities on GPU, reading them back to CPU,
    // sampling an outcome, then collapsing the state on GPU. These steps are tightly
    // coupled and extracting them would complicate the control flow.
    #[allow(clippy::too_many_lines)]
    /// Measure a single qubit using GPU-side workgroup reduction.
    ///
    /// Instead of reading back all 2^n probabilities (O(2^n) transfer), this uses
    /// a reduction kernel that produces ~2^n/256 partial sums, reducing the readback
    /// by 256x. The CPU sums the partial sums and samples the outcome.
    /// CPU-side measurement for small states. Reads the full state, computes
    /// probability, samples outcome, collapses and writes back. Faster than
    /// GPU dispatches when the state is small enough. Returns (outcome, `is_deterministic`).
    fn mz_cpu_path(&mut self, qubit: u32) -> (u32, bool) {
        const DET_EPS: f32 = 1e-6;

        let mut state_data = self.state();
        let target_mask = 1usize << qubit;

        let prob_one: f32 = state_data
            .iter()
            .enumerate()
            .filter(|(i, _)| i & target_mask != 0)
            .map(|(_, [re, im])| re * re + im * im)
            .sum();

        let is_deterministic = !(DET_EPS..=1.0 - DET_EPS).contains(&prob_one);
        let outcome = if is_deterministic {
            u32::from(prob_one > 0.5)
        } else {
            let random: f32 = self.rng.random();
            u32::from(random < prob_one)
        };

        let norm_factor = if outcome == 1 {
            1.0 / prob_one.sqrt()
        } else {
            1.0 / (1.0 - prob_one).sqrt()
        };

        for (i, amp) in state_data.iter_mut().enumerate() {
            let qubit_val = u32::from(i & target_mask != 0);
            if qubit_val == outcome {
                amp[0] *= norm_factor;
                amp[1] *= norm_factor;
            } else {
                *amp = [0.0, 0.0];
            }
        }

        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&state_data));

        (outcome, is_deterministic)
    }

    fn mz_gpu(&mut self, qubit: u32) -> (u32, bool) {
        const DET_EPS: f32 = 1e-6;

        // Fast path for small states: read entire state, compute probability + collapse on CPU.
        // Avoids 2 GPU dispatches (reduction + collapse) and 2 buffer writes.
        if self.num_qubits <= self.persistent_max_qubits {
            return self.mz_cpu_path(qubit);
        }

        // Write target qubit to params buffer
        let params = GateParams {
            target_qubit: qubit,
            control_qubit: 0,
            num_qubits: self.num_qubits,
            _padding: 0,
            matrix_row0: [0.0; 4],
            matrix_row1: [0.0; 4],
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // GPU reduction: each workgroup computes a partial sum of P(qubit = 1)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Marginal probability encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Marginal probability pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.marginal_pipeline);
            pass.set_bind_group(0, &self.marginal_bind_group, &[]);
            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy partial sums to staging buffer (256x smaller than full probability readback)
        let readback_size = self.num_partial_sums * 4;
        encoder.copy_buffer_to_buffer(
            &self.partial_sums_buffer,
            0,
            &self.staging_buffer,
            0,
            readback_size,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back partial sums and compute marginal probability
        let buffer_slice = self.staging_buffer.slice(..readback_size);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU device poll failed");

        let prob_one: f32 = {
            let data = buffer_slice.get_mapped_range();
            let sums: &[f32] = bytemuck::cast_slice(&data);
            sums.iter().sum()
        };
        self.staging_buffer.unmap();

        let is_deterministic = !(DET_EPS..=1.0 - DET_EPS).contains(&prob_one);
        let outcome = if is_deterministic {
            u32::from(prob_one > 0.5)
        } else {
            let random: f32 = self.rng.random();
            u32::from(random < prob_one)
        };

        let norm_factor = if outcome == 1 {
            1.0 / prob_one.sqrt()
        } else {
            1.0 / (1.0 - prob_one).sqrt()
        };

        let measure_params = MeasureParams {
            target_qubit: qubit,
            outcome,
            norm_factor,
            _padding: 0,
        };
        self.queue.write_buffer(
            &self.measure_params_buffer,
            0,
            bytemuck::bytes_of(&measure_params),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Collapse encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Collapse pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.collapse_pipeline);
            pass.set_bind_group(0, &self.collapse_bind_group, &[]);
            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        (outcome, is_deterministic)
    }

    /// Get the number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Get information about the GPU adapter
    #[must_use]
    pub fn adapter_info(&self) -> String {
        "wgpu device".to_string()
    }

    /// Read the state vector from GPU memory.
    ///
    /// Returns amplitudes as `Vec<[f32; 2]>` where each element is `[real, imag]`.
    /// The index corresponds to the computational basis state in little-endian order.
    ///
    /// # Panics
    ///
    /// Panics if the GPU device poll fails (indicates a driver or hardware failure).
    #[must_use]
    pub fn state(&mut self) -> Vec<[f32; 2]> {
        // Combine any pending gate dispatches with the readback copy into a
        // single encoder/submit -- saves one submit round trip vs separate
        // flush + copy submissions.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("State readback encoder"),
            });
        self.record_flush_gates(&mut encoder);

        encoder.copy_buffer_to_buffer(
            &self.state_buffer,
            0,
            &self.staging_buffer,
            0,
            (self.num_amplitudes * 8) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read the staging buffer
        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU device poll failed");

        let state: Vec<[f32; 2]> = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        self.staging_buffer.unmap();

        state
    }

    /// Get the probability of measuring a specific basis state.
    ///
    /// # Arguments
    /// * `basis_state` - The computational basis state index (little-endian)
    #[must_use]
    pub fn probability(&mut self, basis_state: usize) -> f32 {
        let state = self.state();
        let [re, im] = state[basis_state];
        re * re + im * im
    }

    /// Overwrite the GPU state buffer with `amps`. Length must equal
    /// `num_amplitudes`; caller is responsible for the state being normalized.
    /// Pending queued gates are flushed first.
    ///
    /// # Panics
    /// Panics if `amps.len() != num_amplitudes`.
    pub fn write_state(&mut self, amps: &[[f32; 2]]) {
        assert_eq!(
            amps.len(),
            self.num_amplitudes,
            "write_state: slice length mismatch"
        );
        self.flush_gates();
        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(amps));
    }
}

// Trait implementations for PECOS integration

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};

impl QuantumSimulator for GpuStateVec32 {
    fn reset(&mut self) -> &mut Self {
        // Create initial state: |0...0> = [1+0i, 0+0i, 0+0i, ...]
        let mut initial_state = vec![[0.0f32, 0.0f32]; self.num_amplitudes];
        initial_state[0] = [1.0, 0.0];

        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&initial_state));
        self
    }
}

// Trait implementations queue gates for batched dispatch.
#[allow(clippy::cast_possible_truncation)] // Qubit indices from QubitId fit in u32
impl CliffordGateable for GpuStateVec32 {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::H);
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::X);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::Y);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::Z);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::SX);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::SXDG);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::SY);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::SYDG);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::S);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::SDG);
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(c, t) in pairs {
            self.queue_cx(c.index() as u32, t.index() as u32);
        }
        self
    }

    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(c, t) in pairs {
            self.queue_cy(c.index() as u32, t.index() as u32);
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(c, t) in pairs {
            self.queue_cz(c.index() as u32, t.index() as u32);
        }
        self
    }

    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.queue_swap(q0.index() as u32, q1.index() as u32);
        }
        self
    }

    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // SZZ = RZZ(pi/2) -- reuse the existing RZZ shader
        let theta = std::f32::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rzz(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // SZZdg = RZZ(-pi/2)
        let theta = -std::f32::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rzz(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // SXX = RXX(pi/2) -- 1 dispatch instead of 5
        let theta = std::f32::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rxx(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = -std::f32::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rxx(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = std::f32::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_ryy(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = -std::f32::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_ryy(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.flush_gates();

        // Empirical mz path selection (RTX 4090 / PCIe 4.0, 2026-04-11).
        // CPU batch wins only when the state fits in ~128KB (N<=14) and at
        // least 2 qubits are measured. Above N=14, GPU sequential mz beats
        // readback + CPU loop by 2-13x even at full M=N.
        // M=1 always takes the GPU path: a single measurement amortizes the
        // CPU readback poorly (one collapse vs N elements transferred), and
        // the GPU reduction+collapse fuses into one submit.
        // Re-run scripts/native_bench/bench_pecos for a different GPU.
        if qubits.len() >= 2 && self.num_qubits <= 14 {
            self.mz_cpu_batch(qubits)
        } else {
            self.mz_gpu_sequential(qubits)
        }
    }
}

impl GpuStateVec32 {
    /// Read state, measure all qubits on CPU, write state back. Skips path
    /// selection -- intended for benchmarking and tests that need to force a
    /// specific path. Production code should call `mz()`.
    pub fn mz_cpu_batch(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        const DET_EPS: f32 = 1e-6;

        self.flush_gates();
        let mut state_data = self.state();
        let results: Vec<MeasurementResult> = qubits
            .iter()
            .map(|&q| {
                let target_mask = 1usize << q.index();

                let prob_one: f32 = state_data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i & target_mask != 0)
                    .map(|(_, [re, im])| re * re + im * im)
                    .sum();

                // prob_one very close to 0 or 1 means the measurement outcome
                // is forced by the state -- report it as deterministic.
                let is_deterministic = !(DET_EPS..=1.0 - DET_EPS).contains(&prob_one);

                let random: f32 = self.rng.random();
                let outcome = if is_deterministic {
                    u32::from(prob_one > 0.5)
                } else {
                    u32::from(random < prob_one)
                };

                let norm_factor = if outcome == 1 {
                    1.0 / prob_one.sqrt()
                } else {
                    1.0 / (1.0 - prob_one).sqrt()
                };

                for (i, amp) in state_data.iter_mut().enumerate() {
                    let qubit_val = u32::from(i & target_mask != 0);
                    if qubit_val == outcome {
                        amp[0] *= norm_factor;
                        amp[1] *= norm_factor;
                    } else {
                        *amp = [0.0, 0.0];
                    }
                }

                MeasurementResult {
                    outcome: outcome == 1,
                    is_deterministic,
                }
            })
            .collect();

        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&state_data));
        results
    }

    /// Sequential per-qubit GPU measurement. Skips path selection -- intended
    /// for benchmarking and tests that need to force a specific path.
    /// Production code should call `mz()`.
    pub fn mz_gpu_sequential(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.flush_gates();
        qubits
            .iter()
            .map(|&q| {
                #[allow(clippy::cast_possible_truncation)]
                let (outcome, is_deterministic) = self.mz_gpu(q.index() as u32);
                MeasurementResult {
                    outcome: outcome == 1,
                    is_deterministic,
                }
            })
            .collect()
    }
}

#[allow(clippy::cast_possible_truncation)] // Qubit indices from QubitId fit in u32
impl ArbitraryRotationGateable for GpuStateVec32 {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let matrix = gates::rx(theta);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, matrix);
        }
        self
    }

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let matrix = gates::ry(theta);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, matrix);
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let matrix = gates::rz(theta);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, matrix);
        }
        self
    }

    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::T);
        }
        self
    }

    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, gates::TDG);
        }
        self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed() as f32;
        for &(q0, q1) in pairs {
            self.queue_rxx(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed() as f32;
        for &(q0, q1) in pairs {
            self.queue_ryy(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed() as f32;
        for &(q0, q1) in pairs {
            self.queue_rzz(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::qid;
    use pecos_simulators::CliffordGateable;

    // Compile-time assertions that GpuStateVec32 is Send + Sync.
    // This is required for parallel Monte Carlo simulations.
    const _: fn() = || {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<GpuStateVec32>();
        assert_sync::<GpuStateVec32>();
    };

    #[test]
    fn test_initial_state() {
        // Just test that we can create a simulator
        let sim = GpuStateVec32::new(2);
        assert!(sim.is_ok());
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut sim = GpuStateVec32::new(1).unwrap();
        sim.h(&qid(0));

        // Measure many times - should get roughly 50/50
        let mut zeros = 0;
        let mut ones = 0;
        for _ in 0..100 {
            sim.reset();
            sim.h(&qid(0));
            if sim.mz(&qid(0))[0].outcome {
                ones += 1;
            } else {
                zeros += 1;
            }
        }

        // Should be roughly balanced (allow for statistical variation)
        assert!(zeros > 20 && zeros < 80);
        assert!(ones > 20 && ones < 80);
    }

    #[test]
    fn test_bell_state() {
        let mut sim = GpuStateVec32::new(2).unwrap();

        // Create Bell state: H(0), CX(0,1)
        // Should always measure same value on both qubits
        for _ in 0..20 {
            sim.reset();
            sim.h(&qid(0));
            sim.cx(&[(QubitId(0), QubitId(1))]);

            let results = sim.mz(&[QubitId(0), QubitId(1)]);
            assert_eq!(
                results[0].outcome, results[1].outcome,
                "Bell state qubits should be correlated"
            );
        }
    }

    #[test]
    fn test_derived_clifford_gates() {
        // Test that we get derived gates from the CliffordGateable trait
        let mut sim = GpuStateVec32::new(2).unwrap();

        // Test X gate (derived from H and Z, which is derived from SZ)
        sim.x(&qid(0)); // Should flip qubit 0 to |1>
        assert!(sim.mz(&qid(0))[0].outcome, "X gate should flip |0> to |1>");

        // Reset and test Y gate
        sim.reset();
        sim.y(&qid(0));
        assert!(sim.mz(&qid(0))[0].outcome, "Y gate should flip |0> to |1>");

        // Test CZ gate (derived from H and CX)
        sim.reset();
        sim.x(&qid(0)); // |10>
        sim.x(&qid(1)); // |11>
        sim.cz(&[(QubitId(0), QubitId(1))]); // Apply CZ - should add phase but not change computational basis
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        let m0 = &results[0];
        let m1 = &results[1];
        assert!(
            m0.outcome && m1.outcome,
            "CZ should not change |11> in computational basis"
        );

        // Test SWAP gate (derived from CX)
        sim.reset();
        sim.x(&qid(0)); // |10>
        sim.swap(&[(QubitId(0), QubitId(1))]); // Should give |01>
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        let m0 = &results[0];
        let m1 = &results[1];
        assert!(
            !m0.outcome && m1.outcome,
            "SWAP should exchange qubit states"
        );
    }

    #[test]
    fn test_derived_rotation_gates() {
        // Test that we get derived gates from the ArbitraryRotationGateable trait
        let mut sim = GpuStateVec32::new(2).unwrap();

        // Test RY gate (derived from RX and SZ)
        // RY(pi) should flip |0> to |1>
        sim.ry(Angle64::from_radians(std::f64::consts::PI), &qid(0));
        assert!(sim.mz(&qid(0))[0].outcome, "RY(pi) should flip |0> to |1>");

        // Test T gate (derived from RZ)
        sim.reset();
        sim.t(&qid(0)); // T gate is just a phase, shouldn't change measurement
        assert!(
            !sim.mz(&qid(0))[0].outcome,
            "T gate on |0> should still measure 0"
        );
    }

    // =========================================================================
    // Comparison tests against StateVec (CPU reference implementation)
    // =========================================================================

    use pecos_simulators::StateVec;

    /// Compare GPU and CPU state vectors with tolerance for f32 vs f64 precision.
    /// Returns the maximum absolute difference found.
    fn compare_states(gpu: &mut GpuStateVec32, cpu: &mut StateVec) -> f64 {
        let gpu_state = gpu.state();
        let cpu_state = cpu.state();

        assert_eq!(
            gpu_state.len(),
            cpu_state.len(),
            "State vector lengths must match"
        );

        let mut max_diff = 0.0f64;
        for (i, (gpu_amp, cpu_amp)) in gpu_state.iter().zip(cpu_state.iter()).enumerate() {
            let diff_re = (f64::from(gpu_amp[0]) - cpu_amp.re).abs();
            let diff_im = (f64::from(gpu_amp[1]) - cpu_amp.im).abs();
            let diff = diff_re.max(diff_im);
            if diff > max_diff {
                max_diff = diff;
            }
            // Fail fast with detailed info if way off
            assert!(
                diff < 1e-4,
                "State mismatch at index {i}: GPU=[{}, {}], CPU=[{}, {}]",
                gpu_amp[0],
                gpu_amp[1],
                cpu_amp.re,
                cpu_amp.im
            );
        }
        max_diff
    }

    /// Tolerance for comparing f32 GPU results to f64 CPU results
    const TOLERANCE: f64 = 1e-5;

    #[test]
    fn test_compare_initial_state() {
        let mut gpu = GpuStateVec32::new(3).unwrap();
        let mut cpu = StateVec::new(3);

        let max_diff = compare_states(&mut gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "Initial state mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_hadamard() {
        let mut gpu = GpuStateVec32::new(2).unwrap();
        let mut cpu = StateVec::new(2);

        // H on qubit 0
        gpu.h(&qid(0));
        cpu.h(&qid(0));
        let max_diff = compare_states(&mut gpu, &mut cpu);
        assert!(max_diff < TOLERANCE, "H(0) mismatch: max_diff = {max_diff}");

        // H on qubit 1
        gpu.h(&qid(1));
        cpu.h(&qid(1));
        let max_diff = compare_states(&mut gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "H(0)H(1) mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_pauli_gates() {
        // Test X gate
        {
            let mut gpu = GpuStateVec32::new(2).unwrap();
            let mut cpu = StateVec::new(2);
            gpu.x(&qid(0));
            cpu.x(&qid(0));
            let max_diff = compare_states(&mut gpu, &mut cpu);
            assert!(max_diff < TOLERANCE, "X(0) mismatch: max_diff = {max_diff}");
        }

        // Test Y gate
        {
            let mut gpu = GpuStateVec32::new(2).unwrap();
            let mut cpu = StateVec::new(2);
            gpu.y(&qid(1));
            cpu.y(&qid(1));
            let max_diff = compare_states(&mut gpu, &mut cpu);
            assert!(max_diff < TOLERANCE, "Y(1) mismatch: max_diff = {max_diff}");
        }

        // Test Z gate
        {
            let mut gpu = GpuStateVec32::new(2).unwrap();
            let mut cpu = StateVec::new(2);
            gpu.h(&qid(0)); // Put in superposition first so Z has an effect
            cpu.h(&qid(0));
            gpu.z(&qid(0));
            cpu.z(&qid(0));
            let max_diff = compare_states(&mut gpu, &mut cpu);
            assert!(
                max_diff < TOLERANCE,
                "H(0)Z(0) mismatch: max_diff = {max_diff}"
            );
        }
    }

    #[test]
    fn test_compare_phase_gates() {
        // Test S gate
        {
            let mut gpu = GpuStateVec32::new(1).unwrap();
            let mut cpu = StateVec::new(1);
            gpu.h(&qid(0));
            cpu.h(&qid(0));
            gpu.sz(&qid(0));
            cpu.sz(&qid(0));
            let max_diff = compare_states(&mut gpu, &mut cpu);
            assert!(
                max_diff < TOLERANCE,
                "H(0)S(0) mismatch: max_diff = {max_diff}"
            );
        }

        // Test T gate
        {
            let mut gpu = GpuStateVec32::new(1).unwrap();
            let mut cpu = StateVec::new(1);
            gpu.h(&qid(0));
            cpu.h(&qid(0));
            gpu.t(&qid(0));
            cpu.t(&qid(0));
            let max_diff = compare_states(&mut gpu, &mut cpu);
            assert!(
                max_diff < TOLERANCE,
                "H(0)T(0) mismatch: max_diff = {max_diff}"
            );
        }
    }

    #[test]
    fn test_compare_rotation_gates() {
        let angles = [0.0, 0.1, 0.5, 1.0, std::f64::consts::PI, 2.5];

        for &theta in &angles {
            // Test RX
            {
                let mut gpu = GpuStateVec32::new(1).unwrap();
                let mut cpu = StateVec::new(1);
                gpu.rx(Angle64::from_radians(theta), &qid(0));
                cpu.rx(Angle64::from_radians(theta), &qid(0));
                let max_diff = compare_states(&mut gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "RX({theta}) mismatch: max_diff = {max_diff}"
                );
            }

            // Test RY
            {
                let mut gpu = GpuStateVec32::new(1).unwrap();
                let mut cpu = StateVec::new(1);
                gpu.ry(Angle64::from_radians(theta), &qid(0));
                cpu.ry(Angle64::from_radians(theta), &qid(0));
                let max_diff = compare_states(&mut gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "RY({theta}) mismatch: max_diff = {max_diff}"
                );
            }

            // Test RZ
            {
                let mut gpu = GpuStateVec32::new(1).unwrap();
                let mut cpu = StateVec::new(1);
                gpu.h(&qid(0)); // Put in superposition so RZ has visible effect
                cpu.h(&qid(0));
                gpu.rz(Angle64::from_radians(theta), &qid(0));
                cpu.rz(Angle64::from_radians(theta), &qid(0));
                let max_diff = compare_states(&mut gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "H RZ({theta}) mismatch: max_diff = {max_diff}"
                );
            }
        }
    }

    #[test]
    fn test_compare_cx_gate() {
        // Test CX in various configurations
        for control in 0usize..3 {
            for target in 0usize..3 {
                if control == target {
                    continue;
                }

                let mut gpu = GpuStateVec32::new(3).unwrap();
                let mut cpu = StateVec::new(3);

                // Create superposition on control
                gpu.h(&qid(control));
                cpu.h(&qid(control));

                // Apply CX
                gpu.cx(&[(QubitId(control), QubitId(target))]);
                cpu.cx(&[(QubitId(control), QubitId(target))]);

                let max_diff = compare_states(&mut gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "CX({control},{target}) mismatch: max_diff = {max_diff}"
                );
            }
        }
    }

    #[test]
    fn test_compare_cz_gate() {
        let mut gpu = GpuStateVec32::new(2).unwrap();
        let mut cpu = StateVec::new(2);

        // Create |++> state
        gpu.h(&qid(0));
        gpu.h(&qid(1));
        cpu.h(&qid(0));
        cpu.h(&qid(1));

        // Apply CZ
        gpu.cz(&[(QubitId(0), QubitId(1))]);
        cpu.cz(&[(QubitId(0), QubitId(1))]);

        let max_diff = compare_states(&mut gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "H(0)H(1)CZ(0,1) mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_rzz_gate() {
        let angles = [0.1, 0.5, 1.0, std::f64::consts::PI];

        for &theta in &angles {
            let mut gpu = GpuStateVec32::new(2).unwrap();
            let mut cpu = StateVec::new(2);

            // Create superposition
            gpu.h(&qid(0));
            gpu.h(&qid(1));
            cpu.h(&qid(0));
            cpu.h(&qid(1));

            // Apply RZZ
            gpu.rzz(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);
            cpu.rzz(Angle64::from_radians(theta), &[(QubitId(0), QubitId(1))]);

            let max_diff = compare_states(&mut gpu, &mut cpu);
            assert!(
                max_diff < TOLERANCE,
                "RZZ({theta}) mismatch: max_diff = {max_diff}"
            );
        }
    }

    #[test]
    fn test_compare_complex_circuit() {
        // Test a more complex circuit with multiple gates
        let mut gpu = GpuStateVec32::new(4).unwrap();
        let mut cpu = StateVec::new(4);

        // Layer 1: Hadamards
        for q in 0usize..4 {
            gpu.h(&qid(q));
            cpu.h(&qid(q));
        }

        // Layer 2: Rotations
        gpu.rz(Angle64::from_radians(0.3), &qid(0));
        cpu.rz(Angle64::from_radians(0.3), &qid(0));
        gpu.rx(Angle64::from_radians(0.5), &qid(1));
        cpu.rx(Angle64::from_radians(0.5), &qid(1));
        gpu.ry(Angle64::from_radians(0.7), &qid(2));
        cpu.ry(Angle64::from_radians(0.7), &qid(2));
        gpu.rz(Angle64::from_radians(1.1), &qid(3));
        cpu.rz(Angle64::from_radians(1.1), &qid(3));

        // Layer 3: Entangling gates
        gpu.cx(&[(QubitId(0), QubitId(1))]);
        cpu.cx(&[(QubitId(0), QubitId(1))]);
        gpu.cx(&[(QubitId(2), QubitId(3))]);
        cpu.cx(&[(QubitId(2), QubitId(3))]);

        // Layer 4: More rotations
        gpu.rz(Angle64::from_radians(0.2), &qid(0));
        cpu.rz(Angle64::from_radians(0.2), &qid(0));
        gpu.rz(Angle64::from_radians(0.4), &qid(1));
        cpu.rz(Angle64::from_radians(0.4), &qid(1));

        // Layer 5: Cross entanglement
        gpu.cx(&[(QubitId(1), QubitId(2))]);
        cpu.cx(&[(QubitId(1), QubitId(2))]);

        let max_diff = compare_states(&mut gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "Complex circuit mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_reset() {
        let mut gpu = GpuStateVec32::new(2).unwrap();
        let mut cpu = StateVec::new(2);

        // Apply some gates
        gpu.h(&qid(0));
        gpu.cx(&[(QubitId(0), QubitId(1))]);
        cpu.h(&qid(0));
        cpu.cx(&[(QubitId(0), QubitId(1))]);

        // Reset both
        gpu.reset();
        cpu.reset();

        let max_diff = compare_states(&mut gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "Reset state mismatch: max_diff = {max_diff}"
        );
    }
}
