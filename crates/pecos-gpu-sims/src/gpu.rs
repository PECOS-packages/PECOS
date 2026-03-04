//! wgpu-based state vector simulator implementation

use bytemuck::{Pod, Zeroable};
use pecos_rng::PecosRng;
use rand::RngExt;
use std::borrow::Cow;

use crate::gates;

/// Alignment for uniform buffer offsets (wgpu minimum is typically 256 bytes)
const UNIFORM_ALIGNMENT: u64 = 256;

/// Maximum number of gates that can be batched in a single submission
const MAX_BATCH_SIZE: u64 = 256;

/// Size of `GateParams` struct (padded to alignment)
const ALIGNED_GATE_PARAMS_SIZE: u64 = UNIFORM_ALIGNMENT;

/// Error type for GPU operations
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter found
    NoAdapter,
    /// Failed to create device
    DeviceCreation(wgpu::RequestDeviceError),
    /// Buffer mapping failed
    BufferMap(wgpu::BufferAsyncError),
    /// Too many qubits for available memory
    TooManyQubits { requested: u32, max: u32 },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(
                f,
                "No GPU adapter found. GpuStateVec requires a GPU with Vulkan, Metal, or DX12 support. \
                 Check GPU availability with `gpu-check` or use a CPU-based simulator instead (e.g., StateVec)."
            ),
            GpuError::DeviceCreation(e) => write!(f, "Failed to create GPU device: {e}"),
            GpuError::BufferMap(e) => write!(f, "Buffer mapping failed: {e}"),
            GpuError::TooManyQubits { requested, max } => {
                write!(f, "Too many qubits: {requested} requested, max {max}")
            }
        }
    }
}

impl std::error::Error for GpuError {}

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

/// Cross-platform GPU state vector quantum simulator
pub struct GpuStateVec {
    device: wgpu::Device,
    queue: wgpu::Queue,

    num_qubits: u32,
    num_amplitudes: u64,

    // GPU buffers
    state_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    probability_buffer: wgpu::Buffer,
    measure_params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    // Compute pipelines
    single_gate_pipeline: wgpu::ComputePipeline,
    cx_pipeline: wgpu::ComputePipeline,
    cz_pipeline: wgpu::ComputePipeline,
    rzz_pipeline: wgpu::ComputePipeline,
    probability_pipeline: wgpu::ComputePipeline,
    collapse_pipeline: wgpu::ComputePipeline,

    // Bind group layouts (kept for potential future bind group recreation)
    #[allow(dead_code)]
    gate_bind_group_layout: wgpu::BindGroupLayout,
    probability_bind_group_layout: wgpu::BindGroupLayout,
    collapse_bind_group_layout: wgpu::BindGroupLayout,

    // Persistent bind group for gate operations (uses dynamic uniform buffer offsets)
    gate_bind_group: wgpu::BindGroup,

    // RNG for measurements (Send + Sync for parallel Monte Carlo)
    rng: PecosRng,
}

/// Maximum workgroups per dimension (wgpu limit is 65535)
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

impl GpuStateVec {
    /// Compute the number of workgroups needed for a given number of elements.
    /// Uses 256 threads per workgroup (standard for GPU compute).
    /// Returns (x, y) dimensions for dispatch, using 2D dispatch when count exceeds limit.
    fn compute_workgroups(num_elements: u64) -> (u32, u32) {
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

        let num_amplitudes = 1u64 << num_qubits;

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| GpuError::NoAdapter)?;

        // Request the adapter's maximum buffer size limits for large qubit counts
        // This allows supporting as many qubits as the GPU hardware allows
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            ..Default::default()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("PECOS wgpu simulator"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::default(),
        }))
        .map_err(GpuError::DeviceCreation)?;

        // Create shader module
        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Quantum simulation shaders"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
            });

        // Create buffers
        let state_buffer_size = num_amplitudes * 8; // 2 * f32 per amplitude
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
            size: ALIGNED_GATE_PARAMS_SIZE * MAX_BATCH_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let probability_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Probabilities"),
            size: num_amplitudes * 4, // f32 per amplitude
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
            size: num_amplitudes * 8, // For reading state vector (2 * f32 per amplitude)
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

        let probability_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Probability bind group layout"),
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
                        binding: 2,
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
            bind_group_layouts: &[&gate_bind_group_layout],
            immediate_size: 0,
        });

        let probability_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Probability pipeline layout"),
                bind_group_layouts: &[&probability_bind_group_layout],
                immediate_size: 0,
            });

        let collapse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collapse pipeline layout"),
                bind_group_layouts: &[&collapse_bind_group_layout],
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

        let cx_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CX pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_cx"),
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

        let rzz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RZZ pipeline"),
            layout: Some(&gate_pipeline_layout),
            module: &shader,
            entry_point: Some("apply_rzz"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let probability_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Probability pipeline"),
                layout: Some(&probability_pipeline_layout),
                module: &shader,
                entry_point: Some("compute_probabilities"),
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

        let mut sim = Self {
            device,
            queue,
            num_qubits,
            num_amplitudes,
            state_buffer,
            params_buffer,
            probability_buffer,
            measure_params_buffer,
            staging_buffer,
            single_gate_pipeline,
            cx_pipeline,
            cz_pipeline,
            rzz_pipeline,
            probability_pipeline,
            collapse_pipeline,
            gate_bind_group_layout,
            probability_bind_group_layout,
            collapse_bind_group_layout,
            gate_bind_group,
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
        // Create initial state: |0...0> = [1+0i, 0+0i, 0+0i, ...]
        // Safe: with max 30 qubits, num_amplitudes is at most 2^30 which fits in usize on 64-bit.
        // This crate requires 64-bit for practical use (32-bit can't address enough memory anyway).
        #[allow(clippy::cast_possible_truncation)]
        let mut initial_state = vec![[0.0f32, 0.0f32]; self.num_amplitudes as usize];
        initial_state[0] = [1.0, 0.0];

        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&initial_state));
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

    /// Apply the same single-qubit gate to multiple qubits in a single GPU submission.
    ///
    /// This is more efficient than calling `apply_single_gate` multiple times
    /// because it batches all operations into a single command buffer submission,
    /// uses a single buffer write for all parameters, and uses dynamic uniform
    /// buffer offsets to avoid per-gate bind group creation.
    #[allow(clippy::cast_possible_truncation)]
    fn apply_single_gate_batch_qubits(&mut self, qubits: &[QubitId], matrix: [f32; 8]) {
        if qubits.is_empty() {
            return;
        }

        // Build all gate params on CPU first, then write in a single buffer operation.
        // Each GateParams is 64 bytes but we need UNIFORM_ALIGNMENT (256) bytes per entry.
        // We'll write each params at its aligned offset.
        let num_gates = qubits.len();
        let total_size = num_gates * ALIGNED_GATE_PARAMS_SIZE as usize;
        let mut params_data = vec![0u8; total_size];

        for (i, &qubit) in qubits.iter().enumerate() {
            let params = GateParams {
                target_qubit: qubit.index() as u32,
                control_qubit: 0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [matrix[0], matrix[1], matrix[2], matrix[3]],
                matrix_row1: [matrix[4], matrix[5], matrix[6], matrix[7]],
            };

            let offset = i * ALIGNED_GATE_PARAMS_SIZE as usize;
            let params_bytes = bytemuck::bytes_of(&params);
            params_data[offset..offset + params_bytes.len()].copy_from_slice(params_bytes);
        }

        // Single buffer write for all gate parameters
        self.queue
            .write_buffer(&self.params_buffer, 0, &params_data);

        // Create a single command encoder for all dispatches
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched single gate encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched single gate pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.single_gate_pipeline);

            let num_pairs = self.num_amplitudes / 2;
            let (wg_x, wg_y) = Self::compute_workgroups(num_pairs);

            // Use dynamic offset with persistent bind group for each gate
            for i in 0..qubits.len() {
                let offset = (i as u64 * ALIGNED_GATE_PARAMS_SIZE) as u32;
                pass.set_bind_group(0, &self.gate_bind_group, &[offset]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        // Single submission for all gates
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

    /// Apply CX gates to multiple qubit pairs in a single GPU submission.
    ///
    /// Takes qubits as interleaved pairs: [control0, target0, control1, target1, ...]
    #[allow(clippy::cast_possible_truncation)]
    fn cx_batch_qubits(&mut self, qubits: &[QubitId]) {
        let num_pairs = qubits.len() / 2;
        if num_pairs == 0 {
            return;
        }

        // Build all gate params on CPU first, then write in a single buffer operation
        let total_size = num_pairs * ALIGNED_GATE_PARAMS_SIZE as usize;
        let mut params_data = vec![0u8; total_size];

        for (i, pair) in qubits.chunks_exact(2).enumerate() {
            let params = GateParams {
                target_qubit: pair[1].index() as u32,
                control_qubit: pair[0].index() as u32,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [0.0; 4],
                matrix_row1: [0.0; 4],
            };

            let offset = i * ALIGNED_GATE_PARAMS_SIZE as usize;
            let params_bytes = bytemuck::bytes_of(&params);
            params_data[offset..offset + params_bytes.len()].copy_from_slice(params_bytes);
        }

        // Single buffer write for all gate parameters
        self.queue
            .write_buffer(&self.params_buffer, 0, &params_data);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched CX encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched CX pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cx_pipeline);

            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);

            // Use dynamic offset with persistent bind group for each gate pair
            for i in 0..num_pairs {
                let offset = (i as u64 * ALIGNED_GATE_PARAMS_SIZE) as u32;
                pass.set_bind_group(0, &self.gate_bind_group, &[offset]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Apply CZ gates to multiple qubit pairs in a single GPU submission.
    ///
    /// Takes qubits as interleaved pairs: [control0, target0, control1, target1, ...]
    #[allow(clippy::cast_possible_truncation)]
    fn cz_batch_qubits(&mut self, qubits: &[QubitId]) {
        let num_pairs = qubits.len() / 2;
        if num_pairs == 0 {
            return;
        }

        // Build all gate params on CPU first, then write in a single buffer operation
        let total_size = num_pairs * ALIGNED_GATE_PARAMS_SIZE as usize;
        let mut params_data = vec![0u8; total_size];

        for (i, pair) in qubits.chunks_exact(2).enumerate() {
            let params = GateParams {
                target_qubit: pair[1].index() as u32,
                control_qubit: pair[0].index() as u32,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [0.0; 4],
                matrix_row1: [0.0; 4],
            };

            let offset = i * ALIGNED_GATE_PARAMS_SIZE as usize;
            let params_bytes = bytemuck::bytes_of(&params);
            params_data[offset..offset + params_bytes.len()].copy_from_slice(params_bytes);
        }

        // Single buffer write for all gate parameters
        self.queue
            .write_buffer(&self.params_buffer, 0, &params_data);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched CZ encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched CZ pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cz_pipeline);

            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);

            // Use dynamic offset with persistent bind group for each gate pair
            for i in 0..num_pairs {
                let offset = (i as u64 * ALIGNED_GATE_PARAMS_SIZE) as u32;
                pass.set_bind_group(0, &self.gate_bind_group, &[offset]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Apply RZZ gates to multiple qubit pairs in a single GPU submission.
    ///
    /// Takes qubits as interleaved pairs: [q0, q1, q2, q3, ...] for pairs (q0,q1), (q2,q3), ...
    #[allow(clippy::cast_possible_truncation)]
    fn rzz_batch_qubits(&mut self, theta: f64, qubits: &[QubitId]) {
        let num_pairs = qubits.len() / 2;
        if num_pairs == 0 {
            return;
        }

        // Build all gate params on CPU first, then write in a single buffer operation
        let total_size = num_pairs * ALIGNED_GATE_PARAMS_SIZE as usize;
        let mut params_data = vec![0u8; total_size];

        for (i, pair) in qubits.chunks_exact(2).enumerate() {
            let params = GateParams {
                target_qubit: pair[1].index() as u32,
                control_qubit: pair[0].index() as u32,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix_row0: [theta as f32, 0.0, 0.0, 0.0],
                matrix_row1: [0.0; 4],
            };

            let offset = i * ALIGNED_GATE_PARAMS_SIZE as usize;
            let params_bytes = bytemuck::bytes_of(&params);
            params_data[offset..offset + params_bytes.len()].copy_from_slice(params_bytes);
        }

        // Single buffer write for all gate parameters
        self.queue
            .write_buffer(&self.params_buffer, 0, &params_data);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched RZZ encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched RZZ pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rzz_pipeline);

            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);

            // Use dynamic offset with persistent bind group for each gate pair
            for i in 0..num_pairs {
                let offset = (i as u64 * ALIGNED_GATE_PARAMS_SIZE) as u32;
                pass.set_bind_group(0, &self.gate_bind_group, &[offset]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
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
    pub fn measure(&mut self, qubit: u32) -> u32 {
        // Compute probabilities for all amplitudes
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

        let prob_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Probability bind group"),
            layout: &self.probability_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.probability_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Measure encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Probability pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probability_pipeline);
            pass.set_bind_group(0, &prob_bind_group, &[]);

            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy probabilities to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.probability_buffer,
            0,
            &self.staging_buffer,
            0,
            self.num_amplitudes * 4,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back probabilities
        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let probabilities: Vec<f32> = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        self.staging_buffer.unmap();

        // Sum probabilities for |1> outcome on target qubit
        let target_mask = 1u64 << qubit;
        let mut prob_one = 0.0f32;
        for (idx, &prob) in probabilities.iter().enumerate() {
            if (idx as u64 & target_mask) != 0 {
                prob_one += prob;
            }
        }

        // Sample outcome
        let random: f32 = self.rng.random();
        let outcome = u32::from(random < prob_one);

        // Collapse the state
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

        let collapse_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collapse bind group"),
            layout: &self.collapse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.measure_params_buffer.as_entire_binding(),
                },
            ],
        });

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
            pass.set_bind_group(0, &collapse_bind_group, &[]);

            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        outcome
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
    pub fn state(&self) -> Vec<[f32; 2]> {
        // Copy state buffer to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("State readback encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.state_buffer,
            0,
            &self.staging_buffer,
            0,
            self.num_amplitudes * 8,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read the staging buffer
        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

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
    pub fn probability(&self, basis_state: usize) -> f32 {
        let state = self.state();
        let [re, im] = state[basis_state];
        re * re + im * im
    }
}

// Trait implementations for PECOS integration

use pecos_core::{Angle64, QubitId};
use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};

impl QuantumSimulator for GpuStateVec {
    fn reset(&mut self) -> &mut Self {
        // Create initial state: |0...0> = [1+0i, 0+0i, 0+0i, ...]
        // Safe: with max 30 qubits, num_amplitudes fits in usize on 64-bit systems
        #[allow(clippy::cast_possible_truncation)]
        let mut initial_state = vec![[0.0f32, 0.0f32]; self.num_amplitudes as usize];
        initial_state[0] = [1.0, 0.0];

        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&initial_state));
        self
    }
}

// Trait implementations use internal batch methods directly to avoid allocations.
impl CliffordGateable for GpuStateVec {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_single_gate_batch_qubits(qubits, gates::S);
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_single_gate_batch_qubits(qubits, gates::H);
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_single_gate_batch_qubits(qubits, gates::X);
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_single_gate_batch_qubits(qubits, gates::Y);
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.apply_single_gate_batch_qubits(qubits, gates::Z);
        self
    }

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );
        self.cx_batch_qubits(qubits);
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CZ requires pairs of qubits"
        );
        self.cz_batch_qubits(qubits);
        self
    }

    #[allow(clippy::cast_possible_truncation)] // Qubit indices from QubitId fit in u32
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| {
                let outcome = self.measure(q.index() as u32);
                MeasurementResult {
                    outcome: outcome == 1,
                    is_deterministic: false, // State vector sim is never deterministic unless in eigenstate
                }
            })
            .collect()
    }
}

impl ArbitraryRotationGateable for GpuStateVec {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        self.apply_single_gate_batch_qubits(qubits, gates::rx(theta));
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        self.apply_single_gate_batch_qubits(qubits, gates::rz(theta));
        self
    }

    fn rzz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "RZZ requires pairs of qubits"
        );
        self.rzz_batch_qubits(theta, qubits);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{qid, qid2};
    use pecos_qsim::CliffordGateable;

    // Compile-time assertions that GpuStateVec is Send + Sync.
    // This is required for parallel Monte Carlo simulations.
    const _: fn() = || {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<GpuStateVec>();
        assert_sync::<GpuStateVec>();
    };

    #[test]
    fn test_initial_state() {
        // Just test that we can create a simulator
        let sim = GpuStateVec::new(2);
        assert!(sim.is_ok());
    }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut sim = GpuStateVec::new(1).unwrap();
        sim.h(&qid(0));

        // Measure many times - should get roughly 50/50
        let mut zeros = 0;
        let mut ones = 0;
        for _ in 0..100 {
            sim.reset();
            sim.h(&qid(0));
            if sim.measure(0) == 0 {
                zeros += 1;
            } else {
                ones += 1;
            }
        }

        // Should be roughly balanced (allow for statistical variation)
        assert!(zeros > 20 && zeros < 80);
        assert!(ones > 20 && ones < 80);
    }

    #[test]
    fn test_bell_state() {
        let mut sim = GpuStateVec::new(2).unwrap();

        // Create Bell state: H(0), CX(0,1)
        // Should always measure same value on both qubits
        for _ in 0..20 {
            sim.reset();
            sim.h(&qid(0));
            sim.cx(&qid2(0, 1));

            let m0 = sim.measure(0);
            let m1 = sim.measure(1);
            assert_eq!(m0, m1, "Bell state qubits should be correlated");
        }
    }

    #[test]
    fn test_derived_clifford_gates() {
        // Test that we get derived gates from the CliffordGateable trait
        let mut sim = GpuStateVec::new(2).unwrap();

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
        sim.cz(&qid2(0, 1)); // Apply CZ - should add phase but not change computational basis
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
        sim.swap(&qid2(0, 1)); // Should give |01>
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
        let mut sim = GpuStateVec::new(2).unwrap();

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

    use pecos_qsim::StateVec;

    /// Compare GPU and CPU state vectors with tolerance for f32 vs f64 precision.
    /// Returns the maximum absolute difference found.
    fn compare_states(gpu: &GpuStateVec, cpu: &mut StateVec) -> f64 {
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
        let gpu = GpuStateVec::new(3).unwrap();
        let mut cpu = StateVec::new(3);

        let max_diff = compare_states(&gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "Initial state mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_hadamard() {
        let mut gpu = GpuStateVec::new(2).unwrap();
        let mut cpu = StateVec::new(2);

        // H on qubit 0
        gpu.h(&qid(0));
        cpu.h(&qid(0));
        let max_diff = compare_states(&gpu, &mut cpu);
        assert!(max_diff < TOLERANCE, "H(0) mismatch: max_diff = {max_diff}");

        // H on qubit 1
        gpu.h(&qid(1));
        cpu.h(&qid(1));
        let max_diff = compare_states(&gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "H(0)H(1) mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_pauli_gates() {
        // Test X gate
        {
            let mut gpu = GpuStateVec::new(2).unwrap();
            let mut cpu = StateVec::new(2);
            gpu.x(&qid(0));
            cpu.x(&qid(0));
            let max_diff = compare_states(&gpu, &mut cpu);
            assert!(max_diff < TOLERANCE, "X(0) mismatch: max_diff = {max_diff}");
        }

        // Test Y gate
        {
            let mut gpu = GpuStateVec::new(2).unwrap();
            let mut cpu = StateVec::new(2);
            gpu.y(&qid(1));
            cpu.y(&qid(1));
            let max_diff = compare_states(&gpu, &mut cpu);
            assert!(max_diff < TOLERANCE, "Y(1) mismatch: max_diff = {max_diff}");
        }

        // Test Z gate
        {
            let mut gpu = GpuStateVec::new(2).unwrap();
            let mut cpu = StateVec::new(2);
            gpu.h(&qid(0)); // Put in superposition first so Z has an effect
            cpu.h(&qid(0));
            gpu.z(&qid(0));
            cpu.z(&qid(0));
            let max_diff = compare_states(&gpu, &mut cpu);
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
            let mut gpu = GpuStateVec::new(1).unwrap();
            let mut cpu = StateVec::new(1);
            gpu.h(&qid(0));
            cpu.h(&qid(0));
            gpu.sz(&qid(0));
            cpu.sz(&qid(0));
            let max_diff = compare_states(&gpu, &mut cpu);
            assert!(
                max_diff < TOLERANCE,
                "H(0)S(0) mismatch: max_diff = {max_diff}"
            );
        }

        // Test T gate
        {
            let mut gpu = GpuStateVec::new(1).unwrap();
            let mut cpu = StateVec::new(1);
            gpu.h(&qid(0));
            cpu.h(&qid(0));
            gpu.t(&qid(0));
            cpu.t(&qid(0));
            let max_diff = compare_states(&gpu, &mut cpu);
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
                let mut gpu = GpuStateVec::new(1).unwrap();
                let mut cpu = StateVec::new(1);
                gpu.rx(Angle64::from_radians(theta), &qid(0));
                cpu.rx(Angle64::from_radians(theta), &qid(0));
                let max_diff = compare_states(&gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "RX({theta}) mismatch: max_diff = {max_diff}"
                );
            }

            // Test RY
            {
                let mut gpu = GpuStateVec::new(1).unwrap();
                let mut cpu = StateVec::new(1);
                gpu.ry(Angle64::from_radians(theta), &qid(0));
                cpu.ry(Angle64::from_radians(theta), &qid(0));
                let max_diff = compare_states(&gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "RY({theta}) mismatch: max_diff = {max_diff}"
                );
            }

            // Test RZ
            {
                let mut gpu = GpuStateVec::new(1).unwrap();
                let mut cpu = StateVec::new(1);
                gpu.h(&qid(0)); // Put in superposition so RZ has visible effect
                cpu.h(&qid(0));
                gpu.rz(Angle64::from_radians(theta), &qid(0));
                cpu.rz(Angle64::from_radians(theta), &qid(0));
                let max_diff = compare_states(&gpu, &mut cpu);
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

                let mut gpu = GpuStateVec::new(3).unwrap();
                let mut cpu = StateVec::new(3);

                // Create superposition on control
                gpu.h(&qid(control));
                cpu.h(&qid(control));

                // Apply CX
                gpu.cx(&qid2(control, target));
                cpu.cx(&qid2(control, target));

                let max_diff = compare_states(&gpu, &mut cpu);
                assert!(
                    max_diff < TOLERANCE,
                    "CX({control},{target}) mismatch: max_diff = {max_diff}"
                );
            }
        }
    }

    #[test]
    fn test_compare_cz_gate() {
        let mut gpu = GpuStateVec::new(2).unwrap();
        let mut cpu = StateVec::new(2);

        // Create |++> state
        gpu.h(&qid(0));
        gpu.h(&qid(1));
        cpu.h(&qid(0));
        cpu.h(&qid(1));

        // Apply CZ
        gpu.cz(&qid2(0, 1));
        cpu.cz(&qid2(0, 1));

        let max_diff = compare_states(&gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "H(0)H(1)CZ(0,1) mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_rzz_gate() {
        let angles = [0.1, 0.5, 1.0, std::f64::consts::PI];

        for &theta in &angles {
            let mut gpu = GpuStateVec::new(2).unwrap();
            let mut cpu = StateVec::new(2);

            // Create superposition
            gpu.h(&qid(0));
            gpu.h(&qid(1));
            cpu.h(&qid(0));
            cpu.h(&qid(1));

            // Apply RZZ
            gpu.rzz(Angle64::from_radians(theta), &qid2(0, 1));
            cpu.rzz(Angle64::from_radians(theta), &qid2(0, 1));

            let max_diff = compare_states(&gpu, &mut cpu);
            assert!(
                max_diff < TOLERANCE,
                "RZZ({theta}) mismatch: max_diff = {max_diff}"
            );
        }
    }

    #[test]
    fn test_compare_complex_circuit() {
        // Test a more complex circuit with multiple gates
        let mut gpu = GpuStateVec::new(4).unwrap();
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
        gpu.cx(&qid2(0, 1));
        cpu.cx(&qid2(0, 1));
        gpu.cx(&qid2(2, 3));
        cpu.cx(&qid2(2, 3));

        // Layer 4: More rotations
        gpu.rz(Angle64::from_radians(0.2), &qid(0));
        cpu.rz(Angle64::from_radians(0.2), &qid(0));
        gpu.rz(Angle64::from_radians(0.4), &qid(1));
        cpu.rz(Angle64::from_radians(0.4), &qid(1));

        // Layer 5: Cross entanglement
        gpu.cx(&qid2(1, 2));
        cpu.cx(&qid2(1, 2));

        let max_diff = compare_states(&gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "Complex circuit mismatch: max_diff = {max_diff}"
        );
    }

    #[test]
    fn test_compare_reset() {
        let mut gpu = GpuStateVec::new(2).unwrap();
        let mut cpu = StateVec::new(2);

        // Apply some gates
        gpu.h(&qid(0));
        gpu.cx(&qid2(0, 1));
        cpu.h(&qid(0));
        cpu.cx(&qid2(0, 1));

        // Reset both
        gpu.reset();
        cpu.reset();

        let max_diff = compare_states(&gpu, &mut cpu);
        assert!(
            max_diff < TOLERANCE,
            "Reset state mismatch: max_diff = {max_diff}"
        );
    }
}
