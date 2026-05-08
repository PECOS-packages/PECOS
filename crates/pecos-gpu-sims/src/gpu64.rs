// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! wgpu-based state vector simulator with f64 (double) precision.
//!
//! This is the f64 counterpart of [`GpuStateVec`](crate::GpuStateVec). It uses the
//! `SHADER_F64` wgpu feature (Vulkan `shaderFloat64`) for full double-precision
//! computation on the GPU.
//!
//! Note: f64 throughput on consumer GPUs is typically 1/64th of f32 throughput
//! (NVIDIA disables most FP64 units on `GeForce` cards). This simulator is intended
//! for precision-critical workloads and for benchmarking against cuStateVec (which
//! also uses f64).

use bytemuck::{Pod, Zeroable};
use pecos_random::PecosRng;
use rand::RngExt;
use std::borrow::Cow;

use crate::gates;
use crate::gpu::{GpuError, RequiredFeature};
use crate::gpu_probe::gpu_context;

const UNIFORM_ALIGNMENT: usize = 256;
const MAX_BATCH_SIZE: usize = 256;
const ALIGNED_GATE_PARAMS_SIZE: usize = UNIFORM_ALIGNMENT;
const MAX_WORKGROUPS_PER_DIM: u32 = 65535;

/// Gate parameters for f64 precision (matches WGSL struct in `shaders_f64.wgsl`)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GateParams64 {
    target_qubit: u32,
    control_qubit: u32,
    num_qubits: u32,
    _padding: u32,
    // Matrix elements as f64: a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im
    matrix: [f64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeasureParams64 {
    target_qubit: u32,
    outcome: u32,
    norm_factor: f64,
}

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

#[derive(Clone)]
struct QueuedGate {
    pipeline: GatePipeline,
    params: GateParams64,
}

/// Cross-platform GPU state vector simulator with f64 (double) precision.
///
/// Requires a GPU that supports Vulkan `shaderFloat64`.
pub struct GpuStateVec64 {
    device: wgpu::Device,
    queue: wgpu::Queue,

    num_qubits: u32,
    num_amplitudes: usize,

    state_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    measure_params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

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

    gate_bind_group: wgpu::BindGroup,
    collapse_bind_group: wgpu::BindGroup,
    marginal_bind_group: wgpu::BindGroup,

    partial_sums_buffer: wgpu::Buffer,
    _marginal_bind_group_layout: wgpu::BindGroupLayout,
    marginal_pipeline: wgpu::ComputePipeline,
    num_partial_sums: u64,

    // Persistent kernel for small states
    persistent_pipeline: wgpu::ComputePipeline,
    _persistent_bind_group_layout: wgpu::BindGroupLayout,
    persistent_bind_group: wgpu::BindGroup,
    gate_queue_buffer: wgpu::Buffer,
    persistent_max_qubits: u32,

    gate_queue: Vec<QueuedGate>,
    params_staging: Vec<u8>,
    rng: PecosRng,
}

impl GpuStateVec64 {
    fn compute_workgroups(num_elements: usize) -> (u32, u32) {
        #[allow(clippy::cast_possible_truncation)]
        let total_workgroups = num_elements.div_ceil(256) as u32;
        if total_workgroups <= MAX_WORKGROUPS_PER_DIM {
            (total_workgroups, 1)
        } else {
            let y = total_workgroups.div_ceil(MAX_WORKGROUPS_PER_DIM);
            let x = total_workgroups.div_ceil(y);
            (x, y)
        }
    }

    /// Create a new f64 GPU state vector simulator.
    ///
    /// # Errors
    /// Returns an error if no GPU with f64 support is available.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub fn new(num_qubits: u32) -> Result<Self, GpuError> {
        // 29 qubits = 2^29 * 16 bytes = 8 GB (f64 complex is 16 bytes vs f32's 8)
        if num_qubits > 29 {
            return Err(GpuError::TooManyQubits {
                requested: num_qubits,
                max: 29,
            });
        }

        let num_amplitudes = 1usize << num_qubits;

        let ctx = gpu_context()?;
        if !ctx.supports_f64 {
            return Err(GpuError::UnsupportedFeature(RequiredFeature::ShaderF64));
        }
        let device = ctx.device;
        let queue = ctx.queue;

        // f64: each amplitude is vec2<f64> = 16 bytes.
        let shared_mem_bytes = device.limits().max_compute_workgroup_storage_size;
        let persistent_max_qubits = if shared_mem_bytes >= 16 {
            (shared_mem_bytes / 16).ilog2()
        } else {
            0
        };

        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Quantum simulation shaders (f64)"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders_f64.wgsl"))),
            });

        // State buffer: 16 bytes per amplitude (2 x f64)
        let state_buffer_size = (num_amplitudes * 16) as u64;
        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("State vector (f64)"),
            size: state_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gate parameters (f64)"),
            size: (ALIGNED_GATE_PARAMS_SIZE * MAX_BATCH_SIZE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let measure_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Measure parameters (f64)"),
            size: std::mem::size_of::<MeasureParams64>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer (f64)"),
            size: state_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layouts (same structure as f32 version)
        let gate_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gate bind group layout (f64)"),
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
                            has_dynamic_offset: true,
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                                GateParams64,
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
                label: Some("Collapse bind group layout (f64)"),
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

        let gate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gate pipeline layout (f64)"),
            bind_group_layouts: &[Some(&gate_bind_group_layout)],
            immediate_size: 0,
        });

        let collapse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collapse pipeline layout (f64)"),
                bind_group_layouts: &[Some(&collapse_bind_group_layout)],
                immediate_size: 0,
            });

        let make_pipeline = |label, entry_point, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let single_gate_pipeline = make_pipeline(
            "Single gate (f64)",
            "apply_single_gate",
            &gate_pipeline_layout,
        );
        let diagonal_gate_pipeline = make_pipeline(
            "Diagonal gate (f64)",
            "apply_diagonal_gate",
            &gate_pipeline_layout,
        );
        let cx_pipeline = make_pipeline("CX (f64)", "apply_cx", &gate_pipeline_layout);
        let cy_pipeline = make_pipeline("CY (f64)", "apply_cy", &gate_pipeline_layout);
        let cz_pipeline = make_pipeline("CZ (f64)", "apply_cz", &gate_pipeline_layout);
        let swap_pipeline = make_pipeline("SWAP (f64)", "apply_swap", &gate_pipeline_layout);
        let rxx_pipeline = make_pipeline("RXX (f64)", "apply_rxx", &gate_pipeline_layout);
        let ryy_pipeline = make_pipeline("RYY (f64)", "apply_ryy", &gate_pipeline_layout);
        let rzz_pipeline = make_pipeline("RZZ (f64)", "apply_rzz", &gate_pipeline_layout);
        let collapse_pipeline = make_pipeline(
            "Collapse (f64)",
            "collapse_state",
            &collapse_pipeline_layout,
        );

        let gate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gate bind group (f64)"),
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
                        size: std::num::NonZeroU64::new(std::mem::size_of::<GateParams64>() as u64),
                    }),
                },
            ],
        });

        let collapse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collapse bind group (f64)"),
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

        // Marginal probability reduction (partial sums are f64)
        let (meas_wg_x, meas_wg_y) = Self::compute_workgroups(num_amplitudes);
        let num_partial_sums = u64::from(meas_wg_x) * u64::from(meas_wg_y);

        let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Marginal partial sums (f64)"),
            size: num_partial_sums * 8, // f64 = 8 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let marginal_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Marginal bind group layout (f64)"),
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
                label: Some("Marginal pipeline layout (f64)"),
                bind_group_layouts: &[Some(&marginal_bind_group_layout)],
                immediate_size: 0,
            });

        let marginal_pipeline = make_pipeline(
            "Marginal probability (f64)",
            "reduce_marginal_probability",
            &marginal_pipeline_layout,
        );

        let marginal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Marginal bind group (f64)"),
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

        // Persistent kernel: gate queue as array<f64>
        // 12 f64 per gate (type + tgt + ctrl + pad + 8 matrix elements) + 2 f64 header
        let gate_queue_buffer_size = (2 + MAX_BATCH_SIZE * 12) * 8;
        let gate_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Persistent gate queue (f64)"),
            size: gate_queue_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let persistent_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Persistent bind group layout (f64)"),
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
                label: Some("Persistent pipeline layout (f64)"),
                bind_group_layouts: &[Some(&persistent_bind_group_layout)],
                immediate_size: 0,
            });

        // Compile persistent kernel shader with dynamic shared memory size
        let shared_size = 1u32 << persistent_max_qubits;
        let persistent_shader_src = include_str!("persistent_kernel_f64.wgsl")
            .replace("{SHARED_SIZE}", &shared_size.to_string());
        let persistent_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Persistent kernel shader (f64)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(persistent_shader_src)),
        });

        let persistent_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Persistent kernel pipeline (f64)"),
                layout: Some(&persistent_pipeline_layout),
                module: &persistent_shader,
                entry_point: Some("apply_gate_queue_persistent"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let persistent_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Persistent bind group (f64)"),
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

        sim.reset();
        Ok(sim)
    }

    /// Create with a specific RNG seed for reproducibility.
    ///
    /// # Errors
    /// Returns [`GpuError`] if GPU init fails (no adapter, `SHADER_F64` not supported,
    /// or too many qubits).
    pub fn with_seed(num_qubits: u32, seed: u64) -> Result<Self, GpuError> {
        let mut sim = Self::new(num_qubits)?;
        sim.rng = PecosRng::seed_from_u64(seed);
        Ok(sim)
    }

    pub fn reset(&mut self) {
        self.gate_queue.clear();
        let mut initial_state = vec![[0.0f64, 0.0f64]; self.num_amplitudes];
        initial_state[0] = [1.0, 0.0];
        self.queue
            .write_buffer(&self.state_buffer, 0, bytemuck::cast_slice(&initial_state));
    }

    // -- Gate fusion --

    /// Multiply two 2x2 complex matrices in [`a_re`, `a_im`, `b_re`, `b_im`, `c_re`, `c_im`, `d_re`, `d_im`] format.
    fn matrix_mul(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
        // Complex multiply helper: (xr + xi*i) * (yr + yi*i)
        #[inline]
        fn cmul(xr: f64, xi: f64, yr: f64, yi: f64) -> (f64, f64) {
            (xr * yr - xi * yi, xr * yi + xi * yr)
        }

        // C = A * B where A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
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
    /// This turns `H(0), H(1), RZ(0), RZ(1)` into `H(0), RZ(0), H(1), RZ(1)`.
    fn reorder_for_fusion(queue: &mut [QueuedGate]) {
        // Find runs of single-qubit gates (between two-qubit gates) and sort by target
        let mut start = 0;
        while start < queue.len() {
            // Skip two-qubit gates
            if !matches!(
                queue[start].pipeline,
                GatePipeline::Single | GatePipeline::Diagonal
            ) {
                start += 1;
                continue;
            }

            // Find end of single-qubit run
            let mut end = start + 1;
            while end < queue.len()
                && matches!(
                    queue[end].pipeline,
                    GatePipeline::Single | GatePipeline::Diagonal
                )
            {
                end += 1;
            }

            // Sort the run by target qubit (stable sort preserves order within same qubit)
            queue[start..end].sort_by_key(|g| g.params.target_qubit);

            start = end;
        }
    }

    /// Fuse consecutive single-qubit gates on the same qubit by multiplying matrices.
    /// Returns a new queue with fewer gates.
    fn fuse_gate_queue(queue: &mut [QueuedGate]) -> Vec<QueuedGate> {
        Self::reorder_for_fusion(queue);
        if queue.len() <= 1 {
            return queue.to_vec();
        }

        let mut fused = Vec::with_capacity(queue.len());
        let mut i = 0;

        while i < queue.len() {
            let gate = &queue[i];

            // Only fuse single-qubit gates (Single or Diagonal)
            let is_1q = matches!(gate.pipeline, GatePipeline::Single | GatePipeline::Diagonal);
            if !is_1q {
                fused.push(queue[i].clone());
                i += 1;
                continue;
            }

            // Accumulate consecutive single-qubit gates on the same qubit
            let target = gate.params.target_qubit;
            let mut matrix = gate.params.matrix;
            let mut j = i + 1;

            while j < queue.len() {
                let next = &queue[j];
                let next_is_1q =
                    matches!(next.pipeline, GatePipeline::Single | GatePipeline::Diagonal);
                if !next_is_1q || next.params.target_qubit != target {
                    break;
                }
                matrix = Self::matrix_mul(&next.params.matrix, &matrix);
                j += 1;
            }

            // Check if the fused result is diagonal
            let is_diagonal =
                matrix[2] == 0.0 && matrix[3] == 0.0 && matrix[4] == 0.0 && matrix[5] == 0.0;

            fused.push(QueuedGate {
                pipeline: if is_diagonal {
                    GatePipeline::Diagonal
                } else {
                    GatePipeline::Single
                },
                params: GateParams64 {
                    target_qubit: target,
                    control_qubit: 0,
                    num_qubits: gate.params.num_qubits,
                    _padding: 0,
                    matrix,
                },
            });

            i = j;
        }

        fused
    }

    // -- Gate queue methods --

    #[allow(clippy::cast_possible_truncation)]
    /// Encode fused gates into the persistent kernel's storage buffer format.
    /// Buffer is array<f64>. Metadata fields stored as f64-encoded u32 values.
    /// Each gate: 12 f64 [type, tgt, ctrl, pad, matrix(8 x f64)]
    /// Header: [`num_gates`, `num_qubits`] as f64.
    fn encode_persistent_queue_f64(
        fused: &[QueuedGate],
        num_qubits: u32,
        staging: &mut Vec<u8>,
    ) -> usize {
        let num_gates = fused.len();
        let total_f64 = 2 + num_gates * 12;
        let total_bytes = total_f64 * 8;

        if staging.len() < total_bytes {
            staging.resize(total_bytes, 0);
        }

        let buf: &mut [f64] = bytemuck::cast_slice_mut(&mut staging[..total_bytes]);
        #[allow(clippy::cast_precision_loss)] // num_gates <= MAX_BATCH_SIZE (256), safe for f64
        {
            buf[0] = num_gates as f64;
        }
        buf[1] = f64::from(num_qubits);

        for (i, gate) in fused.iter().enumerate() {
            let base = 2 + i * 12;
            buf[base] = match gate.pipeline {
                GatePipeline::Single => 0.0,
                GatePipeline::Diagonal => 1.0,
                GatePipeline::CX => 2.0,
                GatePipeline::CY => 3.0,
                GatePipeline::CZ => 4.0,
                GatePipeline::Swap => 5.0,
                GatePipeline::Rxx => 6.0,
                GatePipeline::Ryy => 7.0,
                GatePipeline::Rzz => 8.0,
            };
            buf[base + 1] = f64::from(gate.params.target_qubit);
            buf[base + 2] = f64::from(gate.params.control_qubit);
            buf[base + 3] = 0.0;
            buf[base + 4..base + 12].copy_from_slice(&gate.params.matrix);
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
                label: Some("Flush gates encoder (f64)"),
            });
        self.record_flush_gates(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn record_flush_gates(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.gate_queue.is_empty() {
            return;
        }

        // Fuse consecutive single-qubit gates on the same qubit
        let fused = Self::fuse_gate_queue(&mut self.gate_queue);

        // Use persistent kernel if state fits in shared memory
        if self.num_qubits <= self.persistent_max_qubits {
            let total_bytes = Self::encode_persistent_queue_f64(
                &fused,
                self.num_qubits,
                &mut self.params_staging,
            );
            self.queue.write_buffer(
                &self.gate_queue_buffer,
                0,
                &self.params_staging[..total_bytes],
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Persistent kernel pass (f64)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.persistent_pipeline);
                pass.set_bind_group(0, &self.persistent_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
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
                label: Some("Batched gate pass (f64)"),
                timestamp_writes: None,
            });

            let num_pairs = self.num_amplitudes / 2;
            let (pair_wg_x, pair_wg_y) = Self::compute_workgroups(num_pairs);
            let (amp_wg_x, amp_wg_y) = Self::compute_workgroups(self.num_amplitudes);

            let mut current_pipeline = None;

            for (i, gate) in fused.iter().enumerate() {
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

    pub fn sync(&mut self) {
        self.flush_gates();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Convert an f32 gate matrix to f64 for the params struct.
    fn matrix_f32_to_f64(m: [f32; 8]) -> [f64; 8] {
        [
            f64::from(m[0]),
            f64::from(m[1]),
            f64::from(m[2]),
            f64::from(m[3]),
            f64::from(m[4]),
            f64::from(m[5]),
            f64::from(m[6]),
            f64::from(m[7]),
        ]
    }

    fn queue_single_gate(&mut self, qubit: u32, matrix: [f64; 8]) {
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
            params: GateParams64 {
                target_qubit: qubit,
                control_qubit: 0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix,
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_cx(&mut self, control: u32, target: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::CX,
            params: GateParams64 {
                target_qubit: target,
                control_qubit: control,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [0.0; 8],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_cz(&mut self, control: u32, target: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::CZ,
            params: GateParams64 {
                target_qubit: target,
                control_qubit: control,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [0.0; 8],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_cy(&mut self, control: u32, target: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::CY,
            params: GateParams64 {
                target_qubit: target,
                control_qubit: control,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [0.0; 8],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_swap(&mut self, qubit0: u32, qubit1: u32) {
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Swap,
            params: GateParams64 {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [0.0; 8],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_rxx(&mut self, qubit0: u32, qubit1: u32, theta: f64) {
        // Precompute cos/sin on the CPU -- wgpu+Vulkan doesn't reliably support
        // f64 transcendental functions in the shader. Pass (c, s) as f64 instead.
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Rxx,
            params: GateParams64 {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [c, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_ryy(&mut self, qubit0: u32, qubit1: u32, theta: f64) {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Ryy,
            params: GateParams64 {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [c, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    fn queue_rzz(&mut self, qubit0: u32, qubit1: u32, theta: f64) {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        self.gate_queue.push(QueuedGate {
            pipeline: GatePipeline::Rzz,
            params: GateParams64 {
                target_qubit: qubit1,
                control_qubit: qubit0,
                num_qubits: self.num_qubits,
                _padding: 0,
                matrix: [c, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        });
        if self.gate_queue.len() >= MAX_BATCH_SIZE {
            self.flush_gates();
        }
    }

    // -- Measurement --

    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn mz_cpu_path(&mut self, qubit: u32) -> (u32, bool) {
        const DET_EPS: f64 = 1e-10;

        let mut state_data = self.state();
        let target_mask = 1usize << qubit;

        let prob_one: f64 = state_data
            .iter()
            .enumerate()
            .filter(|(i, _)| i & target_mask != 0)
            .map(|(_, [re, im])| re * re + im * im)
            .sum();

        let is_deterministic = !(DET_EPS..=1.0 - DET_EPS).contains(&prob_one);
        let outcome = if is_deterministic {
            u32::from(prob_one > 0.5)
        } else {
            let random: f64 = self.rng.random();
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
        const DET_EPS: f64 = 1e-10;

        if self.num_qubits <= self.persistent_max_qubits {
            return self.mz_cpu_path(qubit);
        }
        let params = GateParams64 {
            target_qubit: qubit,
            control_qubit: 0,
            num_qubits: self.num_qubits,
            _padding: 0,
            matrix: [0.0; 8],
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Measurement encoder (f64)"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Marginal probability pass (f64)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.marginal_pipeline);
            pass.set_bind_group(0, &self.marginal_bind_group, &[]);
            let (wg_x, wg_y) = Self::compute_workgroups(self.num_amplitudes);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let readback_size = self.num_partial_sums * 8; // f64
        encoder.copy_buffer_to_buffer(
            &self.partial_sums_buffer,
            0,
            &self.staging_buffer,
            0,
            readback_size,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..readback_size);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU device poll failed");

        let prob_one: f64 = {
            let data = buffer_slice.get_mapped_range();
            let partial_sums: &[f64] = bytemuck::cast_slice(&data);
            partial_sums.iter().sum()
        };
        self.staging_buffer.unmap();

        let is_deterministic = !(DET_EPS..=1.0 - DET_EPS).contains(&prob_one);
        let outcome: u32 = if is_deterministic {
            u32::from(prob_one > 0.5)
        } else {
            let random: f64 = self.rng.random();
            u32::from(random < prob_one)
        };

        // Collapse
        let norm_factor = if outcome == 1 {
            1.0 / prob_one.sqrt()
        } else {
            1.0 / (1.0 - prob_one).sqrt()
        };

        let measure_params = MeasureParams64 {
            target_qubit: qubit,
            outcome,
            norm_factor,
        };
        self.queue.write_buffer(
            &self.measure_params_buffer,
            0,
            bytemuck::bytes_of(&measure_params),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Collapse encoder (f64)"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Collapse pass (f64)"),
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

    /// Read back the full state vector from GPU.
    ///
    /// # Panics
    /// Panics if the GPU device poll or buffer readback fails.
    #[must_use]
    pub fn state(&mut self) -> Vec<[f64; 2]> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("State readback encoder (f64)"),
            });
        self.record_flush_gates(&mut encoder);

        encoder.copy_buffer_to_buffer(
            &self.state_buffer,
            0,
            &self.staging_buffer,
            0,
            (self.num_amplitudes * 16) as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU device poll failed");

        let state: Vec<[f64; 2]> = {
            let data = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&data).to_vec()
        };
        self.staging_buffer.unmap();
        state
    }

    /// Overwrite the GPU state buffer with `amps`. Length must equal
    /// `num_amplitudes`; caller is responsible for the state being normalized.
    /// Pending queued gates are flushed first.
    ///
    /// # Panics
    /// Panics if `amps.len() != num_amplitudes`.
    pub fn write_state(&mut self, amps: &[[f64; 2]]) {
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

// -- Trait implementations --

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};

impl QuantumSimulator for GpuStateVec64 {
    fn reset(&mut self) -> &mut Self {
        self.reset();
        self
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits as usize
    }
}

#[allow(clippy::cast_possible_truncation)]
impl CliffordGateable for GpuStateVec64 {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::H);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::X);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::Y);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::Z);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::SX);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::SXDG);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::SY);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::SYDG);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::S);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::SDG);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
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
        let theta = std::f64::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rzz(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = -std::f64::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rzz(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = std::f64::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rxx(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = -std::f64::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_rxx(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = std::f64::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_ryy(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = -std::f64::consts::FRAC_PI_2;
        for &(q0, q1) in pairs {
            self.queue_ryy(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.flush_gates();

        // Empirical f64 mz path selection (RTX 4090, 2026-04-11).
        // M=1 always GPU: single readback amortizes poorly vs reduction+collapse.
        // Larger M covers the readback and the CPU loop wins for small N.
        // Re-run native_bench's f64 (N,M) probe to recalibrate.
        let min_m_for_batch = match self.num_qubits {
            0..=13 => 2,
            14 => 4,
            _ => usize::MAX,
        };
        if qubits.len() >= min_m_for_batch {
            self.mz_cpu_batch(qubits)
        } else {
            self.mz_gpu_sequential(qubits)
        }
    }
}

impl GpuStateVec64 {
    /// Read state, measure all qubits on CPU, write state back. Skips path
    /// selection -- intended for benchmarking and tests that need to force a
    /// specific path. Production code should call `mz()`.
    pub fn mz_cpu_batch(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        const DET_EPS: f64 = 1e-10;

        self.flush_gates();
        let mut state_data = self.state();
        let results: Vec<MeasurementResult> = qubits
            .iter()
            .map(|&q| {
                let target_mask = 1usize << q.index();

                let prob_one: f64 = state_data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i & target_mask != 0)
                    .map(|(_, [re, im])| re * re + im * im)
                    .sum();

                let is_deterministic = !(DET_EPS..=1.0 - DET_EPS).contains(&prob_one);
                let outcome = if is_deterministic {
                    u32::from(prob_one > 0.5)
                } else {
                    let random: f64 = self.rng.random();
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
impl ArbitraryRotationGateable for GpuStateVec64 {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::rx(theta.to_radians_signed()));
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::ry(theta.to_radians_signed()));
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::rz(theta.to_radians_signed()));
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::T);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        let m = Self::matrix_f32_to_f64(gates::TDG);
        for &q in qubits {
            self.queue_single_gate(q.index() as u32, m);
        }
        self
    }

    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &(q0, q1) in pairs {
            self.queue_rxx(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &(q0, q1) in pairs {
            self.queue_ryy(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        for &(q0, q1) in pairs {
            self.queue_rzz(q0.index() as u32, q1.index() as u32, theta);
        }
        self
    }
}
