//! GPU Stabilizer Simulator
//!
//! This implementation uses a persistent kernel approach that queues gates and
//! processes them in a single GPU dispatch, minimizing dispatch overhead.

// stab_x/stab_z and destab_x/destab_z are standard quantum stabilizer terminology
#![allow(clippy::similar_names)]
// GPU uses 32-bit values; casting from usize to u32 is intentional
#![allow(clippy::cast_possible_truncation)]

use crate::circuit_compiler::{CircuitCompiler, Gate as CompiledGate};
use crate::clifford_fusion::CliffordFuser;
use pecos_core::QubitId;
use pecos_random::{PecosRng, Rng, SeedableRng};
use pecos_simulators::{CliffordGateable, MeasurementResult, QuantumSimulator};
use std::collections::HashMap;
use std::fmt::Debug;

// Gate type constants (must match shader)
const GATE_H: u32 = 0;
const GATE_S: u32 = 1;
const GATE_SDG: u32 = 2;
const GATE_X: u32 = 3;
const GATE_Y: u32 = 4;
const GATE_Z: u32 = 5;
const GATE_CX: u32 = 6;
const GATE_CZ: u32 = 7;
const GATE_SWAP: u32 = 8;

/// Pack a single-qubit gate into the queue format
fn pack_single_gate(gate_type: u32, target: u32) -> u32 {
    (gate_type & 0xF) | ((target & 0x3FFF) << 4)
}

/// Pack a two-qubit gate into the queue format
fn pack_two_qubit_gate(gate_type: u32, control: u32, target: u32) -> u32 {
    (gate_type & 0xF) | ((target & 0x3FFF) << 4) | ((control & 0x3FFF) << 18)
}

/// Decode a packed gate to get (`gate_type`, `target_qubit`)
#[inline]
fn decode_gate(packed: u32) -> (u32, u32) {
    let gate_type = packed & 0xF;
    let target = (packed >> 4) & 0x3FFF;
    (gate_type, target)
}

/// Check if a gate type is single-qubit (can be safely reordered)
#[inline]
fn is_single_qubit_gate(gate_type: u32) -> bool {
    gate_type <= GATE_Z // H, S, SDG, X, Y, Z are single-qubit
}

/// Extract qubits touched by a gate
#[inline]
fn gate_qubits(packed: u32) -> (u32, Option<u32>) {
    let gate_type = packed & 0xF;
    let target = (packed >> 4) & 0x3FFF;
    let control = (packed >> 18) & 0x3FFF;
    if gate_type <= GATE_Z {
        // Single-qubit gate
        (target, None)
    } else {
        // Two-qubit gate
        (target, Some(control))
    }
}

/// Partition gates into independent batches for parallel processing.
/// Gates in the same batch don't share any qubits and can be processed in parallel.
/// Returns a list of (`start_idx`, `end_idx`) ranges into the gate queue.
fn partition_gates_into_batches(gate_queue: &[u32]) -> Vec<(usize, usize)> {
    if gate_queue.len() <= 1 {
        return Vec::new();
    }

    let gates = &gate_queue[1..]; // Skip num_gates header
    if gates.is_empty() {
        return Vec::new();
    }

    let mut batches = Vec::new();
    let mut batch_start = 0;
    let mut used_qubits = std::collections::HashSet::new();

    for (i, &packed) in gates.iter().enumerate() {
        let (q1, q2_opt) = gate_qubits(packed);

        // Check if this gate conflicts with current batch
        let conflicts =
            used_qubits.contains(&q1) || q2_opt.is_some_and(|q2| used_qubits.contains(&q2));

        if conflicts {
            // End current batch, start new one
            if i > batch_start {
                batches.push((batch_start + 1, i + 1)); // +1 to skip header
            }
            batch_start = i;
            used_qubits.clear();
        }

        // Add qubits to current batch
        used_qubits.insert(q1);
        if let Some(q2) = q2_opt {
            used_qubits.insert(q2);
        }
    }

    // Add final batch
    if gates.len() > batch_start {
        batches.push((batch_start + 1, gates.len() + 1));
    }

    batches
}

/// Sort runs of single-qubit gates by target qubit for better cache locality.
/// Two-qubit gates act as barriers - we only sort within runs of single-qubit gates.
/// This is safe because single-qubit gates on different qubits commute.
fn sort_gate_queue(gate_queue: &mut [u32]) {
    if gate_queue.len() <= 2 {
        return; // Nothing to sort (index 0 is num_gates header)
    }

    let gates = &mut gate_queue[1..]; // Skip the num_gates header
    let mut run_start = 0;

    while run_start < gates.len() {
        // Find the end of the current run of single-qubit gates
        let mut run_end = run_start;
        while run_end < gates.len() {
            let (gate_type, _) = decode_gate(gates[run_end]);
            if !is_single_qubit_gate(gate_type) {
                break;
            }
            run_end += 1;
        }

        // Sort the run by target qubit if it has more than one gate
        if run_end - run_start > 1 {
            gates[run_start..run_end].sort_by_key(|&packed| {
                let (_, target) = decode_gate(packed);
                target
            });
        }

        // Skip the two-qubit gate (if any) and continue
        run_start = run_end + 1;
    }
}

// Size of gate queue buffer - supports up to this many gates per sync
const GATE_QUEUE_BUFFER_SIZE: usize = 256 * 1024; // 256K gates

/// GPU Stabilizer simulator using persistent kernel approach.
///
/// Gates are queued and executed in batches to minimize dispatch overhead.
/// Uses deferred submission with multiple buffers to batch `queue.submit()` calls.
///
/// When subgroup operations are available (most modern GPUs), uses optimized
/// subgroup-based measurement reduction for improved performance.
#[allow(clippy::struct_excessive_bools)]
pub struct GpuStab<R: Rng + SeedableRng = PecosRng> {
    num_qubits: u32,
    gen_words: u32,
    rng: R,

    // GPU resources
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Tableau buffers
    stab_x_buffer: wgpu::Buffer,
    stab_z_buffer: wgpu::Buffer,
    destab_x_buffer: wgpu::Buffer,
    destab_z_buffer: wgpu::Buffer,
    sign_minus_buffer: wgpu::Buffer,
    sign_i_buffer: wgpu::Buffer,

    // Persistent kernel resources
    params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,

    // Single gate queue buffer for all pending gates
    gate_queue_buffer: wgpu::Buffer,
    main_bind_group: wgpu::BindGroup,

    // Pipeline
    process_queue_pipeline: wgpu::ComputePipeline,

    // Gate queue (CPU side) - gates accumulate here until sync()
    // Index 0 is reserved for num_gates header
    gate_queue: Vec<u32>,

    // For measurement
    anticommuting_buffer: wgpu::Buffer,
    find_anticommuting_bind_group: wgpu::BindGroup,
    find_anticommuting_pipeline: wgpu::ComputePipeline,

    // Subgroup support (for optimized measurement)
    has_subgroups: bool,
    subgroup_result_buffer: Option<wgpu::Buffer>,
    subgroup_find_pipeline: Option<wgpu::ComputePipeline>,
    subgroup_bind_group: Option<wgpu::BindGroup>,

    // Batch measurement support
    batch_qubits_buffer: wgpu::Buffer,
    batch_results_buffer: wgpu::Buffer,
    batch_random_buffer: wgpu::Buffer,
    batch_bind_group: wgpu::BindGroup,
    batch_pipeline: wgpu::ComputePipeline,
    deterministic_pipeline: wgpu::ComputePipeline,

    // Full measurement support (for non-deterministic measurements)
    measurement_data_buffer: wgpu::Buffer,
    #[allow(dead_code)] // Used by GPU shaders, not read on CPU
    saved_row_x_buffer: wgpu::Buffer,
    #[allow(dead_code)] // Used by GPU shaders, not read on CPU
    saved_row_z_buffer: wgpu::Buffer,
    measurement_bind_group: wgpu::BindGroup,
    meas_compute_weights_pipeline: wgpu::ComputePipeline,
    meas_extract_chosen_pipeline: wgpu::ComputePipeline,
    meas_xor_rows_pipeline: wgpu::ComputePipeline,
    meas_xor_destabs_pipeline: wgpu::ComputePipeline,
    meas_finalize_pipeline: wgpu::ComputePipeline,

    // Deferred measurement support - results accumulate on GPU
    deferred_results_buffer: wgpu::Buffer,
    pending_measurement_count: usize,

    // Batched command mode - accumulates command buffers for single submit
    batched_commands: Vec<wgpu::CommandBuffer>,
    batch_mode: bool,

    // Gate fusion support
    fuser: Option<CliffordFuser>,
    fusion_enabled: bool,

    // Compiled circuit support
    compiled_bind_group: wgpu::BindGroup,
    compiled_bind_group_layout: wgpu::BindGroupLayout,
    compiled_pipelines: HashMap<u64, wgpu::ComputePipeline>,
    circuit_compiler: CircuitCompiler,

    // Gate-parallel processing support
    parallel_bind_group: wgpu::BindGroup,
    parallel_pipeline: wgpu::ComputePipeline,
    parallel_enabled: bool,
}

impl GpuStab<PecosRng> {
    /// Create a new GPU stabilizer simulator with the given number of qubits.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails.
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        Self::with_seed(num_qubits, rand::random())
    }
}

impl<R: Rng + SeedableRng + Debug> GpuStab<R> {
    /// Create a new GPU stabilizer simulator with a specific RNG seed.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails.
    #[allow(clippy::too_many_lines)] // GPU initialization requires many setup steps
    pub fn with_seed(num_qubits: usize, seed: u64) -> Result<Self, String> {
        // Max 32K measurements for deferred measurements (enough for large surface codes)
        const MAX_BATCH_QUBITS_STAGING: u64 = 32 * 1024;
        // Batch measurement support
        // Max 32K measurements per batch (should be plenty for any practical circuit)
        const MAX_BATCH_QUBITS: u64 = 32 * 1024;
        let rng = R::seed_from_u64(seed);

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| "No GPU adapter found")?;

        // Check if subgroups are supported for optimized measurement
        let adapter_features = adapter.features();
        let has_subgroups = adapter_features.contains(wgpu::Features::SUBGROUP);

        // Request subgroup feature if available
        let required_features = if has_subgroups {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("GpuStab Device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        let num_qubits = num_qubits as u32;
        let gen_words = num_qubits.div_ceil(32);

        // Buffer sizes
        let tableau_size = u64::from(num_qubits) * u64::from(gen_words) * 4;
        let packed_signs_size = u64::from(gen_words) * 4; // Packed: one bit per generator
        let params_size = 32u64; // 8 u32s

        // Create tableau buffers
        let stab_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stab X Buffer"),
            size: tableau_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let stab_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stab Z Buffer"),
            size: tableau_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let destab_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destab X Buffer"),
            size: tableau_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let destab_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destab Z Buffer"),
            size: tableau_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sign_minus_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sign Minus Buffer"),
            size: packed_signs_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sign_i_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sign i Buffer"),
            size: packed_signs_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: params_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Single large gate queue buffer for all pending gates
        let gate_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gate Queue Buffer"),
            size: (GATE_QUEUE_BUFFER_SIZE as u64 + 1) * 4, // +1 for num_gates header
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One u32 per generator for anticommuting flags (not packed)
        let anticommuting_size = u64::from(num_qubits) * 4;

        let deferred_size = MAX_BATCH_QUBITS_STAGING * 4;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: tableau_size.max(anticommuting_size).max(deferred_size),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let anticommuting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Anticommuting Buffer"),
            size: anticommuting_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Load shaders
        let gate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Stab Gate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("stab_gate_shader.wgsl").into()),
        });

        let regular_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Regular Stab Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("stab_shaders.wgsl").into()),
        });

        // Create bind group layouts
        let main_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Main Bind Group Layout"),
                entries: &[
                    // stab_x
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
                    // stab_z
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // destab_x
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
                    // destab_z
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sign_minus (packed)
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
                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // gate_queue
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sign_i (packed)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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

        let find_anticommuting_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Find Anticommuting Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Single bind group for gate processing
        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main Bind Group"),
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: stab_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stab_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destab_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: destab_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sign_minus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gate_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sign_i_buffer.as_entire_binding(),
                },
            ],
        });

        // Compiled circuit bind group layout (no gate_queue binding)
        let compiled_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compiled Circuit Bind Group Layout"),
                entries: &[
                    // stab_x
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
                    // stab_z
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // destab_x
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
                    // destab_z
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sign_minus (packed)
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
                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // sign_i (packed) - binding 7
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
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

        // Compiled circuit bind group (no gate_queue)
        let compiled_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compiled Circuit Bind Group"),
            layout: &compiled_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: stab_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stab_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destab_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: destab_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sign_minus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sign_i_buffer.as_entire_binding(),
                },
            ],
        });

        let find_anticommuting_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Find Anticommuting Bind Group"),
            layout: &find_anticommuting_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: anticommuting_buffer.as_entire_binding(),
            }],
        });

        // Create pipelines
        let main_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Main Pipeline Layout"),
            bind_group_layouts: &[&main_bind_group_layout],
            immediate_size: 0,
        });

        let process_queue_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Process Queue Pipeline"),
                layout: Some(&main_pipeline_layout),
                module: &gate_shader,
                entry_point: Some("process_gate_queue"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Gate-parallel shader (uses atomic sign updates)
        let parallel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Parallel Gate Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("stab_gate_shader_parallel.wgsl").into()),
        });

        let parallel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parallel Gate Pipeline"),
            layout: Some(&main_pipeline_layout),
            module: &parallel_shader,
            entry_point: Some("process_gates_parallel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Parallel bind group uses the same layout as main
        let parallel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Parallel Bind Group"),
            layout: &main_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: stab_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stab_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destab_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: destab_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sign_minus_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: gate_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sign_i_buffer.as_entire_binding(),
                },
            ],
        });

        // For find_anticommuting, we need a different layout that uses the regular shader's params
        // We'll create a simplified version that reuses the main bind group
        let find_anticommuting_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Find Anticommuting Pipeline Layout"),
                bind_group_layouts: &[
                    &main_bind_group_layout,
                    &find_anticommuting_bind_group_layout,
                ],
                immediate_size: 0,
            });

        let find_anticommuting_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Find Anticommuting Pipeline"),
                layout: Some(&find_anticommuting_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("find_anticommuting"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Set up subgroup-based measurement if available
        // NOTE: As of wgpu 28/Naga, the `enable subgroups;` WGSL directive is not yet supported
        // in the shader compiler, even though the adapter may report SUBGROUP feature support.
        // We disable subgroups for now but keep the infrastructure for future use.
        // See: https://github.com/gfx-rs/wgpu/issues/5555
        //
        // When Naga adds support, change `false` below to `has_subgroups` to enable.
        let has_subgroups = false; // Disabled until Naga supports `enable subgroups;`
        let _ = has_subgroups; // Suppress unused warning

        let (subgroup_result_buffer, subgroup_find_pipeline, subgroup_bind_group) =
            (None, None, None);

        let batch_qubits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Qubits Buffer"),
            size: (MAX_BATCH_QUBITS + 1) * 4, // +1 for count header
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Results Buffer"),
            size: MAX_BATCH_QUBITS * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let batch_random_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Random Buffer"),
            size: MAX_BATCH_QUBITS * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Deferred results buffer - accumulates measurement outcomes
        let deferred_results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Deferred Results Buffer"),
            size: MAX_BATCH_QUBITS * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Full measurement support buffers (for non-deterministic measurements)
        // measurement_data layout: [min_weight, chosen_row, outcome, qubit, weights...]
        let measurement_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Measurement Data Buffer"),
            size: (4 + u64::from(num_qubits)) * 4, // 4 header fields + per-generator weights
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // saved_row buffers store the X/Z support of chosen generator (one bit per qubit)
        let saved_row_x_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Saved Row X Buffer"),
            size: u64::from(gen_words) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let saved_row_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Saved Row Z Buffer"),
            size: u64::from(gen_words) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let batch_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Batch Bind Group Layout"),
                entries: &[
                    // anticommuting buffer (binding 0, for compatibility)
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
                    // batch_qubits (binding 1)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // batch_results (binding 2)
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
                    // batch_random (binding 3)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let batch_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batch Bind Group"),
            layout: &batch_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: anticommuting_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_qubits_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: batch_results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: batch_random_buffer.as_entire_binding(),
                },
            ],
        });

        let batch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Batch Pipeline Layout"),
                bind_group_layouts: &[&main_bind_group_layout, &batch_bind_group_layout],
                immediate_size: 0,
            });

        let batch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batch Find Anticommuting Pipeline"),
            layout: Some(&batch_pipeline_layout),
            module: &regular_shader,
            entry_point: Some("find_anticommuting_batch"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let deterministic_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Deterministic Outcomes Pipeline"),
                layout: Some(&batch_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("compute_deterministic_outcomes"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Full measurement bind group layout (for non-deterministic measurements)
        // Bindings: 0=measurement_data, 1=saved_row_x, 2=saved_row_z
        let measurement_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Measurement Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        let measurement_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Measurement Bind Group"),
            layout: &measurement_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: measurement_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: saved_row_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: saved_row_z_buffer.as_entire_binding(),
                },
            ],
        });

        let measurement_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Measurement Pipeline Layout"),
                bind_group_layouts: &[&main_bind_group_layout, &measurement_bind_group_layout],
                immediate_size: 0,
            });

        let meas_compute_weights_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Measurement Compute Weights Pipeline"),
                layout: Some(&measurement_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("measurement_compute_weights"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let meas_extract_chosen_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Measurement Extract Chosen Pipeline"),
                layout: Some(&measurement_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("measurement_extract_chosen"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let meas_xor_rows_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Measurement XOR Rows Pipeline"),
                layout: Some(&measurement_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("measurement_xor_rows"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let meas_xor_destabs_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Measurement XOR Destabs Pipeline"),
                layout: Some(&measurement_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("measurement_xor_destabs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let meas_finalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Measurement Finalize Pipeline"),
                layout: Some(&measurement_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("measurement_finalize"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut sim = Self {
            num_qubits,
            gen_words,
            rng,
            device,
            queue,
            stab_x_buffer,
            stab_z_buffer,
            destab_x_buffer,
            destab_z_buffer,
            sign_minus_buffer,
            sign_i_buffer,
            params_buffer,
            staging_buffer,
            gate_queue_buffer,
            main_bind_group,
            process_queue_pipeline,
            gate_queue: {
                let mut q = Vec::with_capacity(1024);
                q.push(0); // Placeholder for num_gates at index 0
                q
            },
            anticommuting_buffer,
            find_anticommuting_bind_group,
            find_anticommuting_pipeline,
            has_subgroups,
            subgroup_result_buffer,
            subgroup_find_pipeline,
            subgroup_bind_group,
            batch_qubits_buffer,
            batch_results_buffer,
            batch_random_buffer,
            batch_bind_group,
            batch_pipeline,
            deterministic_pipeline,
            measurement_data_buffer,
            saved_row_x_buffer,
            saved_row_z_buffer,
            measurement_bind_group,
            meas_compute_weights_pipeline,
            meas_extract_chosen_pipeline,
            meas_xor_rows_pipeline,
            meas_xor_destabs_pipeline,
            meas_finalize_pipeline,
            deferred_results_buffer,
            pending_measurement_count: 0,
            batched_commands: Vec::new(),
            batch_mode: false,
            fuser: None,
            fusion_enabled: false,
            compiled_bind_group,
            compiled_bind_group_layout,
            compiled_pipelines: HashMap::new(),
            circuit_compiler: CircuitCompiler::new(),
            parallel_bind_group,
            parallel_pipeline,
            parallel_enabled: false,
        };

        // Initialize to |0...0> state
        sim.initialize_state();

        // Warmup: do a dummy dispatch to trigger shader JIT compilation
        // This moves the JIT overhead from first user sync to initialization
        sim.warmup_gpu();

        Ok(sim)
    }

    /// Warmup the GPU by doing a dummy dispatch.
    /// This triggers shader JIT compilation so first real sync is fast.
    fn warmup_gpu(&mut self) {
        // Queue a single no-op gate (will be overwritten by initialize_state anyway)
        self.gate_queue.push(pack_single_gate(GATE_H, 0));
        self.gate_queue[0] = 1; // num_gates = 1

        // Write to buffer
        self.queue.write_buffer(
            &self.gate_queue_buffer,
            0,
            bytemuck::cast_slice(&self.gate_queue),
        );

        // Create and submit warmup command
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.process_queue_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.dispatch_workgroups(self.gen_words.div_ceil(256), 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Wait for completion to ensure JIT is done
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Reset gate queue
        self.gate_queue.clear();
        self.gate_queue.push(0);

        // Re-initialize state (warmup modified it)
        self.initialize_state();
    }

    /// Initialize the tableau to the |0...0> state
    fn initialize_state(&mut self) {
        let num_qubits = self.num_qubits as usize;
        let gen_words = self.gen_words as usize;

        // Create initial tableau data
        let mut stab_z = vec![0u32; num_qubits * gen_words];
        let mut destab_x = vec![0u32; num_qubits * gen_words];

        // Set diagonal: stab_z[q, q] = 1, destab_x[q, q] = 1
        for q in 0..num_qubits {
            let word_idx = q / 32;
            let bit_pos = q % 32;
            let idx = q * gen_words + word_idx;
            stab_z[idx] |= 1 << bit_pos;
            destab_x[idx] |= 1 << bit_pos;
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.stab_x_buffer,
            0,
            &vec![0u8; num_qubits * gen_words * 4],
        );
        self.queue
            .write_buffer(&self.stab_z_buffer, 0, bytemuck::cast_slice(&stab_z));
        self.queue
            .write_buffer(&self.destab_x_buffer, 0, bytemuck::cast_slice(&destab_x));
        self.queue.write_buffer(
            &self.destab_z_buffer,
            0,
            &vec![0u8; num_qubits * gen_words * 4],
        );
        // Packed signs: one bit per generator -> gen_words u32s
        self.queue
            .write_buffer(&self.sign_minus_buffer, 0, &vec![0u8; gen_words * 4]);
        self.queue
            .write_buffer(&self.sign_i_buffer, 0, &vec![0u8; gen_words * 4]);

        // Write params once (these don't change per-flush)
        let params = [
            self.num_qubits,
            self.gen_words,
            self.num_qubits, // num_gens
            0u32,            // padding
            0u32,
            0u32,
            0u32,
            0u32,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));
    }

    /// Queue a single-qubit gate
    fn queue_single_gate(&mut self, gate_type: u32, qubit: u32) {
        self.add_single_gate(gate_type, qubit);
    }

    /// Queue a two-qubit gate
    fn queue_two_qubit_gate(&mut self, gate_type: u32, control: u32, target: u32) {
        self.add_two_qubit_gate(gate_type, control, target);
    }

    /// Submit all pending gates to the GPU and execute them.
    /// This is non-blocking - the GPU work may still be in progress when this returns.
    /// Call `wait()` to ensure completion before reading results.
    ///
    /// # Panics
    ///
    /// Panics if the gate queue exceeds the buffer capacity.
    pub fn sync(&mut self) {
        // Flush any pending fused gates first
        self.flush_fused_gates();

        // gate_queue[0] is reserved for num_gates, actual gates start at index 1
        if self.gate_queue.len() <= 1 {
            return;
        }

        // Check buffer capacity
        assert!(
            self.gate_queue.len() <= GATE_QUEUE_BUFFER_SIZE + 1,
            "Gate queue overflow: {} gates exceeds buffer size {}",
            self.gate_queue.len() - 1,
            GATE_QUEUE_BUFFER_SIZE
        );

        if self.parallel_enabled {
            self.sync_parallel_internal();
        } else {
            self.sync_sequential_internal();
        }

        // Reset gate queue, keeping placeholder for num_gates
        self.gate_queue.clear();
        self.gate_queue.push(0);
    }

    /// Sequential sync - processes all gates with one thread per generator word.
    fn sync_sequential_internal(&mut self) {
        // Sort single-qubit gates by target qubit for better cache locality
        sort_gate_queue(&mut self.gate_queue);

        // Update num_gates at index 0
        let num_gates = (self.gate_queue.len() - 1) as u32;
        self.gate_queue[0] = num_gates;

        // Write all gates to GPU buffer
        self.queue.write_buffer(
            &self.gate_queue_buffer,
            0,
            bytemuck::cast_slice(&self.gate_queue),
        );

        // Create and submit command buffer with single dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.process_queue_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.dispatch_workgroups(self.gen_words.div_ceil(256), 1, 1);
        }

        let cmd_buffer = encoder.finish();
        if self.batch_mode {
            self.batched_commands.push(cmd_buffer);
        } else {
            self.queue.submit(std::iter::once(cmd_buffer));
        }
    }

    /// Parallel sync - partitions gates into independent batches and processes in parallel.
    /// Each batch contains gates that don't share any qubits, allowing parallel execution.
    fn sync_parallel_internal(&mut self) {
        // Partition gates into independent batches
        let batches = partition_gates_into_batches(&self.gate_queue);

        if batches.is_empty() {
            return;
        }

        // Process each batch separately - we need to submit between batches because
        // write_buffer happens immediately on the CPU, so later writes would overwrite
        // earlier batches before the GPU processes them.
        for (batch_start, batch_end) in batches {
            let batch_gates = &self.gate_queue[batch_start..batch_end];
            let num_gates_in_batch = batch_gates.len() as u32;

            // Create a temporary gate queue with just this batch
            let mut batch_queue = Vec::with_capacity(batch_gates.len() + 1);
            batch_queue.push(num_gates_in_batch);
            batch_queue.extend_from_slice(batch_gates);

            // Write batch to GPU buffer
            self.queue.write_buffer(
                &self.gate_queue_buffer,
                0,
                bytemuck::cast_slice(&batch_queue),
            );

            // Dispatch with num_gates * gen_words threads
            let total_threads = num_gates_in_batch * self.gen_words;
            let workgroups = total_threads.div_ceil(256);

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Parallel Gate Encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.parallel_pipeline);
                pass.set_bind_group(0, &self.parallel_bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            let cmd_buffer = encoder.finish();
            if self.batch_mode {
                self.batched_commands.push(cmd_buffer);
            } else {
                self.queue.submit(std::iter::once(cmd_buffer));
                // Wait for this batch to complete before writing the next one
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            }
        }
    }

    /// Block until all submitted GPU work completes.
    /// Call this before reading results or when timing GPU execution.
    pub fn wait(&mut self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Submit pending gates and wait for completion (sync + wait).
    /// Use this when you need accurate timing or before reading results.
    pub fn sync_wait(&mut self) {
        self.sync();
        self.wait();
    }

    /// Flush is now a no-op - gates accumulate until `sync()` is called.
    /// This enables batching multiple flushes into a single GPU dispatch.
    pub fn flush(&mut self) {
        // Gates stay in gate_queue until sync() is called
        // This is intentional to reduce dispatch overhead
    }

    /// Begin batched command mode.
    ///
    /// In batch mode, gate and measurement dispatches are accumulated
    /// into a single command buffer submission. Call `end_batch()` to
    /// submit all accumulated commands and wait for completion.
    ///
    /// This reduces CPU-GPU synchronization overhead for workloads with
    /// many rounds of gates followed by measurements.
    pub fn begin_batch(&mut self) {
        self.batch_mode = true;
        self.batched_commands.clear();
    }

    /// End batched command mode and submit all accumulated commands.
    ///
    /// This submits all accumulated command buffers in one go, then
    /// waits for the GPU to complete all work.
    pub fn end_batch(&mut self) {
        if !self.batch_mode {
            return;
        }
        self.batch_mode = false;

        // Submit all accumulated command buffers at once
        if !self.batched_commands.is_empty() {
            self.queue
                .submit(std::mem::take(&mut self.batched_commands));
            self.wait();
        }
    }

    /// Enable gate fusion to reduce single-qubit gate count.
    ///
    /// When enabled, consecutive single-qubit gates on the same qubit are fused
    /// together. Common patterns like H*H=I, S*S=Z, S*Sdg=I are simplified.
    ///
    /// This can improve performance for circuits with many consecutive single-qubit
    /// gates, but has overhead for circuits dominated by two-qubit gates.
    pub fn enable_fusion(&mut self) {
        if !self.fusion_enabled {
            self.fusion_enabled = true;
            self.fuser = Some(CliffordFuser::new());
        }
    }

    /// Disable gate fusion.
    pub fn disable_fusion(&mut self) {
        if self.fusion_enabled {
            // Flush any pending fused gates
            self.flush_fused_gates();
            self.fusion_enabled = false;
            self.fuser = None;
        }
    }

    // =========================================================================
    // Gate-Parallel Processing
    // =========================================================================

    /// Enable gate-parallel processing.
    ///
    /// When enabled, gates are partitioned into independent batches where no two
    /// gates share a qubit. Each batch is processed with `num_gates * gen_words`
    /// threads instead of just `gen_words` threads.
    ///
    /// This can significantly increase GPU utilization for circuits with many
    /// independent gates (e.g., when H gates are applied to all data qubits).
    ///
    /// **Note**: This uses atomic sign updates which may have correctness issues
    /// with SDG gates if multiple SDG gates are in the same batch. For circuits
    /// using only H, S, X, Y, Z, CX, CZ, SWAP (surface code), this is safe.
    pub fn enable_parallel(&mut self) {
        self.parallel_enabled = true;
    }

    /// Disable gate-parallel processing (use sequential processing).
    pub fn disable_parallel(&mut self) {
        self.parallel_enabled = false;
    }

    /// Check if gate-parallel processing is enabled.
    pub fn is_parallel_enabled(&self) -> bool {
        self.parallel_enabled
    }

    // =========================================================================
    // Compiled Circuit Support
    // =========================================================================

    /// Compile a circuit for efficient repeated execution.
    ///
    /// This generates a specialized WGSL shader with all gates inlined,
    /// eliminating loop and switch overhead. Returns a hash that can be
    /// used to execute the compiled circuit.
    ///
    /// Compilation has overhead, so this is only beneficial for circuits
    /// that will be executed many times (e.g., syndrome extraction rounds).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_gpu_sims::GpuStab;
    /// use pecos_gpu_sims::circuit_compiler::Gate;
    ///
    /// let mut sim = GpuStab::new(4).unwrap();
    ///
    /// let gates = vec![
    ///     Gate::h(0),
    ///     Gate::cx(0, 1),
    ///     Gate::h(1),
    /// ];
    ///
    /// let hash = sim.compile_circuit(&gates);
    /// // Execute multiple times
    /// for _ in 0..100 {
    ///     sim.execute_compiled(hash);
    /// }
    /// ```
    pub fn compile_circuit(&mut self, gates: &[CompiledGate]) -> u64 {
        // Flush any pending gates first
        self.flush_fused_gates();

        // Compile the circuit (cached by hash)
        let compiled = self.circuit_compiler.compile(gates);
        let hash = compiled.hash;

        // Create pipeline if not already cached
        if !self.compiled_pipelines.contains_key(&hash) {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("Compiled Circuit {hash:016x}")),
                    source: wgpu::ShaderSource::Wgsl(compiled.wgsl_source.clone().into()),
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("Compiled Pipeline Layout {hash:016x}")),
                        bind_group_layouts: &[&self.compiled_bind_group_layout],
                        immediate_size: 0,
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Compiled Pipeline {hash:016x}")),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(&compiled.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            self.compiled_pipelines.insert(hash, pipeline);
        }

        hash
    }

    /// Execute a previously compiled circuit.
    ///
    /// The hash must come from a previous `compile_circuit` call.
    ///
    /// # Panics
    ///
    /// Panics if the hash doesn't correspond to a compiled circuit.
    pub fn execute_compiled(&mut self, hash: u64) {
        let pipeline = self
            .compiled_pipelines
            .get(&hash)
            .expect("Circuit not compiled - call compile_circuit first");

        // Create and submit command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.compiled_bind_group, &[]);
            pass.dispatch_workgroups(self.gen_words.div_ceil(256), 1, 1);
        }

        let cmd_buffer = encoder.finish();
        if self.batch_mode {
            self.batched_commands.push(cmd_buffer);
        } else {
            self.queue.submit(std::iter::once(cmd_buffer));
        }
    }

    /// Execute a compiled circuit and wait for completion.
    pub fn execute_compiled_wait(&mut self, hash: u64) {
        self.execute_compiled(hash);
        self.wait();
    }

    /// Check if a circuit is already compiled.
    pub fn is_circuit_compiled(&self, gates: &[CompiledGate]) -> bool {
        self.circuit_compiler.is_cached(gates)
    }

    /// Get the number of compiled circuits in the cache.
    pub fn compiled_circuit_count(&self) -> usize {
        self.compiled_pipelines.len()
    }

    /// Clear all compiled circuits from the cache.
    pub fn clear_compiled_circuits(&mut self) {
        self.compiled_pipelines.clear();
    }

    /// Flush pending fused gates to the gate queue.
    fn flush_fused_gates(&mut self) {
        if let Some(ref mut fuser) = self.fuser {
            let fused_gates = fuser.flush_all();
            // Add fused gates to gate queue (they're already packed)
            for packed_gate in fused_gates {
                if self.gate_queue.len() < GATE_QUEUE_BUFFER_SIZE + 1 {
                    self.gate_queue.push(packed_gate);
                }
            }
        }
    }

    /// Add a single-qubit gate, optionally through the fuser.
    fn add_single_gate(&mut self, gate_type: u32, target: u32) {
        if self.fusion_enabled
            && let Some(ref mut fuser) = self.fuser
        {
            fuser.add_gate(gate_type, target, 0);
            return;
        }
        // Direct path - add to gate queue
        self.gate_queue.push(pack_single_gate(gate_type, target));
    }

    /// Add a two-qubit gate, flushing any pending fused gates first.
    fn add_two_qubit_gate(&mut self, gate_type: u32, control: u32, target: u32) {
        if self.fusion_enabled
            && let Some(ref mut fuser) = self.fuser
        {
            // Two-qubit gate acts as barrier - fuser handles flushing
            fuser.add_gate(gate_type, target, control);
            // Get any flushed gates and add to queue
            let fused_gates = fuser.flush_all();
            for packed_gate in fused_gates {
                if self.gate_queue.len() < GATE_QUEUE_BUFFER_SIZE + 1 {
                    self.gate_queue.push(packed_gate);
                }
            }
            return;
        }
        // Direct path - add to gate queue
        self.gate_queue
            .push(pack_two_qubit_gate(gate_type, control, target));
    }

    /// Find first anticommuting stabilizer (for measurement)
    ///
    /// Uses subgroup-based parallel reduction when available for O(1) search.
    fn find_first_anticommuting(&mut self, qubit: u32) -> Option<usize> {
        // Flush and wait to ensure all pending gates are executed
        self.flush();
        self.sync_wait();

        // Update params for find_anticommuting
        let params = [
            self.num_qubits,
            self.gen_words,
            self.num_qubits, // num_gens
            qubit,           // target_qubit
            0,               // control_qubit (unused)
            0,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        // Use subgroup-based implementation if available
        if self.has_subgroups {
            return self.find_first_anticommuting_subgroup();
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Find Anticommuting Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Find Anticommuting Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.find_anticommuting_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.find_anticommuting_bind_group, &[]);
            pass.dispatch_workgroups(self.num_qubits.div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.anticommuting_buffer,
            0,
            &self.staging_buffer,
            0,
            u64::from(self.num_qubits) * 4,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read only the relevant portion
        let read_size = u64::from(self.num_qubits) * 4;
        let buffer_slice = self.staging_buffer.slice(..read_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().ok()?;

        let data = buffer_slice.get_mapped_range();
        let anticommuting: &[u32] = bytemuck::cast_slice(&data);
        let result = anticommuting.iter().position(|&x| x != 0);

        drop(data);
        self.staging_buffer.unmap();

        result
    }

    /// Subgroup-based `find_first_anticommuting` implementation
    ///
    /// Uses subgroupBallot + atomicMin for O(1) parallel search instead of
    /// sequential iteration through results.
    fn find_first_anticommuting_subgroup(&mut self) -> Option<usize> {
        let subgroup_result_buffer = self.subgroup_result_buffer.as_ref()?;
        let subgroup_find_pipeline = self.subgroup_find_pipeline.as_ref()?;
        let subgroup_bind_group = self.subgroup_bind_group.as_ref()?;

        // Initialize result buffer: [0] = 0xFFFFFFFF (no anticommuting found), [1] = 0 (count)
        let init_data = [0xFFFF_FFFFu32, 0u32];
        self.queue
            .write_buffer(subgroup_result_buffer, 0, bytemuck::cast_slice(&init_data));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Subgroup Find Anticommuting Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Subgroup Find Anticommuting Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(subgroup_find_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, subgroup_bind_group, &[]);
            // Dispatch one thread per generator
            pass.dispatch_workgroups(self.num_qubits.div_ceil(256), 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(subgroup_result_buffer, 0, &self.staging_buffer, 0, 8);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read result
        let buffer_slice = self.staging_buffer.slice(..8);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().ok()?;

        let data = buffer_slice.get_mapped_range();
        let result_data: &[u32] = bytemuck::cast_slice(&data);
        let first_anticommuting = result_data[0];

        drop(data);
        self.staging_buffer.unmap();

        if first_anticommuting == 0xFFFF_FFFF {
            None
        } else {
            Some(first_anticommuting as usize)
        }
    }

    /// Measure multiple qubits with fully GPU-accelerated outcome computation.
    ///
    /// All outcomes (both deterministic and non-deterministic) are computed on GPU.
    /// Non-deterministic measurements properly update the stabilizer tableau.
    ///
    /// Returns `MeasurementResult` for each qubit.
    fn measure_batch_gpu(&mut self, qubits: &[u32]) -> Vec<MeasurementResult> {
        if qubits.is_empty() {
            return Vec::new();
        }

        // Flush and wait to ensure all pending gates are executed
        self.flush();
        self.sync_wait();

        // Process each measurement sequentially to handle non-deterministic cases correctly
        // Non-deterministic measurements update the tableau, which affects subsequent measurements
        let mut results = Vec::with_capacity(qubits.len());

        for &qubit in qubits {
            let random_bit = self.rng.next_u32() & 1;
            let result = self.measure_single_qubit_gpu(qubit, random_bit);
            results.push(result);
        }

        results
    }

    /// Measure a single qubit on GPU with proper tableau update for non-deterministic cases.
    fn measure_single_qubit_gpu(&mut self, qubit: u32, random_bit: u32) -> MeasurementResult {
        // Step 1: Find if there's an anticommuting stabilizer
        let anticom_gen = self.find_anticommuting_single(qubit);

        if anticom_gen.is_none() {
            // Deterministic measurement - compute outcome from destabilizers
            let outcome = self.compute_deterministic_outcome_single(qubit);
            return MeasurementResult {
                outcome,
                is_deterministic: true,
            };
        }

        // Non-deterministic measurement - run full 5-stage measurement process
        let outcome = random_bit != 0;
        self.run_full_measurement(qubit, random_bit);

        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    /// Find the first anticommuting stabilizer for a single qubit.
    /// Returns None if deterministic, `Some(gen_index)` if non-deterministic.
    fn find_anticommuting_single(&mut self, qubit: u32) -> Option<u32> {
        // Prepare input: [count=1, qubit]
        let input_data = [1u32, qubit];
        self.queue.write_buffer(
            &self.batch_qubits_buffer,
            0,
            bytemuck::cast_slice(&input_data),
        );

        // Initialize result to 0xFFFFFFFF (deterministic)
        self.queue.write_buffer(
            &self.batch_results_buffer,
            0,
            bytemuck::cast_slice(&[0xFFFF_FFFFu32]),
        );

        // Update params
        let params = [
            self.num_qubits,
            self.gen_words,
            self.num_qubits, // num_gens
            qubit,           // target_qubit
            0,
            0,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        // Run find_anticommuting_batch for single qubit
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Find Anticommuting Single"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Find Anticommuting Single Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.batch_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.batch_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy result to staging
        encoder.copy_buffer_to_buffer(&self.batch_results_buffer, 0, &self.staging_buffer, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read result
        let buffer_slice = self.staging_buffer.slice(..4);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: u32 = bytemuck::cast_slice::<u8, u32>(&data)[0];
        drop(data);
        self.staging_buffer.unmap();

        if result == 0xFFFF_FFFF {
            None
        } else {
            Some(result)
        }
    }

    /// Compute deterministic outcome for a single qubit measurement.
    fn compute_deterministic_outcome_single(&mut self, qubit: u32) -> bool {
        // Prepare input: [count=1, qubit]
        let input_data = [1u32, qubit];
        self.queue.write_buffer(
            &self.batch_qubits_buffer,
            0,
            bytemuck::cast_slice(&input_data),
        );

        // Set result to 0xFFFFFFFF to indicate deterministic
        self.queue.write_buffer(
            &self.batch_results_buffer,
            0,
            bytemuck::cast_slice(&[0xFFFF_FFFFu32]),
        );

        // Random bit doesn't matter for deterministic, but need to provide it
        self.queue
            .write_buffer(&self.batch_random_buffer, 0, bytemuck::cast_slice(&[0u32]));

        // Update params
        let params = [
            self.num_qubits,
            self.gen_words,
            self.num_qubits,
            qubit,
            0,
            0,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        // Run compute_deterministic_outcomes
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Deterministic Single"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Deterministic Single Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.deterministic_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.batch_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy result to staging
        encoder.copy_buffer_to_buffer(&self.batch_results_buffer, 0, &self.staging_buffer, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read result
        let buffer_slice = self.staging_buffer.slice(..4);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: u32 = bytemuck::cast_slice::<u8, u32>(&data)[0];
        drop(data);
        self.staging_buffer.unmap();

        result != 0
    }

    /// Run the full 5-stage measurement process for non-deterministic measurement.
    /// This properly updates the stabilizer tableau.
    fn run_full_measurement(&mut self, qubit: u32, outcome: u32) {
        // Initialize measurement_data: [min_weight=0xFFFFFFFF, chosen_row=0, outcome, qubit, ...]
        let mut init_data = vec![0xFFFF_FFFFu32, 0, outcome, qubit];
        // Add space for per-generator weights
        init_data.resize(4 + self.num_qubits as usize, 0xFFFF_FFFFu32);
        self.queue.write_buffer(
            &self.measurement_data_buffer,
            0,
            bytemuck::cast_slice(&init_data),
        );

        // Update params with target qubit
        let params = [
            self.num_qubits,
            self.gen_words,
            self.num_qubits,
            qubit,
            0,
            0,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        // Stage 1: Compute weights and find minimum
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Full Measurement Stage 1"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Measurement Compute Weights"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.meas_compute_weights_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.measurement_bind_group, &[]);
            pass.dispatch_workgroups(self.num_qubits.div_ceil(256), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Stage 2: Extract chosen generator
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Full Measurement Stage 2"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Measurement Extract Chosen"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.meas_extract_chosen_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.measurement_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Stage 3: XOR chosen into other anticommuting stabilizers
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Full Measurement Stage 3"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Measurement XOR Rows"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.meas_xor_rows_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.measurement_bind_group, &[]);
            pass.dispatch_workgroups(self.gen_words.div_ceil(256), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Stage 4: XOR into anticommuting destabilizers
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Full Measurement Stage 4"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Measurement XOR Destabs"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.meas_xor_destabs_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.measurement_bind_group, &[]);
            pass.dispatch_workgroups(self.gen_words.div_ceil(256), 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        // Stage 5: Finalize - replace chosen stabilizer with Z_q
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Full Measurement Stage 5"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Measurement Finalize"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.meas_finalize_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_bind_group(1, &self.measurement_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Queue measurements without syncing - results accumulate on GPU.
    ///
    /// This is the deferred measurement API. Measurements are dispatched to GPU
    /// but results are not read back until `mz_fetch()` is called.
    ///
    /// Non-deterministic measurements are handled correctly with full tableau updates.
    /// Note: Measurements are processed sequentially to ensure correctness when
    /// non-deterministic measurements affect subsequent ones.
    ///
    /// Returns the number of measurements queued (for tracking purposes).
    pub fn mz_queue(&mut self, qubits: &[QubitId]) -> usize {
        if qubits.is_empty() {
            return 0;
        }

        // Process pending gates (dispatch to GPU, wait for completion)
        self.flush();
        self.sync_wait();

        let qubit_indices: Vec<u32> = qubits.iter().map(|q| q.index() as u32).collect();
        let count = qubit_indices.len();

        // Process each measurement sequentially to handle non-deterministic cases correctly
        // Results are written to deferred_results_buffer for later retrieval
        let mut results: Vec<u32> = Vec::with_capacity(count);

        for &qubit in &qubit_indices {
            let random_bit = self.rng.next_u32() & 1;
            let result = self.measure_single_qubit_gpu(qubit, random_bit);
            results.push(u32::from(result.outcome));
        }

        // Write results to deferred buffer at current offset
        let offset = (self.pending_measurement_count * 4) as u64;
        self.queue.write_buffer(
            &self.deferred_results_buffer,
            offset,
            bytemuck::cast_slice(&results),
        );

        // Track pending measurements
        self.pending_measurement_count += count;

        count
    }

    /// Flush all deferred measurements and return results.
    ///
    /// This reads back all accumulated measurement results from `mz_queue` calls.
    /// If in batch mode, this also ends the batch.
    ///
    /// # Panics
    /// Panics if the GPU buffer mapping fails.
    pub fn mz_fetch(&mut self) -> Vec<MeasurementResult> {
        // End batch mode if active
        if self.batch_mode {
            self.end_batch();
        }

        if self.pending_measurement_count == 0 {
            return Vec::new();
        }

        let count = self.pending_measurement_count;
        let results_size = (count * 4) as u64;

        // Copy deferred results to staging buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Flush Deferred Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.deferred_results_buffer,
            0,
            &self.staging_buffer,
            0,
            results_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results
        let buffer_slice = self.staging_buffer.slice(..results_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result_data: &[u32] = bytemuck::cast_slice(&data);

        let results: Vec<MeasurementResult> = result_data
            .iter()
            .map(|&v| MeasurementResult {
                outcome: v != 0,
                is_deterministic: true, // We don't track this in deferred mode
            })
            .collect();

        drop(data);
        self.staging_buffer.unmap();

        // Reset pending count
        self.pending_measurement_count = 0;

        results
    }

    /// Read a buffer from GPU
    fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u32> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &self.staging_buffer, 0, size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.staging_buffer.unmap();

        result
    }

    /// Get bit for a specific generator on a specific qubit (transposed layout)
    fn get_bit_transposed(data: &[u32], qubit: usize, gen_idx: usize, gen_words: usize) -> bool {
        let word_idx = gen_idx / 32;
        let bit_pos = gen_idx % 32;
        let idx = qubit * gen_words + word_idx;
        (data[idx] & (1 << bit_pos)) != 0
    }

    /// Compute deterministic measurement outcome
    fn compute_deterministic_outcome(&self, qubit: usize) -> bool {
        let num_qubits = self.num_qubits as usize;
        let gen_words = self.gen_words as usize;
        let tableau_size = (num_qubits * gen_words * 4) as u64;
        let packed_signs_size = (gen_words * 4) as u64;

        let destab_x = self.read_buffer(&self.destab_x_buffer, tableau_size);
        let stab_x = self.read_buffer(&self.stab_x_buffer, tableau_size);
        let stab_z = self.read_buffer(&self.stab_z_buffer, tableau_size);
        let sign_minus = self.read_buffer(&self.sign_minus_buffer, packed_signs_size);
        let sign_i = self.read_buffer(&self.sign_i_buffer, packed_signs_size);

        let mut num_minuses = 0usize;
        let mut num_is = 0usize;
        let mut cumulative_x = vec![false; num_qubits];

        for gen_idx in 0..num_qubits {
            if Self::get_bit_transposed(&destab_x, qubit, gen_idx, gen_words) {
                // Read packed sign bits
                let word_idx = gen_idx / 32;
                let bit_pos = gen_idx % 32;
                if (sign_minus[word_idx] & (1 << bit_pos)) != 0 {
                    num_minuses += 1;
                }
                if (sign_i[word_idx] & (1 << bit_pos)) != 0 {
                    num_is += 1;
                }

                for (q2, &cx) in cumulative_x.iter().enumerate().take(num_qubits) {
                    if cx && Self::get_bit_transposed(&stab_z, q2, gen_idx, gen_words) {
                        num_minuses += 1;
                    }
                }

                for (q2, cx) in cumulative_x.iter_mut().enumerate().take(num_qubits) {
                    if Self::get_bit_transposed(&stab_x, q2, gen_idx, gen_words) {
                        *cx = !*cx;
                    }
                }
            }
        }

        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        !num_minuses.is_multiple_of(2)
    }
}

impl<R: Rng + SeedableRng + Debug> CliffordGateable for GpuStab<R> {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(GATE_H, q.index() as u32);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(GATE_S, q.index() as u32);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(GATE_SDG, q.index() as u32);
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(GATE_X, q.index() as u32);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(GATE_Y, q.index() as u32);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.queue_single_gate(GATE_Z, q.index() as u32);
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.queue_two_qubit_gate(GATE_CX, q0.index() as u32, q1.index() as u32);
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.queue_two_qubit_gate(GATE_CZ, q0.index() as u32, q1.index() as u32);
        }
        self
    }

    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.queue_two_qubit_gate(GATE_SWAP, q0.index() as u32, q1.index() as u32);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        if qubits.is_empty() {
            return Vec::new();
        }

        // Use fully GPU-accelerated batch measurement
        let qubit_indices: Vec<u32> = qubits.iter().map(|q| q.index() as u32).collect();
        self.measure_batch_gpu(&qubit_indices)
    }
}

impl<R: Rng + SeedableRng + Debug> GpuStab<R> {
    /// Measure qubit in Z basis with forced outcome for non-deterministic cases.
    ///
    /// If the measurement is deterministic, returns the determined outcome.
    /// If non-deterministic, forces the measurement to the specified outcome
    /// and properly updates the stabilizer tableau.
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        // Must sync to execute queued gates before measuring
        self.sync();

        let qubit_u32 = qubit as u32;
        let first_anticommuting = self.find_first_anticommuting(qubit_u32);

        if first_anticommuting.is_some() {
            // Non-deterministic - force outcome and update tableau
            let outcome_bit = u32::from(forced_outcome);
            self.run_full_measurement(qubit_u32, outcome_bit);
            MeasurementResult {
                outcome: forced_outcome,
                is_deterministic: false,
            }
        } else {
            // Deterministic
            let outcome = self.compute_deterministic_outcome(qubit);
            MeasurementResult {
                outcome,
                is_deterministic: true,
            }
        }
    }
}

impl<R: Rng + SeedableRng + Debug> QuantumSimulator for GpuStab<R> {
    fn reset(&mut self) -> &mut Self {
        // Wait for any pending GPU work before resetting
        self.wait();
        self.gate_queue.clear();
        self.gate_queue.push(0); // Placeholder for num_gates at index 0
        self.initialize_state();
        self
    }
}

impl<R: Rng + SeedableRng + Debug> Debug for GpuStab<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStab")
            .field("num_qubits", &self.num_qubits)
            .field("queued_gates", &self.gate_queue.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_simulators::SparseStab;
    use pecos_simulators::stabilizer_test_utils::{
        ForcedMeasurement, compare_simulators_on_random_circuits_direct,
        run_basic_stabilizer_test_suite,
    };

    impl<R: Rng + SeedableRng + Debug> ForcedMeasurement for GpuStab<R> {
        fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
            GpuStab::mz_forced(self, qubit, forced_outcome)
        }
    }

    // ========================================================================
    // Basic Stabilizer Test Suite (no Clone required)
    // ========================================================================

    /// Run the basic stabilizer test suite on the GPU simulator.
    /// This tests gate identities, entanglement correlations, etc.
    /// Note: GPU simulators don't implement Clone (GPU resources can't be cloned),
    /// so we use the basic suite instead of the full suite.
    #[test]
    fn test_gpu_stab_basic_suite() {
        let Some(mut gpu) = gpu_sim(8, 42) else {
            return;
        };
        run_basic_stabilizer_test_suite(&mut gpu, 8);
    }

    /// Run the basic suite on a smaller number of qubits
    #[test]
    fn test_gpu_stab_basic_suite_small() {
        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        run_basic_stabilizer_test_suite(&mut gpu, 4);
    }

    // ========================================================================
    // GPU vs CPU Comparison Tests
    // ========================================================================

    /// Compare GPU and CPU simulators on random circuits.
    /// This uses the direct comparison method which doesn't require Clone.
    #[test]
    fn test_gpu_vs_cpu_random_circuits() {
        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);
        // Run 20 random circuits of 30 gates each with seed 12345
        compare_simulators_on_random_circuits_direct(&mut gpu, &mut cpu, 4, 30, 20, 12345);
    }

    /// Compare GPU and CPU on larger random circuits
    #[test]
    fn test_gpu_vs_cpu_random_circuits_large() {
        let Some(mut gpu) = gpu_sim(8, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(8);
        // Run 10 random circuits of 50 gates each
        compare_simulators_on_random_circuits_direct(&mut gpu, &mut cpu, 8, 50, 10, 67890);
    }

    /// Compare GPU and CPU with many circuits to ensure consistency
    #[test]
    fn test_gpu_vs_cpu_many_circuits() {
        let Some(mut gpu) = gpu_sim(5, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(5);
        // Run 50 random circuits of 20 gates each
        compare_simulators_on_random_circuits_direct(&mut gpu, &mut cpu, 5, 20, 50, 11111);
    }

    /// Debug test to understand the specific failing circuit
    #[test]
    fn test_gpu_vs_cpu_specific_circuit_debug() {
        use pecos_random::PecosRng;
        use pecos_simulators::stabilizer_test_utils::{
            apply_circuit, generate_random_clifford_circuit,
        };

        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        // Use seed 12353 which was failing (base_seed 12345 + 8)
        let seed = 12353u64;
        let mut rng = PecosRng::seed_from_u64(seed);
        let circuit = generate_random_clifford_circuit(&mut rng, 4, 30);

        // Print the circuit for debugging
        println!("Circuit with seed {seed}:");
        for (i, gate) in circuit.iter().enumerate() {
            println!("  {i}: {gate:?}");
        }

        // Apply to both simulators
        gpu.reset();
        cpu.reset();
        apply_circuit(&mut gpu, &circuit);
        apply_circuit(&mut cpu, &circuit);

        // Make sure GPU has synced
        gpu.sync();
        gpu.wait();

        // Compare determinism for each qubit
        println!("\nDeterminism comparison:");
        for q in 0..4 {
            // Use the find_first_anticommuting logic to check determinism
            let gpu_result = gpu.mz_forced(q, false);
            let cpu_result = cpu.mz_forced(q, false);
            println!(
                "  Qubit {q}: GPU deterministic={}, CPU deterministic={}",
                gpu_result.is_deterministic, cpu_result.is_deterministic
            );
            if gpu_result.is_deterministic != cpu_result.is_deterministic {
                println!("    MISMATCH!");
            }
        }
    }

    /// Test simpler circuits to find where divergence occurs
    #[test]
    fn test_gpu_vs_cpu_simpler_debug() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // Test 1: Just H on qubit 0 - should be non-deterministic
        println!("Test 1: H on qubit 0");
        gpu.reset();
        cpu.reset();
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        gpu.sync();
        gpu.wait();

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);
        println!(
            "  GPU: det={}, CPU: det={}",
            gpu_r.is_deterministic, cpu_r.is_deterministic
        );
        assert_eq!(
            gpu_r.is_deterministic, cpu_r.is_deterministic,
            "H test failed"
        );

        // Test 2: SY gate - H*X decomposition
        println!("\nTest 2: SY on qubit 0 (uses H*X decomposition)");
        gpu.reset();
        cpu.reset();
        gpu.sy(&[QubitId::new(0)]);
        cpu.sy(&[QubitId::new(0)]);
        gpu.sync();
        gpu.wait();

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);
        println!(
            "  GPU: det={}, CPU: det={}",
            gpu_r.is_deterministic, cpu_r.is_deterministic
        );
        assert_eq!(
            gpu_r.is_deterministic, cpu_r.is_deterministic,
            "SY test failed"
        );

        // Test 3: CY gate - Szdg*CX*Sz decomposition
        println!("\nTest 3: CY(0,1) (uses Szdg*CX*Sz decomposition)");
        gpu.reset();
        cpu.reset();
        gpu.cy(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cy(&[(QubitId::new(0), QubitId::new(1))]);
        gpu.sync();
        gpu.wait();

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r1 = cpu.mz_forced(1, false);
        println!(
            "  Q0: GPU det={}, CPU det={}",
            gpu_r0.is_deterministic, cpu_r0.is_deterministic
        );
        println!(
            "  Q1: GPU det={}, CPU det={}",
            gpu_r1.is_deterministic, cpu_r1.is_deterministic
        );
        assert_eq!(
            gpu_r0.is_deterministic, cpu_r0.is_deterministic,
            "CY test Q0 failed"
        );
        assert_eq!(
            gpu_r1.is_deterministic, cpu_r1.is_deterministic,
            "CY test Q1 failed"
        );

        // Test 4: Bell state - H then CX
        println!("\nTest 4: Bell state H(0) CX(0,1)");
        gpu.reset();
        cpu.reset();
        gpu.h(&[QubitId::new(0)]);
        gpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.h(&[QubitId::new(0)]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        gpu.sync();
        gpu.wait();

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r1 = cpu.mz_forced(1, false);
        println!(
            "  Q0: GPU det={}, CPU det={}",
            gpu_r0.is_deterministic, cpu_r0.is_deterministic
        );
        println!(
            "  Q1: GPU det={}, CPU det={}",
            gpu_r1.is_deterministic, cpu_r1.is_deterministic
        );
        assert_eq!(
            gpu_r0.is_deterministic, cpu_r0.is_deterministic,
            "Bell test Q0 failed"
        );
        assert_eq!(
            gpu_r1.is_deterministic, cpu_r1.is_deterministic,
            "Bell test Q1 failed"
        );

        println!("\nAll simple tests passed!");
    }

    /// Test the specific failing circuit - check deterministic outcomes without prior measurements
    #[test]
    fn test_gpu_vs_cpu_failing_circuit_deterministic_only() {
        use pecos_random::PecosRng;
        use pecos_simulators::stabilizer_test_utils::{
            apply_circuit, generate_random_clifford_circuit,
        };

        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        // Use seed 12354 which was failing
        let seed = 12354u64;
        let mut rng = PecosRng::seed_from_u64(seed);
        let circuit = generate_random_clifford_circuit(&mut rng, 4, 30);

        println!("Circuit with seed {seed}:");
        for (i, gate) in circuit.iter().enumerate() {
            println!("  {i}: {gate:?}");
        }

        gpu.reset();
        cpu.reset();
        apply_circuit(&mut gpu, &circuit);
        apply_circuit(&mut cpu, &circuit);
        gpu.sync();
        gpu.wait();

        // Check determinism for all qubits WITHOUT measuring (so we don't modify state)
        println!("\nDeterminism check (without measuring):");
        for q in 0..4 {
            let gpu_det = gpu.find_first_anticommuting(q as u32).is_none();
            let mut cpu_copy = cpu.clone();
            let cpu_det = cpu_copy.mz_forced(q, false).is_deterministic;
            println!("  Q{q}: GPU det={gpu_det}, CPU det={cpu_det}");
        }

        // Now check deterministic outcomes - measure on copies
        println!("\nDeterministic outcome comparison (on fresh copies):");
        for q in 0..4 {
            let gpu_det = gpu.find_first_anticommuting(q as u32).is_none();
            if gpu_det {
                // Both should be deterministic - check the outcome
                let gpu_outcome = gpu.compute_deterministic_outcome(q);
                let mut cpu_copy = cpu.clone();
                let cpu_result = cpu_copy.mz_forced(q, false);
                assert!(
                    cpu_result.is_deterministic,
                    "CPU should be deterministic for Q{q}"
                );
                println!(
                    "  Q{q}: GPU out={}, CPU out={}{}",
                    gpu_outcome,
                    cpu_result.outcome,
                    if gpu_outcome == cpu_result.outcome {
                        ""
                    } else {
                        " MISMATCH!"
                    }
                );
                assert_eq!(gpu_outcome, cpu_result.outcome, "Q{q} outcome mismatch");
            }
        }
    }

    /// Test sequential measurements to find where divergence occurs
    /// Test the failing circuit exactly - gates 0-22
    #[test]
    fn test_gpu_vs_cpu_failing_circuit() {
        use pecos_random::PecosRng;
        use pecos_simulators::stabilizer_test_utils::{
            CliffordGate, generate_random_clifford_circuit,
        };

        // Helper to read GPU sign state
        fn read_gpu_signs(gpu: &GpuStab) -> (Vec<bool>, Vec<bool>) {
            let gen_words = gpu.gen_words as usize;
            let packed_signs_size = (gen_words * 4) as u64;
            let sign_minus_raw = gpu.read_buffer(&gpu.sign_minus_buffer, packed_signs_size);
            let sign_i_raw = gpu.read_buffer(&gpu.sign_i_buffer, packed_signs_size);

            let mut sign_minus = vec![false; gpu.num_qubits as usize];
            let mut sign_i = vec![false; gpu.num_qubits as usize];
            for g in 0..gpu.num_qubits as usize {
                let word = g / 32;
                let bit = g % 32;
                sign_minus[g] = (sign_minus_raw[word] & (1 << bit)) != 0;
                sign_i[g] = (sign_i_raw[word] & (1 << bit)) != 0;
            }
            (sign_minus, sign_i)
        }

        // Helper to read CPU sign state
        fn read_cpu_signs(cpu: &SparseStab, num_qubits: usize) -> (Vec<bool>, Vec<bool>) {
            let mut sign_minus = vec![false; num_qubits];
            let mut sign_i = vec![false; num_qubits];
            for g in &cpu.stabs().signs_minus {
                if g < num_qubits {
                    sign_minus[g] = true;
                }
            }
            for g in &cpu.stabs().signs_i {
                if g < num_qubits {
                    sign_i[g] = true;
                }
            }
            (sign_minus, sign_i)
        }

        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        let seed = 12354u64;
        let mut rng = PecosRng::seed_from_u64(seed);
        let circuit = generate_random_clifford_circuit(&mut rng, 4, 30);

        gpu.reset();
        cpu.reset();

        // Apply gates 0-21 (which all pass)
        println!("Applying gates 0-21:");
        for (i, gate) in circuit.iter().enumerate().take(22) {
            match gate {
                CliffordGate::H(q) => {
                    gpu.h(&[QubitId::new(*q)]);
                    cpu.h(&[QubitId::new(*q)]);
                }
                CliffordGate::S(q) => {
                    gpu.sz(&[QubitId::new(*q)]);
                    cpu.sz(&[QubitId::new(*q)]);
                }
                CliffordGate::Sdg(q) => {
                    gpu.szdg(&[QubitId::new(*q)]);
                    cpu.szdg(&[QubitId::new(*q)]);
                }
                CliffordGate::X(q) => {
                    gpu.x(&[QubitId::new(*q)]);
                    cpu.x(&[QubitId::new(*q)]);
                }
                CliffordGate::Y(q) => {
                    gpu.y(&[QubitId::new(*q)]);
                    cpu.y(&[QubitId::new(*q)]);
                }
                CliffordGate::Z(q) => {
                    gpu.z(&[QubitId::new(*q)]);
                    cpu.z(&[QubitId::new(*q)]);
                }
                CliffordGate::CX(c, t) => {
                    gpu.cx(&[(QubitId::new(*c), QubitId::new(*t))]);
                    cpu.cx(&[(QubitId::new(*c), QubitId::new(*t))]);
                }
                CliffordGate::CZ(a, b) => {
                    gpu.cz(&[(QubitId::new(*a), QubitId::new(*b))]);
                    cpu.cz(&[(QubitId::new(*a), QubitId::new(*b))]);
                }
                CliffordGate::SWAP(a, b) => {
                    gpu.swap(&[(QubitId::new(*a), QubitId::new(*b))]);
                    cpu.swap(&[(QubitId::new(*a), QubitId::new(*b))]);
                }
                CliffordGate::SX(q) => {
                    gpu.sx(&[QubitId::new(*q)]);
                    cpu.sx(&[QubitId::new(*q)]);
                }
                CliffordGate::SXdg(q) => {
                    gpu.sxdg(&[QubitId::new(*q)]);
                    cpu.sxdg(&[QubitId::new(*q)]);
                }
                CliffordGate::SY(q) => {
                    gpu.sy(&[QubitId::new(*q)]);
                    cpu.sy(&[QubitId::new(*q)]);
                }
                CliffordGate::SYdg(q) => {
                    gpu.sydg(&[QubitId::new(*q)]);
                    cpu.sydg(&[QubitId::new(*q)]);
                }
                CliffordGate::CY(c, t) => {
                    gpu.cy(&[(QubitId::new(*c), QubitId::new(*t))]);
                    cpu.cy(&[(QubitId::new(*c), QubitId::new(*t))]);
                }
            }
            gpu.sync();
            gpu.wait();
            println!("  Gate {i}: {gate:?}");
        }

        // Check state after gate 21
        gpu.sync();
        gpu.wait();
        let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
        println!("\nAfter gate 21:");
        println!("  GPU signs_minus: {gpu_sm:?}");
        println!("  CPU signs_minus: {cpu_sm:?}");
        println!("  GPU signs_i:     {gpu_si:?}");
        println!("  CPU signs_i:     {cpu_si:?}");
        assert_eq!(gpu_sm, cpu_sm, "Signs differ after gate 21!");
        assert_eq!(gpu_si, cpu_si, "signs_i differ after gate 21!");

        // Now apply gate 22 (SX(3)) step by step: H*S*H
        println!("\nApplying SX(3) step by step:");

        // Step 1: H(3)
        println!("  Step 1: H(3)");
        gpu.h(&[QubitId::new(3)]);
        cpu.h(&[QubitId::new(3)]);
        gpu.sync();
        gpu.wait();
        let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
        println!("    GPU signs_minus: {gpu_sm:?}");
        println!("    CPU signs_minus: {cpu_sm:?}");
        println!("    GPU signs_i:     {gpu_si:?}");
        println!("    CPU signs_i:     {cpu_si:?}");
        assert_eq!(gpu_sm, cpu_sm, "Signs differ after H!");
        assert_eq!(gpu_si, cpu_si, "signs_i differ after H!");

        // Step 2: S(3)
        println!("  Step 2: S(3)");

        // Before S gate, check X bits at qubit 3
        let gen_words = gpu.gen_words as usize;
        let tableau_size = (4 * gen_words * 4) as u64;
        let gpu_stab_x = gpu.read_buffer(&gpu.stab_x_buffer, tableau_size);
        println!("    Before S: GPU stab_x for qubit 3:");
        for g in 0..4 {
            let word = g / 32;
            let bit = g % 32;
            let offset = 3 * gen_words + word; // qubit 3
            let has_x = (gpu_stab_x[offset] & (1 << bit)) != 0;
            println!("      Generator {g}: X={has_x}");
        }

        // Also check CPU
        println!("    Before S: CPU stab.col_x[3]:");
        for g in &cpu.stabs().col_x[3] {
            println!("      Generator {g} has X");
        }

        gpu.sz(&[QubitId::new(3)]);
        cpu.sz(&[QubitId::new(3)]);
        gpu.sync();
        gpu.wait();
        let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
        println!("    GPU signs_minus: {gpu_sm:?}");
        println!("    CPU signs_minus: {cpu_sm:?}");
        println!("    GPU signs_i:     {gpu_si:?}");
        println!("    CPU signs_i:     {cpu_si:?}");
        assert_eq!(gpu_sm, cpu_sm, "Signs differ after S!");
        assert_eq!(gpu_si, cpu_si, "signs_i differ after S!");

        // Step 3: H(3)
        println!("  Step 3: H(3) (final)");
        gpu.h(&[QubitId::new(3)]);
        cpu.h(&[QubitId::new(3)]);
        gpu.sync();
        gpu.wait();
        let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
        println!("    GPU signs_minus: {gpu_sm:?}");
        println!("    CPU signs_minus: {cpu_sm:?}");
        println!("    GPU signs_i:     {gpu_si:?}");
        println!("    CPU signs_i:     {cpu_si:?}");
        assert_eq!(gpu_sm, cpu_sm, "Signs differ after final H!");
        assert_eq!(gpu_si, cpu_si, "signs_i differ after final H!");

        println!("\nSX decomposition passed!");
    }

    /// Test SX gate in isolation
    #[test]
    fn test_gpu_vs_cpu_sx_gate() {
        // Helper to read GPU sign state
        fn read_gpu_signs(gpu: &GpuStab) -> (Vec<bool>, Vec<bool>) {
            let gen_words = gpu.gen_words as usize;
            let packed_signs_size = (gen_words * 4) as u64;
            let sign_minus_raw = gpu.read_buffer(&gpu.sign_minus_buffer, packed_signs_size);
            let sign_i_raw = gpu.read_buffer(&gpu.sign_i_buffer, packed_signs_size);

            let mut sign_minus = vec![false; gpu.num_qubits as usize];
            let mut sign_i = vec![false; gpu.num_qubits as usize];
            for g in 0..gpu.num_qubits as usize {
                let word = g / 32;
                let bit = g % 32;
                sign_minus[g] = (sign_minus_raw[word] & (1 << bit)) != 0;
                sign_i[g] = (sign_i_raw[word] & (1 << bit)) != 0;
            }
            (sign_minus, sign_i)
        }

        // Helper to read CPU sign state
        fn read_cpu_signs(cpu: &SparseStab, num_qubits: usize) -> (Vec<bool>, Vec<bool>) {
            let mut sign_minus = vec![false; num_qubits];
            let mut sign_i = vec![false; num_qubits];
            for g in &cpu.stabs().signs_minus {
                if g < num_qubits {
                    sign_minus[g] = true;
                }
            }
            for g in &cpu.stabs().signs_i {
                if g < num_qubits {
                    sign_i[g] = true;
                }
            }
            (sign_minus, sign_i)
        }

        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        println!("SX gate test on fresh state:");

        // Apply SX(3) on fresh state
        gpu.reset();
        cpu.reset();
        gpu.sx(&[QubitId::new(3)]);
        cpu.sx(&[QubitId::new(3)]);
        gpu.sync();
        gpu.wait();

        let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
        println!("After SX(3) on fresh state:");
        println!("  GPU signs_minus: {gpu_sm:?}");
        println!("  CPU signs_minus: {cpu_sm:?}");
        println!("  GPU signs_i:     {gpu_si:?}");
        println!("  CPU signs_i:     {cpu_si:?}");

        assert_eq!(
            gpu_sm, cpu_sm,
            "signs_minus mismatch after SX(3) on fresh state"
        );
        assert_eq!(
            gpu_si, cpu_si,
            "signs_i mismatch after SX(3) on fresh state"
        );

        // Now test H then SX - since SX = H*S*H, this tests the components
        println!("\nTesting H(3) then SX(3):");
        gpu.reset();
        cpu.reset();
        gpu.h(&[QubitId::new(3)]);
        cpu.h(&[QubitId::new(3)]);
        gpu.sync();
        gpu.wait();

        let (gpu_sm, _gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, _cpu_si) = read_cpu_signs(&cpu, 4);
        println!("After H(3):");
        println!("  GPU signs_minus: {gpu_sm:?}");
        println!("  CPU signs_minus: {cpu_sm:?}");
        assert_eq!(gpu_sm, cpu_sm, "signs_minus mismatch after H(3)");

        gpu.sx(&[QubitId::new(3)]);
        cpu.sx(&[QubitId::new(3)]);
        gpu.sync();
        gpu.wait();

        let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
        println!("After H(3) then SX(3):");
        println!("  GPU signs_minus: {gpu_sm:?}");
        println!("  CPU signs_minus: {cpu_sm:?}");
        println!("  GPU signs_i:     {gpu_si:?}");
        println!("  CPU signs_i:     {cpu_si:?}");

        assert_eq!(gpu_sm, cpu_sm, "signs_minus mismatch after H(3) then SX(3)");
        assert_eq!(gpu_si, cpu_si, "signs_i mismatch after H(3) then SX(3)");

        println!("SX gate test passed!");
    }

    /// Test gate-by-gate to find where sign divergence occurs
    #[test]
    fn test_gpu_vs_cpu_gate_by_gate_debug() {
        use pecos_random::PecosRng;
        use pecos_simulators::stabilizer_test_utils::{
            CliffordGate, generate_random_clifford_circuit,
        };

        // Helper to read GPU sign state
        fn read_gpu_signs(gpu: &GpuStab) -> (Vec<bool>, Vec<bool>) {
            let gen_words = gpu.gen_words as usize;
            let packed_signs_size = (gen_words * 4) as u64;
            let sign_minus_raw = gpu.read_buffer(&gpu.sign_minus_buffer, packed_signs_size);
            let sign_i_raw = gpu.read_buffer(&gpu.sign_i_buffer, packed_signs_size);

            let mut sign_minus = vec![false; gpu.num_qubits as usize];
            let mut sign_i = vec![false; gpu.num_qubits as usize];
            for g in 0..gpu.num_qubits as usize {
                let word = g / 32;
                let bit = g % 32;
                sign_minus[g] = (sign_minus_raw[word] & (1 << bit)) != 0;
                sign_i[g] = (sign_i_raw[word] & (1 << bit)) != 0;
            }
            (sign_minus, sign_i)
        }

        // Helper to read CPU sign state
        fn read_cpu_signs(cpu: &SparseStab, num_qubits: usize) -> (Vec<bool>, Vec<bool>) {
            let mut sign_minus = vec![false; num_qubits];
            let mut sign_i = vec![false; num_qubits];
            for g in &cpu.stabs().signs_minus {
                if g < num_qubits {
                    sign_minus[g] = true;
                }
            }
            for g in &cpu.stabs().signs_i {
                if g < num_qubits {
                    sign_i[g] = true;
                }
            }
            (sign_minus, sign_i)
        }

        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        let seed = 12358u64;
        let mut rng = PecosRng::seed_from_u64(seed);
        let circuit = generate_random_clifford_circuit(&mut rng, 4, 30);

        gpu.reset();
        cpu.reset();
        gpu.sync();
        gpu.wait();

        println!("Gate-by-gate test with seed {seed}:");
        println!("After reset:");
        let (gpu_sm, _gpu_si) = read_gpu_signs(&gpu);
        let (cpu_sm, _cpu_si) = read_cpu_signs(&cpu, 4);
        println!("  GPU signs_minus: {gpu_sm:?}");
        println!("  CPU signs_minus: {cpu_sm:?}");
        assert_eq!(gpu_sm, cpu_sm, "Signs differ after reset!");

        // Apply gates one by one
        for (i, gate) in circuit.iter().enumerate() {
            // Apply gate to both
            match gate {
                CliffordGate::H(q) => {
                    gpu.h(&[QubitId::new(*q)]);
                    cpu.h(&[QubitId::new(*q)]);
                }
                CliffordGate::S(q) => {
                    gpu.sz(&[QubitId::new(*q)]);
                    cpu.sz(&[QubitId::new(*q)]);
                }
                CliffordGate::Sdg(q) => {
                    gpu.szdg(&[QubitId::new(*q)]);
                    cpu.szdg(&[QubitId::new(*q)]);
                }
                CliffordGate::X(q) => {
                    gpu.x(&[QubitId::new(*q)]);
                    cpu.x(&[QubitId::new(*q)]);
                }
                CliffordGate::Y(q) => {
                    gpu.y(&[QubitId::new(*q)]);
                    cpu.y(&[QubitId::new(*q)]);
                }
                CliffordGate::Z(q) => {
                    gpu.z(&[QubitId::new(*q)]);
                    cpu.z(&[QubitId::new(*q)]);
                }
                CliffordGate::CX(c, t) => {
                    gpu.cx(&[(QubitId::new(*c), QubitId::new(*t))]);
                    cpu.cx(&[(QubitId::new(*c), QubitId::new(*t))]);
                }
                CliffordGate::CZ(a, b) => {
                    gpu.cz(&[(QubitId::new(*a), QubitId::new(*b))]);
                    cpu.cz(&[(QubitId::new(*a), QubitId::new(*b))]);
                }
                CliffordGate::SWAP(a, b) => {
                    gpu.swap(&[(QubitId::new(*a), QubitId::new(*b))]);
                    cpu.swap(&[(QubitId::new(*a), QubitId::new(*b))]);
                }
                CliffordGate::SX(q) => {
                    gpu.sx(&[QubitId::new(*q)]);
                    cpu.sx(&[QubitId::new(*q)]);
                }
                CliffordGate::SXdg(q) => {
                    gpu.sxdg(&[QubitId::new(*q)]);
                    cpu.sxdg(&[QubitId::new(*q)]);
                }
                CliffordGate::SY(q) => {
                    gpu.sy(&[QubitId::new(*q)]);
                    cpu.sy(&[QubitId::new(*q)]);
                }
                CliffordGate::SYdg(q) => {
                    gpu.sydg(&[QubitId::new(*q)]);
                    cpu.sydg(&[QubitId::new(*q)]);
                }
                CliffordGate::CY(c, t) => {
                    gpu.cy(&[(QubitId::new(*c), QubitId::new(*t))]);
                    cpu.cy(&[(QubitId::new(*c), QubitId::new(*t))]);
                }
            }

            // Sync GPU and compare
            gpu.sync();
            gpu.wait();

            let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
            let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);

            if gpu_sm != cpu_sm || gpu_si != cpu_si {
                println!("\nDivergence at gate {i}: {gate:?}");
                println!("  GPU signs_minus: {gpu_sm:?}");
                println!("  CPU signs_minus: {cpu_sm:?}");
                println!("  GPU signs_i:     {gpu_si:?}");
                println!("  CPU signs_i:     {cpu_si:?}");
                println!("\nGates applied (0 to {i}):");
                for (j, g) in circuit.iter().enumerate().take(i + 1) {
                    println!("  {j}: {g:?}");
                }
                panic!("Signs diverged at gate {i}: {gate:?}");
            }
        }

        println!("\nAll {} gates passed!", circuit.len());
    }

    /// Test sequential measurements to find where divergence occurs
    #[test]
    fn test_gpu_vs_cpu_sequential_measurement_debug() {
        use pecos_random::{PecosRng, RngExt};
        use pecos_simulators::stabilizer_test_utils::{
            apply_circuit, generate_random_clifford_circuit,
        };

        // Helper to read GPU sign state
        fn read_gpu_signs(gpu: &GpuStab) -> (Vec<bool>, Vec<bool>) {
            let gen_words = gpu.gen_words as usize;
            let packed_signs_size = (gen_words * 4) as u64;
            let sign_minus_raw = gpu.read_buffer(&gpu.sign_minus_buffer, packed_signs_size);
            let sign_i_raw = gpu.read_buffer(&gpu.sign_i_buffer, packed_signs_size);

            let mut sign_minus = vec![false; gpu.num_qubits as usize];
            let mut sign_i = vec![false; gpu.num_qubits as usize];
            for g in 0..gpu.num_qubits as usize {
                let word = g / 32;
                let bit = g % 32;
                sign_minus[g] = (sign_minus_raw[word] & (1 << bit)) != 0;
                sign_i[g] = (sign_i_raw[word] & (1 << bit)) != 0;
            }
            (sign_minus, sign_i)
        }

        // Helper to read CPU sign state
        fn read_cpu_signs(cpu: &SparseStab, num_qubits: usize) -> (Vec<bool>, Vec<bool>) {
            let mut sign_minus = vec![false; num_qubits];
            let mut sign_i = vec![false; num_qubits];
            for g in &cpu.stabs().signs_minus {
                if g < num_qubits {
                    sign_minus[g] = true;
                }
            }
            for g in &cpu.stabs().signs_i {
                if g < num_qubits {
                    sign_i[g] = true;
                }
            }
            (sign_minus, sign_i)
        }

        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        // Use seed 12354 which was failing
        let seed = 12354u64;
        let mut rng = PecosRng::seed_from_u64(seed);
        let circuit = generate_random_clifford_circuit(&mut rng, 4, 30);

        gpu.reset();
        cpu.reset();
        apply_circuit(&mut gpu, &circuit);
        apply_circuit(&mut cpu, &circuit);
        gpu.sync();
        gpu.wait();

        // Measure qubits one by one and compare
        println!("Sequential measurement test with seed {seed}:");
        let mut meas_rng = PecosRng::seed_from_u64(seed.wrapping_add(1_000_000));
        for q in 0..4 {
            let forced: bool = meas_rng.random();

            // Check determinism before this measurement
            let gpu_det = gpu.find_first_anticommuting(q as u32).is_none();
            let cpu_det = cpu.clone().mz_forced(q, false).is_deterministic;

            println!("\nBefore measuring Q{q}:");
            println!("  GPU det={gpu_det}, CPU det={cpu_det}");

            // Print sign state before measurement
            let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
            let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
            println!("  GPU signs_minus: {gpu_sm:?}");
            println!("  CPU signs_minus: {cpu_sm:?}");
            println!("  GPU signs_i:     {gpu_si:?}");
            println!("  CPU signs_i:     {cpu_si:?}");
            if gpu_sm != cpu_sm {
                println!("  SIGNS_MINUS MISMATCH!");
            }
            if gpu_si != cpu_si {
                println!("  SIGNS_I MISMATCH!");
            }

            if gpu_det != cpu_det {
                println!("  DETERMINISM MISMATCH BEFORE MEASUREMENT!");
            }

            // Actually measure
            let gpu_r = gpu.mz_forced(q, forced);
            let cpu_r = cpu.mz_forced(q, forced);

            println!("After measuring Q{q} (forced={forced}):");
            println!(
                "  GPU: det={}, out={}",
                gpu_r.is_deterministic, gpu_r.outcome
            );
            println!(
                "  CPU: det={}, out={}",
                cpu_r.is_deterministic, cpu_r.outcome
            );

            if gpu_r.is_deterministic != cpu_r.is_deterministic {
                println!("  DETERMINISM MISMATCH!");
                panic!("Determinism mismatch at Q{q}");
            }
            if gpu_r.outcome != cpu_r.outcome {
                println!("  OUTCOME MISMATCH!");
                // Print sign state at point of divergence
                let (gpu_sm, gpu_si) = read_gpu_signs(&gpu);
                let (cpu_sm, cpu_si) = read_cpu_signs(&cpu, 4);
                println!("  After meas - GPU signs_minus: {gpu_sm:?}");
                println!("  After meas - CPU signs_minus: {cpu_sm:?}");
                println!("  After meas - GPU signs_i:     {gpu_si:?}");
                println!("  After meas - CPU signs_i:     {cpu_si:?}");
                panic!("Outcome mismatch at Q{q}");
            }
        }

        println!("\nAll measurements agreed!");
    }

    /// Test Bell state measurement - this should be deterministic after first measurement
    #[test]
    fn test_gpu_vs_cpu_bell_state_measurement() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // Create Bell state: H(0), CX(0,1) -> (|00> + |11>)/sqrt(2)
        gpu.reset();
        cpu.reset();
        gpu.h(&[QubitId::new(0)]);
        gpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.h(&[QubitId::new(0)]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        gpu.sync();
        gpu.wait();

        println!("Bell state measurement test:");

        // Q0 should be non-deterministic
        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        println!(
            "Q0: GPU det={} out={}, CPU det={} out={}",
            gpu_r0.is_deterministic, gpu_r0.outcome, cpu_r0.is_deterministic, cpu_r0.outcome
        );
        assert_eq!(gpu_r0.is_deterministic, cpu_r0.is_deterministic);
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome);

        // Q1 should now be deterministic (same as Q0 outcome)
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r1 = cpu.mz_forced(1, false);
        println!(
            "Q1: GPU det={} out={}, CPU det={} out={}",
            gpu_r1.is_deterministic, gpu_r1.outcome, cpu_r1.is_deterministic, cpu_r1.outcome
        );
        assert_eq!(
            gpu_r1.is_deterministic, cpu_r1.is_deterministic,
            "Q1 determinism mismatch"
        );
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "Q1 outcome mismatch");

        // Q1 outcome should equal Q0 outcome (Bell state correlation)
        assert!(
            !gpu_r1.outcome,
            "Bell state: Q1 should equal Q0 (both false)"
        );
    }

    /// Test GHZ state measurement - demonstrates entanglement correlation
    #[test]
    fn test_gpu_vs_cpu_ghz_state_measurement() {
        let Some(mut gpu) = gpu_sim(3, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(3);

        // Create GHZ state: H(0), CX(0,1), CX(0,2) -> (|000> + |111>)/sqrt(2)
        gpu.reset();
        cpu.reset();
        gpu.h(&[QubitId::new(0)]);
        gpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        gpu.cx(&[(QubitId::new(0), QubitId::new(2))]);
        cpu.h(&[QubitId::new(0)]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(2))]);
        gpu.sync();
        gpu.wait();

        println!("GHZ state measurement test:");

        // Q0 should be non-deterministic
        let forced_outcome = true; // Force Q0 to 1
        let gpu_r0 = gpu.mz_forced(0, forced_outcome);
        let cpu_r0 = cpu.mz_forced(0, forced_outcome);
        println!(
            "Q0: GPU det={} out={}, CPU det={} out={}",
            gpu_r0.is_deterministic, gpu_r0.outcome, cpu_r0.is_deterministic, cpu_r0.outcome
        );
        assert!(!gpu_r0.is_deterministic, "Q0 should be non-deterministic");
        assert_eq!(
            gpu_r0.outcome, forced_outcome,
            "Q0 should have forced outcome"
        );

        // Q1 and Q2 should now be deterministic and equal to Q0
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r1 = cpu.mz_forced(1, false);
        println!(
            "Q1: GPU det={} out={}, CPU det={} out={}",
            gpu_r1.is_deterministic, gpu_r1.outcome, cpu_r1.is_deterministic, cpu_r1.outcome
        );
        assert!(
            gpu_r1.is_deterministic,
            "Q1 should be deterministic after Q0 measured"
        );
        assert_eq!(
            gpu_r1.is_deterministic, cpu_r1.is_deterministic,
            "Q1 determinism mismatch"
        );
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "Q1 outcome mismatch");
        assert_eq!(gpu_r1.outcome, forced_outcome, "GHZ: Q1 should equal Q0");

        let gpu_r2 = gpu.mz_forced(2, false);
        let cpu_r2 = cpu.mz_forced(2, false);
        println!(
            "Q2: GPU det={} out={}, CPU det={} out={}",
            gpu_r2.is_deterministic, gpu_r2.outcome, cpu_r2.is_deterministic, cpu_r2.outcome
        );
        assert!(
            gpu_r2.is_deterministic,
            "Q2 should be deterministic after Q0 measured"
        );
        assert_eq!(
            gpu_r2.is_deterministic, cpu_r2.is_deterministic,
            "Q2 determinism mismatch"
        );
        assert_eq!(gpu_r2.outcome, cpu_r2.outcome, "Q2 outcome mismatch");
        assert_eq!(gpu_r2.outcome, forced_outcome, "GHZ: Q2 should equal Q0");
    }

    fn gpu_sim(num_qubits: usize, seed: u64) -> Option<GpuStab> {
        GpuStab::with_seed(num_qubits, seed).ok()
    }

    // ========================================================================
    // Basic Tests
    // ========================================================================

    #[test]
    fn test_creation() {
        let Some(sim) = gpu_sim(4, 42) else { return };
        assert_eq!(sim.num_qubits, 4);
    }

    #[test]
    fn test_queue_batching() {
        let Some(mut sim) = gpu_sim(100, 42) else {
            return;
        };

        // Queue many gates
        for i in 0..100 {
            sim.h(&[QubitId::new(i)]);
        }

        // All gates should be queued (gate_queue[0] is num_gates placeholder)
        assert_eq!(sim.gate_queue.len() - 1, 100);

        // Flush is now a no-op, gates stay until sync
        sim.flush();
        assert_eq!(sim.gate_queue.len() - 1, 100);

        // Sync executes all gates (only placeholder remains)
        sim.sync();
        assert_eq!(sim.gate_queue.len() - 1, 0);
    }

    // ========================================================================
    // Deterministic Measurement Tests (|0> and |1> states)
    // ========================================================================

    #[test]
    fn test_initial_state_measurement() {
        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        // Initial state should measure as |0> deterministically
        for q in 0..4 {
            let gpu_r = gpu.mz_forced(q, false);
            let cpu_r = cpu.mz_forced(q, false);

            assert!(
                gpu_r.is_deterministic,
                "Initial state should be deterministic"
            );
            assert_eq!(
                gpu_r.is_deterministic, cpu_r.is_deterministic,
                "Determinism should match CPU"
            );
            assert!(!gpu_r.outcome, "Initial state should measure 0");
            assert_eq!(gpu_r.outcome, cpu_r.outcome, "Outcome should match CPU");
        }
    }

    #[test]
    fn test_x_gate_deterministic() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // Apply X to qubit 0 - should flip to |1>
        gpu.x(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "X|0> should be deterministic");
        assert!(gpu_r0.outcome, "X|0> should measure 1");
        assert_eq!(
            gpu_r0.outcome, cpu_r0.outcome,
            "X gate: q0 outcome mismatch"
        );

        assert!(
            gpu_r1.is_deterministic,
            "Unmodified qubit should be deterministic"
        );
        assert!(!gpu_r1.outcome, "Unmodified qubit should measure 0");
        assert_eq!(
            gpu_r1.outcome, cpu_r1.outcome,
            "X gate: q1 outcome mismatch"
        );
    }

    #[test]
    fn test_z_gate_on_computational_basis() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // Z on |0> should have no effect on measurement
        gpu.z(&[QubitId::new(0)]);
        cpu.z(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "Z|0> should be deterministic");
        assert!(!gpu_r.outcome, "Z|0> should still measure 0");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "Z gate: outcome mismatch");

        // Z on |1> should have no effect on Z-basis measurement
        gpu.reset();
        cpu.reset();
        gpu.x(&[QubitId::new(0)]);
        gpu.z(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);
        cpu.z(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "ZX|0> should be deterministic");
        assert!(gpu_r.outcome, "ZX|0> should measure 1");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "ZX gate: outcome mismatch");
    }

    #[test]
    fn test_y_gate_deterministic() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // Y on |0> should flip to |1> (with phase)
        gpu.y(&[QubitId::new(0)]);
        cpu.y(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "Y|0> should be deterministic");
        assert!(gpu_r.outcome, "Y|0> should measure 1");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "Y gate: outcome mismatch");
    }

    // ========================================================================
    // H Gate Tests (Non-Deterministic)
    // ========================================================================

    #[test]
    fn test_h_gate_non_deterministic() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // H on |0> creates superposition - non-deterministic
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(!gpu_r.is_deterministic, "H|0> should be non-deterministic");
        assert_eq!(
            gpu_r.is_deterministic, cpu_r.is_deterministic,
            "H gate: determinism mismatch"
        );
    }

    #[test]
    fn test_h_h_identity() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // H H = I, should return to |0>
        gpu.h(&[QubitId::new(0)]);
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "HH|0> should be deterministic");
        assert!(!gpu_r.outcome, "HH|0> should measure 0");
        assert_eq!(
            gpu_r.outcome, cpu_r.outcome,
            "HH identity: outcome mismatch"
        );
    }

    // ========================================================================
    // S Gate Tests
    // ========================================================================

    #[test]
    fn test_s_gate_gpu_vs_cpu() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // S on |0> has no effect on Z-measurement
        gpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "S|0> should be deterministic");
        assert!(!gpu_r.outcome, "S|0> should measure 0");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "S gate: outcome mismatch");

        // Test S on |+>
        gpu.reset();
        cpu.reset();
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        gpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert_eq!(
            gpu_r.is_deterministic, cpu_r.is_deterministic,
            "S gate: determinism mismatch after H S"
        );
    }

    #[test]
    fn test_s_s_s_s_identity() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // S^4 = I
        gpu.sz(&[QubitId::new(0)]);
        gpu.sz(&[QubitId::new(0)]);
        gpu.sz(&[QubitId::new(0)]);
        gpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "S^4|0> should be deterministic");
        assert!(!gpu_r.outcome, "S^4|0> should measure 0");
        assert_eq!(
            gpu_r.outcome, cpu_r.outcome,
            "S^4 identity: outcome mismatch"
        );
    }

    // ========================================================================
    // Sdg Gate Tests
    // ========================================================================

    #[test]
    fn test_sdg_gate_gpu_vs_cpu() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // Sdg on |0> has no effect on Z-measurement
        gpu.szdg(&[QubitId::new(0)]);
        cpu.szdg(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "Sdg|0> should be deterministic");
        assert!(!gpu_r.outcome, "Sdg|0> should measure 0");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "Sdg gate: outcome mismatch");
    }

    #[test]
    fn test_s_sdg_identity() {
        let Some(mut gpu) = gpu_sim(1, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(1);

        // S Sdg = I
        gpu.h(&[QubitId::new(0)]); // Create superposition first
        gpu.sz(&[QubitId::new(0)]);
        gpu.szdg(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);
        cpu.szdg(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert_eq!(
            gpu_r.is_deterministic, cpu_r.is_deterministic,
            "S Sdg: determinism mismatch"
        );
    }

    // ========================================================================
    // CX Gate Tests
    // ========================================================================

    #[test]
    fn test_cx_deterministic() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // CX with control in |0> - target unchanged
        gpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(1))]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(
            gpu_r0.is_deterministic,
            "CX: control should be deterministic"
        );
        assert!(
            gpu_r1.is_deterministic,
            "CX: target should be deterministic"
        );
        assert!(!gpu_r0.outcome, "CX: control should measure 0");
        assert!(!gpu_r1.outcome, "CX: target should measure 0");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "CX: control mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "CX: target mismatch");

        // CX with control in |1> - target flips
        gpu.reset();
        cpu.reset();
        gpu.x(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);
        gpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(1))]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(
            gpu_r0.is_deterministic,
            "CX |1>: control should be deterministic"
        );
        assert!(
            gpu_r1.is_deterministic,
            "CX |1>: target should be deterministic"
        );
        assert!(gpu_r0.outcome, "CX |1>: control should measure 1");
        assert!(gpu_r1.outcome, "CX |1>: target should measure 1");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "CX |1>: control mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "CX |1>: target mismatch");
    }

    #[test]
    fn test_cx_entanglement() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // H CX creates Bell state - both measurements non-deterministic but correlated
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        gpu.cx(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cx(&[(QubitId::new(0), QubitId::new(1))]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);

        // First measurement should be non-deterministic
        assert!(
            !gpu_r0.is_deterministic,
            "Bell state: first meas non-deterministic"
        );
        assert_eq!(
            gpu_r0.is_deterministic, cpu_r0.is_deterministic,
            "Bell state: determinism mismatch"
        );
    }

    // ========================================================================
    // CZ Gate Tests
    // ========================================================================

    #[test]
    fn test_cz_deterministic() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // CZ on computational basis - no effect on Z measurement
        gpu.cz(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cz(&[(QubitId::new(0), QubitId::new(1))]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "CZ: q0 should be deterministic");
        assert!(gpu_r1.is_deterministic, "CZ: q1 should be deterministic");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "CZ: q0 mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "CZ: q1 mismatch");
    }

    #[test]
    fn test_cz_on_superposition() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // Put both qubits in superposition, then CZ
        gpu.h(&[QubitId::new(0)]);
        gpu.h(&[QubitId::new(1)]);
        cpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(1)]);
        gpu.cz(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.cz(&[(QubitId::new(0), QubitId::new(1))]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);

        assert_eq!(
            gpu_r0.is_deterministic, cpu_r0.is_deterministic,
            "CZ superposition: determinism mismatch"
        );
    }

    // ========================================================================
    // SWAP Gate Tests
    // ========================================================================

    #[test]
    fn test_swap_gate() {
        let Some(mut gpu) = gpu_sim(2, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(2);

        // Set q0 to |1>, q1 to |0>, then swap
        gpu.x(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);
        gpu.swap(&[(QubitId::new(0), QubitId::new(1))]);
        cpu.swap(&[(QubitId::new(0), QubitId::new(1))]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "SWAP: q0 should be deterministic");
        assert!(gpu_r1.is_deterministic, "SWAP: q1 should be deterministic");
        assert!(!gpu_r0.outcome, "SWAP: q0 should now be 0");
        assert!(gpu_r1.outcome, "SWAP: q1 should now be 1");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "SWAP: q0 mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "SWAP: q1 mismatch");
    }

    // ========================================================================
    // Multi-Qubit Tests
    // ========================================================================

    #[test]
    fn test_multi_qubit_circuit() {
        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        // Apply X to all qubits
        for i in 0..4 {
            gpu.x(&[QubitId::new(i)]);
            cpu.x(&[QubitId::new(i)]);
        }

        // Verify all measure 1
        for i in 0..4 {
            let gpu_r = gpu.mz_forced(i, false);
            let cpu_r = cpu.mz_forced(i, false);

            assert!(
                gpu_r.is_deterministic,
                "Multi X: q{i} should be deterministic"
            );
            assert!(gpu_r.outcome, "Multi X: q{i} should measure 1");
            assert_eq!(gpu_r.outcome, cpu_r.outcome, "Multi X: q{i} mismatch");
        }
    }

    #[test]
    fn test_batched_gates() {
        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(4);

        // Apply H to all, then S to all, then H to all
        // This should be equivalent to Sdg (up to phase)
        for i in 0..4 {
            gpu.h(&[QubitId::new(i)]);
            cpu.h(&[QubitId::new(i)]);
        }
        for i in 0..4 {
            gpu.sz(&[QubitId::new(i)]);
            cpu.sz(&[QubitId::new(i)]);
        }
        for i in 0..4 {
            gpu.h(&[QubitId::new(i)]);
            cpu.h(&[QubitId::new(i)]);
        }

        for i in 0..4 {
            let gpu_r = gpu.mz_forced(i, false);
            let cpu_r = cpu.mz_forced(i, false);

            assert_eq!(
                gpu_r.is_deterministic, cpu_r.is_deterministic,
                "Batched HSH: q{i} determinism mismatch"
            );
            assert_eq!(
                gpu_r.outcome, cpu_r.outcome,
                "Batched HSH: q{i} outcome mismatch"
            );
        }
    }

    // ========================================================================
    // Large System Tests
    // ========================================================================

    #[test]
    fn test_larger_system() {
        let Some(mut gpu) = gpu_sim(50, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(50);

        // Apply alternating X and Z gates
        for i in 0..50 {
            if i % 2 == 0 {
                gpu.x(&[QubitId::new(i)]);
                cpu.x(&[QubitId::new(i)]);
            } else {
                gpu.z(&[QubitId::new(i)]);
                cpu.z(&[QubitId::new(i)]);
            }
        }

        // Verify measurements
        for i in 0..50 {
            let gpu_r = gpu.mz_forced(i, false);
            let cpu_r = cpu.mz_forced(i, false);

            let expected = i % 2 == 0; // X flips, Z doesn't
            assert!(
                gpu_r.is_deterministic,
                "Large: q{i} should be deterministic"
            );
            assert_eq!(gpu_r.outcome, expected, "Large: q{i} wrong outcome");
            assert_eq!(gpu_r.outcome, cpu_r.outcome, "Large: q{i} mismatch");
        }
    }

    #[test]
    fn test_random_circuit() {
        let Some(mut gpu) = gpu_sim(10, 42) else {
            return;
        };
        let mut cpu = SparseStab::new(10);

        // Apply a deterministic sequence of gates
        let gates = [
            (0u8, 0usize),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (2, 3),
            (2, 4),
        ];

        for &(gate_type, qubit) in &gates {
            match gate_type {
                0 => {
                    gpu.x(&[QubitId::new(qubit)]);
                    cpu.x(&[QubitId::new(qubit)]);
                }
                1 => {
                    gpu.z(&[QubitId::new(qubit)]);
                    cpu.z(&[QubitId::new(qubit)]);
                }
                2 => {
                    gpu.y(&[QubitId::new(qubit)]);
                    cpu.y(&[QubitId::new(qubit)]);
                }
                _ => {}
            }
        }

        // Verify all measurements match
        for i in 0..10 {
            let gpu_r = gpu.mz_forced(i, false);
            let cpu_r = cpu.mz_forced(i, false);

            assert_eq!(
                gpu_r.is_deterministic, cpu_r.is_deterministic,
                "Random circuit: q{i} determinism mismatch"
            );
            assert_eq!(
                gpu_r.outcome, cpu_r.outcome,
                "Random circuit: q{i} outcome mismatch"
            );
        }
    }

    /// Test that parallel processing produces same results as sequential.
    #[test]
    fn test_parallel_processing() {
        let num_qubits = 20;
        let seed = 12345;

        // Run same circuit with sequential and parallel processing
        let mut gpu_seq = GpuStab::<PecosRng>::with_seed(num_qubits, seed).unwrap();
        let mut gpu_par = GpuStab::<PecosRng>::with_seed(num_qubits, seed).unwrap();

        gpu_par.enable_parallel();

        // Apply H to all qubits (all independent, should batch together)
        for q in 0..num_qubits {
            gpu_seq.h(&[QubitId(q)]);
            gpu_par.h(&[QubitId(q)]);
        }

        // Apply some CX gates
        for q in (0..num_qubits - 1).step_by(2) {
            gpu_seq.cx(&[(QubitId(q), QubitId(q + 1))]);
            gpu_par.cx(&[(QubitId(q), QubitId(q + 1))]);
        }

        // Apply S gates
        for q in 0..num_qubits {
            gpu_seq.sz(&[QubitId(q)]);
            gpu_par.sz(&[QubitId(q)]);
        }

        gpu_seq.sync_wait();
        gpu_par.sync_wait();

        // Measure all qubits and compare - sequential and parallel should match
        for q in 0..num_qubits {
            let seq_result = gpu_seq.mz(&[QubitId(q)]);
            let par_result = gpu_par.mz(&[QubitId(q)]);

            assert_eq!(
                seq_result[0].outcome, par_result[0].outcome,
                "Qubit {q}: sequential={}, parallel={}",
                seq_result[0].outcome, par_result[0].outcome
            );
        }
    }

    #[test]
    fn test_deferred_measurement_bell_state() {
        // Test that deferred measurements correctly handle non-deterministic cases
        // by properly updating the tableau
        let num_trials = 50;
        let mut correlated_count = 0;

        for seed in 0..num_trials {
            let Some(mut gpu) = gpu_sim(2, seed) else {
                return;
            };

            // Create Bell state: |00> + |11>
            gpu.h(&[QubitId(0)]);
            gpu.cx(&[(QubitId(0), QubitId(1))]);

            // Use deferred measurements
            gpu.mz_queue(&[QubitId(0), QubitId(1)]);
            let results = gpu.mz_fetch();

            // Both qubits should have the same outcome (perfectly correlated)
            if results[0].outcome == results[1].outcome {
                correlated_count += 1;
            }
        }

        // All trials should be perfectly correlated
        assert_eq!(
            correlated_count,
            num_trials,
            "Deferred Bell state measurements should be 100% correlated, got {}%",
            correlated_count * 100 / num_trials
        );
    }

    #[test]
    fn test_deferred_measurement_multiple_calls() {
        // Test multiple mz_queue calls followed by a single fetch
        let Some(mut gpu) = gpu_sim(4, 42) else {
            return;
        };

        // Prepare states: qubit 0 in |0>, qubit 1 in |1>, qubit 2,3 in Bell state
        gpu.x(&[QubitId(1)]);
        gpu.h(&[QubitId(2)]);
        gpu.cx(&[(QubitId(2), QubitId(3))]);

        // Queue measurements in separate calls
        gpu.mz_queue(&[QubitId(0)]); // Should be 0
        gpu.mz_queue(&[QubitId(1)]); // Should be 1
        gpu.mz_queue(&[QubitId(2), QubitId(3)]); // Should be correlated

        // Flush all at once
        let results = gpu.mz_fetch();

        assert_eq!(results.len(), 4);
        assert!(!results[0].outcome, "Qubit 0 should measure 0");
        assert!(results[1].outcome, "Qubit 1 should measure 1");
        assert_eq!(
            results[2].outcome, results[3].outcome,
            "Qubits 2,3 (Bell state) should be correlated"
        );
    }
}
