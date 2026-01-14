//! GPU Stabilizer Simulator
//!
//! This implementation uses a persistent kernel approach that queues gates and
//! processes them in a single GPU dispatch, minimizing dispatch overhead.

use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, MeasurementResult, QuantumSimulator};
use rand::{RngCore, SeedableRng};
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

// Maximum gates per batch (buffer size / 4 bytes per gate)
const MAX_GATE_QUEUE_SIZE: usize = 65536;

/// Pack a single-qubit gate into the queue format
fn pack_single_gate(gate_type: u32, target: u32) -> u32 {
    (gate_type & 0xF) | ((target & 0x3FFF) << 4)
}

/// Pack a two-qubit gate into the queue format
fn pack_two_qubit_gate(gate_type: u32, control: u32, target: u32) -> u32 {
    (gate_type & 0xF) | ((target & 0x3FFF) << 4) | ((control & 0x3FFF) << 18)
}

// Number of gate queue buffers for deferred submission
const NUM_QUEUE_BUFFERS: usize = 8;

/// GPU Stabilizer simulator using persistent kernel approach.
///
/// Gates are queued and executed in batches to minimize dispatch overhead.
/// Uses deferred submission with multiple buffers to batch queue.submit() calls.
pub struct GpuStab<R: RngCore + SeedableRng = rand::rngs::StdRng> {
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

    // Pool of gate queue buffers for deferred submission
    gate_queue_buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    current_buffer_idx: usize,

    // Pipeline
    process_queue_pipeline: wgpu::ComputePipeline,

    // Gate queue (CPU side)
    gate_queue: Vec<u32>,

    // Deferred submission state
    pending_command_buffers: Vec<wgpu::CommandBuffer>,

    // For measurement
    anticommuting_buffer: wgpu::Buffer,
    find_anticommuting_bind_group: wgpu::BindGroup,
    find_anticommuting_pipeline: wgpu::ComputePipeline,
}

impl GpuStab<rand::rngs::StdRng> {
    /// Create a new GPU stabilizer simulator with the given number of qubits.
    pub fn new(num_qubits: usize) -> Result<Self, String> {
        Self::with_seed(num_qubits, rand::random())
    }
}

impl<R: RngCore + SeedableRng + Debug> GpuStab<R> {
    /// Create a new GPU stabilizer simulator with a specific RNG seed.
    pub fn with_seed(num_qubits: usize, seed: u64) -> Result<Self, String> {
        let rng = R::seed_from_u64(seed);

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| "No GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("GpuStab Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        let num_qubits = num_qubits as u32;
        let gen_words = num_qubits.div_ceil(32);

        // Buffer sizes
        let tableau_size = (num_qubits as u64) * (gen_words as u64) * 4;
        let packed_signs_size = (gen_words as u64) * 4; // Packed: one bit per generator
        let params_size = 32u64; // 8 u32s
        let gate_queue_size = (MAX_GATE_QUEUE_SIZE * 4) as u64;

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

        // Pool of gate queue buffers for deferred submission
        let mut gate_queue_buffers = Vec::with_capacity(NUM_QUEUE_BUFFERS);
        for i in 0..NUM_QUEUE_BUFFERS {
            gate_queue_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Gate Queue Buffer {}", i)),
                size: gate_queue_size + 4, // +4 for num_gates header
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // One u32 per generator for anticommuting flags (not packed)
        let anticommuting_size = (num_qubits as u64) * 4;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: tableau_size.max(anticommuting_size),
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

        // Create bind groups - one per gate queue buffer for deferred submission
        let mut bind_groups = Vec::with_capacity(NUM_QUEUE_BUFFERS);
        for (i, gate_queue_buffer) in gate_queue_buffers.iter().enumerate() {
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Main Bind Group {}", i)),
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
            }));
        }

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
                compilation_options: Default::default(),
                cache: None,
            });

        // For find_anticommuting, we need a different layout that uses the regular shader's params
        // We'll create a simplified version that reuses the main bind group
        let find_anticommuting_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Find Anticommuting Pipeline Layout"),
                bind_group_layouts: &[&main_bind_group_layout, &find_anticommuting_bind_group_layout],
                immediate_size: 0,
            });

        let find_anticommuting_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Find Anticommuting Pipeline"),
                layout: Some(&find_anticommuting_pipeline_layout),
                module: &regular_shader,
                entry_point: Some("find_anticommuting"),
                compilation_options: Default::default(),
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
            gate_queue_buffers,
            bind_groups,
            current_buffer_idx: 0,
            process_queue_pipeline,
            gate_queue: Vec::with_capacity(1024),
            pending_command_buffers: Vec::with_capacity(NUM_QUEUE_BUFFERS),
            anticommuting_buffer,
            find_anticommuting_bind_group,
            find_anticommuting_pipeline,
        };

        // Initialize to |0...0> state
        sim.initialize_state();

        Ok(sim)
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
        self.queue
            .write_buffer(&self.stab_x_buffer, 0, &vec![0u8; num_qubits * gen_words * 4]);
        self.queue
            .write_buffer(&self.stab_z_buffer, 0, bytemuck::cast_slice(&stab_z));
        self.queue
            .write_buffer(&self.destab_x_buffer, 0, bytemuck::cast_slice(&destab_x));
        self.queue
            .write_buffer(&self.destab_z_buffer, 0, &vec![0u8; num_qubits * gen_words * 4]);
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
        self.gate_queue.push(pack_single_gate(gate_type, qubit));
    }

    /// Queue a two-qubit gate
    fn queue_two_qubit_gate(&mut self, gate_type: u32, control: u32, target: u32) {
        self.gate_queue
            .push(pack_two_qubit_gate(gate_type, control, target));
    }

    /// Submit all pending command buffers to the GPU.
    /// Called automatically before measurement or when buffer pool is exhausted.
    pub fn sync(&mut self) {
        if self.pending_command_buffers.is_empty() {
            return;
        }

        // Submit all pending command buffers in one call
        self.queue
            .submit(self.pending_command_buffers.drain(..));

        // Reset buffer index
        self.current_buffer_idx = 0;
    }

    /// Flush the gate queue - record command buffer for deferred execution.
    /// Command buffers are batched and submitted together when sync() is called
    /// or when the buffer pool is exhausted.
    pub fn flush(&mut self) {
        if self.gate_queue.is_empty() {
            return;
        }

        // If buffer pool is exhausted, sync first
        if self.current_buffer_idx >= NUM_QUEUE_BUFFERS {
            self.sync();
        }

        let buffer_idx = self.current_buffer_idx;
        let num_gates = self.gate_queue.len() as u32;

        // Write gate queue with num_gates as header to current buffer
        let current_buffer = &self.gate_queue_buffers[buffer_idx];
        self.queue
            .write_buffer(current_buffer, 0, bytemuck::bytes_of(&num_gates));
        self.queue.write_buffer(
            current_buffer,
            4,
            bytemuck::cast_slice(&self.gate_queue),
        );

        // Record command buffer (don't submit yet)
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: None,
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.process_queue_pipeline);
            pass.set_bind_group(0, &self.bind_groups[buffer_idx], &[]);
            pass.dispatch_workgroups(self.gen_words.div_ceil(256), 1, 1);
        }

        // Add to pending list instead of submitting
        self.pending_command_buffers.push(encoder.finish());
        self.current_buffer_idx += 1;

        // Clear the gate queue
        self.gate_queue.clear();
    }

    /// Find first anticommuting stabilizer (for measurement)
    fn find_first_anticommuting(&mut self, qubit: u32) -> Option<usize> {
        // Flush and sync to ensure all pending gates are executed
        self.flush();
        self.sync();

        // Update params for find_anticommuting
        let params = [
            self.num_qubits,
            self.gen_words,
            self.num_qubits, // num_gens
            qubit,          // target_qubit
            0,              // control_qubit (unused)
            0,
            0,
            0,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

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
            pass.set_bind_group(0, &self.bind_groups[0], &[]);
            pass.set_bind_group(1, &self.find_anticommuting_bind_group, &[]);
            pass.dispatch_workgroups(self.num_qubits.div_ceil(256), 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.anticommuting_buffer,
            0,
            &self.staging_buffer,
            0,
            (self.num_qubits as u64) * 4,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read only the relevant portion
        let read_size = (self.num_qubits as u64) * 4;
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

                for q2 in 0..num_qubits {
                    if cumulative_x[q2]
                        && Self::get_bit_transposed(&stab_z, q2, gen_idx, gen_words)
                    {
                        num_minuses += 1;
                    }
                }

                for q2 in 0..num_qubits {
                    if Self::get_bit_transposed(&stab_x, q2, gen_idx, gen_words) {
                        cumulative_x[q2] = !cumulative_x[q2];
                    }
                }
            }
        }

        if num_is & 3 != 0 {
            num_minuses += 1;
        }

        num_minuses % 2 != 0
    }
}

impl<R: RngCore + SeedableRng + Debug> CliffordGateable for GpuStab<R> {
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

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "CX requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            self.queue_two_qubit_gate(GATE_CX, pair[0].index() as u32, pair[1].index() as u32);
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "CZ requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            self.queue_two_qubit_gate(GATE_CZ, pair[0].index() as u32, pair[1].index() as u32);
        }
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(qubits.len() % 2 == 0, "SWAP requires pairs of qubits");
        for pair in qubits.chunks_exact(2) {
            self.queue_two_qubit_gate(GATE_SWAP, pair[0].index() as u32, pair[1].index() as u32);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        // Flush before measurement
        self.flush();

        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qubit = q.index() as usize;
            let first_anticommuting = self.find_first_anticommuting(qubit as u32);

            if first_anticommuting.is_some() {
                // Non-deterministic - for now just return random
                // Full implementation would need the measurement update shaders
                let outcome = self.rng.next_u32() & 1 != 0;
                results.push(MeasurementResult {
                    outcome,
                    is_deterministic: false,
                });
            } else {
                let outcome = self.compute_deterministic_outcome(qubit);
                results.push(MeasurementResult {
                    outcome,
                    is_deterministic: true,
                });
            }
        }

        results
    }
}

impl<R: RngCore + SeedableRng + Debug> GpuStab<R> {
    /// Measure qubit in Z basis with forced outcome for non-deterministic cases.
    ///
    /// If the measurement is deterministic, returns the determined outcome.
    /// If non-deterministic, forces the measurement to the specified outcome.
    ///
    /// Note: This implementation does not update the tableau for non-deterministic
    /// measurements (used for testing deterministic cases only).
    pub fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        self.flush();

        let first_anticommuting = self.find_first_anticommuting(qubit as u32);

        if first_anticommuting.is_some() {
            // Non-deterministic - return forced outcome
            // Note: Tableau update not implemented for persistent kernel
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

impl<R: RngCore + SeedableRng + Debug> QuantumSimulator for GpuStab<R> {
    fn reset(&mut self) -> &mut Self {
        self.gate_queue.clear();
        self.pending_command_buffers.clear();
        self.current_buffer_idx = 0;
        self.initialize_state();
        self
    }
}

impl<R: RngCore + SeedableRng + Debug> Debug for GpuStab<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStab")
            .field("num_qubits", &self.num_qubits)
            .field("queued_gates", &self.gate_queue.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_qsim::stabilizer_test_utils::ForcedMeasurement;
    use pecos_qsim::SparseStab;

    impl<R: RngCore + SeedableRng + Debug> ForcedMeasurement for GpuStab<R> {
        fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
            GpuStab::mz_forced(self, qubit, forced_outcome)
        }
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
        let Some(mut sim) = gpu_sim(100, 42) else { return };

        // Queue many gates
        for i in 0..100 {
            sim.h(&[QubitId::new(i)]);
        }

        // All gates should be queued
        assert_eq!(sim.gate_queue.len(), 100);

        // Flush executes them all
        sim.flush();
        assert_eq!(sim.gate_queue.len(), 0);
    }

    // ========================================================================
    // Deterministic Measurement Tests (|0> and |1> states)
    // ========================================================================

    #[test]
    fn test_initial_state_measurement() {
        let Some(mut gpu) = gpu_sim(4, 42) else { return };
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
            assert_eq!(gpu_r.outcome, false, "Initial state should measure 0");
            assert_eq!(gpu_r.outcome, cpu_r.outcome, "Outcome should match CPU");
        }
    }

    #[test]
    fn test_x_gate_deterministic() {
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // Apply X to qubit 0 - should flip to |1>
        gpu.x(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "X|0> should be deterministic");
        assert_eq!(gpu_r0.outcome, true, "X|0> should measure 1");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "X gate: q0 outcome mismatch");

        assert!(gpu_r1.is_deterministic, "Unmodified qubit should be deterministic");
        assert_eq!(gpu_r1.outcome, false, "Unmodified qubit should measure 0");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "X gate: q1 outcome mismatch");
    }

    #[test]
    fn test_z_gate_on_computational_basis() {
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // Z on |0> should have no effect on measurement
        gpu.z(&[QubitId::new(0)]);
        cpu.z(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "Z|0> should be deterministic");
        assert_eq!(gpu_r.outcome, false, "Z|0> should still measure 0");
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
        assert_eq!(gpu_r.outcome, true, "ZX|0> should measure 1");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "ZX gate: outcome mismatch");
    }

    #[test]
    fn test_y_gate_deterministic() {
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
        let mut cpu = SparseStab::new(1);

        // Y on |0> should flip to |1> (with phase)
        gpu.y(&[QubitId::new(0)]);
        cpu.y(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "Y|0> should be deterministic");
        assert_eq!(gpu_r.outcome, true, "Y|0> should measure 1");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "Y gate: outcome mismatch");
    }

    // ========================================================================
    // H Gate Tests (Non-Deterministic)
    // ========================================================================

    #[test]
    fn test_h_gate_non_deterministic() {
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
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
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
        let mut cpu = SparseStab::new(1);

        // H H = I, should return to |0>
        gpu.h(&[QubitId::new(0)]);
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "HH|0> should be deterministic");
        assert_eq!(gpu_r.outcome, false, "HH|0> should measure 0");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "HH identity: outcome mismatch");
    }

    // ========================================================================
    // S Gate Tests
    // ========================================================================

    #[test]
    fn test_s_gate_gpu_vs_cpu() {
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
        let mut cpu = SparseStab::new(1);

        // S on |0> has no effect on Z-measurement
        gpu.sz(&[QubitId::new(0)]);
        cpu.sz(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "S|0> should be deterministic");
        assert_eq!(gpu_r.outcome, false, "S|0> should measure 0");
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
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
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
        assert_eq!(gpu_r.outcome, false, "S^4|0> should measure 0");
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
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
        let mut cpu = SparseStab::new(1);

        // Sdg on |0> has no effect on Z-measurement
        gpu.szdg(&[QubitId::new(0)]);
        cpu.szdg(&[QubitId::new(0)]);

        let gpu_r = gpu.mz_forced(0, false);
        let cpu_r = cpu.mz_forced(0, false);

        assert!(gpu_r.is_deterministic, "Sdg|0> should be deterministic");
        assert_eq!(gpu_r.outcome, false, "Sdg|0> should measure 0");
        assert_eq!(gpu_r.outcome, cpu_r.outcome, "Sdg gate: outcome mismatch");
    }

    #[test]
    fn test_s_sdg_identity() {
        let Some(mut gpu) = gpu_sim(1, 42) else { return };
        let mut cpu = SparseStab::new(1);

        // S Sdg = I
        gpu.h(&[QubitId::new(0)]);  // Create superposition first
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
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // CX with control in |0> - target unchanged
        gpu.cx(&[QubitId::new(0), QubitId::new(1)]);
        cpu.cx(&[QubitId::new(0), QubitId::new(1)]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "CX: control should be deterministic");
        assert!(gpu_r1.is_deterministic, "CX: target should be deterministic");
        assert_eq!(gpu_r0.outcome, false, "CX: control should measure 0");
        assert_eq!(gpu_r1.outcome, false, "CX: target should measure 0");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "CX: control mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "CX: target mismatch");

        // CX with control in |1> - target flips
        gpu.reset();
        cpu.reset();
        gpu.x(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);
        gpu.cx(&[QubitId::new(0), QubitId::new(1)]);
        cpu.cx(&[QubitId::new(0), QubitId::new(1)]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "CX |1>: control should be deterministic");
        assert!(gpu_r1.is_deterministic, "CX |1>: target should be deterministic");
        assert_eq!(gpu_r0.outcome, true, "CX |1>: control should measure 1");
        assert_eq!(gpu_r1.outcome, true, "CX |1>: target should measure 1");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "CX |1>: control mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "CX |1>: target mismatch");
    }

    #[test]
    fn test_cx_entanglement() {
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // H CX creates Bell state - both measurements non-deterministic but correlated
        gpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(0)]);
        gpu.cx(&[QubitId::new(0), QubitId::new(1)]);
        cpu.cx(&[QubitId::new(0), QubitId::new(1)]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let cpu_r0 = cpu.mz_forced(0, false);

        // First measurement should be non-deterministic
        assert!(!gpu_r0.is_deterministic, "Bell state: first meas non-deterministic");
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
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // CZ on computational basis - no effect on Z measurement
        gpu.cz(&[QubitId::new(0), QubitId::new(1)]);
        cpu.cz(&[QubitId::new(0), QubitId::new(1)]);

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
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // Put both qubits in superposition, then CZ
        gpu.h(&[QubitId::new(0)]);
        gpu.h(&[QubitId::new(1)]);
        cpu.h(&[QubitId::new(0)]);
        cpu.h(&[QubitId::new(1)]);
        gpu.cz(&[QubitId::new(0), QubitId::new(1)]);
        cpu.cz(&[QubitId::new(0), QubitId::new(1)]);

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
        let Some(mut gpu) = gpu_sim(2, 42) else { return };
        let mut cpu = SparseStab::new(2);

        // Set q0 to |1>, q1 to |0>, then swap
        gpu.x(&[QubitId::new(0)]);
        cpu.x(&[QubitId::new(0)]);
        gpu.swap(&[QubitId::new(0), QubitId::new(1)]);
        cpu.swap(&[QubitId::new(0), QubitId::new(1)]);

        let gpu_r0 = gpu.mz_forced(0, false);
        let gpu_r1 = gpu.mz_forced(1, false);
        let cpu_r0 = cpu.mz_forced(0, false);
        let cpu_r1 = cpu.mz_forced(1, false);

        assert!(gpu_r0.is_deterministic, "SWAP: q0 should be deterministic");
        assert!(gpu_r1.is_deterministic, "SWAP: q1 should be deterministic");
        assert_eq!(gpu_r0.outcome, false, "SWAP: q0 should now be 0");
        assert_eq!(gpu_r1.outcome, true, "SWAP: q1 should now be 1");
        assert_eq!(gpu_r0.outcome, cpu_r0.outcome, "SWAP: q0 mismatch");
        assert_eq!(gpu_r1.outcome, cpu_r1.outcome, "SWAP: q1 mismatch");
    }

    // ========================================================================
    // Multi-Qubit Tests
    // ========================================================================

    #[test]
    fn test_multi_qubit_circuit() {
        let Some(mut gpu) = gpu_sim(4, 42) else { return };
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

            assert!(gpu_r.is_deterministic, "Multi X: q{} should be deterministic", i);
            assert_eq!(gpu_r.outcome, true, "Multi X: q{} should measure 1", i);
            assert_eq!(gpu_r.outcome, cpu_r.outcome, "Multi X: q{} mismatch", i);
        }
    }

    #[test]
    fn test_batched_gates() {
        let Some(mut gpu) = gpu_sim(4, 42) else { return };
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
                "Batched HSH: q{} determinism mismatch",
                i
            );
            assert_eq!(
                gpu_r.outcome, cpu_r.outcome,
                "Batched HSH: q{} outcome mismatch",
                i
            );
        }
    }

    // ========================================================================
    // Large System Tests
    // ========================================================================

    #[test]
    fn test_larger_system() {
        let Some(mut gpu) = gpu_sim(50, 42) else { return };
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
            assert!(gpu_r.is_deterministic, "Large: q{} should be deterministic", i);
            assert_eq!(gpu_r.outcome, expected, "Large: q{} wrong outcome", i);
            assert_eq!(gpu_r.outcome, cpu_r.outcome, "Large: q{} mismatch", i);
        }
    }

    #[test]
    fn test_random_circuit() {
        let Some(mut gpu) = gpu_sim(10, 42) else { return };
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
                "Random circuit: q{} determinism mismatch",
                i
            );
            assert_eq!(
                gpu_r.outcome, cpu_r.outcome,
                "Random circuit: q{} outcome mismatch",
                i
            );
        }
    }
}
