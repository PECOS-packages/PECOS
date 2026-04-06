//! GPU-accelerated Pauli Propagator
//!
//! Efficiently tracks how Pauli faults propagate through Clifford circuits
//! across many shots in parallel. This is lighter weight than full stabilizer
//! simulation - it only tracks X and Z fault bits, not the full tableau.
//!
//! # Design
//!
//! Based on cuQuantum's Pauli frame model and PECOS's CPU `PauliProp`:
//! - Dense bit arrays for X and Z faults (one bit per qubit per shot)
//! - All shots processed in parallel on GPU
//! - Gate propagation via bit operations
//! - Fault injection for noise simulation
//!
//! # Use Cases
//!
//! - Fast noisy Clifford circuit sampling
//! - Fault tolerance analysis (tracking fault propagation)
//! - Monte Carlo error threshold estimation
//!
//! # Example
//!
//! ```
//! use pecos_gpu_sims::GpuPauliProp;
//!
//! let mut prop = GpuPauliProp::new(100, 1024).unwrap();  // 100 qubits, 1024 shots
//!
//! // Inject X fault on qubit 0 for all shots
//! prop.inject_x_fault(0);
//!
//! // Propagate through circuit
//! prop.h(&[0]);      // X -> Z
//! prop.cx(&[(0, 1)]);  // Z propagates to control
//!
//! prop.flush();
//!
//! // Check if faults would flip Z measurement on qubit 0
//! let flips = prop.measure_z_flips(&[0, 1]);
//! ```

use pecos_random::{PecosRng, time_seed};
use wgpu::util::DeviceExt;

// Gate type constants (matching shader)
const GATE_H: u32 = 1;
const GATE_SZ: u32 = 2;
const GATE_SZDG: u32 = 3;
const GATE_X: u32 = 4;
const GATE_Y: u32 = 5;
const GATE_Z: u32 = 6;
const GATE_CX: u32 = 7;
const GATE_CZ: u32 = 8;
const GATE_SWAP: u32 = 9;

// Fault injection constants
const FAULT_X: u32 = 16;
const FAULT_Z: u32 = 17;
const FAULT_Y: u32 = 18;
const FAULT_DEPOL1: u32 = 19; // Single-qubit depolarizing
const FAULT_DEPOL2: u32 = 20; // Two-qubit depolarizing

// Max gates per batch before auto-flush
const MAX_GATE_QUEUE_SIZE: usize = 4096;

/// GPU-accelerated Pauli fault propagator.
///
/// Tracks X and Z faults across many shots in parallel, propagating them
/// through Clifford gates. Much lighter weight than full stabilizer simulation.
pub struct GpuPauliProp {
    num_qubits: usize,
    num_shots: u32,
    shot_words: u32, // Number of u32 words per qubit (shots / 32, rounded up)

    // GPU resources
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Fault tables on GPU: x_faults[qubit * shot_words + word_idx]
    x_faults_buffer: wgpu::Buffer,
    z_faults_buffer: wgpu::Buffer,

    // Parameters uniform buffer (kept alive for GPU bind group)
    #[allow(dead_code)]
    params_buffer: wgpu::Buffer,

    // Gate queue buffer
    gate_queue_buffer: wgpu::Buffer,

    // Random bits for probabilistic fault injection
    random_buffer: wgpu::Buffer,

    // Pipeline and bind group
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    // Gate queue (CPU side, flushed to GPU periodically)
    gate_queue: Vec<u32>,

    // RNG for fault injection
    rng: PecosRng,
}

/// Parameters passed to the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    num_qubits: u32,
    num_shots: u32,
    shot_words: u32,
    _padding: u32,
}

impl GpuPauliProp {
    /// Create a new GPU Pauli propagator.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits to track
    /// * `num_shots` - Number of parallel shots
    ///
    /// # Returns
    /// A new `GpuPauliProp` instance or an error
    ///
    /// # Errors
    /// Returns an error if no GPU adapter is found or device creation fails.
    pub fn new(num_qubits: usize, num_shots: u32) -> Result<Self, String> {
        Self::with_seed(num_qubits, num_shots, time_seed())
    }

    /// Create a new GPU Pauli propagator with a specific seed.
    ///
    /// # Errors
    /// Returns an error if no GPU adapter is found or device creation fails.
    #[allow(clippy::cast_possible_truncation)] // GPU params: qubit/shot counts fit in u32
    pub fn with_seed(num_qubits: usize, num_shots: u32, seed: u64) -> Result<Self, String> {
        // Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| "No GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("PauliProp Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        // Calculate dimensions
        let shot_words = num_shots.div_ceil(32);
        let table_size = u64::from(num_qubits as u32 * shot_words * 4); // bytes

        // Create fault table buffers (initialized to zero = no faults)
        let x_faults_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("X Faults Buffer"),
            size: table_size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let z_faults_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Z Faults Buffer"),
            size: table_size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = Params {
            num_qubits: num_qubits as u32,
            num_shots,
            shot_words,
            _padding: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create gate queue buffer
        let gate_queue_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gate Queue Buffer"),
            size: ((MAX_GATE_QUEUE_SIZE + 1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create random buffer for probabilistic faults
        let random_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Random Buffer"),
            size: u64::from(num_shots * 4), // One u32 per shot
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shader
        let shader_source = include_str!("pauli_prop_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PauliProp Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PauliProp Bind Group Layout"),
            entries: &[
                // 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: x_faults (storage, read-write)
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
                // 2: z_faults (storage, read-write)
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
                // 3: gate_queue (storage, read-only)
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
                // 4: random bits (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PauliProp Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_faults_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: z_faults_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gate_queue_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: random_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PauliProp Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PauliProp Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            num_qubits,
            num_shots,
            shot_words,
            device,
            queue,
            x_faults_buffer,
            z_faults_buffer,
            params_buffer,
            gate_queue_buffer,
            random_buffer,
            pipeline,
            bind_group,
            gate_queue: Vec::with_capacity(MAX_GATE_QUEUE_SIZE),
            rng: PecosRng::seed_from_u64(seed),
        })
    }

    /// Reset all faults to zero (identity).
    pub fn reset(&mut self) {
        self.gate_queue.clear();

        #[allow(clippy::cast_possible_truncation)] // qubit count fits in u32
        let table_size = self.num_qubits as u32 * self.shot_words * 4;
        let zeros = vec![0u8; table_size as usize];

        self.queue.write_buffer(&self.x_faults_buffer, 0, &zeros);
        self.queue.write_buffer(&self.z_faults_buffer, 0, &zeros);
    }

    /// Get the number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the number of shots.
    #[must_use]
    pub fn num_shots(&self) -> u32 {
        self.num_shots
    }

    // =========================================================================
    // Gate operations (queue Clifford gates for propagation)
    // =========================================================================

    /// Apply Hadamard gate. Transforms: X <-> Z, Y -> -Y
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn h(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.queue_gate(GATE_H, q as u32, 0);
        }
    }

    /// Apply SZ (S) gate. Transforms: X -> Y, Y -> -X, Z -> Z
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn sz(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.queue_gate(GATE_SZ, q as u32, 0);
        }
    }

    /// Apply SZ-dagger gate. Transforms: X -> -Y, Y -> X, Z -> Z
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn szdg(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.queue_gate(GATE_SZDG, q as u32, 0);
        }
    }

    /// Apply X gate. Toggles X fault.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn x(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.queue_gate(GATE_X, q as u32, 0);
        }
    }

    /// Apply Y gate. Toggles both X and Z faults.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn y(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.queue_gate(GATE_Y, q as u32, 0);
        }
    }

    /// Apply Z gate. Toggles Z fault.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn z(&mut self, qubits: &[usize]) {
        for &q in qubits {
            self.queue_gate(GATE_Z, q as u32, 0);
        }
    }

    /// Apply CX (CNOT) gate.
    /// Transforms: `ctrl_X` -> `tgt_X`, `tgt_Z` -> `ctrl_Z`
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn cx(&mut self, pairs: &[(usize, usize)]) {
        for &(control, target) in pairs {
            self.queue_gate(GATE_CX, control as u32, target as u32);
        }
    }

    /// Apply CZ gate.
    /// Transforms: `ctrl_X` -> `tgt_Z`, `tgt_X` -> `ctrl_Z`
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn cz(&mut self, pairs: &[(usize, usize)]) {
        for &(a, b) in pairs {
            self.queue_gate(GATE_CZ, a as u32, b as u32);
        }
    }

    /// Apply SWAP gate.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn swap(&mut self, pairs: &[(usize, usize)]) {
        for &(a, b) in pairs {
            self.queue_gate(GATE_SWAP, a as u32, b as u32);
        }
    }

    // =========================================================================
    // Fault injection
    // =========================================================================

    /// Inject X fault on a qubit for all shots.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn inject_x_fault(&mut self, qubit: usize) {
        self.queue_gate(FAULT_X, qubit as u32, 0);
    }

    /// Inject Z fault on a qubit for all shots.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn inject_z_fault(&mut self, qubit: usize) {
        self.queue_gate(FAULT_Z, qubit as u32, 0);
    }

    /// Inject Y fault on a qubit for all shots.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32
    pub fn inject_y_fault(&mut self, qubit: usize) {
        self.queue_gate(FAULT_Y, qubit as u32, 0);
    }

    /// Inject probabilistic single-qubit depolarizing fault.
    ///
    /// With probability p, applies a random Pauli (X, Y, or Z) independently
    /// per shot.
    ///
    /// # Arguments
    /// * `qubit` - The qubit to potentially apply fault to
    /// * `probability` - Error probability (0.0 to 1.0)
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32; f32 threshold is intentional
    pub fn inject_depol1(&mut self, qubit: usize, probability: f32) {
        // Upload fresh random bits
        self.upload_random_bits();

        // Encode probability as threshold in the "target" field
        #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
        // probability in [0,1] so product is non-negative
        let threshold = (probability * u32::MAX as f32) as u32;
        self.queue_gate(FAULT_DEPOL1, qubit as u32, threshold);
    }

    /// Inject probabilistic two-qubit depolarizing fault.
    ///
    /// With probability p, applies a random two-qubit Pauli (one of 15 non-II)
    /// independently per shot.
    #[allow(clippy::cast_possible_truncation)] // qubit index fits in u32; f32 threshold is intentional
    pub fn inject_depol2(&mut self, qubit_a: usize, qubit_b: usize, probability: f32) {
        // Upload fresh random bits
        self.upload_random_bits();

        // For 2Q faults, we need to encode both qubits and probability
        // Use a separate queue entry for the probability threshold
        #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
        // probability in [0,1] so product is non-negative
        let threshold = (probability * u32::MAX as f32) as u32;

        // Queue as: [FAULT_DEPOL2, qubit_a, qubit_b, threshold]
        if self.gate_queue.len() + 4 > MAX_GATE_QUEUE_SIZE {
            self.flush();
        }
        self.gate_queue.push(FAULT_DEPOL2);
        self.gate_queue.push(qubit_a as u32);
        self.gate_queue.push(qubit_b as u32);
        self.gate_queue.push(threshold);
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Flush pending operations to GPU.
    ///
    /// This processes operations in two passes:
    /// 1. Single-qubit operations (gates and fault injection)
    /// 2. Two-qubit operations
    ///
    /// This ensures that faults injected on one qubit are visible to two-qubit
    /// gates that read from that qubit.
    #[allow(clippy::cast_possible_truncation)] // GPU params: counts fit in u32
    pub fn flush(&mut self) {
        if self.gate_queue.is_empty() {
            return;
        }

        // Separate 1Q and 2Q operations
        let mut ops_1q: Vec<u32> = Vec::new();
        let mut ops_2q: Vec<u32> = Vec::new();

        let mut i = 0;
        while i < self.gate_queue.len() {
            let gate_type = self.gate_queue[i];

            // Check if this is a 2Q gate
            let is_2q = matches!(gate_type, GATE_CX | GATE_CZ | GATE_SWAP);

            // FAULT_DEPOL2 uses 4 words
            let op_len = if gate_type == FAULT_DEPOL2 { 4 } else { 3 };

            if is_2q {
                ops_2q.extend_from_slice(&self.gate_queue[i..i + op_len]);
            } else {
                ops_1q.extend_from_slice(&self.gate_queue[i..i + op_len]);
            }

            i += op_len;
        }

        let total_work_items = self.num_qubits as u32 * self.shot_words;
        let workgroups = total_work_items.div_ceil(256);

        // Dispatch 1Q operations first
        if !ops_1q.is_empty() {
            let mut queue_data = Vec::with_capacity(ops_1q.len() + 1);
            queue_data.push(ops_1q.len() as u32);
            queue_data.extend_from_slice(&ops_1q);

            self.queue.write_buffer(
                &self.gate_queue_buffer,
                0,
                bytemuck::cast_slice(&queue_data),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PauliProp 1Q Encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PauliProp 1Q Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));

            // Wait for 1Q to complete before 2Q
            if !ops_2q.is_empty() {
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            }
        }

        // Dispatch 2Q operations
        if !ops_2q.is_empty() {
            let mut queue_data = Vec::with_capacity(ops_2q.len() + 1);
            queue_data.push(ops_2q.len() as u32);
            queue_data.extend_from_slice(&ops_2q);

            self.queue.write_buffer(
                &self.gate_queue_buffer,
                0,
                bytemuck::cast_slice(&queue_data),
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("PauliProp 2Q Encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PauliProp 2Q Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        self.gate_queue.clear();
    }

    /// Synchronously wait for GPU operations to complete.
    pub fn sync(&mut self) {
        self.flush();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    // =========================================================================
    // Measurement (check fault effects)
    // =========================================================================

    /// Check which shots would have their Z measurement flipped due to X faults.
    ///
    /// Returns a vector of bools, one per shot, indicating if the measurement
    /// outcome would be flipped (true = X or Y fault present = flip).
    pub fn measure_z_flips(&mut self, qubits: &[usize]) -> Vec<Vec<bool>> {
        self.sync();

        // Read X faults from GPU
        let x_faults = self.read_fault_table(&self.x_faults_buffer);

        let mut results = vec![vec![false; qubits.len()]; self.num_shots as usize];

        for (meas_idx, &qubit) in qubits.iter().enumerate() {
            let base = qubit * self.shot_words as usize;
            for (shot, result) in results.iter_mut().enumerate() {
                let word_idx = shot / 32;
                let bit_idx = shot % 32;
                let has_x = (x_faults[base + word_idx] >> bit_idx) & 1 != 0;
                result[meas_idx] = has_x;
            }
        }

        results
    }

    /// Check which shots would have their X measurement flipped due to Z faults.
    pub fn measure_x_flips(&mut self, qubits: &[usize]) -> Vec<Vec<bool>> {
        self.sync();

        let z_faults = self.read_fault_table(&self.z_faults_buffer);

        let mut results = vec![vec![false; qubits.len()]; self.num_shots as usize];

        for (meas_idx, &qubit) in qubits.iter().enumerate() {
            let base = qubit * self.shot_words as usize;
            for (shot, result) in results.iter_mut().enumerate() {
                let word_idx = shot / 32;
                let bit_idx = shot % 32;
                let has_z = (z_faults[base + word_idx] >> bit_idx) & 1 != 0;
                result[meas_idx] = has_z;
            }
        }

        results
    }

    /// Get the X and Z fault tables (for analysis).
    ///
    /// Returns (`x_faults`, `z_faults`) where each is indexed as [qubit][shot].
    pub fn get_fault_tables(&mut self) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        self.sync();

        let x_raw = self.read_fault_table(&self.x_faults_buffer);
        let z_raw = self.read_fault_table(&self.z_faults_buffer);

        let mut x_faults = vec![vec![false; self.num_shots as usize]; self.num_qubits];
        let mut z_faults = vec![vec![false; self.num_shots as usize]; self.num_qubits];

        for qubit in 0..self.num_qubits {
            let base = qubit * self.shot_words as usize;
            for shot in 0..self.num_shots as usize {
                let word_idx = shot / 32;
                let bit_idx = shot % 32;
                x_faults[qubit][shot] = (x_raw[base + word_idx] >> bit_idx) & 1 != 0;
                z_faults[qubit][shot] = (z_raw[base + word_idx] >> bit_idx) & 1 != 0;
            }
        }

        (x_faults, z_faults)
    }

    /// Check if a Pauli string anticommutes with the accumulated faults.
    ///
    /// This is used to check for logical errors: if the fault anticommutes
    /// with a logical operator, it's a logical error.
    ///
    /// # Arguments
    /// * `x_qubits` - Qubits with X in the Pauli string
    /// * `z_qubits` - Qubits with Z in the Pauli string
    ///
    /// # Returns
    /// A vector of bools, one per shot, true if anticommutes (logical error).
    pub fn check_anticommutation(&mut self, x_qubits: &[usize], z_qubits: &[usize]) -> Vec<bool> {
        self.sync();

        let x_faults = self.read_fault_table(&self.x_faults_buffer);
        let z_faults = self.read_fault_table(&self.z_faults_buffer);

        let mut results = vec![false; self.num_shots as usize];

        for (shot, result) in results.iter_mut().enumerate() {
            let word_idx = shot / 32;
            let bit_idx = shot % 32;

            let mut anticom_count = 0u32;

            // X in logical anticommutes with Z faults
            for &q in x_qubits {
                let base = q * self.shot_words as usize;
                if (z_faults[base + word_idx] >> bit_idx) & 1 != 0 {
                    anticom_count += 1;
                }
            }

            // Z in logical anticommutes with X faults
            for &q in z_qubits {
                let base = q * self.shot_words as usize;
                if (x_faults[base + word_idx] >> bit_idx) & 1 != 0 {
                    anticom_count += 1;
                }
            }

            // Odd number of anticommutations = overall anticommutes
            *result = anticom_count % 2 == 1;
        }

        results
    }

    // =========================================================================
    // Internal helpers
    // =========================================================================

    fn queue_gate(&mut self, gate_type: u32, qubit1: u32, qubit2: u32) {
        if self.gate_queue.len() + 3 > MAX_GATE_QUEUE_SIZE {
            self.flush();
        }
        self.gate_queue.push(gate_type);
        self.gate_queue.push(qubit1);
        self.gate_queue.push(qubit2);
    }

    fn upload_random_bits(&mut self) {
        let random_bits: Vec<u32> = (0..self.num_shots).map(|_| self.rng.next_u32()).collect();

        self.queue
            .write_buffer(&self.random_buffer, 0, bytemuck::cast_slice(&random_bits));
    }

    fn read_fault_table(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let table_size = self.num_qubits * self.shot_words as usize;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (table_size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (table_size * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("GPU worker channel closed")
            .expect("GPU buffer mapping failed");

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        result
    }
}

crate::impl_gpu_drop!(GpuPauliProp);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let prop = GpuPauliProp::new(10, 64);
        assert!(prop.is_ok());
        let prop = prop.unwrap();
        assert_eq!(prop.num_qubits(), 10);
        assert_eq!(prop.num_shots(), 64);
    }

    #[test]
    fn test_reset() {
        let mut prop = GpuPauliProp::new(4, 32).unwrap();
        prop.inject_x_fault(0);
        prop.flush();

        // Should have X fault on qubit 0
        let (x, _z) = prop.get_fault_tables();
        assert!(x[0].iter().all(|&b| b)); // All shots have X on qubit 0

        prop.reset();
        let (x, z) = prop.get_fault_tables();
        assert!(x.iter().all(|row| row.iter().all(|&b| !b))); // No X faults
        assert!(z.iter().all(|row| row.iter().all(|&b| !b))); // No Z faults
    }

    #[test]
    fn test_h_propagation() {
        let mut prop = GpuPauliProp::with_seed(2, 32, 42).unwrap();

        // X fault on qubit 0
        prop.inject_x_fault(0);

        // H transforms X -> Z
        prop.h(&[0]);
        prop.flush();

        let (x, z) = prop.get_fault_tables();

        // Qubit 0 should now have Z fault, not X
        assert!(x[0].iter().all(|&b| !b)); // No X fault
        assert!(z[0].iter().all(|&b| b)); // Z fault present
    }

    #[test]
    fn test_sz_propagation() {
        let mut prop = GpuPauliProp::with_seed(2, 32, 42).unwrap();

        // X fault on qubit 0
        prop.inject_x_fault(0);

        // SZ transforms X -> Y (XZ)
        prop.sz(&[0]);
        prop.flush();

        let (x, z) = prop.get_fault_tables();

        // Qubit 0 should have both X and Z (Y fault)
        assert!(x[0].iter().all(|&b| b)); // X present
        assert!(z[0].iter().all(|&b| b)); // Z present (Y = XZ)
    }

    #[test]
    fn test_cx_propagation() {
        let mut prop = GpuPauliProp::with_seed(2, 32, 42).unwrap();

        // X fault on control (qubit 0)
        prop.inject_x_fault(0);

        // CX: control X propagates to target
        prop.cx(&[(0, 1)]);
        prop.flush();

        let (x, _z) = prop.get_fault_tables();

        // Both qubits should have X fault
        assert!(x[0].iter().all(|&b| b)); // Control still has X
        assert!(x[1].iter().all(|&b| b)); // Target now has X
    }

    #[test]
    fn test_cx_z_propagation() {
        let mut prop = GpuPauliProp::with_seed(2, 32, 42).unwrap();

        // Z fault on target (qubit 1)
        prop.inject_z_fault(1);

        // CX: target Z propagates to control
        prop.cx(&[(0, 1)]);
        prop.flush();

        let (_x, z) = prop.get_fault_tables();

        // Both qubits should have Z fault
        assert!(z[0].iter().all(|&b| b)); // Control now has Z
        assert!(z[1].iter().all(|&b| b)); // Target still has Z
    }

    #[test]
    fn test_measure_z_flips() {
        let mut prop = GpuPauliProp::with_seed(3, 32, 42).unwrap();

        // X fault on qubit 0 - would flip Z measurement
        prop.inject_x_fault(0);

        // Z fault on qubit 1 - would NOT flip Z measurement
        prop.inject_z_fault(1);

        // No fault on qubit 2
        prop.flush();

        let flips = prop.measure_z_flips(&[0, 1, 2]);

        // All shots: qubit 0 flipped, qubits 1 and 2 not flipped
        for shot_result in &flips {
            assert!(shot_result[0]); // X fault flips Z measurement
            assert!(!shot_result[1]); // Z fault doesn't flip Z measurement
            assert!(!shot_result[2]); // No fault, no flip
        }
    }

    #[test]
    fn test_anticommutation_check() {
        let mut prop = GpuPauliProp::with_seed(4, 32, 42).unwrap();

        // X fault on qubit 0
        prop.inject_x_fault(0);
        prop.flush();

        // Check against Z logical on qubit 0 (should anticommute)
        let results = prop.check_anticommutation(&[], &[0]);
        assert!(results.iter().all(|&b| b)); // All shots: anticommutes

        // Check against X logical on qubit 0 (should commute)
        let results = prop.check_anticommutation(&[0], &[]);
        assert!(results.iter().all(|&b| !b)); // All shots: commutes
    }
}
