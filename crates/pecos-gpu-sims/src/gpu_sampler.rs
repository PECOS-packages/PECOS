//! GPU-accelerated measurement sampler for symbolic stabilizer simulation.
//!
//! This sampler takes measurement dependencies computed by `SymbolicSparseStab` and
//! samples millions of shots in parallel on the GPU. It supports both clean and
//! noisy sampling.
//!
//! # Usage
//!
//! ```
//! use pecos_simulators::SymbolicSparseStab;
//! use pecos_simulators::measurement_sampler::MeasurementKind;
//! use pecos_gpu_sims::GpuMeasurementSampler;
//!
//! // Run symbolic simulation
//! let mut sim = SymbolicSparseStab::new(4);
//! sim.h(&[0]);
//! sim.cx(&[(0, 1)]);
//! sim.mz(&[0]);
//! sim.mz(&[1]);
//!
//! // Convert measurement history to MeasurementKind for GPU sampler
//! let measurements = MeasurementKind::from_history(sim.measurement_history());
//!
//! // Create GPU sampler from measurement kinds
//! let sampler = GpuMeasurementSampler::new(&measurements).unwrap();
//!
//! // Sample 1000 shots (returns full shot data)
//! let results = sampler.sample(1000);
//!
//! // Sample with noise (1% error rate)
//! let noisy_results = sampler.sample_noisy(1000, 0.01);
//!
//! // Or use sample_counts for faster statistics-only sampling
//! // (only works for Fixed/Random measurements, returns Vec<u32> of counts)
//! if let Ok(counts) = sampler.sample_counts(1000) {
//!     // counts[i] = number of 1s in measurement i across all shots
//! }
//! ```

// GPU uses 32-bit values throughout; casting from usize/u64 to u32 is intentional
// and values are bounded by buffer sizes that fit in u32
#![allow(clippy::cast_possible_truncation)]
// Converting probability (f64) to fixed-point threshold (u32) is intentional
#![allow(clippy::cast_sign_loss)]

use crate::gpu_probe::gpu_context;
use pecos_random::PecosRng;
use pecos_simulators::measurement_sampler::MeasurementKind;

/// Maximum number of dependencies per Computed measurement.
/// This limit simplifies GPU buffer layout.
const MAX_DEPS_PER_MEASUREMENT: usize = 16;

/// GPU buffer layout constants
const WORKGROUP_SIZE: u32 = 256;

/// GPU-accelerated measurement sampler.
///
/// Takes measurement dependencies from symbolic stabilizer simulation and
/// samples many shots in parallel on the GPU.
pub struct GpuMeasurementSampler {
    device: wgpu::Device,
    queue: wgpu::Queue,

    /// Number of measurements
    num_measurements: u32,

    /// Measurement metadata buffer (type, source/flip, dep count)
    /// Kept alive to maintain GPU buffer; data is accessed via bind group
    _measurement_meta_buffer: wgpu::Buffer,

    /// Dependency indices buffer (flattened, `MAX_DEPS_PER_MEASUREMENT` per measurement)
    /// Kept alive to maintain GPU buffer; data is accessed via bind group
    _deps_buffer: wgpu::Buffer,

    /// Parameters buffer
    params_buffer: wgpu::Buffer,

    /// RNG seed data buffer (4 u32s per word, used as input to stateless hash)
    seeds_buffer: wgpu::Buffer,

    /// Output results buffer
    results_buffer: wgpu::Buffer,

    /// Counts buffer for statistics (one u32 per measurement)
    counts_buffer: wgpu::Buffer,

    /// Staging buffer for readback
    staging_buffer: wgpu::Buffer,

    /// Small staging buffer for counts
    counts_staging_buffer: wgpu::Buffer,

    /// Bind group
    bind_group: wgpu::BindGroup,

    /// Sequential sampling pipeline (for complex dependency patterns)
    sample_pipeline: wgpu::ComputePipeline,

    /// Parallel sampling pipeline (for Fixed/Random heavy workloads)
    parallel_pipeline: wgpu::ComputePipeline,

    /// Noisy sampling pipeline (applies bit flips)
    noisy_pipeline: wgpu::ComputePipeline,

    /// Sample and count combined pipeline (for stats-only mode)
    sample_count_pipeline: wgpu::ComputePipeline,

    /// Maximum shots this sampler can handle (based on buffer sizes)
    max_shots: usize,

    /// Whether all measurements are independent (Fixed or Random)
    all_independent: bool,
}

/// Packed measurement metadata for GPU.
/// Layout: [type: 4 bits][flip: 1 bit][`dep_count`: 4 bits][source/padding: 23 bits]
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct MeasurementMeta {
    /// Packed: type (4 bits) | flip (1 bit) | `dep_count` (4 bits) | source (23 bits)
    packed: u32,
}

// GPU uses 32-bit packed values; truncation is intentional and values fit within bit fields
#[allow(clippy::cast_possible_truncation)]
impl MeasurementMeta {
    const TYPE_FIXED_0: u32 = 0;
    const TYPE_FIXED_1: u32 = 1;
    const TYPE_RANDOM: u32 = 2;
    const TYPE_COPY: u32 = 3;
    const TYPE_COPY_FLIPPED: u32 = 4;
    const TYPE_COMPUTED: u32 = 5;

    fn fixed(value: bool) -> Self {
        let type_bits = if value {
            Self::TYPE_FIXED_1
        } else {
            Self::TYPE_FIXED_0
        };
        Self {
            packed: type_bits & 0xF,
        }
    }

    fn random() -> Self {
        Self {
            packed: Self::TYPE_RANDOM & 0xF,
        }
    }

    fn copy(source: usize) -> Self {
        Self {
            packed: (Self::TYPE_COPY & 0xF) | ((source as u32 & 0x007F_FFFF) << 9),
        }
    }

    fn copy_flipped(source: usize) -> Self {
        Self {
            packed: (Self::TYPE_COPY_FLIPPED & 0xF) | ((source as u32 & 0x007F_FFFF) << 9),
        }
    }

    fn computed(dep_count: usize, flip: bool) -> Self {
        let flip_bit = if flip { 1u32 << 4 } else { 0 };
        Self {
            packed: (Self::TYPE_COMPUTED & 0xF) | flip_bit | ((dep_count as u32 & 0xF) << 5),
        }
    }
}

unsafe impl bytemuck::Pod for MeasurementMeta {}
unsafe impl bytemuck::Zeroable for MeasurementMeta {}

/// Result of GPU sampling.
#[derive(Clone, Debug)]
pub struct GpuSampleResult {
    /// Column-major storage: columns[measurement][word]
    /// Each u32 contains 32 shots (bit-packed)
    columns: Vec<Vec<u32>>,
    /// Number of shots
    shots: usize,
}

impl GpuSampleResult {
    /// Get the outcome for a specific shot and measurement.
    #[must_use]
    pub fn get(&self, shot: usize, measurement: usize) -> bool {
        let word_idx = shot / 32;
        let bit_idx = shot % 32;
        (self.columns[measurement][word_idx] & (1 << bit_idx)) != 0
    }

    /// Number of shots.
    #[must_use]
    pub fn shots(&self) -> usize {
        self.shots
    }

    /// Number of measurements.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.columns.len()
    }

    /// Get raw column data for a measurement.
    #[must_use]
    pub fn column(&self, measurement: usize) -> &[u32] {
        &self.columns[measurement]
    }

    /// Count ones in a measurement column.
    #[must_use]
    pub fn count_ones(&self, measurement: usize) -> usize {
        let full_words = self.shots / 32;
        let remaining_bits = self.shots % 32;

        let mut count: usize = self.columns[measurement][..full_words]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();

        if remaining_bits > 0 {
            let mask = (1u32 << remaining_bits) - 1;
            count += (self.columns[measurement][full_words] & mask).count_ones() as usize;
        }

        count
    }

    /// Count zeros in a measurement column.
    #[must_use]
    pub fn count_zeros(&self, measurement: usize) -> usize {
        self.shots - self.count_ones(measurement)
    }
}

impl GpuMeasurementSampler {
    /// Maximum buffer binding size (use 120 MiB to stay safely under 128 MiB limit)
    const MAX_BUFFER_SIZE: u64 = 120 * 1024 * 1024;

    /// Create a new GPU sampler from measurement kinds.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails or if the measurement
    /// structure is invalid (e.g., too many dependencies).
    pub fn new(measurements: &[MeasurementKind]) -> Result<Self, String> {
        // Calculate max_shots based on GPU buffer limits
        // Results buffer size = num_measurements * num_words * 4 bytes
        // num_words = shots / 32
        // So: buffer_size = num_measurements * (shots / 32) * 4 = num_measurements * shots / 8
        // max_shots = MAX_BUFFER_SIZE * 8 / num_measurements
        let num_measurements = measurements.len().max(1) as u64;
        let max_shots = ((Self::MAX_BUFFER_SIZE * 8) / num_measurements) as usize;
        // Cap at 100M shots for sanity
        let max_shots = max_shots.min(100_000_000);
        Self::with_max_shots(measurements, max_shots)
    }

    /// Create a new GPU sampler with a specific maximum shot capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU initialization fails.
    #[allow(clippy::too_many_lines)] // GPU initialization requires many setup steps
    pub fn with_max_shots(
        measurements: &[MeasurementKind],
        max_shots: usize,
    ) -> Result<Self, String> {
        // Validate measurements
        for (i, kind) in measurements.iter().enumerate() {
            if let MeasurementKind::Computed { deps, .. } = kind
                && deps.len() > MAX_DEPS_PER_MEASUREMENT
            {
                return Err(format!(
                    "Measurement {} has {} dependencies, max is {}",
                    i,
                    deps.len(),
                    MAX_DEPS_PER_MEASUREMENT
                ));
            }
        }

        let ctx = gpu_context().map_err(|e| e.to_string())?;
        let device = ctx.device;
        let queue = ctx.queue;

        let num_measurements = measurements.len() as u32;
        let num_words = max_shots.div_ceil(32) as u32;

        // Convert measurements to GPU format and check if all are independent
        let mut meta_data = Vec::with_capacity(measurements.len());
        let mut deps_data = vec![0u32; measurements.len() * MAX_DEPS_PER_MEASUREMENT];
        let mut all_independent = true;

        for (i, kind) in measurements.iter().enumerate() {
            match kind {
                MeasurementKind::Fixed(value) => {
                    meta_data.push(MeasurementMeta::fixed(*value));
                }
                MeasurementKind::Random => {
                    meta_data.push(MeasurementMeta::random());
                }
                MeasurementKind::Copy(src) => {
                    meta_data.push(MeasurementMeta::copy(*src));
                    all_independent = false;
                }
                MeasurementKind::CopyFlipped(src) => {
                    meta_data.push(MeasurementMeta::copy_flipped(*src));
                    all_independent = false;
                }
                MeasurementKind::Computed { deps, flip } => {
                    meta_data.push(MeasurementMeta::computed(deps.len(), *flip));
                    let base = i * MAX_DEPS_PER_MEASUREMENT;
                    for (j, &dep) in deps.iter().enumerate() {
                        deps_data[base + j] = dep as u32;
                    }
                    all_independent = false;
                }
            }
        }

        // Create buffers
        let measurement_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Measurement Meta Buffer"),
            size: (measurements.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let deps_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dependencies Buffer"),
            size: (deps_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: 32, // 8 u32s
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let seeds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Seeds Buffer"),
            size: u64::from(num_words) * 4 * 4, // 4 u32s per word for RNG state
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Results buffer: one u32 per (measurement, word) pair
        let results_size = u64::from(num_measurements) * u64::from(num_words) * 4;
        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: results_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Counts buffer: one u32 per measurement for statistics
        let counts_size = u64::from(num_measurements) * 4;
        let counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counts Buffer"),
            size: counts_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let counts_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counts Staging Buffer"),
            size: counts_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload measurement data
        queue.write_buffer(
            &measurement_meta_buffer,
            0,
            bytemuck::cast_slice(&meta_data),
        );
        queue.write_buffer(&deps_buffer, 0, bytemuck::cast_slice(&deps_data));

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sampler Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sampler_shader.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sampler Bind Group Layout"),
            entries: &[
                // measurement_meta
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // deps
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
                // params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // seeds
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
                // results
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
                // counts
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sampler Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: measurement_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: deps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: seeds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counts_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sampler Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let sample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sample Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sample_measurements"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let parallel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parallel Sample Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sample_parallel"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let noisy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Noisy Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_noise"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sample_count_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sample and Count Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("sample_and_count"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(Self {
            device,
            queue,
            num_measurements,
            _measurement_meta_buffer: measurement_meta_buffer,
            _deps_buffer: deps_buffer,
            params_buffer,
            seeds_buffer,
            results_buffer,
            counts_buffer,
            staging_buffer,
            counts_staging_buffer,
            bind_group,
            sample_pipeline,
            parallel_pipeline,
            noisy_pipeline,
            sample_count_pipeline,
            max_shots,
            all_independent,
        })
    }

    /// Sample the specified number of shots without noise.
    ///
    /// # Panics
    ///
    /// Panics if `shots` exceeds `max_shots`.
    #[must_use]
    pub fn sample(&self, shots: usize) -> GpuSampleResult {
        self.sample_with_seed(shots, rand::random())
    }

    /// Sample with a specific RNG seed.
    ///
    /// # Panics
    ///
    /// Panics if `shots` exceeds `max_shots`.
    #[must_use]
    pub fn sample_with_seed(&self, shots: usize, seed: u64) -> GpuSampleResult {
        assert!(
            shots <= self.max_shots,
            "shots ({}) exceeds max_shots ({})",
            shots,
            self.max_shots
        );

        let num_words = shots.div_ceil(32) as u32;

        // Generate seeds for each word using PecosRng
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut seeds = Vec::with_capacity(num_words as usize * 4);
        for _ in 0..num_words {
            // Xorshift state: 4 u32s per word
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
        }
        self.queue
            .write_buffer(&self.seeds_buffer, 0, bytemuck::cast_slice(&seeds));

        // Write params
        let params = [
            self.num_measurements,
            num_words,
            shots as u32,
            0u32, // error_rate (not used for clean sampling)
            0u32,
            0u32,
            0u32,
            0u32,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        // Run sampling kernel
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sample Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sample Pass"),
                timestamp_writes: None,
            });

            if self.all_independent {
                // Use parallel kernel: one thread per (measurement, word) pair
                pass.set_pipeline(&self.parallel_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                let total_elements = self.num_measurements * num_words;
                let workgroups_needed = total_elements.div_ceil(WORKGROUP_SIZE);
                let workgroups_x = workgroups_needed.min(65535);
                let workgroups_y = workgroups_needed.div_ceil(65535);
                pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            } else {
                // Use sequential kernel: one thread per word, processes all measurements
                pass.set_pipeline(&self.sample_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(num_words.div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }

        // Copy results to staging
        let results_size = u64::from(self.num_measurements) * u64::from(num_words) * 4;
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.staging_buffer,
            0,
            results_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        self.read_results(shots, num_words)
    }

    /// Sample with uniform noise (each measurement has probability `error_rate` of being flipped).
    ///
    /// This models depolarizing measurement noise.
    #[must_use]
    pub fn sample_noisy(&self, shots: usize, error_rate: f64) -> GpuSampleResult {
        self.sample_noisy_with_seed(shots, error_rate, rand::random())
    }

    /// Sample with noise using a specific seed.
    ///
    /// # Panics
    ///
    /// Panics if `shots` exceeds `max_shots` or if `error_rate` is not in `[0.0, 1.0]`.
    #[must_use]
    pub fn sample_noisy_with_seed(
        &self,
        shots: usize,
        error_rate: f64,
        seed: u64,
    ) -> GpuSampleResult {
        assert!(
            shots <= self.max_shots,
            "shots ({}) exceeds max_shots ({})",
            shots,
            self.max_shots
        );
        assert!(
            (0.0..=1.0).contains(&error_rate),
            "error_rate must be between 0 and 1"
        );

        let num_words = shots.div_ceil(32) as u32;

        // Generate seeds using PecosRng
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut seeds = Vec::with_capacity(num_words as usize * 4);
        for _ in 0..num_words {
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
        }
        self.queue
            .write_buffer(&self.seeds_buffer, 0, bytemuck::cast_slice(&seeds));

        // Convert error rate to threshold (for comparison with uniform random)
        // We use fixed-point arithmetic: threshold = error_rate * 2^32
        let error_threshold = (error_rate * f64::from(u32::MAX)) as u32;

        let params = [
            self.num_measurements,
            num_words,
            shots as u32,
            error_threshold,
            0u32,
            0u32,
            0u32,
            0u32,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Noisy Sample Encoder"),
            });

        // First pass: clean sampling
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sample Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_words.div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        // Second pass: apply noise
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Noise Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noisy_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            // Dispatch one workgroup per (measurement, word_chunk)
            // Use 2D dispatch to avoid exceeding 65535 limit per dimension
            let total_elements = self.num_measurements * num_words;
            let workgroups_needed = total_elements.div_ceil(WORKGROUP_SIZE);
            // Split into x and y dimensions (each max 65535)
            let workgroups_x = workgroups_needed.min(65535);
            let workgroups_y = workgroups_needed.div_ceil(65535);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let results_size = u64::from(self.num_measurements) * u64::from(num_words) * 4;
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.staging_buffer,
            0,
            results_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        self.read_results(shots, num_words)
    }

    /// Sample and return only the count of 1s per measurement.
    ///
    /// This is much faster than `sample()` because it only transfers one u32 per measurement
    /// instead of the full shot data. Use this when you only need statistics, not individual
    /// shot outcomes.
    ///
    /// Only works for independent measurements (Fixed, Random). Returns an error if there are
    /// dependent measurements.
    ///
    /// # Errors
    ///
    /// Returns an error if measurements include dependencies (`Copy`, `CopyFlipped`, `Computed`).
    ///
    /// # Panics
    ///
    /// Panics if `shots` exceeds `max_shots`.
    pub fn sample_counts(&self, shots: usize) -> Result<Vec<u32>, &'static str> {
        self.sample_counts_with_seed(shots, rand::random())
    }

    /// Sample counts with a specific seed.
    ///
    /// # Errors
    ///
    /// Returns an error if measurements include dependencies (`Copy`, `CopyFlipped`, `Computed`).
    ///
    /// # Panics
    ///
    /// Panics if `shots` exceeds `max_shots`.
    pub fn sample_counts_with_seed(
        &self,
        shots: usize,
        seed: u64,
    ) -> Result<Vec<u32>, &'static str> {
        if !self.all_independent {
            return Err("sample_counts only works for independent measurements (Fixed, Random)");
        }

        assert!(
            shots <= self.max_shots,
            "shots ({}) exceeds max_shots ({})",
            shots,
            self.max_shots
        );

        let num_words = shots.div_ceil(32) as u32;

        // Generate seeds
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut seeds = Vec::with_capacity(num_words as usize * 4);
        for _ in 0..num_words {
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
            seeds.push(rng.next_u32());
        }
        self.queue
            .write_buffer(&self.seeds_buffer, 0, bytemuck::cast_slice(&seeds));

        // Write params
        let params = [
            self.num_measurements,
            num_words,
            shots as u32,
            0u32,
            0u32,
            0u32,
            0u32,
            0u32,
        ];
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

        // Zero out the counts buffer
        let zeros = vec![0u32; self.num_measurements as usize];
        self.queue
            .write_buffer(&self.counts_buffer, 0, bytemuck::cast_slice(&zeros));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sample Counts Encoder"),
            });

        // Run combined sample+count kernel
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sample Counts Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sample_count_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            let total_elements = self.num_measurements * num_words;
            let workgroups_needed = total_elements.div_ceil(WORKGROUP_SIZE);
            let workgroups_x = workgroups_needed.min(65535);
            let workgroups_y = workgroups_needed.div_ceil(65535);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy counts to staging (much smaller than full results!)
        let counts_size = u64::from(self.num_measurements) * 4;
        encoder.copy_buffer_to_buffer(
            &self.counts_buffer,
            0,
            &self.counts_staging_buffer,
            0,
            counts_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read counts
        let buffer_slice = self.counts_staging_buffer.slice(..counts_size);

        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .expect("GPU worker channel closed")
            .expect("Failed to map counts buffer");

        let data = buffer_slice.get_mapped_range();
        let counts: &[u32] = bytemuck::cast_slice(&data);
        let result = counts.to_vec();

        drop(data);
        self.counts_staging_buffer.unmap();

        Ok(result)
    }

    /// Read results from staging buffer.
    fn read_results(&self, shots: usize, num_words: u32) -> GpuSampleResult {
        let results_size = u64::from(self.num_measurements) * u64::from(num_words) * 4;
        let buffer_slice = self.staging_buffer.slice(..results_size);

        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .expect("GPU worker channel closed")
            .expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);

        // Convert to column-major format
        let mut columns = Vec::with_capacity(self.num_measurements as usize);
        for m in 0..self.num_measurements as usize {
            let start = m * num_words as usize;
            let end = start + num_words as usize;
            columns.push(results[start..end].to_vec());
        }

        drop(data);
        self.staging_buffer.unmap();

        GpuSampleResult { columns, shots }
    }

    /// Number of measurements this sampler handles.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.num_measurements as usize
    }

    /// Maximum shots this sampler can handle.
    #[must_use]
    pub fn max_shots(&self) -> usize {
        self.max_shots
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // Test code computes ratios from counts
mod tests {
    use super::*;

    #[test]
    fn test_fixed_measurements() {
        let measurements = vec![
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
            MeasurementKind::Fixed(false),
        ];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let result = sampler.sample_with_seed(1000, 42);

        assert_eq!(result.count_ones(0), 0); // All zeros
        assert_eq!(result.count_ones(1), 1000); // All ones
        assert_eq!(result.count_ones(2), 0); // All zeros
    }

    #[test]
    fn test_random_measurements() {
        let measurements = vec![MeasurementKind::Random];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let result = sampler.sample_with_seed(10000, 42);

        // Should be roughly 50/50
        let ones = result.count_ones(0);
        let ratio = ones as f64 / 10000.0;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "Random should be ~50/50, got {:.2}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_copy_measurement() {
        let measurements = vec![MeasurementKind::Random, MeasurementKind::Copy(0)];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let result = sampler.sample_with_seed(1000, 42);

        // m1 should equal m0 for all shots
        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                result.get(shot, 1),
                "Copy failed at shot {shot}"
            );
        }
    }

    #[test]
    fn test_copy_flipped_measurement() {
        let measurements = vec![MeasurementKind::Random, MeasurementKind::CopyFlipped(0)];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let result = sampler.sample_with_seed(1000, 42);

        // m1 should be NOT m0 for all shots
        for shot in 0..1000 {
            assert_eq!(
                result.get(shot, 0),
                !result.get(shot, 1),
                "CopyFlipped failed at shot {shot}"
            );
        }
    }

    #[test]
    fn test_computed_xor() {
        let measurements = vec![
            MeasurementKind::Random,
            MeasurementKind::Random,
            MeasurementKind::Computed {
                deps: vec![0, 1],
                flip: false,
            },
        ];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let result = sampler.sample_with_seed(1000, 42);

        // m2 should equal m0 XOR m1
        for shot in 0..1000 {
            let expected = result.get(shot, 0) ^ result.get(shot, 1);
            assert_eq!(
                result.get(shot, 2),
                expected,
                "Computed XOR failed at shot {shot}"
            );
        }
    }

    #[test]
    fn test_noisy_sampling() {
        let measurements = vec![MeasurementKind::Fixed(false)]; // All zeros

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        // 10% error rate
        let result = sampler.sample_noisy_with_seed(10000, 0.10, 42);

        // Should have ~10% ones due to noise
        let ones = result.count_ones(0);
        let ratio = ones as f64 / 10000.0;
        assert!(
            (0.08..=0.12).contains(&ratio),
            "Noisy sampling should have ~10% errors, got {:.2}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_sample_counts_fixed() {
        let measurements = vec![
            MeasurementKind::Fixed(false),
            MeasurementKind::Fixed(true),
            MeasurementKind::Fixed(false),
        ];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let counts = sampler.sample_counts_with_seed(10000, 42).unwrap();

        assert_eq!(counts[0], 0, "Fixed(false) should have 0 ones");
        assert_eq!(counts[1], 10000, "Fixed(true) should have all ones");
        assert_eq!(counts[2], 0, "Fixed(false) should have 0 ones");
    }

    #[test]
    fn test_sample_counts_random() {
        let measurements = vec![MeasurementKind::Random];

        let sampler = match GpuMeasurementSampler::new(&measurements) {
            Ok(s) => s,
            Err(e) => {
                println!("No GPU available: {e}");
                return;
            }
        };

        let counts = sampler.sample_counts_with_seed(10000, 42).unwrap();

        // Should be roughly 50/50
        let ratio = f64::from(counts[0]) / 10000.0;
        assert!(
            (0.45..=0.55).contains(&ratio),
            "Random should be ~50/50, got {:.2}%",
            ratio * 100.0
        );
    }
}
