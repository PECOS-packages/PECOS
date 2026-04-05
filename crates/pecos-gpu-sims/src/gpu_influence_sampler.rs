//! GPU Influence Sampler
//!
//! Each thread handles ONE shot and ALL locations.
//! This eliminates atomic contention since each shot has its own output region.

use bytemuck::{Pod, Zeroable};
use pecos_random::{PecosRng, time_seed};
use wgpu::util::DeviceExt;

/// Influence map data for GPU sampling.
///
/// Contains CSR (Compressed Sparse Row) arrays mapping fault locations
/// to their detector and logical influences for X, Y, and Z Pauli faults.
pub struct GpuInfluenceMapData {
    /// Number of fault locations.
    pub num_locations: u32,
    /// Number of detectors.
    pub num_detectors: u32,
    /// Number of logicals.
    pub num_logicals: u32,

    // CSR arrays for detector influences
    /// Offsets for X detector influences: `offsets_x`[loc] to `offsets_x`[loc+1]
    pub detector_offsets_x: Vec<u32>,
    /// Detector indices for X faults.
    pub detector_data_x: Vec<u32>,
    /// Offsets for Y detector influences.
    pub detector_offsets_y: Vec<u32>,
    /// Detector indices for Y faults.
    pub detector_data_y: Vec<u32>,
    /// Offsets for Z detector influences.
    pub detector_offsets_z: Vec<u32>,
    /// Detector indices for Z faults.
    pub detector_data_z: Vec<u32>,

    // CSR arrays for logical influences
    /// Offsets for X logical influences.
    pub logical_offsets_x: Vec<u32>,
    /// Logical indices for X faults.
    pub logical_data_x: Vec<u32>,
    /// Offsets for Y logical influences.
    pub logical_offsets_y: Vec<u32>,
    /// Logical indices for Y faults.
    pub logical_data_y: Vec<u32>,
    /// Offsets for Z logical influences.
    pub logical_offsets_z: Vec<u32>,
    /// Logical indices for Z faults.
    pub logical_data_z: Vec<u32>,
}

impl GpuInfluenceMapData {
    /// Create an empty influence map.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            num_locations: 0,
            num_detectors: 0,
            num_logicals: 0,
            detector_offsets_x: vec![0],
            detector_data_x: vec![],
            detector_offsets_y: vec![0],
            detector_data_y: vec![],
            detector_offsets_z: vec![0],
            detector_data_z: vec![],
            logical_offsets_x: vec![0],
            logical_data_x: vec![],
            logical_offsets_y: vec![0],
            logical_data_y: vec![],
            logical_offsets_z: vec![0],
            logical_data_z: vec![],
        }
    }

    /// Create from CSR arrays exported from a CPU influence map.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn from_csr(
        num_locations: u32,
        num_detectors: u32,
        num_logicals: u32,
        detector_offsets_x: Vec<u32>,
        detector_data_x: Vec<u32>,
        detector_offsets_y: Vec<u32>,
        detector_data_y: Vec<u32>,
        detector_offsets_z: Vec<u32>,
        detector_data_z: Vec<u32>,
        logical_offsets_x: Vec<u32>,
        logical_data_x: Vec<u32>,
        logical_offsets_y: Vec<u32>,
        logical_data_y: Vec<u32>,
        logical_offsets_z: Vec<u32>,
        logical_data_z: Vec<u32>,
    ) -> Self {
        Self {
            num_locations,
            num_detectors,
            num_logicals,
            detector_offsets_x,
            detector_data_x,
            detector_offsets_y,
            detector_data_y,
            detector_offsets_z,
            detector_data_z,
            logical_offsets_x,
            logical_data_x,
            logical_offsets_y,
            logical_data_y,
            logical_offsets_z,
            logical_data_z,
        }
    }
}

/// Parameters for the sampling shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SamplerParams {
    num_locations: u32,
    num_shots: u32,
    num_detectors: u32,
    num_logicals: u32,
    p_error_threshold: u32,
    detector_words: u32,
    logical_words: u32,
    _padding: u32,
}

/// Optimized GPU influence sampler using per-shot parallelization.
///
/// This version eliminates atomic contention by having each thread
/// process all locations for a single shot, writing to its own output region.
pub struct GpuInfluenceSampler {
    num_locations: u32,
    num_detectors: u32,
    num_logicals: u32,

    device: wgpu::Device,
    queue: wgpu::Queue,

    params_buffer: wgpu::Buffer,

    // Influence map CSR buffers
    detector_offsets_x_buffer: wgpu::Buffer,
    detector_data_x_buffer: wgpu::Buffer,
    detector_offsets_y_buffer: wgpu::Buffer,
    detector_data_y_buffer: wgpu::Buffer,
    detector_offsets_z_buffer: wgpu::Buffer,
    detector_data_z_buffer: wgpu::Buffer,
    logical_offsets_x_buffer: wgpu::Buffer,
    logical_data_x_buffer: wgpu::Buffer,
    logical_offsets_y_buffer: wgpu::Buffer,
    logical_data_y_buffer: wgpu::Buffer,
    logical_offsets_z_buffer: wgpu::Buffer,
    logical_data_z_buffer: wgpu::Buffer,

    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    rng: PecosRng,
}

impl GpuInfluenceSampler {
    /// Create a new optimized GPU influence sampler.
    ///
    /// # Errors
    /// Returns an error if no GPU adapter is found or device creation fails.
    pub fn new(influence_map: &GpuInfluenceMapData, seed: u64) -> Result<Self, String> {
        Self::create_internal(influence_map, seed)
    }

    /// Create with a random seed.
    ///
    /// # Errors
    /// Returns an error if no GPU adapter is found or device creation fails.
    pub fn new_random(influence_map: &GpuInfluenceMapData) -> Result<Self, String> {
        Self::new(influence_map, time_seed())
    }

    fn create_internal(map: &GpuInfluenceMapData, seed: u64) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| "No GPU adapter found")?;

        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 16,
            ..wgpu::Limits::default()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("InfluenceSampler Device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("Failed to create device: {e}"))?;

        // Helper to create buffer from data
        let create_buffer = |data: &[u32], label: &str| -> wgpu::Buffer {
            let data = if data.is_empty() { &[0u32] } else { data };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Create influence map buffers
        let detector_offsets_x_buffer = create_buffer(&map.detector_offsets_x, "DetOffX");
        let detector_data_x_buffer = create_buffer(&map.detector_data_x, "DetDataX");
        let detector_offsets_y_buffer = create_buffer(&map.detector_offsets_y, "DetOffY");
        let detector_data_y_buffer = create_buffer(&map.detector_data_y, "DetDataY");
        let detector_offsets_z_buffer = create_buffer(&map.detector_offsets_z, "DetOffZ");
        let detector_data_z_buffer = create_buffer(&map.detector_data_z, "DetDataZ");
        let logical_offsets_x_buffer = create_buffer(&map.logical_offsets_x, "LogOffX");
        let logical_data_x_buffer = create_buffer(&map.logical_data_x, "LogDataX");
        let logical_offsets_y_buffer = create_buffer(&map.logical_offsets_y, "LogOffY");
        let logical_data_y_buffer = create_buffer(&map.logical_data_y, "LogDataY");
        let logical_offsets_z_buffer = create_buffer(&map.logical_offsets_z, "LogOffZ");
        let logical_data_z_buffer = create_buffer(&map.logical_data_z, "LogDataZ");

        // Create params buffer
        let params = SamplerParams {
            num_locations: map.num_locations,
            num_shots: 0,
            num_detectors: map.num_detectors,
            num_logicals: map.num_logicals,
            p_error_threshold: 0,
            detector_words: 0,
            logical_words: 0,
            _padding: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sampler Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load shader
        let shader_source = include_str!("influence_sampler_shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("InfluenceSampler Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("InfluenceSampler BindGroupLayout"),
            entries: &[
                // 0: params uniform
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
                // 1-12: CSR buffers (read-only)
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 13: random seeds
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 14-15: output buffers
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("InfluenceSampler PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("InfluenceSampler Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            num_locations: map.num_locations,
            num_detectors: map.num_detectors,
            num_logicals: map.num_logicals,
            device,
            queue,
            params_buffer,
            detector_offsets_x_buffer,
            detector_data_x_buffer,
            detector_offsets_y_buffer,
            detector_data_y_buffer,
            detector_offsets_z_buffer,
            detector_data_z_buffer,
            logical_offsets_x_buffer,
            logical_data_x_buffer,
            logical_offsets_y_buffer,
            logical_data_y_buffer,
            logical_offsets_z_buffer,
            logical_data_z_buffer,
            bind_group_layout,
            pipeline,
            rng: PecosRng::seed_from_u64(seed),
        })
    }

    /// Sample with uniform depolarizing noise.
    pub fn sample_uniform(&mut self, num_shots: u32, p_error: f64) -> GpuSamplingResult {
        let detector_words = self.num_detectors.div_ceil(32).max(1);
        let logical_words = self.num_logicals.div_ceil(32).max(1);

        // Update params
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // probability in [0,1] maps to [0, u32::MAX]
        let p_threshold = (p_error * f64::from(u32::MAX)) as u32;
        let params = SamplerParams {
            num_locations: self.num_locations,
            num_shots,
            num_detectors: self.num_detectors,
            num_logicals: self.num_logicals,
            p_error_threshold: p_threshold,
            detector_words,
            logical_words,
            _padding: 0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Generate random seeds (one per shot)
        let seeds: Vec<u32> = (0..num_shots).map(|_| self.rng.next_u32()).collect();

        let random_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Random Seeds"),
                contents: bytemuck::cast_slice(&seeds),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output buffers - layout: [shot * words + word_idx]
        let detector_output_size = (num_shots as usize * detector_words as usize * 4) as u64;
        let logical_output_size = (num_shots as usize * logical_words as usize * 4) as u64;

        let detector_output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Detector Output"),
            size: detector_output_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let logical_output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Logical Output"),
            size: logical_output_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("InfluenceSampler BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.detector_offsets_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.detector_data_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.detector_offsets_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.detector_data_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.detector_offsets_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.detector_data_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.logical_offsets_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.logical_data_x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.logical_offsets_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.logical_data_y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.logical_offsets_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.logical_data_z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: random_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: detector_output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: logical_output_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch: one thread per shot
        let workgroups = num_shots.div_ceil(256);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("InfluenceSampler Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("InfluenceSampler Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results - layout is [shot][word]
        let detector_flips = self.read_output(
            &detector_output_buffer,
            num_shots as usize,
            detector_words as usize,
        );
        let logical_flips = self.read_output(
            &logical_output_buffer,
            num_shots as usize,
            logical_words as usize,
        );

        GpuSamplingResult {
            num_shots: num_shots as usize,
            detector_flips,
            logical_flips,
            num_detectors: self.num_detectors as usize,
            num_logicals: self.num_logicals as usize,
            detector_words: detector_words as usize,
            logical_words: logical_words as usize,
        }
    }

    fn read_output(&self, buffer: &wgpu::Buffer, num_shots: usize, words: usize) -> Vec<u32> {
        let total_size = (num_shots * words * 4) as u64;
        if total_size == 0 {
            return vec![];
        }

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: total_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, total_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let raw: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        raw
    }
}

/// Result from GPU sampling.
pub struct GpuSamplingResult {
    pub num_shots: usize,
    /// Flat array: [shot * `detector_words` + word]
    pub detector_flips: Vec<u32>,
    /// Flat array: [shot * `logical_words` + word]
    pub logical_flips: Vec<u32>,
    pub num_detectors: usize,
    pub num_logicals: usize,
    pub detector_words: usize,
    pub logical_words: usize,
}

impl GpuSamplingResult {
    /// Count shots with any logical error.
    #[must_use]
    pub fn count_logical_errors(&self) -> usize {
        if self.num_logicals == 0 {
            return 0;
        }

        let mut count = 0;
        for shot in 0..self.num_shots {
            let base = shot * self.logical_words;
            let has_error = (0..self.logical_words)
                .any(|w| self.logical_flips.get(base + w).copied().unwrap_or(0) != 0);
            if has_error {
                count += 1;
            }
        }
        count
    }

    /// Check if a specific shot has a logical error.
    #[must_use]
    pub fn has_logical_error(&self, shot: usize) -> bool {
        if shot >= self.num_shots || self.num_logicals == 0 {
            return false;
        }
        let base = shot * self.logical_words;
        (0..self.logical_words).any(|w| self.logical_flips.get(base + w).copied().unwrap_or(0) != 0)
    }

    /// Get detector flip bits for a specific shot.
    #[must_use]
    pub fn detector_flips_for_shot(&self, shot: usize) -> Vec<u32> {
        if shot >= self.num_shots {
            return vec![];
        }
        let base = shot * self.detector_words;
        self.detector_flips[base..base + self.detector_words].to_vec()
    }
}
