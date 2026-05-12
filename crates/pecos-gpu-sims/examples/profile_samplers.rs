//! Profile CPU and GPU influence samplers to identify bottlenecks
//!
//! Run with: cargo run --example `profile_samplers` --release -p pecos-gpu-sims

use bytemuck::{Pod, Zeroable};
use pecos_gpu_sims::GpuInfluenceMapData;
use pecos_qec::fault_tolerance::InfluenceBuilder;
use pecos_qec::fault_tolerance::dem_builder::DemSampler;
use pecos_quantum::DagCircuit;
use pecos_random::PecosRng;
use std::time::Instant;
use wgpu::util::DeviceExt;

fn build_surface_code_grid(distance: usize, num_rounds: usize) -> DagCircuit {
    let mut dag = DagCircuit::new();

    let num_data = distance * distance;
    let num_x_ancillas = (distance - 1) * (distance - 1);
    let num_z_ancillas = (distance - 1) * (distance - 1);

    let x_ancilla_start = num_data;
    let z_ancilla_start = num_data + num_x_ancillas;

    for _round in 0..num_rounds {
        for a in 0..num_x_ancillas {
            dag.pz(&[x_ancilla_start + a]);
            dag.h(&[x_ancilla_start + a]);
        }
        for a in 0..num_z_ancillas {
            dag.pz(&[z_ancilla_start + a]);
        }

        for row in 0..(distance - 1) {
            for col in 0..(distance - 1) {
                let ancilla = x_ancilla_start + row * (distance - 1) + col;
                let d0 = row * distance + col;
                let d1 = row * distance + col + 1;
                let d2 = (row + 1) * distance + col;
                let d3 = (row + 1) * distance + col + 1;

                dag.cx(&[(ancilla, d0)]);
                dag.cx(&[(ancilla, d1)]);
                dag.cx(&[(ancilla, d2)]);
                dag.cx(&[(ancilla, d3)]);
            }
        }

        for row in 0..(distance - 1) {
            for col in 0..(distance - 1) {
                let ancilla = z_ancilla_start + row * (distance - 1) + col;
                let d0 = row * distance + col;
                let d1 = row * distance + col + 1;
                let d2 = (row + 1) * distance + col;
                let d3 = (row + 1) * distance + col + 1;

                dag.cx(&[(d0, ancilla)]);
                dag.cx(&[(d1, ancilla)]);
                dag.cx(&[(d2, ancilla)]);
                dag.cx(&[(d3, ancilla)]);
            }
        }

        for a in 0..num_x_ancillas {
            dag.h(&[x_ancilla_start + a]);
            dag.mz(&[x_ancilla_start + a]);
        }
        for a in 0..num_z_ancillas {
            dag.mz(&[z_ancilla_start + a]);
        }
    }

    dag
}

/// Profile the CPU `DemSampler` with detailed timing
fn profile_cpu_sampler(
    influence_map: &pecos_qec::fault_tolerance::DagFaultInfluenceMap,
    p_error: f64,
    seed: u64,
    num_shots: usize,
) -> CpuProfile {
    let probs = vec![p_error; influence_map.locations.len()];
    let sampler = DemSampler::from_influence_map(influence_map, &probs);

    let start = Instant::now();
    let _stats = sampler.sample_statistics(num_shots, seed);
    let total_time = start.elapsed();

    CpuProfile {
        total_ms: total_time.as_secs_f64() * 1000.0,
        shots: num_shots,
        locations: influence_map.locations.len(),
    }
}

struct CpuProfile {
    total_ms: f64,
    shots: usize,
    #[allow(dead_code)]
    locations: usize,
}

/// Parameters for the sampling shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SamplerParams {
    num_locations: u32,
    num_shots: u32,
    num_detectors: u32,
    num_dem_outputs: u32,
    p_error_threshold: u32,
    detector_words: u32,
    dem_output_words: u32,
    _padding: u32,
}

/// Profile the GPU sampler with detailed timing for each phase
fn profile_gpu_sampler(
    gpu_map: &GpuInfluenceMapData,
    p_error: f64,
    seed: u64,
    num_shots: u32,
) -> GpuProfile {
    let mut rng = PecosRng::seed_from_u64(seed);

    // Phase 1: GPU initialization (done once, amortized)
    let init_start = Instant::now();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No GPU adapter found");

    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 16,
        ..wgpu::Limits::default()
    };

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Profiling Device"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("Failed to create device");
    let init_time = init_start.elapsed();

    // Phase 2: Upload influence map buffers (done once)
    let upload_map_start = Instant::now();
    let create_buffer = |data: &[u32], label: &str| -> wgpu::Buffer {
        let data = if data.is_empty() { &[0u32] } else { data };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };

    let detector_offsets_x_buffer = create_buffer(&gpu_map.detector_offsets_x, "DetOffX");
    let detector_data_x_buffer = create_buffer(&gpu_map.detector_data_x, "DetDataX");
    let detector_offsets_y_buffer = create_buffer(&gpu_map.detector_offsets_y, "DetOffY");
    let detector_data_y_buffer = create_buffer(&gpu_map.detector_data_y, "DetDataY");
    let detector_offsets_z_buffer = create_buffer(&gpu_map.detector_offsets_z, "DetOffZ");
    let detector_data_z_buffer = create_buffer(&gpu_map.detector_data_z, "DetDataZ");
    let dem_output_offsets_x_buffer = create_buffer(&gpu_map.dem_output_offsets_x, "DemOutOffX");
    let dem_output_data_x_buffer = create_buffer(&gpu_map.dem_output_data_x, "DemOutDataX");
    let dem_output_offsets_y_buffer = create_buffer(&gpu_map.dem_output_offsets_y, "DemOutOffY");
    let dem_output_data_y_buffer = create_buffer(&gpu_map.dem_output_data_y, "DemOutDataY");
    let dem_output_offsets_z_buffer = create_buffer(&gpu_map.dem_output_offsets_z, "DemOutOffZ");
    let dem_output_data_z_buffer = create_buffer(&gpu_map.dem_output_data_z, "DemOutDataZ");
    let upload_map_time = upload_map_start.elapsed();

    // Phase 3: Create shader and pipeline (done once)
    let pipeline_start = Instant::now();
    let shader_source = include_str!("../src/influence_sampler_shader.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Profiling Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Profiling BindGroupLayout"),
        entries: &(0..16)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else if i < 14 {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Profiling PipelineLayout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Profiling Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let pipeline_time = pipeline_start.elapsed();

    // Phase 4: Create params buffer
    let params_start = Instant::now();
    let detector_words = gpu_map.num_detectors.div_ceil(32).max(1);
    let dem_output_words = gpu_map.num_dem_outputs.div_ceil(32).max(1);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // probability in [0,1] maps to [0, u32::MAX]
    let p_threshold = (p_error * f64::from(u32::MAX)) as u32;
    let params = SamplerParams {
        num_locations: gpu_map.num_locations,
        num_shots,
        num_detectors: gpu_map.num_detectors,
        num_dem_outputs: gpu_map.num_dem_outputs,
        p_error_threshold: p_threshold,
        detector_words,
        dem_output_words,
        _padding: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let params_time = params_start.elapsed();

    // Phase 5: Generate and upload random seeds
    let seeds_start = Instant::now();
    let seeds: Vec<u32> = (0..num_shots).map(|_| rng.next_u32()).collect();
    let seeds_gen_time = seeds_start.elapsed();

    let seeds_upload_start = Instant::now();
    let random_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Random Seeds"),
        contents: bytemuck::cast_slice(&seeds),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let seeds_upload_time = seeds_upload_start.elapsed();

    // Phase 6: Create output buffers
    let output_start = Instant::now();
    let detector_output_size = (num_shots as usize * detector_words as usize * 4) as u64;
    let dem_output_size = (num_shots as usize * dem_output_words as usize * 4) as u64;

    let detector_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Detector Output"),
        size: detector_output_size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dem_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("DEM Output"),
        size: dem_output_size.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_time = output_start.elapsed();

    // Phase 7: Create bind group
    let bind_start = Instant::now();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Profiling BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: detector_offsets_x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: detector_data_x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: detector_offsets_y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: detector_data_y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: detector_offsets_z_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: detector_data_z_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: dem_output_offsets_x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: dem_output_data_x_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: dem_output_offsets_y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: dem_output_data_y_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: dem_output_offsets_z_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: dem_output_data_z_buffer.as_entire_binding(),
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
                resource: dem_output_buffer.as_entire_binding(),
            },
        ],
    });
    let bind_time = bind_start.elapsed();

    // Phase 8: Dispatch compute shader
    let dispatch_start = Instant::now();
    let workgroups = num_shots.div_ceil(256);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Profiling Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Profiling Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    let dispatch_time = dispatch_start.elapsed();

    // Phase 9: Wait for GPU and read results
    let read_start = Instant::now();

    // Create staging buffers
    let detector_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Detector Staging"),
        size: detector_output_size.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let dem_output_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("DEM Output Staging"),
        size: dem_output_size.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(
        &detector_output_buffer,
        0,
        &detector_staging,
        0,
        detector_output_size.max(4),
    );
    encoder.copy_buffer_to_buffer(
        &dem_output_buffer,
        0,
        &dem_output_staging,
        0,
        dem_output_size.max(4),
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Map and read detector results
    let det_slice = detector_staging.slice(..);
    let (tx1, rx1) = std::sync::mpsc::channel();
    det_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx1.send(result).unwrap();
    });

    let dem_output_slice = dem_output_staging.slice(..);
    let (tx2, rx2) = std::sync::mpsc::channel();
    dem_output_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx2.send(result).unwrap();
    });

    let _ = device.poll(wgpu::PollType::wait_indefinitely());

    rx1.recv().unwrap().unwrap();
    rx2.recv().unwrap().unwrap();

    let det_data = det_slice.get_mapped_range();
    let _det_results: Vec<u32> = bytemuck::cast_slice(&det_data).to_vec();
    drop(det_data);

    let dem_output_data = dem_output_slice.get_mapped_range();
    let _dem_output_results: Vec<u32> = bytemuck::cast_slice(&dem_output_data).to_vec();
    drop(dem_output_data);

    let read_time = read_start.elapsed();

    GpuProfile {
        init_ms: init_time.as_secs_f64() * 1000.0,
        upload_map_ms: upload_map_time.as_secs_f64() * 1000.0,
        pipeline_ms: pipeline_time.as_secs_f64() * 1000.0,
        params_ms: params_time.as_secs_f64() * 1000.0,
        seeds_gen_ms: seeds_gen_time.as_secs_f64() * 1000.0,
        seeds_upload_ms: seeds_upload_time.as_secs_f64() * 1000.0,
        output_alloc_ms: output_time.as_secs_f64() * 1000.0,
        bind_group_ms: bind_time.as_secs_f64() * 1000.0,
        dispatch_ms: dispatch_time.as_secs_f64() * 1000.0,
        read_results_ms: read_time.as_secs_f64() * 1000.0,
        shots: num_shots as usize,
        #[allow(clippy::cast_possible_truncation)] // 64-bit target
        detector_output_bytes: detector_output_size as usize,
        #[allow(clippy::cast_possible_truncation)] // 64-bit target
        dem_output_bytes: dem_output_size as usize,
    }
}

struct GpuProfile {
    init_ms: f64,
    upload_map_ms: f64,
    pipeline_ms: f64,
    params_ms: f64,
    seeds_gen_ms: f64,
    seeds_upload_ms: f64,
    output_alloc_ms: f64,
    bind_group_ms: f64,
    dispatch_ms: f64,
    read_results_ms: f64,
    #[allow(dead_code)]
    shots: usize,
    detector_output_bytes: usize,
    dem_output_bytes: usize,
}

impl GpuProfile {
    fn total_ms(&self) -> f64 {
        self.init_ms
            + self.upload_map_ms
            + self.pipeline_ms
            + self.params_ms
            + self.seeds_gen_ms
            + self.seeds_upload_ms
            + self.output_alloc_ms
            + self.bind_group_ms
            + self.dispatch_ms
            + self.read_results_ms
    }

    fn one_time_ms(&self) -> f64 {
        self.init_ms + self.upload_map_ms + self.pipeline_ms
    }

    fn per_sample_ms(&self) -> f64 {
        self.params_ms
            + self.seeds_gen_ms
            + self.seeds_upload_ms
            + self.output_alloc_ms
            + self.bind_group_ms
            + self.dispatch_ms
            + self.read_results_ms
    }
}

#[allow(clippy::cast_precision_loss)] // profiling calculations use count as f64
fn main() {
    println!("CPU vs GPU Influence Sampler Profiling");
    println!("======================================\n");

    let p_error = 0.001;
    let seed = 42u64;
    let num_shots = 100_000u32;

    for distance in [5, 7, 9] {
        let num_rounds = 2 * distance;
        let circuit = build_surface_code_grid(distance, num_rounds);
        let num_data = distance * distance;

        // Build influence map
        let tracked_pauli_qubits: Vec<usize> = (0..num_data).collect();
        let builder = InfluenceBuilder::new(&circuit).with_z(&tracked_pauli_qubits);
        let influence_map = builder.build();
        let num_locations = influence_map.locations.len();

        // Export for GPU
        let (
            num_loc,
            num_det,
            num_dem_outputs,
            det_off_x,
            det_data_x,
            det_off_y,
            det_data_y,
            det_off_z,
            det_data_z,
            dem_output_offsets_x,
            dem_output_data_x,
            dem_output_offsets_y,
            dem_output_data_y,
            dem_output_offsets_z,
            dem_output_data_z,
        ) = influence_map.export_csr();

        let gpu_map = GpuInfluenceMapData::from_csr(
            num_loc,
            num_det,
            num_dem_outputs,
            det_off_x,
            det_data_x,
            det_off_y,
            det_data_y,
            det_off_z,
            det_data_z,
            dem_output_offsets_x,
            dem_output_data_x,
            dem_output_offsets_y,
            dem_output_data_y,
            dem_output_offsets_z,
            dem_output_data_z,
        );

        println!(
            "Surface code d={distance}, {num_rounds} rounds, {num_locations} locations, {num_shots} shots"
        );
        println!("{:-<70}", "");

        // CPU profile (DemSampler)
        let cpu = profile_cpu_sampler(&influence_map, p_error, seed, num_shots as usize);
        println!("\nCPU Pipeline (DemSampler):");
        println!("  Total time:            {:>10.2} ms", cpu.total_ms);
        println!(
            "  Per-shot:              {:>10.2} us",
            cpu.total_ms * 1000.0 / cpu.shots as f64
        );
        println!(
            "  Throughput:            {:>10.3} M shots/sec",
            cpu.shots as f64 / cpu.total_ms / 1000.0
        );

        // GPU profile
        let gpu = profile_gpu_sampler(&gpu_map, p_error, seed, num_shots);
        println!("\nGPU Pipeline:");
        println!("  One-time setup:");
        println!("    GPU init:            {:>10.2} ms", gpu.init_ms);
        println!("    Upload influence map:{:>10.2} ms", gpu.upload_map_ms);
        println!("    Create pipeline:     {:>10.2} ms", gpu.pipeline_ms);
        println!("    Subtotal (one-time): {:>10.2} ms", gpu.one_time_ms());
        println!("  Per-sample:");
        println!("    Create params:       {:>10.2} ms", gpu.params_ms);
        println!("    Generate seeds:      {:>10.2} ms", gpu.seeds_gen_ms);
        println!("    Upload seeds:        {:>10.2} ms", gpu.seeds_upload_ms);
        println!("    Alloc output bufs:   {:>10.2} ms", gpu.output_alloc_ms);
        println!("    Create bind group:   {:>10.2} ms", gpu.bind_group_ms);
        println!("    Dispatch + wait:     {:>10.2} ms", gpu.dispatch_ms);
        println!("    Read results:        {:>10.2} ms", gpu.read_results_ms);
        println!("    Subtotal (per-call): {:>10.2} ms", gpu.per_sample_ms());
        println!("  Total:                 {:>10.2} ms", gpu.total_ms());
        println!(
            "  Output size:           {:>10.2} KB (det) + {:.2} KB (DEM out)",
            gpu.detector_output_bytes as f64 / 1024.0,
            gpu.dem_output_bytes as f64 / 1024.0
        );

        println!("\nComparison:");
        println!("  CPU (DemSampler):      {:>10.2} ms", cpu.total_ms);
        println!("  GPU total (with init): {:>10.2} ms", gpu.total_ms());
        println!("  GPU per-call only:     {:>10.2} ms", gpu.per_sample_ms());
        let speedup_gpu_vs_cpu = cpu.total_ms / gpu.per_sample_ms();
        println!("  GPU vs CPU:            {speedup_gpu_vs_cpu:>10.1}x");

        println!("\n");
    }

    println!("Profiling complete!");
}
