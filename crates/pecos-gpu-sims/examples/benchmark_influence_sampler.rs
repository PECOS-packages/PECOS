//! Benchmark comparing CPU vs GPU influence map sampling
//!
//! Run with: cargo run --example `benchmark_influence_sampler` --release

use pecos_gpu_sims::{GpuInfluenceMapData, GpuInfluenceSampler};
use std::time::Instant;

#[allow(clippy::cast_possible_truncation)] // benchmark indices fit in u32
fn create_test_influence_map(num_locations: usize, num_detectors: usize) -> GpuInfluenceMapData {
    // Each location i: X fault -> detector i % num_detectors
    // This creates a realistic sparse pattern
    let mut detector_offsets_x = Vec::with_capacity(num_locations + 1);
    let mut detector_data_x = Vec::with_capacity(num_locations);
    detector_offsets_x.push(0);
    for i in 0..num_locations {
        detector_data_x.push((i % num_detectors) as u32);
        detector_offsets_x.push((i + 1) as u32);
    }

    // Y faults: every 3rd location affects detector (i+1) % num_detectors
    let mut detector_offsets_y = Vec::with_capacity(num_locations + 1);
    let mut detector_data_y = Vec::new();
    detector_offsets_y.push(0);
    for i in 0..num_locations {
        if i % 3 == 0 {
            detector_data_y.push(((i + 1) % num_detectors) as u32);
        }
        detector_offsets_y.push(detector_data_y.len() as u32);
    }

    // Z faults: every 2nd location affects detector (i+2) % num_detectors
    let mut detector_offsets_z = Vec::with_capacity(num_locations + 1);
    let mut detector_data_z = Vec::new();
    detector_offsets_z.push(0);
    for i in 0..num_locations {
        if i % 2 == 0 {
            detector_data_z.push(((i + 2) % num_detectors) as u32);
        }
        detector_offsets_z.push(detector_data_z.len() as u32);
    }

    GpuInfluenceMapData {
        num_locations: num_locations as u32,
        num_detectors: num_detectors as u32,
        num_logicals: 2,
        detector_offsets_x,
        detector_data_x,
        detector_offsets_y,
        detector_data_y,
        detector_offsets_z,
        detector_data_z,
        // Logicals: location 0 X -> log 0, location 1 X -> log 1
        logical_offsets_x: {
            let mut v = vec![0u32; num_locations + 1];
            if num_locations > 0 {
                v[1] = 1;
            }
            if num_locations > 1 {
                v[2] = 2;
            }
            for vi in &mut v[3..=num_locations] {
                *vi = 2;
            }
            v
        },
        logical_data_x: vec![0, 1],
        logical_offsets_y: vec![0; num_locations + 1],
        logical_data_y: vec![],
        logical_offsets_z: vec![0; num_locations + 1],
        logical_data_z: vec![],
    }
}

/// Simple CPU sampler for comparison (mirrors the pecos-qec `NoisySampler` logic)
struct CpuSampler {
    num_locations: usize,
    num_detectors: usize,
    num_logicals: usize,
    detector_offsets_x: Vec<u32>,
    detector_data_x: Vec<u32>,
    detector_offsets_y: Vec<u32>,
    detector_data_y: Vec<u32>,
    detector_offsets_z: Vec<u32>,
    detector_data_z: Vec<u32>,
    logical_offsets_x: Vec<u32>,
    logical_data_x: Vec<u32>,
    rng_state: u64,
}

impl CpuSampler {
    fn new(map: &GpuInfluenceMapData, seed: u64) -> Self {
        Self {
            num_locations: map.num_locations as usize,
            num_detectors: map.num_detectors as usize,
            num_logicals: map.num_logicals as usize,
            detector_offsets_x: map.detector_offsets_x.clone(),
            detector_data_x: map.detector_data_x.clone(),
            detector_offsets_y: map.detector_offsets_y.clone(),
            detector_data_y: map.detector_data_y.clone(),
            detector_offsets_z: map.detector_offsets_z.clone(),
            detector_data_z: map.detector_data_z.clone(),
            logical_offsets_x: map.logical_offsets_x.clone(),
            logical_data_x: map.logical_data_x.clone(),
            rng_state: seed,
        }
    }

    fn next_u32(&mut self) -> u32 {
        // Simple xorshift for fast RNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        #[allow(clippy::cast_possible_truncation)] // intentional low-32-bit extraction
        {
            self.rng_state as u32
        }
    }

    fn sample(&mut self, num_shots: usize, p_error: f64) -> usize {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        // probability in [0,1] maps to [0, u32::MAX]
        let threshold = (p_error * f64::from(u32::MAX)) as u32;
        let mut logical_errors = 0;

        for _ in 0..num_shots {
            let mut detector_flips = vec![0u8; self.num_detectors.max(1)];
            let mut logical_flips = vec![0u8; self.num_logicals.max(1)];

            for loc in 0..self.num_locations {
                let rand = self.next_u32();
                if rand >= threshold {
                    continue;
                }

                // Sample Pauli type
                let pauli = self.next_u32() % 3;

                // Get affected detectors and logicals
                let (det_start, det_end, det_data) = match pauli {
                    0 => (
                        self.detector_offsets_x[loc] as usize,
                        self.detector_offsets_x[loc + 1] as usize,
                        &self.detector_data_x,
                    ),
                    1 => (
                        self.detector_offsets_y[loc] as usize,
                        self.detector_offsets_y[loc + 1] as usize,
                        &self.detector_data_y,
                    ),
                    _ => (
                        self.detector_offsets_z[loc] as usize,
                        self.detector_offsets_z[loc + 1] as usize,
                        &self.detector_data_z,
                    ),
                };

                for &det_val in &det_data[det_start..det_end] {
                    let det_idx = det_val as usize;
                    if det_idx < detector_flips.len() {
                        detector_flips[det_idx] ^= 1;
                    }
                }

                // Only X affects logicals in our test map
                if pauli == 0 {
                    let log_start = self.logical_offsets_x[loc] as usize;
                    let log_end = self.logical_offsets_x[loc + 1] as usize;
                    for i in log_start..log_end {
                        let log_idx = self.logical_data_x[i] as usize;
                        if log_idx < logical_flips.len() {
                            logical_flips[log_idx] ^= 1;
                        }
                    }
                }
            }

            if logical_flips.contains(&1) {
                logical_errors += 1;
            }
        }

        logical_errors
    }
}

fn benchmark_gpu(
    map: &GpuInfluenceMapData,
    num_shots: u32,
    p_error: f64,
    seed: u64,
) -> (std::time::Duration, usize) {
    let mut sampler = GpuInfluenceSampler::new(map, seed).expect("Failed to create GPU sampler");

    let start = Instant::now();
    let result = sampler.sample_uniform(num_shots, p_error);
    let elapsed = start.elapsed();

    let logical_errors = result.count_logical_errors();
    (elapsed, logical_errors)
}

fn benchmark_cpu(
    map: &GpuInfluenceMapData,
    num_shots: usize,
    p_error: f64,
    seed: u64,
) -> (std::time::Duration, usize) {
    let mut sampler = CpuSampler::new(map, seed);

    let start = Instant::now();
    let logical_errors = sampler.sample(num_shots, p_error);
    let elapsed = start.elapsed();

    (elapsed, logical_errors)
}

fn main() {
    println!("Influence Map Sampler Benchmark: CPU vs GPU\n");
    println!("{:=<70}", "");

    // Note: GPU has a limit of 65535 workgroups per dimension
    // Total work items = num_locations * ceil(num_shots/32)
    // Max supported: ~65535 * 256 * 32 = ~537M work items
    let configs = [
        (100, 50, 10_000),    // Small: 100 locations, 50 detectors, 10k shots
        (500, 100, 50_000),   // Medium: 500 locations, 100 detectors, 50k shots
        (1000, 200, 100_000), // Large: 1000 locations, 200 detectors, 100k shots
        (2000, 400, 200_000), // XL: 2000 locations, 400 detectors, 200k shots
        (5000, 500, 100_000), // XXL: 5000 locations, 500 detectors, 100k shots (high location count)
    ];

    let p_error = 0.001; // 0.1% error rate
    let seed = 42u64;

    println!(
        "{:<12} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "Config", "Locations", "Detectors", "Shots", "CPU Time", "GPU Time"
    );
    println!("{:-<70}", "");

    for (num_locations, num_detectors, num_shots) in configs {
        let map = create_test_influence_map(num_locations, num_detectors);

        // Warm up GPU
        let mut warmup = GpuInfluenceSampler::new(&map, seed).expect("Failed to create sampler");
        let _ = warmup.sample_uniform(100, p_error);

        // Benchmark
        let (cpu_time, _cpu_errors) = benchmark_cpu(&map, num_shots, p_error, seed);
        #[allow(clippy::cast_possible_truncation)] // benchmark shot count fits in u32
        let (gpu_time, _gpu_errors) = benchmark_gpu(&map, num_shots as u32, p_error, seed);

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<12} {:>10} {:>12} {:>12} {:>10.2}ms {:>10.2}ms  ({:.1}x)",
            format!("{}x{}", num_locations, num_shots),
            num_locations,
            num_detectors,
            num_shots,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }

    println!("{:=<70}", "");
    println!("\nNote: GPU includes data transfer overhead. Speedup increases with scale.");
}
