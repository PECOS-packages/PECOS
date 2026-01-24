// Analysis: Would caching help with lazy gate fusion?
//
// Current implementation: Matrix multiplication happens incrementally
//   H -> pending = H
//   SZ -> pending = SZ * H (1 matrix multiply)
//   H -> pending = H * SZ * H (1 matrix multiply)
//   Total: 2 matrix multiplies for 3 gates
//
// With caching by sequence [H, SZ, H]:
//   - Need to track gate sequence (memory)
//   - Need hash lookup (time)
//   - Save matrix multiplications on cache hit

use std::collections::HashMap;
use std::time::Instant;

// Simplified 2x2 complex matrix (same as in StateVecSoA)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Matrix2x2 {
    data: [f64; 8], // a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im
}

impl Matrix2x2 {
    fn mul(&self, other: &Self) -> Self {
        let (a1_re, a1_im) = (self.data[0], self.data[1]);
        let (b1_re, b1_im) = (self.data[2], self.data[3]);
        let (c1_re, c1_im) = (self.data[4], self.data[5]);
        let (d1_re, d1_im) = (self.data[6], self.data[7]);

        let (a2_re, a2_im) = (other.data[0], other.data[1]);
        let (b2_re, b2_im) = (other.data[2], other.data[3]);
        let (c2_re, c2_im) = (other.data[4], other.data[5]);
        let (d2_re, d2_im) = (other.data[6], other.data[7]);

        Self {
            data: [
                a1_re * a2_re - a1_im * a2_im + b1_re * c2_re - b1_im * c2_im,
                a1_re * a2_im + a1_im * a2_re + b1_re * c2_im + b1_im * c2_re,
                a1_re * b2_re - a1_im * b2_im + b1_re * d2_re - b1_im * d2_im,
                a1_re * b2_im + a1_im * b2_re + b1_re * d2_im + b1_im * d2_re,
                c1_re * a2_re - c1_im * a2_im + d1_re * c2_re - d1_im * c2_im,
                c1_re * a2_im + c1_im * a2_re + d1_re * c2_im + d1_im * c2_re,
                c1_re * b2_re - c1_im * b2_im + d1_re * d2_re - d1_im * d2_im,
                c1_re * b2_im + c1_im * b2_re + d1_re * d2_im + d1_im * d2_re,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GateType {
    H,
    X,
    Y,
    Z,
    SZ,
    SZDG,
    SX,
}

fn gate_matrix(g: GateType) -> Matrix2x2 {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    match g {
        GateType::H => Matrix2x2 {
            data: [inv_sqrt2, 0.0, inv_sqrt2, 0.0, inv_sqrt2, 0.0, -inv_sqrt2, 0.0],
        },
        GateType::X => Matrix2x2 {
            data: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        },
        GateType::Y => Matrix2x2 {
            data: [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        },
        GateType::Z => Matrix2x2 {
            data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        },
        GateType::SZ => Matrix2x2 {
            data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        GateType::SZDG => Matrix2x2 {
            data: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        },
        GateType::SX => Matrix2x2 {
            data: [0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5],
        },
    }
}

// Approach 1: No caching (current implementation)
fn fuse_no_cache(gates: &[GateType]) -> Matrix2x2 {
    let mut result = gate_matrix(gates[0]);
    for &g in &gates[1..] {
        result = gate_matrix(g).mul(&result);
    }
    result
}

// Approach 2: Cache by gate sequence
fn fuse_with_cache(gates: &[GateType], cache: &mut HashMap<Vec<GateType>, Matrix2x2>) -> Matrix2x2 {
    if let Some(&cached) = cache.get(gates) {
        return cached;
    }
    let result = fuse_no_cache(gates);
    cache.insert(gates.to_vec(), result);
    result
}

// Approach 3: Cache with small fixed-size key (for sequences up to 8 gates)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SmallGateSeq {
    gates: [u8; 8],
    len: u8,
}

impl SmallGateSeq {
    fn from_gates(gates: &[GateType]) -> Option<Self> {
        if gates.len() > 8 {
            return None;
        }
        let mut seq = SmallGateSeq {
            gates: [0; 8],
            len: gates.len() as u8,
        };
        for (i, &g) in gates.iter().enumerate() {
            seq.gates[i] = g as u8;
        }
        Some(seq)
    }
}

fn fuse_with_small_cache(
    gates: &[GateType],
    cache: &mut HashMap<SmallGateSeq, Matrix2x2>,
) -> Matrix2x2 {
    if let Some(key) = SmallGateSeq::from_gates(gates) {
        if let Some(&cached) = cache.get(&key) {
            return cached;
        }
        let result = fuse_no_cache(gates);
        cache.insert(key, result);
        result
    } else {
        fuse_no_cache(gates)
    }
}

fn main() {
    println!("Gate Fusion Caching Analysis");
    println!("============================\n");

    // Test patterns that might benefit from caching
    let patterns: Vec<(&str, Vec<GateType>)> = vec![
        ("H-SZ-H (common)", vec![GateType::H, GateType::SZ, GateType::H]),
        ("H-Z-H", vec![GateType::H, GateType::Z, GateType::H]),
        ("SZ-H-SZ", vec![GateType::SZ, GateType::H, GateType::SZ]),
        ("H-SX-H-SZ", vec![GateType::H, GateType::SX, GateType::H, GateType::SZ]),
    ];

    println!("Cost Analysis (per fusion operation):");
    println!("--------------------------------------");
    println!("Matrix multiply: ~64 floating point ops");
    println!("HashMap lookup: ~20-50 ops (hash + compare)");
    println!("Vec allocation: ~100+ ops (for cache key)\n");

    let iterations = 1_000_000;
    let num_qubits = 16;

    println!("Benchmark: {} iterations, simulating {} qubits\n", iterations, num_qubits);

    for (name, pattern) in &patterns {
        let num_multiplies = pattern.len() - 1;

        println!("Pattern: {} ({} gates, {} matrix multiplies)", name, pattern.len(), num_multiplies);

        // Approach 1: No caching
        let start = Instant::now();
        let mut dummy = 0.0;
        for _ in 0..iterations {
            for _ in 0..num_qubits {
                let m = fuse_no_cache(pattern);
                dummy += m.data[0];
            }
        }
        let no_cache_time = start.elapsed();
        println!("  No cache:        {:>8.2}ms (dummy: {:.2})", no_cache_time.as_secs_f64() * 1000.0, dummy);

        // Approach 2: HashMap cache (new cache each "circuit")
        let start = Instant::now();
        let mut dummy = 0.0;
        for _ in 0..iterations {
            let mut cache = HashMap::new();
            for _ in 0..num_qubits {
                let m = fuse_with_cache(pattern, &mut cache);
                dummy += m.data[0];
            }
        }
        let cache_new_time = start.elapsed();
        let speedup = no_cache_time.as_nanos() as f64 / cache_new_time.as_nanos() as f64;
        println!("  Cache (new/iter): {:>8.2}ms ({:.2}x)", cache_new_time.as_secs_f64() * 1000.0, speedup);

        // Approach 3: HashMap cache (persistent across circuits)
        let start = Instant::now();
        let mut dummy = 0.0;
        let mut cache = HashMap::new();
        for _ in 0..iterations {
            for _ in 0..num_qubits {
                let m = fuse_with_cache(pattern, &mut cache);
                dummy += m.data[0];
            }
        }
        let cache_persist_time = start.elapsed();
        let speedup = no_cache_time.as_nanos() as f64 / cache_persist_time.as_nanos() as f64;
        println!("  Cache (persist):  {:>8.2}ms ({:.2}x)", cache_persist_time.as_secs_f64() * 1000.0, speedup);

        // Approach 4: Small fixed-size cache key
        let start = Instant::now();
        let mut dummy = 0.0;
        let mut cache = HashMap::new();
        for _ in 0..iterations {
            for _ in 0..num_qubits {
                let m = fuse_with_small_cache(pattern, &mut cache);
                dummy += m.data[0];
            }
        }
        let small_cache_time = start.elapsed();
        let speedup = no_cache_time.as_nanos() as f64 / small_cache_time.as_nanos() as f64;
        println!("  Small key cache:  {:>8.2}ms ({:.2}x)\n", small_cache_time.as_secs_f64() * 1000.0, speedup);
    }

    println!("Analysis:");
    println!("---------");
    println!("1. Caching helps when the SAME gate sequence is applied to MULTIPLE qubits");
    println!("2. Cache overhead (hash + allocation) can exceed matrix multiply cost for short sequences");
    println!("3. Persistent cache across circuit executions helps most");
    println!("4. Fixed-size keys avoid allocation overhead\n");

    println!("Recommendation:");
    println!("---------------");
    println!("For the current lazy fusion implementation:");
    println!("- Short sequences (2-3 gates): Caching overhead likely exceeds benefit");
    println!("- Long sequences (4+ gates): Caching could help, but rare in practice");
    println!("- Better approach: Use precomputed fused gates (hz, hs, etc.) for common patterns");
}

// Additional test: At what sequence length does caching break even?
