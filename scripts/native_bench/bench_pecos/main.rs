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

//! Standalone PECOS benchmark using the same circuits and timing
//! methodology as `bench_quest.c`, for direct apples-to-apples comparison.
//!
//! CPU-only by default. Enable GPU backends with feature flags:
//!   --features gpu        (GpuStateVec via wgpu, f32)
//!   --features cuquantum  (CuStateVec via cuQuantum, f64)

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, DensityMatrix, QuantumSimulator, StateVecSoA,
};
use std::hint::black_box;
use std::time::Instant;

#[cfg(feature = "gpu")]
use pecos_gpu_sims::{GpuStateVec32, GpuStateVec64};

#[cfg(feature = "cuquantum")]
use pecos_cuquantum::CuStateVec;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

// ---------------------------------------------------------------------------
// Generic circuit runner for any simulator
// ---------------------------------------------------------------------------

fn run_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    let angle = Angle64::from_radians(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(angle, &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

/// Circuit using SXX two-qubit gates (tests RXX shader path)
fn run_sxx_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    let angle = Angle64::from_radians(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(angle, &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.sxx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

/// Circuit using RXX two-qubit gates (for RXX parallel path validation)
fn run_rxx_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    let angle = Angle64::from_radians(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(angle, &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.rxx(angle, &[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

/// Circuit using RYY two-qubit gates (for RYY parallel path validation)
fn run_ryy_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    let angle = Angle64::from_radians(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(angle, &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.ryy(angle, &[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

/// Circuit using CZ two-qubit gates (for CZ parallel path validation)
fn run_cz_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    let angle = Angle64::from_radians(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(angle, &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.cz(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

/// Circuit using RZZ two-qubit gates (for RZZ parallel path validation)
fn run_rzz_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    let angle = Angle64::from_radians(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(angle, &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.rzz(angle, &[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU StateVecSoA benchmarks
// ---------------------------------------------------------------------------

fn bench_circuit(
    num_qubits: usize,
    num_layers: usize,
    reps: usize,
    fusion: bool,
    parallel: bool,
) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(parallel);
    sim.set_fusion(fusion);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        let t0 = Instant::now();
        run_circuit(&mut sim, num_qubits, num_layers);
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    let tag = match (fusion, parallel) {
        (false, false) => "nofuse ",
        (true, false) => "fused  ",
        (false, true) => "nf+par ",
        (true, true) => "fu+par ",
    };
    let med = median(&mut times);
    println!("circuit  {num_qubits:2}q {num_layers:2}l  {tag}{med:12.3} us", med = med * 1e6);
}

fn bench_2q_circuit(
    label: &str,
    run: fn(&mut StateVecSoA, usize, usize),
    num_qubits: usize,
    num_layers: usize,
    reps: usize,
    parallel: bool,
) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(parallel);
    sim.set_fusion(true);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        let t0 = Instant::now();
        run(&mut sim, num_qubits, num_layers);
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    let tag = if parallel { "fu+par " } else { "fused  " };
    let med = median(&mut times);
    println!("{label}  {num_qubits:2}q {num_layers:2}l  {tag}{med:12.3} us", med = med * 1e6);
}

fn bench_gate_h(num_qubits: usize, iters: usize, reps: usize) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(false);
    sim.set_fusion(false);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        let t0 = Instant::now();
        for _ in 0..iters {
            for q in 0..num_qubits {
                sim.h(&[QubitId(q)]);
            }
        }
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    println!("gate     H        {med:12.3} us", med = median(&mut times) * 1e6);
}

fn bench_gate_x(num_qubits: usize, iters: usize, reps: usize) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(false);
    sim.set_fusion(false);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        let t0 = Instant::now();
        for _ in 0..iters {
            for q in 0..num_qubits {
                sim.x(&[QubitId(q)]);
            }
        }
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    println!("gate     X        {med:12.3} us", med = median(&mut times) * 1e6);
}

fn bench_gate_cx_pair(
    num_qubits: usize,
    c: usize,
    t_q: usize,
    iters: usize,
    reps: usize,
    parallel: bool,
) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(parallel);
    sim.set_fusion(false);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        let t0 = Instant::now();
        for _ in 0..iters {
            sim.cx(&[(QubitId(c), QubitId(t_q))]);
        }
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    let med = median(&mut times);
    let per_call_us = med * 1e6 / (iters as f64);
    let tag = if parallel { "par " } else { "ser " };
    println!("cx  N={num_qubits:2} ({c},{t_q}) {tag} per_call={per_call_us:10.2} us");
}

fn bench_gate_cx(num_qubits: usize, iters: usize, reps: usize) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(false);
    sim.set_fusion(false);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        let t0 = Instant::now();
        for _ in 0..iters {
            for q in 0..num_qubits - 1 {
                sim.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
        }
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    println!("gate     CX       {med:12.3} us", med = median(&mut times) * 1e6);
}

fn bench_gate_rz(num_qubits: usize, iters: usize, reps: usize) {
    let mut sim = StateVecSoA::new(num_qubits);
    sim.set_parallel(false);
    sim.set_fusion(false);
    let angle = Angle64::from_radians(0.1);
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        let t0 = Instant::now();
        for _ in 0..iters {
            for q in 0..num_qubits {
                sim.rz(angle, &[QubitId(q)]);
            }
        }
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    println!("gate     RZ       {med:12.3} us", med = median(&mut times) * 1e6);
}

// ---------------------------------------------------------------------------
// Density matrix benchmark
// ---------------------------------------------------------------------------

fn bench_dm_circuit(num_qubits: usize, num_layers: usize, reps: usize, parallel: bool) {
    let mut sim = DensityMatrix::new(num_qubits);
    if parallel {
        sim.state_vector_mut().set_parallel(true);
    }
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        let t0 = Instant::now();
        run_circuit(&mut sim, num_qubits, num_layers);
        black_box(&sim);
        *t = t0.elapsed().as_secs_f64();
    }

    let tag = if parallel { "par   " } else { "serial" };
    let med = median(&mut times);
    println!("dm_circ  {num_qubits:2}q {num_layers:2}l  {tag}  {med:12.3} us", med = med * 1e6);
}

// ---------------------------------------------------------------------------
// GPU benchmarks: GpuStateVec (wgpu, f32)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn bench_gpu_circuit(num_qubits: usize, num_layers: usize, reps: usize) {
    let mut sim = match GpuStateVec32::new(num_qubits as u32) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GpuStateVec({num_qubits}): {e}");
            return;
        }
    };
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        sim.sync();
        let t0 = Instant::now();
        run_circuit(&mut sim, num_qubits, num_layers);
        sim.sync();
        *t = t0.elapsed().as_secs_f64();
    }

    let med = median(&mut times);
    println!("circuit  {num_qubits:2}q {num_layers:2}l  {med:12.3} us", med = med * 1e6);
}

#[cfg(feature = "gpu")]
fn run_gpu_benchmarks(reps: usize) {
    println!();
    println!("=== PECOS GpuStateVec (wgpu, f32) standalone benchmarks ===");
    println!();
    println!("-- Layered circuits (median of {reps} runs) --");

    let configs = [
        (10, 20),
        (14, 20),
        (18, 20),
        (20, 20),
        (22, 20),
        (24, 10),
        (26, 5),
    ];

    for (nq, nl) in configs {
        bench_gpu_circuit(nq, nl, reps);
    }

    // Measurement-heavy circuit (tests CPU measurement fast path)
    println!();
    println!("-- Gate+Measure circuit (median of {reps} runs) --");
    for nq in [4, 8, 10] {
        let mut sim = match GpuStateVec32::new(nq as u32) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("GpuStateVec32({nq}): {e}");
                continue;
            }
        };
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            // 50 rounds of: H on all qubits, then measure all qubits
            for _round in 0..50 {
                for q in 0..nq {
                    sim.h(&[QubitId(q)]);
                }
                for q in 0..nq {
                    sim.mz(&[QubitId(q)]);
                }
            }
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        println!("mz_circ  {nq:2}q 50r  {med:12.3} us", med = med * 1e6);
    }

    // Check if small-state workloads are limited by workgroup size (thread utilization)
    println!();
    println!("-- Small-state gate throughput (N=4..10, 100 gates, {reps} runs) --");
    for nq in [4u32, 6, 8, 10] {
        let mut sim = GpuStateVec32::new(nq).unwrap();
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            for _iter in 0..50 {
                for i in 0..100 {
                    sim.h(&[QubitId((i % nq as usize))]);
                }
                let _s = sim.state();
            }
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        let per_iter_us = med * 1e6 / 50.0;
        let num_amps = 1u64 << nq;
        let threads_used = num_amps / 2; // num pairs
        let thread_util = (threads_used as f64 / 256.0 * 100.0).min(100.0);
        println!("N={nq:2}  amps={num_amps:4}  per_iter={per_iter_us:7.1}us  thread_util={thread_util:5.1}%");
    }

    // Persistent vs dispatched path: measure per-gate-count overhead
    println!();
    println!("-- Persistent kernel overhead vs gate count (N=10, persistent) ({reps} runs) --");
    let nq = 10;
    for num_gates in [1, 2, 5, 10, 20, 50, 100, 200] {
        let mut sim = GpuStateVec32::new(nq).unwrap();
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            for _iter in 0..50 {
                // Queue N gates
                for i in 0..num_gates {
                    sim.h(&[QubitId((i % nq as usize))]);
                }
                // Force flush by reading state
                let _s = sim.state();
            }
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        let per_iter_us = med * 1e6 / 50.0;
        let per_gate_us = per_iter_us / (num_gates as f64);
        println!("gates={num_gates:3}  per_iter={per_iter_us:7.1}us  per_gate={per_gate_us:5.2}us");
    }

    // Same gate count test at N=14 (above persistent_max_qubits on this GPU -> uses dispatched path)
    println!();
    println!("-- Dispatched path: gate count overhead (N=14, dispatched) ({reps} runs) --");
    let nq = 14;
    for num_gates in [1, 2, 5, 10, 20, 50, 100, 200] {
        let mut sim = GpuStateVec32::new(nq).unwrap();
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            for _iter in 0..50 {
                for i in 0..num_gates {
                    sim.h(&[QubitId((i % nq as usize))]);
                }
                let _s = sim.state();
            }
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        let per_iter_us = med * 1e6 / 50.0;
        let per_gate_us = per_iter_us / (num_gates as f64);
        println!("gates={num_gates:3}  per_iter={per_iter_us:7.1}us  per_gate={per_gate_us:5.2}us");
    }

    // Measure time for a bare state readback (for calibration cost analysis)
    println!();
    println!("-- Calibration cost: state readback time at various N --");
    for nq in [8, 10, 12, 14, 16, 18, 20] {
        let mut sim = GpuStateVec32::new(nq as u32).unwrap();
        sim.reset();
        sim.sync();
        // Warm up
        let _ = sim.state();
        let _ = sim.state();
        // Measure 20 readbacks
        let t0 = Instant::now();
        for _ in 0..20 {
            let _state = sim.state();
        }
        let total = t0.elapsed().as_secs_f64();
        let per_readback_us = total * 1e6 / 20.0;
        let state_bytes = (1u64 << nq) * 8;
        let effective_gbps = (state_bytes as f64) / (per_readback_us * 1e-6) / 1e9;
        println!("N={nq:2}  readback={per_readback_us:8.1}us  state_size={kb:7.1}KB  effective_bw={gbps:.2}GB/s",
            kb = state_bytes as f64 / 1024.0, gbps = effective_gbps);
    }

    // (N, M) grid benchmark: probe crossover surface for path selection.
    // Uses mz_gpu_sequential() and mz_cpu_batch() directly to bypass the
    // path-selection lookup table -- otherwise both columns would converge.
    println!();
    println!("-- (N, M) f32 batch vs sequential mz probe (median of {reps} runs, 10 rounds) --");
    for nq in [10, 12, 14, 16, 18, 20] {
        for &m in &[1, 2, 4, 8, nq / 2, nq] {
            let m = m.min(nq);
            if m == 0 { continue; }
            let ancillas: Vec<QubitId> = (0..m).map(QubitId).collect();

            // Sequential GPU path (forced)
            let mut sim = GpuStateVec32::new(nq as u32).unwrap();
            let mut times = vec![0.0_f64; reps];
            for t in &mut times {
                sim.reset();
                sim.sync();
                let t0 = Instant::now();
                for _round in 0..10 {
                    for q in 0..nq { sim.h(&[QubitId(q)]); }
                    sim.mz_gpu_sequential(&ancillas);
                }
                sim.sync();
                *t = t0.elapsed().as_secs_f64();
            }
            let seq_med = median(&mut times);

            // CPU batch path (forced)
            let mut sim = GpuStateVec32::new(nq as u32).unwrap();
            let mut times = vec![0.0_f64; reps];
            for t in &mut times {
                sim.reset();
                sim.sync();
                let t0 = Instant::now();
                for _round in 0..10 {
                    for q in 0..nq { sim.h(&[QubitId(q)]); }
                    sim.mz_cpu_batch(&ancillas);
                }
                sim.sync();
                *t = t0.elapsed().as_secs_f64();
            }
            let batch_med = median(&mut times);

            let ratio = seq_med / batch_med;
            let winner = if ratio > 1.0 { "BATCH" } else { "SEQ  " };
            println!(
                "N={nq:2} M={m:2}  seq={seq:10.1}us  batch={bat:10.1}us  ratio={ratio:.2}  {winner}",
                seq = seq_med * 1e6,
                bat = batch_med * 1e6,
            );
        }
    }

    // Same probe for f64 -- transfers and CPU loops are 2x f32 due to wider amps.
    println!();
    println!("-- (N, M) f64 batch vs sequential mz probe (median of {reps} runs, 10 rounds) --");
    for nq in [10, 12, 14, 16, 18, 20] {
        for &m in &[1, 2, 4, 8, nq / 2, nq] {
            let m = m.min(nq);
            if m == 0 { continue; }
            let ancillas: Vec<QubitId> = (0..m).map(QubitId).collect();

            let mut sim = GpuStateVec64::new(nq as u32).unwrap();
            let mut times = vec![0.0_f64; reps];
            for t in &mut times {
                sim.reset();
                sim.sync();
                let t0 = Instant::now();
                for _round in 0..10 {
                    for q in 0..nq { sim.h(&[QubitId(q)]); }
                    sim.mz_gpu_sequential(&ancillas);
                }
                sim.sync();
                *t = t0.elapsed().as_secs_f64();
            }
            let seq_med = median(&mut times);

            let mut sim = GpuStateVec64::new(nq as u32).unwrap();
            let mut times = vec![0.0_f64; reps];
            for t in &mut times {
                sim.reset();
                sim.sync();
                let t0 = Instant::now();
                for _round in 0..10 {
                    for q in 0..nq { sim.h(&[QubitId(q)]); }
                    sim.mz_cpu_batch(&ancillas);
                }
                sim.sync();
                *t = t0.elapsed().as_secs_f64();
            }
            let batch_med = median(&mut times);

            let ratio = seq_med / batch_med;
            let winner = if ratio > 1.0 { "BATCH" } else { "SEQ  " };
            println!(
                "N={nq:2} M={m:2}  seq={seq:10.1}us  batch={bat:10.1}us  ratio={ratio:.2}  {winner}",
                seq = seq_med * 1e6,
                bat = batch_med * 1e6,
            );
        }
    }

    // QEC-pattern benchmark: block of gates + batch measurements + conditionals
    println!();
    println!("-- QEC-pattern: gates + mz + conditional gates (median of {reps} runs) --");
    for nq in [10, 14, 18] {
        let ancillas: Vec<QubitId> = (0..nq/2).map(QubitId).collect();
        let mut sim = GpuStateVec32::new(nq as u32).unwrap();
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            for _round in 0..20 {
                // Block of gates
                for q in 0..nq { sim.h(&[QubitId(q)]); }
                for q in 0..nq-1 { sim.cx(&[(QubitId(q), QubitId(q+1))]); }
                // Batch measure ancillas
                let outcomes = sim.mz(&ancillas);
                // Conditional gates based on outcome (simulate classical logic)
                for (i, r) in outcomes.iter().enumerate() {
                    if r.outcome {
                        sim.z(&[QubitId(i)]);
                    }
                }
            }
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        println!("qec_batch{nq:2}q 20r  {med:12.3} us", med = med * 1e6);

        // Sequential mz (per-ancilla) for comparison
        let mut sim = GpuStateVec32::new(nq as u32).unwrap();
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            for _round in 0..20 {
                for q in 0..nq { sim.h(&[QubitId(q)]); }
                for q in 0..nq-1 { sim.cx(&[(QubitId(q), QubitId(q+1))]); }
                for a in &ancillas {
                    let outcomes = sim.mz(&[*a]);
                    if outcomes[0].outcome {
                        sim.z(&[QubitId(a.0)]);
                    }
                }
            }
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        println!("qec_seq  {nq:2}q 20r  {med:12.3} us", med = med * 1e6);
    }

    // Batch measurement (measure all qubits at once vs one at a time)
    println!();
    println!("-- Batch vs sequential mz (median of {reps} runs) --");
    for nq in [4, 8, 10] {
        let all_qubits: Vec<QubitId> = (0..nq).map(QubitId).collect();
        // Sequential: mz one qubit at a time
        {
            let mut sim = GpuStateVec32::new(nq as u32).unwrap();
            let mut times = vec![0.0_f64; reps];
            for t in &mut times {
                sim.reset();
                sim.sync();
                let t0 = Instant::now();
                for _round in 0..50 {
                    for q in 0..nq { sim.h(&[QubitId(q)]); }
                    for q in 0..nq { sim.mz(&[QubitId(q)]); }
                }
                sim.sync();
                *t = t0.elapsed().as_secs_f64();
            }
            let med = median(&mut times);
            println!("mz_seq   {nq:2}q 50r  {med:12.3} us", med = med * 1e6);
        }
        // Batch: mz all qubits at once
        {
            let mut sim = GpuStateVec32::new(nq as u32).unwrap();
            let mut times = vec![0.0_f64; reps];
            for t in &mut times {
                sim.reset();
                sim.sync();
                let t0 = Instant::now();
                for _round in 0..50 {
                    for q in 0..nq { sim.h(&[QubitId(q)]); }
                    sim.mz(&all_qubits);
                }
                sim.sync();
                *t = t0.elapsed().as_secs_f64();
            }
            let med = median(&mut times);
            println!("mz_batch {nq:2}q 50r  {med:12.3} us", med = med * 1e6);
        }
    }

    // SXX circuit (tests RXX shader vs decomposition)
    println!();
    println!("-- SXX circuit (median of {reps} runs) --");
    for (nq, nl) in [(10, 20), (18, 20)] {
        let mut sim = match GpuStateVec32::new(nq as u32) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("GpuStateVec32({nq}): {e}");
                continue;
            }
        };
        let mut times = vec![0.0_f64; reps];
        for t in &mut times {
            sim.reset();
            sim.sync();
            let t0 = Instant::now();
            run_sxx_circuit(&mut sim, nq, nl);
            sim.sync();
            *t = t0.elapsed().as_secs_f64();
        }
        let med = median(&mut times);
        println!("sxx_circ {nq:2}q {nl:2}l  {med:12.3} us", med = med * 1e6);
    }
}

// ---------------------------------------------------------------------------
// GPU benchmarks: GpuStateVec64 (wgpu, f64)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn bench_gpu64_circuit(num_qubits: usize, num_layers: usize, reps: usize) {
    let mut sim = match GpuStateVec64::new(num_qubits as u32) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GpuStateVec64({num_qubits}): {e}");
            return;
        }
    };
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        sim.sync();
        let t0 = Instant::now();
        run_circuit(&mut sim, num_qubits, num_layers);
        sim.sync();
        *t = t0.elapsed().as_secs_f64();
    }

    let med = median(&mut times);
    println!("circuit  {num_qubits:2}q {num_layers:2}l  {med:12.3} us", med = med * 1e6);
}

#[cfg(feature = "gpu")]
fn run_gpu64_benchmarks(reps: usize) {
    println!();
    println!("=== PECOS GpuStateVec64 (wgpu, f64) standalone benchmarks ===");
    println!();
    println!("-- Layered circuits (median of {reps} runs) --");

    let configs = [
        (10, 20),
        (14, 20),
        (18, 20),
        (20, 20),
        (22, 20),
        (24, 10),
        (26, 5),
    ];

    for (nq, nl) in configs {
        bench_gpu64_circuit(nq, nl, reps);
    }

    println!();
    println!("=== PECOS GpuDensityMatrix (Choi on GpuStateVec32) benchmarks ===");
    println!();
    println!("-- Density matrix: layered circuits (median of {reps} runs) --");

    let dm_configs = [(6, 20), (8, 20), (10, 20), (12, 10), (13, 5)];
    for (nq, nl) in dm_configs {
        bench_gpu_dm_circuit(nq, nl, reps);
    }
}

#[cfg(feature = "gpu")]
fn bench_gpu_dm_circuit(num_qubits: usize, num_layers: usize, reps: usize) {
    use pecos_gpu_sims::GpuDensityMatrix32;
    let mut sim = match GpuDensityMatrix32::new(num_qubits) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("GpuDensityMatrix({num_qubits}): {e}");
            return;
        }
    };
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        sim.sync();
        let t0 = Instant::now();
        run_circuit(&mut sim, num_qubits, num_layers);
        sim.sync();
        *t = t0.elapsed().as_secs_f64();
    }

    let med = median(&mut times);
    println!("dm_circ  {num_qubits:2}q {num_layers:2}l  {med:12.3} us", med = med * 1e6);
}

// ---------------------------------------------------------------------------
// GPU benchmarks: CuStateVec (cuQuantum, f64)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuquantum")]
fn bench_cuquantum_circuit(num_qubits: usize, num_layers: usize, reps: usize) {
    let mut sim = match CuStateVec::new(num_qubits) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("CuStateVec({num_qubits}): {e}");
            return;
        }
    };
    let mut times = vec![0.0_f64; reps];

    for t in &mut times {
        sim.reset();
        sim.sync();
        let t0 = Instant::now();
        run_circuit(&mut sim, num_qubits, num_layers);
        sim.sync();
        *t = t0.elapsed().as_secs_f64();
    }

    let med = median(&mut times);
    println!("circuit  {num_qubits:2}q {num_layers:2}l  {med:12.3} us", med = med * 1e6);
}

#[cfg(feature = "cuquantum")]
fn run_cuquantum_benchmarks(reps: usize) {
    println!();
    println!("=== PECOS CuStateVec (cuQuantum, f64) standalone benchmarks ===");
    println!();
    println!("-- Layered circuits (median of {reps} runs) --");

    let configs = [
        (10, 20),
        (14, 20),
        (18, 20),
        (20, 20),
        (22, 20),
        (24, 10),
        (26, 5),
    ];

    for (nq, nl) in configs {
        bench_cuquantum_circuit(nq, nl, reps);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let reps = 5;

    println!("=== PECOS StateVecSoA standalone benchmarks ===");
    println!();
    println!("-- Layered circuits (median of {reps} runs) --");

    let configs = [(10, 20), (14, 20), (18, 20), (20, 20), (22, 10), (24, 5)];

    for (num_qubits, num_layers) in configs {
        bench_circuit(num_qubits, num_layers, reps, false, false);
        bench_circuit(num_qubits, num_layers, reps, true, false);
        bench_circuit(num_qubits, num_layers, reps, false, true);
        bench_circuit(num_qubits, num_layers, reps, true, true);
    }

    println!();
    println!("-- CX scalar-path overhead at various N (low-qubit scalar fallback) --");
    for n in [18, 20, 22] {
        bench_gate_cx_pair(n, 0, 1, 20, reps, false);
        bench_gate_cx_pair(n, 0, 1, 20, reps, true);
        bench_gate_cx_pair(n, 2, 3, 20, reps, false);
        bench_gate_cx_pair(n, 2, 3, 20, reps, true);
    }

    println!();
    println!("-- 2-qubit gate circuits: fused vs fused+parallel at 22q 10l (median of {reps} runs) --");
    for (label, run) in [
        ("cz_circ ", run_cz_circuit::<StateVecSoA> as fn(&mut StateVecSoA, usize, usize)),
        ("rzz_circ", run_rzz_circuit::<StateVecSoA>),
        ("rxx_circ", run_rxx_circuit::<StateVecSoA>),
        ("ryy_circ", run_ryy_circuit::<StateVecSoA>),
    ] {
        bench_2q_circuit(label, run, 22, 10, reps, false);
        bench_2q_circuit(label, run, 22, 10, reps, true);
    }

    println!();
    println!("-- Individual gates at 18 qubits, 100 iters (median of {reps} runs) --");
    bench_gate_h(18, 100, reps);
    bench_gate_x(18, 100, reps);
    bench_gate_cx(18, 100, reps);
    bench_gate_rz(18, 100, reps);

    println!();
    println!("-- Density matrix: layered circuits (median of {reps} runs) --");

    let dm_configs = [(6, 20), (8, 20), (10, 20), (12, 10), (13, 5)];

    for (num_qubits, num_layers) in dm_configs {
        bench_dm_circuit(num_qubits, num_layers, reps, false);
        bench_dm_circuit(num_qubits, num_layers, reps, true);
    }

    #[cfg(feature = "gpu")]
    run_gpu_benchmarks(reps);

    #[cfg(feature = "gpu")]
    run_gpu64_benchmarks(reps);

    #[cfg(feature = "cuquantum")]
    run_cuquantum_benchmarks(reps);
}
