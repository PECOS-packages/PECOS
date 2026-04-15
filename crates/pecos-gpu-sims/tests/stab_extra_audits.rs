// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

//! Audits for `GpuStab` compile-circuit path and `GpuStab` / `GpuStabMulti`
//! mid-circuit measurement queues against CPU `SparseStab` reference.

use pecos_core::QubitId;
use pecos_gpu_sims::{CompiledGate, DefaultGpuStab, GateType, GpuStabMulti};
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, QuantumSimulator, SparseStab};

// ---------------------------------------------------------------------------
// compile_circuit replay: compiled dispatch should match normal per-gate
// dispatch exactly on the same deterministic circuit.
// ---------------------------------------------------------------------------

/// Build an equivalent (`CompiledGate`, trait-call) pair for each op so we can
/// apply the same circuit through either dispatch path.
#[derive(Clone, Copy)]
#[allow(dead_code)] // X and Cz reserved for expanded test coverage
enum Op {
    H(u32),
    X(u32),
    S(u32),
    Cx(u32, u32),
    Cz(u32, u32),
}

fn to_compiled(op: Op) -> CompiledGate {
    match op {
        Op::H(t) => CompiledGate::h(t),
        Op::X(t) => CompiledGate::x(t),
        Op::S(t) => CompiledGate::s(t),
        Op::Cx(c, t) => CompiledGate {
            gate_type: GateType::Cx,
            target: t,
            control: Some(c),
        },
        Op::Cz(c, t) => CompiledGate {
            gate_type: GateType::Cz,
            target: t,
            control: Some(c),
        },
    }
}

fn apply_trait<S: CliffordGateable>(sim: &mut S, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q as usize)]);
        }
        Op::X(q) => {
            sim.x(&[QubitId(q as usize)]);
        }
        Op::S(q) => {
            sim.sz(&[QubitId(q as usize)]);
        }
        Op::Cx(c, t) => {
            sim.cx(&[(QubitId(c as usize), QubitId(t as usize))]);
        }
        Op::Cz(c, t) => {
            sim.cz(&[(QubitId(c as usize), QubitId(t as usize))]);
        }
    }
}

#[test]
fn compile_circuit_matches_normal_path() {
    let Ok(mut gpu_compiled) = DefaultGpuStab::with_seed(6, 42) else {
        return;
    };
    let Ok(mut gpu_normal) = DefaultGpuStab::with_seed(6, 42) else {
        return;
    };

    // A Bell-state-like deterministic Clifford circuit: all measurements
    // forced by the state.
    let ops = [
        Op::H(0),
        Op::Cx(0, 1),
        Op::H(2),
        Op::Cx(2, 3),
        Op::H(4),
        Op::Cx(4, 5),
        Op::S(1),
        Op::S(3),
    ];

    let compiled_gates: Vec<CompiledGate> = ops.iter().copied().map(to_compiled).collect();
    let hash = gpu_compiled.compile_circuit(&compiled_gates);
    gpu_compiled.execute_compiled_wait(hash);

    for op in ops {
        apply_trait(&mut gpu_normal, op);
    }
    gpu_normal.sync_wait();

    let results_compiled = gpu_compiled.mz(&(0..6).map(QubitId).collect::<Vec<_>>());
    let results_normal = gpu_normal.mz(&(0..6).map(QubitId).collect::<Vec<_>>());
    for (a, b) in results_compiled.iter().zip(results_normal.iter()) {
        assert_eq!(a.outcome, b.outcome, "compile_circuit vs normal mz differ");
    }
}

#[test]
fn compile_circuit_cached_second_call() {
    // Compiling the same circuit twice should return the same hash.
    let Ok(mut gpu) = DefaultGpuStab::with_seed(4, 7) else {
        return;
    };
    let gates = vec![
        CompiledGate::h(0),
        CompiledGate {
            gate_type: GateType::Cx,
            target: 1,
            control: Some(0),
        },
    ];
    let h1 = gpu.compile_circuit(&gates);
    let h2 = gpu.compile_circuit(&gates);
    assert_eq!(h1, h2);
    assert!(gpu.is_circuit_compiled(&gates));
}

#[test]
fn compile_circuit_matches_cpu_ghz() {
    // GHZ-state preparation. Z-basis measurement is deterministic (all agree
    // on random 0 or 1). Compiled and normal paths must agree with CPU on at
    // least the correlation structure.
    let n: u32 = 5;
    let gates: Vec<CompiledGate> = std::iter::once(CompiledGate::h(0))
        .chain((0..n - 1).map(|q| CompiledGate {
            gate_type: GateType::Cx,
            target: q + 1,
            control: Some(q),
        }))
        .collect();

    let Ok(mut gpu) = DefaultGpuStab::with_seed(n as usize, 99) else {
        return;
    };
    let hash = gpu.compile_circuit(&gates);
    gpu.execute_compiled_wait(hash);

    let mut cpu = SparseStab::new(n as usize);
    cpu.h(&[QubitId(0)]);
    for q in 0..n - 1 {
        cpu.cx(&[(QubitId(q as usize), QubitId((q + 1) as usize))]);
    }

    // After GHZ prep all measurement outcomes must be identical across
    // qubits in each individual shot.
    let gpu_results = gpu.mz(&(0..n as usize).map(QubitId).collect::<Vec<_>>());
    let cpu_results = cpu.mz(&(0..n as usize).map(QubitId).collect::<Vec<_>>());
    let gpu_val = gpu_results[0].outcome;
    for r in &gpu_results {
        assert_eq!(r.outcome, gpu_val, "GHZ: GPU qubits disagree");
    }
    let cpu_val = cpu_results[0].outcome;
    for r in &cpu_results {
        assert_eq!(r.outcome, cpu_val, "GHZ: CPU qubits disagree");
    }
}

// ---------------------------------------------------------------------------
// GpuStab mz_queue / mz_fetch mid-circuit measurement
// ---------------------------------------------------------------------------

#[test]
fn stab_mz_queue_fetch_matches_direct_mz() {
    // For deterministic Clifford circuits, mz_queue + more gates + mz_fetch
    // should produce outcomes identical to calling mz() at each intermediate
    // point.
    let Ok(mut gpu_queue) = DefaultGpuStab::with_seed(4, 11) else {
        return;
    };
    let Ok(mut gpu_direct) = DefaultGpuStab::with_seed(4, 11) else {
        return;
    };

    // Prepare |0101> by applying X on qubits 1, 3. Measure.
    // Then apply X on qubit 0 and measure again.
    // Both outcomes forced by state.
    gpu_queue.x(&[QubitId(1), QubitId(3)]);
    gpu_queue.mz_queue(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    gpu_queue.x(&[QubitId(0)]);
    gpu_queue.mz_queue(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    let all = gpu_queue.mz_fetch();
    assert_eq!(all.len(), 8);
    let first_round: Vec<bool> = all[..4].iter().map(|r| r.outcome).collect();
    let second_round: Vec<bool> = all[4..].iter().map(|r| r.outcome).collect();
    assert_eq!(first_round, vec![false, true, false, true]);
    assert_eq!(second_round, vec![true, true, false, true]);

    // Compare with direct-mz reference.
    gpu_direct.x(&[QubitId(1), QubitId(3)]);
    let first_direct = gpu_direct.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    gpu_direct.x(&[QubitId(0)]);
    let second_direct = gpu_direct.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    let first_direct: Vec<bool> = first_direct.iter().map(|r| r.outcome).collect();
    let second_direct: Vec<bool> = second_direct.iter().map(|r| r.outcome).collect();
    assert_eq!(first_round, first_direct);
    assert_eq!(second_round, second_direct);
}

// ---------------------------------------------------------------------------
// GpuStabMulti mz_queue / mz_fetch
// ---------------------------------------------------------------------------

#[test]
fn stab_multi_fresh_state_mz_queue_all_zero() {
    // Absolute simplest: fresh |0000> via constructor, mz_queue immediately.
    let shots = 4;
    let Ok(mut gpu) = GpuStabMulti::<PecosRng>::with_seed(4, shots, 42) else {
        return;
    };
    gpu.mz_queue(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    let all = gpu.mz_fetch();
    for (shot, row) in all.iter().enumerate() {
        assert_eq!(
            *row,
            vec![false, false, false, false],
            "shot {shot}: fresh |0000> mz_queue should be all-false, got {row:?}"
        );
    }
}

#[test]
fn stab_multi_x_then_mz_queue() {
    // Simplest per-qubit determinism: X(q), mz_queue(q). Should be true.
    let shots = 4;
    for q in 0..4 {
        let Ok(mut gpu) = GpuStabMulti::<PecosRng>::with_seed(4, shots, 42) else {
            return;
        };
        gpu.x(&[QubitId(q)]);
        gpu.mz_queue(&[QubitId(q)]);
        let all = gpu.mz_fetch();
        for (shot, row) in all.iter().enumerate() {
            assert_eq!(
                *row,
                vec![true],
                "shot {shot}: X({q}) mz_queue({q}) should be true, got {row:?}"
            );
        }
    }
}

#[test]
fn stab_multi_mz_queue_deterministic_shots() {
    // Deterministic circuit: all 32 shots must agree.
    let shots = 32;
    let Ok(mut gpu) = GpuStabMulti::<PecosRng>::with_seed(4, shots, 42) else {
        return;
    };
    gpu.x(&[QubitId(1), QubitId(3)]);
    gpu.mz_queue(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    gpu.x(&[QubitId(0)]);
    gpu.mz_queue(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    let all = gpu.mz_fetch();
    assert_eq!(all.len(), shots);
    for row in &all {
        assert_eq!(row.len(), 8, "8 queued measurements per shot");
        let expected = vec![false, true, false, true, true, true, false, true];
        assert_eq!(*row, expected);
    }
}

#[test]
fn stab_multi_reset_clears_queue() {
    // After reset the measurement queue must be empty.
    let Ok(mut gpu) = GpuStabMulti::<PecosRng>::with_seed(3, 4, 42) else {
        return;
    };
    gpu.x(&[QubitId(0)]);
    gpu.mz_queue(&[QubitId(0)]);
    gpu.reset();
    // Fresh state: mz_queue on fresh |000> then fetch should be all zeros.
    gpu.mz_queue(&[QubitId(0), QubitId(1), QubitId(2)]);
    let all = gpu.mz_fetch();
    for row in &all {
        assert_eq!(
            *row,
            vec![false, false, false],
            "reset should clear residual queue and state"
        );
    }
}

// ---------------------------------------------------------------------------
// Reset semantics: multiple reuse cycles on the same GpuStab should not leak
// state or RNG entropy between runs.
// ---------------------------------------------------------------------------

#[test]
fn stab_reset_reuse_deterministic() {
    let Ok(mut gpu) = DefaultGpuStab::with_seed(4, 42) else {
        return;
    };
    for cycle in 0..5 {
        // Each cycle: reset, apply X(2), measure all.
        gpu.reset();
        gpu.x(&[QubitId(2)]);
        let r = gpu.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        let outs: Vec<bool> = r.iter().map(|x| x.outcome).collect();
        assert_eq!(
            outs,
            vec![false, false, true, false],
            "cycle {cycle}: reset+X(2) should give |0010>"
        );
    }
}
