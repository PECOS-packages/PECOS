//! Hardware integration coverage. Pure boundary coverage lives in `qubit_validation`.
use crate::qubit_validation::tests::assert_bounds_panic;
use crate::{
    CompiledGate, GpuDensityMatrix32, GpuPauliProp, GpuStab, GpuStabMulti, GpuStateVec32,
    GpuStateVec64,
};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, SparseStab};

pub(crate) fn check_clifford_bounds<S: CliffordGateable>(sim: &mut S, name: &str) {
    for bad in [2, 16383, 16384, usize::MAX] {
        let qubits = [QubitId(0), QubitId(bad)];
        macro_rules! single {
            ($($gate:ident),+ $(,)?) => {$(
                assert_bounds_panic(name, bad, 2, || { sim.$gate(&qubits); });
            )+};
        }
        single!(
            identity, x, y, z, h, h2, h3, h4, h5, h6, sx, sxdg, sy, sydg, sz, szdg, f, fdg, f2,
            f2dg, f3, f3dg, f4, f4dg, mz, mx, my, mnx, mny, mnz, px, pnx, py, pny, pz, pnz, mpx,
            mpnx, mpy, mpny, mpz, mpnz
        );
        assert_bounds_panic(name, bad, 2, || {
            sim.apply_global_phase(Angle64::ZERO, &qubits);
        });
        for pair in [(QubitId(0), QubitId(bad)), (QubitId(bad), QubitId(0))] {
            macro_rules! pair {
                ($($gate:ident),+ $(,)?) => {$(
                    assert_bounds_panic(name, bad, 2, || { sim.$gate(&[pair]); });
                )+};
            }
            pair!(
                cx, cy, cz, swap, sxx, sxxdg, syy, syydg, szz, szzdg, iswap, iswapdg, g, gdg
            );
        }
    }
    // Every rejected operation must leave |00> unchanged.
    assert!(sim.mz(&[QubitId(0), QubitId(1)]).iter().all(|r| !r.outcome));
    sim.reset();
    let mut cpu = SparseStab::new(2);
    // Entangle and uncompute to compare deterministic results with the CPU reference.
    sim.h(&[QubitId(0)])
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(0), QubitId(1))])
        .h(&[QubitId(0)])
        .x(&[QubitId(1)]);
    cpu.h(&[QubitId(0)])
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(0), QubitId(1))])
        .h(&[QubitId(0)])
        .x(&[QubitId(1)]);
    assert_eq!(
        sim.mz(&[QubitId(0), QubitId(1)])
            .iter()
            .map(|r| (r.outcome, r.is_deterministic))
            .collect::<Vec<_>>(),
        cpu.mz(&[QubitId(0), QubitId(1)])
            .iter()
            .map(|r| (r.outcome, r.is_deterministic))
            .collect::<Vec<_>>()
    );
}

pub(crate) fn check_rotation_bounds<S: ArbitraryRotationGateable>(sim: &mut S, name: &str) {
    let bad = QubitId(usize::MAX);
    let qubits = [QubitId(0), bad];
    macro_rules! rotation {
        ($($gate:ident),+ $(,)?) => {$(
            assert_bounds_panic(name, bad.index(), 2, || { sim.$gate(Angle64::ZERO, &qubits); });
        )+};
    }
    rotation!(rx, ry, rz);
    for pair in [(QubitId(0), bad), (bad, QubitId(0))] {
        macro_rules! rotation_pair {
            ($($gate:ident),+ $(,)?) => {$(
                assert_bounds_panic(name, bad.index(), 2, || { sim.$gate(Angle64::ZERO, &[pair]); });
            )+};
        }
        rotation_pair!(rxx, ryy, rzz);
        assert_bounds_panic(name, bad.index(), 2, || {
            sim.u2q(
                [[Angle64::ZERO; 3]; 2],
                [Angle64::ZERO; 3],
                [[Angle64::ZERO; 3]; 2],
                &[pair],
            );
        });
    }
}

fn cpu_reference_outcomes() -> Vec<bool> {
    let mut cpu = SparseStab::new(2);
    cpu.h(&[QubitId(0)])
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(0), QubitId(1))])
        .h(&[QubitId(0)])
        .x(&[QubitId(1)]);
    cpu.mz(&[QubitId(0), QubitId(1)])
        .iter()
        .map(|result| result.outcome)
        .collect()
}

#[test]
fn gpu_stab_bounds() {
    let mut sim = match GpuStab::new(2) {
        Ok(sim) => sim,
        Err(error) => {
            eprintln!("SKIP gpu_stab_bounds: {error}");
            return;
        }
    };
    check_clifford_bounds(&mut sim, "GpuStab");
    for bad in [2, 16383, 16384, usize::MAX] {
        assert_bounds_panic("GpuStab", bad, 2, || {
            sim.mz_forced(bad, true);
        });
        assert_bounds_panic("GpuStab", bad, 2, || {
            sim.mz_queue(&[QubitId(bad)]);
        });
        sim.enable_fusion();
        assert_bounds_panic("GpuStab", bad, 2, || {
            sim.h(&[QubitId(bad)]);
        });
        assert_bounds_panic("GpuStab", bad, 2, || {
            sim.cx(&[(QubitId(0), QubitId(bad))]);
        });
        sim.disable_fusion();
    }
    for gate in [
        CompiledGate::h(2),
        CompiledGate::cx(2, 0),
        CompiledGate::cx(0, 2),
    ] {
        assert_bounds_panic("GpuStab", 2, 2, || {
            sim.compile_circuit(&[gate]);
        });
    }
    sim.reset();
    let circuit = sim.compile_circuit(&[
        CompiledGate::h(0),
        CompiledGate::cx(0, 1),
        CompiledGate::cx(0, 1),
        CompiledGate::h(0),
        CompiledGate::x(1),
    ]);
    sim.execute_compiled_wait(circuit);
    assert_eq!(
        sim.mz(&[QubitId(0), QubitId(1)])
            .iter()
            .map(|r| r.outcome)
            .collect::<Vec<_>>(),
        cpu_reference_outcomes()
    );
}

#[test]
fn gpu_stab_multi_bounds() {
    let mut sim = match GpuStabMulti::<pecos_random::PecosRng>::new(2, 4) {
        Ok(sim) => sim,
        Err(error) => {
            eprintln!("SKIP gpu_stab_multi_bounds: {error}");
            return;
        }
    };
    for bad in [2, 16383, 16384, usize::MAX] {
        macro_rules! single {
            ($($gate:ident),+ $(,)?) => {$(
                assert_bounds_panic("GpuStabMulti", bad, 2, || { sim.$gate(&[QubitId(0), QubitId(bad)]); });
            )+};
        }
        single!(h, x, y, z, sz, szdg, mx, my, mz, mz_queue);
        for pair in [(QubitId(0), QubitId(bad)), (QubitId(bad), QubitId(0))] {
            assert_bounds_panic("GpuStabMulti", bad, 2, || {
                sim.cx(&[pair]);
            });
            assert_bounds_panic("GpuStabMulti", bad, 2, || {
                sim.cz(&[pair]);
            });
            assert_bounds_panic("GpuStabMulti", bad, 2, || {
                sim.swap(&[pair]);
            });
        }
    }
    assert_eq!(
        sim.mz(&[QubitId(0), QubitId(1)]),
        vec![vec![false, false]; 4]
    );
    sim.h(&[QubitId(0)])
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(0), QubitId(1))])
        .h(&[QubitId(0)])
        .x(&[QubitId(1)]);
    assert_eq!(
        sim.mz(&[QubitId(0), QubitId(1)]),
        vec![cpu_reference_outcomes(); 4]
    );
}

#[test]
fn gpu_statevec32_bounds() {
    let mut sim = match GpuStateVec32::new(2) {
        Ok(sim) => sim,
        Err(error) => {
            eprintln!("SKIP gpu_statevec32_bounds: {error}");
            return;
        }
    };
    eprintln!("GPU bounds adapter: {}", sim.adapter_info());
    check_clifford_bounds(&mut sim, "GpuStateVec32");
    check_rotation_bounds(&mut sim, "GpuStateVec32");
    assert_bounds_panic("GpuStateVec32", 2, 2, || {
        sim.apply_single_gate(2, crate::gates::H);
    });
    assert_bounds_panic("GpuStateVec32", 2, 2, || {
        sim.apply_cx(2, 0);
    });
    assert_bounds_panic("GpuStateVec32", 2, 2, || {
        sim.mz_cpu_batch(&[QubitId(2)]);
    });
    assert_bounds_panic("GpuStateVec32", 2, 2, || {
        sim.mz_gpu_sequential(&[QubitId(2)]);
    });
}

#[test]
fn gpu_statevec64_bounds() {
    let mut sim = match GpuStateVec64::new(2) {
        Ok(sim) => sim,
        Err(error) => {
            eprintln!("SKIP gpu_statevec64_bounds: {error}");
            return;
        }
    };
    check_clifford_bounds(&mut sim, "GpuStateVec64");
    check_rotation_bounds(&mut sim, "GpuStateVec64");
    assert_bounds_panic("GpuStateVec64", 2, 2, || {
        sim.mz_cpu_batch(&[QubitId(2)]);
    });
    assert_bounds_panic("GpuStateVec64", 2, 2, || {
        sim.mz_gpu_sequential(&[QubitId(2)]);
    });
}

#[test]
fn gpu_density_bounds() {
    let mut sim = match GpuDensityMatrix32::new(2) {
        Ok(sim) => sim,
        Err(error) => {
            eprintln!("SKIP gpu_density_bounds: {error}");
            return;
        }
    };
    check_clifford_bounds(&mut sim, "GpuDensityMatrix");
    check_rotation_bounds(&mut sim, "GpuDensityMatrix");
    assert_bounds_panic("GpuDensityMatrix", 2, 2, || {
        sim.apply_bit_flip(2, 0.0);
    });
}

#[test]
fn gpu_pauli_bounds() {
    let mut sim = match GpuPauliProp::new(2, 4) {
        Ok(sim) => sim,
        Err(error) => {
            eprintln!("SKIP gpu_pauli_bounds: {error}");
            return;
        }
    };
    assert_bounds_panic("GpuPauliProp", 2, 2, || {
        sim.h(&[2]);
    });
    assert_bounds_panic("GpuPauliProp", 2, 2, || {
        sim.cx(&[(2, 0)]);
    });
    assert_bounds_panic("GpuPauliProp", usize::MAX, 2, || {
        sim.inject_x_fault(usize::MAX);
    });
    assert_bounds_panic("GpuPauliProp", 2, 2, || {
        sim.inject_depol1(2, 0.0);
    });
    assert_bounds_panic("GpuPauliProp", 2, 2, || {
        sim.inject_depol2(0, 2, 0.0);
    });
    assert_bounds_panic("GpuPauliProp", 2, 2, || {
        sim.measure_z_flips(&[2]);
    });
    assert_bounds_panic("GpuPauliProp", 2, 2, || {
        sim.check_anticommutation(&[0], &[2]);
    });
}
