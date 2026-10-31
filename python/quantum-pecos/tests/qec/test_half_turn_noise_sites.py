# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""A physical half-turn interaction retains its two-qubit noise and idle sites."""

from __future__ import annotations

import math

import pytest
import stim
from guppylang import guppy
from guppylang.std.angles import pi
from guppylang.std.builtins import array, result
from guppylang.std.qsystem.helios import zz_phase
from guppylang.std.quantum import measure, qubit
from pecos import NoiseParameters
from pecos._traced_circuit import normalize_traced_tick_circuit
from pecos.qec import DemInferenceSettings, Detector, DetectorErrorModel, Observable, rec
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.quantum import TickCircuit
from pecos.tracing import capture_qis_operation_trace, qis_operation_trace_to_tick_circuit


@guppy
def _half_turn() -> None:
    a = qubit()
    b = qubit()
    zz_phase(a, b, pi)
    ma = measure(a).read()
    result("raw measurements", ma)
    mb = measure(b).read()
    result("raw measurements", mb)
    result("DETECTOR", array(ma, mb))
    result("obs", ma)


@pytest.mark.parametrize("runtime_name", ["simple", "soft_rz"])
@pytest.mark.parametrize("inferred", [False, True])
def test_guppy_half_turn_retains_correlated_two_qubit_noise(runtime_name: str, *, inferred: bool) -> None:
    """A 20% XX channel produces one joint event, independent of p1."""
    plugin = pytest.importorskip(f"selene_{runtime_name}_runtime_plugin")
    runtime = getattr(plugin, "SimpleRuntimePlugin" if runtime_name == "simple" else "SoftRZRuntimePlugin")()
    traced = qis_operation_trace_to_tick_circuit(capture_qis_operation_trace(_half_turn, 2, runtime=runtime))
    assert traced.gate_counts_by_type()["RZZ"] == 1
    noise = (
        NoiseParameters()
        .with_p1(0.07)
        .with_p1_weights({"X": 1.0})
        .with_p2(0.2)
        .with_p2_weights({"XX": 1.0})
        .with_p_prep(0)
        .with_p_meas(0)
    )
    builder = (
        DetectorErrorModel.builder().with_program(_half_turn).with_qubits(2).with_runtime(runtime).with_noise(noise)
    )
    if inferred:
        builder = builder.with_inference(DemInferenceSettings().with_validation_rows(8))
    else:
        builder = builder.with_detectors([Detector(rec[-2]), Detector(rec[-1])]).with_observables([Observable(rec[-2])])
    build = builder.build()
    assert build.circuit.gate_counts_by_type()["RZZ"] == 1
    assert [line for line in build.dem.to_string().splitlines() if line.startswith("error(")] == ["error(0.2) D0 D1 L0"]
    with_idles = builder.with_idle_after_2q(0.001).build()
    assert with_idles.circuit.gate_counts_by_type()["Idle"] == 2
    idle_batches = [gate for _, gate in with_idles.circuit.gate_batches() if gate.gate_type.name == "Idle"]
    assert all(gate.params == [0.001] for gate in idle_batches)
    assert sorted(q for gate in idle_batches for q in gate.qubits) == [0, 1]
    override = DetectorErrorModel.from_circuit(
        build.circuit,
        p1=0,
        p2=0,
        p2_gate_rates={"RZZ": 0.3},
        p2_weights={"XX": 1.0},
        p_prep=0,
        p_meas=0,
    )
    assert [line for line in override.to_string().splitlines() if line.startswith("error(")] == ["error(0.3) D0 D1 L0"]


@pytest.mark.parametrize("axis", ["xx", "yy", "zz"])
@pytest.mark.parametrize("theta", [0.0, math.pi, -math.pi])
def test_traced_half_turn_lowering_and_stim_keep_two_qubit_sites(axis: str, theta: float) -> None:
    """Native and Stim paths preserve a batched physical channel and its metadata."""
    tc = TickCircuit()
    getattr(tc.tick(), f"r{axis}")(theta, [(2, 3), (0, 1)])
    tc.set_gate_meta(0, 0, "site", "physical-pairs")
    normalize_traced_tick_circuit(tc, simplify_single_qubit_clifford_chains=False)
    gates = list(tc.gate_batches())
    assert len(gates) == 1
    assert gates[0][1].gate_type.name == f"R{axis.upper()}"
    assert gates[0][1].qubits == [2, 3, 0, 1]
    assert tc.get_gate_meta(0, 0, "site") == "physical-pairs"
    stim_circuit = tick_circuit_to_stim(tc, p1=0.07, p2=0.2)
    parsed = stim.Circuit(stim_circuit)
    ideal_name = "I" if theta == 0 else axis[0].upper()
    ideal = [instruction for instruction in parsed if instruction.name == ideal_name]
    assert len(ideal) == 1
    assert [target.value for target in ideal[0].targets_copy()] == [2, 3, 0, 1]
    lines = str(stim_circuit).splitlines()
    assert not any(line.startswith("DEPOLARIZE1") for line in lines)
    assert [line for line in lines if line.startswith("DEPOLARIZE2")] == ["DEPOLARIZE2(0.2) 2 3 0 1"]


def test_default_lowering_still_decomposes_half_turn() -> None:
    tc = TickCircuit()
    tc.tick().rzz(math.pi, [(0, 1)])
    tc.lower_clifford_rotations()
    assert tc.gate_counts_by_type() == {"Z": 2}


@pytest.mark.parametrize("phi", [0.0, math.pi / 2])
def test_axis_aligned_rxyxy_half_turn_retains_pair(phi: float) -> None:
    tc = TickCircuit()
    tc.tick().rxyxy2q(math.pi, phi, [(0, 1)])
    normalize_traced_tick_circuit(tc, simplify_single_qubit_clifford_chains=False)
    assert tc.gate_counts_by_type() == {"RXYXY2Q": 1}
    assert "DEPOLARIZE2(0.2) 0 1" in tick_circuit_to_stim(tc, p2=0.2)
