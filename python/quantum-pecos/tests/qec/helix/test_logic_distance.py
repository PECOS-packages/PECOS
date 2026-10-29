# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Logical gate distance bounds under uniform p1=p2=p_meas=p_prep=0.001."""

from time import perf_counter

import pytest
import stim
from pecos.qec import DetectorErrorModel, discover_detectors
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix import (
    HelixAllocation,
    c4_helix_code,
    helix_result_refs,
    logic_circuit,
    logical_gate_steps,
    lower_steps,
    memory_conjugates,
    set_discovered_metadata,
    syndrome_round_gadgets,
)
from pecos.qec.helix.gadgets import memory_gadgets
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

from qec.helix.test_memory_distance import UNIFORM_NOISE


def _witness(circuit, expected, label):
    start = perf_counter()
    witness = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    print(f"{label}: stim weight={len(witness)}, {perf_counter() - start:.3f}s", flush=True)
    assert len(witness) == expected, str(witness)


def _cluster(circuit, max_weight, label):
    dem = DetectorErrorModel.from_circuit(circuit, **UNIFORM_NOISE)
    start = perf_counter()
    distances = dem.per_observable_fault_distances(max_weight)
    values = [None if result is None else result.distance for result in distances]
    print(f"{label}: cluster max_weight={max_weight}: {values}, {perf_counter() - start:.3f}s", flush=True)
    assert values == [None, None], str(distances)


def test_logic_gate_fault_upper_bound():
    """A weight-six witness is an upper bound with noise on the S0S1 gate only."""
    prefix = memory_gadgets("X", 1)[0][1] + [step for _, steps in syndrome_round_gadgets("a") for step in steps]
    through_gate = [*prefix, CircuitStep(OpType.TICK), *logical_gate_steps("S0S1")]
    b0 = lower_steps(prefix, HelixAllocation.NUM_SLOTS).tick_circuit.num_ticks()
    b1 = lower_steps(through_gate, HelixAllocation.NUM_SLOTS).tick_circuit.num_ticks()
    assert (b0, b1) == (16, 24)
    steps = [*through_gate, CircuitStep(OpType.TICK)]
    steps.extend(step for _, gadget in syndrome_round_gadgets("b") for step in gadget)
    for slot in range(20):
        steps.extend(
            [
                CircuitStep(OpType.SZDG, [slot]),
                CircuitStep(OpType.H, [slot]),
                CircuitStep(OpType.MEASURE, [slot], f"data:{slot}"),
            ],
        )
    circuit = lower_steps(steps, HelixAllocation.NUM_SLOTS).tick_circuit
    code = c4_helix_code()
    x_support = {q for q in range(20) if (code.logical_x[0][q] != "I") != (code.logical_x[1][q] != "I")}
    z_support = {q for q in range(20) if (code.logical_z[0][q] != "I") != (code.logical_z[1][q] != "I")}
    assert x_support == z_support
    records = {ref: record for record, ref in enumerate(helix_result_refs(circuit))}
    observables = [[records[(f"data:{q}", 0)] for q in sorted(x_support)]]
    found = discover_detectors(circuit, observables, memory_conjugates("X")[:1])
    set_discovered_metadata(circuit, found)
    noisy = stim.Circuit(tick_circuit_to_stim(circuit, **UNIFORM_NOISE))
    gate_noise = stim.Circuit()
    tick = 0
    for instruction in noisy:
        if instruction.name == "TICK":
            tick += 1
        if instruction.name in {"X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"} and not b0 <= tick < b1:
            continue
        gate_noise.append(instruction)
    _witness(gate_noise, 6, "gate only")


@pytest.mark.slow
def test_logic_full_noise_z_bounds():
    """For prepare, round, S0S1, round, S0S1, round, Z readout: 5 <= d <= 6.

    Weight four is exhausted; the expensive weight-five cluster search is not attempted.
    """
    circuit = logic_circuit("Z", ("S0S1", "S0S1"))
    _witness(stim.Circuit(tick_circuit_to_stim(circuit, **UNIFORM_NOISE)), 6, "full-noise Z")
    _cluster(circuit, 4, "full-noise Z")


def test_logic_full_noise_x_floor():
    """Exact distance four is the owner-accepted full-noise X floor of 2026-10-29.

    Round faults feeding a gate spread through its CZs. Six holds with gate
    faults only (test_logic_gate_fault_upper_bound). This circuit has prepare,
    round, S0S1, round, S0S1, round, and X readout. Both a drop to three and a
    rise fail this test.
    """
    circuit = logic_circuit("X", ("S0S1", "S0S1"))
    _witness(stim.Circuit(tick_circuit_to_stim(circuit, **UNIFORM_NOISE)), 4, "full-noise X")
    _cluster(circuit, 3, "full-noise X")
