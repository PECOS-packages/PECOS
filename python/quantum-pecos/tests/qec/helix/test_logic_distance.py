# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Logical gate distances under uniform p1=p2=p_meas=p_prep=0.001.

PECOS connected-cluster searches give exact distances or lower bounds;
randomized witnesses give upper bounds.
"""

from time import perf_counter

import pytest
from pecos.qec import DetectorErrorModel, discover_detectors
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix import (
    HelixAllocation,
    c4_helix_code,
    helix_result_refs,
    logic_circuit,
    logic_conjugates,
    logical_gate_steps,
    lower_steps,
    memory_conjugates,
    set_discovered_metadata,
    syndrome_round_gadgets,
)
from pecos.qec.helix.gadgets import memory_gadgets

from qec.helix.test_memory_distance import UNIFORM_NOISE, upper_bound_config


def _exact_cluster(circuit, max_weight, expected, label):
    dem = DetectorErrorModel.from_circuit(circuit, **UNIFORM_NOISE)
    start = perf_counter()
    result = dem.connected_cluster_fault_distance(max_weight)
    value = None if result is None else result.distance
    print(f"{label}: exact cluster max_weight={max_weight}: {value}, {perf_counter() - start:.3f}s", flush=True)
    assert value == expected


def _cluster(circuit, max_weight, label):
    dem = DetectorErrorModel.from_circuit(circuit, **UNIFORM_NOISE)
    start = perf_counter()
    distances = dem.per_observable_fault_distances(max_weight)
    values = [None if result is None else result.distance for result in distances]
    print(f"{label}: cluster max_weight={max_weight}: {values}, {perf_counter() - start:.3f}s", flush=True)
    assert values == [None, None], str(distances)


def test_logic_gate_fault_exact_six():
    """PECOS connected-cluster search proves distance six with S0S1 gate noise only."""
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
    baseline = DetectorErrorModel.from_circuit(circuit, **UNIFORM_NOISE)
    for tick in range(circuit.num_ticks()):
        if not b0 <= tick < b1:
            for batch_index in range(len(circuit.get_tick(tick))):
                circuit.set_gate_meta(tick, batch_index, "noiseless", True)
    marked = DetectorErrorModel.from_circuit(circuit, **UNIFORM_NOISE)
    assert len(marked.contribution_effect_summaries()) < len(baseline.contribution_effect_summaries())
    _exact_cluster(circuit, 6, 6, "gate only")


@pytest.mark.slow
def test_logic_full_noise_z_bounds():
    """For prepare, round, S0S1, round, S0S1, round, Z readout: 5 <= d <= 6.

    A PECOS randomized witness gives the upper bound. Connected-cluster
    weight four is exhausted; the expensive weight-five cluster search is not attempted.
    """
    circuit = logic_circuit("Z", ("S0S1", "S0S1"))
    dem = DetectorErrorModel.from_circuit(circuit, **UNIFORM_NOISE)
    start = perf_counter()
    # With 150 samples, seeds 0..4 all reach weight six; retain margin for future decoder or sampler changes.
    witness = dem.randomized_fault_distance_upper_bound(
        upper_bound_config(150, seed=0, osd_method="osd_cs", osd_order=1),
    )
    assert witness is not None
    print(f"full-noise Z: randomized weight={witness.weight}, {perf_counter() - start:.3f}s", flush=True)
    assert witness.weight == 6
    _cluster(circuit, 4, "full-noise Z")


def test_logic_full_noise_x_floor():
    """Exact distance four is the owner-accepted full-noise X floor of 2026-10-29.

    PECOS connected-cluster search through weight four proves the value.
    This circuit has prepare, round, S0S1, round, S0S1, round, and X readout.
    The floor comes from the X-basis boundaries: X preparation gives the first
    round's Z checks no reference and X readout gives no final Z values, so X
    errors next to a boundary go undetected and S0S1 turns them into Z errors.
    It stays four with both gates noiseless; six holds with gate faults only
    (test_logic_gate_fault_exact_six). Both a drop to three and a rise fail
    this test.
    """
    circuit = logic_circuit("X", ("S0S1", "S0S1"))
    _exact_cluster(circuit, 4, 4, "full-noise X")


@pytest.mark.slow
def test_logic_full_noise_x_padded_exact_five():
    """Three rounds before the first S0S1 and two after the last give exactly five.

    PECOS connected-cluster search through weight five proves the value.
    Padding moves both X boundaries two Z-check rounds away from the gates.
    Five is also the most an S0S1 inside a circuit reaches: four faults in the
    two rounds before it and one in its CZ flags stay undetectable with the
    gate moved two rounds deeper and four rounds after it. Both a drop to four
    and a rise fail this test.
    """
    tokens = ("S0S1", "S0S1")
    memory = memory_gadgets("X", 3)
    steps = [step for _, gadget in memory[:-1] for step in gadget]
    for variant in "ba":
        steps.extend(logical_gate_steps("S0S1"))
        steps.extend(step for _, gadget in syndrome_round_gadgets(variant) for step in gadget)
    steps.extend(step for _, gadget in syndrome_round_gadgets("b") for step in gadget)
    steps.extend(memory[-1][1])
    circuit = lower_steps(steps, HelixAllocation.NUM_SLOTS).tick_circuit
    records = {ref: record for record, ref in enumerate(helix_result_refs(circuit))}
    observables = [
        [records[(f"data:{q}", 0)] for q, axis in enumerate(row) if axis != "I"] for row in c4_helix_code().logical_x
    ]
    set_discovered_metadata(circuit, discover_detectors(circuit, observables, logic_conjugates("X", tokens)))
    _exact_cluster(circuit, 5, 5, "padded full-noise X")
