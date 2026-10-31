# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Two-block CNOT distance bounds under uniform p1=p2=p_meas=p_prep=0.001.

Prepare, one round, transversal CNOT, one round, readout; no TICK barriers.
The post-merge slow lane establishes 5 <= d <= 6 for ZZ, XX and ZX.
"""

from time import perf_counter

import pytest
import stim
from pecos.qec import DetectorErrorModel
from pecos.qec.helix import cnot_circuit
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

from qec.helix.test_cnot import PAIRS
from qec.helix.test_memory_distance import UNIFORM_NOISE


def _cluster(pair: str, max_weight: int) -> None:
    dem = DetectorErrorModel.from_circuit(cnot_circuit(*pair), **UNIFORM_NOISE)
    start = perf_counter()
    distances = dem.per_observable_fault_distances(max_weight)
    values = [None if result is None else result.distance for result in distances]
    print(f"CNOT {pair}: cluster max_weight={max_weight}: {values}, {perf_counter() - start:.3f}s", flush=True)
    assert values == [None] * 4, str(distances)


@pytest.mark.parametrize("pair", PAIRS)
def test_cluster_weight_three(pair: str) -> None:
    """Exhaust weight three in the default lane; the slow lane pins 5 <= d <= 6."""
    _cluster(pair, 3)


@pytest.mark.slow
@pytest.mark.parametrize("pair", PAIRS)
def test_stim_weight_six(pair: str) -> None:
    """A weight-six witness supplies the upper bound in 5 <= d <= 6."""
    circuit = stim.Circuit(tick_circuit_to_stim(cnot_circuit(*pair), **UNIFORM_NOISE))
    start = perf_counter()
    witness = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    print(f"CNOT {pair}: stim weight={len(witness)}, {perf_counter() - start:.3f}s", flush=True)
    assert len(witness) == 6, str(witness)


@pytest.mark.slow
@pytest.mark.parametrize("pair", PAIRS)
def test_cluster_weight_four(pair: str) -> None:
    """Exhaust weight four for all four observables, supplying 5 <= d in 5 <= d <= 6."""
    _cluster(pair, 4)
