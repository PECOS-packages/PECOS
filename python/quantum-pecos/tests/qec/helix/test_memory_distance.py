# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fault-distance acceptance under uniform p1=p2=p_meas=p_prep=0.001 noise."""

from time import perf_counter

import pytest
import stim
from pecos.qec import DetectorErrorModel
from pecos.qec.helix import memory_circuit
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

UNIFORM_NOISE = {"p1": 0.001, "p2": 0.001, "p_meas": 0.001, "p_prep": 0.001}


def _cluster(basis: str, rounds: int, max_weight: int) -> None:
    tc = memory_circuit(basis, rounds)
    dem = DetectorErrorModel.from_circuit(tc, **UNIFORM_NOISE)
    start = perf_counter()
    distances = dem.per_observable_fault_distances(max_weight)
    elapsed = perf_counter() - start
    values = [None if result is None else result.distance for result in distances]
    print(f"cluster basis={basis} rounds={rounds} max_weight={max_weight}: {values}, {elapsed:.3f}s", flush=True)
    assert values == [None, None]


def _stim_witness(basis: str, rounds: int) -> None:
    circuit = stim.Circuit(tick_circuit_to_stim(memory_circuit(basis, rounds), **UNIFORM_NOISE))
    start = perf_counter()
    witness = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=6,
        dont_explore_edges_with_degree_above=6,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    elapsed = perf_counter() - start
    print(f"stim basis={basis} rounds={rounds}: weight={len(witness)}, {elapsed:.3f}s", flush=True)
    assert len(witness) == 6


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize(("rounds", "max_weight"), [(1, 5), (2, 4)])
def test_uniform_noise_cluster_lower_bound(basis: str, rounds: int, max_weight: int) -> None:
    _cluster(basis, rounds, max_weight)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_uniform_noise_stim_upper_bound(basis: str) -> None:
    _stim_witness(basis, 1)


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_uniform_noise_two_round_distance_six(basis: str) -> None:
    _cluster(basis, 2, 5)
    _stim_witness(basis, 2)
