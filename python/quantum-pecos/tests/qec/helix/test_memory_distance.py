# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""PECOS connected-cluster lower bounds and randomized witness upper bounds.

Uniform p1=p2=p_meas=p_prep=0.001 gives exact distance six for one round;
the slow lane establishes exact distance six for two rounds too.
"""

from time import perf_counter

import pytest
from pecos.qec import DetectorErrorModel, FaultDistanceUpperBoundConfig
from pecos.qec.helix import memory_circuit

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


def upper_bound_config(samples=200, *, seed=17, osd_method="osd_0", osd_order=0):
    return FaultDistanceUpperBoundConfig(
        samples=samples,
        seed=seed,
        observable_subset_strategy="each_single_then_random",
        error_rate=0.01,
        max_iterations=100,
        bp_method="product_sum",
        bp_schedule="parallel",
        min_sum_scaling_factor=1.0,
        osd_method=osd_method,
        osd_order=osd_order,
        omp_threads=1,
    )


def _randomized_witness(basis: str, rounds: int) -> None:
    dem = DetectorErrorModel.from_circuit(memory_circuit(basis, rounds), **UNIFORM_NOISE)
    start = perf_counter()
    witness = dem.randomized_fault_distance_upper_bound(upper_bound_config())
    elapsed = perf_counter() - start
    assert witness is not None
    print(f"randomized basis={basis} rounds={rounds}: weight={witness.weight}, {elapsed:.3f}s", flush=True)
    assert witness.weight == 6


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize(("rounds", "max_weight"), [(1, 5), (2, 4)])
def test_uniform_noise_cluster_lower_bound(basis: str, rounds: int, max_weight: int) -> None:
    _cluster(basis, rounds, max_weight)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_uniform_noise_randomized_upper_bound(basis: str) -> None:
    _randomized_witness(basis, 1)


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_uniform_noise_two_round_distance_six(basis: str) -> None:
    _cluster(basis, 2, 5)
    _randomized_witness(basis, 2)
