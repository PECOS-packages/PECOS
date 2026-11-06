# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Two-block CNOT distance bounds under uniform p1=p2=p_meas=p_prep=0.001.

Prepare, one round, transversal CNOT, one round, readout; no TICK barriers.
PECOS connected-cluster lower bounds and randomized witnesses in the slow lane establish 5 <= d <= 6 for ZZ, XX and ZX.
"""

from time import perf_counter

import pytest
from pecos.qec import DetectorErrorModel
from pecos.qec.helix import cnot_circuit

from qec.helix.test_cnot import PAIRS
from qec.helix.test_memory_distance import UNIFORM_NOISE, upper_bound_config


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
def test_randomized_weight_six(pair: str) -> None:
    """A PECOS randomized weight-six witness supplies the upper bound in 5 <= d <= 6."""
    dem = DetectorErrorModel.from_circuit(cnot_circuit(*pair), **UNIFORM_NOISE)
    start = perf_counter()
    witness = dem.randomized_fault_distance_upper_bound(upper_bound_config())
    assert witness is not None
    print(f"CNOT {pair}: randomized weight={witness.weight}, {perf_counter() - start:.3f}s", flush=True)
    assert witness.weight == 6, str(witness)


@pytest.mark.slow
@pytest.mark.parametrize("pair", PAIRS)
def test_cluster_weight_four(pair: str) -> None:
    """Exhaust weight four for all four observables, supplying 5 <= d in 5 <= d <= 6."""
    _cluster(pair, 4)
