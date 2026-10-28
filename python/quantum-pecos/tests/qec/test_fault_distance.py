# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.

"""Python coverage for detector-error-model fault distance."""

from itertools import combinations

import pytest


def _upper_bound_config(samples: int, seed: int):
    from pecos.qec import FaultDistanceUpperBoundConfig

    return FaultDistanceUpperBoundConfig(
        samples=samples,
        seed=seed,
        observable_subset_strategy="each_single_then_random",
        error_rate=0.01,
        max_iterations=100,
        bp_method="product_sum",
        bp_schedule="parallel",
        min_sum_scaling_factor=1.0,
        osd_method="osd_0",
        osd_order=0,
        omp_threads=1,
    )


def _two_observable_dem():
    from pecos.qec import DetectorErrorModel
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().mz([0, 1, 2, 3, 4])
    circuit.add_detector(records=[-5, -4])
    circuit.add_detector(records=[-5, -3])
    circuit.add_detector(records=[-2, -1])
    circuit.add_observable(records=[-5])
    circuit.add_observable(records=[-2])
    return DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )


def _triad_dem():
    from pecos.qec import DetectorErrorModel
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().mz([0, 1, 2])
    circuit.add_detector(records=[-3, -2])
    circuit.add_detector(records=[-3, -1])
    circuit.add_observable(records=[-3])
    return DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )


def test_repetition_triad_agrees_across_all_fault_distance_methods() -> None:
    dem = _triad_dem()

    graphlike = dem.graphlike_fault_distance()
    connected = dem.connected_cluster_fault_distance(3)
    exhaustive = dem.exhaustive_fault_distance(3)
    assert graphlike is not None
    assert connected is not None
    assert exhaustive is not None
    assert graphlike.distance == connected.distance == exhaustive.distance == 3
    assert graphlike.mechanism_indices == connected.mechanism_indices
    assert connected.mechanism_indices == exhaustive.mechanism_indices


def test_repetition_triad_randomized_upper_bound_is_sound_and_tight() -> None:
    dem = _triad_dem()
    exact = dem.connected_cluster_fault_distance(3)
    sampled = dem.randomized_fault_distance_upper_bound(_upper_bound_config(8, 17))

    assert exact is not None
    assert sampled is not None
    assert sampled.weight >= exact.distance
    equality_reached = sampled.weight == exact.distance
    assert equality_reached, f"sampled upper bound equality reached: {equality_reached}"
    assert sampled.bound_kind == "upper_bound"
    assert sampled.samples_run == 8


def test_distance_three_rotated_surface_memory_cross_method_agreement() -> None:
    from pecos.qec import DetectorErrorModel, FaultDistanceResult
    from pecos.qec.surface import build_memory_circuit

    circuit = build_memory_circuit(distance=3, rounds=3, basis="Z")
    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )

    graphlike = dem.graphlike_fault_distance()
    connected = dem.connected_cluster_fault_distance(3)
    exhaustive = dem.exhaustive_fault_distance(3)

    assert isinstance(graphlike, FaultDistanceResult)
    assert isinstance(connected, FaultDistanceResult)
    assert isinstance(exhaustive, FaultDistanceResult)
    assert graphlike.distance == connected.distance == exhaustive.distance == 3
    assert graphlike.mechanism_indices == exhaustive.mechanism_indices
    assert connected.mechanism_indices == exhaustive.mechanism_indices
    assert graphlike.mechanism_indices == sorted(graphlike.mechanism_indices)
    assert repr(graphlike).startswith("FaultDistanceResult(distance=3, mechanism_indices=[")


def test_distance_three_surface_memory_randomized_upper_bound_is_sound_and_tight() -> None:
    from pecos.qec import DetectorErrorModel, FaultDistanceUpperBoundResult
    from pecos.qec.surface import build_memory_circuit

    circuit = build_memory_circuit(distance=3, rounds=3, basis="Z")
    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )
    exact = dem.graphlike_fault_distance()
    sampled = dem.randomized_fault_distance_upper_bound(_upper_bound_config(64, 23))

    assert exact is not None
    assert isinstance(sampled, FaultDistanceUpperBoundResult)
    assert sampled.weight >= exact.distance
    equality_reached = sampled.weight == exact.distance
    assert equality_reached, f"sampled upper bound equality reached: {equality_reached}"


def test_randomized_upper_bound_is_deterministic_for_same_seed() -> None:
    dem = _two_observable_dem()
    config = _upper_bound_config(32, 991)

    first = dem.randomized_fault_distance_upper_bound(config)
    second = dem.randomized_fault_distance_upper_bound(config)

    assert first is not None
    assert second is not None
    assert first.weight == second.weight
    assert first.mechanism_indices == second.mechanism_indices


def test_randomized_upper_bound_zero_samples_returns_none() -> None:
    assert _triad_dem().randomized_fault_distance_upper_bound(_upper_bound_config(0, 1)) is None


def test_randomized_upper_bound_returns_none_without_undetectable_logical_fault() -> None:
    from pecos.qec import DetectorErrorModel
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().mz([0])
    circuit.add_detector(records=[-1])
    circuit.add_observable(records=[-1])
    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.01,
        p_prep=0.0,
    )

    assert dem.randomized_fault_distance_upper_bound(_upper_bound_config(32, 5)) is None


def test_two_observable_connected_cluster_distances_differ() -> None:
    dem = _two_observable_dem()

    per_observable = dem.per_observable_fault_distances(3)
    assert [result.distance if result is not None else None for result in per_observable] == [3, 2]

    overall = dem.connected_cluster_fault_distance(3)
    assert overall is not None
    assert overall.distance == 2


def test_graphlike_fault_distance_reports_hyperedge_count() -> None:
    from pecos.qec import DetectorErrorModel
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().mz([0])
    for _ in range(3):
        circuit.add_detector(records=[-1])

    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    with pytest.raises(ValueError, match=r"found 1 hyperedge mechanism\(s\)"):
        dem.graphlike_fault_distance()


def _repetition_extraction_circuit(distance: int):
    """Extract adjacent Z parities, then destructively measure the data."""
    from pecos.quantum import TickCircuit

    num_qubits = 2 * distance - 1
    circuit = TickCircuit()
    circuit.tick().pz(list(range(num_qubits)))
    for check in range(distance - 1):
        ancilla = distance + check
        circuit.tick().cx([(check, ancilla)])
        circuit.tick().cx([(check + 1, ancilla)])
    circuit.tick().mz(list(range(num_qubits)))
    for check in range(distance - 1):
        circuit.add_detector(records=[check - num_qubits, check + 1 - num_qubits])
    for check in range(distance - 1):
        circuit.add_detector(
            records=[distance + check - num_qubits, check - num_qubits, check + 1 - num_qubits],
        )
    circuit.add_observable(records=[-num_qubits])
    return circuit


def _simulate_preparation_faults(distance: int, faults: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Inject faults into a state vector, without the DEM or Pauli fault propagator."""
    from pecos_rslib import StateVec

    num_qubits = 2 * distance - 1
    state = StateVec(num_qubits, seed=0)
    for qubit in range(num_qubits):
        state.run_1q_gate("PZ", qubit)
    for qubit in faults:
        state.run_1q_gate("X", qubit)
    for check in range(distance - 1):
        state.run_2q_gate("CX", (check, distance + check), None)
        state.run_2q_gate("CX", (check + 1, distance + check), None)
    measurements = [state.run_1q_gate("MZ", qubit) for qubit in range(num_qubits)]
    parities = tuple(measurements[check] ^ measurements[check + 1] for check in range(distance - 1))
    consistency = tuple(measurements[distance + check] ^ parities[check] for check in range(distance - 1))
    return parities + consistency, (measurements[0],)


@pytest.mark.parametrize("distance", [2, 3, 4, 5])
def test_circuit_dem_correctability_matches_explicit_statevector_faults(distance: int) -> None:
    """Check d=2t versus d=2t+1 using physical fault sets and a syndrome lookup table.

    Only preparations are noisy: each faulty location injects one X after PZ.
    These are independent binary events, so physical fault counts and DEM event
    counts have the same meaning. Enumerate all sets through weight two, including
    the empty set; never infer the oracle's logical effects from DEM mechanisms.
    """
    from pecos.qec import DetectorErrorModel, DistanceProblem

    num_qubits = 2 * distance - 1
    dem = DetectorErrorModel.from_circuit(
        _repetition_extraction_circuit(distance),
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.01,
    )
    outcomes = [
        (weight, _simulate_preparation_faults(distance, faults))
        for weight in range(3)
        for faults in combinations(range(num_qubits), weight)
    ]
    assert outcomes[0] == (0, ((0,) * (2 * (distance - 1)), (0,)))

    # Pin the individual effects too: an incorrect CNOT propagation must not pass
    # merely because the resulting DEM happens to have the same minimum distance.
    simulated_effects = {
        (tuple(i for i, bit in enumerate(detectors) if bit), tuple(i for i, bit in enumerate(observables) if bit))
        for weight, (detectors, observables) in outcomes
        if weight == 1
    }
    dem_effects = {
        (tuple(effect["detectors"]), tuple(effect["observables"])) for effect in dem.contribution_effect_summaries()
    }
    assert dem_effects == simulated_effects

    problem = DistanceProblem.from_dem(dem)
    graphlike = dem.graphlike_fault_distance()
    assert graphlike is not None
    assert graphlike.distance == distance
    for max_faults in range(3):
        decoder = {}
        unambiguous = True
        for weight, (detectors, observables) in outcomes:
            if weight <= max_faults and decoder.setdefault(detectors, observables) != observables:
                unambiguous = False
        assert unambiguous == (distance > 2 * max_faults)
        assert (graphlike.distance > 2 * max_faults) == unambiguous
        assert (dem.connected_cluster_fault_distance(2 * max_faults) is None) == unambiguous
        assert (dem.exhaustive_fault_distance(2 * max_faults) is None) == unambiguous
        assert (problem.certified_distance(2 * max_faults) is None) == unambiguous
