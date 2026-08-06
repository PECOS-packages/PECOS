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

import pytest


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
