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

"""Python binding tests for generic coloration-scheduled CSS memory circuits."""

import pytest
from pecos.qec import coloration_memory_circuit
from pecos.quantum import ParityCheckMatrix

STEANE_H = [
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]


def test_coloration_memory_binding_is_deterministic_and_reports_depth() -> None:
    h = ParityCheckMatrix(STEANE_H)

    first = coloration_memory_circuit(h, h, 2, "Z")
    second = coloration_memory_circuit(h, h, 2, "z")

    assert repr(first) == repr(second)
    assert first.get_meta("circuit_type") == "coloration_css_memory"
    assert first.get_meta("num_data_qubits") == "7"
    assert first.get_meta("num_ancilla_qubits") == "6"
    assert first.get_meta("num_detectors") == "12"
    assert first.get_meta("num_observables") == "1"
    assert first.get_meta("x_coloration_depth") == "4"
    assert first.get_meta("z_coloration_depth") == "4"
    assert first.get_meta("entangling_depth_per_cycle") == "8"
    assert first.get_meta("syndrome_cycle_depth") == "11"
    assert first.num_ticks() == 23


def test_coloration_memory_binding_rejects_invalid_inputs() -> None:
    h = ParityCheckMatrix(STEANE_H)
    nonorthogonal = ParityCheckMatrix([[1, 0, 0, 0, 0, 0, 0]])

    with pytest.raises(ValueError, match="basis must be"):
        coloration_memory_circuit(h, h, 2, "Y")
    with pytest.raises(ValueError, match="at least one syndrome cycle"):
        coloration_memory_circuit(h, h, 0, "Z")
    with pytest.raises(ValueError, match="not orthogonal"):
        coloration_memory_circuit(h, nonorthogonal, 2, "Z")
