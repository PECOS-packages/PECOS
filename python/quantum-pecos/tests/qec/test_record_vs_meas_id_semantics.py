# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Records count measurement emission order; meas_ids name stable identities.

Circuit-backed builders resolve both forms through the shared reader and accept
co-present references only when they name the same measurement multiset. Stamps
can differ from emission positions: the first measurement of [2, 0, 1] has id 2,
so records[-3] names id 2, not the smallest id.
"""

from __future__ import annotations

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import cx, measure, qubit
from pecos.qec import DetectorErrorModel
from pecos.quantum import TickCircuit

_NOISE = {"p1": 0.001, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}


@guppy
def _five_measurement_program() -> None:
    d0, d1, d2 = qubit(), qubit(), qubit()
    a0, a1 = qubit(), qubit()
    cx(d0, a0)
    cx(d1, a0)
    cx(d1, a1)
    cx(d2, a1)
    result("s0", measure(a0).read())
    result("s1", measure(a1).read())
    result("m0", measure(d0).read())
    result("m1", measure(d1).read())
    result("m2", measure(d2).read())


def _stamped_circuit() -> TickCircuit:
    # Stamped ids deliberately differ from execution position.
    circuit = TickCircuit()
    circuit.tick().pz([0, 1, 2])
    circuit.tick().mz_with_ids([2, 0, 1], [2, 0, 1])
    circuit.set_meta("num_measurements", "3")
    return circuit


def _accepts(detectors_json: str, *, circuit: TickCircuit | None = None) -> bool:
    """True when the builder accepts the co-present references as redundant."""
    if circuit is None:
        try:
            DetectorErrorModel.from_guppy(
                _five_measurement_program,
                num_qubits=5,
                detectors_json=detectors_json,
                **_NOISE,
            )
        except ValueError:
            return False
        return True
    circuit.set_meta("detectors", detectors_json)
    try:
        DetectorErrorModel.from_circuit(circuit, **_NOISE)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("offset_from_end", range(1, 6))
def test_traced_records_agree_with_meas_ids(offset_from_end: int) -> None:
    num_measurements = 5
    expected_meas_id = num_measurements - offset_from_end
    detectors = f'[{{"id":0,"records":[-{offset_from_end}],"meas_ids":[{expected_meas_id}]}}]'

    assert _accepts(detectors), f"records[-{offset_from_end}] should name meas_id {expected_meas_id}"


def test_traced_records_reject_every_other_meas_id() -> None:
    # Guards against an accept-everything redundancy check.
    mismatched = [
        meas_id
        for meas_id in range(5)
        if meas_id != 0 and _accepts(f'[{{"id":0,"records":[-5],"meas_ids":[{meas_id}]}}]')
    ]

    assert mismatched == []


def test_stamped_circuit_records_agree_with_meas_ids() -> None:
    # The batch emits q2/id2, q0/id0, q1/id1; -3 names the first, id 2.
    assert _accepts('[{"id":0,"records":[-3],"meas_ids":[2]}]', circuit=_stamped_circuit())
    assert not _accepts('[{"id":0,"records":[-3],"meas_ids":[0]}]', circuit=_stamped_circuit())
