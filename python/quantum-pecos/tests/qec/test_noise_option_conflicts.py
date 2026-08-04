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

"""Base-idle-channel combinations must fail loud, not resolve silently (issue #426).

`set_t1_t2` makes T1/T2 the base channel that shadows ``p_idle``, and
`set_idle_rz` zeroes ``p_idle`` and overwrites T1/T2. Each combination used to
discard a caller-supplied rate with no signal.
"""

from __future__ import annotations

import pytest
from pecos.quantum import TickCircuit
from pecos.qec import DemSampler, DetectorErrorModel

_GATE_NOISE = {"p1": 0.001, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}


def _circuit() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().idle(1, [0])
    circuit.tick().mz_with_ids([0], [0])
    circuit.set_meta("num_measurements", "1")
    circuit.set_meta("detectors", '[{"id": 0, "records": [-1]}]')
    return circuit


@pytest.mark.parametrize(
    ("conflict", "message"),
    [
        pytest.param(
            {"p_idle": 0.01, "t1": 100.0, "t2": 50.0},
            "T1/T2 channel replaces",
            id="p_idle-with-t1t2",
        ),
        pytest.param(
            {"idle_rz": 0.01, "p_idle": 0.01},
            "coherent RZ conversion replaces",
            id="idle_rz-with-p_idle",
        ),
        pytest.param(
            {"idle_rz": 0.01, "t1": 100.0, "t2": 50.0},
            "overwrites the T1/T2 channel",
            id="idle_rz-with-t1t2",
        ),
    ],
)
def test_from_circuit_rejects_shadowed_idle_channels(conflict: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DetectorErrorModel.from_circuit(_circuit(), **conflict, **_GATE_NOISE)


def test_dem_sampler_shares_the_same_guard() -> None:
    # The guard lives in the shared noise-option helper, so every ingest path
    # is protected -- not only DetectorErrorModel.from_circuit.
    with pytest.raises(ValueError, match="T1/T2 channel replaces"):
        DemSampler.from_circuit(_circuit(), p_idle=0.01, t1=100.0, t2=50.0, **_GATE_NOISE)


@pytest.mark.parametrize(
    "idle_noise",
    [
        pytest.param({"p_idle": 0.01}, id="p_idle-alone"),
        pytest.param({"t1": 100.0, "t2": 50.0}, id="t1t2-alone"),
        pytest.param({"idle_rz": 0.01}, id="idle_rz-alone"),
    ],
)
def test_each_base_idle_channel_alone_still_builds(idle_noise: dict[str, float]) -> None:
    dem = DetectorErrorModel.from_circuit(_circuit(), **idle_noise, **_GATE_NOISE)

    assert dem.num_detectors == 1
