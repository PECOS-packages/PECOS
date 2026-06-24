# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""sim_neo Python bindings mirror the Rust builder's explicit-by-default rules.

No silent backend, shot count, or seed. Missing required config fails with
the same messages as the Rust builder; .auto() opts into automatic selection.
"""

from __future__ import annotations

import pytest

pecos_rslib_exp = pytest.importorskip("pecos_rslib_exp")

from pecos.quantum import TickCircuit  # noqa: E402
from pecos_rslib_exp import monte_carlo, sim_neo, stabilizer  # noqa: E402


def one_qubit_circuit() -> TickCircuit:
    tc = TickCircuit()
    tc.tick().x([0])
    tc.tick().mz([0])
    return tc


def test_missing_quantum_backend_is_error() -> None:
    with pytest.raises(ValueError, match="No quantum backend set"):
        sim_neo(one_qubit_circuit()).sampling(monte_carlo(2)).run()


def test_missing_sampling_is_error() -> None:
    with pytest.raises(ValueError, match="No sampling strategy set"):
        sim_neo(one_qubit_circuit()).quantum(stabilizer()).run()


def test_auto_selects_stabilizer_backend() -> None:
    result = sim_neo(one_qubit_circuit()).auto().sampling(monte_carlo(3)).seed(7).run()
    assert result.num_shots == 3
    assert [list(shot) for shot in result] == [[1], [1], [1]]


def test_explicit_quantum_overrides_auto() -> None:
    result = (
        sim_neo(one_qubit_circuit())
        .auto()
        .quantum(stabilizer())
        .sampling(monte_carlo(2))
        .seed(7)
        .run()
    )
    assert result.num_shots == 2


def test_deprecated_shots_forwarder_warns_and_works() -> None:
    with pytest.deprecated_call():
        builder = sim_neo(one_qubit_circuit()).quantum(stabilizer()).shots(2)
    result = builder.seed(7).run()
    assert result.num_shots == 2


def test_deprecated_shots_conflicts_with_sampling() -> None:
    with pytest.deprecated_call():
        builder = sim_neo(one_qubit_circuit()).quantum(stabilizer()).shots(2)
    with pytest.raises(ValueError, match=r"deprecated \.shots\(\) cannot be combined"):
        builder.sampling(monte_carlo(5)).run()
