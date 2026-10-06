# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Canonical Z measurements use the same forced path as their aliases (issue #801)."""

from collections.abc import Callable

import pytest
from pecos_rslib.simulators import SparseStab, Stabilizer


@pytest.mark.parametrize("factory", [SparseStab, Stabilizer], ids=["SparseStab", "Stab"])
@pytest.mark.parametrize("forced_outcome", [0, 1])
def test_forced_mz_matches_measure(factory: Callable[..., object], forced_outcome: int) -> None:
    """Both spellings select the requested branch and leave the measured state there."""
    for seed in range(16):
        outcomes = []
        for spelling in ("MZ", "Measure"):
            state = factory(1, seed=seed)
            state.run_gate("H", {0})
            result = state.run_gate(spelling, {0}, forced_outcome=forced_outcome)
            outcomes.append(result.get(0, 0))
            assert state.run_gate("MZ", {0}).get(0, 0) == forced_outcome
        assert outcomes == [forced_outcome, forced_outcome]
