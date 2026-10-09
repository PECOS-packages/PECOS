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

import re
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


@pytest.mark.parametrize("factory", [SparseStab, Stabilizer], ids=["SparseStab", "Stab"])
@pytest.mark.parametrize("spelling", ["MZ", "Measure", "measure Z", "Measure +Z", "MZForced"])
def test_random_sentinel_matches_unforced_z(factory: Callable[..., object], spelling: str) -> None:
    """The random sentinel preserves outcomes, state, and RNG progress for each seed."""
    outcomes = set()
    for seed in range(16):
        actual = factory(2, seed=seed)
        expected = factory(2, seed=seed)
        for state in (actual, expected):
            state.run_gate("H", {0})
            state.run_gate("CX", {(0, 1)})
        result = actual.run_gate(spelling, {0}, forced_outcome=-1)
        reference = expected.run_gate("MZ", {0})
        assert result.get(0, 0) == reference.get(0, 0)
        outcomes.add(result.get(0, 0))
        assert actual.stab_tableau() == expected.stab_tableau()
        assert actual.destab_tableau() == expected.destab_tableau()
        # Further random measurements detect accidental RNG consumption differences.
        for _ in range(4):
            for state in (actual, expected):
                state.run_gate("H", {0})
            assert actual.run_gate("MZ", {0}) == expected.run_gate("MZ", {0})
    assert outcomes == {0, 1}


@pytest.mark.parametrize("factory", [SparseStab, Stabilizer], ids=["SparseStab", "Stab"])
@pytest.mark.parametrize(
    "spelling",
    ["MZ", "Measure", "measure Z", "Measure +Z", "MZForced", "PZ", "Init", "PZForced"],
)
@pytest.mark.parametrize("forced_outcome", [-2, 2, None, 0.5, "1", 2**64])
def test_invalid_forced_z_outcome(
    factory: Callable[..., object],
    spelling: str,
    forced_outcome: object,
) -> None:
    """Invalid values fail explicitly before changing the simulator state."""
    state = factory(1, seed=0)
    state.run_gate("H", {0})
    before = state.stab_tableau()
    with pytest.raises(ValueError, match=rf"{re.escape(spelling)}.*forced_outcome"):
        state.run_gate(spelling, {0}, forced_outcome=forced_outcome)
    assert state.stab_tableau() == before
