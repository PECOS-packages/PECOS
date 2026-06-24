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

"""Subset simulation through the sim_neo Python bindings.

Mirrors the Rust tool API: score/failure are Python callables receiving the
sample's measurement bits as list[int]; the estimate arrives in
result.subset (rows are empty for subset runs).
"""

from __future__ import annotations

import pytest

pecos_rslib_exp = pytest.importorskip("pecos_rslib_exp")

from pecos.quantum import TickCircuit  # noqa: E402
from pecos_rslib_exp import (  # noqa: E402
    sim_neo,
    stabilizer,
    statevec,
    subset_simulation,
)


def three_h_circuit() -> TickCircuit:
    tc = TickCircuit()
    tc.tick().h([0, 1, 2])
    tc.tick().mz([0, 1, 2])
    return tc


def x_circuit() -> TickCircuit:
    tc = TickCircuit()
    tc.tick().x([0])
    tc.tick().mz([0])
    return tc


def test_subset_estimates_known_probability() -> None:
    # P(all three H measurements give 1) = 1/8.
    result = (
        sim_neo(three_h_circuit())
        .quantum(stabilizer())
        .sampling(
            subset_simulation(2000)
            .score(lambda bits: float(sum(bits)))
            .failure(lambda bits: all(b == 1 for b in bits)),
        )
        .seed(42)
        .run()
    )

    assert result.num_shots == 0, "Subset runs produce an estimate, not rows"
    subset = result.subset
    assert subset is not None
    assert 0.08 <= subset.probability <= 0.20
    assert subset.total_samples >= 2000
    lower, upper = subset.confidence_interval_95()
    assert lower <= subset.probability <= upper
    assert len(subset.levels()) >= 1


def test_subset_certain_event() -> None:
    result = (
        sim_neo(x_circuit())
        .auto()
        .sampling(
            subset_simulation(200)
            .score(lambda bits: float(sum(bits)))
            .failure(lambda bits: bits[0] == 1),
        )
        .seed(7)
        .run()
    )
    assert abs(result.subset.probability - 1.0) < 1e-9


def test_subset_deterministic_with_seed() -> None:
    def run() -> float:
        return (
            sim_neo(three_h_circuit())
            .auto()
            .sampling(
                subset_simulation(500)
                .score(lambda bits: float(sum(bits)))
                .failure(lambda bits: all(b == 1 for b in bits)),
            )
            .seed(99)
            .run()
            .subset.probability
        )

    assert run() == run()


def test_subset_requires_score_and_failure() -> None:
    with pytest.raises(ValueError, match=r"requires both \.score\(\.\.\) and \.failure\(\.\.\)"):
        sim_neo(three_h_circuit()).auto().sampling(subset_simulation(100)).run()


def test_subset_rejects_statevec_backend() -> None:
    with pytest.raises(ValueError, match="only the stabilizer"):
        sim_neo(three_h_circuit()).quantum(statevec()).sampling(
            subset_simulation(100)
            .score(lambda bits: float(sum(bits)))
            .failure(lambda bits: all(b == 1 for b in bits)),
        ).run()


def test_subset_callable_exception_propagates() -> None:
    def bad_score(_bits: list[int]) -> float:
        msg = "score exploded"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="score exploded"):
        sim_neo(three_h_circuit()).auto().sampling(
            subset_simulation(50).score(bad_score).failure(lambda _bits: False),
        ).run()


def _ones_score(bits: list[int]) -> float:
    return float(sum(bits))


def _all_ones(bits: list[int]) -> bool:
    return all(b == 1 for b in bits)


def test_subset_multilevel_without_opt_in_raises() -> None:
    """max_levels > 1 engages the biased multi-level estimator and must be
    refused unless explicitly acknowledged (mirrors the Rust guard)."""
    with pytest.raises(ValueError, match="biased upward"):
        sim_neo(three_h_circuit()).auto().sampling(
            subset_simulation(100).score(_ones_score).failure(_all_ones).max_levels(5),
        ).run()


def test_subset_multilevel_runs_with_opt_in() -> None:
    result = (
        sim_neo(three_h_circuit())
        .auto()
        .sampling(
            subset_simulation(500)
            .score(_ones_score)
            .failure(_all_ones)
            .max_levels(5)
            .allow_biased_multilevel(),
        )
        .seed(42)
        .run()
    )
    assert result.subset is not None
