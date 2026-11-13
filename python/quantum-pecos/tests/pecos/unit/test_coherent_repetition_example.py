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

"""Record integrity and actual-runtime checks for the coherent model example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pecos.experimental.detector_models import RepetitionConfig, RepetitionModel

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="module")
def example() -> ModuleType:
    """Load the standalone example through its public file location."""
    path = Path(__file__).resolve().parents[5] / "examples" / "coherent_repetition_detector_model.py"
    name = "_coherent_repetition_example_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_repeated_check_tags_retain_chronological_values(example: ModuleType) -> None:
    """A dictionary conversion would silently erase the first two rounds."""
    record = [
        ("check_0", 0),
        ("check_1", 1),
        ("check_0", 1),
        ("check_1", 0),
        ("check_0", 1),
        ("check_1", 1),
        ("data_0", 0),
        ("data_1", 1),
        ("data_2", 0),
    ]
    checks, final_data = example.parse_record(record)
    assert checks == [[0, 1], [1, 0], [1, 1]]
    assert final_data == [0, 1, 0]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonbinary", "unexpected"])
def test_malformed_record_is_rejected(example: ModuleType, mutation: str) -> None:
    """Missing, repeated terminal, and nonbinary values must not be coerced."""
    record = [("check_0", [0, 0, 0]), ("check_1", [0, 0, 0]), ("data_0", 0), ("data_1", 0), ("data_2", 0)]
    if mutation == "missing":
        record.pop()
    elif mutation == "duplicate":
        record.append(("data_0", 1))
    elif mutation == "nonbinary":
        record[0] = ("check_0", [0, 2, 0])
    else:
        record.append(("hidden_leakage", 1))
    with pytest.raises(ValueError, match=r"Each shot|Nonbinary|Unexpected"):
        example.parse_record(record)


def test_target_complement_changes_only_scoring(example: ModuleType) -> None:
    """Complemented raw readout cannot change decisions for identical checks."""
    model = RepetitionModel(RepetitionConfig())
    checks = [[[0, 0], [0, 0], [0, 0]], [[1, 0], [1, 1], [0, 1]]]
    data = [[0, 0, 0], [0, 0, 1]]
    complemented = [[1 - value for value in row] for row in data]
    original = example.score_records(model, checks, data)
    changed = example.score_records(model, checks, complemented)
    assert original["different_corrections"] == changed["different_corrections"]
    assert original["canonical_failures"] + changed["canonical_failures"] == len(checks)
    assert original["coherent_failures"] + changed["coherent_failures"] == len(checks)


def test_extra_final_parity_noise_is_rejected(example: ModuleType) -> None:
    """The example's ideal final-boundary assumption is checked explicitly."""
    with pytest.raises(ValueError, match="Final data"):
        example.score_records(RepetitionModel(RepetitionConfig()), [[[0, 0]] * 3], [[1, 0, 0]])


def test_zero_failures_retains_nonzero_uncertainty(example: ModuleType) -> None:
    """A zero observed failure count must not produce a zero-width interval."""
    low, high = example.wilson_interval(0, 2000)
    assert low == 0
    assert high == pytest.approx(0.0019170472812529346)
    low, high = example.wilson_interval(2000, 2000)
    assert low == pytest.approx(1 - 0.0019170472812529346)
    assert high == 1


def test_wilson_interval_known_count(example: ModuleType) -> None:
    """Check an independently evaluated score interval for a balanced count."""
    low, high = example.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038315303659956)
    assert high == pytest.approx(0.5961684696340044)


@pytest.mark.parametrize("counts", [(-1, 10), (11, 10), (0, 0), (0.5, 10), (True, 10)])
def test_wilson_interval_rejects_invalid_counts(example: ModuleType, counts: tuple[int, int]) -> None:
    """Rates cannot silently use invalid denominators or non-count inputs."""
    with pytest.raises(ValueError, match="Wilson intervals require"):
        example.wilson_interval(*counts)


@pytest.mark.slow
@pytest.mark.parametrize("basis", ["bit", "phase"])
def test_real_guppy_selene_quest_cohort(example: ModuleType, basis: str) -> None:
    """Compile and simulate each basis using actual ordinary measurement tags."""
    report = example.run_example(basis=basis, shots=64, seed=20261113)
    assert report["basis"] == basis
    assert report["shots"] == 64
    assert 0 <= report["canonical_failures"] <= 64
    assert 0 <= report["coherent_failures"] <= 64
    assert report["maximum_live_qubits"] == 4
    assert report["inference_input"] == "ordinary check history only"
    assert len(report["observed_records_sha256"]) == 64
    assert report["versions"]["selene-sim"] is not None
