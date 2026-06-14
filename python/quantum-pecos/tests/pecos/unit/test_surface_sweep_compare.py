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

"""Unit tests for examples/surface/compare_surface_sweep_json.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "Justfile").is_file() and (candidate / "examples").is_dir():
            return candidate
    msg = f"Could not locate repo root above {cur}"
    raise RuntimeError(msg)


_COMPARE_MODULE_NAME = "_surface_sweep_compare_under_test"


def _load_compare_module() -> ModuleType:
    example_path = _repo_root() / "examples" / "surface" / "compare_surface_sweep_json.py"
    spec = importlib.util.spec_from_file_location(_COMPARE_MODULE_NAME, example_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load comparison module from {example_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_COMPARE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_COMPARE_MODULE_NAME, None)
        raise
    return module


@pytest.fixture(scope="module")
def compare() -> ModuleType:
    return _load_compare_module()


def _point(*, distance: int, p: float, errors: int, shots: int = 1000) -> dict[str, object]:
    return {
        "backend": "native_sampler",
        "basis": "Z",
        "distance": distance,
        "physical_error_rate": p,
        "total_rounds": distance,
        "num_logical_errors": errors,
        "num_shots": shots,
    }


def _write_sweep(path: Path, points: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"points": points}), encoding="utf-8")


def test_report_defaults_to_jeffreys_and_omits_cross_distance_pooled(
    compare: ModuleType,
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_sweep(left, [_point(distance=3, p=0.004, errors=5), _point(distance=5, p=0.004, errors=2)])
    _write_sweep(right, [_point(distance=3, p=0.004, errors=6), _point(distance=5, p=0.004, errors=1)])

    report = compare.build_report(
        left,
        right,
        left_label="CX",
        right_label="SZZ",
        include_ci=True,
        interval_method="jeffreys",
        include_cross_distance_pooled=False,
    )

    assert "- intervals: jeffreys 95%" in report
    assert "- z-scores: descriptive unpooled Wald z-scores" in report
    assert "## Aggregate Over Physical Error Rates" in report
    assert "## Pooled Across Distances" not in report


def test_cross_distance_pooled_table_is_explicitly_labeled(
    compare: ModuleType,
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_sweep(left, [_point(distance=3, p=0.004, errors=5), _point(distance=5, p=0.004, errors=2)])
    _write_sweep(right, [_point(distance=3, p=0.004, errors=6), _point(distance=5, p=0.004, errors=1)])

    report = compare.build_report(
        left,
        right,
        left_label="CX",
        right_label="SZZ",
        include_ci=True,
        interval_method="jeffreys",
        include_cross_distance_pooled=True,
    )

    assert "## Pooled Across Distances By Backend And Basis" in report
    assert "not a scaling or threshold statement" in report


def test_duplicate_point_keys_raise(compare: ModuleType, tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    point = _point(distance=3, p=0.004, errors=5)
    _write_sweep(duplicate, [point, point])

    with pytest.raises(ValueError, match="duplicate point key"):
        compare.load_points(duplicate)


def test_wilson_interval_remains_available(compare: ModuleType) -> None:
    low, high = compare.binomial_interval(0, 100, "wilson")
    assert low == pytest.approx(0.0, abs=1e-15)
    assert 0.0 < high < 0.1
