# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""NumPy-oracle tests for the graphlike DEM projection benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pecos.decoders

if TYPE_CHECKING:
    from types import ModuleType

    import pytest


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "Justfile").is_file() and (candidate / "examples").is_dir():
            return candidate
    msg = f"Could not locate repo root above {current}"
    raise RuntimeError(msg)


def _load_benchmark_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    surface_examples = _repo_root() / "examples" / "surface"
    monkeypatch.syspath_prepend(str(surface_examples))
    module_path = surface_examples / "graphlike_dem_projection_benchmark.py"
    module_name = "_graphlike_dem_projection_benchmark_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load benchmark module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class _FakeDecoder:
    predictions: ClassVar[list[list[int]]] = []
    seen_flat: ClassVar[list[int] | None] = None
    expected_shots: ClassVar[int] = 0

    @classmethod
    def from_dem_with_correlations(cls, dem_text: str, *, enable_correlations: bool) -> _FakeDecoder:
        assert dem_text == "fixed-seed-dem"
        assert enable_correlations is True
        return cls()

    def decode_batch(self, flat: list[int], num_shots: int) -> list[list[int]]:
        type(self).seen_flat = flat
        assert num_shots == type(self).expected_shots
        return type(self).predictions


def test_logical_error_count_matches_fixed_seed_numpy_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _load_benchmark_module(monkeypatch)
    seed = 20260810
    shots = 17
    rng = np.random.default_rng(seed)
    detection_events = rng.integers(0, 2, size=(shots, 6), dtype=np.uint8).astype(bool)
    observable_flips = rng.integers(0, 2, size=(shots, 1), dtype=np.uint8).astype(bool)
    prediction_bits = rng.integers(0, 2, size=shots, dtype=np.uint8)
    predictions = [[] if index % 4 == 0 else [int(bit)] for index, bit in enumerate(prediction_bits)]

    _FakeDecoder.predictions = predictions
    _FakeDecoder.seen_flat = None
    _FakeDecoder.expected_shots = shots
    monkeypatch.setattr(pecos.decoders, "PyMatchingDecoder", _FakeDecoder)

    logical_errors, _, _ = benchmark.decode_with_correlated_pymatching(
        "fixed-seed-dem",
        detection_events,
        observable_flips,
    )

    predicted = np.array([prediction[0] if prediction else 0 for prediction in predictions], dtype=np.uint8)
    expected = observable_flips[:, 0].astype(np.uint8)
    expected_flat = detection_events.astype(np.uint8).flatten().tolist()
    assert logical_errors == int(np.sum(predicted != expected)) == 12
    assert _FakeDecoder.seen_flat == expected_flat
    assert _FakeDecoder.seen_flat is not None
    assert all(type(value) is int for value in _FakeDecoder.seen_flat)
