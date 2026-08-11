# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""NumPy-oracle tests for the DEM decomposition diagnostics example."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pecos.decoders
from pecos import asarray, dtypes

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


def _load_diagnostics_module() -> ModuleType:
    module_path = _repo_root() / "examples" / "surface" / "dem_decomposition_diagnostics.py"
    module_name = "_dem_decomposition_diagnostics_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load diagnostics module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _fixed_seed_dems() -> tuple[str, str]:
    rng = np.random.default_rng(20260811)
    probabilities = rng.uniform(0.001, 0.2, size=6)
    native_dem = "\n".join(
        [
            "detector(0, 0) D0",
            f"error({probabilities[0]:.17g}) D0 D2 ^ D4 L0",
            f"error({probabilities[1]:.17g}) D1 L0",
            f"error({probabilities[2]:.17g}) D3",
            f"error({probabilities[3]:.17g}) D3",
            "logical_observable L0",
        ],
    )
    stim_dem = "\n".join(
        [
            "detector(0, 0) D0",
            f"error({probabilities[4]:.17g}) D0 D2 D4 L0",
            f"error({probabilities[5]:.17g}) D5",
            "logical_observable L0",
        ],
    )
    return native_dem, stim_dem


def test_dem_stats_and_compare_raw_dems_match_captured_fixed_seed_values() -> None:
    diagnostics = _load_diagnostics_module()
    native_dem, stim_dem = _fixed_seed_dems()
    comparison_native_dem = "\n".join(native_dem.splitlines()[1:3])
    comparison_stim_dem = "\n".join(stim_dem.splitlines()[1:3])

    assert asdict(diagnostics.dem_stats(native_dem)) == {
        "error_lines": 4,
        "probability_sum": 0.34778671791333904,
        "separator_lines": 1,
        "hyperedge_lines": 1,
        "logical_lines": 2,
        "max_component_detectors": 2,
        "max_line_detectors": 3,
        "pure_logical_components": 0,
    }
    assert asdict(diagnostics.dem_stats(stim_dem)) == {
        "error_lines": 2,
        "probability_sum": 0.21691480573842542,
        "separator_lines": 0,
        "hyperedge_lines": 1,
        "logical_lines": 1,
        "max_component_detectors": 3,
        "max_line_detectors": 3,
        "pure_logical_components": 0,
    }
    assert asdict(diagnostics.compare_raw_dems(comparison_native_dem, comparison_stim_dem)) == {
        "native_errors": 2,
        "stim_errors": 2,
        "only_native": 1,
        "only_stim": 1,
        "common": 1,
        "max_abs_probability_diff": 0.04490873066068533,
        "max_rel_probability_diff": 0.44347431107696206,
        "l1_probability_diff": 0.21975757274400112,
    }


def test_dense_effect_arrays_matches_fixed_seed_numpy_oracle() -> None:
    diagnostics = _load_diagnostics_module()
    rng = np.random.default_rng(20260811)
    effects = {
        "D0 D3 L0": float(rng.uniform(0.001, 0.2)),
        "D1": float(rng.uniform(0.001, 0.2)),
        "D2 D4": float(rng.uniform(0.001, 0.2)),
    }

    expected_probabilities = np.array(list(effects.values()), dtype=float)
    expected_detection_events = np.zeros((3, 5), dtype=np.uint8)
    expected_detection_events[0, [0, 3]] = 1
    expected_detection_events[1, [1]] = 1
    expected_detection_events[2, [2, 4]] = 1
    expected_observable_flips = np.array([1, 0, 0], dtype=np.uint8)

    keys, probabilities, detection_events, observable_flips = diagnostics.dense_effect_arrays(effects)

    assert keys == list(effects)
    assert probabilities.dtype == dtypes.float64
    assert detection_events.dtype == dtypes.uint8
    assert observable_flips.dtype == dtypes.uint8
    assert probabilities.tolist() == expected_probabilities.tolist()
    assert detection_events.tolist() == expected_detection_events.tolist()
    assert observable_flips.tolist() == expected_observable_flips.tolist()


def test_true_observable_flips_matches_numpy_oracle() -> None:
    diagnostics = _load_diagnostics_module()
    rng = np.random.default_rng(20260811)
    numpy_cases = [
        rng.integers(0, 2, size=9, dtype=np.uint8),
        np.zeros((9, 0), dtype=np.uint8),
        rng.integers(0, 2, size=(9, 3), dtype=np.uint8),
    ]

    for numpy_case in numpy_cases:
        if numpy_case.ndim == 1:
            expected = numpy_case.astype(np.uint8)
        elif numpy_case.shape[1] == 0:
            expected = np.zeros(numpy_case.shape[0], dtype=np.uint8)
        else:
            expected = numpy_case[:, 0].astype(np.uint8)
        actual = diagnostics.true_observable_flips(numpy_case)
        assert actual.dtype == dtypes.uint8
        assert actual.tolist() == expected.tolist()


@dataclass(frozen=True)
class _FakeTesseractResult:
    observable_flips: list[int]


class _FakeTesseractDecoder:
    prediction_bits: ClassVar[list[int]] = []
    seen_rows: ClassVar[list[list[int]] | None] = None

    @classmethod
    def from_dem(cls, dem_text: str, *, preset: str, det_beam: int) -> _FakeTesseractDecoder:
        assert dem_text == "fixed-seed-dem"
        assert preset == "fast"
        assert det_beam == 7
        return cls()

    def decode_batch(self, rows: list[list[int]]) -> list[_FakeTesseractResult]:
        type(self).seen_rows = rows
        return [_FakeTesseractResult([bit]) for bit in type(self).prediction_bits]


class _FakePyMatchingDecoder:
    predictions: ClassVar[list[list[int]]] = []
    seen_flat: ClassVar[list[int] | None] = None
    expected_shots: ClassVar[int] = 0

    @classmethod
    def from_dem(cls, dem_text: str) -> _FakePyMatchingDecoder:
        assert dem_text == "fixed-seed-dem"
        return cls()

    @classmethod
    def from_dem_with_correlations(
        cls,
        dem_text: str,
        *,
        enable_correlations: bool,
    ) -> _FakePyMatchingDecoder:
        assert dem_text == "fixed-seed-dem"
        assert enable_correlations is True
        return cls()

    def decode_batch(self, flat: list[int], num_shots: int) -> list[list[int]]:
        type(self).seen_flat = flat
        assert num_shots == type(self).expected_shots
        return type(self).predictions


def test_decoder_predictions_and_reductions_match_fixed_seed_numpy_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = _load_diagnostics_module()
    rng = np.random.default_rng(20260811)
    shots = 19
    numpy_events = rng.integers(0, 2, size=(shots, 6), dtype=np.uint8)
    numpy_observables = rng.integers(0, 2, size=(shots, 1), dtype=np.uint8)
    tesseract_bits = rng.integers(0, 2, size=shots, dtype=np.uint8)
    pymatching_bits = rng.integers(0, 2, size=shots, dtype=np.uint8)
    pymatching_predictions = [[] if index % 5 == 0 else [int(bit)] for index, bit in enumerate(pymatching_bits)]
    normalized_pymatching_bits = np.array(
        [prediction[0] if prediction else 0 for prediction in pymatching_predictions],
        dtype=np.uint8,
    )

    _FakeTesseractDecoder.prediction_bits = tesseract_bits.tolist()
    _FakeTesseractDecoder.seen_rows = None
    _FakePyMatchingDecoder.predictions = pymatching_predictions
    _FakePyMatchingDecoder.seen_flat = None
    _FakePyMatchingDecoder.expected_shots = shots
    monkeypatch.setattr(pecos.decoders, "TesseractDecoder", _FakeTesseractDecoder)
    monkeypatch.setattr(pecos.decoders, "PyMatchingDecoder", _FakePyMatchingDecoder)

    events = asarray(numpy_events, dtype=dtypes.uint8)
    observables = asarray(numpy_observables, dtype=dtypes.uint8)
    expected = numpy_observables[:, 0].astype(np.uint8)

    tesseract_errors = diagnostics.decode_with_tesseract(
        "fixed-seed-dem",
        events,
        observables,
        beam=7,
    )
    pymatching_errors = diagnostics.decode_with_pymatching(
        "fixed-seed-dem",
        events,
        observables,
        correlated=True,
    )

    assert tesseract_errors == int(np.sum(tesseract_bits != expected)) == 9
    assert pymatching_errors == int(np.sum(normalized_pymatching_bits != expected)) == 13
    assert _FakeTesseractDecoder.seen_rows == numpy_events.tolist()
    assert _FakePyMatchingDecoder.seen_flat == numpy_events.astype(np.uint8).flatten().tolist()
    assert _FakePyMatchingDecoder.seen_flat is not None
    assert all(type(value) is int for value in _FakePyMatchingDecoder.seen_flat)


def test_two_fault_pair_analysis_matches_numpy_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics = _load_diagnostics_module()
    probabilities = np.array([0.03125, 0.0625, 0.125, 0.25], dtype=float)
    native_raw = "\n".join(
        [
            f"error({probabilities[0]:.17g}) D0 L0",
            f"error({probabilities[1]:.17g}) D1",
            f"error({probabilities[2]:.17g}) D0 D2",
            f"error({probabilities[3]:.17g}) D2 L0",
        ],
    )
    variant_offsets = {
        ("native", False): 0,
        ("native", True): 1,
        ("stim", False): 1,
        ("stim", True): 0,
        ("terminal", False): 0,
        ("terminal", True): 1,
    }

    def predictions(events: object, offset: int) -> object:
        numpy_events = np.asarray(events.tolist(), dtype=np.uint8)
        bits = (np.sum(numpy_events, axis=1) + offset) % 2
        return asarray(bits.astype(np.uint8), dtype=dtypes.uint8)

    def fake_tesseract(dem_text: str, detection_events: object, *, beam: int) -> object:
        assert dem_text == native_raw
        assert beam == 5
        return predictions(detection_events, 0)

    def fake_pymatching(dem_text: str, detection_events: object, *, correlated: bool) -> object:
        return predictions(detection_events, variant_offsets[(dem_text, correlated)])

    monkeypatch.setattr(diagnostics, "tesseract_predictions", fake_tesseract)
    monkeypatch.setattr(diagnostics, "pymatching_predictions", fake_pymatching)

    actual = diagnostics.two_fault_pair_analysis(
        native_raw=native_raw,
        native_decomposed="native",
        stim_decomposed="stim",
        terminal_decomposed="terminal",
        max_effects=4,
    )

    event_rows = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.uint8)
    observable_flips = np.array([1, 0, 0, 1], dtype=np.uint8)
    pair_rows = []
    pair_observables = []
    pair_weights = []
    for left in range(4):
        for right in range(left + 1, 4):
            pair_rows.append(event_rows[left] ^ event_rows[right])
            pair_observables.append(observable_flips[left] ^ observable_flips[right])
            pair_weights.append(probabilities[left] * probabilities[right])
    pair_rows_array = np.asarray(pair_rows, dtype=np.uint8)
    pair_observables_array = np.asarray(pair_observables, dtype=np.uint8)
    pair_weights_array = np.asarray(pair_weights, dtype=float)
    reference = (np.sum(pair_rows_array, axis=1) % 2).astype(np.uint8)
    expected_predictions = {
        "native_raw_tesseract_b5": reference,
        "native_decomp_pymatching": reference,
        "native_decomp_pymatching_correlated": 1 - reference,
        "stim_decomp_pymatching": 1 - reference,
        "stim_decomp_pymatching_correlated": reference,
        "terminal_decomp_pymatching": reference,
        "terminal_decomp_pymatching_correlated": 1 - reference,
    }
    total_weight = float(np.sum(pair_weights_array))
    expected = []
    for name, predicted in expected_predictions.items():
        wrong = predicted != pair_observables_array
        disagree = predicted != reference
        wrong_mass = float(np.sum(pair_weights_array[wrong]))
        disagree_mass = float(np.sum(pair_weights_array[disagree]))
        expected.append(
            {
                "decoder": name,
                "pair_probability_mass": total_weight,
                "wrong_probability_mass": wrong_mass,
                "wrong_probability_fraction": wrong_mass / total_weight,
                "disagree_tesseract_probability_mass": disagree_mass,
                "disagree_tesseract_probability_fraction": disagree_mass / total_weight,
                "wrong_count": int(np.sum(wrong)),
                "disagree_tesseract_count": int(np.sum(disagree)),
            },
        )

    assert actual is not None
    assert [asdict(summary) for summary in actual] == expected
