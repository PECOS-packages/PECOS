# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""NumPy-oracle tests for the native DEM threshold-sweep example."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

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


def _load_sweep_module() -> ModuleType:
    module_path = _repo_root() / "examples" / "surface" / "native_dem_threshold_sweep.py"
    module_name = "_native_dem_threshold_sweep_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load threshold-sweep module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _captured_points(sweep: ModuleType) -> list[Any]:
    return [
        sweep.SweepPoint(
            backend="native_sampler",
            distance=5,
            basis="Z",
            physical_error_rate=0.006,
            total_rounds=rounds,
            num_shots=2000,
            num_logical_errors=errors,
            num_raw_errors=None,
            logical_error_rate=errors / 2000,
            raw_error_rate=None,
        )
        for rounds, errors in [(10, 91), (12, 111), (14, 137), (15, 144)]
    ]


def _private(sweep: ModuleType, name: str) -> Any:
    return getattr(sweep, name)


def test_deterministic_fit_values_match_captured_numpy_baseline() -> None:
    sweep = _load_sweep_module()
    summary = asdict(_private(sweep, "_fit_summary_from_points")(_captured_points(sweep)))
    for bootstrap_field in [
        "fitted_logical_error_rate_per_round_ci_low",
        "fitted_logical_error_rate_per_round_ci_high",
        "fitted_projected_logical_error_rate_over_d_rounds_ci_low",
        "fitted_projected_logical_error_rate_over_d_rounds_ci_high",
    ]:
        summary.pop(bootstrap_field)

    assert summary == {
        "backend": "native_sampler",
        "basis": "Z",
        "distance": 5,
        "fit_root_mean_square_error": 0.0021243697782072015,
        "fitted_logical_error_rate_per_round": 0.005051183185870917,
        "fitted_projected_logical_error_rate_over_d_rounds": 0.024750756037684896,
        "num_shots_per_round_point": 2000,
        "observed_logical_error_counts": (91, 111, 137, 144),
        "observed_logical_error_rate_lower_bounds": (
            0.03720527135020145,
            0.04629144858684902,
            0.05823636538977862,
            0.06147313331726136,
        ),
        "observed_logical_error_rate_upper_bounds": (
            0.05553732462845745,
            0.06641280644618483,
            0.08041804641394268,
            0.08416785915549113,
        ),
        "observed_logical_error_rates": (0.0455, 0.0555, 0.0685, 0.072),
        "observed_raw_error_rates": (None, None, None, None),
        "physical_error_rate": 0.006,
        "round_values": (10, 12, 14, 15),
    }


def test_native_array_transforms_match_numpy_exactly() -> None:
    sweep = _load_sweep_module()
    flat_values = list(range(12))
    reshaped = _private(sweep, "_reshape_round_values")(flat_values, 3, 4, "synx")

    assert [row.tolist() for row in reshaped] == np.asarray(flat_values, dtype=np.uint8).reshape(3, 4).tolist()
    assert tuple(
        _private(sweep, "_duration_rounds_for_distance")(
            5,
            explicit_multipliers=None,
            duration_min_multiplier=2.0,
            duration_max_multiplier=3.0,
            duration_num_points=4,
        ),
    ) == (10, 12, 13, 15)
    assert [float(value) for value in sweep.linspace(0.003, 0.009, 4)] == np.linspace(0.003, 0.009, 4).tolist()


class _FakePyMatchingDecoder:
    predictions: ClassVar[list[list[int]]] = []
    seen_flat: ClassVar[list[int] | None] = None

    def decode_batch(self, flat: list[int], num_shots: int) -> list[list[int]]:
        type(self).seen_flat = flat
        assert num_shots == len(type(self).predictions)
        return type(self).predictions


def test_batch_decoding_matches_numpy_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = _load_sweep_module()
    numpy_events = np.array(
        [[0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    numpy_observables = np.array([[0], [0], [1], [1], [1]], dtype=np.uint8)
    _FakePyMatchingDecoder.predictions = [[0], [1], [], [1], [0]]
    _FakePyMatchingDecoder.seen_flat = None
    monkeypatch.setattr(pecos.decoders, "PyMatchingDecoder", _FakePyMatchingDecoder)

    logical_errors = _private(sweep, "_decode_all_shots")(
        _FakePyMatchingDecoder(),
        asarray(numpy_events, dtype=dtypes.uint8),
        asarray(numpy_observables, dtype=dtypes.uint8),
        len(numpy_events),
    )
    normalized_predictions = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

    assert logical_errors == int(np.sum(normalized_predictions != numpy_observables[:, 0])) == 3
    assert _FakePyMatchingDecoder.seen_flat == numpy_events.flatten().tolist()


class _CapturingMemoryDecoder:
    calls: list[tuple[list[list[bool]], list[list[bool]], list[bool], list[bool]]]

    def __init__(self) -> None:
        self.calls = []

    def decode_memory_z(
        self,
        synx: list[Any],
        synz: list[Any],
        final: Any,
        *,
        init_synx: Any,
    ) -> tuple[bool, None]:
        self.calls.append(
            (
                [row.tolist() for row in synx],
                [row.tolist() for row in synz],
                final.tolist(),
                init_synx.tolist(),
            ),
        )
        return len(self.calls) == 1, None


def test_gate_trajectory_xor_matches_captured_numpy_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = _load_sweep_module()
    decoder = _CapturingMemoryDecoder()
    runtime = SimpleNamespace(
        patch=SimpleNamespace(geometry=SimpleNamespace(num_data=4)),
        num_x_stab=2,
        num_z_stab=1,
        logical_qubits=(0, 3),
        decoder=decoder,
    )
    result_dict = {
        "synx": [[1, 1, 0, 1], [0, 0, 1, 1]],
        "synz": [[0, 0], [1, 1]],
        "final": [[0, 0, 1, 1], [1, 1, 0, 0]],
        "init_synx": [[1, 1], [0, 0]],
    }
    monkeypatch.setattr(sweep, "_decoder_runtime", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(
        sweep,
        "_sim_reference_trajectory",
        lambda *_args, **_kwargs: (((1, 0), (0, 1)), ((1,), (0,)), (1, 0, 1, 0), (1, 0)),
    )
    monkeypatch.setattr(sweep, "_run_gate_backend_result_dict", lambda **_kwargs: result_dict)

    point = _private(sweep, "_run_memory_point")(
        sample_backend="sim",
        distance=3,
        basis="Z",
        physical_error_rate=0.006,
        total_rounds=2,
        num_shots=2,
        dem_mode="native_decomposed",
        native_circuit_source="abstract",
        seed=123,
    )

    assert decoder.calls == [
        ([[False, True], [False, False]], [[True], [False]], [True, False, False, True], [False, True]),
        ([[True, False], [True, False]], [[False], [True]], [False, True, True, False], [True, False]),
    ]
    assert asdict(point) == {
        "backend": "sim",
        "basis": "Z",
        "distance": 3,
        "logical_error_rate": 0.5,
        "num_logical_errors": 1,
        "num_raw_errors": 0,
        "num_shots": 2,
        "physical_error_rate": 0.006,
        "raw_error_rate": 0.0,
        "total_rounds": 2,
    }


def test_native_dem_point_matches_captured_pre_migration_values() -> None:
    sweep = _load_sweep_module()

    point = _private(sweep, "_run_memory_point")(
        sample_backend="native_sampler",
        distance=3,
        basis="Z",
        physical_error_rate=0.01,
        total_rounds=6,
        num_shots=20,
        dem_mode="native_decomposed",
        native_circuit_source="abstract",
        seed=124,
        p1_scale=1.0 / 30.0,
        p_meas_scale=1.0 / 3.0,
        p_prep_scale=1.0 / 3.0,
    )

    assert asdict(point) == {
        "backend": "native_sampler",
        "basis": "Z",
        "distance": 3,
        "logical_error_rate": 0.15,
        "num_logical_errors": 3,
        "num_raw_errors": None,
        "num_shots": 20,
        "physical_error_rate": 0.01,
        "raw_error_rate": None,
        "total_rounds": 6,
    }


def test_bootstrap_ci_is_deliberately_rebaselined_for_pecos_rng() -> None:
    sweep = _load_sweep_module()
    summary = _private(sweep, "_fit_summary_from_points")(_captured_points(sweep))

    # Deliberate re-baseline: PECOS's seeded RNG stream is expected to differ from NumPy's.
    assert (
        summary.fitted_logical_error_rate_per_round_ci_low,
        summary.fitted_logical_error_rate_per_round_ci_high,
        summary.fitted_projected_logical_error_rate_over_d_rounds_ci_low,
        summary.fitted_projected_logical_error_rate_over_d_rounds_ci_high,
    ) == (
        0.004567745863951806,
        0.005495604203696777,
        0.02242523794681137,
        0.02688059024790983,
    )


def test_bootstrap_ci_properties_and_reproducibility() -> None:
    sweep = _load_sweep_module()
    point = sweep.SweepPoint(
        backend="native_sampler",
        distance=3,
        basis="Z",
        physical_error_rate=0.006,
        total_rounds=5,
        num_shots=2000,
        num_logical_errors=200,
        num_raw_errors=None,
        logical_error_rate=0.1,
        raw_error_rate=None,
    )
    first = _private(sweep, "_fit_summary_confidence_intervals")([point])
    repeated = _private(sweep, "_fit_summary_confidence_intervals")([point])
    per_round = _private(sweep, "_fit_per_round_rate")([point])
    projected = sweep.ler_over_rounds(per_round, point.distance)

    assert first == repeated
    assert first[0] <= per_round <= first[1]
    assert first[2] <= projected <= first[3]

    observed_standard_error = math.sqrt(point.logical_error_rate * (1.0 - point.logical_error_rate) / point.num_shots)
    per_round_derivative = (1.0 / point.total_rounds) * (1.0 - 2.0 * point.logical_error_rate) ** (
        1.0 / point.total_rounds - 1.0
    )
    per_round_standard_error = per_round_derivative * observed_standard_error
    projected_derivative = point.distance * (1.0 - 2.0 * per_round) ** (point.distance - 1.0)
    projected_standard_error = projected_derivative * per_round_standard_error

    assert 2.0 * per_round_standard_error < first[1] - first[0] < 6.0 * per_round_standard_error
    assert 2.0 * projected_standard_error < first[3] - first[2] < 6.0 * projected_standard_error


def test_stable_bootstrap_seed_matches_captured_value() -> None:
    sweep = _load_sweep_module()

    assert _private(sweep, "_stable_bootstrap_seed")(_captured_points(sweep)) == 1869656115688596537
