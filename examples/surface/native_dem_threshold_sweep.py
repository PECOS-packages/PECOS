#!/usr/bin/env python3
r"""Surface-code X/Z memory threshold sweep with native PECOS DEMs.

This example runs rotated surface-code memory experiments using:

- Guppy surface-memory programs from ``pecos.guppy.surface.make_surface_code``
- ``sim(...).classical(selene_engine())`` for end-to-end execution
- direct ``selene_sim`` execution with either Selene ``Stim`` or the PECOS
  Selene stabilizer plugin
- optional native DEM sampling via ``build_native_sampler(...)``
- a depolarizing noise model with ``p2 = p``, ``p1 = p/30``, ``p_meas = p_prep = p/3``
- ``SurfaceDecoder(...)`` with PECOS-native DEMs (PyMatching or Tesseract)

For the ``sim`` backend, decoding is performed relative to a cached noiseless
reference trajectory from the same Guppy/QIS circuit. This makes the gate-level
path compatible with the native DEM's "deviations from ideal trajectory" view.

Instead of relying on one fixed memory duration, the default workflow samples
about four evenly spaced integer round counts across the window
``r in [2d, 3d]`` for each ``(distance, basis, p)`` point and fits a
per-round logical error rate ``epsilon`` via

    p_L(r) ~= 0.5 * (1 - (1 - 2 * epsilon) ** r)

This is a cleaner way to reduce temporal-boundary sensitivity than trying to
decode only the "middle" rounds of a finite spacetime volume.

Example:
    python examples/surface/native_dem_threshold_sweep.py --shots 200

    python examples/surface/native_dem_threshold_sweep.py \\
        --distances 3 5 7 9 \\
        --duration-multipliers 2 2.25 2.5 2.75 3 \\
        --error-rates 0.001 0.002 0.003 0.004 0.005 0.006 \\
        --bases X Z \\
        --shots 500 \\
        --save-json --save-svg

    python examples/surface/native_dem_threshold_sweep.py \\
        --sample-backend compare \\
        --distances 3 5 \\
        --error-rates 0.001 0.002

    python examples/surface/native_dem_threshold_sweep.py \\
        --sample-backend compare_all \\
        --distances 3 5 \\
        --error-rates 0.003
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import html
import itertools
import json
import math
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class SweepPoint:
    """Decoded statistics for one memory experiment duration."""

    backend: str
    distance: int
    basis: str
    physical_error_rate: float
    total_rounds: int
    num_shots: int
    num_logical_errors: int
    num_raw_errors: int | None
    logical_error_rate: float
    raw_error_rate: float | None


@dataclass(frozen=True)
class FitSummary:
    """Fitted per-round logical error summary for one ``(d, basis, p)`` point."""

    backend: str
    distance: int
    basis: str
    physical_error_rate: float
    num_shots_per_round_point: int
    round_values: tuple[int, ...]
    observed_logical_error_rates: tuple[float, ...]
    observed_raw_error_rates: tuple[float | None, ...]
    fitted_logical_error_rate_per_round: float
    fitted_projected_logical_error_rate_over_d_rounds: float
    fit_root_mean_square_error: float
    observed_logical_error_counts: tuple[int, ...] = ()
    observed_logical_error_rate_lower_bounds: tuple[float, ...] = ()
    observed_logical_error_rate_upper_bounds: tuple[float, ...] = ()
    fitted_logical_error_rate_per_round_ci_low: float | None = None
    fitted_logical_error_rate_per_round_ci_high: float | None = None
    fitted_projected_logical_error_rate_over_d_rounds_ci_low: float | None = None
    fitted_projected_logical_error_rate_over_d_rounds_ci_high: float | None = None


@dataclass(frozen=True)
class DistanceScalingFitSummary:
    """Fit ``epsilon(d) ~= A * (p / p_th) ** ((d + 1) / 2)`` at fixed ``p``."""

    backend: str
    basis: str
    physical_error_rate: float
    distances: tuple[int, ...]
    fitted_prefactor: float
    fitted_threshold: float
    fitted_suppression_factor: float
    fit_root_mean_square_log_error: float


@dataclass(frozen=True)
class GlobalScalingFitSummary:
    """Fit the standard below-threshold surface-code scaling ansatz."""

    backend: str
    basis: str
    distances: tuple[int, ...]
    physical_error_rates: tuple[float, ...]
    fitted_prefactor: float
    fitted_threshold: float
    fit_root_mean_square_log_error: float


@dataclass(frozen=True)
class PerDistancePowerLawFitSummary:
    """Fit ``epsilon(p) ~= C_d * p ** beta_d`` for one distance."""

    backend: str
    basis: str
    distance: int
    physical_error_rates: tuple[float, ...]
    fitted_prefactor: float
    fitted_exponent: float
    expected_distance_scaling_exponent: float
    fit_root_mean_square_log_error: float
    fitted_exponent_std_error: float = 0.0


@dataclass(frozen=True)
class FSSThresholdFitSummary:
    """Polynomial finite-size-scaling threshold fit using the Wang-Harrington-Preskill form.

    Reference: Wang, Harrington, Preskill, *Confinement-Higgs transition in a disordered
    gauge theory and the accuracy threshold for quantum memory* (arXiv:quant-ph/0207088).
    Fit model: ``p_L = a + b*x + c*x**2`` with ``x = (p - p_th) * d**(1 / nu)``. Data
    should bracket the threshold; the Watson-Barrett constraint ``p > 1/(4d)``
    (arXiv:1312.5213) also applies to the domain of validity.

    Delegates to ``pecos.analysis.threshold_curve.threshold_fit`` + ``func`` so
    every surface-code performance report uses the same canonical fit routine
    as the rest of PECOS.
    """

    backend: str
    basis: str
    p_th: float
    p_th_std_error: float
    nu: float
    nu_std_error: float
    coeff_a: float
    coeff_a_std_error: float
    coeff_b: float
    coeff_b_std_error: float
    coeff_c: float
    coeff_c_std_error: float
    num_points: int
    fit_window_low: float
    fit_window_high: float
    reference: str = "Wang-Harrington-Preskill (arXiv:quant-ph/0207088)"


@dataclass(frozen=True)
class PairwiseLambdaSummary:
    """Empirical ``Lambda_{d/(d+2)}`` ratios at one fixed physical error rate."""

    backend: str
    basis: str
    physical_error_rate: float
    distance_low: int
    distance_high: int
    lambda_d_over_d_plus_2: float


@dataclass(frozen=True)
class DashboardPlot:
    """One SVG plot entry for the optional HTML dashboard."""

    section: str
    title: str
    filename: str
    backend: str
    basis: str | None = None
    physical_error_rate: float | None = None


@dataclass(frozen=True)
class _DecoderRuntime:
    """Reusable decoder-side runtime for one native comparison point shape."""

    patch: Any
    logical_qubits: tuple[int, ...]
    num_x_stab: int
    num_z_stab: int
    noise: Any
    decoder: Any


@dataclass(frozen=True)
class _NativeSamplerRuntime:
    """Reusable sampler + decoder bundle for one traced/native DEM shape."""

    decoder_runtime: _DecoderRuntime
    sampler: Any
    dem_decoder: Any
    dem_str: str | None = None


_CACHED_SELENE_INSTANCES: list[Any] = []
_FIT_CONFIDENCE_LEVEL = 0.95
_FIT_BOOTSTRAP_SAMPLES = 200


def _cleanup_cached_selene_instances() -> None:
    """Best-effort cleanup for temporary Selene build directories."""
    while _CACHED_SELENE_INSTANCES:
        instance = _CACHED_SELENE_INSTANCES.pop()
        with contextlib.suppress(Exception):
            instance.delete_files()


atexit.register(_cleanup_cached_selene_instances)


def _backend_runtime_label(sample_backend: str, native_circuit_source: str = "abstract") -> str:
    """Describe one sampling backend in human-readable terms."""
    # Handle "backend:decoder" labels from multi-decoder comparison
    if ":" in sample_backend:
        base, decoder = sample_backend.split(":", 1)
        base_label = _backend_runtime_label(base, native_circuit_source)
        return f"{base_label} [decoder={decoder}]"
    if sample_backend == "sim":
        return (
            "sim(Guppy(...)).classical(selene_engine()).quantum(pecos.stabilizer()) "
            f"+ PECOS depolarizing noise + native DEM source={native_circuit_source} + noiseless "
            "reference-trajectory calibration"
        )
    if sample_backend == "selene_sim":
        return (
            "direct selene_sim (compile_guppy_to_hugr + build/run_shots) with Selene Stim "
            f"+ Selene DepolarizingErrorModel + native DEM source={native_circuit_source} "
            "+ noiseless reference-trajectory calibration"
        )
    if sample_backend == "selene_stabilizer_plugin":
        return (
            "direct selene_sim (compile_guppy_to_hugr + build/run_shots) with the PECOS "
            "Selene StabilizerPlugin + Selene DepolarizingErrorModel + native DEM source="
            f"{native_circuit_source} + noiseless reference-trajectory calibration"
        )
    if sample_backend == "native_sampler":
        return f"build_native_sampler(..., circuit_source={native_circuit_source!r}) + DEM decoder on the native DEM"
    msg = f"Unknown sample backend: {sample_backend}"
    raise ValueError(msg)


def _predicted_observable_flip(result: object) -> int:
    """Extract the predicted logical observable flip from a DEM decoder result."""
    observables_mask = getattr(result, "observables_mask", None)
    if observables_mask is not None:
        return int(observables_mask & 1)
    correction = getattr(result, "correction", [])
    return int(correction[0]) if len(correction) > 0 else 0


def _format_rate(value: float | None) -> str:
    """Format a logical or raw error rate for compact terminal output."""
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def ler_per_round_exp(logical_error_rate: float, num_rounds: int) -> float:
    """Extract a per-round logical error rate from one duration point."""
    if num_rounds <= 0:
        msg = "num_rounds must be positive"
        raise ValueError(msg)
    if logical_error_rate <= 0.0:
        return 0.0
    if logical_error_rate >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * logical_error_rate) ** (1.0 / num_rounds))


def ler_over_rounds(per_round_rate: float, num_rounds: float) -> float:
    """Project a per-round logical error rate over ``num_rounds`` rounds."""
    if num_rounds <= 0:
        msg = "num_rounds must be positive"
        raise ValueError(msg)
    if per_round_rate <= 0.0:
        return 0.0
    if per_round_rate >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * per_round_rate) ** num_rounds)


def _wilson_interval(
    num_successes: int,
    num_trials: int,
    *,
    confidence_level: float = _FIT_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Return a Wilson score interval for one binomial proportion."""
    if num_trials <= 0:
        msg = "num_trials must be positive"
        raise ValueError(msg)
    z = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    p_hat = num_successes / num_trials
    z_sq_over_n = (z * z) / num_trials
    denom = 1.0 + z_sq_over_n
    center = (p_hat + 0.5 * z_sq_over_n) / denom
    half_width = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * num_trials)) / num_trials) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _fit_summary_metric_interval(summary: FitSummary, metric: str) -> tuple[float, float, float]:
    """Return ``(value, low, high)`` for one plotted fit metric."""
    value = getattr(summary, metric)
    if metric == "fitted_logical_error_rate_per_round":
        low = summary.fitted_logical_error_rate_per_round_ci_low
        high = summary.fitted_logical_error_rate_per_round_ci_high
    elif metric == "fitted_projected_logical_error_rate_over_d_rounds":
        low = summary.fitted_projected_logical_error_rate_over_d_rounds_ci_low
        high = summary.fitted_projected_logical_error_rate_over_d_rounds_ci_high
    else:
        low = None
        high = None
    return (
        value,
        value if low is None else low,
        value if high is None else high,
    )


def _format_interval(low: float | None, high: float | None, value: float) -> str:
    """Format one fit interval for terminal output."""
    resolved_low = value if low is None else low
    resolved_high = value if high is None else high
    return f"[{resolved_low:.6e}, {resolved_high:.6e}]"


def _stable_bootstrap_seed(points: list[SweepPoint]) -> int:
    """Derive a stable RNG seed for one fit-summary point group."""
    first = points[0]
    key = "|".join(
        [
            first.backend,
            first.basis,
            str(first.distance),
            f"{first.physical_error_rate:.12g}",
            *(f"{point.total_rounds}:{point.num_shots}:{point.num_logical_errors}" for point in points),
        ],
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _percentile_interval(
    values: list[float],
    *,
    confidence_level: float = _FIT_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Return an empirical central percentile interval for a sample."""
    if not values:
        msg = "Need at least one sample value"
        raise ValueError(msg)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0], ordered[0]

    lower_q = 0.5 * (1.0 - confidence_level)
    upper_q = 1.0 - lower_q

    def interpolate(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return ordered[lower_index]
        fraction = position - lower_index
        return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction

    return interpolate(lower_q), interpolate(upper_q)


def _fit_summary_confidence_intervals(points: list[SweepPoint]) -> tuple[float, float, float, float]:
    """Bootstrap fit uncertainty for one ``(d, basis, p)`` point group."""
    ordered = sorted(points, key=lambda point: point.total_rounds)
    fitted_per_round = _fit_per_round_rate(ordered)
    fitted_projected = ler_over_rounds(fitted_per_round, ordered[0].distance)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return fitted_per_round, fitted_per_round, fitted_projected, fitted_projected

    shot_counts = np.asarray([point.num_shots for point in ordered], dtype=np.int64)
    observed_rates = np.asarray(
        [min(max(point.logical_error_rate, 0.0), 1.0) for point in ordered],
        dtype=np.float64,
    )
    rng = np.random.default_rng(_stable_bootstrap_seed(ordered))
    bootstrap_counts = rng.binomial(n=shot_counts, p=observed_rates, size=(_FIT_BOOTSTRAP_SAMPLES, len(ordered)))

    bootstrap_per_round: list[float] = []
    bootstrap_projected: list[float] = []
    for sample_counts in bootstrap_counts:
        sample_points: list[SweepPoint] = []
        for point, sample_count in zip(ordered, sample_counts, strict=True):
            count = int(sample_count)
            sample_points.append(
                SweepPoint(
                    backend=point.backend,
                    distance=point.distance,
                    basis=point.basis,
                    physical_error_rate=point.physical_error_rate,
                    total_rounds=point.total_rounds,
                    num_shots=point.num_shots,
                    num_logical_errors=count,
                    num_raw_errors=point.num_raw_errors,
                    logical_error_rate=(count / point.num_shots) if point.num_shots else 0.0,
                    raw_error_rate=point.raw_error_rate,
                ),
            )
        sample_fit = _fit_per_round_rate(sample_points)
        bootstrap_per_round.append(sample_fit)
        bootstrap_projected.append(ler_over_rounds(sample_fit, ordered[0].distance))

    per_round_low, per_round_high = _percentile_interval(bootstrap_per_round)
    projected_low, projected_high = _percentile_interval(bootstrap_projected)
    return per_round_low, per_round_high, projected_low, projected_high


def _rounds_from_multiplier(distance: int, duration_multiplier: float) -> int:
    """Convert a duration multiplier into an integer round count."""
    total_rounds = round(duration_multiplier * distance)
    if total_rounds <= 0:
        msg = f"duration multiplier {duration_multiplier!r} produced non-positive rounds for d={distance}"
        raise ValueError(msg)
    return total_rounds


def _evenly_spaced_values(start: float, stop: float, num_points: int) -> list[float]:
    """Return ``num_points`` evenly spaced values from ``start`` to ``stop`` inclusive."""
    if num_points <= 0:
        msg = "num_points must be positive"
        raise ValueError(msg)
    if num_points == 1:
        return [0.5 * (start + stop)]
    step = (stop - start) / (num_points - 1)
    return [start + index * step for index in range(num_points)]


def _duration_rounds_for_distance(
    distance: int,
    *,
    explicit_multipliers: list[float] | None,
    duration_min_multiplier: float,
    duration_max_multiplier: float,
    duration_num_points: int,
) -> tuple[int, ...]:
    """Return the effective integer round counts to sample for one distance."""
    if explicit_multipliers is not None:
        return tuple(sorted({_rounds_from_multiplier(distance, multiplier) for multiplier in explicit_multipliers}))

    start_round = _rounds_from_multiplier(distance, duration_min_multiplier)
    stop_round = _rounds_from_multiplier(distance, duration_max_multiplier)
    if stop_round < start_round:
        msg = "duration_max_multiplier must be at least duration_min_multiplier"
        raise ValueError(msg)
    raw_rounds = _evenly_spaced_values(float(start_round), float(stop_round), duration_num_points)
    return tuple(sorted({max(1, round(value)) for value in raw_rounds}))


def _reshape_round_values(flat_values: list[int], num_rounds: int, width: int, label: str) -> list[Any]:
    """Reshape a flattened per-shot result register into round slices."""
    import numpy as np

    if width <= 0:
        return []
    expected = num_rounds * width
    values = np.asarray(flat_values, dtype=np.uint8)
    if values.size != expected:
        msg = (
            f"Register {label!r} has {values.size} bits for one shot, "
            f"expected {expected} = {num_rounds} rounds * {width} bits"
        )
        raise ValueError(msg)
    return [values[i * width : (i + 1) * width] for i in range(num_rounds)]


def _logical_qubits_for_basis(patch: object, basis: str) -> tuple[int, ...]:
    """Get the logical support used for the final parity check."""
    geom = patch.geometry
    if basis.upper() == "Z":
        return tuple(geom.logical_z.data_qubits if geom.logical_z else ())
    return tuple(geom.logical_x.data_qubits if geom.logical_x else ())


def _result_rows_for_key(result_dict: dict[str, Any], key: str) -> list[Any]:
    """Fetch per-shot rows for a named result register."""
    if key in result_dict:
        rows = result_dict[key]
        if isinstance(rows, list):
            return rows
    available = ", ".join(sorted(result_dict))
    msg = f"Expected result register {key!r}, available registers: {available}"
    raise KeyError(msg)


@cache
def _surface_patch(distance: int) -> object:
    """Cache surface patch geometry shared across many sweep points."""
    from pecos.qec.surface import SurfacePatch

    return SurfacePatch.create(distance=distance)


_CHECK_MATRIX_DECODERS = {"bp_osd", "bp_lsd", "union_find", "relay_bp", "min_sum_bp"}


def _noise_model_description(args: argparse.Namespace) -> str:
    """Human-readable noise model string for reports."""
    p1s = getattr(args, "p1_scale", 1.0 / 30.0)
    pms = getattr(args, "p_meas_scale", 1.0 / 3.0)
    pps = getattr(args, "p_prep_scale", 1.0 / 3.0)
    return f"depolarizing with p1={p1s:.4g}*p, p2=p, p_meas={pms:.4g}*p, p_prep={pps:.4g}*p"


def _create_dem_decoder(decoder_type: str, dem_str: str, *, tesseract_beam: int = 5) -> object:
    """Create a DEM-level decoder from a DEM string.

    Supports MWPM decoders (pymatching), search decoders (tesseract), and
    check-matrix decoders (bp_osd, bp_lsd, union_find, relay_bp, min_sum_bp)
    via DemAwareDecoder which extracts the check matrix from the DEM.
    """
    if decoder_type == "tesseract":
        from pecos_rslib.decoders import TesseractDecoder

        dem_filtered = "\n".join(line for line in dem_str.split("\n") if not line.startswith("logical_observable"))
        return TesseractDecoder.from_dem(dem_filtered, preset="fast", det_beam=tesseract_beam)

    if decoder_type in _CHECK_MATRIX_DECODERS:
        from pecos_rslib.decoders import DemAwareDecoder

        dem_filtered = "\n".join(line for line in dem_str.split("\n") if not line.startswith("logical_observable"))
        return DemAwareDecoder.from_dem(dem_filtered, decoder_type=decoder_type)

    from pecos_rslib.decoders import PyMatchingDecoder

    return PyMatchingDecoder.from_dem(dem_str)


def _decode_one_shot(dem_decoder: object, events_flat: list[int]) -> object:
    """Decode one shot using whichever DEM decoder was created.

    Tesseract.decode() wants sparse indices; decode_syndrome() accepts dense vectors.
    PyMatching.decode() accepts dense vectors directly.
    """
    if hasattr(dem_decoder, "decode_syndrome"):
        return dem_decoder.decode_syndrome(events_flat)
    return dem_decoder.decode(events_flat)


def _decode_all_shots(
    dem_decoder: object,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
    num_shots: int,
) -> int:
    """Decode all shots using the fastest available path.

    For PyMatching: uses decode_batch with flattened numpy array (no Python loop).
    For Tesseract: uses decode_batch with parallel rayon workers.
    For others: falls back to per-shot Python loop.

    Returns the number of logical errors.
    """
    import numpy as np

    true_flips = (
        observable_flips[:, 0].astype(np.uint8)
        if observable_flips.shape[1] > 0
        else np.zeros(num_shots, dtype=np.uint8)
    )

    # PyMatching batch: takes flattened (num_shots * num_detectors) u8 array
    from pecos_rslib.decoders import PyMatchingDecoder

    if isinstance(dem_decoder, PyMatchingDecoder):
        flat = detection_events.astype(np.uint8).flatten().tolist()
        predictions = dem_decoder.decode_batch(flat, num_shots)
        # Each prediction is a list of observables; check index 0
        predicted = np.array([p[0] if p else 0 for p in predictions], dtype=np.uint8)
        return int(np.sum(predicted != true_flips))

    # Tesseract batch: takes list of syndromes, parallel rayon
    from pecos_rslib.decoders import TesseractDecoder

    if isinstance(dem_decoder, TesseractDecoder):
        syndromes = [detection_events[i].astype(np.uint8).tolist() for i in range(num_shots)]
        batch_results = dem_decoder.decode_batch(syndromes)
        num_errors = 0
        for shot_idx, result in enumerate(batch_results):
            predicted_flip = int(result.observables_mask & 1)
            num_errors += int(predicted_flip != true_flips[shot_idx])
        return num_errors

    # Fallback: per-shot loop (DemAwareDecoder, etc.)
    num_errors = 0
    for shot_idx in range(num_shots):
        events_flat = detection_events[shot_idx].astype(np.uint8).tolist()
        decode_result = _decode_one_shot(dem_decoder, events_flat)
        predicted_flip = _predicted_observable_flip(decode_result)
        num_errors += int(predicted_flip != true_flips[shot_idx])
    return num_errors


@cache
def _decoder_runtime(
    distance: int,
    total_rounds: int,
    basis: str,
    physical_error_rate: float,
    dem_mode: str,
    native_circuit_source: str,
    decoder_type: str = "pymatching",
    ancilla_budget: int | None = None,
    p1_scale: float = 0.1,
    p_meas_scale: float = 0.5,
    p_prep_scale: float = 0.5,
) -> _DecoderRuntime:
    """Build and cache the expensive native decoder-side objects once."""
    from pecos.qec.surface import NoiseModel, SurfaceDecoder

    basis = basis.upper()
    patch = _surface_patch(distance)
    noise = NoiseModel(
        p1=physical_error_rate * p1_scale,
        p2=physical_error_rate,
        p_meas=physical_error_rate * p_meas_scale,
        p_prep=physical_error_rate * p_prep_scale,
    )
    decoder = SurfaceDecoder(
        patch,
        num_rounds=total_rounds,
        noise=noise,
        decoder_type=decoder_type,
        use_circuit_level_dem=True,
        circuit_level_dem_mode=dem_mode,
        circuit_level_dem_source=native_circuit_source,
        ancilla_budget=ancilla_budget,
    )
    return _DecoderRuntime(
        patch=patch,
        logical_qubits=_logical_qubits_for_basis(patch, basis),
        num_x_stab=len(patch.geometry.x_stabilizers),
        num_z_stab=len(patch.geometry.z_stabilizers),
        noise=noise,
        decoder=decoder,
    )


@cache
def _native_sampler_runtime(
    distance: int,
    total_rounds: int,
    basis: str,
    physical_error_rate: float,
    dem_mode: str,
    native_circuit_source: str,
    decoder_type: str = "pymatching",
    ancilla_budget: int | None = None,
    p1_scale: float = 0.1,
    p_meas_scale: float = 0.5,
    p_prep_scale: float = 0.5,
) -> _NativeSamplerRuntime:
    """Build and cache the native sampler + decoder bundle once."""
    from pecos.qec.surface import build_native_sampler
    from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

    runtime = _decoder_runtime(
        distance,
        total_rounds,
        basis,
        physical_error_rate,
        dem_mode,
        native_circuit_source,
        decoder_type=decoder_type,
        ancilla_budget=ancilla_budget,
        p1_scale=p1_scale,
        p_meas_scale=p_meas_scale,
        p_prep_scale=p_prep_scale,
    )
    sampler = build_native_sampler(
        runtime.patch,
        total_rounds,
        runtime.noise,
        basis=basis,
        circuit_source=native_circuit_source,
        ancilla_budget=ancilla_budget,
    )
    # PyMatching needs decomposed (graph-like) DEMs; Tesseract and check-matrix
    # decoders handle hyperedges natively and should get the full DEM.
    if decoder_type == "pymatching":
        dem_str = runtime.decoder.get_dem(basis.upper(), circuit_level=True)
    else:
        dem_str = generate_circuit_level_dem_from_builder(
            runtime.patch,
            total_rounds,
            runtime.noise,
            basis=basis,
            decompose_errors=False,
            circuit_source=native_circuit_source,
            ancilla_budget=ancilla_budget,
        )
    dem_decoder = _create_dem_decoder(decoder_type, dem_str)
    # The traced-QIS sampler stack has a noticeable one-time initialization cost
    # on its first sample. Pay that once when the cached runtime is created so
    # subsequent point evaluations stay on the true steady-state path.
    warm_det_events, _ = sampler.sample(num_shots=1, seed=0)
    _decode_one_shot(dem_decoder, warm_det_events[0].astype(int).tolist())
    # Filter logical_observable lines for decoders that need it
    dem_str_filtered = "\n".join(line for line in dem_str.split("\n") if not line.startswith("logical_observable"))
    return _NativeSamplerRuntime(
        decoder_runtime=runtime,
        sampler=sampler,
        dem_decoder=dem_decoder,
        dem_str=dem_str_filtered,
    )


@cache
def _sim_reference_trajectory(
    sample_backend: str,
    distance: int,
    total_rounds: int,
    basis: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Cache a noiseless gate-level trajectory used as a decoding reference."""
    import numpy as np
    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=distance)
    result_dict = _run_gate_backend_result_dict(
        sample_backend=sample_backend,
        distance=distance,
        basis=basis,
        physical_error_rate=0.0,
        total_rounds=total_rounds,
        num_shots=1,
        seed=0,
    )

    synx_rows = _reshape_round_values(
        _result_rows_for_key(result_dict, "synx")[0],
        total_rounds,
        len(patch.geometry.x_stabilizers),
        "synx",
    )
    synz_rows = _reshape_round_values(
        _result_rows_for_key(result_dict, "synz")[0],
        total_rounds,
        len(patch.geometry.z_stabilizers),
        "synz",
    )
    final = np.asarray(_result_rows_for_key(result_dict, "final")[0], dtype=np.uint8)

    return (
        tuple(tuple(int(v) for v in row) for row in synx_rows),
        tuple(tuple(int(v) for v in row) for row in synz_rows),
        tuple(int(v) for v in final.tolist()),
    )


@cache
def _compiled_guppy_hugr(distance: int, total_rounds: int, basis: str) -> bytes:
    """Cache compiled HUGR bytes for the direct selene_sim backend."""
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from pecos.guppy import make_surface_code

    program = make_surface_code(distance=distance, num_rounds=total_rounds, basis=basis)
    return compile_guppy_to_hugr(program)


@cache
def _selene_instance(distance: int, total_rounds: int, basis: str) -> object:
    """Cache a built Selene instance for one circuit shape."""
    from selene_sim import build

    instance = build(
        _compiled_guppy_hugr(distance, total_rounds, basis),
        name=f"surface_d{distance}_{basis.lower()}_r{total_rounds}",
    )
    _CACHED_SELENE_INSTANCES.append(instance)
    return instance


def _run_gate_backend_result_dict(
    *,
    sample_backend: str,
    distance: int,
    basis: str,
    physical_error_rate: float,
    total_rounds: int,
    num_shots: int,
    seed: int,
    timing_sink: dict[str, float] | None = None,
    p1_scale: float = 0.1,
    p_meas_scale: float = 0.5,
    p_prep_scale: float = 0.5,
) -> dict[str, list[list[int]]]:
    """Run one gate-level backend and normalize results to a shot-map-like dict."""
    import os
    import tempfile
    from collections import defaultdict

    import pecos
    from pecos.guppy import get_num_qubits, make_surface_code

    def run_direct_selene_backend(*, simulator: object) -> dict[str, list[list[int]]]:
        from selene_sim import DepolarizingErrorModel, SimpleRuntime

        backend_start = time.perf_counter()
        os.environ.setdefault(
            "ZIG_GLOBAL_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "pecos_zig_global_cache"),
        )
        os.environ.setdefault(
            "ZIG_LOCAL_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "pecos_zig_local_cache"),
        )

        compile_start = time.perf_counter()
        _compiled_guppy_hugr(distance, total_rounds, basis)
        compile_seconds = time.perf_counter() - compile_start

        build_start = time.perf_counter()
        instance = _selene_instance(distance, total_rounds, basis)
        build_seconds = time.perf_counter() - build_start

        reset_start = time.perf_counter()
        instance.delete_run_directories()
        instance.runs.mkdir(parents=True, exist_ok=True)
        reset_seconds = time.perf_counter() - reset_start

        error_model_start = time.perf_counter()
        error_model = DepolarizingErrorModel(
            p_1q=physical_error_rate * p1_scale,
            p_2q=physical_error_rate,
            p_meas=physical_error_rate * p_meas_scale,
            p_prep=physical_error_rate * p_prep_scale,
        )
        error_model_seconds = time.perf_counter() - error_model_start

        result_dict: dict[str, list[list[int]]] = defaultdict(list)
        run_start = time.perf_counter()
        for shot_results in instance.run_shots(
            simulator=simulator,
            n_qubits=get_num_qubits(distance),
            n_shots=num_shots,
            error_model=error_model,
            runtime=SimpleRuntime(),
            random_seed=seed,
            n_processes=1,
        ):
            shot_rows: dict[str, list[int]] = defaultdict(list)
            for name, values in shot_results:
                shot_rows[name].extend(int(v) for v in values)
            for name, values in shot_rows.items():
                result_dict[name].append(values)
        run_seconds = time.perf_counter() - run_start
        if timing_sink is not None:
            timing_sink.update(
                {
                    "compile_hugr_seconds": compile_seconds,
                    "instance_build_seconds": build_seconds,
                    "instance_reset_seconds": reset_seconds,
                    "error_model_seconds": error_model_seconds,
                    "run_and_parse_seconds": run_seconds,
                    "total_seconds": time.perf_counter() - backend_start,
                },
            )
        return dict(result_dict)

    if sample_backend == "sim":
        backend_start = time.perf_counter()
        noise_start = time.perf_counter()
        noise_model = pecos.depolarizing_noise()
        noise_model.set_probabilities(
            physical_error_rate * p_prep_scale,  # p_prep
            physical_error_rate * p_meas_scale,  # p_meas_0
            physical_error_rate * p_meas_scale,  # p_meas_1
            physical_error_rate * p1_scale,  # p1 (single-qubit gates)
            physical_error_rate,  # p2 (two-qubit gates)
        )
        noise_seconds = time.perf_counter() - noise_start
        program_start = time.perf_counter()
        program = make_surface_code(distance=distance, num_rounds=total_rounds, basis=basis)
        program_seconds = time.perf_counter() - program_start
        run_start = time.perf_counter()
        shot_vec = (
            pecos.sim(program)
            .classical(pecos.selene_engine())
            .quantum(pecos.stabilizer())
            .qubits(get_num_qubits(distance))
            .noise(noise_model)
            .seed(seed)
            .run(num_shots)
        )
        run_seconds = time.perf_counter() - run_start
        shot_map_start = time.perf_counter()
        shot_map = shot_vec.to_shot_map()
        shot_map_seconds = time.perf_counter() - shot_map_start
        dict_start = time.perf_counter()
        result_dict = shot_map.to_dict()
        dict_seconds = time.perf_counter() - dict_start
        if timing_sink is not None:
            timing_sink.update(
                {
                    "noise_model_seconds": noise_seconds,
                    "program_build_seconds": program_seconds,
                    "run_seconds": run_seconds,
                    "to_shot_map_seconds": shot_map_seconds,
                    "to_dict_seconds": dict_seconds,
                    "total_seconds": time.perf_counter() - backend_start,
                },
            )
        return result_dict

    if sample_backend == "selene_sim":
        from selene_sim import Stim

        return run_direct_selene_backend(simulator=Stim())

    if sample_backend == "selene_stabilizer_plugin":
        from pecos_selene_stabilizer import StabilizerPlugin

        return run_direct_selene_backend(simulator=StabilizerPlugin())

    msg = f"Unknown gate backend: {sample_backend}"
    raise ValueError(msg)


def _profile_gate_backends(
    *,
    backends: list[str],
    distances: list[int],
    bases: list[str],
    error_rates: list[float],
    duration_rounds_by_distance: dict[int, tuple[int, ...]],
    shots: int,
    seed: int,
    warmup_repetitions: int,
    benchmark_repetitions: int,
) -> None:
    """Profile gate backends and print a phase breakdown."""
    if warmup_repetitions < 0:
        msg = "warmup_repetitions must be non-negative"
        raise ValueError(msg)
    if benchmark_repetitions <= 0:
        msg = "benchmark_repetitions must be positive"
        raise ValueError(msg)

    print()
    print("Gate Backend Profile")
    print(f"  warmup repetitions : {warmup_repetitions}")
    print(f"  timed repetitions  : {benchmark_repetitions}")

    profile_keys: dict[str, list[str]] = {
        "selene_sim": [
            "compile_hugr_seconds",
            "instance_build_seconds",
            "instance_reset_seconds",
            "error_model_seconds",
            "run_and_parse_seconds",
        ],
        "selene_stabilizer_plugin": [
            "compile_hugr_seconds",
            "instance_build_seconds",
            "instance_reset_seconds",
            "error_model_seconds",
            "run_and_parse_seconds",
        ],
        "sim": [
            "noise_model_seconds",
            "program_build_seconds",
            "run_seconds",
            "to_shot_map_seconds",
            "to_dict_seconds",
        ],
    }

    combinations = [
        (distance, basis, physical_error_rate, total_rounds)
        for basis in bases
        for distance in distances
        for physical_error_rate in error_rates
        for total_rounds in duration_rounds_by_distance[distance]
    ]

    for combo_idx, (distance, basis, physical_error_rate, total_rounds) in enumerate(combinations, start=1):
        print()
        print(
            f"[profile {combo_idx}/{len(combinations)}] "
            f"basis={basis} d={distance} p={physical_error_rate:.5g} r={total_rounds} shots={shots}",
        )
        backend_totals: dict[str, float] = {}
        for backend_index, backend in enumerate(backends, start=1):
            combo_seed = seed + combo_idx * 1000 + backend_index * 100
            for rep in range(warmup_repetitions):
                _run_gate_backend_result_dict(
                    sample_backend=backend,
                    distance=distance,
                    basis=basis,
                    physical_error_rate=physical_error_rate,
                    total_rounds=total_rounds,
                    num_shots=shots,
                    seed=combo_seed + rep,
                )

            runs: list[dict[str, float]] = []
            for rep in range(benchmark_repetitions):
                timing: dict[str, float] = {}
                _run_gate_backend_result_dict(
                    sample_backend=backend,
                    distance=distance,
                    basis=basis,
                    physical_error_rate=physical_error_rate,
                    total_rounds=total_rounds,
                    num_shots=shots,
                    seed=combo_seed + warmup_repetitions + rep,
                    timing_sink=timing,
                )
                runs.append(timing)

            total_values = [run["total_seconds"] for run in runs]
            mean_total = statistics.fmean(total_values)
            median_total = statistics.median(total_values)
            shots_per_second = shots / mean_total if mean_total > 0 else float("inf")
            backend_totals[backend] = mean_total
            print(
                f"  [{backend}] mean={mean_total:.3f}s "
                f"median={median_total:.3f}s throughput={shots_per_second:.3f} shots/s",
            )
            for key in profile_keys[backend]:
                phase_values = [run[key] for run in runs]
                mean_phase = statistics.fmean(phase_values)
                phase_fraction = mean_phase / mean_total if mean_total > 0 else 0.0
                print(f"    {key}: {mean_phase:.3f}s ({phase_fraction:.1%})")

        if "selene_sim" in backend_totals:
            reference = backend_totals["selene_sim"]
            print("  relative_to_selene_sim:")
            for backend in backends:
                ratio = backend_totals[backend] / reference if reference > 0 else float("inf")
                print(f"    {backend}: {ratio:.3f}")


def _run_memory_point(
    *,
    sample_backend: str,
    distance: int,
    basis: str,
    physical_error_rate: float,
    total_rounds: int,
    num_shots: int,
    dem_mode: str,
    native_circuit_source: str,
    seed: int,
    decoder_type: str = "pymatching",
    backend_label: str | None = None,
    ancilla_budget: int | None = None,
    p1_scale: float = 0.1,
    p_meas_scale: float = 0.5,
    p_prep_scale: float = 0.5,
) -> SweepPoint:
    """Run one surface-memory point and decode it with native PECOS DEMs."""
    import numpy as np

    basis = basis.upper()
    decoder_runtime = _decoder_runtime(
        distance,
        total_rounds,
        basis,
        physical_error_rate,
        dem_mode,
        native_circuit_source,
        decoder_type=decoder_type,
        ancilla_budget=ancilla_budget,
        p1_scale=p1_scale,
        p_meas_scale=p_meas_scale,
        p_prep_scale=p_prep_scale,
    )
    patch = decoder_runtime.patch
    num_x_stab = decoder_runtime.num_x_stab
    num_z_stab = decoder_runtime.num_z_stab
    logical_qubits = decoder_runtime.logical_qubits
    decoder = decoder_runtime.decoder

    num_logical_errors = 0
    num_raw_errors: int | None = 0

    if sample_backend in {"sim", "selene_sim", "selene_stabilizer_plugin"}:
        ref_synx_rows, ref_synz_rows, ref_final_row = _sim_reference_trajectory(
            sample_backend,
            distance,
            total_rounds,
            basis.upper(),
        )
        ref_synx_list = [np.asarray(row, dtype=np.uint8) for row in ref_synx_rows]
        ref_synz_list = [np.asarray(row, dtype=np.uint8) for row in ref_synz_rows]
        ref_final = np.asarray(ref_final_row, dtype=np.uint8)
        result_dict = _run_gate_backend_result_dict(
            sample_backend=sample_backend,
            distance=distance,
            basis=basis,
            physical_error_rate=physical_error_rate,
            total_rounds=total_rounds,
            num_shots=num_shots,
            seed=seed,
            p1_scale=p1_scale,
            p_meas_scale=p_meas_scale,
            p_prep_scale=p_prep_scale,
        )

        synx_rows = _result_rows_for_key(result_dict, "synx")
        synz_rows = _result_rows_for_key(result_dict, "synz")
        final_rows = _result_rows_for_key(result_dict, "final")

        if len(synx_rows) != num_shots or len(synz_rows) != num_shots or len(final_rows) != num_shots:
            msg = (
                "Result register lengths do not match the requested shot count: "
                f"synx={len(synx_rows)}, synz={len(synz_rows)}, final={len(final_rows)}, shots={num_shots}"
            )
            raise ValueError(
                msg,
            )

        for shot_idx in range(num_shots):
            synx_list = _reshape_round_values(synx_rows[shot_idx], total_rounds, num_x_stab, "synx")
            synz_list = _reshape_round_values(synz_rows[shot_idx], total_rounds, num_z_stab, "synz")
            final = np.asarray(final_rows[shot_idx], dtype=np.uint8)

            if final.size != patch.geometry.num_data:
                msg = f"Register 'final' has {final.size} bits for one shot, expected {patch.geometry.num_data}"
                raise ValueError(
                    msg,
                )

            # Decode relative to the noiseless gate-level baseline so the native
            # DEM sees deviations from the actual circuit trajectory.
            synx_list = [
                np.asarray(synx, dtype=np.uint8) ^ ref_synx
                for synx, ref_synx in zip(synx_list, ref_synx_list, strict=True)
            ]
            synz_list = [
                np.asarray(synz, dtype=np.uint8) ^ ref_synz
                for synz, ref_synz in zip(synz_list, ref_synz_list, strict=True)
            ]
            final = final ^ ref_final

            raw_parity = int(sum(int(final[q]) for q in logical_qubits) % 2)
            if num_raw_errors is None:
                msg = "Gate-level backends must track raw parity counts"
                raise RuntimeError(msg)
            num_raw_errors += raw_parity

            if basis.upper() == "Z":
                is_error, _ = decoder.decode_memory_z(synx_list, synz_list, final)
            else:
                is_error, _ = decoder.decode_memory_x(synx_list, synz_list, final)
            num_logical_errors += int(is_error)
    elif sample_backend == "native_sampler":
        native_runtime = _native_sampler_runtime(
            distance,
            total_rounds,
            basis,
            physical_error_rate,
            dem_mode,
            native_circuit_source,
            decoder_type=decoder_type,
            ancilla_budget=ancilla_budget,
            p1_scale=p1_scale,
            p_meas_scale=p_meas_scale,
            p_prep_scale=p_prep_scale,
        )
        sampler = native_runtime.sampler
        dem_decoder = native_runtime.dem_decoder
        detection_events, observable_flips = sampler.sample(num_shots=num_shots, seed=seed)

        num_raw_errors = None
        # Fast path: sample+decode entirely in Rust via ObservableDecoder trait.
        # The DemSampler keeps all per-shot data in Rust -- nothing crosses to Python.
        dem_str_for_rust = native_runtime.dem_str
        rust_sampler = getattr(sampler, "sampler", None)
        if dem_str_for_rust and rust_sampler and hasattr(rust_sampler, "sample_decode_count"):
            # Use parallel path for slow decoders (Tesseract, BP+OSD, etc.)
            if decoder_type != "pymatching" and hasattr(rust_sampler, "sample_decode_count_parallel"):
                num_logical_errors = rust_sampler.sample_decode_count_parallel(
                    dem_str_for_rust,
                    num_shots,
                    decoder_type,
                    seed,
                )
            else:
                num_logical_errors = rust_sampler.sample_decode_count(
                    dem_str_for_rust,
                    num_shots,
                    decoder_type,
                    seed,
                )
        else:
            detection_events, observable_flips = sampler.sample(num_shots=num_shots, seed=seed)
            num_logical_errors = _decode_all_shots(dem_decoder, detection_events, observable_flips, num_shots)
    else:
        msg = f"Unknown sample backend: {sample_backend}"
        raise ValueError(msg)

    logical_error_rate = num_logical_errors / num_shots if num_shots else 0.0
    raw_error_rate = None if num_raw_errors is None else (num_raw_errors / num_shots if num_shots else 0.0)

    return SweepPoint(
        backend=backend_label or sample_backend,
        distance=distance,
        basis=basis.upper(),
        physical_error_rate=physical_error_rate,
        total_rounds=total_rounds,
        num_shots=num_shots,
        num_logical_errors=num_logical_errors,
        num_raw_errors=num_raw_errors,
        logical_error_rate=logical_error_rate,
        raw_error_rate=raw_error_rate,
    )


def _fit_per_round_rate(points: list[SweepPoint]) -> float:
    """Fit one per-round logical error rate to several memory durations."""
    if not points:
        msg = "Need at least one point to fit a per-round logical error rate"
        raise ValueError(msg)
    if all(point.logical_error_rate <= 0.0 for point in points):
        return 0.0
    if all(point.logical_error_rate >= 0.5 for point in points):
        return 0.5
    if len(points) == 1:
        point = points[0]
        return ler_per_round_exp(point.logical_error_rate, point.total_rounds)

    def objective(per_round_rate: float) -> float:
        return sum(
            (ler_over_rounds(per_round_rate, point.total_rounds) - point.logical_error_rate) ** 2 for point in points
        )

    left = 0.0
    right = 0.5 - 1e-12  # exclusive upper bound: per-round rate must be strictly below 0.5
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    c = right - (right - left) * inv_phi
    d = left + (right - left) * inv_phi
    fc = objective(c)
    fd = objective(d)

    for _ in range(96):
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - (right - left) * inv_phi
            fc = objective(c)
        else:
            left = c
            c = d
            fc = fd
            d = left + (right - left) * inv_phi
            fd = objective(d)

    return 0.5 * (left + right)


def _fit_summary_from_points(points: list[SweepPoint]) -> FitSummary:
    """Fit a per-round logical rate for one ``(d, basis, p)`` group."""
    if not points:
        msg = "Cannot summarize an empty point group"
        raise ValueError(msg)

    ordered = sorted(points, key=lambda point: point.total_rounds)
    first = ordered[0]
    fitted_per_round = _fit_per_round_rate(ordered)
    per_round_ci_low, per_round_ci_high, projected_ci_low, projected_ci_high = _fit_summary_confidence_intervals(
        ordered,
    )
    residuals = [ler_over_rounds(fitted_per_round, point.total_rounds) - point.logical_error_rate for point in ordered]
    rms_error = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    logical_rate_intervals = [_wilson_interval(point.num_logical_errors, point.num_shots) for point in ordered]
    return FitSummary(
        backend=first.backend,
        distance=first.distance,
        basis=first.basis,
        physical_error_rate=first.physical_error_rate,
        num_shots_per_round_point=first.num_shots,
        round_values=tuple(point.total_rounds for point in ordered),
        observed_logical_error_rates=tuple(point.logical_error_rate for point in ordered),
        observed_raw_error_rates=tuple(point.raw_error_rate for point in ordered),
        fitted_logical_error_rate_per_round=fitted_per_round,
        fitted_projected_logical_error_rate_over_d_rounds=ler_over_rounds(fitted_per_round, first.distance),
        fit_root_mean_square_error=rms_error,
        observed_logical_error_counts=tuple(point.num_logical_errors for point in ordered),
        observed_logical_error_rate_lower_bounds=tuple(interval[0] for interval in logical_rate_intervals),
        observed_logical_error_rate_upper_bounds=tuple(interval[1] for interval in logical_rate_intervals),
        fitted_logical_error_rate_per_round_ci_low=per_round_ci_low,
        fitted_logical_error_rate_per_round_ci_high=per_round_ci_high,
        fitted_projected_logical_error_rate_over_d_rounds_ci_low=projected_ci_low,
        fitted_projected_logical_error_rate_over_d_rounds_ci_high=projected_ci_high,
    )


def _fit_rms_warning_text(summary: FitSummary) -> str:
    """Return a warning string when the fit residual dwarfs the fitted quantity.

    When ``fit_root_mean_square_error`` is at least the fitted per-round rate
    itself, the fit is dominated by statistical noise and the reported
    ``fit_epsilon`` should not be trusted. Empty string means "no warning".
    Skips the degenerate cases where every observed rate is 0 or >= 0.5.
    """
    epsilon = summary.fitted_logical_error_rate_per_round
    if epsilon <= 0.0 or epsilon >= 0.5:
        return ""
    if summary.fit_root_mean_square_error < epsilon:
        return ""
    return (
        f"WARNING: fit_rms ({summary.fit_root_mean_square_error:.3e}) "
        f">= fit_epsilon ({epsilon:.3e}); fit is noise-dominated, increase --shots"
    )


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return ``(slope, intercept)`` for a least-squares line fit."""
    if len(xs) != len(ys):
        msg = "xs and ys must have the same length"
        raise ValueError(msg)
    if len(xs) < 2:
        msg = "Need at least two points for linear regression"
        raise ValueError(msg)

    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx <= 0.0:
        msg = "Linear regression requires at least two distinct x values"
        raise ValueError(msg)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _fit_distance_scaling_at_fixed_p(summaries: list[FitSummary]) -> DistanceScalingFitSummary | None:
    """Fit the standard below-threshold ansatz across distance at one fixed ``p``.

    Requires at least three distinct distances -- fitting a line through two
    points is a tautology (``log_rmse == 0`` always) and the reported threshold
    has no meaning.
    """
    usable = sorted(
        [summary for summary in summaries if summary.fitted_logical_error_rate_per_round > 0.0],
        key=lambda summary: summary.distance,
    )
    if len({summary.distance for summary in usable}) < 3:
        return None

    xs = [0.5 * (summary.distance + 1) for summary in usable]
    ys = [math.log(summary.fitted_logical_error_rate_per_round) for summary in usable]
    slope, intercept = _linear_regression(xs, ys)
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys, strict=True)]
    rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    physical_error_rate = usable[0].physical_error_rate
    suppression_factor = math.exp(-slope)
    threshold = physical_error_rate * suppression_factor
    return DistanceScalingFitSummary(
        backend=usable[0].backend,
        basis=usable[0].basis,
        physical_error_rate=physical_error_rate,
        distances=tuple(summary.distance for summary in usable),
        fitted_prefactor=math.exp(intercept),
        fitted_threshold=threshold,
        fitted_suppression_factor=suppression_factor,
        fit_root_mean_square_log_error=rmse,
    )


def _fit_global_scaling_law(summaries: list[FitSummary]) -> GlobalScalingFitSummary | None:
    """Fit ``epsilon ~= A * (p / p_th) ** ((d + 1) / 2)`` across all ``(d, p)`` points.

    Requires at least three ``(d, p)`` points -- two points fit two parameters
    perfectly (``log_rmse == 0`` always) so the reported threshold is tautological.
    """
    usable = [summary for summary in summaries if summary.fitted_logical_error_rate_per_round > 0.0]
    if len(usable) < 3:
        return None

    xs = [0.5 * (summary.distance + 1) for summary in usable]
    zs = [
        math.log(summary.fitted_logical_error_rate_per_round) - x * math.log(summary.physical_error_rate)
        for summary, x in zip(usable, xs, strict=True)
    ]
    slope, intercept = _linear_regression(xs, zs)
    threshold = math.exp(-slope)
    residuals = []
    for summary in usable:
        x = 0.5 * (summary.distance + 1)
        predicted = intercept + x * (math.log(summary.physical_error_rate) - math.log(threshold))
        residuals.append(math.log(summary.fitted_logical_error_rate_per_round) - predicted)
    rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    return GlobalScalingFitSummary(
        backend=usable[0].backend,
        basis=usable[0].basis,
        distances=tuple(sorted({summary.distance for summary in usable})),
        physical_error_rates=tuple(sorted({summary.physical_error_rate for summary in usable})),
        fitted_prefactor=math.exp(intercept),
        fitted_threshold=threshold,
        fit_root_mean_square_log_error=rmse,
    )


def _fit_per_distance_power_law(
    summaries: list[FitSummary],
    *,
    max_physical_error_rate: float | None = None,
) -> list[PerDistancePowerLawFitSummary]:
    """Fit ``epsilon_d(p) ~= C_d * p ** beta_d`` independently for each distance.

    The power law only holds below threshold -- including p values near or
    above threshold systematically pulls the fitted exponent down from its
    true below-threshold value. Callers that have an estimated threshold
    should pass ``max_physical_error_rate=p_th`` (or a fraction of it) so
    only the sub-threshold regime is fit.

    Also returns the OLS standard error of the slope so callers can display
    uncertainty alongside the exponent.
    """
    fits: list[PerDistancePowerLawFitSummary] = []
    for distance in sorted({summary.distance for summary in summaries}):
        rows = sorted(
            [
                summary
                for summary in summaries
                if summary.distance == distance
                and summary.fitted_logical_error_rate_per_round > 0.0
                and (max_physical_error_rate is None or summary.physical_error_rate <= max_physical_error_rate)
            ],
            key=lambda summary: summary.physical_error_rate,
        )
        if len(rows) < 2:
            continue
        xs = [math.log(summary.physical_error_rate) for summary in rows]
        ys = [math.log(summary.fitted_logical_error_rate_per_round) for summary in rows]
        slope, intercept = _linear_regression(xs, ys)
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys, strict=True)]
        rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
        # Standard error of the OLS slope: sqrt(residual_var / sum((x - x_mean)^2)),
        # where residual_var has Bessel correction (n - 2) for the two fitted parameters.
        n_points = len(rows)
        x_mean = sum(xs) / n_points
        ss_xx = sum((x - x_mean) ** 2 for x in xs)
        if n_points > 2 and ss_xx > 0.0:
            residual_variance = sum(r * r for r in residuals) / (n_points - 2)
            slope_std_error = math.sqrt(residual_variance / ss_xx)
        else:
            slope_std_error = 0.0
        fits.append(
            PerDistancePowerLawFitSummary(
                backend=rows[0].backend,
                basis=rows[0].basis,
                distance=distance,
                physical_error_rates=tuple(summary.physical_error_rate for summary in rows),
                fitted_prefactor=math.exp(intercept),
                fitted_exponent=slope,
                expected_distance_scaling_exponent=0.5 * (distance + 1),
                fit_root_mean_square_log_error=rmse,
                fitted_exponent_std_error=slope_std_error,
            ),
        )
    return fits


def _fit_fss_threshold(
    summaries: list[FitSummary],
    *,
    seed_threshold: float | None = None,
    seed_nu: float = 1.0,
    window_factor_low: float = 0.55,
    window_factor_high: float = 1.5,
) -> FSSThresholdFitSummary | None:
    """Fit the Wang-Harrington-Preskill polynomial FSS form to ``summaries``.

    Uses ``pecos.analysis.threshold_curve.threshold_fit`` with the default
    ``func`` (``p_L = a + b*x + c*x**2`` with ``x = (p - p_th) * d**(1/nu)``).
    The polynomial expansion is only accurate near threshold, so points are
    filtered to the window ``[window_factor_low, window_factor_high] * seed_threshold``
    before fitting. ``seed_threshold`` defaults to the per-round-rate crossing
    estimate from ``_estimate_threshold``. Returns ``None`` when the estimator
    cannot seed, too few points remain in the window, or ``curve_fit`` raises.
    """
    if not summaries:
        return None

    if seed_threshold is None:
        seed_threshold = _estimate_threshold(summaries)
    if seed_threshold is None or seed_threshold <= 0.0:
        return None

    low = seed_threshold * window_factor_low
    high = seed_threshold * window_factor_high
    windowed = [
        summary
        for summary in summaries
        if low <= summary.physical_error_rate <= high and summary.fitted_logical_error_rate_per_round > 0.0
    ]
    if len({summary.distance for summary in windowed}) < 2 or len(windowed) < 5:
        return None

    plist = [summary.physical_error_rate for summary in windowed]
    dlist = [summary.distance for summary in windowed]
    plog = [summary.fitted_logical_error_rate_per_round for summary in windowed]
    mean_plog = sum(plog) / len(plog)
    initial = [seed_threshold, seed_nu, mean_plog, 1.0, 1.0]

    try:
        from pecos.analysis.threshold_curve import func as _fss_func
        from pecos.analysis.threshold_curve import threshold_fit as _fss_threshold_fit
    except ImportError:
        return None

    try:
        popt, stdev = _fss_threshold_fit(plist, dlist, plog, _fss_func, initial)
    except Exception:  # pragma: no cover - scipy fit can fail many ways on bad data
        return None

    p_th, nu, a, b, c = (float(popt[i]) for i in range(5))
    p_th_se, nu_se, a_se, b_se, c_se = (float(stdev[i]) for i in range(5))
    if p_th <= 0.0 or nu <= 0.0:
        # Non-physical fit result -- treat as failure so callers can fall back.
        return None

    first = windowed[0]
    return FSSThresholdFitSummary(
        backend=first.backend,
        basis=first.basis,
        p_th=p_th,
        p_th_std_error=p_th_se,
        nu=nu,
        nu_std_error=nu_se,
        coeff_a=a,
        coeff_a_std_error=a_se,
        coeff_b=b,
        coeff_b_std_error=b_se,
        coeff_c=c,
        coeff_c_std_error=c_se,
        num_points=len(windowed),
        fit_window_low=low,
        fit_window_high=high,
    )


def _estimate_threshold(
    summaries: list[FitSummary],
    *,
    metric: str = "fitted_logical_error_rate_per_round",
) -> float | None:
    """Estimate the ``p`` where the smallest- and largest-distance curves cross.

    Defaults to the canonical threshold definition -- crossing of
    ``fitted_logical_error_rate_per_round``, independent of code distance at
    threshold. Pass ``metric="fitted_projected_logical_error_rate_over_d_rounds"``
    to instead find the crossing on the ``d``-scaled metric, which lies at a
    different (lower) ``p`` because that metric itself scales with ``d``.
    """
    if not summaries:
        return None

    distances = sorted({summary.distance for summary in summaries})
    if len(distances) < 2:
        return None

    d_small = distances[0]
    d_large = distances[-1]
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}
    error_rates = sorted({summary.physical_error_rate for summary in summaries})

    diffs: list[tuple[float, float]] = []
    for p in error_rates:
        small = by_key.get((d_small, p))
        large = by_key.get((d_large, p))
        if small is None or large is None:
            continue
        diffs.append((p, getattr(large, metric) - getattr(small, metric)))

    for (p0, diff0), (p1, diff1) in itertools.pairwise(diffs):
        if diff0 == 0.0:
            return p0
        if diff0 * diff1 < 0.0:
            t = abs(diff0) / (abs(diff0) + abs(diff1))
            return math.exp((1.0 - t) * math.log(p0) + t * math.log(p1))
    return None


def _suppression_summary(summaries: list[FitSummary]) -> list[tuple[float, bool]]:
    """Check whether fitted projected ``d``-round rates decrease with distance."""
    distances = sorted({summary.distance for summary in summaries})
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}

    rows: list[tuple[float, bool]] = []
    for p in error_rates:
        available = [by_key[(d, p)] for d in distances if (d, p) in by_key]
        if len(available) < 2:
            continue
        ordered = [s.fitted_projected_logical_error_rate_over_d_rounds for s in available]
        rows.append((p, all(next_value < value for value, next_value in itertools.pairwise(ordered))))
    return rows


def _distance_scaling_fits(summaries: list[FitSummary]) -> list[DistanceScalingFitSummary]:
    """Fit the distance-scaling ansatz separately at each physical error rate."""
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    fits: list[DistanceScalingFitSummary] = []
    for physical_error_rate in error_rates:
        fit = _fit_distance_scaling_at_fixed_p(
            [summary for summary in summaries if summary.physical_error_rate == physical_error_rate],
        )
        if fit is not None:
            fits.append(fit)
    return fits


def _pairwise_lambda_ratios(summaries: list[FitSummary]) -> list[PairwiseLambdaSummary]:
    """Compute empirical ``Lambda_{d/(d+2)}`` ratios from fitted per-round rates."""
    distances = sorted({summary.distance for summary in summaries})
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}

    ratios: list[PairwiseLambdaSummary] = []
    for physical_error_rate in error_rates:
        for distance_low, distance_high in itertools.pairwise(distances):
            low = by_key.get((distance_low, physical_error_rate))
            high = by_key.get((distance_high, physical_error_rate))
            if low is None or high is None:
                continue
            if low.fitted_logical_error_rate_per_round <= 0.0 or high.fitted_logical_error_rate_per_round <= 0.0:
                continue
            ratios.append(
                PairwiseLambdaSummary(
                    backend=low.backend,
                    basis=low.basis,
                    physical_error_rate=physical_error_rate,
                    distance_low=distance_low,
                    distance_high=distance_high,
                    lambda_d_over_d_plus_2=(
                        low.fitted_logical_error_rate_per_round / high.fitted_logical_error_rate_per_round
                    ),
                ),
            )
    return ratios


def _print_basis_table(summaries: list[FitSummary], *, metric: str, title: str) -> None:
    """Print a compact table for one basis and one fitted metric."""
    distances = sorted({summary.distance for summary in summaries})
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}

    print()
    print(title)
    print("p".ljust(10) + "".join(f"d={distance}".rjust(14) for distance in distances))
    print("-" * (10 + 14 * len(distances)))

    for p in error_rates:
        row = [f"{p:<10.5g}"]
        for distance in distances:
            summary = by_key.get((distance, p))
            if summary is None:
                row.append(f"{'--':>14}")
            else:
                row.append(f"{getattr(summary, metric):>14.6e}")
        print("".join(row))


def _resolve_output_dir(output_dir: str | None, *, wants_outputs: bool) -> Path | None:
    """Choose where optional artifacts should be written."""
    if not wants_outputs:
        return None
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="pecos_surface_threshold_"))


def _basis_summary(summaries: list[FitSummary]) -> dict[str, Any]:
    """Create a compact JSON-friendly summary for one basis."""
    distance_scaling = _distance_scaling_fits(summaries)
    global_scaling = _fit_global_scaling_law(summaries)
    power_law_fits = _fit_per_distance_power_law(summaries)
    lambda_ratios = _pairwise_lambda_ratios(summaries)
    return {
        "per_distance_power_law_fits": [asdict(fit) for fit in power_law_fits],
        "pairwise_lambda_ratios": [asdict(ratio) for ratio in lambda_ratios],
        "fixed_p_distance_scaling_fits": [
            {
                "backend": fit.backend,
                "basis": fit.basis,
                "physical_error_rate": fit.physical_error_rate,
                "distances": fit.distances,
                "fitted_prefactor": fit.fitted_prefactor,
                "fitted_lambda_d_over_d_plus_2": fit.fitted_suppression_factor,
                "fit_root_mean_square_log_error": fit.fit_root_mean_square_log_error,
                "background_implied_threshold": fit.fitted_threshold,
            }
            for fit in distance_scaling
        ],
        "suppression": [
            {
                "physical_error_rate": p,
                "is_suppressed": is_suppressed,
            }
            for p, is_suppressed in _suppression_summary(summaries)
        ],
        "background_threshold_crossing": _estimate_threshold(summaries),
        "background_threshold_crossing_per_round": _estimate_threshold(
            summaries,
            metric="fitted_logical_error_rate_per_round",
        ),
        "background_threshold_crossing_d_rounds": _estimate_threshold(
            summaries,
            metric="fitted_projected_logical_error_rate_over_d_rounds",
        ),
        "background_threshold_style_global_scaling_fit": None if global_scaling is None else asdict(global_scaling),
    }


def _timing_summary(point_timings: list[dict[str, Any]], *, total_wall_clock_seconds: float) -> dict[str, Any]:
    """Aggregate end-to-end sweep timings in a user-facing way."""

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total_seconds = sum(float(row["elapsed_seconds"]) for row in rows)
        total_shots = sum(int(row["num_shots"]) for row in rows)
        return {
            "seconds": total_seconds,
            "shots": total_shots,
            "shots_per_second": (total_shots / total_seconds) if total_seconds > 0.0 else 0.0,
        }

    backends = sorted({str(row["backend"]) for row in point_timings})
    bases = sorted({str(row["basis"]) for row in point_timings})

    per_backend = {
        backend: aggregate([row for row in point_timings if row["backend"] == backend]) for backend in backends
    }
    per_basis = {basis: aggregate([row for row in point_timings if row["basis"] == basis]) for basis in bases}
    per_backend_basis = {
        backend: {
            basis: aggregate(
                [row for row in point_timings if row["backend"] == backend and row["basis"] == basis],
            )
            for basis in bases
            if any(row["backend"] == backend and row["basis"] == basis for row in point_timings)
        }
        for backend in backends
    }

    return {
        "total_wall_clock_seconds": total_wall_clock_seconds,
        "total_point_seconds": sum(float(row["elapsed_seconds"]) for row in point_timings),
        "total_points": len(point_timings),
        "total_shots": sum(int(row["num_shots"]) for row in point_timings),
        "overall_shots_per_second": (
            sum(int(row["num_shots"]) for row in point_timings) / total_wall_clock_seconds
            if total_wall_clock_seconds > 0.0
            else 0.0
        ),
        "per_backend": per_backend,
        "per_basis": per_basis,
        "per_backend_basis": per_backend_basis,
    }


def _print_timing_summary(timing_summary: dict[str, Any]) -> None:
    """Print a compact end-to-end timing summary."""
    print()
    print("Timing Summary")
    print(f"  total wall clock : {timing_summary['total_wall_clock_seconds']:.3f}s")
    print(f"  total point time : {timing_summary['total_point_seconds']:.3f}s")
    print(f"  total points     : {timing_summary['total_points']}")
    print(f"  total shots      : {timing_summary['total_shots']}")
    print(f"  overall throughput: {timing_summary['overall_shots_per_second']:.3f} shots/s")

    print("  by backend:")
    for backend, entry in timing_summary["per_backend"].items():
        print(
            f"    {backend}: {entry['seconds']:.3f}s over {entry['shots']} shots "
            f"({entry['shots_per_second']:.3f} shots/s)",
        )

    print("  by basis:")
    for basis, entry in timing_summary["per_basis"].items():
        print(
            f"    {basis}: {entry['seconds']:.3f}s over {entry['shots']} shots "
            f"({entry['shots_per_second']:.3f} shots/s)",
        )

    print("  by backend+basis:")
    for backend, basis_rows in timing_summary["per_backend_basis"].items():
        basis_text = ", ".join(
            f"{basis}={entry['seconds']:.3f}s/{entry['shots']} shots" for basis, entry in basis_rows.items()
        )
        print(f"    {backend}: {basis_text}")


def _write_json_results(
    output_path: Path,
    *,
    args: argparse.Namespace,
    points: list[SweepPoint],
    summaries: list[FitSummary],
    point_timings: list[dict[str, Any]],
    timing_summary: dict[str, Any],
) -> None:
    """Write sweep results to a JSON artifact."""
    bases = sorted({summary.basis for summary in summaries})
    payload = {
        "config": {
            "distances": sorted(set(args.distances)),
            "bases": bases,
            "sample_backend_mode": args.sample_backend,
            "executed_backends": sorted({point.backend for point in points}),
            "duration_multipliers": sorted(set(args.duration_multipliers)),
            "duration_min_multiplier": args.duration_min_multiplier,
            "duration_max_multiplier": args.duration_max_multiplier,
            "duration_num_points": args.duration_num_points,
            "duration_schedule_description": args.duration_schedule_description,
            "duration_rounds_by_distance": {
                str(distance): list(rounds) for distance, rounds in sorted(args.duration_rounds_by_distance.items())
            },
            "error_rates": sorted(set(args.error_rates)),
            "shots": args.shots,
            "dem_mode": args.dem_mode,
            "native_circuit_source": args.native_circuit_source,
            "seed": args.seed,
            "backend_runtime_descriptions": {
                backend: _backend_runtime_label(backend, args.native_circuit_source)
                for backend in sorted({point.backend for point in points})
            },
            "noise_model": _noise_model_description(args),
            "fit_model": "p_L(r) = 0.5 * (1 - (1 - 2 * epsilon) ** r)",
            "primary_power_law_model": "epsilon_d(p) ~= A_d * p ** c_d",
            "primary_lambda_model": "Lambda_{d/(d+2)}(p) = epsilon_d(p) / epsilon_{d+2}(p)",
            "background_distance_scaling_model": "epsilon ~= A * (p / p_th)^((d + 1) / 2)",
        },
        "points": [asdict(point) for point in points],
        "point_timings": point_timings,
        "fit_summaries": [asdict(summary) for summary in summaries],
        "timing_summary": timing_summary,
        "summary": {
            backend: {
                basis: _basis_summary(
                    [summary for summary in summaries if summary.backend == backend and summary.basis == basis],
                )
                for basis in bases
            }
            for backend in sorted({summary.backend for summary in summaries})
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_matplotlib_pyplot() -> ModuleType:
    """Return ``matplotlib.pyplot``, raising a clear error if it is not installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        msg = "matplotlib is required to render plot output (install matplotlib)"
        raise RuntimeError(msg) from exc
    return plt


# Shared palette used by every plot writer so colors are consistent across the
# duration overlay, per-round overlay, and per-basis curves. Indexed by distance.
_DISTANCE_COLOR_PALETTE: list[str] = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#059669",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0f766e",  # teal
]


def _color_for_distance(distance_index: int) -> str:
    """Return the palette color for the ``distance_index``-th distance (wraps if needed)."""
    return _DISTANCE_COLOR_PALETTE[distance_index % len(_DISTANCE_COLOR_PALETTE)]


def _color_by_distance(distances: list[int]) -> dict[int, str]:
    """Return a mapping from each distance to its palette color."""
    return {distance: _color_for_distance(index) for index, distance in enumerate(distances)}


def _format_rate_for_filename(value: float) -> str:
    """Render a rate in a filename-friendly compact form."""
    return f"{value:.6g}".replace(".", "p")


def _basis_linestyle(basis: str) -> str:
    """Return the matplotlib line style for one basis."""
    return "-" if basis.upper() == "X" else "--"


def _duration_fit_curve_points(
    summary: FitSummary,
    *,
    num_samples: int = 120,
) -> list[tuple[float, float]]:
    """Return a smooth fitted duration curve for one ``(basis, distance, p)`` summary."""
    if not summary.round_values:
        return []

    min_rounds = float(min(summary.round_values))
    max_rounds = float(max(summary.round_values))
    if math.isclose(min_rounds, max_rounds):
        round_samples = [min_rounds]
    else:
        round_samples = [
            min_rounds + (max_rounds - min_rounds) * index / (num_samples - 1) for index in range(num_samples)
        ]

    return [
        (
            total_rounds / summary.distance,
            ler_over_rounds(summary.fitted_logical_error_rate_per_round, total_rounds),
        )
        for total_rounds in round_samples
    ]


def _build_duration_overlay_figure(
    *,
    points: list[SweepPoint],
    summaries: list[FitSummary],
    backend: str,
    physical_error_rate: float,
    figsize: tuple[float, float] = (9.5, 6.5),
) -> Figure:
    """Build the fixed-``p`` logical-error-vs-duration overlay figure (caller closes it)."""
    plt = _require_matplotlib_pyplot()
    distances = sorted({point.distance for point in points})
    bases = sorted({point.basis for point in points})
    color_by_distance = _color_by_distance(distances)

    fig, ax = plt.subplots(figsize=figsize)
    summary_by_series = {(summary.basis, summary.distance): summary for summary in summaries}
    for basis in bases:
        for distance in distances:
            series = sorted(
                [point for point in points if point.basis == basis and point.distance == distance],
                key=lambda point: point.total_rounds,
            )
            if not series:
                continue
            summary = summary_by_series.get((basis, distance))
            color = color_by_distance[distance]
            if summary is not None:
                fit_curve = _duration_fit_curve_points(summary)
                fit_xs = [curve_x for curve_x, _ in fit_curve]
                fit_ys = [max(curve_y, 1e-12) for _, curve_y in fit_curve]
                ax.plot(
                    fit_xs,
                    fit_ys,
                    linewidth=2.5,
                    linestyle=_basis_linestyle(basis),
                    color=color,
                    label=f"{basis} d={distance}",
                )
            xs = [point.total_rounds / point.distance for point in series]
            ys = [max(point.logical_error_rate, 1e-12) for point in series]
            lower_bounds = [_wilson_interval(point.num_logical_errors, point.num_shots)[0] for point in series]
            upper_bounds = [_wilson_interval(point.num_logical_errors, point.num_shots)[1] for point in series]
            yerr_lower = [max(y - low, 0.0) for y, low in zip(ys, lower_bounds, strict=True)]
            yerr_upper = [max(high - y, 0.0) for y, high in zip(ys, upper_bounds, strict=True)]
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lower, yerr_upper],
                marker="o",
                linestyle="none",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                elinewidth=1.2,
                alpha=0.85,
                capsize=3,
            )

    ax.set_title(
        "Logical Memory Error vs Duration "
        f"({backend}, p={physical_error_rate:.4g})\n"
        "Points show observed logical error rates with 95% Wilson intervals; lines show fitted duration curves.",
    )
    ax.set_xlabel("Memory duration (rounds / d)")
    ax.set_ylabel("Observed logical error rate")
    ax.set_yscale("log")
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def _write_duration_overlay_plot(
    output_dir: Path,
    stem: str,
    *,
    points: list[SweepPoint],
    summaries: list[FitSummary],
    backend: str,
    physical_error_rate: float,
    formats: list[str],
) -> list[Path]:
    """Write one fixed-``p`` logical-error-vs-duration overlay to each requested format."""
    if not points or not formats:
        return []
    plt = _require_matplotlib_pyplot()
    fig = _build_duration_overlay_figure(
        points=points,
        summaries=summaries,
        backend=backend,
        physical_error_rate=physical_error_rate,
    )
    output_paths = [output_dir / f"{stem}.{fmt}" for fmt in formats]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def _draw_power_law_exponents(
    ax: Axes,
    summaries: list[FitSummary],
) -> None:
    """Annotate per-distance below-threshold power-law exponents on the plot.

    The power law ``epsilon_d(p) ~= A_d * p ** c_d`` only holds below threshold
    -- including p values near or above threshold compresses the fitted
    exponent toward the noise-dominated value. For each basis, this helper
    estimates ``p_th`` from the per-round rate crossing, then fits only the
    points with ``p <= p_th`` so the reported ``c_d`` reflects the true
    below-threshold scaling (close to the theoretical ``(d + 1) / 2``).

    The annotation shows ``c_d ± se`` where ``se`` is the OLS standard error
    of the slope, and notes how many points fed each fit.
    """
    bases_in_data = sorted({summary.basis for summary in summaries})
    blocks: list[str] = []
    for basis in bases_in_data:
        basis_summaries = [summary for summary in summaries if summary.basis == basis]
        threshold = _estimate_threshold(basis_summaries)
        fits = _fit_per_distance_power_law(basis_summaries, max_physical_error_rate=threshold)
        if not fits:
            # Fall back to fitting all points if threshold estimation fails --
            # better to show a compressed exponent than nothing.
            fits = _fit_per_distance_power_law(basis_summaries)
        if not fits:
            continue
        n_points_used = len(fits[0].physical_error_rates) if fits else 0
        pieces = []
        for fit in fits:
            if fit.fitted_exponent_std_error > 0.0:
                pieces.append(f"c_{fit.distance}={fit.fitted_exponent:.2f}±{fit.fitted_exponent_std_error:.2f}")
            else:
                pieces.append(f"c_{fit.distance}={fit.fitted_exponent:.2f}")
        basis_tag = f"{basis}: " if len(bases_in_data) > 1 else ""
        line = basis_tag + ", ".join(pieces)
        if threshold is not None:
            line += f"  (fit p≤{threshold:.3g}, n={n_points_used})"
        blocks.append(line)

    if not blocks:
        return

    header = "Power-law fit eps_d(p) ≈ A_d · p^c_d   [theory c_d=(d+1)/2]:"
    text = header + "\n" + "\n".join(blocks)
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="#0f172a",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.35"},
    )


def _draw_threshold_markers(
    ax: Axes,
    summaries: list[FitSummary],
    *,
    metric: str = "fitted_logical_error_rate_per_round",
    label_prefix: str = "p_th",
) -> None:
    """Draw a dotted grey vertical line where this metric's curves cross, per basis.

    The crossing point is computed with ``_estimate_threshold(summaries, metric=metric)``
    so the marker matches *this plot's* visual intersection rather than the canonical
    per-round threshold. Callers override ``label_prefix`` (e.g. ``"p_cross"``) when
    the metric is not the canonical per-round rate, to avoid implying these crossings
    are all the threshold. Skipped when the estimator returns ``None``.
    """
    bases_in_data = sorted({summary.basis for summary in summaries})
    # Prefer the Wang-Harrington-Preskill FSS fit (``p_th ± sigma``) for the
    # canonical per-round metric. Fall back to the simpler per-curve crossing
    # estimator when the FSS fit cannot converge (too few near-threshold
    # points) or when the caller is plotting a non-canonical metric.
    use_fss = metric == "fitted_logical_error_rate_per_round"
    # Stack per-basis labels vertically so they do not overlap when two
    # thresholds land near the same ``p``. Top-down in sorted basis order.
    for label_row, basis in enumerate(bases_in_data):
        basis_summaries = [summary for summary in summaries if summary.basis == basis]
        threshold: float | None
        uncertainty: float | None
        fss = _fit_fss_threshold(basis_summaries) if use_fss else None
        if fss is not None:
            threshold = fss.p_th
            uncertainty = fss.p_th_std_error
        else:
            threshold = _estimate_threshold(basis_summaries, metric=metric)
            uncertainty = None
        if threshold is None or threshold <= 0.0:
            continue
        ax.axvline(
            threshold,
            color="#334155",
            linestyle=":",
            linewidth=1.8,
            alpha=0.7,
            zorder=0,
        )
        if uncertainty is not None and uncertainty > 0.0:
            # Shade a +/- one-sigma band so readers can see fit uncertainty directly.
            ax.axvspan(
                max(threshold - uncertainty, 1e-12),
                threshold + uncertainty,
                color="#334155",
                alpha=0.09,
                zorder=0,
            )
        basis_tag = f"({basis})" if len(bases_in_data) > 1 else ""
        if uncertainty is not None and uncertainty > 0.0:
            label = f" {label_prefix}{basis_tag}≈{threshold:.3g}±{uncertainty:.1g}"
        else:
            label = f" {label_prefix}{basis_tag}≈{threshold:.3g}"
        ax.text(
            threshold,
            0.98 - 0.045 * label_row,
            label,
            transform=ax.get_xaxis_transform(),
            color="#334155",
            alpha=0.9,
            fontsize=8,
            ha="left",
            va="top",
        )


def _build_per_round_overlay_figure(
    *,
    summaries: list[FitSummary],
    backend: str,
    figsize: tuple[float, float] = (9.5, 6.5),
) -> Figure:
    """Build the combined X/Z per-round-epsilon-vs-``p`` overlay figure (caller closes it)."""
    plt = _require_matplotlib_pyplot()
    distances = sorted({summary.distance for summary in summaries})
    bases = sorted({summary.basis for summary in summaries})
    color_by_distance = _color_by_distance(distances)

    fig, ax = plt.subplots(figsize=figsize)
    for basis in bases:
        for distance in distances:
            series = sorted(
                [summary for summary in summaries if summary.basis == basis and summary.distance == distance],
                key=lambda summary: summary.physical_error_rate,
            )
            if not series:
                continue
            xs = [summary.physical_error_rate for summary in series]
            intervals = [
                _fit_summary_metric_interval(summary, "fitted_logical_error_rate_per_round") for summary in series
            ]
            ys = [max(value, 1e-12) for value, _, _ in intervals]
            yerr_lower = [max(value - low, 0.0) for value, low, _ in intervals]
            yerr_upper = [max(high - value, 0.0) for value, _, high in intervals]
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_lower, yerr_upper],
                marker="o",
                linewidth=2,
                linestyle=_basis_linestyle(basis),
                color=color_by_distance[distance],
                label=f"{basis} d={distance}",
                capsize=3,
            )

    ax.set_title(f"Per-round logical error rate vs p ({backend})")
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Fitted logical error rate per round")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(ncol=2)
    _draw_threshold_markers(ax, summaries)
    _draw_power_law_exponents(ax, summaries)
    fig.tight_layout()
    return fig


def _write_per_round_overlay_plot(
    output_dir: Path,
    stem: str,
    *,
    summaries: list[FitSummary],
    backend: str,
    formats: list[str],
) -> list[Path]:
    """Write the combined X/Z per-round-epsilon-vs-``p`` overlay to each requested format."""
    if not summaries or not formats:
        return []
    plt = _require_matplotlib_pyplot()
    fig = _build_per_round_overlay_figure(summaries=summaries, backend=backend)
    output_paths = [output_dir / f"{stem}.{fmt}" for fmt in formats]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def _build_plot_figure(
    *,
    summaries: list[FitSummary],
    metric: str,
    title: str,
    y_label: str,
    figsize: tuple[float, float] = (9, 6),
) -> Figure:
    """Build a per-basis epsilon-vs-``p`` curve figure (caller closes it)."""
    plt = _require_matplotlib_pyplot()
    distances = sorted({summary.distance for summary in summaries})
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}
    color_by_distance = _color_by_distance(distances)

    fig, ax = plt.subplots(figsize=figsize)
    for distance in distances:
        available_ps = [p for p in error_rates if (distance, p) in by_key]
        if not available_ps:
            continue
        intervals = [_fit_summary_metric_interval(by_key[(distance, p)], metric) for p in available_ps]
        ys = [max(value, 1e-12) for value, _, _ in intervals]
        yerr_lower = [max(value - low, 0.0) for value, low, _ in intervals]
        yerr_upper = [max(high - value, 0.0) for value, _, high in intervals]
        ax.errorbar(
            available_ps,
            ys,
            yerr=[yerr_lower, yerr_upper],
            marker="o",
            linewidth=2,
            color=color_by_distance[distance],
            label=f"d={distance}",
            capsize=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel(y_label)
    # LER-vs-p plots read most naturally on log-log: the below-threshold
    # power law eps_d(p) ~ A_d * p^c_d becomes a straight line with slope c_d,
    # which matches the annotated exponents and makes crossings easy to eyeball.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend()
    # Per-round rate is the canonical threshold metric -- call that crossing ``p_th``.
    # The projected-over-d-rounds curves cross at a different ``p`` because the metric
    # itself scales with ``d``, so we mark it but label it ``p_cross`` to be honest
    # about what the line represents.
    is_per_round = metric == "fitted_logical_error_rate_per_round"
    _draw_threshold_markers(
        ax,
        summaries,
        metric=metric,
        label_prefix="p_th" if is_per_round else "p_cross",
    )
    # Power-law fit eps_d ~= A_d * p^c_d is defined against the per-round rate,
    # so annotate it only on the per-round per-basis plot.
    if is_per_round:
        _draw_power_law_exponents(ax, summaries)
    fig.tight_layout()
    return fig


def _write_plot(
    output_dir: Path,
    stem: str,
    *,
    summaries: list[FitSummary],
    metric: str,
    title: str,
    y_label: str,
    formats: list[str],
) -> list[Path]:
    """Write a per-basis epsilon-vs-``p`` curve plot to each requested format."""
    if not summaries or not formats:
        return []
    plt = _require_matplotlib_pyplot()
    fig = _build_plot_figure(summaries=summaries, metric=metric, title=title, y_label=y_label)
    output_paths = [output_dir / f"{stem}.{fmt}" for fmt in formats]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def _write_html_dashboard(
    output_path: Path,
    *,
    args: argparse.Namespace,
    summaries: list[FitSummary],
    timing_summary: dict[str, Any],
    plots: list[DashboardPlot],
    json_filename: str | None,
) -> None:
    """Write a simple browsable HTML report for the generated SVG artifacts."""
    from textwrap import dedent

    def meta_card(label: str, value_html: str) -> str:
        return f'      <div class="meta-card"><strong>{html.escape(label)}</strong>{value_html}</div>'

    def plot_card(plot: DashboardPlot) -> list[str]:
        detail_bits = [f"backend={plot.backend}"]
        if plot.basis is not None:
            detail_bits.append(f"basis={plot.basis}")
        if plot.physical_error_rate is not None:
            detail_bits.append(f"p={plot.physical_error_rate:.4g}")
        details = ", ".join(detail_bits)
        image_link = html.escape(plot.filename)
        title = html.escape(plot.title)
        return [
            '      <article class="plot-card">',
            f"        <header><h3>{title}</h3><p>{html.escape(details)}</p></header>",
            '        <div class="image-wrap">',
            f'          <a href="{image_link}">',
            f'            <img src="{image_link}" alt="{title}" />',
            "          </a>",
            "        </div>",
            f'        <footer><a href="{image_link}">Open SVG</a></footer>',
            "      </article>",
        ]

    backend_names = sorted({summary.backend for summary in summaries})
    basis_names = sorted({summary.basis for summary in summaries})
    plot_sections = [
        ("Combined Overlays", [plot for plot in plots if plot.section == "combined"]),
        ("Fixed-p Duration Overlays", [plot for plot in plots if plot.section == "duration"]),
        ("Per-basis Curves", [plot for plot in plots if plot.section == "basis"]),
    ]
    style = dedent(
        """
        :root {
          color-scheme: light dark;
          --bg: #f8fafc; --fg: #0f172a;
          --hero-bg: linear-gradient(135deg, #e0f2fe, #f8fafc 55%, #dcfce7);
          --hero-border: #cbd5e1;
          --card-bg: white; --card-border: #dbeafe; --card-shadow: rgba(15,23,42,0.05);
          --meta-bg: rgba(255,255,255,0.82);
          --muted: #475569; --link: #2563eb; --link-alt: #0369a1;
          --header-border: #e2e8f0;
          --img-bg: white;
        }
        [data-theme="dark"] {
          --bg: #0f172a; --fg: #e2e8f0;
          --hero-bg: linear-gradient(135deg, #1e293b, #0f172a 55%, #1a2e1a);
          --hero-border: #334155;
          --card-bg: #1e293b; --card-border: #334155; --card-shadow: rgba(0,0,0,0.3);
          --meta-bg: rgba(30,41,59,0.82);
          --muted: #94a3b8; --link: #60a5fa; --link-alt: #38bdf8;
          --header-border: #334155;
          --img-bg: #1e293b;
        }
        body {
          margin: 0;
          font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif;
          background: var(--bg);
          color: var(--fg);
        }
        main { max-width: 1500px; margin: 0 auto; padding: 32px 24px 56px; }
        h1, h2, h3, p { margin-top: 0; }
        .theme-toggle {
          position: fixed; top: 16px; right: 16px; z-index: 100;
          background: var(--card-bg); border: 1px solid var(--card-border);
          border-radius: 8px; padding: 6px 12px; cursor: pointer;
          color: var(--fg); font-size: 0.85rem; font-weight: 600;
        }
        .theme-toggle:hover { opacity: 0.8; }
        .hero {
          background: var(--hero-bg);
          border: 1px solid var(--hero-border);
          border-radius: 20px;
          padding: 24px;
          margin-bottom: 24px;
        }
        .meta {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
          margin-top: 18px;
        }
        .meta-card {
          background: var(--meta-bg);
          border: 1px solid var(--card-border);
          border-radius: 14px;
          padding: 14px 16px;
        }
        .meta-card strong {
          display: block;
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: var(--muted);
          margin-bottom: 6px;
        }
        .section { margin-top: 30px; }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
          gap: 18px;
        }
        .plot-card {
          background: var(--card-bg);
          border: 1px solid var(--card-border);
          border-radius: 18px;
          overflow: hidden;
          box-shadow: 0 10px 24px var(--card-shadow);
        }
        .plot-card header {
          padding: 16px 18px 10px;
          border-bottom: 1px solid var(--header-border);
        }
        .plot-card header p {
          margin-bottom: 0;
          color: var(--muted);
          font-size: 0.92rem;
        }
        .plot-card .image-wrap { padding: 14px; background: var(--img-bg); }
        .plot-card img {
          width: 100%;
          height: auto;
          display: block;
          border-radius: 12px;
          background: var(--img-bg);
        }
        .plot-card footer { padding: 0 18px 16px; font-size: 0.92rem; }
        .plot-card a { color: var(--link); text-decoration: none; font-weight: 600; }
        .plot-card a:hover { text-decoration: underline; }
        .links { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 12px; }
        .links a { color: var(--link-alt); text-decoration: none; font-weight: 600; }
        .links a:hover { text-decoration: underline; }
        code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) {
            --bg: #0f172a; --fg: #e2e8f0;
            --hero-bg: linear-gradient(135deg, #1e293b, #0f172a 55%, #1a2e1a);
            --hero-border: #334155;
            --card-bg: #1e293b; --card-border: #334155; --card-shadow: rgba(0,0,0,0.3);
            --meta-bg: rgba(30,41,59,0.82);
            --muted: #94a3b8; --link: #60a5fa; --link-alt: #38bdf8;
            --header-border: #334155;
            --img-bg: #1e293b;
          }
        }
        """,
    ).strip()
    theme_script = dedent(
        """
        <script>
        (function() {
          var html = document.documentElement;
          var btn = document.getElementById('theme-toggle');
          var stored = localStorage.getItem('pecos-theme');
          if (stored) html.setAttribute('data-theme', stored);
          btn.addEventListener('click', function() {
            var current = html.getAttribute('data-theme');
            var next = current === 'dark' ? 'light' : 'dark';
            if (!current) next = 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('pecos-theme', next);
          });
        })();
        </script>
        """,
    ).strip()
    distances_text = ", ".join(str(distance) for distance in sorted(set(args.distances)))
    multipliers_text = getattr(args, "duration_schedule_description", None)
    if not multipliers_text:
        multipliers_text = ", ".join(f"{value:g}" for value in sorted(set(args.duration_multipliers)))
    rounds_by_distance = getattr(args, "duration_rounds_by_distance", {})
    rounds_lines = [
        f"d={distance}: [{', '.join(str(value) for value in rounds)}]"
        for distance, rounds in sorted(rounds_by_distance.items())
    ]
    # Render one distance per line; html-escape each line individually since we
    # join with literal <br> markup that must not itself be escaped.
    rounds_html = "<br>".join(html.escape(line) for line in rounds_lines)
    error_rates_text = ", ".join(f"{value:.4g}" for value in sorted(set(args.error_rates)))

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        "  <title>PECOS Surface Sweep Dashboard</title>",
        "  <style>",
        style,
        "  </style>",
        "</head>",
        "<body>",
        '<button id="theme-toggle" class="theme-toggle">Light / Dark</button>',
        "<main>",
        '  <section class="hero">',
        "    <h1>PECOS Surface Sweep Dashboard</h1>",
        (
            "    <p>This report bundles the generated SVG plots for the rotated "
            "surface-code memory sweep so the run is easy to browse and compare.</p>"
        ),
        '    <div class="meta">',
        meta_card("Backends", html.escape(", ".join(backend_names))),
        meta_card("Bases", html.escape(", ".join(basis_names))),
        meta_card("Distances", html.escape(distances_text)),
        meta_card("Round Schedule", html.escape(multipliers_text)),
        meta_card("Error Rates", html.escape(error_rates_text)),
        meta_card("Shots / Point", html.escape(str(args.shots))),
        meta_card(
            "Noise Model",
            html.escape(_noise_model_description(args)),
        ),
        meta_card(
            "Overall Throughput",
            html.escape(f"{timing_summary['overall_shots_per_second']:.3f} shots/s"),
        ),
        meta_card("Effective Rounds", rounds_html) if rounds_html else "",
        "    </div>",
    ]
    if json_filename is not None:
        parts.extend(
            [
                '    <div class="links">',
                f'      <a href="{html.escape(json_filename)}">Open JSON results</a>',
                "    </div>",
            ],
        )
    parts.append("  </section>")

    for section_title, section_plots in plot_sections:
        if not section_plots:
            continue
        parts.extend(
            [
                f'  <section class="section"><h2>{html.escape(section_title)}</h2>',
                '    <div class="grid">',
            ],
        )
        for plot in section_plots:
            parts.extend(plot_card(plot))
        parts.extend(["    </div>", "  </section>"])

    parts.extend(["</main>", theme_script, "</body>", "</html>"])
    output_path.write_text("\n".join(parts) + "\n")


def _maybe_open_html_dashboard(output_path: Path) -> None:
    """Open the generated dashboard in the default browser."""
    import webbrowser

    opened = webbrowser.open(output_path.resolve().as_uri())
    if not opened:
        msg = f"Failed to open HTML dashboard at {output_path}"
        raise RuntimeError(msg)


def load_sweep_data_from_json(
    json_path: Path,
) -> tuple[list[SweepPoint], list[FitSummary], dict[str, Any]]:
    """Reconstruct ``(points, fit_summaries, payload)`` from a saved JSON results file.

    Used by ``surface_sweep_report.py --render-plots`` to rebuild plots from the
    canonical JSON data without rerunning the simulation. ``points`` and
    ``fit_summaries`` are returned as the original frozen dataclasses, with
    ``FitSummary``'s tuple fields recovered from JSON-array form.
    """
    payload = json.loads(json_path.read_text())
    points = [SweepPoint(**row) for row in payload.get("points", [])]
    tuple_fields = {
        "round_values",
        "observed_logical_error_rates",
        "observed_raw_error_rates",
        "observed_logical_error_counts",
        "observed_logical_error_rate_lower_bounds",
        "observed_logical_error_rate_upper_bounds",
    }
    summaries: list[FitSummary] = []
    for row in payload.get("fit_summaries", []):
        kwargs = {key: (tuple(value) if key in tuple_fields else value) for key, value in row.items()}
        summaries.append(FitSummary(**kwargs))
    return points, summaries, payload


def _merge_sweep_point_group(points: list[SweepPoint]) -> SweepPoint:
    """Merge same-key SweepPoints by summing counts and recomputing rates.

    All points in the input must share the same
    ``(backend, distance, basis, physical_error_rate, total_rounds)`` key.
    """
    if not points:
        msg = "Cannot merge an empty point group"
        raise ValueError(msg)

    first = points[0]
    total_shots = sum(point.num_shots for point in points)
    total_logical_errors = sum(point.num_logical_errors for point in points)
    raw_values = [point.num_raw_errors for point in points]
    if all(value is not None for value in raw_values):
        total_raw_errors: int | None = sum(v for v in raw_values if v is not None)
        raw_rate: float | None = total_raw_errors / total_shots if total_shots > 0 else 0.0
    else:
        total_raw_errors = None
        raw_rate = None
    logical_rate = total_logical_errors / total_shots if total_shots > 0 else 0.0
    return SweepPoint(
        backend=first.backend,
        distance=first.distance,
        basis=first.basis,
        physical_error_rate=first.physical_error_rate,
        total_rounds=first.total_rounds,
        num_shots=total_shots,
        num_logical_errors=total_logical_errors,
        num_raw_errors=total_raw_errors,
        logical_error_rate=logical_rate,
        raw_error_rate=raw_rate,
    )


def _merge_sweep_configs(configs: list[dict[str, Any]], source_paths: list[str]) -> dict[str, Any]:
    """Merge shard configs: union list fields, keep per-shard shot detail in ``shots_per_shard``."""
    if not configs:
        return {"source_shards": list(source_paths)}

    merged: dict[str, Any] = dict(configs[0])

    for field in ("distances", "error_rates", "bases", "executed_backends"):
        union: set[Any] = set()
        for config in configs:
            for value in config.get(field, []) or []:
                union.add(value)
        if union:
            merged[field] = sorted(union)

    rounds_by_distance: dict[str, set[int]] = {}
    for config in configs:
        for distance_key, rounds in (config.get("duration_rounds_by_distance") or {}).items():
            rounds_by_distance.setdefault(str(distance_key), set()).update(int(r) for r in rounds)
    if rounds_by_distance:
        merged["duration_rounds_by_distance"] = {
            distance_key: sorted(rounds) for distance_key, rounds in rounds_by_distance.items()
        }

    shot_values = [config.get("shots") for config in configs if config.get("shots") is not None]
    if len(set(shot_values)) == 1:
        merged["shots"] = shot_values[0]
    else:
        # When shards used different per-point targets, reporting a single
        # scalar would mislead readers. Record the distinct values instead.
        merged["shots"] = ", ".join(str(value) for value in sorted(set(shot_values)))

    # Provenance: one row per shard with path + shots + description.
    shards_metadata = []
    for path, config in zip(source_paths, configs, strict=True):
        shards_metadata.append(
            {
                "path": path,
                "shots": config.get("shots"),
                "duration_schedule_description": config.get("duration_schedule_description"),
                "executed_backends": config.get("executed_backends", []),
            },
        )
    merged["source_shards"] = shards_metadata

    return merged


def _merge_sweep_timings(timings: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum scalar timing totals across shards; recompute throughput from totals.

    Note: ``total_wall_clock_seconds`` is the sum of each shard's wall-clock
    time. When shards ran in parallel this overstates the actual elapsed time
    and should be interpreted as total CPU time across shards.
    """
    total_wall = sum(timing.get("total_wall_clock_seconds", 0.0) for timing in timings)
    total_point = sum(timing.get("total_point_seconds", 0.0) for timing in timings)
    total_shots = sum(timing.get("total_shots", 0) or 0 for timing in timings)
    total_points = sum(timing.get("total_points", 0) or 0 for timing in timings)
    return {
        "total_wall_clock_seconds": total_wall,
        "total_point_seconds": total_point,
        "total_shots": total_shots,
        "total_points": total_points,
        "overall_shots_per_second": (total_shots / total_wall) if total_wall > 0 else 0.0,
    }


def merge_sweep_shards(
    paths: list[Path],
) -> tuple[list[SweepPoint], list[FitSummary], dict[str, Any], dict[str, Any]]:
    """Load ``paths`` as sweep shards, merge points by key, re-fit, return merged data.

    Two ``SweepPoint`` entries with the same
    ``(backend, distance, basis, physical_error_rate, total_rounds)`` key are
    merged by summing ``num_shots`` and ``num_logical_errors`` and recomputing
    the rate. ``FitSummary`` entries are re-derived from the merged points --
    not carried over from the shards -- because fit statistics depend on the
    merged shot counts.

    Returns ``(merged_points, merged_summaries, merged_config, merged_timing_summary)``.
    """
    if not paths:
        msg = "At least one shard path is required"
        raise ValueError(msg)

    all_shard_points: list[SweepPoint] = []
    shard_configs: list[dict[str, Any]] = []
    shard_timings: list[dict[str, Any]] = []
    for path in paths:
        points, _summaries, payload = load_sweep_data_from_json(path)
        all_shard_points.extend(points)
        shard_configs.append(dict(payload.get("config", {})))
        shard_timings.append(dict(payload.get("timing_summary", {})))

    # Group by merge key and merge each group.
    point_groups: dict[tuple[str, int, str, float, int], list[SweepPoint]] = {}
    for point in all_shard_points:
        key = (point.backend, point.distance, point.basis, point.physical_error_rate, point.total_rounds)
        point_groups.setdefault(key, []).append(point)
    merged_points = [_merge_sweep_point_group(group) for group in point_groups.values()]
    merged_points.sort(
        key=lambda point: (point.backend, point.distance, point.basis, point.physical_error_rate, point.total_rounds),
    )

    # Re-fit: group merged points by (backend, basis, distance, p) and fit each group.
    fit_groups: dict[tuple[str, str, int, float], list[SweepPoint]] = {}
    for point in merged_points:
        fit_groups.setdefault((point.backend, point.basis, point.distance, point.physical_error_rate), []).append(
            point,
        )
    merged_summaries = [
        _fit_summary_from_points(sorted(group, key=lambda point: point.total_rounds)) for group in fit_groups.values()
    ]
    merged_summaries.sort(
        key=lambda summary: (summary.backend, summary.basis, summary.distance, summary.physical_error_rate),
    )

    merged_config = _merge_sweep_configs(shard_configs, [str(path) for path in paths])
    merged_timing = _merge_sweep_timings(shard_timings)

    return merged_points, merged_summaries, merged_config, merged_timing


def render_plot_artifacts(
    output_dir: Path,
    *,
    prefix: str,
    points: list[SweepPoint],
    summaries: list[FitSummary],
    formats: list[str],
) -> list[DashboardPlot]:
    """Render every plot type from in-memory data and return the dashboard plot list.

    Used both by the live sweep (which feeds in just-collected data) and by
    ``surface_sweep_report.py --render-plots`` (which feeds in data
    reconstructed from a saved JSON results file). Only SVG paths get a
    ``DashboardPlot`` entry, since the dashboard embeds SVG only.
    """
    dashboard_plots: list[DashboardPlot] = []
    backends = sorted({summary.backend for summary in summaries})

    def _report_written(paths: list[Path]) -> None:
        for path in paths:
            print(f"Wrote {path.suffix.lstrip('.').upper()} plot to {path}")

    def _svg_path_for(paths: list[Path]) -> Path | None:
        return next((path for path in paths if path.suffix == ".svg"), None)

    for backend in backends:
        backend_summaries = [summary for summary in summaries if summary.backend == backend]
        overlay_paths = _write_per_round_overlay_plot(
            output_dir,
            f"{prefix}_{backend}_per_round_overlay",
            summaries=backend_summaries,
            backend=backend,
            formats=formats,
        )
        _report_written(overlay_paths)
        overlay_svg = _svg_path_for(overlay_paths)
        if overlay_svg is not None:
            dashboard_plots.append(
                DashboardPlot(
                    section="combined",
                    title=f"Per-round logical error rate vs p ({backend})",
                    filename=overlay_svg.name,
                    backend=backend,
                ),
            )

        for physical_error_rate in sorted({point.physical_error_rate for point in points if point.backend == backend}):
            rate_points = [
                point
                for point in points
                if point.backend == backend and point.physical_error_rate == physical_error_rate
            ]
            rate_summaries = [
                summary for summary in backend_summaries if summary.physical_error_rate == physical_error_rate
            ]
            stem = f"{prefix}_{backend}_p_{_format_rate_for_filename(physical_error_rate)}_duration_overlay"
            duration_paths = _write_duration_overlay_plot(
                output_dir,
                stem,
                points=rate_points,
                summaries=rate_summaries,
                backend=backend,
                physical_error_rate=physical_error_rate,
                formats=formats,
            )
            _report_written(duration_paths)
            duration_svg = _svg_path_for(duration_paths)
            if duration_svg is not None:
                dashboard_plots.append(
                    DashboardPlot(
                        section="duration",
                        title=f"Logical memory error vs duration ({backend}, p={physical_error_rate:.4g})",
                        filename=duration_svg.name,
                        backend=backend,
                        physical_error_rate=physical_error_rate,
                    ),
                )

    for backend in backends:
        for basis in sorted({summary.basis for summary in summaries if summary.backend == backend}):
            basis_summaries = [
                summary for summary in summaries if summary.backend == backend and summary.basis == basis
            ]
            plot_specs = [
                (
                    "fitted_projected_logical_error_rate_over_d_rounds",
                    f"{prefix}_{backend}_{basis.lower()}_projected_d_rounds",
                    f"{basis}-Basis Fitted Logical Error Rate Over d Rounds ({backend})",
                    "Fitted logical error rate over d rounds",
                ),
                (
                    "fitted_logical_error_rate_per_round",
                    f"{prefix}_{backend}_{basis.lower()}_per_round",
                    f"{basis}-Basis Fitted Logical Error Rate Per Round ({backend})",
                    "Fitted logical error rate per round",
                ),
            ]
            for metric, stem, title, y_label in plot_specs:
                plot_paths = _write_plot(
                    output_dir,
                    stem,
                    summaries=basis_summaries,
                    metric=metric,
                    title=title,
                    y_label=y_label,
                    formats=formats,
                )
                _report_written(plot_paths)
                plot_svg = _svg_path_for(plot_paths)
                if plot_svg is not None:
                    dashboard_plots.append(
                        DashboardPlot(
                            section="basis",
                            title=title,
                            filename=plot_svg.name,
                            backend=backend,
                            basis=basis,
                        ),
                    )

    return dashboard_plots


# Letter-landscape page size for every PDF report page -- matches the 11x8.5
# aspect ratio used by cover, section dividers, and each plot page so the
# reader does not see page-size jitter when flipping through the report.
_REPORT_PAGE_SIZE: tuple[float, float] = (11.0, 8.5)


def _draw_meta_card(
    page: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    value_lines: list[str],
) -> None:
    """Draw a single HTML-like meta-card (label + value block) onto the page axes.

    Long values are wrapped to fit the card width so nothing runs past the
    card border. The wrap width is a conservative character-count heuristic
    based on ``width`` (in figure coordinates) and the 11" report page width.
    """
    import textwrap

    from matplotlib.patches import FancyBboxPatch

    card = FancyBboxPatch(
        (x + 0.002, y + 0.002),
        width - 0.004,
        height - 0.004,
        boxstyle="round,pad=0.002,rounding_size=0.012",
        linewidth=1.2,
        facecolor="#ffffff",
        edgecolor="#dbeafe",
        transform=page.transAxes,
    )
    page.add_patch(card)
    page.text(
        x + 0.018,
        y + height - 0.028,
        label.upper(),
        fontsize=8.5,
        color="#475569",
        weight="bold",
        transform=page.transAxes,
    )

    # Wrap values that would otherwise overflow the card width.
    page_width_inches = _REPORT_PAGE_SIZE[0]
    padding_inches = 0.036 * page_width_inches  # 0.018 fig-coord padding each side
    # DejaVu Sans at 10.5pt averages ~0.09in per character; overestimate slightly
    # so we wrap sooner rather than clip.
    char_width_inches = 0.09
    usable_inches = max(0.5, width * page_width_inches - padding_inches)
    max_chars = max(10, int(usable_inches / char_width_inches))
    wrapped: list[str] = []
    for line in value_lines:
        if len(line) <= max_chars:
            wrapped.append(line)
            continue
        pieces = textwrap.wrap(line, width=max_chars, break_long_words=False) or [line]
        wrapped.extend(pieces)

    for offset, line in enumerate(wrapped):
        page.text(
            x + 0.018,
            y + height - 0.06 - offset * 0.024,
            line,
            fontsize=10.5,
            color="#0f172a",
            transform=page.transAxes,
        )


def _build_report_cover_figure(
    *,
    config: dict[str, Any] | None,
    summaries: list[FitSummary],
    title: str = "PECOS Surface Sweep Report",
) -> Figure:
    """Build a styled cover page: hero band + meta-card grid + footer timing line."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=_REPORT_PAGE_SIZE, facecolor="#f8fafc")

    # Full-page axes used to position meta cards and footer text in normalized coords.
    page = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    page.set_xlim(0, 1)
    page.set_ylim(0, 1)
    page.axis("off")
    page.patch.set_facecolor("#f8fafc")

    # --- Hero band with gradient + title + subtitle ---
    hero = fig.add_axes((0.04, 0.74, 0.92, 0.22))
    hero.set_xticks([])
    hero.set_yticks([])
    for spine in hero.spines.values():
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(1.0)
    hero_gradient = [[column / 255.0 for column in range(256)]]
    hero_cmap = LinearSegmentedColormap.from_list("pecos-hero", ["#e0f2fe", "#f8fafc", "#dcfce7"])
    hero.imshow(hero_gradient, aspect="auto", cmap=hero_cmap, extent=(0.0, 1.0, 0.0, 1.0))
    hero.text(
        0.5,
        0.62,
        title,
        transform=hero.transAxes,
        ha="center",
        va="center",
        fontsize=26,
        weight="bold",
        color="#0f172a",
    )
    hero.text(
        0.5,
        0.30,
        "Rotated surface code memory experiments",
        transform=hero.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#475569",
    )

    # --- Meta-card grid: only the scientific headline parameters. Run-level
    # details (shots, timing, DEM mode, schedule, effective rounds) live in the
    # appendix so this cover stays focused on what was studied.
    backends_in_data = sorted({summary.backend for summary in summaries}) or ["(none)"]
    bases_in_data = sorted({summary.basis for summary in summaries}) or ["(none)"]
    config = config or {}

    cards: list[tuple[str, list[str], int]] = [
        ("Backends", [", ".join(backends_in_data)], 1),
        ("Bases", [", ".join(bases_in_data)], 1),
        ("Distances", [", ".join(str(d) for d in config.get("distances", [])) or "(none)"], 1),
        ("Error Rates", [", ".join(f"{p:.4g}" for p in config.get("error_rates", [])) or "(none)"], 1),
        (
            "Noise Model",
            [config.get("noise_model", "depolarizing")],
            2,
        ),
    ]

    cols = 3
    gap = 0.015
    grid_left = 0.04
    grid_right = 0.96
    unit_width = (grid_right - grid_left - (cols - 1) * gap) / cols
    card_height = 0.18
    row_y_positions = [0.50, 0.28]
    col_cursor = 0
    row_cursor = 0
    for label, value_lines, span in cards:
        if col_cursor + span > cols:
            row_cursor += 1
            col_cursor = 0
        if row_cursor >= len(row_y_positions):
            break
        card_x = grid_left + col_cursor * (unit_width + gap)
        card_w = unit_width * span + gap * (span - 1)
        _draw_meta_card(
            page,
            x=card_x,
            y=row_y_positions[row_cursor],
            width=card_w,
            height=card_height,
            label=label,
            value_lines=value_lines,
        )
        col_cursor += span

    # --- Footer hint pointing readers at the appendix for methods/timing ---
    fig.text(
        0.5,
        0.12,
        "See the Appendix at the end of this report for methods, shot counts, and timing details.",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
        style="italic",
    )

    return fig


def _build_appendix_figure(
    *,
    config: dict[str, Any] | None,
    timing_summary: dict[str, Any] | None,
    summaries: list[FitSummary] | None = None,
) -> Figure:
    """Build the "Methods and Timing" appendix page (two columns of key/value rows).

    When ``summaries`` are provided and cover enough near-threshold data, a
    third section lists the Wang-Harrington-Preskill FSS fit per (backend,
    basis) with fitted ``p_th`` and ``nu`` plus their standard errors.
    """
    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=_REPORT_PAGE_SIZE, facecolor="#f8fafc")
    page = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    page.set_xlim(0, 1)
    page.set_ylim(0, 1)
    page.axis("off")
    page.patch.set_facecolor("#f8fafc")

    fig.text(0.5, 0.92, "Appendix: Methods and Timing", ha="center", fontsize=24, weight="bold", color="#0f172a")
    fig.text(
        0.5,
        0.87,
        "Run-level parameters and timing for reproducibility",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    config = config or {}
    timing_summary = timing_summary or {}

    rounds_by_distance = {
        int(distance): tuple(values) for distance, values in config.get("duration_rounds_by_distance", {}).items()
    }
    rounds_lines = [f"d={distance}: {list(rounds)}" for distance, rounds in sorted(rounds_by_distance.items())] or [
        "(no schedule recorded)",
    ]

    method_rows: list[tuple[str, list[str]]] = [
        ("Shots / Point", [str(config.get("shots", "?"))]),
        ("Sample Backend Mode", [str(config.get("sample_backend_mode", "(unspecified)"))]),
        ("Executed Backends", [", ".join(config.get("executed_backends", [])) or "(unspecified)"]),
        ("DEM Mode", [str(config.get("dem_mode", "(unspecified)"))]),
        ("Native Circuit Source", [str(config.get("native_circuit_source", "(unspecified)"))]),
        ("RNG Seed", [str(config.get("seed", "(unspecified)"))]),
        ("Round Schedule", [str(config.get("duration_schedule_description", "(unspecified)"))]),
        ("Effective Rounds", rounds_lines),
    ]

    timing_rows: list[tuple[str, list[str]]] = [
        (
            "Total Wall Clock",
            [f"{timing_summary.get('total_wall_clock_seconds', 0.0):.2f} s"],
        ),
        ("Total Shots", [str(timing_summary.get("total_shots", "?"))]),
        (
            "Overall Throughput",
            [f"{timing_summary.get('overall_shots_per_second', 0.0):.1f} shots/s"],
        ),
        (
            "Total Point Time",
            [f"{timing_summary.get('total_point_seconds', 0.0):.2f} s"],
        ),
        ("Total Points", [str(timing_summary.get("total_points", "?"))]),
    ]

    def _render_column(x: float, heading: str, rows: list[tuple[str, list[str]]]) -> None:
        fig.text(x, 0.80, heading, fontsize=14, weight="bold", color="#0f172a")
        page.add_patch(
            _section_accent(x_left=x, x_right=x + 0.42, y=0.785),
        )
        cursor = 0.75
        for label, values in rows:
            fig.text(x, cursor, f"{label}:", fontsize=10.5, color="#475569", weight="bold")
            for value in values:
                cursor -= 0.028
                fig.text(x + 0.015, cursor, value, fontsize=10.5, color="#0f172a")
            cursor -= 0.018

    _render_column(0.08, "Methods", method_rows)
    _render_column(0.54, "Timing", timing_rows)

    # --- Optional third section: FSS threshold fits per (backend, basis) ---
    fss_rows = _collect_fss_fit_rows(summaries or [])
    if fss_rows:
        fig.text(
            0.08,
            0.36,
            "Threshold Fit (Wang-Harrington-Preskill)",
            fontsize=14,
            weight="bold",
            color="#0f172a",
        )
        page.add_patch(_section_accent(x_left=0.08, x_right=0.62, y=0.345))
        fig.text(
            0.08,
            0.325,
            "p_L = a + b*x + c*x^2,  x = (p - p_th)*d^(1/nu)    [arXiv:quant-ph/0207088]",
            fontsize=9,
            color="#475569",
            family="monospace",
        )
        # Column headers + one row per fit in a simple fixed-grid layout.
        header_y = 0.29
        fig.text(0.08, header_y, "Backend", fontsize=9.5, weight="bold", color="#475569")
        fig.text(0.26, header_y, "Basis", fontsize=9.5, weight="bold", color="#475569")
        fig.text(0.33, header_y, "p_th", fontsize=9.5, weight="bold", color="#475569")
        fig.text(0.48, header_y, "nu", fontsize=9.5, weight="bold", color="#475569")
        fig.text(0.60, header_y, "n", fontsize=9.5, weight="bold", color="#475569")
        fig.text(0.66, header_y, "fit window (p)", fontsize=9.5, weight="bold", color="#475569")
        row_y = header_y - 0.025
        for backend, basis, fss in fss_rows:
            fig.text(0.08, row_y, backend, fontsize=10, color="#0f172a")
            fig.text(0.26, row_y, basis, fontsize=10, color="#0f172a")
            fig.text(
                0.33,
                row_y,
                f"{fss.p_th:.5g} ± {fss.p_th_std_error:.2g}",
                fontsize=10,
                color="#0f172a",
                family="monospace",
            )
            fig.text(
                0.48,
                row_y,
                f"{fss.nu:.3g} ± {fss.nu_std_error:.2g}",
                fontsize=10,
                color="#0f172a",
                family="monospace",
            )
            fig.text(0.60, row_y, str(fss.num_points), fontsize=10, color="#0f172a", family="monospace")
            fig.text(
                0.66,
                row_y,
                f"[{fss.fit_window_low:.4g}, {fss.fit_window_high:.4g}]",
                fontsize=10,
                color="#0f172a",
                family="monospace",
            )
            row_y -= 0.025

    return fig


def _collect_fss_fit_rows(
    summaries: list[FitSummary],
) -> list[tuple[str, str, FSSThresholdFitSummary]]:
    """Return one ``(backend, basis, fit)`` triple per (backend, basis) that has an FSS fit."""
    rows: list[tuple[str, str, FSSThresholdFitSummary]] = []
    for backend in sorted({summary.backend for summary in summaries}):
        for basis in sorted({summary.basis for summary in summaries if summary.backend == backend}):
            basis_summaries = [
                summary for summary in summaries if summary.backend == backend and summary.basis == basis
            ]
            fit = _fit_fss_threshold(basis_summaries)
            if fit is not None:
                rows.append((backend, basis, fit))
    return rows


def _section_accent(*, x_left: float, x_right: float, y: float) -> Rectangle:
    """Return the small blue accent bar drawn under an appendix column heading."""
    from matplotlib.patches import Rectangle as _Rectangle

    return _Rectangle((x_left, y), x_right - x_left, 0.003, facecolor="#2563eb", edgecolor="none")


def _build_section_divider_figure(
    title: str,
    subtitle: str | None = None,
) -> Figure:
    """Build a minimal section-title page -- centered title + optional subtitle + accent bar."""
    from matplotlib.patches import Rectangle

    plt = _require_matplotlib_pyplot()
    fig = plt.figure(figsize=_REPORT_PAGE_SIZE, facecolor="#f8fafc")
    page = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    page.set_xlim(0, 1)
    page.set_ylim(0, 1)
    page.axis("off")
    page.patch.set_facecolor("#f8fafc")

    page.add_patch(
        Rectangle(
            (0.25, 0.555),
            0.5,
            0.004,
            facecolor="#2563eb",
            edgecolor="none",
            transform=page.transAxes,
        ),
    )
    fig.text(0.5, 0.60, title, ha="center", va="center", fontsize=34, weight="bold", color="#0f172a")
    if subtitle:
        fig.text(0.5, 0.50, subtitle, ha="center", va="center", fontsize=14, color="#475569")
    return fig


def write_pdf_report(
    output_path: Path,
    *,
    points: list[SweepPoint],
    summaries: list[FitSummary],
    timing_summary: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    title: str = "PECOS Surface Sweep Report",
) -> Path:
    """Write a multi-page PDF report (cover + every plot) to ``output_path``.

    The cover page lists configuration and timing; subsequent pages contain
    the same plots the dashboard embeds, in the same order. Returns the
    output path on success.
    """
    plt = _require_matplotlib_pyplot()
    from matplotlib.backends.backend_pdf import PdfPages

    backends = sorted({summary.backend for summary in summaries})

    def _save_and_close(fig: Figure) -> None:
        pdf.savefig(fig)
        plt.close(fig)

    def _write_divider(section_title: str, subtitle: str | None = None) -> None:
        _save_and_close(_build_section_divider_figure(section_title, subtitle))

    with PdfPages(output_path) as pdf:
        _save_and_close(
            _build_report_cover_figure(
                config=config,
                summaries=summaries,
                title=title,
            ),
        )

        _write_divider("Combined Overlays", "Per-round logical error rate versus physical error rate")
        for backend in backends:
            backend_summaries = [summary for summary in summaries if summary.backend == backend]
            _save_and_close(
                _build_per_round_overlay_figure(
                    summaries=backend_summaries,
                    backend=backend,
                    figsize=_REPORT_PAGE_SIZE,
                ),
            )

        _write_divider("Fixed-p Duration Overlays", "Logical memory error versus memory duration")
        for backend in backends:
            backend_summaries = [summary for summary in summaries if summary.backend == backend]
            for physical_error_rate in sorted(
                {point.physical_error_rate for point in points if point.backend == backend},
            ):
                rate_points = [
                    point
                    for point in points
                    if point.backend == backend and point.physical_error_rate == physical_error_rate
                ]
                rate_summaries = [
                    summary for summary in backend_summaries if summary.physical_error_rate == physical_error_rate
                ]
                _save_and_close(
                    _build_duration_overlay_figure(
                        points=rate_points,
                        summaries=rate_summaries,
                        backend=backend,
                        physical_error_rate=physical_error_rate,
                        figsize=_REPORT_PAGE_SIZE,
                    ),
                )

        _write_divider("Per-basis Curves", "Fitted logical error versus physical error rate")
        for backend in backends:
            for basis in sorted({summary.basis for summary in summaries if summary.backend == backend}):
                basis_summaries = [
                    summary for summary in summaries if summary.backend == backend and summary.basis == basis
                ]
                plot_specs = [
                    (
                        "fitted_projected_logical_error_rate_over_d_rounds",
                        f"{basis}-Basis Fitted Logical Error Rate Over d Rounds ({backend})",
                        "Fitted logical error rate over d rounds",
                    ),
                    (
                        "fitted_logical_error_rate_per_round",
                        f"{basis}-Basis Fitted Logical Error Rate Per Round ({backend})",
                        "Fitted logical error rate per round",
                    ),
                ]
                for metric, plot_title, y_label in plot_specs:
                    _save_and_close(
                        _build_plot_figure(
                            summaries=basis_summaries,
                            metric=metric,
                            title=plot_title,
                            y_label=y_label,
                            figsize=_REPORT_PAGE_SIZE,
                        ),
                    )

        _write_divider("Appendix", "Methods and timing details")
        _save_and_close(
            _build_appendix_figure(config=config, timing_summary=timing_summary, summaries=summaries),
        )

    return output_path


def _write_artifacts(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    points: list[SweepPoint],
    summaries: list[FitSummary],
    point_timings: list[dict[str, Any]],
    timing_summary: dict[str, Any],
) -> None:
    """Write any optional JSON or plot artifacts requested by the user."""
    prefix = args.output_prefix
    json_filename: str | None = None
    if args.save_json:
        json_path = output_dir / f"{prefix}_results.json"
        _write_json_results(
            json_path,
            args=args,
            points=points,
            summaries=summaries,
            point_timings=point_timings,
            timing_summary=timing_summary,
        )
        print(f"Wrote JSON results to {json_path}")
        json_filename = json_path.name

    formats: list[str] = []
    if args.save_svg:
        formats.append("svg")
    if args.save_pdf:
        formats.append("pdf")

    dashboard_plots = render_plot_artifacts(
        output_dir,
        prefix=prefix,
        points=points,
        summaries=summaries,
        formats=formats,
    )

    if args.save_html:
        html_path = output_dir / f"{prefix}_dashboard.html"
        _write_html_dashboard(
            html_path,
            args=args,
            summaries=summaries,
            timing_summary=timing_summary,
            plots=dashboard_plots,
            json_filename=json_filename,
        )
        print(f"Wrote HTML dashboard to {html_path}")
        if args.open_html:
            _maybe_open_html_dashboard(html_path)
            print(f"Opened HTML dashboard at {html_path}")

    if args.save_report_pdf:
        report_path = output_dir / f"{prefix}_report.pdf"
        write_pdf_report(
            report_path,
            points=points,
            summaries=summaries,
            timing_summary=timing_summary,
            config=_config_for_report(args),
        )
        print(f"Wrote PDF report to {report_path}")


def _config_for_report(args: argparse.Namespace) -> dict[str, Any]:
    """Build the ``config`` dict the PDF report cover + appendix pages expect from CLI args."""
    return {
        "distances": sorted(set(args.distances)),
        "error_rates": sorted(set(args.error_rates)),
        "shots": args.shots,
        "dem_mode": getattr(args, "dem_mode", None),
        "duration_schedule_description": getattr(args, "duration_schedule_description", None),
        "duration_rounds_by_distance": dict(getattr(args, "duration_rounds_by_distance", {})),
        # Match the key name that ``_write_json_results`` serializes so the
        # appendix page can read the same field from either source.
        "sample_backend_mode": getattr(args, "sample_backend", None),
        "native_circuit_source": getattr(args, "native_circuit_source", None),
        "decoder": getattr(args, "decoder", ["pymatching"]),
        "noise_model": _noise_model_description(args),
        "seed": getattr(args, "seed", None),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9], help="Odd code distances to sweep.")
    parser.add_argument(
        "--duration-multipliers",
        "--round-multipliers",
        dest="duration_multipliers",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit duration multipliers to use for the fit, where r = multiplier * distance. "
            "When omitted, the sweep uses about four evenly spaced integer round counts "
            "across the default [2d, 3d] window."
        ),
    )
    parser.add_argument(
        "--duration-min-multiplier",
        type=float,
        default=2.0,
        help="Lower bound of the default duration window, in units of distance.",
    )
    parser.add_argument(
        "--duration-max-multiplier",
        type=float,
        default=3.0,
        help="Upper bound of the default duration window, in units of distance.",
    )
    parser.add_argument(
        "--duration-num-points",
        type=int,
        default=4,
        help=(
            "Number of approximately evenly spaced integer round counts to sample within the "
            "default duration window when --duration-multipliers is not provided."
        ),
    )
    parser.add_argument(
        "--p1-scale",
        type=float,
        default=1.0 / 30.0,
        help=(
            "Scale factor for single-qubit gate error rate relative to p. p1 = p * p1_scale. Default: 1/30 (~0.033)."
        ),
    )
    parser.add_argument(
        "--p-meas-scale",
        type=float,
        default=1.0 / 3.0,
        help="Scale factor for measurement error rate. p_meas = p * p_meas_scale. Default: 1/3.",
    )
    parser.add_argument(
        "--p-prep-scale",
        type=float,
        default=1.0 / 3.0,
        help="Scale factor for preparation error rate. p_prep = p * p_prep_scale. Default: 1/3.",
    )
    parser.add_argument(
        "--error-rates",
        nargs="+",
        type=float,
        default=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        help="Uniform physical error rates p to sweep.",
    )
    parser.add_argument("--bases", nargs="+", default=["X", "Z"], help="Memory bases to sweep.")
    parser.add_argument("--shots", type=int, default=200, help="Shots per (distance, basis, p, rounds) point.")
    parser.add_argument(
        "--sample-backend",
        choices=[
            "sim",
            "selene_sim",
            "selene_stabilizer_plugin",
            "native_sampler",
            "compare",
            "compare_gate_backends",
            "compare_all",
            "profile_gate_backends",
        ],
        default="sim",
        help=(
            "Sampling backend. 'sim' uses sim(Guppy(...)).classical(selene_engine()), "
            "'selene_sim' uses direct selene_sim execution with Selene Stim, "
            "'selene_stabilizer_plugin' uses direct selene_sim execution with the PECOS Selene StabilizerPlugin, "
            "'native_sampler' uses the PECOS native DEM sampler, "
            "'compare' runs sim + native_sampler, "
            "'compare_gate_backends' runs selene_sim + selene_stabilizer_plugin + sim, "
            "'compare_all' runs selene_sim + selene_stabilizer_plugin + sim + native_sampler, "
            "and 'profile_gate_backends' reports timing breakdowns for selene_sim + "
            "selene_stabilizer_plugin + sim without decoding."
        ),
    )
    parser.add_argument(
        "--native-circuit-source",
        choices=["abstract", "traced_qis"],
        default="traced_qis",
        help=(
            "Which ideal circuit the native PECOS DEM/sampler path should analyze. "
            "'traced_qis' (default) traces the lowered ideal Selene/QIS gate stream "
            "(decomposed into native gates like RZZ+rotations), matching the actual "
            "hardware gate set. Use this for hardware-realistic threshold estimation. "
            "'abstract' uses the high-level surface TickCircuit with CX/H gates, "
            "matching the standard circuit-level noise model from the QEC literature."
        ),
    )
    parser.add_argument(
        "--dem-mode",
        choices=["native_decomposed", "native_full"],
        default="native_decomposed",
        help="PECOS native DEM mode. PyMatching typically wants native_decomposed.",
    )
    parser.add_argument(
        "--decoder",
        nargs="+",
        choices=["pymatching", "tesseract", "bp_osd", "bp_lsd", "union_find", "relay_bp", "min_sum_bp"],
        default=["pymatching"],
        help=(
            "Decoder(s) for circuit-level DEM decoding. Specify multiple to "
            "compare them side-by-side in plots and reports. Default: pymatching. "
            "Check-matrix decoders (bp_osd, bp_lsd, union_find, relay_bp, min_sum_bp) "
            "extract a check matrix from the DEM automatically."
        ),
    )
    parser.add_argument(
        "--tesseract-beam",
        type=int,
        default=5,
        help=(
            "Tesseract det_beam parameter (number of detectors to consider in beam search). "
            "Default: 5 (matches upstream). With BFS orderings, det_beam=5 gives identical "
            "accuracy to 50 or 100 at d<=5 while being 10x faster."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed for the runtime noise model.")
    parser.add_argument("--save-json", action="store_true", help="Write a JSON artifact with all sweep results.")
    parser.add_argument("--save-svg", action="store_true", help="Write SVG plots for each basis and fitted metric.")
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Write an HTML dashboard that links the generated SVG plots. Implies --save-svg.",
    )
    parser.add_argument(
        "--open-html",
        action="store_true",
        help="Open the generated HTML dashboard after the run. Implies --save-html and --save-svg.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Write PDF plots for each basis and fitted metric. Requires matplotlib.",
    )
    parser.add_argument(
        "--save-report-pdf",
        action="store_true",
        help=(
            "Write a single multi-page PDF report (cover page with config + timing, "
            "then one plot per page). Requires matplotlib."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for optional artifacts. Defaults to a temporary directory outside the repo.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="surface_threshold_sweep",
        help="Filename prefix for optional artifacts.",
    )
    parser.add_argument(
        "--refine-threshold",
        action="store_true",
        help=(
            "After the initial sweep, estimate the threshold and automatically "
            "run a refined sweep with tighter error-rate spacing around it. "
            "The refinement uses the same distances, bases, and shots."
        ),
    )
    parser.add_argument(
        "--refine-window",
        type=float,
        default=0.5,
        help=(
            "Half-width of the refinement window as a fraction of the estimated "
            "threshold. E.g., 0.5 means sweep from 0.5*p_th to 1.5*p_th. "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--refine-points",
        type=int,
        default=6,
        help="Number of error-rate points in the refinement sweep. Default: 6.",
    )
    parser.add_argument(
        "--ancilla-budget",
        type=int,
        default=None,
        help=(
            "Optional cap on simultaneously live ancilla qubits. When set, the "
            "circuit builder batches stabilizer measurements to stay within this "
            "budget. Affects both the abstract and traced_qis circuit sources."
        ),
    )
    parser.add_argument(
        "--benchmark-repetitions",
        type=int,
        default=3,
        help="Timed repetitions for 'profile_gate_backends'.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="Warmup repetitions before timed runs for 'profile_gate_backends'.",
    )
    return parser.parse_args()


_BACKEND_MODE_EXPANSIONS: dict[str, list[str]] = {
    "compare": ["sim", "native_sampler"],
    "compare_gate_backends": ["selene_sim", "selene_stabilizer_plugin", "sim"],
    "compare_all": ["selene_sim", "selene_stabilizer_plugin", "sim", "native_sampler"],
    "profile_gate_backends": ["selene_sim", "selene_stabilizer_plugin", "sim"],
}


def _resolve_backends(sample_backend: str, decoders: list[str] | None = None) -> list[str]:
    """Resolve ``--sample-backend`` and ``--decoder`` to the concrete list of backends to run.

    When multiple decoders are given, each base backend is expanded to
    ``backend:decoder`` pairs so the plotting infrastructure sees them as
    separate series.  With a single decoder the backend name is unchanged
    for backwards compatibility.
    """
    base = _BACKEND_MODE_EXPANSIONS.get(sample_backend, [sample_backend])
    if decoders is None or len(decoders) <= 1:
        return base
    return [f"{b}:{d}" for b in base for d in decoders]


def _resolve_duration_schedule(
    args: argparse.Namespace,
    distances: list[int],
) -> tuple[list[float], dict[int, tuple[int, ...]], str]:
    """Return (multipliers, rounds-by-distance, human-readable description)."""
    explicit_multipliers = None if args.duration_multipliers is None else sorted(set(args.duration_multipliers))
    if explicit_multipliers is not None:
        multipliers = explicit_multipliers
        description = (
            "explicit multipliers: "
            + ", ".join(f"{value:g}" for value in multipliers)
            + " (meaning r = multiplier * distance)"
        )
    else:
        multipliers = _evenly_spaced_values(
            args.duration_min_multiplier,
            args.duration_max_multiplier,
            args.duration_num_points,
        )
        description = (
            f"about {args.duration_num_points} evenly spaced round counts over "
            f"[{args.duration_min_multiplier:g}d, {args.duration_max_multiplier:g}d]"
        )
    rounds_by_distance = {
        distance: _duration_rounds_for_distance(
            distance,
            explicit_multipliers=explicit_multipliers,
            duration_min_multiplier=args.duration_min_multiplier,
            duration_max_multiplier=args.duration_max_multiplier,
            duration_num_points=args.duration_num_points,
        )
        for distance in distances
    }
    return multipliers, rounds_by_distance, description


def _validate_sweep_inputs(
    distances: list[int],
    duration_multipliers: list[float],
    args: argparse.Namespace,
) -> None:
    """Raise ``ValueError`` on any invalid sweep configuration input."""
    if any(distance <= 0 or distance % 2 == 0 for distance in distances):
        msg = "Distances must be positive odd integers"
        raise ValueError(msg)
    if any(multiplier <= 0 for multiplier in duration_multipliers):
        msg = "Duration multipliers must be positive"
        raise ValueError(msg)
    if args.duration_min_multiplier <= 0.0 or args.duration_max_multiplier <= 0.0:
        msg = "Duration window multipliers must be positive"
        raise ValueError(msg)
    if args.duration_max_multiplier < args.duration_min_multiplier:
        msg = "duration-max-multiplier must be at least duration-min-multiplier"
        raise ValueError(msg)
    if args.duration_num_points <= 0:
        msg = "duration-num-points must be positive"
        raise ValueError(msg)


def _print_config_banner(
    args: argparse.Namespace,
    *,
    backends: list[str],
    distances: list[int],
    bases: list[str],
    error_rates: list[float],
    output_dir: Path | None,
    duration_schedule_description: str,
    duration_rounds_by_distance: dict[int, tuple[int, ...]],
) -> None:
    """Print the sweep configuration summary at the top of a run."""
    print("Native PECOS Surface Threshold Sweep")
    print("=" * 40)
    print(f"distances        : {distances}")
    print(f"bases            : {bases}")
    print(f"duration schedule: {duration_schedule_description}")
    print(
        "effective rounds : "
        + "; ".join(
            f"d={distance} -> {list(rounds)}" for distance, rounds in sorted(duration_rounds_by_distance.items())
        ),
    )
    print(f"error rates      : {error_rates}")
    print(f"shots / point    : {args.shots}")
    print(f"sample backend mode: {args.sample_backend}")
    print(f"executed backends: {backends}")
    print(f"DEM mode         : {args.dem_mode}")
    print(f"native circuit source: {args.native_circuit_source}")
    decoders = getattr(args, "decoder", ["pymatching"])
    print(f"decoder(s)       : {', '.join(decoders)} via SurfaceDecoder(native PECOS DEM)")
    for backend in backends:
        print(f"runtime[{backend}]  : {_backend_runtime_label(backend, args.native_circuit_source)}")
    p1s = getattr(args, "p1_scale", 0.1)
    pms = getattr(args, "p_meas_scale", 0.5)
    pps = getattr(args, "p_prep_scale", 0.5)
    print(f"noise model      : depolarizing with p1={p1s}*p, p2=p, p_meas={pms}*p, p_prep={pps}*p")
    print("fit model        : p_L(r) = 0.5 * (1 - (1 - 2 * epsilon) ** r)")
    if output_dir is not None:
        print(f"artifact dir     : {output_dir}")


def _parse_backend_decoder(backend: str, args: argparse.Namespace) -> tuple[str, str]:
    """Split ``backend:decoder`` into base backend and decoder type.

    When a single decoder is configured the backend string has no colon
    and the decoder comes from ``args.decoder[0]``.
    """
    if ":" in backend:
        base, decoder = backend.split(":", 1)
        return base, decoder
    decoders = getattr(args, "decoder", ["pymatching"])
    return backend, decoders[0] if isinstance(decoders, list) else decoders


def _run_one_memory_point(
    args: argparse.Namespace,
    *,
    backend: str,
    basis: str,
    distance: int,
    physical_error_rate: float,
    total_rounds: int,
    seed: int,
) -> tuple[SweepPoint, dict[str, Any]]:
    """Run one sampling point, print the per-point line, return the point + its timing row."""
    base_backend, decoder_type = _parse_backend_decoder(backend, args)
    point_start = time.perf_counter()
    point = _run_memory_point(
        sample_backend=base_backend,
        distance=distance,
        basis=basis,
        physical_error_rate=physical_error_rate,
        total_rounds=total_rounds,
        num_shots=args.shots,
        dem_mode=args.dem_mode,
        native_circuit_source=args.native_circuit_source,
        seed=seed,
        decoder_type=decoder_type,
        backend_label=backend,
        ancilla_budget=getattr(args, "ancilla_budget", None),
        p1_scale=getattr(args, "p1_scale", 0.1),
        p_meas_scale=getattr(args, "p_meas_scale", 0.5),
        p_prep_scale=getattr(args, "p_prep_scale", 0.5),
    )
    elapsed_seconds = time.perf_counter() - point_start
    naive_per_round = ler_per_round_exp(point.logical_error_rate, point.total_rounds)
    print(
        "    "
        f"LER={point.logical_error_rate:.6e} "
        f"raw={_format_rate(point.raw_error_rate)} "
        f"naive_per_round={naive_per_round:.6e} "
        f"elapsed={elapsed_seconds:.3f}s",
    )
    timing_row = {
        "backend": backend,
        "basis": basis,
        "distance": distance,
        "physical_error_rate": physical_error_rate,
        "total_rounds": total_rounds,
        "num_shots": args.shots,
        "elapsed_seconds": elapsed_seconds,
    }
    return point, timing_row


def _fit_and_print_group(
    group_points: list[SweepPoint],
    backend: str,
) -> FitSummary:
    """Fit one ``(backend, basis, d, p)`` group and print the fit line + any warning."""
    fit_summary = _fit_summary_from_points(group_points)
    observed = ", ".join(
        f"r={round_value}:{logical_rate:.3e}"
        for round_value, logical_rate in zip(
            fit_summary.round_values,
            fit_summary.observed_logical_error_rates,
            strict=True,
        )
    )
    epsilon_interval = _format_interval(
        fit_summary.fitted_logical_error_rate_per_round_ci_low,
        fit_summary.fitted_logical_error_rate_per_round_ci_high,
        fit_summary.fitted_logical_error_rate_per_round,
    )
    proj_interval = _format_interval(
        fit_summary.fitted_projected_logical_error_rate_over_d_rounds_ci_low,
        fit_summary.fitted_projected_logical_error_rate_over_d_rounds_ci_high,
        fit_summary.fitted_projected_logical_error_rate_over_d_rounds,
    )
    print(
        "    "
        f"[{backend}] "
        f"fit_epsilon={fit_summary.fitted_logical_error_rate_per_round:.6e} {epsilon_interval} "
        f"fit_proj_d={fit_summary.fitted_projected_logical_error_rate_over_d_rounds:.6e} {proj_interval} "
        f"fit_rms={fit_summary.fit_root_mean_square_error:.3e} "
        f"[{observed}]",
    )
    warning_text = _fit_rms_warning_text(fit_summary)
    if warning_text:
        print(f"    [{backend}] {warning_text}")
    return fit_summary


def _print_cross_backend_deltas(
    group_fit_summaries: dict[str, FitSummary],
    backends: list[str],
) -> None:
    """Print delta lines across backends in one ``(basis, d, p)`` group."""
    if "selene_sim" in group_fit_summaries:
        ref_summary = group_fit_summaries["selene_sim"]
        for backend in backends:
            if backend == "selene_sim":
                continue
            summary = group_fit_summaries[backend]
            delta_epsilon = (
                summary.fitted_logical_error_rate_per_round - ref_summary.fitted_logical_error_rate_per_round
            )
            delta_proj_d = (
                summary.fitted_projected_logical_error_rate_over_d_rounds
                - ref_summary.fitted_projected_logical_error_rate_over_d_rounds
            )
            print(
                "    "
                f"compare_vs_selene_sim[{backend}] "
                f"delta_epsilon={delta_epsilon:+.3e} "
                f"delta_proj_d={delta_proj_d:+.3e}",
            )
    elif len(backends) == 2 and "sim" in group_fit_summaries and "native_sampler" in group_fit_summaries:
        sim_summary = group_fit_summaries["sim"]
        sampler_summary = group_fit_summaries["native_sampler"]
        delta_epsilon = (
            sampler_summary.fitted_logical_error_rate_per_round - sim_summary.fitted_logical_error_rate_per_round
        )
        delta_proj_d = (
            sampler_summary.fitted_projected_logical_error_rate_over_d_rounds
            - sim_summary.fitted_projected_logical_error_rate_over_d_rounds
        )
        print(f"    compare delta_epsilon={delta_epsilon:+.3e} delta_proj_d={delta_proj_d:+.3e}")


def _run_sweep_and_fit(
    args: argparse.Namespace,
    *,
    backends: list[str],
    distances: list[int],
    bases: list[str],
    error_rates: list[float],
    duration_rounds_by_distance: dict[int, tuple[int, ...]],
) -> tuple[list[SweepPoint], list[FitSummary], list[dict[str, Any]]]:
    """Run the full sweep, fit each ``(basis, d, p)`` group, return (points, fits, timings)."""
    all_points: list[SweepPoint] = []
    fit_summaries: list[FitSummary] = []
    point_timings: list[dict[str, Any]] = []

    total_points = (
        sum(len(duration_rounds_by_distance[distance]) for distance in distances)
        * len(bases)
        * len(error_rates)
        * len(backends)
    )
    point_idx = 0

    for basis in bases:
        for distance in distances:
            for physical_error_rate in error_rates:
                for total_rounds in duration_rounds_by_distance[distance]:
                    for backend in backends:
                        point_idx += 1
                        print(
                            f"[{point_idx:>3}/{total_points}] "
                            f"backend={backend} basis={basis} d={distance} "
                            f"p={physical_error_rate:.5g} r={total_rounds} ...",
                        )
                        point, timing_row = _run_one_memory_point(
                            args,
                            backend=backend,
                            basis=basis,
                            distance=distance,
                            physical_error_rate=physical_error_rate,
                            total_rounds=total_rounds,
                            seed=args.seed + point_idx,
                        )
                        all_points.append(point)
                        point_timings.append(timing_row)

                group_fit_summaries: dict[str, FitSummary] = {}
                for backend in backends:
                    group_points = [
                        point
                        for point in all_points
                        if point.backend == backend
                        and point.basis == basis
                        and point.distance == distance
                        and point.physical_error_rate == physical_error_rate
                    ]
                    fit_summary = _fit_and_print_group(group_points, backend)
                    fit_summaries.append(fit_summary)
                    group_fit_summaries[backend] = fit_summary

                _print_cross_backend_deltas(group_fit_summaries, backends)

    return all_points, fit_summaries, point_timings


def _print_post_sweep_analysis(
    *,
    backends: list[str],
    bases: list[str],
    distances: list[int],
    fit_summaries: list[FitSummary],
) -> None:
    """Print all per-basis tables, scaling fits, and threshold summaries."""
    for backend in backends:
        for basis in bases:
            basis_summaries = [
                summary for summary in fit_summaries if summary.backend == backend and summary.basis == basis
            ]
            _print_basis_table(
                basis_summaries,
                metric="fitted_projected_logical_error_rate_over_d_rounds",
                title=f"{basis}-Basis Fitted Logical Error Rate Over d Rounds ({backend})",
            )
            _print_basis_table(
                basis_summaries,
                metric="fitted_logical_error_rate_per_round",
                title=f"{basis}-Basis Fitted Logical Error Rate Per Round ({backend})",
            )

            # Restrict to points at p <= estimated threshold so the power-law
            # fit reflects the below-threshold regime where ``eps ~ A * p^c``
            # actually holds. Above threshold, curves bend and the fitted
            # exponent falls away from the theoretical ``(d + 1) / 2``.
            below_threshold_cut = _estimate_threshold(basis_summaries)
            power_law_fits = _fit_per_distance_power_law(
                basis_summaries,
                max_physical_error_rate=below_threshold_cut,
            ) or _fit_per_distance_power_law(basis_summaries)
            if power_law_fits:
                print()
                cut_note = (
                    f" (fit restricted to p<={below_threshold_cut:.4g})" if below_threshold_cut is not None else ""
                )
                print(
                    f"{basis} basis [{backend}] primary epsilon_d(p) ~= A_d * p^c_d fits{cut_note}:",
                )
                for fit in power_law_fits:
                    se_text = f" ±{fit.fitted_exponent_std_error:.3f}" if fit.fitted_exponent_std_error > 0.0 else ""
                    print(
                        "  "
                        f"d={fit.distance}: A_d={fit.fitted_prefactor:.4g} "
                        f"c_d={fit.fitted_exponent:.3f}{se_text} "
                        f"theory=(d+1)/2={fit.expected_distance_scaling_exponent:.1f} "
                        f"log_rmse={fit.fit_root_mean_square_log_error:.3e} "
                        f"n={len(fit.physical_error_rates)}",
                    )

            lambda_ratios = _pairwise_lambda_ratios(basis_summaries)
            if lambda_ratios:
                print(f"{basis} basis [{backend}] primary Lambda_(d/(d+2)) ratios:")
                for ratio in lambda_ratios:
                    print(
                        "  "
                        f"p={ratio.physical_error_rate:.5g}: "
                        f"Lambda_{{{ratio.distance_low}/{ratio.distance_high}}}="
                        f"{ratio.lambda_d_over_d_plus_2:.4g}",
                    )

            print(f"{basis} basis [{backend}] suppression check (fitted d-round LER decreases with distance):")
            for p, is_suppressed in _suppression_summary(basis_summaries):
                status = "suppressed" if is_suppressed else "not suppressed"
                print(f"  p={p:.5g}: {status}")

            distance_scaling_fits = _distance_scaling_fits(basis_summaries)
            if distance_scaling_fits:
                print(f"{basis} basis [{backend}] background fixed-p distance-scaling fits:")
                for fit in distance_scaling_fits:
                    print(
                        "  "
                        f"p={fit.physical_error_rate:.5g}: "
                        f"A={fit.fitted_prefactor:.4g} "
                        f"Lambda_(d/(d+2))={fit.fitted_suppression_factor:.4g} "
                        f"log_rmse={fit.fit_root_mean_square_log_error:.3e}",
                    )

            crossing_per_round = _estimate_threshold(
                basis_summaries,
                metric="fitted_logical_error_rate_per_round",
            )
            crossing_d_rounds = _estimate_threshold(
                basis_summaries,
                metric="fitted_projected_logical_error_rate_over_d_rounds",
            )
            global_scaling_fit = _fit_global_scaling_law(basis_summaries)
            # Try FSS fit seeded from both crossings; prefer d-round seed
            fss_seed = crossing_d_rounds or crossing_per_round
            fss_fit = _fit_fss_threshold(basis_summaries, seed_threshold=fss_seed)
            if (
                crossing_per_round is not None
                or crossing_d_rounds is not None
                or global_scaling_fit is not None
                or fss_fit is not None
            ):
                print(f"{basis} basis [{backend}] background threshold-style summary:")
                if crossing_per_round is None and crossing_d_rounds is None:
                    print(f"  no d={min(distances)} vs d={max(distances)} crossing was detected on this sweep.")
                if crossing_per_round is not None:
                    print(f"  per-round epsilon crossing: p ~= {crossing_per_round:.6g}")
                if crossing_d_rounds is not None:
                    print(f"  projected d-round crossing: p ~= {crossing_d_rounds:.6g}")
                if global_scaling_fit is not None:
                    print(
                        "  "
                        f"global ansatz epsilon ~= A * (p / p_th)^((d + 1) / 2): "
                        f"A={global_scaling_fit.fitted_prefactor:.4g} "
                        f"p_th={global_scaling_fit.fitted_threshold:.4g} "
                        f"log_rmse={global_scaling_fit.fit_root_mean_square_log_error:.3e}",
                    )
                if fss_fit is not None:
                    print(
                        "  "
                        f"FSS fit p_L = a + b*x + c*x^2, x = (p - p_th) * d^(1/nu) "
                        f"[Wang-Harrington-Preskill, arXiv:quant-ph/0207088; "
                        f"window {fss_fit.fit_window_low:.4g} <= p <= {fss_fit.fit_window_high:.4g}, "
                        f"n={fss_fit.num_points}]:",
                    )
                    print(
                        "    "
                        f"p_th = {fss_fit.p_th:.5g} ± {fss_fit.p_th_std_error:.3g}    "
                        f"nu = {fss_fit.nu:.4g} ± {fss_fit.nu_std_error:.3g}",
                    )


def main() -> int:
    """Run the threshold sweep CLI and optionally write summary artifacts."""
    args = _parse_args()
    if args.open_html:
        args.save_html = True
    if args.save_html:
        args.save_svg = True

    wants_outputs = args.save_json or args.save_svg or args.save_pdf or args.save_html or args.save_report_pdf
    output_dir = _resolve_output_dir(args.output_dir, wants_outputs=wants_outputs)
    sweep_start = time.perf_counter()

    distances = sorted(set(args.distances))
    bases = [basis.upper() for basis in args.bases]
    backends = _resolve_backends(args.sample_backend, args.decoder)
    duration_multipliers, duration_rounds_by_distance, duration_schedule_description = _resolve_duration_schedule(
        args,
        distances,
    )
    error_rates = sorted(set(args.error_rates))

    _validate_sweep_inputs(distances, duration_multipliers, args)

    args.duration_multipliers = duration_multipliers
    args.duration_rounds_by_distance = duration_rounds_by_distance
    args.duration_schedule_description = duration_schedule_description

    _print_config_banner(
        args,
        backends=backends,
        distances=distances,
        bases=bases,
        error_rates=error_rates,
        output_dir=output_dir,
        duration_schedule_description=duration_schedule_description,
        duration_rounds_by_distance=duration_rounds_by_distance,
    )

    if args.sample_backend == "profile_gate_backends":
        _profile_gate_backends(
            backends=backends,
            distances=distances,
            bases=bases,
            error_rates=error_rates,
            duration_rounds_by_distance=duration_rounds_by_distance,
            shots=args.shots,
            seed=args.seed,
            warmup_repetitions=args.benchmark_warmup,
            benchmark_repetitions=args.benchmark_repetitions,
        )
        return 0

    all_points, fit_summaries, point_timings = _run_sweep_and_fit(
        args,
        backends=backends,
        distances=distances,
        bases=bases,
        error_rates=error_rates,
        duration_rounds_by_distance=duration_rounds_by_distance,
    )

    _print_post_sweep_analysis(
        backends=backends,
        bases=bases,
        distances=distances,
        fit_summaries=fit_summaries,
    )

    # --- Adaptive threshold refinement ---
    if args.refine_threshold:
        # Estimate threshold from the initial sweep
        threshold_estimates = []
        for basis in bases:
            basis_summaries = [s for s in fit_summaries if s.basis == basis]
            for backend in backends:
                backend_summaries = [s for s in basis_summaries if s.backend == backend]
                if not backend_summaries:
                    continue
                # Use d-round crossing as initial estimate (more conservative)
                crossing = _estimate_threshold(
                    backend_summaries,
                    metric="fitted_projected_logical_error_rate_over_d_rounds",
                )
                if crossing is None:
                    # Fall back to per-round crossing
                    crossing = _estimate_threshold(backend_summaries)
                if crossing is not None:
                    threshold_estimates.append((backend, basis, crossing))

        if threshold_estimates:
            # Use the median estimate across all backends/bases
            median_th = sorted(t[2] for t in threshold_estimates)[len(threshold_estimates) // 2]
            half_w = args.refine_window
            p_low = median_th * (1.0 - half_w)
            p_high = median_th * (1.0 + half_w)
            n_pts = args.refine_points
            import numpy as np

            refined_rates = sorted({float(f"{r:.6g}") for r in np.linspace(p_low, p_high, n_pts)})
            # Exclude rates already in the initial sweep
            refined_rates = [r for r in refined_rates if r not in error_rates and r > 0]

            if refined_rates:
                print()
                print(
                    f"=== Threshold refinement: {len(refined_rates)} additional points "
                    f"in [{p_low:.5g}, {p_high:.5g}] around estimate p_th ~= {median_th:.5g} ===",
                )
                refine_points, refine_fits, refine_timings = _run_sweep_and_fit(
                    args,
                    backends=backends,
                    distances=distances,
                    bases=bases,
                    error_rates=refined_rates,
                    duration_rounds_by_distance=duration_rounds_by_distance,
                )
                # Merge with initial results
                all_points.extend(refine_points)
                fit_summaries.extend(refine_fits)
                point_timings.extend(refine_timings)

                # Re-run analysis with merged data
                print()
                print("=== Combined analysis (initial + refinement) ===")
                _print_post_sweep_analysis(
                    backends=backends,
                    bases=bases,
                    distances=distances,
                    fit_summaries=fit_summaries,
                )
        else:
            print("\n  No threshold detected -- skipping refinement.")

    timing_summary = _timing_summary(
        point_timings,
        total_wall_clock_seconds=time.perf_counter() - sweep_start,
    )
    _print_timing_summary(timing_summary)

    if output_dir is not None:
        _write_artifacts(
            output_dir,
            args=args,
            points=all_points,
            summaries=fit_summaries,
            point_timings=point_timings,
            timing_summary=timing_summary,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
