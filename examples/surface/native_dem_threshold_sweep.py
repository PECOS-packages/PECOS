#!/usr/bin/env python3
r"""Surface-code X/Z memory threshold sweep with native PECOS DEMs.

This example runs rotated surface-code memory experiments using:

- Guppy surface-memory programs from ``pecos.guppy.surface.make_surface_code``
- ``sim(...).classical(selene_engine())`` for end-to-end execution
- direct ``selene_sim`` execution with either Selene ``Stim`` or the PECOS
  Selene stabilizer plugin
- optional native DEM sampling via ``build_native_sampler(...)``
- a uniform depolarizing noise model with ``p1 = p2 = p_meas = p_init = p``
- ``SurfaceDecoder(..., decoder_type="pymatching")`` with PECOS-native DEMs

For the ``sim`` backend, decoding is performed relative to a cached noiseless
reference trajectory from the same Guppy/QIS circuit. This makes the gate-level
path compatible with the native DEM's "deviations from ideal trajectory" view.

Instead of relying on one fixed memory duration, the default workflow sweeps
round counts ``r in {2d, 3d, 4d}`` for each ``(distance, basis, p)`` point
and fits a per-round logical error rate ``epsilon`` via

    p_L(r) ~= 0.5 * (1 - (1 - 2 * epsilon) ** r)

This is a cleaner way to reduce temporal-boundary sensitivity than trying to
decode only the "middle" rounds of a finite spacetime volume.

Example:
    python examples/surface/native_dem_threshold_sweep.py --shots 200

    python examples/surface/native_dem_threshold_sweep.py \\
        --distances 3 5 7 9 \\
        --duration-multipliers 2 3 4 \\
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
from typing import Any


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


_CACHED_SELENE_INSTANCES: list[Any] = []


def _cleanup_cached_selene_instances() -> None:
    """Best-effort cleanup for temporary Selene build directories."""
    while _CACHED_SELENE_INSTANCES:
        instance = _CACHED_SELENE_INSTANCES.pop()
        with contextlib.suppress(Exception):
            instance.delete_files()


atexit.register(_cleanup_cached_selene_instances)


def _backend_runtime_label(sample_backend: str, native_circuit_source: str = "abstract") -> str:
    """Describe one sampling backend in human-readable terms."""
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
        return f"build_native_sampler(..., circuit_source={native_circuit_source!r}) + PyMatching on the native DEM"
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


def ler_over_rounds(per_round_rate: float, num_rounds: int) -> float:
    """Project a per-round logical error rate over ``num_rounds`` rounds."""
    if num_rounds <= 0:
        msg = "num_rounds must be positive"
        raise ValueError(msg)
    if per_round_rate <= 0.0:
        return 0.0
    if per_round_rate >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * per_round_rate) ** num_rounds)


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


@cache
def _decoder_runtime(
    distance: int,
    total_rounds: int,
    basis: str,
    physical_error_rate: float,
    dem_mode: str,
    native_circuit_source: str,
) -> _DecoderRuntime:
    """Build and cache the expensive native decoder-side objects once."""
    from pecos.qec.surface import NoiseModel, SurfaceDecoder

    basis = basis.upper()
    patch = _surface_patch(distance)
    noise = NoiseModel(
        p1=physical_error_rate,
        p2=physical_error_rate,
        p_meas=physical_error_rate,
        p_init=physical_error_rate,
    )
    decoder = SurfaceDecoder(
        patch,
        num_rounds=total_rounds,
        noise=noise,
        decoder_type="pymatching",
        use_circuit_level_dem=True,
        circuit_level_dem_mode=dem_mode,
        circuit_level_dem_source=native_circuit_source,
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
) -> _NativeSamplerRuntime:
    """Build and cache the native sampler + PyMatching decoder bundle once."""
    from pecos.qec.surface import build_native_sampler
    from pecos_rslib.decoders import PyMatchingDecoder

    runtime = _decoder_runtime(
        distance,
        total_rounds,
        basis,
        physical_error_rate,
        dem_mode,
        native_circuit_source,
    )
    sampler = build_native_sampler(
        runtime.patch,
        total_rounds,
        runtime.noise,
        basis=basis,
        circuit_source=native_circuit_source,
    )
    dem_str = runtime.decoder.get_dem(basis.upper(), circuit_level=True)
    dem_decoder = PyMatchingDecoder.from_dem(dem_str)
    # The traced-QIS sampler stack has a noticeable one-time initialization cost
    # on its first sample. Pay that once when the cached runtime is created so
    # subsequent point evaluations stay on the true steady-state path.
    warm_det_events, _ = sampler.sample(num_shots=1, seed=0)
    dem_decoder.decode(warm_det_events[0].astype(int).tolist())
    return _NativeSamplerRuntime(
        decoder_runtime=runtime,
        sampler=sampler,
        dem_decoder=dem_decoder,
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
            p_1q=physical_error_rate,
            p_2q=physical_error_rate,
            p_meas=physical_error_rate,
            p_init=physical_error_rate,
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
        noise_model = pecos.depolarizing_noise().with_uniform_probability(physical_error_rate)
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
    duration_multipliers: list[int],
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
        (distance, basis, physical_error_rate, duration_multiplier * distance)
        for basis in bases
        for distance in distances
        for physical_error_rate in error_rates
        for duration_multiplier in duration_multipliers
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
        )
        sampler = native_runtime.sampler
        dem_decoder = native_runtime.dem_decoder
        detection_events, observable_flips = sampler.sample(num_shots=num_shots, seed=seed)

        num_raw_errors = None
        for shot_idx in range(num_shots):
            events_flat = detection_events[shot_idx].astype(np.uint8).tolist()
            decode_result = dem_decoder.decode(events_flat)
            predicted_flip = _predicted_observable_flip(decode_result)
            true_flip = int(observable_flips[shot_idx, 0]) if observable_flips.shape[1] > 0 else 0
            num_logical_errors += int(predicted_flip != true_flip)
    else:
        msg = f"Unknown sample backend: {sample_backend}"
        raise ValueError(msg)

    logical_error_rate = num_logical_errors / num_shots if num_shots else 0.0
    raw_error_rate = None if num_raw_errors is None else (num_raw_errors / num_shots if num_shots else 0.0)

    return SweepPoint(
        backend=sample_backend,
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
    if len(points) == 1:
        point = points[0]
        return ler_per_round_exp(point.logical_error_rate, point.total_rounds)

    def objective(per_round_rate: float) -> float:
        return sum(
            (ler_over_rounds(per_round_rate, point.total_rounds) - point.logical_error_rate) ** 2 for point in points
        )

    left = 0.0
    right = 0.499999999999
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
    residuals = [ler_over_rounds(fitted_per_round, point.total_rounds) - point.logical_error_rate for point in ordered]
    rms_error = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
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
    )


def _estimate_threshold(summaries: list[FitSummary]) -> float | None:
    """Estimate a crossing between the smallest and largest distance curves."""
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
        diffs.append(
            (
                p,
                large.fitted_projected_logical_error_rate_over_d_rounds
                - small.fitted_projected_logical_error_rate_over_d_rounds,
            ),
        )

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
        ordered = [by_key[(distance, p)].fitted_projected_logical_error_rate_over_d_rounds for distance in distances]
        rows.append((p, all(next_value < value for value, next_value in itertools.pairwise(ordered))))
    return rows


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
            summary = by_key[(distance, p)]
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
    return {
        "approx_threshold_crossing": _estimate_threshold(summaries),
        "suppression": [
            {
                "physical_error_rate": p,
                "is_suppressed": is_suppressed,
            }
            for p, is_suppressed in _suppression_summary(summaries)
        ],
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
            "error_rates": sorted(set(args.error_rates)),
            "shots": args.shots,
            "dem_mode": args.dem_mode,
            "native_circuit_source": args.native_circuit_source,
            "seed": args.seed,
            "backend_runtime_descriptions": {
                backend: _backend_runtime_label(backend, args.native_circuit_source)
                for backend in sorted({point.backend for point in points})
            },
            "noise_model": "uniform depolarizing with p1 = p2 = p_meas = p_init = p",
            "fit_model": "p_L(r) = 0.5 * (1 - (1 - 2 * epsilon) ** r)",
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


def _value_ticks(min_value: float, max_value: float, *, count: int = 5) -> list[float]:
    """Produce simple linear ticks between two values."""
    if max_value <= min_value:
        return [min_value]
    if count <= 1:
        return [min_value, max_value]
    return [min_value + (max_value - min_value) * i / (count - 1) for i in range(count)]


def _x_pos(value: float, x_min: float, x_max: float, plot_left: float, plot_width: float) -> float:
    """Map an x value into SVG coordinates."""
    if x_max <= x_min:
        return plot_left + plot_width / 2.0
    return plot_left + (value - x_min) / (x_max - x_min) * plot_width


def _y_pos(value: float, y_min: float, y_max: float, plot_top: float, plot_height: float) -> float:
    """Map a positive y value into SVG coordinates using log scaling."""
    value = max(value, y_min)
    if y_max <= y_min:
        return plot_top + plot_height / 2.0
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)
    log_value = math.log10(value)
    return plot_top + (log_max - log_value) / (log_max - log_min) * plot_height


def _write_svg_plot(
    output_path: Path,
    *,
    summaries: list[FitSummary],
    metric: str,
    title: str,
    y_label: str,
) -> None:
    """Write a simple standalone SVG curve plot."""
    distances = sorted({summary.distance for summary in summaries})
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}

    values = [getattr(summary, metric) for summary in summaries if getattr(summary, metric) > 0.0]
    if values:
        y_min = max(min(values) * 0.8, 1e-12)
        y_max = max(max(values) * 1.2, y_min * 10.0)
    else:
        y_min = 1e-12
        y_max = 1e-6

    x_min = min(error_rates)
    x_max = max(error_rates)

    width = 980.0
    height = 640.0
    plot_left = 110.0
    plot_right = 40.0
    plot_top = 70.0
    plot_bottom = 90.0
    plot_width = width - plot_left - plot_right
    plot_height = height - plot_top - plot_bottom
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0f766e"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" fill="#0f172a">'
        f"{html.escape(title)}</text>",
        f'<text x="{width / 2:.1f}" y="{height - 20:.1f}" text-anchor="middle" font-size="18" fill="#334155">'
        "Physical error rate p</text>",
        f'<text x="28" y="{height / 2:.1f}" text-anchor="middle" font-size="18" fill="#334155" '
        'transform="rotate(-90 28 '
        f'{height / 2:.1f})">{html.escape(y_label)}</text>',
    ]

    for tick in _value_ticks(y_min, y_max):
        y = _y_pos(tick, y_min, y_max, plot_top, plot_height)
        parts.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>',
        )
        parts.append(
            f'<text x="{plot_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">'
            f"{tick:.2e}</text>",
        )

    for p in error_rates:
        x = _x_pos(p, x_min, x_max, plot_left, plot_width)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" y2="{plot_top + plot_height:.1f}" '
            'stroke="#f1f5f9" stroke-width="1"/>',
        )
        parts.append(
            f'<text x="{x:.1f}" y="{plot_top + plot_height + 22:.1f}" '
            'text-anchor="middle" font-size="12" fill="#475569">'
            f"{p:.4g}</text>",
        )

    parts.append(
        f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" '
        'fill="none" stroke="#0f172a" stroke-width="1.5"/>',
    )

    legend_x = plot_left + 14.0
    legend_y = plot_top + 20.0

    for index, distance in enumerate(distances):
        color = colors[index % len(colors)]
        curve_points = []
        for p in error_rates:
            summary = by_key[(distance, p)]
            value = max(getattr(summary, metric), y_min)
            curve_points.append(
                f"{_x_pos(p, x_min, x_max, plot_left, plot_width):.1f},"
                f"{_y_pos(value, y_min, y_max, plot_top, plot_height):.1f}",
            )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(curve_points)}"/>',
        )
        for p in error_rates:
            summary = by_key[(distance, p)]
            x = _x_pos(p, x_min, x_max, plot_left, plot_width)
            y = _y_pos(max(getattr(summary, metric), y_min), y_min, y_max, plot_top, plot_height)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

        legend_row_y = legend_y + index * 24.0
        parts.append(
            f'<line x1="{legend_x:.1f}" y1="{legend_row_y:.1f}" x2="{legend_x + 22:.1f}" y2="{legend_row_y:.1f}" '
            f'stroke="{color}" stroke-width="3"/>',
        )
        parts.append(
            f'<text x="{legend_x + 30:.1f}" y="{legend_row_y + 4:.1f}" font-size="14" fill="#0f172a">'
            f"d={distance}</text>",
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n")


def _write_pdf_plot(
    output_path: Path,
    *,
    summaries: list[FitSummary],
    metric: str,
    title: str,
    y_label: str,
) -> None:
    """Write a PDF plot using matplotlib if it is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        msg = "matplotlib is required for --save-pdf"
        raise RuntimeError(msg) from exc

    distances = sorted({summary.distance for summary in summaries})
    error_rates = sorted({summary.physical_error_rate for summary in summaries})
    by_key = {(summary.distance, summary.physical_error_rate): summary for summary in summaries}

    fig, ax = plt.subplots(figsize=(9, 6))
    for distance in distances:
        ys = [max(getattr(by_key[(distance, p)], metric), 1e-12) for p in error_rates]
        ax.semilogy(error_rates, ys, marker="o", linewidth=2, label=f"d={distance}")

    ax.set_title(title)
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel(y_label)
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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
    backends = sorted({summary.backend for summary in summaries})
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
                if args.save_svg:
                    svg_path = output_dir / f"{stem}.svg"
                    _write_svg_plot(svg_path, summaries=basis_summaries, metric=metric, title=title, y_label=y_label)
                    print(f"Wrote SVG plot to {svg_path}")
                if args.save_pdf:
                    pdf_path = output_dir / f"{stem}.pdf"
                    _write_pdf_plot(pdf_path, summaries=basis_summaries, metric=metric, title=title, y_label=y_label)
                    print(f"Wrote PDF plot to {pdf_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7, 9], help="Odd code distances to sweep.")
    parser.add_argument(
        "--duration-multipliers",
        "--round-multipliers",
        dest="duration_multipliers",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help=(
            "Use total memory durations r = multiplier * distance for the fit. "
            "These multipliers are in units of code distance, not raw rounds."
        ),
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
        default="abstract",
        help=(
            "Which ideal circuit the native PECOS DEM/sampler path should analyze. "
            "'abstract' uses the existing high-level surface TickCircuit, while "
            "'traced_qis' traces the lowered ideal Selene/QIS gate stream and "
            "replays that exact circuit into the native PECOS analysis."
        ),
    )
    parser.add_argument(
        "--dem-mode",
        choices=["native_decomposed", "native_full"],
        default="native_decomposed",
        help="PECOS native DEM mode. PyMatching typically wants native_decomposed.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed for the runtime noise model.")
    parser.add_argument("--save-json", action="store_true", help="Write a JSON artifact with all sweep results.")
    parser.add_argument("--save-svg", action="store_true", help="Write SVG plots for each basis and fitted metric.")
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Write PDF plots for each basis and fitted metric. Requires matplotlib.",
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


def main() -> int:
    """Run the threshold sweep CLI and optionally write summary artifacts."""
    args = _parse_args()
    wants_outputs = args.save_json or args.save_svg or args.save_pdf
    output_dir = _resolve_output_dir(args.output_dir, wants_outputs=wants_outputs)
    sweep_start = time.perf_counter()

    distances = sorted(set(args.distances))
    bases = [basis.upper() for basis in args.bases]
    if args.sample_backend == "compare":
        backends = ["sim", "native_sampler"]
    elif args.sample_backend == "compare_gate_backends":
        backends = ["selene_sim", "selene_stabilizer_plugin", "sim"]
    elif args.sample_backend == "compare_all":
        backends = ["selene_sim", "selene_stabilizer_plugin", "sim", "native_sampler"]
    elif args.sample_backend == "profile_gate_backends":
        backends = ["selene_sim", "selene_stabilizer_plugin", "sim"]
    else:
        backends = [args.sample_backend]
    duration_multipliers = sorted(set(args.duration_multipliers))
    error_rates = sorted(set(args.error_rates))

    if any(distance <= 0 or distance % 2 == 0 for distance in distances):
        msg = "Distances must be positive odd integers"
        raise ValueError(msg)
    if any(multiplier <= 0 for multiplier in duration_multipliers):
        msg = "Duration multipliers must be positive"
        raise ValueError(msg)

    print("Native PECOS Surface Threshold Sweep")
    print("=" * 40)
    print(f"distances        : {distances}")
    print(f"bases            : {bases}")
    print(f"duration multipliers: {duration_multipliers} (meaning r = multiplier * distance)")
    print(f"error rates      : {error_rates}")
    print(f"shots / point    : {args.shots}")
    print(f"sample backend mode: {args.sample_backend}")
    print(f"executed backends: {backends}")
    print(f"DEM mode         : {args.dem_mode}")
    print(f"native circuit source: {args.native_circuit_source}")
    print("decoder          : PyMatching via SurfaceDecoder(native PECOS DEM)")
    for backend in backends:
        print(f"runtime[{backend}]  : {_backend_runtime_label(backend, args.native_circuit_source)}")
    print("noise model      : depolarizing with p1 = p2 = p_meas = p_init = p")
    print("fit model        : p_L(r) = 0.5 * (1 - (1 - 2 * epsilon) ** r)")
    if output_dir is not None:
        print(f"artifact dir     : {output_dir}")

    if args.sample_backend == "profile_gate_backends":
        _profile_gate_backends(
            backends=backends,
            distances=distances,
            bases=bases,
            error_rates=error_rates,
            duration_multipliers=duration_multipliers,
            shots=args.shots,
            seed=args.seed,
            warmup_repetitions=args.benchmark_warmup,
            benchmark_repetitions=args.benchmark_repetitions,
        )
        return 0

    all_points: list[SweepPoint] = []
    fit_summaries: list[FitSummary] = []
    point_timings: list[dict[str, Any]] = []

    total_points = len(distances) * len(bases) * len(error_rates) * len(duration_multipliers) * len(backends)
    point_idx = 0

    for basis in bases:
        for distance in distances:
            for physical_error_rate in error_rates:
                for duration_multiplier in duration_multipliers:
                    total_rounds = duration_multiplier * distance
                    for backend in backends:
                        point_idx += 1
                        point_seed = args.seed + point_idx
                        print(
                            f"[{point_idx:>3}/{total_points}] "(
                                f"backend={backend} basis={basis} d={distance} "
                                f"p={physical_error_rate:.5g} r={total_rounds} ..."
                            ),
                        )
                        point_start = time.perf_counter()
                        point = _run_memory_point(
                            sample_backend=backend,
                            distance=distance,
                            basis=basis,
                            physical_error_rate=physical_error_rate,
                            total_rounds=total_rounds,
                            num_shots=args.shots,
                            dem_mode=args.dem_mode,
                            native_circuit_source=args.native_circuit_source,
                            seed=point_seed,
                        )
                        elapsed_seconds = time.perf_counter() - point_start
                        all_points.append(point)
                        point_timings.append(
                            {
                                "backend": backend,
                                "basis": basis,
                                "distance": distance,
                                "physical_error_rate": physical_error_rate,
                                "total_rounds": total_rounds,
                                "num_shots": args.shots,
                                "elapsed_seconds": elapsed_seconds,
                            },
                        )
                        naive_per_round = ler_per_round_exp(point.logical_error_rate, point.total_rounds)
                        print(
                            "    "
                            f"LER={point.logical_error_rate:.6e} "
                            f"raw={_format_rate(point.raw_error_rate)} "
                            f"naive_per_round={naive_per_round:.6e} "
                            f"elapsed={elapsed_seconds:.3f}s",
                        )

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
                    fit_summary = _fit_summary_from_points(group_points)
                    fit_summaries.append(fit_summary)
                    group_fit_summaries[backend] = fit_summary
                    observed = ", ".join(
                        f"r={round_value}:{logical_rate:.3e}"
                        for round_value, logical_rate in zip(
                            fit_summary.round_values,
                            fit_summary.observed_logical_error_rates,
                            strict=False,
                        )
                    )
                    print(
                        "    "
                        f"[{backend}] "
                        f"fit_epsilon={fit_summary.fitted_logical_error_rate_per_round:.6e} "
                        f"fit_proj_d={fit_summary.fitted_projected_logical_error_rate_over_d_rounds:.6e} "
                        f"fit_rms={fit_summary.fit_root_mean_square_error:.3e} "
                        f"[{observed}]",
                    )

                if "selene_sim" in group_fit_summaries:
                    ref_summary = group_fit_summaries["selene_sim"]
                    for backend in backends:
                        if backend == "selene_sim":
                            continue
                        summary = group_fit_summaries[backend]
                        delta_epsilon = (
                            summary.fitted_logical_error_rate_per_round
                            - ref_summary.fitted_logical_error_rate_per_round
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
                        sampler_summary.fitted_logical_error_rate_per_round
                        - sim_summary.fitted_logical_error_rate_per_round
                    )
                    delta_proj_d = (
                        sampler_summary.fitted_projected_logical_error_rate_over_d_rounds
                        - sim_summary.fitted_projected_logical_error_rate_over_d_rounds
                    )
                    print(
                        f"    compare delta_epsilon={delta_epsilon:+.3e} delta_proj_d={delta_proj_d:+.3e}",
                    )

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

            crossing = _estimate_threshold(basis_summaries)
            print()
            if crossing is None:
                print(
                    (
                        f"{basis} basis [{backend}]: no d={min(distances)} vs "
                        f"d={max(distances)} crossing was detected on this sweep."
                    ),
                )
            else:
                print(
                    f"{basis} basis [{backend}]: approximate threshold crossing "
                    f"(smallest vs largest distance, fitted d-round LER) is near p ~= {crossing:.6g}.",
                )

            print(f"{basis} basis [{backend}] suppression check (fitted d-round LER decreases with distance):")
            for p, is_suppressed in _suppression_summary(basis_summaries):
                status = "suppressed" if is_suppressed else "not suppressed"
                print(f"  p={p:.5g}: {status}")

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
