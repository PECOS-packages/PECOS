"""Benchmark graphlike DEM projections on fixed traced-QIS surface-code samples.

This is a narrower companion to ``dem_decomposition_diagnostics.py``. It builds
each DEM view once, samples once from the exact native influence model, then
times correlated PyMatching construction and batch decoding for the selected
graphlike projections.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dem_decomposition_diagnostics import (
    compare_raw_dems,
    dem_stats,
    terminal_graphlike_projection,
    true_observable_flips,
)

SZZ_Z_FRAME_P1_GATE_RATES = {"Z": 0.0, "SZ": 0.0, "SZdg": 0.0}


@dataclass(frozen=True)
class TimedValue:
    label: str
    elapsed_s: float


@dataclass(frozen=True)
class VariantResult:
    variant: str
    dem_stats: dict[str, Any]
    dem_build_s: float
    decoder_build_s: float
    decode_s: float
    logical_errors: int
    logical_error_rate: float


@dataclass(frozen=True)
class BenchmarkResult:
    distance: int
    rounds: int
    basis: str
    interaction_basis: str
    p: float
    shots: int
    setup_timings: list[TimedValue]
    raw_comparison: dict[str, Any]
    variants: list[VariantResult]


def timed(label: str, callback: Any) -> tuple[Any, float]:
    print(f"[start] {label}", flush=True)
    start = time.perf_counter()
    value = callback()
    elapsed = time.perf_counter() - start
    print(f"[done]  {label}: {elapsed:.3f}s", flush=True)
    return value, elapsed


def decode_with_correlated_pymatching(
    dem_text: str,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
) -> tuple[int, float, float]:
    from pecos.decoders import PyMatchingDecoder

    decoder, decoder_build_s = timed(
        "build correlated PyMatching",
        lambda: PyMatchingDecoder.from_dem_with_correlations(dem_text, enable_correlations=True),
    )

    expected = true_observable_flips(observable_flips)

    def decode() -> list[list[int]]:
        flat = detection_events.astype(np.uint8).flatten().tolist()
        return decoder.decode_batch(flat, len(detection_events))

    predictions, decode_s = timed("decode batch", decode)
    predicted = np.array([prediction[0] if prediction else 0 for prediction in predictions], dtype=np.uint8)
    logical_errors = int(np.sum(predicted != expected))
    return logical_errors, decoder_build_s, decode_s


def build_case(
    *,
    distance: int,
    rounds: int,
    basis: str,
    interaction_basis: str,
    p: float,
    shots: int,
    seed: int,
    variants: list[str],
) -> BenchmarkResult:
    from pecos.qec.surface import NoiseModel, SurfacePatch, build_native_sampler
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit_via_stim,
        normalize_traced_qis_tick_circuit,
    )
    from pecos.qec.surface.decode import (
        _build_surface_tick_circuit_for_native_model,
        generate_circuit_level_dem_from_builder,
    )

    print(
        f"\n=== d={distance} r={rounds} basis={basis} basis2q={interaction_basis} "
        f"p={p:g} shots={shots} ===",
        flush=True,
    )
    setup_timings: list[TimedValue] = []
    patch = SurfacePatch.create(distance=distance)
    noise = NoiseModel(p1=p / 30.0, p2=p, p_meas=p / 3.0, p_prep=p / 3.0)
    noise_args = {
        "p1": noise.p1,
        "p1_gate_rates": SZZ_Z_FRAME_P1_GATE_RATES if interaction_basis == "szz" else None,
        "p2": noise.p2,
        "p_meas": noise.p_meas,
        "p_prep": noise.p_prep,
    }

    tick_circuit, elapsed = timed(
        "build traced-QIS tick circuit",
        lambda: _build_surface_tick_circuit_for_native_model(
            patch,
            rounds,
            basis,
            circuit_source="traced_qis",
            interaction_basis=interaction_basis,
        ),
    )
    setup_timings.append(TimedValue("build_traced_qis_tick_circuit", elapsed))
    normalize_traced_qis_tick_circuit(tick_circuit, context="graphlike DEM projection benchmark")

    native_raw, elapsed = timed(
        "build native raw DEM",
        lambda: generate_circuit_level_dem_from_builder(
            patch,
            rounds,
            noise,
            basis=basis,
            decompose_errors=False,
            circuit_source="traced_qis",
            interaction_basis=interaction_basis,
        ),
    )
    setup_timings.append(TimedValue("build_native_raw_dem", elapsed))

    stim_raw, elapsed = timed(
        "build Stim raw DEM",
        lambda: generate_dem_from_tick_circuit_via_stim(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )
    setup_timings.append(TimedValue("build_stim_raw_dem", elapsed))
    raw_comparison = asdict(compare_raw_dems(native_raw, stim_raw))
    print(
        "raw native vs Stim: "
        f"only_native={raw_comparison['only_native']} only_stim={raw_comparison['only_stim']} "
        f"max_rel={raw_comparison['max_rel_probability_diff']:.3e}",
        flush=True,
    )

    sampler, elapsed = timed(
        "build native influence sampler",
        lambda: build_native_sampler(
            patch,
            rounds,
            noise,
            basis=basis,
            circuit_source="traced_qis",
            interaction_basis=interaction_basis,
            sampling_model="influence_dem",
        ),
    )
    setup_timings.append(TimedValue("build_native_influence_sampler", elapsed))

    (detection_events, observable_flips), elapsed = timed(
        "sample native influence events",
        lambda: sampler.sample(num_shots=shots, seed=seed),
    )
    setup_timings.append(TimedValue("sample_native_influence_events", elapsed))

    def build_variant_dem(variant: str) -> str:
        if variant == "native_source":
            return generate_circuit_level_dem_from_builder(
                patch,
                rounds,
                noise,
                basis=basis,
                decompose_errors=True,
                dem_decomposition="source_graphlike",
                circuit_source="traced_qis",
                interaction_basis=interaction_basis,
            )
        if variant == "native_terminal":
            try:
                return generate_circuit_level_dem_from_builder(
                    patch,
                    rounds,
                    noise,
                    basis=basis,
                    decompose_errors=True,
                    dem_decomposition="terminal_graphlike",
                    circuit_source="traced_qis",
                    interaction_basis=interaction_basis,
                )
            except RuntimeError as exc:
                if "terminal graphlike" not in str(exc):
                    raise
                print("[info] using Python terminal graphlike projection fallback", flush=True)
                return terminal_graphlike_projection(native_raw)
        if variant == "stim":
            return generate_dem_from_tick_circuit_via_stim(
                tick_circuit,
                decompose_errors=True,
                **noise_args,
            )
        msg = f"unknown variant {variant!r}"
        raise ValueError(msg)

    results: list[VariantResult] = []
    for variant in variants:
        dem_text, dem_build_s = timed(f"build {variant} DEM", lambda variant=variant: build_variant_dem(variant))
        stats = asdict(dem_stats(dem_text))
        print(
            f"{variant} DEM: errors={stats['error_lines']} sep={stats['separator_lines']} "
            f"hyper={stats['hyperedge_lines']} max_line={stats['max_line_detectors']}",
            flush=True,
        )
        logical_errors, decoder_build_s, decode_s = decode_with_correlated_pymatching(
            dem_text,
            detection_events,
            observable_flips,
        )
        result = VariantResult(
            variant=variant,
            dem_stats=stats,
            dem_build_s=dem_build_s,
            decoder_build_s=decoder_build_s,
            decode_s=decode_s,
            logical_errors=logical_errors,
            logical_error_rate=logical_errors / shots if shots else 0.0,
        )
        print(
            f"{variant}: errors={result.logical_errors} "
            f"LER={result.logical_error_rate:.6f} "
            f"build={result.dem_build_s:.3f}s "
            f"matcher={result.decoder_build_s:.3f}s "
            f"decode={result.decode_s:.3f}s",
            flush=True,
        )
        results.append(result)

    return BenchmarkResult(
        distance=distance,
        rounds=rounds,
        basis=basis,
        interaction_basis=interaction_basis,
        p=p,
        shots=shots,
        setup_timings=setup_timings,
        raw_comparison=raw_comparison,
        variants=results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[7])
    parser.add_argument("--rounds", type=int, default=None, help="Rounds to use. Defaults to distance.")
    parser.add_argument("--bases", nargs="+", choices=["X", "Z"], default=["X"])
    parser.add_argument("--interaction-bases", nargs="+", choices=["cx", "szz"], default=["cx", "szz"])
    parser.add_argument("--p", type=float, default=0.006)
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["native_source", "native_terminal", "stim"],
        default=["native_terminal", "stim", "native_source"],
    )
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    for distance in args.distances:
        rounds = args.rounds if args.rounds is not None else distance
        results.extend(
            build_case(
                distance=distance,
                rounds=rounds,
                basis=basis,
                interaction_basis=interaction_basis,
                p=args.p,
                shots=args.shots,
                seed=args.seed,
                variants=args.variants,
            )
            for basis in args.bases
            for interaction_basis in args.interaction_bases
        )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nWrote {args.save_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
