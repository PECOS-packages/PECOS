"""Diagnose raw DEMs and graphlike decompositions for traced-QIS surface circuits.

The script keeps sampling fixed: each case samples once from the exact native
influence-model DEM, then decodes the same detector events with several decoder
views of the model. This separates raw DEM generation from graphlike
decomposition quality.

Example:
    uv run python examples/surface/dem_decomposition_diagnostics.py \\
        --distances 3 5 --bases X Z --interaction-bases cx szz \\
        --p 0.006 --shots 10000 --tesseract-beams 5 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ERROR_RE = re.compile(r"error\(([^)]+)\)\s*(.*)")
DET_RE = re.compile(r"\bD(\d+)\b")
OBS_RE = re.compile(r"\bL(\d+)\b")


@dataclass(frozen=True)
class DemStats:
    error_lines: int
    probability_sum: float
    separator_lines: int
    hyperedge_lines: int
    logical_lines: int
    max_component_detectors: int
    max_line_detectors: int
    pure_logical_components: int


@dataclass(frozen=True)
class RawDemComparison:
    native_errors: int
    stim_errors: int
    only_native: int
    only_stim: int
    common: int
    max_abs_probability_diff: float
    max_rel_probability_diff: float
    l1_probability_diff: float


@dataclass(frozen=True)
class DecodeSummary:
    decoder: str
    logical_errors: int
    logical_error_rate: float
    elapsed_s: float


@dataclass(frozen=True)
class CaseResult:
    distance: int
    rounds: int
    basis: str
    interaction_basis: str
    p: float
    shots: int
    raw_comparison: RawDemComparison
    dem_stats: dict[str, DemStats]
    decoders: list[DecodeSummary]


def _combine_independent_probabilities(left: float, right: float) -> float:
    """Combine independent mechanisms with the same XOR effect."""
    return left * (1.0 - right) + right * (1.0 - left)


def _toggle(values: set[int], value: int) -> None:
    if value in values:
        values.remove(value)
    else:
        values.add(value)


def _canonical_effect_key(targets: str) -> str:
    """Canonicalize DEM targets by XORing all ``^`` components."""
    detectors: set[int] = set()
    observables: set[int] = set()
    for component in targets.split("^"):
        for detector in DET_RE.findall(component):
            _toggle(detectors, int(detector))
        for observable in OBS_RE.findall(component):
            _toggle(observables, int(observable))
    tokens = [f"D{det}" for det in sorted(detectors)]
    tokens.extend(f"L{obs}" for obs in sorted(observables))
    return " ".join(tokens)


def dem_effect_probabilities(dem_text: str) -> dict[str, float]:
    """Aggregate DEM error probabilities by combined detector/observable effect."""
    effects: dict[str, float] = {}
    for line in dem_text.splitlines():
        match = ERROR_RE.match(line.strip())
        if not match:
            continue
        probability = float(match.group(1))
        key = _canonical_effect_key(match.group(2))
        if not key:
            continue
        effects[key] = _combine_independent_probabilities(effects.get(key, 0.0), probability)
    return effects


def compare_raw_dems(native_dem: str, stim_dem: str) -> RawDemComparison:
    """Compare raw native and Stim DEMs after aggregating duplicate effects."""
    native = dem_effect_probabilities(native_dem)
    stim = dem_effect_probabilities(stim_dem)
    native_keys = set(native)
    stim_keys = set(stim)
    common = native_keys & stim_keys

    max_abs = 0.0
    max_rel = 0.0
    l1 = 0.0
    for key in common:
        diff = abs(native[key] - stim[key])
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, diff / max(native[key], stim[key], 1e-18))
        l1 += diff
    for key in native_keys - stim_keys:
        l1 += native[key]
    for key in stim_keys - native_keys:
        l1 += stim[key]

    return RawDemComparison(
        native_errors=len(native),
        stim_errors=len(stim),
        only_native=len(native_keys - stim_keys),
        only_stim=len(stim_keys - native_keys),
        common=len(common),
        max_abs_probability_diff=max_abs,
        max_rel_probability_diff=max_rel,
        l1_probability_diff=l1,
    )


def dem_stats(dem_text: str) -> DemStats:
    """Summarize the structure of a DEM string."""
    error_lines = 0
    probability_sum = 0.0
    separator_lines = 0
    hyperedge_lines = 0
    logical_lines = 0
    max_component_detectors = 0
    max_line_detectors = 0
    pure_logical_components = 0

    for line in dem_text.splitlines():
        match = ERROR_RE.match(line.strip())
        if not match:
            continue
        error_lines += 1
        probability_sum += float(match.group(1))
        targets = match.group(2)
        components = targets.split(" ^ ")
        if len(components) > 1:
            separator_lines += 1
        if "L" in targets:
            logical_lines += 1

        line_detectors = len(DET_RE.findall(targets))
        max_line_detectors = max(max_line_detectors, line_detectors)
        if line_detectors > 2:
            hyperedge_lines += 1

        for component in components:
            component_detectors = len(DET_RE.findall(component))
            component_observables = len(OBS_RE.findall(component))
            max_component_detectors = max(max_component_detectors, component_detectors)
            if component_detectors == 0 and component_observables > 0:
                pure_logical_components += 1

    return DemStats(
        error_lines=error_lines,
        probability_sum=probability_sum,
        separator_lines=separator_lines,
        hyperedge_lines=hyperedge_lines,
        logical_lines=logical_lines,
        max_component_detectors=max_component_detectors,
        max_line_detectors=max_line_detectors,
        pure_logical_components=pure_logical_components,
    )


def strip_logical_observable_lines(dem_text: str) -> str:
    return "\n".join(line for line in dem_text.splitlines() if not line.startswith("logical_observable"))


def true_observable_flips(observable_flips: np.ndarray) -> np.ndarray:
    if observable_flips.ndim == 1:
        return observable_flips.astype(np.uint8)
    if observable_flips.shape[1] == 0:
        return np.zeros(observable_flips.shape[0], dtype=np.uint8)
    return observable_flips[:, 0].astype(np.uint8)


def decode_with_tesseract(
    dem_text: str,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
    *,
    beam: int,
) -> int:
    from pecos.decoders import TesseractDecoder

    decoder = TesseractDecoder.from_dem(
        strip_logical_observable_lines(dem_text),
        preset="fast",
        det_beam=beam,
    )
    expected = true_observable_flips(observable_flips)
    syndromes = [detection_events[index].astype(np.uint8).tolist() for index in range(len(detection_events))]
    results = decoder.decode_batch(syndromes)
    return sum(int(int(result.observables_mask & 1) != expected[index]) for index, result in enumerate(results))


def decode_with_pymatching(
    dem_text: str,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
    *,
    correlated: bool,
) -> int:
    from pecos.decoders import PyMatchingDecoder

    if correlated:
        decoder = PyMatchingDecoder.from_dem_with_correlations(dem_text, enable_correlations=True)
    else:
        decoder = PyMatchingDecoder.from_dem(dem_text)

    expected = true_observable_flips(observable_flips)
    predictions = decoder.decode_batch(
        detection_events.astype(np.uint8).flatten().tolist(),
        len(detection_events),
    )
    predicted = np.array([prediction[0] if prediction else 0 for prediction in predictions], dtype=np.uint8)
    return int(np.sum(predicted != expected))


def _timed_decode(label: str, callback: Any, shots: int) -> DecodeSummary:
    start = time.perf_counter()
    errors = int(callback())
    elapsed = time.perf_counter() - start
    return DecodeSummary(
        decoder=label,
        logical_errors=errors,
        logical_error_rate=errors / shots if shots else 0.0,
        elapsed_s=elapsed,
    )


def run_case(
    *,
    distance: int,
    rounds: int,
    basis: str,
    interaction_basis: str,
    p: float,
    shots: int,
    seed: int,
    tesseract_beams: list[int],
) -> CaseResult:
    from pecos.qec.surface import NoiseModel, SurfacePatch, build_native_sampler
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit_via_stim,
        normalize_traced_qis_tick_circuit,
    )
    from pecos.qec.surface.decode import (
        _build_surface_tick_circuit_for_native_model,
        generate_circuit_level_dem_from_builder,
    )

    patch = SurfacePatch.create(distance=distance)
    noise = NoiseModel(p1=p / 30.0, p2=p, p_meas=p / 3.0, p_prep=p / 3.0)
    noise_args = {
        "p1": noise.p1,
        "p2": noise.p2,
        "p_meas": noise.p_meas,
        "p_prep": noise.p_prep,
    }

    tick_circuit = _build_surface_tick_circuit_for_native_model(
        patch,
        rounds,
        basis,
        circuit_source="traced_qis",
        interaction_basis=interaction_basis,
    )
    normalize_traced_qis_tick_circuit(tick_circuit, context="DEM decomposition diagnostics")

    native_raw = generate_circuit_level_dem_from_builder(
        patch,
        rounds,
        noise,
        basis=basis,
        decompose_errors=False,
        circuit_source="traced_qis",
        interaction_basis=interaction_basis,
    )
    native_decomposed = generate_circuit_level_dem_from_builder(
        patch,
        rounds,
        noise,
        basis=basis,
        decompose_errors=True,
        circuit_source="traced_qis",
        interaction_basis=interaction_basis,
    )
    stim_raw = generate_dem_from_tick_circuit_via_stim(
        tick_circuit,
        decompose_errors=False,
        **noise_args,
    )
    stim_decomposed = generate_dem_from_tick_circuit_via_stim(
        tick_circuit,
        decompose_errors=True,
        **noise_args,
    )

    sampler = build_native_sampler(
        patch,
        rounds,
        noise,
        basis=basis,
        circuit_source="traced_qis",
        interaction_basis=interaction_basis,
        sampling_model="influence_dem",
    )
    detection_events, observable_flips = sampler.sample(num_shots=shots, seed=seed)

    decoders = [
        _timed_decode(
            f"native_raw_tesseract_b{beam}",
            lambda beam=beam: decode_with_tesseract(
                native_raw,
                detection_events,
                observable_flips,
                beam=beam,
            ),
            shots,
        )
        for beam in tesseract_beams
    ]
    decoders.extend(
        [
            _timed_decode(
                "native_decomp_pymatching",
                lambda: decode_with_pymatching(
                    native_decomposed,
                    detection_events,
                    observable_flips,
                    correlated=False,
                ),
                shots,
            ),
            _timed_decode(
                "native_decomp_pymatching_correlated",
                lambda: decode_with_pymatching(
                    native_decomposed,
                    detection_events,
                    observable_flips,
                    correlated=True,
                ),
                shots,
            ),
            _timed_decode(
                "stim_decomp_pymatching",
                lambda: decode_with_pymatching(
                    stim_decomposed,
                    detection_events,
                    observable_flips,
                    correlated=False,
                ),
                shots,
            ),
            _timed_decode(
                "stim_decomp_pymatching_correlated",
                lambda: decode_with_pymatching(
                    stim_decomposed,
                    detection_events,
                    observable_flips,
                    correlated=True,
                ),
                shots,
            ),
        ],
    )

    return CaseResult(
        distance=distance,
        rounds=rounds,
        basis=basis,
        interaction_basis=interaction_basis,
        p=p,
        shots=shots,
        raw_comparison=compare_raw_dems(native_raw, stim_raw),
        dem_stats={
            "native_raw": dem_stats(native_raw),
            "native_decomposed": dem_stats(native_decomposed),
            "stim_raw": dem_stats(stim_raw),
            "stim_decomposed": dem_stats(stim_decomposed),
        },
        decoders=decoders,
    )


def print_case(result: CaseResult) -> None:
    print(
        f"\n=== d={result.distance} r={result.rounds} basis={result.basis} "
        f"basis2q={result.interaction_basis} p={result.p:g} shots={result.shots} ===",
    )
    raw = result.raw_comparison
    print(
        "raw native vs Stim: "
        f"native={raw.native_errors} stim={raw.stim_errors} "
        f"only_native={raw.only_native} only_stim={raw.only_stim} "
        f"max_rel={raw.max_rel_probability_diff:.3e} "
        f"l1={raw.l1_probability_diff:.3e}",
    )

    print("DEM stats:")
    print("  source              errors       psum   sep hyper max_comp max_line pure_L")
    for name, stats in result.dem_stats.items():
        print(
            f"  {name:<18} {stats.error_lines:6d} {stats.probability_sum:10.6f} "
            f"{stats.separator_lines:5d} {stats.hyperedge_lines:5d} "
            f"{stats.max_component_detectors:8d} {stats.max_line_detectors:8d} "
            f"{stats.pure_logical_components:6d}",
        )

    print("Decode on identical raw influence samples:")
    print("  decoder                                  errors        LER   elapsed")
    for summary in result.decoders:
        print(
            f"  {summary.decoder:<36} {summary.logical_errors:6d} "
            f"{summary.logical_error_rate:10.6f} {summary.elapsed_s:8.3f}s",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--rounds", type=int, default=None, help="Rounds to use. Defaults to distance.")
    parser.add_argument("--bases", nargs="+", choices=["X", "Z"], default=["X", "Z"])
    parser.add_argument("--interaction-bases", nargs="+", choices=["cx", "szz"], default=["cx", "szz"])
    parser.add_argument("--p", nargs="+", type=float, default=[0.006])
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--tesseract-beams", nargs="+", type=int, default=[5])
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[CaseResult] = []
    for distance in args.distances:
        rounds = args.rounds if args.rounds is not None else distance
        for basis in args.bases:
            for interaction_basis in args.interaction_bases:
                for p in args.p:
                    result = run_case(
                        distance=distance,
                        rounds=rounds,
                        basis=basis,
                        interaction_basis=interaction_basis,
                        p=p,
                        shots=args.shots,
                        seed=args.seed,
                        tesseract_beams=args.tesseract_beams,
                    )
                    results.append(result)
                    print_case(result)

    if args.save_json is not None:
        payload = [asdict(result) for result in results]
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
