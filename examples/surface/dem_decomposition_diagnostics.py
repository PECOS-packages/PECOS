"""Diagnose raw DEMs and graphlike decompositions for traced-QIS surface circuits.

The script keeps sampling fixed: each case samples once from the exact native
influence-model DEM, then decodes the same detector events with several decoder
views of the model. The graphlike views are lossy hyperedge-to-edge projections
for graph decoders, so this separates raw DEM generation from graphlike
decomposition quality.

Example:
    uv run python examples/surface/dem_decomposition_diagnostics.py \\
        --distances 3 5 --bases X Z --interaction-bases cx szz \\
        --p 0.006 --shots 10000 --tesseract-beams 5 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ERROR_RE = re.compile(r"error\(([^)]+)\)\s*(.*)")
DET_RE = re.compile(r"\bD(\d+)\b")
OBS_RE = re.compile(r"\bL(\d+)\b")
DETECTOR_COORD_RE = re.compile(r"detector\(([^)]*)\) D(\d+)")
GRAPHLIKE_DECODER_CHOICES = [
    "native_decomp_pymatching",
    "native_decomp_pymatching_correlated",
    "stim_decomp_pymatching",
    "stim_decomp_pymatching_correlated",
    "terminal_decomp_pymatching",
    "terminal_decomp_pymatching_correlated",
]
# The staged SZZ device model treats Z/SZ/SZdg frame updates as noiseless
# virtual operations. CX-vs-SZZ p1 location comparisons include this assumption
# as well as the gate-basis difference.
SZZ_Z_FRAME_P1_GATE_RATES = {"Z": 0.0, "SZ": 0.0, "SZdg": 0.0}


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
class PairAnalysisSummary:
    decoder: str
    pair_probability_mass: float
    wrong_probability_mass: float
    wrong_probability_fraction: float
    disagree_tesseract_probability_mass: float
    disagree_tesseract_probability_fraction: float
    wrong_count: int
    disagree_tesseract_count: int


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
    pair_analysis: list[PairAnalysisSummary] | None


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
    """Compare raw native and Stim DEMs after aggregating duplicate effects.

    ``only_native`` and ``only_stim`` report structural differences. Nonzero
    probability deltas with zero structural differences reflect independent
    probability-combination and serialization-rounding conventions.
    """
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


def parse_detector_coords(dem_text: str) -> dict[int, tuple[float, ...]]:
    """Parse detector coordinate annotations from DEM text."""
    coords: dict[int, tuple[float, ...]] = {}
    for line in dem_text.splitlines():
        match = DETECTOR_COORD_RE.match(line.strip())
        if not match:
            continue
        values = tuple(float(value.strip()) for value in match.group(1).split(",") if value.strip())
        coords[int(match.group(2))] = values
    return coords


def detector_coord_distance(
    left: int,
    right: int,
    coords: dict[int, tuple[float, ...]],
) -> float:
    """Coordinate distance with detector-id fallback for missing annotations."""
    left_coords = coords.get(left, (float(left),))
    right_coords = coords.get(right, (float(right),))
    dims = max(len(left_coords), len(right_coords))
    left_coords = left_coords + (0.0,) * (dims - len(left_coords))
    right_coords = right_coords + (0.0,) * (dims - len(right_coords))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_coords, right_coords, strict=True)))


def min_coord_terminal_pairs(
    detectors: tuple[int, ...],
    coords: dict[int, tuple[float, ...]],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Pair terminals by minimum coordinate distance, leaving one singleton if odd."""
    if len(detectors) <= 1:
        return [], list(detectors)
    if len(detectors) == 2:
        return [(detectors[0], detectors[1])], []

    best_cost: float | None = None
    best_pairs: list[tuple[int, int]] = []
    best_singles: list[int] = []
    for left_index, left in enumerate(detectors):
        for right in detectors[left_index + 1 :]:
            rest = tuple(detector for detector in detectors if detector not in {left, right})
            pairs, singles = min_coord_terminal_pairs(rest, coords)
            pairs = [(left, right), *pairs]
            cost = sum(detector_coord_distance(a, b, coords) for a, b in pairs)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pairs = pairs
                best_singles = singles
    return best_pairs, best_singles


def terminal_graphlike_projection(raw_dem: str) -> str:
    """Project raw DEM effects into minimum-span terminal-only graphlike pieces.

    Each raw mechanism keeps its combined detector/observable effect exactly,
    but the rendered decomposition uses only detectors present in that raw
    effect. This avoids cancellation/path detectors introduced by graph-path
    decompositions while still producing graphlike components for matching
    decoders. The result is a lossy decoder-facing projection of hyperedge
    correlations, not an exact raw DEM.
    """
    coords = parse_detector_coords(raw_dem)
    annotation_lines: list[str] = []
    by_targets: dict[str, float] = {}

    for line in raw_dem.splitlines():
        stripped = line.strip()
        if stripped.startswith(("detector", "logical_observable")):
            annotation_lines.append(line)
            continue

        match = ERROR_RE.match(stripped)
        if not match:
            continue
        probability = float(match.group(1))
        effect = _canonical_effect_key(match.group(2))
        detectors = tuple(sorted(int(detector) for detector in DET_RE.findall(effect)))
        observables = sorted(int(observable) for observable in OBS_RE.findall(effect))
        pairs, singles = min_coord_terminal_pairs(detectors, coords)

        components = [f"D{left} D{right}" for left, right in pairs]
        components.extend(f"D{detector}" for detector in singles)
        if not components and observables:
            # PyMatching cannot use a pure logical component. Preserve the raw
            # effect so construction fails instead of silently changing it.
            components = [f"L{observable}" for observable in observables]
        else:
            for observable in observables:
                components[-1] = f"{components[-1]} L{observable}"

        rendered_targets = " ^ ".join(components)
        if rendered_targets:
            by_targets[rendered_targets] = _combine_independent_probabilities(
                by_targets.get(rendered_targets, 0.0),
                probability,
            )

    rendered_lines = [
        f"error({probability:.16g}) {targets}"
        for targets, probability in sorted(by_targets.items())
        if probability > 0.0
    ]
    return "\n".join([*annotation_lines, *rendered_lines])


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


def dense_effect_arrays(effects: dict[str, float]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Convert effect keys into dense detector rows and observable flips."""
    keys = list(effects)
    probabilities = np.array([effects[key] for key in keys], dtype=float)
    max_detector = max((int(detector) for key in keys for detector in DET_RE.findall(key)), default=-1)
    detection_events = np.zeros((len(keys), max_detector + 1), dtype=np.uint8)
    observable_flips = np.zeros(len(keys), dtype=np.uint8)

    for index, key in enumerate(keys):
        detectors = [int(detector) for detector in DET_RE.findall(key)]
        if detectors:
            detection_events[index, detectors] = 1
        observable_flips[index] = 1 if "0" in OBS_RE.findall(key) else 0

    return keys, probabilities, detection_events, observable_flips


def tesseract_predictions(dem_text: str, detection_events: np.ndarray, *, beam: int) -> np.ndarray:
    from pecos.decoders import TesseractDecoder

    decoder = TesseractDecoder.from_dem(
        strip_logical_observable_lines(dem_text),
        preset="fast",
        det_beam=beam,
    )
    results = decoder.decode_batch([row.tolist() for row in detection_events])
    return np.array([int(result.observables_mask & 1) for result in results], dtype=np.uint8)


def pymatching_predictions(dem_text: str, detection_events: np.ndarray, *, correlated: bool) -> np.ndarray:
    from pecos.decoders import PyMatchingDecoder

    if correlated:
        decoder = PyMatchingDecoder.from_dem_with_correlations(dem_text, enable_correlations=True)
    else:
        decoder = PyMatchingDecoder.from_dem(dem_text)
    predictions = decoder.decode_batch(
        detection_events.astype(np.uint8).flatten().tolist(),
        len(detection_events),
    )
    return np.array([prediction[0] if prediction else 0 for prediction in predictions], dtype=np.uint8)


def decode_with_tesseract(
    dem_text: str,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
    *,
    beam: int,
) -> int:
    expected = true_observable_flips(observable_flips)
    predicted = tesseract_predictions(dem_text, detection_events, beam=beam)
    return int(np.sum(predicted != expected))


def decode_with_pymatching(
    dem_text: str,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
    *,
    correlated: bool,
) -> int:
    expected = true_observable_flips(observable_flips)
    predicted = pymatching_predictions(dem_text, detection_events, correlated=correlated)
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


def two_fault_pair_analysis(
    *,
    native_raw: str,
    native_decomposed: str,
    stim_decomposed: str,
    terminal_decomposed: str,
    max_effects: int,
) -> list[PairAnalysisSummary] | None:
    """Exhaustively compare decoders on all two-mechanism XOR combinations."""
    effects = dem_effect_probabilities(native_raw)
    keys, probabilities, detection_events, observable_flips = dense_effect_arrays(effects)
    if len(keys) > max_effects:
        return None

    pair_rows: list[np.ndarray] = []
    pair_observables: list[int] = []
    pair_weights: list[float] = []
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            pair_rows.append(detection_events[left] ^ detection_events[right])
            pair_observables.append(int(observable_flips[left] ^ observable_flips[right]))
            pair_weights.append(float(probabilities[left] * probabilities[right]))

    if not pair_rows:
        return []

    pair_detection_events = np.asarray(pair_rows, dtype=np.uint8)
    pair_observable_flips = np.asarray(pair_observables, dtype=np.uint8)
    weights = np.asarray(pair_weights, dtype=float)
    total_weight = float(np.sum(weights))

    predictions = {
        "native_raw_tesseract_b5": tesseract_predictions(native_raw, pair_detection_events, beam=5),
        "native_decomp_pymatching": pymatching_predictions(
            native_decomposed,
            pair_detection_events,
            correlated=False,
        ),
        "native_decomp_pymatching_correlated": pymatching_predictions(
            native_decomposed,
            pair_detection_events,
            correlated=True,
        ),
        "stim_decomp_pymatching": pymatching_predictions(
            stim_decomposed,
            pair_detection_events,
            correlated=False,
        ),
        "stim_decomp_pymatching_correlated": pymatching_predictions(
            stim_decomposed,
            pair_detection_events,
            correlated=True,
        ),
        "terminal_decomp_pymatching": pymatching_predictions(
            terminal_decomposed,
            pair_detection_events,
            correlated=False,
        ),
        "terminal_decomp_pymatching_correlated": pymatching_predictions(
            terminal_decomposed,
            pair_detection_events,
            correlated=True,
        ),
    }
    reference = predictions["native_raw_tesseract_b5"]

    summaries = []
    for name, predicted in predictions.items():
        wrong = predicted != pair_observable_flips
        disagree = predicted != reference
        wrong_mass = float(np.sum(weights[wrong]))
        disagree_mass = float(np.sum(weights[disagree]))
        summaries.append(
            PairAnalysisSummary(
                decoder=name,
                pair_probability_mass=total_weight,
                wrong_probability_mass=wrong_mass,
                wrong_probability_fraction=wrong_mass / total_weight if total_weight else 0.0,
                disagree_tesseract_probability_mass=disagree_mass,
                disagree_tesseract_probability_fraction=disagree_mass / total_weight if total_weight else 0.0,
                wrong_count=int(np.sum(wrong)),
                disagree_tesseract_count=int(np.sum(disagree)),
            ),
        )
    return summaries


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
    decoder_names: set[str],
    pair_analysis: bool,
    pair_analysis_max_effects: int,
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
        "p1_gate_rates": SZZ_Z_FRAME_P1_GATE_RATES if interaction_basis == "szz" else None,
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
    try:
        terminal_decomposed = generate_circuit_level_dem_from_builder(
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
        terminal_decomposed = terminal_graphlike_projection(native_raw)
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
    graphlike_decoder_specs = [
        (
            "native_decomp_pymatching",
            lambda: decode_with_pymatching(
                native_decomposed,
                detection_events,
                observable_flips,
                correlated=False,
            ),
        ),
        (
            "native_decomp_pymatching_correlated",
            lambda: decode_with_pymatching(
                native_decomposed,
                detection_events,
                observable_flips,
                correlated=True,
            ),
        ),
        (
            "stim_decomp_pymatching",
            lambda: decode_with_pymatching(
                stim_decomposed,
                detection_events,
                observable_flips,
                correlated=False,
            ),
        ),
        (
            "stim_decomp_pymatching_correlated",
            lambda: decode_with_pymatching(
                stim_decomposed,
                detection_events,
                observable_flips,
                correlated=True,
            ),
        ),
        (
            "terminal_decomp_pymatching",
            lambda: decode_with_pymatching(
                terminal_decomposed,
                detection_events,
                observable_flips,
                correlated=False,
            ),
        ),
        (
            "terminal_decomp_pymatching_correlated",
            lambda: decode_with_pymatching(
                terminal_decomposed,
                detection_events,
                observable_flips,
                correlated=True,
            ),
        ),
    ]
    decoders.extend(
        _timed_decode(name, callback, shots)
        for name, callback in graphlike_decoder_specs
        if name in decoder_names
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
            "terminal_decomposed": dem_stats(terminal_decomposed),
        },
        decoders=decoders,
        pair_analysis=two_fault_pair_analysis(
            native_raw=native_raw,
            native_decomposed=native_decomposed,
            stim_decomposed=stim_decomposed,
            terminal_decomposed=terminal_decomposed,
            max_effects=pair_analysis_max_effects,
        )
        if pair_analysis
        else None,
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
    if raw.only_native == 0 and raw.only_stim == 0 and raw.max_rel_probability_diff > 0:
        print(
            "  raw structures match; probability deltas reflect "
            "combination/rounding conventions.",
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

    if result.pair_analysis is None:
        return
    print("Exact two-fault analysis:")
    print("  decoder                                  wrong_mass wrong_frac disagree_mass disagree_frac")
    for summary in result.pair_analysis:
        print(
            f"  {summary.decoder:<36} "
            f"{summary.wrong_probability_mass:10.6f} "
            f"{summary.wrong_probability_fraction:10.4f} "
            f"{summary.disagree_tesseract_probability_mass:13.6f} "
            f"{summary.disagree_tesseract_probability_fraction:13.4f}",
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
    parser.add_argument(
        "--skip-tesseract",
        action="store_true",
        help="Skip raw-DEM Tesseract decoding for larger graphlike-only sampled comparisons.",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        choices=GRAPHLIKE_DECODER_CHOICES,
        default=GRAPHLIKE_DECODER_CHOICES,
        help="Graphlike decoder variants to run in sampled comparisons.",
    )
    parser.add_argument(
        "--pair-analysis",
        action="store_true",
        help="Exhaustively compare decoders on all two-fault combinations when the effect count is small enough.",
    )
    parser.add_argument("--pair-analysis-max-effects", type=int, default=400)
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
                        tesseract_beams=[] if args.skip_tesseract else args.tesseract_beams,
                        decoder_names=set(args.decoders),
                        pair_analysis=args.pair_analysis,
                        pair_analysis_max_effects=args.pair_analysis_max_effects,
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
