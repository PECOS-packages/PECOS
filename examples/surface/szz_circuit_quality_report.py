"""Report CX vs SZZ/SZZdg surface-memory circuit quality metrics.

This script is intentionally descriptive rather than a threshold sweep. It
counts the gate locations that drive the simple circuit-level noise model and,
optionally, compares raw PECOS and Stim DEMs for the traced-QIS circuit path.
For SZZ/SZZdg cases the staged device model treats Z/SZ/SZdg frame updates as
noiseless virtual operations, so p1 location comparisons include that
assumption as well as the gate-basis change.

Example:
    uv run python examples/surface/szz_circuit_quality_report.py \\
        --distances 3 5 --bases X Z --interaction-bases cx szz
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dem_decomposition_diagnostics import compare_raw_dems, dem_stats
from pecos._traced_circuit import normalize_traced_tick_circuit
from pecos.qec.surface import NoiseModel, OpType, SurfacePatch, build_surface_code_circuit
from pecos.qec.surface.circuit_builder import (
    _analyze_szz_forward_flow,
    generate_dem_from_tick_circuit_via_stim,
)
from pecos.qec.surface.decode import (
    _build_surface_tick_circuit_for_native_model,
    generate_circuit_level_dem_from_builder,
)
from pecos.quantum import PHYSICAL_DURATION_META_KEY

PREP_GATES = {"PZ", "PX", "PY", "QAlloc"}
MEASUREMENT_GATES = {"MZ", "MX", "MY", "MeasureFree"}
IDLE_GATES = {"Idle", "I"}
TWO_QUBIT_GATES = {
    "CX",
    "CY",
    "CZ",
    "CH",
    "CRZ",
    "SXX",
    "SXXdg",
    "SYY",
    "SYYdg",
    "SZZ",
    "SZZdg",
    "RXX",
    "RYY",
    "RZZ",
    "SWAP",
    "ISWAP",
}
THREE_QUBIT_GATES = {"CCX", "CCZ"}
SZZ_ABSTRACT_PREFIX_P1_FREE_GATES = {"Z", "SZ", "SZdg"}
SZZ_Z_FRAME_P1_FREE_SOURCES = {"abstract_physical_prefix", "traced_qis"}
# SZZ/SZZdg surface diagnostics model Z-frame gates as virtual and p1-free.
SZZ_Z_FRAME_P1_GATE_RATES = {"Z": 0.0, "SZ": 0.0, "SZdg": 0.0}


@dataclass(frozen=True)
class TickCircuitStats:
    source: str
    build_s: float
    total_ticks: int
    nonempty_ticks: int
    gate_batches: int
    gate_locations: int
    prep_locations: int
    measurement_locations: int
    single_qubit_locations: int
    p1_model_locations: int
    p1_exempt_locations: int
    two_qubit_locations: int
    idle_locations: int
    zero_duration_locations: int
    max_tick_width: int
    gate_counts: dict[str, int]
    first_order_fault_mass: dict[str, float]


@dataclass(frozen=True)
class AbstractCircuitStats:
    step_counts: dict[str, int]
    szz_forward_flow: dict[str, Any] | None


@dataclass(frozen=True)
class DemReport:
    native_stats: dict[str, Any]
    stim_stats: dict[str, Any]
    native_vs_stim: dict[str, Any]
    native_build_s: float
    stim_build_s: float


@dataclass(frozen=True)
class CaseReport:
    distance: int
    rounds: int
    basis: str
    interaction_basis: str
    p: float
    p1_ratio: float
    abstract: AbstractCircuitStats
    tick_circuits: list[TickCircuitStats]
    dem: DemReport | None


def _gate_type_name(gate: Any) -> str:
    gate_type = getattr(gate, "gate_type", "")
    return str(getattr(gate_type, "name", gate_type)).rsplit(".", maxsplit=1)[-1]


def _gate_location_count(gate_name: str, qubits: list[int]) -> int:
    if gate_name in TWO_QUBIT_GATES:
        return len(qubits) // 2
    if gate_name in THREE_QUBIT_GATES:
        return len(qubits) // 3
    return len(qubits)


def _is_zero_duration(tick: Any, gate_index: int) -> bool:
    value = tick.get_gate_attr(gate_index, PHYSICAL_DURATION_META_KEY)
    return value == 0


def _noise_fault_mass(
    *,
    p: float,
    p1_ratio: float,
    p1_locations: int,
    p2_locations: int,
    prep_locations: int,
    measurement_locations: int,
    idle_locations: int,
    p_idle: float,
) -> dict[str, float]:
    p1 = p / p1_ratio
    p_prep = p / 3.0
    p_meas = p / 3.0
    return {
        "p1": p1_locations * p1,
        "p2": p2_locations * p,
        "prep": prep_locations * p_prep,
        "measurement": measurement_locations * p_meas,
        "idle": idle_locations * p_idle,
        "total": p1_locations * p1
        + p2_locations * p
        + prep_locations * p_prep
        + measurement_locations * p_meas
        + idle_locations * p_idle,
    }


def _tick_circuit_stats(
    *,
    source: str,
    tick_circuit: Any,
    build_s: float,
    interaction_basis: str,
    p: float,
    p1_ratio: float,
    p_idle: float,
) -> TickCircuitStats:
    gate_counts: Counter[str] = Counter()
    total_locations = 0
    gate_batches = 0
    prep_locations = 0
    measurement_locations = 0
    single_qubit_locations = 0
    p1_exempt_locations = 0
    two_qubit_locations = 0
    idle_locations = 0
    zero_duration_locations = 0
    nonempty_ticks = 0
    max_tick_width = 0
    p1_exempt_names = (
        SZZ_ABSTRACT_PREFIX_P1_FREE_GATES
        if source in SZZ_Z_FRAME_P1_FREE_SOURCES and interaction_basis == "szz"
        else set()
    )

    for tick_index in range(int(tick_circuit.num_ticks())):
        tick = tick_circuit.get_tick(tick_index)
        if tick is None or tick.is_empty():
            continue
        nonempty_ticks += 1
        tick_width = 0
        for gate_index, gate in enumerate(tick.gate_batches()):
            gate_name = _gate_type_name(gate)
            qubits = [int(qubit) for qubit in getattr(gate, "qubits", [])]
            locations = _gate_location_count(gate_name, qubits)
            gate_batches += 1
            total_locations += locations
            tick_width += locations
            gate_counts[gate_name] += locations
            if _is_zero_duration(tick, gate_index):
                zero_duration_locations += locations
            if gate_name in PREP_GATES:
                prep_locations += locations
            elif gate_name in MEASUREMENT_GATES:
                measurement_locations += locations
            elif gate_name in IDLE_GATES:
                idle_locations += locations
            elif gate_name in TWO_QUBIT_GATES or gate_name in THREE_QUBIT_GATES:
                two_qubit_locations += locations
            else:
                single_qubit_locations += locations
                if gate_name in p1_exempt_names or _is_zero_duration(tick, gate_index):
                    p1_exempt_locations += locations
        max_tick_width = max(max_tick_width, tick_width)

    p1_model_locations = single_qubit_locations - p1_exempt_locations
    return TickCircuitStats(
        source=source,
        build_s=build_s,
        total_ticks=int(tick_circuit.num_ticks()),
        nonempty_ticks=nonempty_ticks,
        gate_batches=gate_batches,
        gate_locations=total_locations,
        prep_locations=prep_locations,
        measurement_locations=measurement_locations,
        single_qubit_locations=single_qubit_locations,
        p1_model_locations=p1_model_locations,
        p1_exempt_locations=p1_exempt_locations,
        two_qubit_locations=two_qubit_locations,
        idle_locations=idle_locations,
        zero_duration_locations=zero_duration_locations,
        max_tick_width=max_tick_width,
        gate_counts=dict(sorted(gate_counts.items())),
        first_order_fault_mass=_noise_fault_mass(
            p=p,
            p1_ratio=p1_ratio,
            p1_locations=p1_model_locations,
            p2_locations=two_qubit_locations,
            prep_locations=prep_locations,
            measurement_locations=measurement_locations,
            idle_locations=idle_locations,
            p_idle=p_idle,
        ),
    )


def _timed(callback: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = callback()
    return value, time.perf_counter() - start


def _abstract_stats(patch: SurfacePatch, rounds: int, basis: str, interaction_basis: str) -> AbstractCircuitStats:
    ops, _allocation = build_surface_code_circuit(
        patch,
        rounds,
        basis=basis,
        interaction_basis=interaction_basis,
    )
    step_counts = Counter(op.op_type.name for op in ops if op.op_type not in {OpType.COMMENT, OpType.TICK})
    flow = None
    if interaction_basis == "szz":
        summary = _analyze_szz_forward_flow(ops)
        flow = asdict(summary)
        flow.pop("pulses", None)
    return AbstractCircuitStats(step_counts=dict(sorted(step_counts.items())), szz_forward_flow=flow)


def _build_tick_view(
    *,
    patch: SurfacePatch,
    rounds: int,
    basis: str,
    interaction_basis: str,
    source: str,
) -> tuple[Any, float]:
    circuit_source = "abstract"
    szz_physical_prefixes = False
    if source == "traced_qis":
        circuit_source = "traced_qis"
    elif source == "abstract_physical_prefix":
        if interaction_basis != "szz":
            msg = "abstract_physical_prefix is only meaningful for interaction_basis='szz'"
            raise ValueError(msg)
        szz_physical_prefixes = True
    elif source != "abstract":
        msg = f"unknown tick circuit source {source!r}"
        raise ValueError(msg)

    tick_circuit, elapsed = _timed(
        lambda: _build_surface_tick_circuit_for_native_model(
            patch,
            rounds,
            basis,
            circuit_source=circuit_source,
            interaction_basis=interaction_basis,
            szz_physical_prefixes=szz_physical_prefixes,
        ),
    )
    if source == "traced_qis":
        normalize_traced_tick_circuit(tick_circuit, context="SZZ circuit quality report")
    return tick_circuit, elapsed


def _dem_report(
    *,
    patch: SurfacePatch,
    rounds: int,
    basis: str,
    interaction_basis: str,
    tick_circuit: Any,
    p: float,
    p1_ratio: float,
) -> DemReport:
    noise = NoiseModel(p1=p / p1_ratio, p2=p, p_prep=p / 3.0, p_meas=p / 3.0)
    noise_args = {
        "p1": noise.p1,
        "p1_gate_rates": SZZ_Z_FRAME_P1_GATE_RATES if interaction_basis == "szz" else None,
        "p2": noise.p2,
        "p_prep": noise.p_prep,
        "p_meas": noise.p_meas,
    }
    native_dem, native_build_s = _timed(
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
    stim_dem, stim_build_s = _timed(
        lambda: generate_dem_from_tick_circuit_via_stim(
            tick_circuit,
            decompose_errors=False,
            **noise_args,
        ),
    )
    return DemReport(
        native_stats=asdict(dem_stats(native_dem)),
        stim_stats=asdict(dem_stats(stim_dem)),
        native_vs_stim=asdict(compare_raw_dems(native_dem, stim_dem)),
        native_build_s=native_build_s,
        stim_build_s=stim_build_s,
    )


def build_case(
    *,
    distance: int,
    rounds: int,
    basis: str,
    interaction_basis: str,
    p: float,
    p1_ratio: float,
    p_idle: float,
    sources: list[str],
    include_dem: bool,
) -> CaseReport:
    patch = SurfacePatch.create(distance=distance)
    abstract = _abstract_stats(patch, rounds, basis, interaction_basis)
    tick_stats: list[TickCircuitStats] = []
    traced_tick_circuit = None

    print(f"\n=== d={distance} r={rounds} basis={basis} basis2q={interaction_basis} ===", flush=True)
    for source in sources:
        tick_circuit, build_s = _build_tick_view(
            patch=patch,
            rounds=rounds,
            basis=basis,
            interaction_basis=interaction_basis,
            source=source,
        )
        if source == "traced_qis":
            traced_tick_circuit = tick_circuit
        stats = _tick_circuit_stats(
            source=source,
            tick_circuit=tick_circuit,
            build_s=build_s,
            interaction_basis=interaction_basis,
            p=p,
            p1_ratio=p1_ratio,
            p_idle=p_idle,
        )
        tick_stats.append(stats)
        mass = stats.first_order_fault_mass
        print(
            f"{source:24} ticks={stats.nonempty_ticks:5d}/{stats.total_ticks:<5d} "
            f"p1_locs={stats.p1_model_locations:5d} "
            f"p2_locs={stats.two_qubit_locations:5d} "
            f"prep={stats.prep_locations:5d} meas={stats.measurement_locations:5d} "
            f"mass={mass['total']:.6g} "
            f"build={stats.build_s:.3f}s",
            flush=True,
        )

    dem = None
    if include_dem:
        if traced_tick_circuit is None:
            traced_tick_circuit, _build_s = _build_tick_view(
                patch=patch,
                rounds=rounds,
                basis=basis,
                interaction_basis=interaction_basis,
                source="traced_qis",
            )
        print("building traced-QIS raw DEM comparison...", flush=True)
        dem = _dem_report(
            patch=patch,
            rounds=rounds,
            basis=basis,
            interaction_basis=interaction_basis,
            tick_circuit=traced_tick_circuit,
            p=p,
            p1_ratio=p1_ratio,
        )
        comparison = dem.native_vs_stim
        print(
            "raw native vs Stim: "
            f"only_native={comparison['only_native']} only_stim={comparison['only_stim']} "
            f"max_rel={comparison['max_rel_probability_diff']:.3e}",
            flush=True,
        )
        if (
            comparison["only_native"] == 0
            and comparison["only_stim"] == 0
            and comparison["max_rel_probability_diff"] > 0
        ):
            print(
                "raw structures match; max_rel is a probability-combination/rounding delta.",
                flush=True,
            )

    return CaseReport(
        distance=distance,
        rounds=rounds,
        basis=basis,
        interaction_basis=interaction_basis,
        p=p,
        p1_ratio=p1_ratio,
        abstract=abstract,
        tick_circuits=tick_stats,
        dem=dem,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--rounds", type=int, default=None, help="Rounds to use. Defaults to distance.")
    parser.add_argument("--bases", nargs="+", choices=["X", "Z"], default=["X", "Z"])
    parser.add_argument("--interaction-bases", nargs="+", choices=["cx", "szz"], default=["cx", "szz"])
    parser.add_argument("--sources", nargs="+", choices=["abstract", "abstract_physical_prefix", "traced_qis"])
    parser.add_argument("--p", type=float, default=0.006)
    parser.add_argument("--p1-ratio", type=float, default=30.0, help="Use p1=p/p1_ratio.")
    parser.add_argument("--p-idle", type=float, default=0.0, help="Optional idle rate for the rough mass estimate.")
    parser.add_argument("--include-dem", action="store_true", help="Also compare traced-QIS raw native and Stim DEMs.")
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def _default_sources(interaction_basis: str) -> list[str]:
    if interaction_basis == "szz":
        return ["abstract", "abstract_physical_prefix", "traced_qis"]
    return ["abstract", "traced_qis"]


def main() -> int:
    args = parse_args()
    results: list[CaseReport] = []
    for distance in args.distances:
        rounds = args.rounds if args.rounds is not None else distance
        for basis in args.bases:
            for interaction_basis in args.interaction_bases:
                sources = args.sources if args.sources is not None else _default_sources(interaction_basis)
                results.append(
                    build_case(
                        distance=distance,
                        rounds=rounds,
                        basis=basis,
                        interaction_basis=interaction_basis,
                        p=args.p,
                        p1_ratio=args.p1_ratio,
                        p_idle=args.p_idle,
                        sources=sources,
                        include_dem=args.include_dem,
                    ),
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
