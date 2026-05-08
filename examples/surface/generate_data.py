r"""Generate decoder performance data for surface code memory experiments.

Samples detection events once per (distance, error_rate, rounds) point,
then decodes with each requested decoder. Writes a JSON shard that can
be fed to ``analyze_data.py`` and ``build_report.py``.

Example:
    uv run python examples/surface/generate_data.py \\
        --distances 3 5 --error-rates 0.004 0.008 \\
        --decoders pymatching mwpf tesseract bp_osd \\
        --shots 5000

    uv run python examples/surface/generate_data.py \\
        --distances 3 5 7 \\
        --error-rates 0.002 0.004 0.006 0.008 \\
        --decoders pymatching mwpf tesseract \\
        --duration-multipliers 2 2.5 3 \\
        --shots 2000 --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.surface import NoiseModel

# -- Data model ---------------------------------------------------------------


@dataclass
class DecoderStats:
    """Decode statistics for one decoder on one set of samples."""

    decoder: str
    num_errors: int
    logical_error_rate: float
    total_decode_seconds: float
    per_shot_mean: float
    per_shot_median: float
    per_shot_p99: float
    per_shot_max: float
    # 21 quantiles at [0%, 5%, 10%, ..., 95%, 100%] for violin plots
    quantiles: list[float] = field(default_factory=list)


@dataclass
class DataPoint:
    """Raw data for one (distance, basis, p, rounds, decoder-set) cell."""

    distance: int
    basis: str
    physical_error_rate: float
    num_rounds: int
    num_shots: int
    sample_seconds: float
    decoder_stats: list[DecoderStats] = field(default_factory=list)


@dataclass
class DataShard:
    """One complete data-generation run. Serialised to JSON."""

    config: dict
    points: list[DataPoint] = field(default_factory=list)
    total_seconds: float = 0.0


# -- DEM sets for different decoder families ----------------------------------

# MWPM decoders need decomposed (graphlike) DEMs.
_MWPM_DECODERS = {
    "pymatching",
    "pymatching_uncorrelated",
    "fusion_blossom",
    "fusion_blossom_serial",
    "fusion_blossom_parallel",
}

# Slow decoders benefit from parallel decode.
_SLOW_DECODERS = {"tesseract", "mwpf", "bp_osd", "relay_bp"}


def _decoder_base_name(name: str) -> str:
    """Strip config suffix: 'mwpf:c=30' -> 'mwpf'."""
    return name.split(":", maxsplit=1)[0]


# -- Sampler + DEM construction -----------------------------------------------


def _build_sampler(
    distance: int,
    num_rounds: int,
    noise: NoiseModel,
    basis: str,
    circuit_source: str,
) -> tuple:
    """Build native sampler and return (sampler, dem_decomposed, dem_full)."""
    from pecos.qec.surface import SurfacePatch, build_native_sampler
    from pecos.qec.surface.decode import SurfaceDecoder, generate_circuit_level_dem_from_builder

    patch = SurfacePatch.create(distance=distance)
    sampler = build_native_sampler(
        patch,
        num_rounds,
        noise,
        basis=basis,
        circuit_source=circuit_source,
    )

    # Decomposed DEM for MWPM decoders
    dec = SurfaceDecoder(
        patch,
        num_rounds=num_rounds,
        noise=noise,
        decoder_type="pymatching",
        use_circuit_level_dem=True,
        circuit_level_dem_mode="native_decomposed",
        circuit_level_dem_source=circuit_source,
    )
    dem_decomp = dec.get_dem(basis.upper(), circuit_level=True)
    dem_decomp = "\n".join(line for line in dem_decomp.split("\n") if not line.startswith("logical_observable"))

    # Full DEM for non-MWPM decoders
    dem_full = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds,
        noise,
        basis=basis,
        decompose_errors=False,
        circuit_source=circuit_source,
    )
    dem_full = "\n".join(line for line in dem_full.split("\n") if not line.startswith("logical_observable"))

    return sampler, dem_decomp, dem_full


# -- Main generation loop -----------------------------------------------------


def generate(
    *,
    distances: list[int],
    error_rates: list[float],
    decoders: list[str],
    basis: str,
    shots: int,
    seed: int,
    circuit_source: str,
    p1_scale: float,
    p_meas_scale: float,
    p_prep_scale: float,
    duration_multipliers: list[float],
) -> DataShard:
    """Run the full data generation and return a shard."""
    from pecos.qec.surface import NoiseModel

    config = {
        "distances": distances,
        "error_rates": error_rates,
        "decoders": decoders,
        "basis": basis.upper(),
        "shots": shots,
        "seed": seed,
        "circuit_source": circuit_source,
        "p1_scale": p1_scale,
        "p_meas_scale": p_meas_scale,
        "p_prep_scale": p_prep_scale,
        "duration_multipliers": duration_multipliers,
    }

    shard = DataShard(config=config)
    t_start = time.perf_counter()

    # Compute distinct round counts per distance.
    # For small d, multipliers may collide after truncation.
    # Ensure at least len(duration_multipliers) distinct round counts
    # by extending the range upward if needed.
    rounds_per_d: dict[int, list[int]] = {}
    for d in distances:
        seen = set()
        for mult in duration_multipliers:
            seen.add(max(2, int(d * mult)))
        # If we got fewer than requested, fill in consecutive integers from 2*d
        target = len(duration_multipliers)
        r_start = 2 * d
        while len(seen) < target:
            seen.add(r_start)
            r_start += 1
        rounds_per_d[d] = sorted(seen)

    total_cells = sum(len(rounds_per_d[d]) for d in distances) * len(error_rates)
    cell_idx = 0

    for d in distances:
        for p in error_rates:
            noise = NoiseModel(
                p1=p * p1_scale,
                p2=p,
                p_meas=p * p_meas_scale,
                p_prep=p * p_prep_scale,
            )

            for num_rounds in rounds_per_d[d]:
                cell_idx += 1
                print(f"[{cell_idx}/{total_cells}] d={d} p={p:.4g} r={num_rounds} ...")

                sampler, dem_decomp, dem_full = _build_sampler(
                    d,
                    num_rounds,
                    noise,
                    basis,
                    circuit_source,
                )

                # Sample once
                t0 = time.perf_counter()
                batch = sampler.sampler.generate_samples(shots, seed=seed + cell_idx)
                sample_seconds = time.perf_counter() - t0

                point = DataPoint(
                    distance=d,
                    basis=basis.upper(),
                    physical_error_rate=p,
                    num_rounds=num_rounds,
                    num_shots=shots,
                    sample_seconds=sample_seconds,
                )

                # Decode with each decoder
                for decoder_name in decoders:
                    base = _decoder_base_name(decoder_name)
                    dem = dem_decomp if base in _MWPM_DECODERS else dem_full

                    if base in _SLOW_DECODERS:
                        stats = batch.decode_stats_parallel(dem, decoder_name)
                    else:
                        stats = batch.decode_stats(dem, decoder_name)

                    point.decoder_stats.append(
                        DecoderStats(
                            decoder=decoder_name,
                            num_errors=stats.num_errors,
                            logical_error_rate=stats.logical_error_rate,
                            total_decode_seconds=stats.total_seconds,
                            per_shot_mean=stats.per_shot_mean,
                            per_shot_median=stats.per_shot_median,
                            per_shot_p99=stats.per_shot_p99,
                            per_shot_max=stats.per_shot_max,
                            quantiles=list(stats.quantiles),
                        ),
                    )

                    print(
                        f"    {decoder_name:20s}: {stats.num_errors:>4d}/{shots}  "
                        f"LER={stats.logical_error_rate:.4f}  "
                        f"median={stats.per_shot_median:.1e}s  "
                        f"p99={stats.per_shot_p99:.1e}s",
                    )

                shard.points.append(point)

    shard.total_seconds = time.perf_counter() - t_start
    return shard


# -- CLI ----------------------------------------------------------------------


def main() -> int:
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--error-rates", nargs="+", type=float, default=[0.004, 0.006, 0.008])
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["pymatching", "mwpf", "tesseract", "bp_osd"],
        help="Decoders to run. Use 'mwpf:c=30,t=0.5' for config overrides.",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--basis", default="Z")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--circuit-source", default="traced_qis", choices=["traced_qis", "abstract"])
    parser.add_argument("--p1-scale", type=float, default=1.0 / 30.0)
    parser.add_argument("--p-meas-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--p-prep-scale", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--duration-multipliers",
        nargs="+",
        type=float,
        default=[2.0],
        help="Round count = distance * multiplier. Use multiple for threshold fitting.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("PECOS Data Generation")
    print("=" * 40)
    for k, v in vars(args).items():
        if k != "output_dir":
            print(f"  {k}: {v}")
    print()

    shard = generate(
        distances=sorted(args.distances),
        error_rates=sorted(args.error_rates),
        decoders=args.decoders,
        basis=args.basis,
        shots=args.shots,
        seed=args.seed,
        circuit_source=args.circuit_source,
        p1_scale=args.p1_scale,
        p_meas_scale=args.p_meas_scale,
        p_prep_scale=args.p_prep_scale,
        duration_multipliers=sorted(args.duration_multipliers),
    )

    print(f"\nTotal time: {shard.total_seconds:.1f}s")

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        import tempfile

        out = Path(tempfile.mkdtemp(prefix="pecos_data_"))

    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "data.json"
    json_path.write_text(json.dumps(asdict(shard), indent=2))
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
