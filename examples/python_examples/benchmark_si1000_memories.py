"""Compare PECOS and released SI1000 memories with the paper's Tesseract settings.

Source: Shutty, arXiv:2609.21376v3, DOI 10.5281/zenodo.22820614. This runs new
samples; it does not reuse the released shots or claim published-curve precision.
Use a separately built CLI at source revision 024db1d3b5b038f565c476dd1b51885271f7b0bf.
A provenance JSON must bind that source revision to the executable SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import stim
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.generic import SI1000, evaluate_tick_circuit
from pecos.qec.memory_benchmark import MemoryCounts, equal_weight_memory_estimate
from pecos.qec.surface import tick_circuit_to_stim

if TYPE_CHECKING:
    from pecos.qec.generic import NoisyScheduledCircuit
    from pecos.quantum import TickCircuit

SOURCE_COMMIT = "024db1d3b5b038f565c476dd1b51885271f7b0bf"
FIXTURES = Path(__file__).resolve().parents[2] / "python/quantum-pecos/tests/qec/fixtures/si1000"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _noisy_case(name: str) -> NoisyScheduledCircuit:
    topology, _, distance, rounds, basis, phase, rate = name.split("_")
    d, r = int(distance[1:]), int(rounds[1:])
    if topology == "planar":
        impl = color_code_implementation(color_code_spec("488", d), ColorExtraction.BRICKWORK_CZ)
        plan = color_memory(impl, r, basis, start_phase=int(phase[5:]))
    else:
        impl = periodic_488_implementation(periodic_488_code(d), gate_set="native-cz")
        plan = periodic_488_memory(impl, r, basis)
    return SI1000(float(rate[1:].replace("p", "."))).apply(plan)


def _constructed(name: str) -> stim.Circuit:
    return stim.Circuit(tick_circuit_to_stim(_noisy_case(name).to_tick_circuit()))


def write_native_samples(
    noisy: NoisyScheduledCircuit,
    ticks: TickCircuit,
    path: Path,
    *,
    shots: int,
    seed: int,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Write native detector/observable deviations in Stim's appended 01 format.

    Subtract an ideal native reference, so authored affine signs are not counted
    as noise. DEM preflight in the caller separately establishes determinism.
    """
    from pecos_rslib_exp import monte_carlo, sim_neo, stabilizer

    reference_record = SI1000(0).apply(noisy.ideal).compile_native().sample_records(1, seed=0)[0]
    reference = noisy.ideal.evaluate(reference_record)
    rows = sim_neo(ticks).quantum(stabilizer()).sampling(monte_carlo(shots)).seed(seed).run().to_list()
    with path.open("w") as stream:
        for record in rows:
            values = evaluate_tick_circuit(ticks, record)
            deviations = [
                value ^ base
                for group, ideal in zip(values, reference, strict=True)
                for value, base in zip(group, ideal, strict=True)
            ]
            stream.write("".join(str(int(bit)) for bit in deviations) + "\n")
    return reference


def _mechanisms(dem: stim.DetectorErrorModel) -> dict:
    result = {}
    for instruction in dem.flattened():
        if instruction.type == "error":
            key = tuple(str(t) for t in instruction.targets_copy())
            p = instruction.args_copy()[0]
            old = result.get(key, 0.0)
            result[key] = old * (1 - p) + p * (1 - old)
    return result


def _native_worker(request_path: Path) -> None:
    """Sample in an isolated process that the parent can stop at its deadline."""
    request = json.loads(request_path.read_text())
    noisy = _noisy_case(request["case"])
    ticks = noisy.to_tick_circuit()
    circuit_text = str(stim.Circuit(tick_circuit_to_stim(ticks)))
    if hashlib.sha256(circuit_text.encode()).hexdigest() != request["circuit_sha256"]:
        msg = "Native worker circuit differs from the preflight circuit"
        raise ValueError(msg)
    reference = write_native_samples(
        noisy,
        ticks,
        Path(request["samples_path"]),
        shots=request["shots"],
        seed=request["seed"],
    )
    extensions = {}
    for name in ("pecos_rslib.pecos_rslib", "pecos_rslib_exp.pecos_rslib_exp"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            msg = f"Cannot record native extension provenance: {name}"
            raise RuntimeError(msg)
        path = Path(spec.origin)
        extensions[name] = {"path": str(path), "sha256": _sha(path)}
    Path(request["receipt_path"]).write_text(
        json.dumps(
            {
                "ideal_reference": reference,
                "native_extensions": extensions,
            },
            indent=2,
        ),
    )


def group_estimates(rows: list[dict]) -> dict:
    """Report only complete basis/phase groups using the released estimator."""
    groups = {}
    for row in rows:
        topology, _, distance, rounds, basis, phase, rate = row["case"].split("_")
        key = (row["variant"], topology + "_488", int(distance[1:]), int(rounds[1:]), rate)
        groups.setdefault(key, {})[basis, int(phase[5:])] = row
    complete, incomplete = [], []
    for (variant, family, distance, rounds, rate), strata in sorted(groups.items()):
        expected = {(basis, phase) for basis in "XZ" for phase in ((0, 1) if family == "planar_488" else (0,))}
        identity = {
            "variant": variant,
            "family": family,
            "distance": distance,
            "rounds": rounds,
            "p": float(rate[1:].replace("p", ".")),
        }
        if set(strata) != expected:
            incomplete.append({**identity, "missing": sorted(expected - set(strata))})
            continue
        estimate = equal_weight_memory_estimate(
            [MemoryCounts(**strata[key]["counts"]) for key in sorted(strata)],
            num_logicals=1 if family == "planar_488" else 4,
            rounds=rounds,
        )
        complete.append({**identity, "estimate": asdict(estimate)})
    return {"complete_groups": complete, "incomplete_groups": incomplete}


def main() -> None:
    """Prepare, verify, and decode independent fixed-shot samples on both paths."""
    if len(sys.argv) == 3 and sys.argv[1] == "--native-worker":
        _native_worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tesseract", type=Path, required=True)
    parser.add_argument("--decoder-provenance", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help="Repeat to select pinned fixture stems; default d4/r4, both bases/rates",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=FIXTURES,
        help="Directory containing pinned released .stim files; supports full release sizes",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1, help="Independent decoder shot workers (default: 1)")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--sampling-backend",
        choices=("stim", "pecos"),
        default="stim",
        help="PECOS variant sampling; reference always uses pinned Tesseract/Stim",
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--wall-time-budget", type=float, help="Overall seconds, including circuit preparation")
    parser.add_argument("--output", type=Path, default=Path("output/si1000-decoder-benchmark"))
    args = parser.parse_args()
    if args.shots < 1 or args.seed < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("shots/timeout must be positive and finite, and seed nonnegative")
    if not 1 <= args.threads <= 1000:
        parser.error("threads must be between 1 and the decoder limit of 1000")
    if args.wall_time_budget is not None and (not math.isfinite(args.wall_time_budget) or args.wall_time_budget <= 0):
        parser.error("wall-time-budget must be positive and finite")
    provenance = json.loads(args.decoder_provenance.read_text())
    if provenance.get("source_commit") != SOURCE_COMMIT or provenance.get("binary_sha256") != _sha(args.tesseract):
        parser.error("decoder provenance must bind the paper's source revision to this binary")
    catalog_rows = json.loads((FIXTURES / "benchmark_circuits.json").read_text())["cases"]
    catalog = {Path(c["fixture"]).stem: c for c in catalog_rows}
    names = args.case or [name for name in catalog if name.startswith("torus_488_d4_")]
    if len(set(names)) != len(names) or any(name not in catalog for name in names):
        parser.error("cases must be distinct pinned fixture stems")
    if any(not (args.reference_dir / catalog[name]["fixture"]).is_file() for name in names):
        parser.error("missing released circuit; set --reference-dir to the release's circuits/simulated directory")
    if args.seed + 2 * len(names) - 1 >= 2**64:
        parser.error("sampling seeds must fit in uint64")
    if (args.output / "results.json").exists():
        parser.error("output already contains a report; choose a new directory to preserve its samples")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()

    def checkpoint(status: str, active: str | None = None) -> None:
        report = {
            "paper": "https://arxiv.org/abs/2609.21376v3",
            "decoder": provenance,
            "dem_generation_stim_version": stim.__version__,
            "results": rows,
            **group_estimates(rows),
            "status": status,
            "active_case": active,
            "elapsed_seconds": time.monotonic() - started,
            "wall_time_budget_seconds": args.wall_time_budget,
            "planned_cases": names,
            "planned_shots_per_variant": args.shots,
            "pecos_sampling_backend": args.sampling_backend,
            "decoder_threads": args.threads,
            "pending_runs": [
                f"{name}/{variant}"
                for name in names
                for variant in ("reference", "pecos")
                if not any(row["case"] == name and row["variant"] == variant for row in rows)
            ],
            "scope": "Fixed-shot validation with equal-stratum groups; not threshold reproduction",
        }
        temporary = args.output / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output / "results.json")

    def remaining() -> float:
        return float("inf") if args.wall_time_budget is None else args.wall_time_budget - (time.monotonic() - started)

    checkpoint("running")
    for index, name in enumerate(names):
        if remaining() <= 0:
            checkpoint("budget_exhausted", name)
            return
        source = args.reference_dir / catalog[name]["fixture"]
        if _sha(source) != catalog[name]["sha256"]:
            msg = f"Reference checksum mismatch: {name}"
            raise ValueError(msg)
        noisy = _noisy_case(name)
        ticks = noisy.to_tick_circuit()
        circuits = {
            "reference": stim.Circuit(source.read_text()),
            "pecos": stim.Circuit(str(stim.Circuit(tick_circuit_to_stim(ticks)))),
        }
        dems = {
            key: c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
            for key, c in circuits.items()
        }
        a, b = (_mechanisms(dems[key]) for key in ("reference", "pecos"))
        if a.keys() != b.keys() or any(not math.isclose(a[k], b[k], rel_tol=1e-10, abs_tol=0.0) for k in a):
            msg = f"DEM mismatch before decoding: {name}"
            raise ValueError(msg)
        # Stim serialization applies the same decimal precision as the released
        # files. Compare those decoder-visible coordinates, including time.
        if circuits["reference"].get_detector_coordinates() != circuits["pecos"].get_detector_coordinates():
            msg = f"Decoder coordinates differ: {name}"
            raise ValueError(msg)
        if dems["reference"].get_detector_coordinates() != dems["pecos"].get_detector_coordinates():
            msg = f"Decoder DEM coordinates differ: {name}"
            raise ValueError(msg)
        for variant, circuit in circuits.items():
            folder = args.output / name / variant
            folder.mkdir(parents=True, exist_ok=True)
            circuit_path, dem_path, stats_path = (folder / f for f in ("circuit.stim", "decoder.dem", "stats.json"))
            circuit_path.write_text(str(circuit))
            dem_path.write_text(str(dems[variant]))
            seed = args.seed + 2 * index + (variant == "pecos")
            command = [
                str(args.tesseract.resolve()),
                "--circuit",
                str(circuit_path),
                "--dem",
                str(dem_path),
                "--sample-num-shots",
                str(args.shots),
                "--sample-seed",
                str(seed),
                "--stats-out",
                str(stats_path),
                "--threads",
                str(args.threads),
                "--num-det-orders",
                "21",
                "--det-order-bfs",
                "--det-order-coordinate",
                "--det-order-index",
                "--det-order-seed",
                "518278944",
                "--beam",
                "20",
                "--beam-climbing",
                "--no-revisit-dets",
                "--pqlimit",
                "1000000",
            ]
            sample_metadata = {"backend": "tesseract_stim"}
            if variant == "pecos" and args.sampling_backend == "pecos":
                available = remaining()
                if available <= 0:
                    checkpoint("budget_exhausted", f"{name}/{variant}")
                    return
                samples_path = folder / "detectors-observables.01"
                request_path, receipt_path = folder / "sampling-request.json", folder / "sampling-receipt.json"
                request_path.write_text(
                    json.dumps(
                        {
                            "case": name,
                            "shots": args.shots,
                            "seed": seed,
                            "circuit_sha256": _sha(circuit_path),
                            "samples_path": str(samples_path.resolve()),
                            "receipt_path": str(receipt_path.resolve()),
                        },
                        indent=2,
                    ),
                )
                checkpoint("sampling", f"{name}/{variant}")
                try:
                    with (folder / "sampling.log").open("w") as log:
                        subprocess.run(  # noqa: S603 - current interpreter and repository worker, no shell
                            [sys.executable, str(Path(__file__).resolve()), "--native-worker", str(request_path)],
                            check=True,
                            timeout=min(args.timeout, available),
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                except subprocess.TimeoutExpired:
                    checkpoint(
                        "budget_exhausted" if available <= args.timeout else "native_sampling_timeout",
                        f"{name}/{variant}",
                    )
                    return
                except subprocess.CalledProcessError:
                    checkpoint("native_sampling_error", f"{name}/{variant}")
                    raise
                receipt = json.loads(receipt_path.read_text())
                # File input and built-in circuit sampling are mutually exclusive.
                start = command.index("--sample-num-shots")
                del command[start : start + 4]
                command.extend(["--in", str(samples_path), "--in-format", "01", "--in-includes-appended-observables"])
                sample_metadata = {
                    "backend": "pecos_sim_neo_stabilizer",
                    "samples_sha256": _sha(samples_path),
                    **receipt,
                }
            (folder / "command.json").write_text(json.dumps(command, indent=2))
            available = remaining()
            if available <= 0:
                checkpoint("budget_exhausted", f"{name}/{variant}")
                return
            checkpoint("running", f"{name}/{variant}")
            try:
                with (folder / "decoder.log").open("w") as log:
                    subprocess.run(  # noqa: S603 - hash-verified user-selected executable, no shell
                        command,
                        check=True,
                        timeout=min(args.timeout, available),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.TimeoutExpired:
                checkpoint("budget_exhausted" if available <= args.timeout else "decoder_timeout", f"{name}/{variant}")
                return
            except subprocess.CalledProcessError:
                checkpoint("decoder_error", f"{name}/{variant}")
                raise
            stats = json.loads(stats_path.read_text())
            required = {
                "num_shots": args.shots,
                "num_threads": args.threads,
                "num_low_confidence": 0,
                "num_det_orders": 63,
                "det_beam": 20,
                "beam_climbing": True,
                "no_revisit_dets": True,
                "pqlimit": 1000000,
                "det_order_seed": 518278944,
                "sparsify_errors": False,
            }
            if any(stats.get(key) != value for key, value in required.items()):
                checkpoint("invalid_decoder_result", f"{name}/{variant}")
                msg = f"Incomplete/unresolved or mismatched decoder run: {name}/{variant}"
                raise ValueError(msg)
            counts = MemoryCounts(stats["num_shots"], stats["num_errors"], stats["num_low_confidence"])
            rounds = int(name.split("_")[3][1:])
            estimate = equal_weight_memory_estimate([counts], num_logicals=circuit.num_observables, rounds=rounds)
            row = {
                "case": name,
                "variant": variant,
                "counts": asdict(counts),
                "estimate": asdict(estimate),
                "circuit_sha256": _sha(circuit_path),
                "dem_sha256": _sha(dem_path),
                "seed": seed,
                "sampling": sample_metadata,
            }
            rows.append(row)
            checkpoint("running")
            print(f"{name}/{variant}: {counts.failures}/{counts.shots} failures; unresolved=0", flush=True)
    checkpoint("complete")


if __name__ == "__main__":
    main()
