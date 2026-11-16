"""Measure decoder worker scaling on common saved samples, without changing its settings.

Consumes a completed case from benchmark_si1000_memories.py. This is a timing
experiment, not additional independent logical-error-rate evidence. The pinned
Shutty decoder configuration is retained (arXiv:2609.21376v3,
https://doi.org/10.5281/zenodo.22820614).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import stim

SOURCE_COMMIT = "024db1d3b5b038f565c476dd1b51885271f7b0bf"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    circuit_path = Path(request["circuit_path"])
    if _sha(circuit_path) != request["circuit_sha256"]:
        msg = "Sampling circuit changed after input validation"
        raise ValueError(msg)
    stim.Circuit(circuit_path.read_text()).compile_detector_sampler(seed=request["seed"]).sample_write(
        shots=request["shots"],
        filepath=request["samples_path"],
        format="01",
        append_observables=True,
    )


def scaling_summary(rows: list[dict], threads: list[int], repetitions: int) -> dict:
    """Compare only complete repeated measurements with identical predictions."""
    expected = {(worker, repeat) for worker in threads for repeat in range(repetitions)}
    successful = [row for row in rows if row["status"] == "complete"]
    identities = [(row["threads"], row["repetition"]) for row in successful]
    complete = len(identities) == len(expected) and set(identities) == expected
    identical = bool(successful) and len({row["predictions_sha256"] for row in successful}) == 1
    if not complete or not identical:
        return {
            "comparison_complete": False,
            "predictions_identical": identical if complete else None,
            "speedups": [],
            "reason": "All planned runs must complete with identical predictions",
        }
    medians = {
        worker: statistics.median(row["wall_seconds"] for row in successful if row["threads"] == worker)
        for worker in threads
    }
    return {
        "comparison_complete": True,
        "predictions_identical": True,
        "speedups": [
            {
                "threads": worker,
                "median_wall_seconds": medians[worker],
                "speedup_over_serial": medians[1] / medians[worker],
                "shots_per_wall_second": successful[0]["shots"] / medians[worker],
            }
            for worker in threads
        ],
    }


def main() -> None:
    """Validate a completed benchmark artifact and profile identical new samples."""
    if len(sys.argv) == 3 and sys.argv[1] == "--sample-worker":
        _sample_worker(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="Directory containing results.json")
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", choices=("reference", "pecos"), default="pecos")
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261171)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--wall-time-budget", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shots < 1 or args.repetitions < 1 or not 0 <= args.seed < 2**64:
        parser.error("shots/repetitions must be positive and seed must fit uint64")
    if (
        1 not in args.threads
        or len(set(args.threads)) != len(args.threads)
        or any(not 1 <= n <= 1000 for n in args.threads)
    ):
        parser.error("threads must be distinct values in 1..1000, including a serial (1) control")
    if any(not math.isfinite(n) or n <= 0 for n in (args.timeout, args.wall_time_budget)):
        parser.error("timeout and wall-time-budget must be positive and finite")
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory to preserve the experiment")
    benchmark = json.loads((args.benchmark / "results.json").read_text())
    matches = [row for row in benchmark["results"] if row["case"] == args.case and row["variant"] == args.variant]
    if len(matches) != 1:
        parser.error("case/variant must identify one completed benchmark row")
    source = args.benchmark / args.case / args.variant
    command = json.loads((source / "command.json").read_text())
    binary = Path(command[0])
    if benchmark["decoder"]["source_commit"] != SOURCE_COMMIT or _sha(binary) != benchmark["decoder"]["binary_sha256"]:
        parser.error("decoder binary/source provenance mismatch")
    for name, key in (("circuit.stim", "circuit_sha256"), ("decoder.dem", "dem_sha256")):
        if _sha(source / name) != matches[0][key]:
            parser.error(f"modified input: {name}")
    # Accept only the recipe emitted by our benchmark, then construct fresh argv.
    fixed = {"--num-det-orders": "21", "--det-order-seed": "518278944", "--beam": "20", "--pqlimit": "1000000"}
    flags = {"--det-order-bfs", "--det-order-coordinate", "--det-order-index", "--beam-climbing", "--no-revisit-dets"}
    if any(command.count(key) != 1 or command[command.index(key) + 1] != value for key, value in fixed.items()):
        parser.error("input command does not use the pinned decoder configuration")
    if not flags.issubset(command) or "--sparsify-errors" in command:
        parser.error("input command does not use the pinned decoder configuration")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    circuit = stim.Circuit((source / "circuit.stim").read_text())
    samples = args.output / "common-samples.01"
    provenance = {
        "benchmark_report_sha256": _sha(args.benchmark / "results.json"),
        "driver_sha256": _sha(Path(__file__)),
        "python_version": sys.version,
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "case": args.case,
        "variant": args.variant,
        "circuit_sha256": matches[0]["circuit_sha256"],
        "dem_sha256": matches[0]["dem_sha256"],
        "decoder": benchmark["decoder"],
        "sampling_stim_version": stim.__version__,
        "sampling_seed": args.seed,
        "samples_sha256": None,
        "shots": args.shots,
        "threads": args.threads,
        "repetitions": args.repetitions,
        "wall_time_budget_seconds": args.wall_time_budget,
        "per_run_timeout_seconds": args.timeout,
        "scope": "Common-sample throughput probe; repeated predictions are not new independent shots",
        "sampling_scope": "New Python Stim samples; not a replay of the earlier C++ sampler seed stream",
    }
    rows = []

    def checkpoint(status: str, active: dict | None = None) -> None:
        temporary = args.output / "profile.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    **provenance,
                    "status": status,
                    "active_run": active,
                    "elapsed_seconds": time.monotonic() - started,
                    "runs": rows,
                    **scaling_summary(rows, args.threads, args.repetitions),
                },
                indent=2,
            ),
        )
        temporary.replace(args.output / "profile.json")

    checkpoint("sampling")
    request_path = args.output / "sampling-request.json"
    request_path.write_text(
        json.dumps(
            {
                "circuit_path": str((source / "circuit.stim").resolve()),
                "circuit_sha256": matches[0]["circuit_sha256"],
                "samples_path": str(samples.resolve()),
                "seed": args.seed,
                "shots": args.shots,
            },
            indent=2,
        ),
    )
    remaining = args.wall_time_budget - (time.monotonic() - started)
    if remaining <= 0:
        checkpoint("budget_exhausted")
        return
    try:
        with (args.output / "sampling.log").open("w") as log:
            subprocess.run(  # noqa: S603 - current interpreter and repository worker, no shell
                [sys.executable, str(Path(__file__).resolve()), "--sample-worker", str(request_path.resolve())],
                check=True,
                timeout=min(args.timeout, remaining),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except subprocess.TimeoutExpired:
        checkpoint("budget_exhausted" if remaining <= args.timeout else "sampling_timeout")
        return
    except subprocess.CalledProcessError:
        checkpoint("sampling_error")
        raise
    provenance["samples_sha256"] = _sha(samples)
    checkpoint("running")
    for repeat in range(args.repetitions):
        # Reverse order on alternating passes to reduce fixed ordering effects.
        for worker in args.threads if repeat % 2 == 0 else args.threads[::-1]:
            remaining = args.wall_time_budget - (time.monotonic() - started)
            if remaining <= 0:
                checkpoint("budget_exhausted")
                return
            folder = args.output / f"repeat-{repeat}-threads-{worker}"
            folder.mkdir()
            stats_path, predictions = folder / "stats.json", folder / "predictions.01"
            argv = [
                str(binary.resolve()),
                "--circuit",
                str((source / "circuit.stim").resolve()),
                "--dem",
                str((source / "decoder.dem").resolve()),
                "--in",
                str(samples.resolve()),
                "--in-format",
                "01",
                "--in-includes-appended-observables",
                "--threads",
                str(worker),
                "--stats-out",
                str(stats_path.resolve()),
                "--out",
                str(predictions.resolve()),
                "--out-format",
                "01",
                "--print-stats",
            ]
            for key, value in fixed.items():
                argv.extend([key, value])
            argv.extend(sorted(flags))
            (folder / "command.json").write_text(json.dumps(argv, indent=2))
            identity = {"threads": worker, "repetition": repeat, "shots": args.shots}
            checkpoint("running", identity)
            run_started = time.monotonic()
            try:
                with (folder / "decoder.log").open("w") as log:
                    subprocess.run(  # noqa: S603 - hash-verified decoder with constructed arguments, no shell
                        argv,
                        check=True,
                        timeout=min(remaining, args.timeout),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.TimeoutExpired:
                rows.append({**identity, "status": "timeout", "wall_seconds": time.monotonic() - run_started})
                checkpoint("running")
                continue
            except subprocess.CalledProcessError:
                checkpoint("decoder_error", identity)
                raise
            elapsed = time.monotonic() - run_started
            try:
                stats = json.loads(stats_path.read_text())
            except (OSError, ValueError):
                checkpoint("invalid_decoder_result", identity)
                raise
            required = {
                "num_shots": args.shots,
                "num_threads": worker,
                "num_low_confidence": 0,
                "num_det_orders": 63,
                "det_beam": 20,
                "det_order_seed": 518278944,
                "pqlimit": 1000000,
                "beam_climbing": True,
                "no_revisit_dets": True,
                "sparsify_errors": False,
            }
            shot_time = stats.get("total_time_seconds") if isinstance(stats, dict) else None
            if (
                not isinstance(stats, dict)
                or any(stats.get(key) != value for key, value in required.items())
                or type(shot_time) not in (int, float)
                or not math.isfinite(shot_time)
                or shot_time < 0
            ):
                checkpoint("invalid_decoder_result", identity)
                msg = "Incomplete, unresolved, or mismatched decoder result"
                raise ValueError(msg)
            try:
                lines = predictions.read_text().splitlines()
            except OSError:
                checkpoint("invalid_predictions", identity)
                raise
            if len(lines) != args.shots or any(
                len(line) != circuit.num_observables or set(line) - {"0", "1"} for line in lines
            ):
                checkpoint("invalid_predictions", identity)
                msg = "Decoder prediction count/width differs from the input"
                raise ValueError(msg)
            rows.append(
                {
                    **identity,
                    "status": "complete",
                    "wall_seconds": elapsed,
                    "summed_shot_seconds": stats["total_time_seconds"],
                    "predictions_sha256": _sha(predictions),
                },
            )
            checkpoint("running")
            print(
                f"repeat={repeat} threads={worker}: {elapsed:.3f}s wall, "
                f"{stats['total_time_seconds']:.3f}s summed shots",
                flush=True,
            )
    summary = scaling_summary(rows, args.threads, args.repetitions)
    checkpoint("complete" if summary["comparison_complete"] else "inconclusive")


if __name__ == "__main__":
    main()
