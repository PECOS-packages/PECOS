#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Build Frontier Wasm with an optional DEM and embedded hardware shots."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "pecos-frontier-wasm"
TARGET = "wasm32-unknown-unknown"
MODULE = "pecos_frontier_wasm.wasm"
GETRANDOM_CFG = '--cfg getrandom_backend="unsupported"'
REPLAY_MAGIC = b"FWR1"
REPLAY_VERSION = 2
MAX_OBSERVABLES = 128


def cargo_executable() -> str:
    """Return cargo from PATH or fail with a useful message."""
    cargo = shutil.which("cargo")
    if cargo is None:
        message = "cargo was not found on PATH"
        raise RuntimeError(message)
    return cargo


def target_directory(cargo: str) -> Path:
    """Ask Cargo for its configured target directory."""
    result = subprocess.run(
        [cargo, "metadata", "--no-deps", "--format-version", "1"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(json.loads(result.stdout)["target_directory"])


def pack_replay(
    corpus_path: Path,
    destination: Path,
    dem: str,
    max_shots: int | None = None,
) -> dict[str, int]:
    """Validate a PECOS corpus and pack its shots into an FWR1 v2 fixture."""
    from pecos.qec import SampleBatch

    batch = SampleBatch.load(corpus_path.resolve(strict=True))
    if batch.dem != dem:
        message = "corpus DEM does not exactly match the model being compiled"
        raise ValueError(message)
    if max_shots is not None and max_shots <= 0:
        message = "max_shots must be positive"
        raise ValueError(message)
    if batch.num_observables > MAX_OBSERVABLES:
        message = f"fixture exceeds the {MAX_OBSERVABLES}-observable result ABI"
        raise ValueError(message)
    # SampleBatch exposes detector width through each syndrome. Empty corpora
    # contain no useful replay work and cannot provide this through the API.
    if batch.num_shots == 0:
        message = "replay corpus must contain at least one shot"
        raise ValueError(message)
    detectors = len(batch.get_syndrome(0))
    observables = batch.num_observables
    shots = batch.num_shots if max_shots is None else min(batch.num_shots, max_shots)
    detector_bytes = ((detectors + 31) // 32) * 4
    observable_bytes = ((observables + 31) // 32) * 4
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(struct.pack("<4sIIII", REPLAY_MAGIC, REPLAY_VERSION, shots, detectors, observables))
        for shot in range(shots):
            syndrome = batch.get_syndrome(shot)
            mask = sum(int(bit) << index for index, bit in enumerate(syndrome))
            output.write(mask.to_bytes(detector_bytes, "little"))
            output.write(batch.get_observable_flips(shot).mask.to_bytes(observable_bytes, "little"))
    return {"shots": shots, "detectors": detectors, "observables": observables}


def build(
    dem_path: Path | None,
    replay_path: Path | None = None,
    corpus_path: Path | None = None,
    max_shots: int | None = None,
    replay_output: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, int] | None]:
    """Build and copy the module, returning the distribution path."""
    cargo = cargo_executable()
    target_dir = target_directory(cargo)
    environment = os.environ.copy()
    environment["RUSTFLAGS"] = " ".join(value for value in (environment.get("RUSTFLAGS"), GETRANDOM_CFG) if value)
    resolved_dem = dem_path or REPO_ROOT / "exp" / PACKAGE / "model.dem"
    if dem_path is None:
        environment.pop("FRONTIER_DEM_PATH", None)
    else:
        resolved_dem = dem_path.resolve(strict=True)
        if not resolved_dem.is_file():
            message = f"DEM path is not a file: {resolved_dem}"
            raise ValueError(message)
        environment["FRONTIER_DEM_PATH"] = str(resolved_dem)

    replay_metadata = None
    with tempfile.TemporaryDirectory(prefix="frontier-wasm-") as temporary_directory:
        if replay_path is not None:
            resolved_replay = replay_path.resolve(strict=True)
            if not resolved_replay.is_file():
                message = f"replay fixture is not a file: {resolved_replay}"
                raise ValueError(message)
            environment["FRONTIER_REPLAY_PATH"] = str(resolved_replay)
        elif corpus_path is not None:
            packed_replay = Path(temporary_directory) / "replay.fwr"
            replay_metadata = pack_replay(
                corpus_path,
                packed_replay,
                resolved_dem.read_text(encoding="utf-8"),
                max_shots,
            )
            environment["FRONTIER_REPLAY_PATH"] = str(packed_replay)
            if replay_output is not None:
                resolved_output = replay_output.resolve()
                resolved_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(packed_replay, resolved_output)
        else:
            environment.pop("FRONTIER_REPLAY_PATH", None)

        features = ["--features", "replay"] if replay_path is not None or corpus_path is not None else []
        subprocess.run(
            [cargo, "build", "--release", "--target", TARGET, "-p", PACKAGE, *features],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )

    source = target_dir / TARGET / "release" / MODULE
    destination = output_path.resolve() if output_path is not None else REPO_ROOT / "dist" / MODULE
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination, replay_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem",
        nargs="?",
        help="flattened Stim detector error model to embed",
    )
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument("--replay", type=Path, help="prepacked .fwr hardware-shot fixture to embed")
    replay_group.add_argument(
        "--corpus",
        type=Path,
        help="PECOS SampleBatch corpus containing the embedded DEM and shots",
    )
    parser.add_argument("--max-shots", type=int, help="embed at most this many corpus shots")
    parser.add_argument("--replay-output", type=Path, help="also save the generated .fwr fixture here")
    parser.add_argument("--output", type=Path, help="Wasm output path (default: dist/pecos_frontier_wasm.wasm)")
    args = parser.parse_args()
    if args.max_shots is not None and args.max_shots <= 0:
        parser.error("--max-shots must be positive")
    if args.max_shots is not None and args.corpus is None:
        parser.error("--max-shots requires --corpus")
    if args.replay_output is not None and args.corpus is None:
        parser.error("--replay-output requires --corpus")
    dem_path = Path(args.dem) if args.dem else None
    output, replay_metadata = build(
        dem_path,
        replay_path=args.replay,
        corpus_path=args.corpus,
        max_shots=args.max_shots,
        replay_output=args.replay_output,
        output_path=args.output,
    )
    print(f"Built {output} ({output.stat().st_size:,} bytes)")
    if replay_metadata is not None:
        print(
            "Embedded "
            f"{replay_metadata['shots']:,} shots with {replay_metadata['detectors']} detectors and "
            f"{replay_metadata['observables']} observables",
        )


if __name__ == "__main__":
    main()
