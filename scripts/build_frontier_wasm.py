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


def pack_replay(npz_path: Path, destination: Path, max_shots: int | None = None) -> dict[str, int]:
    """Pack detector and observable bits from an NPZ into a Wasm fixture."""
    try:
        import numpy as np  # noqa: TID251 - NPZ fixture ingestion is a build-time compatibility boundary.
    except ImportError as error:
        message = "embedding NPZ shots requires NumPy; rerun with `uv run --group numpy-compat`"
        raise RuntimeError(message) from error

    resolved = npz_path.resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as arrays:
        try:
            detectors = np.asarray(arrays["detection_events"], dtype=np.uint8)
            observables = np.asarray(arrays["observable_flips"], dtype=np.uint8)
        except KeyError as error:
            message = f"{resolved} must contain detection_events and observable_flips"
            raise ValueError(message) from error

    if detectors.ndim != 2 or observables.ndim != 2:
        message = "detection_events and observable_flips must both be two-dimensional"
        raise ValueError(message)
    if detectors.shape[0] != observables.shape[0]:
        message = "detection_events and observable_flips must have the same number of shots"
        raise ValueError(message)
    if observables.shape[1] > MAX_OBSERVABLES:
        message = (
            f"fixture has {observables.shape[1]} observables; "
            f"the Wasm result ABI supports at most {MAX_OBSERVABLES}"
        )
        raise ValueError(message)
    if not np.all((detectors == 0) | (detectors == 1)) or not np.all((observables == 0) | (observables == 1)):
        message = "fixture arrays must contain only binary values"
        raise ValueError(message)

    shot_count = detectors.shape[0] if max_shots is None else min(detectors.shape[0], max_shots)
    detectors = detectors[:shot_count]
    observables = observables[:shot_count]

    def pack_words(bits: object) -> object:
        values = np.asarray(bits, dtype=np.uint8)
        words = np.zeros((shot_count, (values.shape[1] + 31) // 32), dtype="<u4")
        for bit in range(values.shape[1]):
            words[:, bit // 32] |= values[:, bit].astype(np.uint32) << np.uint32(bit % 32)
        return words

    records = np.concatenate((pack_words(detectors), pack_words(observables)), axis=1).astype("<u4", copy=False)
    header = struct.pack(
        "<4sIIII",
        REPLAY_MAGIC,
        REPLAY_VERSION,
        shot_count,
        detectors.shape[1],
        observables.shape[1],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + records.tobytes(order="C"))
    return {
        "shots": shot_count,
        "detectors": detectors.shape[1],
        "observables": observables.shape[1],
    }


def build(
    dem_path: Path | None,
    replay_path: Path | None = None,
    syndromes_path: Path | None = None,
    max_shots: int | None = None,
    replay_output: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, int] | None]:
    """Build and copy the module, returning the distribution path."""
    cargo = cargo_executable()
    target_dir = target_directory(cargo)
    environment = os.environ.copy()
    environment["RUSTFLAGS"] = " ".join(value for value in (environment.get("RUSTFLAGS"), GETRANDOM_CFG) if value)
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
        elif syndromes_path is not None:
            packed_replay = Path(temporary_directory) / "replay.fwr"
            replay_metadata = pack_replay(syndromes_path, packed_replay, max_shots)
            environment["FRONTIER_REPLAY_PATH"] = str(packed_replay)
            if replay_output is not None:
                resolved_output = replay_output.resolve()
                resolved_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(packed_replay, resolved_output)
        else:
            environment.pop("FRONTIER_REPLAY_PATH", None)

        subprocess.run(
            [cargo, "build", "--release", "--target", TARGET, "-p", PACKAGE],
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
    replay_group.add_argument("--syndromes", type=Path, help="NPZ containing detection_events and observable_flips")
    parser.add_argument("--max-shots", type=int, help="embed at most this many NPZ shots")
    parser.add_argument("--replay-output", type=Path, help="also save the generated .fwr fixture here")
    parser.add_argument("--output", type=Path, help="Wasm output path (default: dist/pecos_frontier_wasm.wasm)")
    args = parser.parse_args()
    if args.max_shots is not None and args.max_shots <= 0:
        parser.error("--max-shots must be positive")
    if args.replay_output is not None and args.syndromes is None:
        parser.error("--replay-output requires --syndromes")
    dem_path = Path(args.dem) if args.dem else None
    output, replay_metadata = build(
        dem_path,
        replay_path=args.replay,
        syndromes_path=args.syndromes,
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
