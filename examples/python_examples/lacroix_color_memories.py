# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Run Lacroix native-CZ memories and export Guppy, TickCircuit views, and Stim.

Reference: arXiv:2412.14256v1, Supplement C.3/D.1; dataset DOI
10.5281/zenodo.14238944. Normalized ideal circuits, not device/noise replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pecos import sim, stabilizer
from pecos.qec import DetectorErrorModel
from pecos.qec.color import lacroix_memory_program
from pecos.qec.generic import SchedulePolicy


def main() -> None:
    """Export both backends from one selected schedule and check Guppy outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/lacroix-color-memories"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--schedule", choices=[s.value for s in SchedulePolicy], default="reference")
    parser.add_argument("--dem", action="store_true", help="Build a native TickCircuit DEM with example Pauli noise")
    parser.add_argument("--audit-parities", action="store_true", help="Check closed ideal detector/observable signs")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for patch in ("d3_west", "d3_east", "d3_north", "d5"):
        for basis in "XZ":
            name = f"{patch}_{basis}"
            authored = lacroix_memory_program(patch, args.rounds, basis)
            plan = authored.compile(SchedulePolicy(args.schedule))
            if args.audit_parities:
                audit = plan.audit_parities()
                audit.require_valid()
                (args.output / f"{name}.parity-audit.json").write_text(json.dumps(audit.to_dict(), indent=2))
                (args.output / f"{name}.parity-audit.txt").write_text(audit.describe() + "\n")
            ticks = plan.to_tick_circuit()
            (args.output / f"{name}.py").write_text(plan.guppy_source())
            (args.output / f"{name}.stim").write_text(str(plan.to_stim()))
            (args.output / f"{name}.txt").write_text(authored.describe() + "\n" + plan.describe())
            (args.output / f"{name}.definitions.json").write_text(json.dumps(ticks.circuit_definitions(), indent=2))
            (args.output / f"{name}.svg").write_text(
                ticks.render(
                    "svg", start=0, stop=min(12, ticks.num_ticks()), title=f"{name}: first 12 physical moments",
                ),
            )
            columns = (
                sim(plan.guppy_program())
                .qubits(plan.num_qubits)
                .quantum(stabilizer())
                .seed(42)
                .run(args.shots)
                .to_dict()
            )
            manifest_bytes = (json.dumps(plan.measurement_manifest(), indent=2) + "\n").encode("utf-8")
            (args.output / f"{name}.measurements.json").write_bytes(manifest_bytes)
            (args.output / f"{name}.raw-results.json").write_text(
                json.dumps({
                    "measurement_manifest": f"{name}.measurements.json",
                    "measurement_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "source": "ideal Guppy stabilizer simulation", "seed": 42, "shots": args.shots,
                    "columns": columns,
                }, indent=2) + "\n", encoding="utf-8",
            )
            for shot in range(args.shots):
                detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
                if any(detectors) or any(observables):
                    msg = f"Unexpected ideal parity in {name}, shot {shot}"
                    raise RuntimeError(msg)
            if args.dem:
                dem = DetectorErrorModel.from_circuit(ticks, p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
                (args.output / f"{name}.dem").write_text(dem.to_string())
            print(
                f"{name}: {plan.num_qubits} active qubits, {ticks.num_ticks()} ticks, {args.shots} Guppy shots passed",
            )


if __name__ == "__main__":
    main()
