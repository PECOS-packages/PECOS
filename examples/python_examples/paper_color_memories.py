# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Run cited 4.8.8 and 6.6.6 memory constructions through Guppy and PECOS.

References: https://arxiv.org/abs/2609.21376v3 and
https://arxiv.org/abs/2610.24077v1. See docs/user-guide/paper-color-codes.md
for construction provenance and the distinction between functional Guppy
execution and the paper's reference timing/noise experiments.

Example:
    python examples/python_examples/paper_color_memories.py --output /tmp/color-memories
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pecos import sim, stabilizer
from pecos.qec import build_dem_from_guppy
from pecos.qec.color import ColorExtraction, color_code_implementation, color_code_spec, color_memory_program
from pecos.qec.generic import SchedulePolicy


def main() -> None:
    """Compare all four constructions using the same memory interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3, help="Shared distance: 3, 7, 11, ...")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--basis", choices=("X", "Z"), default="Z")
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--schedule", choices=[p.value for p in SchedulePolicy], default="reference")
    parser.add_argument("--output", type=Path, help="Write circuits, measurement manifests, and raw shots here")
    parser.add_argument("--dem", action="store_true", help="Also build a noisy DEM from each actual Guppy trace")
    parser.add_argument("--audit-flows", action="store_true",
                        help="Verify signed ideal contracts of every extraction phase")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    for choice in ColorExtraction:
        family = "488" if choice in {ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ} else "666"
        implementation = color_code_implementation(color_code_spec(family, args.distance), choice)
        if args.audit_flows:
            audits = [implementation.audit_flows(phase) for phase in range(implementation.period)]
            for audit in audits:
                audit.require_valid()
            count = sum(len(a.to_dict()["contracts"]) for a in audits)
            print(f"{choice.value}: {count} signed ideal contracts passed over {len(audits)} phases")
            if args.output:
                (args.output / f"{choice.value}.flows.json").write_text(
                    json.dumps([a.to_dict() for a in audits], indent=2), encoding="utf-8",
                )
                (args.output / f"{choice.value}.flows.txt").write_text(
                    "\n\n".join(a.describe() for a in audits) + "\n", encoding="utf-8",
                )
        authored = color_memory_program(implementation, args.rounds, args.basis)
        plan = authored.compile(SchedulePolicy(args.schedule))
        program = plan.guppy_program()
        columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(42).run(args.shots).to_dict()
        failures = 0
        for shot in range(args.shots):
            detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
            failures += any(detectors) or any(observables)
        print(
            f"{choice.value}: {plan.num_qubits} qubits, {plan.num_measurements} measurements, "
            f"{len(plan.detectors)} detectors; {failures}/{args.shots} ideal shots with nonzero checks",
        )
        if failures:
            msg = f"Noiseless memory check failed for {choice.value}"
            raise RuntimeError(msg)
        if args.output:
            manifest_bytes = (json.dumps(plan.measurement_manifest(), indent=2) + "\n").encode("utf-8")
            (args.output / f"{choice.value}.measurements.json").write_bytes(manifest_bytes)
            (args.output / f"{choice.value}.raw-results.json").write_text(
                json.dumps({
                    "measurement_manifest": f"{choice.value}.measurements.json",
                    "measurement_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "source": "ideal Guppy stabilizer simulation", "seed": 42, "shots": args.shots,
                    "columns": columns,
                }, indent=2) + "\n", encoding="utf-8",
            )
            (args.output / f"{choice.value}.code.txt").write_text(implementation.code.describe(), encoding="utf-8")
            (args.output / f"{choice.value}.gadgets.txt").write_text(authored.describe(), encoding="utf-8")
            (args.output / f"{choice.value}.schedule.txt").write_text(plan.describe(), encoding="utf-8")
            (args.output / f"{choice.value}.py").write_text(plan.guppy_source(), encoding="utf-8")
            (args.output / f"{choice.value}.stim").write_text(str(plan.to_stim()), encoding="utf-8")
        if args.dem:
            detectors, observables = plan.dem_spec()
            build = build_dem_from_guppy(
                program,
                num_qubits=plan.num_qubits,
                detectors=detectors,
                observables=observables,
                p1=0.001,
                p2=0.002,
                p_meas=0.003,
                p_prep=0.004,
            )
            dem = build.dem.to_string()
            print(f"  Actual Guppy trace DEM: {len(dem.splitlines())} lines")
            if args.output:
                (args.output / f"{choice.value}.dem").write_text(dem, encoding="utf-8")


if __name__ == "__main__":
    main()
