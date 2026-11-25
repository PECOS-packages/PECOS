# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Build, inspect, and run Shutty's four-logical-qubit periodic 4.8.8 memory.

Reference: arXiv:2609.21376v3; author source DOI 10.5281/zenodo.22820614.
Choose the twelve-CNOT-layer or 25-moment native-CZ circuit, without SI1000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pecos import sim, stabilizer
from pecos.qec import DetectorErrorModel
from pecos.qec.color import periodic_488_code, periodic_488_implementation, periodic_488_memory_program
from pecos.qec.generic import SchedulePolicy


def main() -> None:
    """Export both backends from the same named code/gadget/schedule description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gate-set", choices=("cnot", "native-cz"), default="cnot")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--schedule", choices=[s.value for s in SchedulePolicy], default="reference")
    parser.add_argument("--output", type=Path, default=Path("output/periodic-color-memories"))
    parser.add_argument("--dem", action="store_true", help="Build a native DEM with example Pauli noise")
    parser.add_argument(
        "--audit-flows",
        action="store_true",
        help="Check signed extraction flows and each memory's closed ideal parities",
    )
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    code = periodic_488_code(args.distance)
    implementation = periodic_488_implementation(code, gate_set=args.gate_set)
    (args.output / "code.txt").write_text(code.describe())
    (args.output / "extraction.txt").write_text(implementation.gadget().describe())
    if args.audit_flows:
        audit = implementation.audit_flows()
        audit.require_valid()
        (args.output / "extraction-flows.json").write_text(json.dumps(audit.to_dict(), indent=2))
        (args.output / "extraction-flows.txt").write_text(audit.describe() + "\n")
    for basis in "XZ":
        authored = periodic_488_memory_program(implementation, args.rounds, basis)
        plan = authored.compile(SchedulePolicy(args.schedule))
        if args.audit_flows:
            audit = plan.audit_parities()
            audit.require_valid()
            (args.output / f"{basis}.parity-audit.json").write_text(json.dumps(audit.to_dict(), indent=2))
            (args.output / f"{basis}.parity-audit.txt").write_text(audit.describe() + "\n")
        ticks = plan.to_tick_circuit()
        (args.output / f"{basis}.py").write_text(plan.guppy_source())
        (args.output / f"{basis}.stim").write_text(str(plan.to_stim()))
        (args.output / f"{basis}.gadgets.txt").write_text(authored.describe())
        (args.output / f"{basis}.schedule.txt").write_text(plan.describe())
        (args.output / f"{basis}.definitions.json").write_text(
            json.dumps(
                {
                    "supports": ticks.circuit_definitions(),
                    "detector_metadata": json.loads(ticks.get_meta("detectors") or "[]"),
                    "observable_metadata": json.loads(ticks.get_meta("observables") or "[]"),
                },
                indent=2,
            ),
        )
        columns = (
            sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(42).run(args.shots).to_dict()
        )
        manifest_bytes = (json.dumps(plan.measurement_manifest(), indent=2) + "\n").encode("utf-8")
        (args.output / f"{basis}.measurements.json").write_bytes(manifest_bytes)
        (args.output / f"{basis}.raw-results.json").write_text(
            json.dumps(
                {
                    "measurement_manifest": f"{basis}.measurements.json",
                    "measurement_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "source": "ideal Guppy stabilizer simulation",
                    "seed": 42,
                    "shots": args.shots,
                    "columns": columns,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for shot in range(args.shots):
            dets, obs = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
            if any(dets) or obs != (False, False, False, False):
                msg = f"Unexpected ideal parity in {basis} memory, shot {shot}"
                raise RuntimeError(msg)
        if args.dem:
            dem = DetectorErrorModel.from_circuit(ticks, p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
            (args.output / f"{basis}.dem").write_text(dem.to_string())
        print(
            f"{basis}: {plan.num_qubits} qubits, {ticks.num_ticks()} ticks, "
            f"4 logicals; {args.shots} Guppy shots passed",
        )


if __name__ == "__main__":
    main()
