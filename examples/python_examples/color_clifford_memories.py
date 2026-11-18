# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Run six-state memories and two-block Clifford circuits through Guppy.

Circuit ingredients follow arXiv:2610.24077v1. Frame-aware detector composition
is also discussed in arXiv:2412.14256v1, Sec. IV. This is an ideal circuit
example, not a reproduction of either hardware benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from pecos import sim, stabilizer
from pecos.qec.color import (
    ColorCliffordExperiment,
    ColorExtraction,
    LogicalState,
    color_code_implementation,
    color_code_spec,
    color_state_memory,
)

if TYPE_CHECKING:
    from pecos.qec.generic import ScheduledCircuit


def run(label: str, plan: ScheduledCircuit, shots: int, output: Path | None) -> None:
    """Execute one ideal experiment and check all declared success parities."""
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(42).run(shots).to_dict()
    for shot in range(shots):
        detectors, observables = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        if any(detectors) or any(observables):
            msg = f"Noiseless detector/logical failure in {label}"
            raise RuntimeError(msg)
    print(f"{label}: {shots} ideal shots passed; {plan.num_qubits} roles, {len(plan.detectors)} detectors")
    if output is not None:
        (output / f"{label}.py").write_text(plan.guppy_source(), encoding="utf-8")
        (output / f"{label}.stim").write_text(str(plan.to_stim()), encoding="utf-8")
        (output / f"{label}.schedule.txt").write_text(plan.describe(), encoding="utf-8")


def main() -> None:
    """Exercise all six states and Bell XX/ZZ correlations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
    impl = color_code_implementation(color_code_spec("666", args.distance), ColorExtraction.SWAP_SUPERDENSE)
    for state in LogicalState:
        run(state.name.lower(), color_state_memory(impl, args.rounds, state), args.shots, args.output)
    for basis in ("X", "Z"):
        experiment = ColorCliffordExperiment(impl, ("alice", "bob"))
        experiment.prepare("alice", "+X", start_phase=2)
        experiment.prepare("bob", "+Z", start_phase=1)
        experiment.extract("alice")
        experiment.extract("bob", 2)
        experiment.gate("CX", "alice", "bob")
        experiment.extract("alice", args.rounds)
        experiment.extract("bob", args.rounds)
        a, b = experiment.readout("alice", basis), experiment.readout("bob", basis)
        experiment.observable(f"Bell {basis}{basis}", a ^ b)
        run(f"bell_{basis.lower()}", experiment.compile(), args.shots, args.output)


if __name__ == "__main__":
    main()
