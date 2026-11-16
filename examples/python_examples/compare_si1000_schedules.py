# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Compare a released native-CZ memory with rescheduled SI1000 alternatives.

Run from a PECOS checkout. Witnesses and unchanged reference circuits are pinned
under tests/qec/fixtures/si1000. Source: Shutty et al., arXiv:2609.21376v3,
https://doi.org/10.5281/zenodo.22820614 (distance/witnesses.json and check.py).
This is a finite-memory fault/DEM comparison, not a decoder benchmark or Guppy
runtime noise model. Alternative schedules are separately redecorated with SI1000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isclose
from pathlib import Path

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
from pecos.qec.generic import SI1000, PauliFault, SchedulePolicy
from pecos_rslib.qec import DetectorErrorModel


def _stim_mechanisms(circuit: stim.Circuit) -> dict:
    mechanisms = {}
    model = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened()
    for instruction in model:
        if instruction.type != "error":
            continue
        targets = instruction.targets_copy()
        d = tuple(sorted(t.val for t in targets if t.is_relative_detector_id()))
        o = tuple(sorted(t.val for t in targets if t.is_logical_observable_id()))
        p = instruction.args_copy()[0]
        if p and (d or o):
            previous = mechanisms.get((d, o), 0.)
            mechanisms[d, o] = previous * (1 - p) + p * (1 - previous)
    return mechanisms


def _check_mechanisms(actual: dict, expected: dict) -> None:
    if actual.keys() != expected.keys() or any(not isclose(actual[k], expected[k], rel_tol=1e-10, abs_tol=0.)
                                             for k in actual):
        msg = "Independent DEM propagation disagrees with the reference/export"
        raise ValueError(msg)


def main() -> None:
    """Replay a published witness and retain independent backend cross-checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="planar_488_d3_r3_Z_phase0_p0p001")
    parser.add_argument("--exclude-through", type=int, choices=range(4))
    parser.add_argument("--max-pair-checks", type=int, default=10_000_000)
    parser.add_argument("--output", type=Path, default=Path("output/si1000-schedule-comparison"))
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[2] / "python/quantum-pecos/tests/qec/fixtures/si1000"
    cases = json.loads((fixtures / "witnesses.json").read_text())["cases"]
    case = next((c for c in cases if Path(c["fixture"]).stem == args.case), None)
    if case is None:
        parser.error("--case must name a pinned SI1000 witness fixture")
    raw = (fixtures / case["fixture"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != case["sha256"]:
        msg = "Published reference fixture hash mismatch"
        raise ValueError(msg)
    topology, _, distance, rounds, basis, phase, rate = args.case.split("_")
    d, r = int(distance[1:]), int(rounds[1:])
    if topology == "planar":
        impl = color_code_implementation(color_code_spec("488", d), ColorExtraction.BRICKWORK_CZ)
        plan = color_memory(impl, r, basis, start_phase=int(phase[5:]))
    else:
        impl = periodic_488_implementation(periodic_488_code(d), gate_set="native-cz")
        plan = periodic_488_memory(impl, r, basis)
    noisy = SI1000(float(rate[1:].replace("p", "."))).apply(plan)
    selected = [c for c in noisy.channels if c.kind == "X_ERROR" and c.moment == case["tick"]
                and c.qubits[0] in case["x_error_qubits"]]
    if sorted(c.qubits[0] for c in selected) != sorted(case["x_error_qubits"]):
        msg = "Published physical witness is absent or ambiguous in the reference schedule"
        raise ValueError(msg)
    faults = [PauliFault(c.id, "X") for c in selected]
    through = min(3, case["distance"] - 1) if args.exclude_through is None else args.exclude_through
    published = _stim_mechanisms(stim.Circuit(raw.decode()))
    args.output.mkdir(parents=True, exist_ok=True)
    for policy in (SchedulePolicy.ASAP_UNITARIES, SchedulePolicy.COMPACT_UNITARIES):
        comparison = noisy.compare_schedule(faults, candidate_policy=policy,
                                             exclude_through=through, max_pair_checks=args.max_pair_checks)
        report = comparison.to_dict()
        if report["reference"]["bounds"]["upper_bound"] != case["distance"]:
            msg = "Reference witness does not reproduce the published upper bound"
            raise ValueError(msg)
        destination = args.output / policy.value
        destination.mkdir(parents=True, exist_ok=True)
        for label, variant in (("reference", comparison.reference), ("candidate", comparison.candidate)):
            actual = {(tuple(row["detectors"]), tuple(row["observables"])): row["probability"]
                      for row in report[label]["independent_dem"]}
            if label == "reference":
                _check_mechanisms(actual, published)
            _check_mechanisms(actual, _stim_mechanisms(variant.to_stim()))
            inline = DetectorErrorModel.from_circuit(variant.to_tick_circuit())
            _check_mechanisms(actual, {(tuple(ds), tuple(os)): p for p, ds, os in inline.independent_mechanisms()})
            (destination / f"{label}.stim").write_text(str(variant.to_stim()))
            (destination / f"{label}-schedule.txt").write_text(variant.ideal.describe())
        report["example_validation"] = {
            "published_reference": case, "published_reference_dem_matches": True,
            "stim_dems_match_native": True, "inline_tick_dems_match_native": True,
            "noise_applied_to_guppy": False,
        }
        (destination / "comparison.json").write_text(json.dumps(report, indent=2))
        (destination / "comparison.txt").write_text(comparison.describe() + "\n")
        (destination / "explanation.txt").write_text(comparison.explain() + "\n")
        print(comparison.describe(), flush=True)
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
