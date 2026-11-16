# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Replay a released Shutty memory witness under the matched SI1000 schedule.

Run from a PECOS checkout: the immutable author circuits and witnesses are in
tests/qec/fixtures/si1000. Source: arXiv:2609.21376v3,
https://doi.org/10.5281/zenodo.22820614 (distance/check.py and witnesses.json).
This verifies a finite memory instance, not a logical-error-rate benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
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
from pecos.qec.generic import SI1000, GuppyOrdering, PauliFault, evaluate_tick_circuit, fault_distance_bounds
from pecos.qec.surface import tick_circuit_to_stim
from pecos_rslib.qec import DetectorErrorModel
from pecos_rslib_exp import monte_carlo, sim_neo, stabilizer


def main() -> None:
    """Export the schedule/noise receipt and physically replay a pinned witness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="torus_488_d4_r4_Z_phase0_p0p001")
    parser.add_argument("--exclude-through", type=int, choices=range(4), default=3)
    parser.add_argument("--max-pair-checks", type=int, default=10_000_000)
    parser.add_argument("--native-shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--audit-guppy", action="store_true", help="Trace ideal runtime lowering and save its audit")
    parser.add_argument("--guppy-ordering", choices=("functional", "unitary_regions", "all_moments"),
                        default="functional",
                        help="Optional compiler ordering cuts; not a physical timing or noise contract")
    parser.add_argument("--output", type=Path, default=Path("output/si1000-color-memory"))
    args = parser.parse_args()
    guppy_ordering = GuppyOrdering(args.guppy_ordering)
    fixtures = Path(__file__).resolve().parents[2] / "python/quantum-pecos/tests/qec/fixtures/si1000"
    cases = json.loads((fixtures / "witnesses.json").read_text())["cases"]
    matches = [c for c in cases if Path(c["fixture"]).stem == args.case]
    if len(matches) != 1:
        parser.error("--case must name one of the pinned cases in fixtures/si1000/witnesses.json")
    case = matches[0]
    reference = (fixtures / case["fixture"]).read_bytes()
    if hashlib.sha256(reference).hexdigest() != case["sha256"]:
        msg = "Released reference circuit checksum mismatch"
        raise ValueError(msg)
    topology, _, distance, rounds, basis, phase, rate = args.case.split("_")
    d, r = int(distance[1:]), int(rounds[1:])
    if topology == "torus":
        implementation = periodic_488_implementation(periodic_488_code(d), gate_set="native-cz")
        plan = periodic_488_memory(implementation, r, basis)
    else:
        implementation = color_code_implementation(color_code_spec("488", d), ColorExtraction.BRICKWORK_CZ)
        plan = color_memory(implementation, r, basis, start_phase=int(phase[5:]))
    noisy = SI1000(float(rate[1:].replace("p", "."))).apply(plan)
    selected = [
        c for c in noisy.channels
        if c.kind == "X_ERROR" and c.moment == case["tick"] and c.qubits[0] in case["x_error_qubits"]
    ]
    if sorted(c.qubits[0] for c in selected) != sorted(case["x_error_qubits"]):
        msg = "Released witness locators are absent or ambiguous in the constructed schedule"
        raise ValueError(msg)
    bounds = fault_distance_bounds(
        noisy, [PauliFault(c.id, "X") for c in selected],
        exclude_through=args.exclude_through, max_pair_checks=args.max_pair_checks,
    )
    if bounds.upper_bound != case["distance"]:
        msg = "Replayed witness weight differs from the released bound"
        raise ValueError(msg)
    native = noisy.compile_native()
    scheduled_dem = native.detector_error_model()
    native_flips = native.fault_effect(bounds.witness.faults)
    expected_flips = ((False,) * len(plan.detectors), bounds.witness.observable_flips)
    if native_flips != expected_flips or plan.evaluate(native.replay_faults(bounds.witness.faults)) != expected_flips:
        msg = "Native propagation/replay disagrees with the independently verified physical witness"
        raise ValueError(msg)
    records = native.sample_records(args.native_shots, seed=args.seed)
    values = [plan.evaluate(record) for record in records]
    noisy_ticks = noisy.to_tick_circuit()
    native_dem = DetectorErrorModel.from_circuit(noisy_ticks)
    actual = {(tuple(d), tuple(o)): p for p, d, o in native_dem.independent_mechanisms()}
    expected = {(tuple(d), tuple(o)): p for p, d, o in scheduled_dem.independent_mechanisms()}
    if (actual.keys() != expected.keys()
            or any(not math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=0.) for key in actual)
            or native_dem.detector_coordinates() != scheduled_dem.detector_coordinates()
            or native_dem.num_observables != scheduled_dem.num_observables):
        msg = "Generic inline DEM disagrees with scheduled native propagation"
        raise ValueError(msg)
    tick_records = [list(row) for row in sim_neo(noisy_ticks).quantum(stabilizer()).sampling(
        monte_carlo(args.native_shots),
    ).seed(args.seed).run().to_list()]
    tick_values = [evaluate_tick_circuit(noisy_ticks, record) for record in tick_records]
    if any(value != plan.evaluate(record) for record, value in zip(tick_records, tick_values, strict=True)):
        msg = "Noisy TickCircuit records do not match the scheduled parity contract"
        raise ValueError(msg)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tick-samples.json").write_text(json.dumps({
        "seed": args.seed, "shots": len(tick_records), "measurement_records": tick_records,
        "detector_values": [v[0] for v in tick_values], "observable_values": [v[1] for v in tick_values],
        "backend": "sim_neo stabilizer with inline categorical Pauli channels",
    }, indent=2))
    (args.output / "tick-execution-steps.json").write_text(json.dumps({
        "physical_moments": noisy_ticks.get_meta("qec_physical_moments"),
        "execution_steps": [
            {"tick": i, "physical_moment": noisy_ticks.get_tick_meta(i, "qec_moment"),
             "phase": noisy_ticks.get_tick_meta(i, "qec_execution_phase")}
            for i in range(noisy_ticks.num_ticks())
        ],
        "convention": "Channel-only steps serialize noise within a physical moment; no added hardware time",
    }, indent=2))
    (args.output / "native-dem.json").write_text(json.dumps({
        "mechanisms": native_dem.independent_mechanisms(),
        "num_detectors": native_dem.num_detectors,
        "num_observables": native_dem.num_observables,
        "detector_coordinates": native_dem.detector_coordinates(),
        "convention": "Independent Bernoulli mechanisms, not categorical physical faults",
        "backend": "DetectorErrorModel.from_circuit(noisy_ticks)",
        "scheduled_native_dem_verified": True,
    }, indent=2))
    (args.output / "native-samples.json").write_text(json.dumps({
        "seed": args.seed, "shots": len(records), "measurement_records": records,
        "detector_values": [v[0] for v in values], "observable_values": [v[1] for v in values],
        "convention": "Authored affine parity values; these paper memories have zero ideal reference",
    }, indent=2))
    if args.audit_guppy:
        audit = plan.audit_guppy(seed=args.seed, ordering=guppy_ordering)
        (args.output / "guppy-runtime-audit.json").write_text(json.dumps(audit.to_dict(), indent=2))
        print(audit.describe())
    noisy_text = str(noisy.to_stim())
    (args.output / "memory.si1000.stim").write_text(noisy_text)
    (args.output / "memory.inline.stim").write_text(tick_circuit_to_stim(noisy_ticks))
    (args.output / "memory.ideal.py").write_text(plan.guppy_source(ordering=guppy_ordering))
    (args.output / "guppy-ordering.json").write_text(
        json.dumps(plan.guppy_ordering_plan(guppy_ordering).to_dict(), indent=2),
    )
    ticks = plan.to_tick_circuit()
    (args.output / "memory.ideal.definitions.json").write_text(json.dumps({
        "supports": ticks.circuit_definitions(),
        "detectors": json.loads(ticks.get_meta("detectors") or "[]"),
        "observables": json.loads(ticks.get_meta("observables") or "[]"),
    }, indent=2))
    (args.output / "schedule.txt").write_text(plan.describe())
    (args.output / "noise.json").write_text(json.dumps({
        "model": "Shutty native-CZ SI1000", "p": noisy.noise.p,
        "schedule": plan.schedule.value, "channels": [asdict(c) for c in noisy.channels],
        "noisy_backends": ["stim", "pecos-native", "sim_neo-inline-tick"],
        "guppy_noise_applied": False, "inline_tick_noise_applied": True,
        "inline_stim_ticks": "execution steps, not physical moments",
    }, indent=2))
    (args.output / "fault-distance.json").write_text(json.dumps({
        "case": args.case, "source": case, "bounds": asdict(bounds), "exact": bounds.exact,
        "stim_version": stim.__version__,
        "native_witness_verified": True,
        "noisy_export_sha256": hashlib.sha256(noisy_text.encode()).hexdigest(),
        "scope": "This finite memory and its selected SI1000 physical noise locations",
    }, indent=2))
    print(noisy.describe())
    print(f"Native witness verified; {len(records)} native noisy trajectories; "
          f"{len(native_dem.independent_mechanisms())} independent DEM mechanisms")
    print(f"Inline TickCircuit: {len(tick_records)} noisy shots, {noisy_ticks.num_ticks()} execution steps "
          f"for {len(plan.moments)} physical moments")
    print(f"Physical fault-distance bounds: [{bounds.lower_bound}, {bounds.upper_bound}]; exact={bounds.exact}")


if __name__ == "__main__":
    main()
