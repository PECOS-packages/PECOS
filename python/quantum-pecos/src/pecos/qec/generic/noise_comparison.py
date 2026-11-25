# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""SI1000 schedule comparisons with physical witness replay and bounded exclusions.

The noise convention and bounded support exclusion follow Shutty et al.,
arXiv:2609.21376v3 and https://doi.org/10.5281/zenodo.22820614.
This analyzes explicit native-CZ schedules, not compiler-lowered Guppy noise.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import cached_property
from math import isclose
from typing import TYPE_CHECKING

from pecos.qec.generic.scheduled import SchedulePolicy
from pecos.qec.generic.scheduled_faults import fault_distance_bounds, remap_operation_faults

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.generic.native_noise import NativeScheduledNoise
    from pecos.qec.generic.scheduled_faults import PauliFault
    from pecos.qec.generic.scheduled_noise import NoisyScheduledCircuit


@dataclass(frozen=True)
class ScheduledNoiseComparison:
    """Independent DEM differences and finite-memory physical distance bounds."""

    reference: NoisyScheduledCircuit
    candidate: NoisyScheduledCircuit
    _report_json: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return an independent snapshot with witness locations and evidence limits."""
        return json.loads(self._report_json)

    def describe(self) -> str:
        """Summarize bounds without inferring noisy logical error rates."""
        report = self.to_dict()
        lines = [f"SI1000 p={report['p']:g}; {report['candidate_policy']} versus reference"]
        for key in ("reference", "candidate"):
            row = report[key]
            bounds = row["bounds"]
            lines.append(
                f"{key}: {row['physical_moments']} moments, {row['noise_locations']} noise locations; "
                f"physical distance [{bounds['lower_bound']}, {bounds['upper_bound']}], "
                f"exact={bounds['exact']}",
            )
        dem = report["independent_dem_comparison"]
        lines.append(
            f"Independent DEMs match at this p within stated tolerance: {dem['matches_at_p']}; "
            f"{len(dem['changes'])} changed signatures/probabilities",
        )
        lines.append("Finite scheduled memories only; logical error rates and Guppy noise equivalence unverified")
        return "\n".join(lines)

    @cached_property
    def _native_pair(self) -> tuple[NativeScheduledNoise, NativeScheduledNoise]:
        return self.reference.compile_native(), self.candidate.compile_native()

    def explain_change(self, index: int) -> dict:
        """Explain one changed support using each schedule's own channel locations.

        The index refers to ``independent_dem_comparison.changes`` in to_dict().
        Each explanation reconstructs and checks the stored independent factor
        weight. Physical branch masses apply within a single channel only.
        """
        dem = self.to_dict()["independent_dem_comparison"]
        if type(index) is not int or not 0 <= index < len(dem["changes"]):
            msg = "index must identify an existing changed signature"
            raise ValueError(msg)
        change = dem["changes"][index]
        result = {"change": change}
        for key, native in zip(("reference", "candidate"), self._native_pair, strict=True):
            explanation = native.explain_signature(detectors=change["detectors"], observables=change["observables"])
            row = explanation.to_dict()
            if not isclose(
                row["dem_probability"],
                change[f"{key}_probability"],
                rel_tol=dem["rel_tol"],
                abs_tol=dem["abs_tol"],
            ):
                msg = f"{key} explanation disagrees with the stored DEM factor weight"
                raise ValueError(msg)
            result[key] = row
        return result

    def explain(self, *, top_changes: int = 3, max_channels: int = 5) -> str:
        """Describe changed independent factors and their source-local contributors."""
        from pecos.qec.generic.fault_explanations import FaultSignatureExplanation  # noqa: PLC0415

        if any(type(n) is not int or n < 0 for n in (top_changes, max_channels)):
            msg = "top_changes and max_channels must be nonnegative integers"
            raise ValueError(msg)
        changes = self.to_dict()["independent_dem_comparison"]["changes"]
        lines = [self.describe()]
        if not changes:
            lines.append("No changed independent DEM terms at this p within the stated tolerance")
        for index in range(min(top_changes, len(changes))):
            report = self.explain_change(index)
            lines.append(f"Change {index}: {report['change']['status']}")
            for key in ("reference", "candidate"):
                lines.append(f"{key}:")
                lines.append(FaultSignatureExplanation(json.dumps(report[key])).describe(max_channels=max_channels))
        if len(changes) > top_changes:
            lines.append(f"{len(changes) - top_changes} further changes available through explain_change(index)")
        return "\n".join(lines)


def _assess(noisy: NoisyScheduledCircuit, faults: Sequence[PauliFault], through: int, budget: int) -> tuple[dict, dict]:
    from pecos.qec.generic.scheduled_noise import SI1000  # noqa: PLC0415

    bounds = fault_distance_bounds(noisy, faults, exclude_through=through, max_pair_checks=budget)
    native = noisy.compile_native()
    expected = ((False,) * len(noisy.ideal.detectors), bounds.witness.observable_flips)
    if native.fault_effect(faults) != expected:
        msg = "Native fault propagation disagrees with the independently replayed witness"
        raise ValueError(msg)
    ideal = SI1000(0).apply(noisy.ideal).compile_native().sample_records(1, seed=0)[0]
    baseline = noisy.ideal.evaluate(ideal)
    actual = noisy.ideal.evaluate(native.replay_faults(faults, seed=0))
    flips = tuple(tuple(x != y for x, y in zip(a, b, strict=True)) for a, b in zip(actual, baseline, strict=True))
    if flips != expected:
        msg = "Native trajectory replay disagrees with the independently replayed witness"
        raise ValueError(msg)
    mechanisms = {(tuple(d), tuple(o)): p for p, d, o in native.detector_error_model().independent_mechanisms()}
    return {
        "physical_moments": len(noisy.ideal.moments),
        "noise_locations": len(noisy.channels),
        "channels_by_reason": dict(sorted(Counter(c.reason for c in noisy.channels).items())),
        "bounds": {**asdict(bounds), "exact": bounds.exact},
        "native_propagation_verified": True,
        "native_trajectory_replay_verified": True,
        "independent_dem": [
            {"detectors": list(d), "observables": list(o), "probability": p} for (d, o), p in sorted(mechanisms.items())
        ],
    }, mechanisms


def compare_si1000_schedules(
    noisy: NoisyScheduledCircuit,
    faults: Sequence[PauliFault],
    *,
    candidate_policy: SchedulePolicy = SchedulePolicy.ASAP_UNITARIES,
    exclude_through: int = 2,
    max_pair_checks: int = 10_000_000,
) -> ScheduledNoiseComparison:
    """Compare a reference schedule with an alternative at one explicit SI1000 p.

    Supply an operation-attached physical logical witness on the reference. Its
    remapped branches must replay as undetected logical faults in both schedules.
    Stim independently replays/excludes faults; PECOS propagates and executes the
    witness and computes the independent-error DEM. Exclusion uses a conservative
    relaxation and an explicit pair budget. Equal DEMs within the stated floating
    tolerance at one p do not establish all-rate or physical noise equivalence.
    """
    if noisy.ideal.schedule != SchedulePolicy.REFERENCE:
        msg = "Noise comparison requires the original REFERENCE schedule"
        raise ValueError(msg)
    if candidate_policy not in (SchedulePolicy.ASAP_UNITARIES, SchedulePolicy.COMPACT_UNITARIES):
        msg = "candidate_policy must be ASAP_UNITARIES or COMPACT_UNITARIES"
        raise ValueError(msg)
    faults = tuple(faults)
    candidate = noisy.noise.apply(noisy.ideal.rescheduled(candidate_policy))
    transferred = remap_operation_faults(noisy, candidate, faults)
    old, old_dem = _assess(noisy, faults, exclude_through, max_pair_checks)
    new, new_dem = _assess(candidate, transferred, exclude_through, max_pair_checks)
    changes, max_delta = [], 0.0
    for key in sorted(old_dem.keys() | new_dem.keys()):
        a, b = old_dem.get(key, 0.0), new_dem.get(key, 0.0)
        max_delta = max(max_delta, abs(b - a))
        if key not in old_dem or key not in new_dem or not isclose(a, b, rel_tol=1e-10, abs_tol=0.0):
            changes.append(
                {
                    "detectors": list(key[0]),
                    "observables": list(key[1]),
                    "reference_probability": a,
                    "candidate_probability": b,
                    "status": (
                        "added" if key not in old_dem else "removed" if key not in new_dem else "probability_changed"
                    ),
                },
            )
    old_channels, new_channels = ({c.id: c for c in x.channels} for x in (noisy, candidate))
    report = {
        "p": noisy.noise.p,
        "reference_policy": noisy.ideal.schedule.value,
        "candidate_policy": candidate_policy.value,
        "source_citations": list(noisy.ideal.citations),
        "noise_reference": "Shutty et al., arXiv:2609.21376v3; https://doi.org/10.5281/zenodo.22820614",
        "reference": old,
        "candidate": new,
        "witness_mapping": [
            {
                "reference": asdict(a),
                "candidate": asdict(b),
                "reference_channel": asdict(old_channels[a.channel_id]),
                "candidate_channel": asdict(new_channels[b.channel_id]),
            }
            for a, b in zip(faults, transferred, strict=True)
        ],
        "independent_dem_comparison": {
            "matches_at_p": not changes,
            "rel_tol": 1e-10,
            "abs_tol": 0.0,
            "max_probability_delta": max_delta,
            "changes": changes,
            "scope": "Aggregated independent Bernoulli mechanisms at this p",
        },
        "scope": "Finite native-CZ schedules with explicitly redecorated SI1000 channels",
        "exclusion": {
            "through": exclude_through,
            "max_pair_checks": max_pair_checks,
            "relaxation": "Full undecomposed supports, including mutually exclusive branches",
        },
        "source_rescheduling_contract_verified": True,
        "physical_witnesses_verified": True,
        "logical_error_rates_verified": False,
        "guppy_noise_equivalence_verified": False,
        "all_rate_noise_equivalence_verified": False,
    }
    return ScheduledNoiseComparison(noisy, candidate, json.dumps(report))
