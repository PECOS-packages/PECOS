# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Source locations behind a DEM factor, with categorical and independent weights."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from pecos.qec.generic._schedule_analysis import _source_placements

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.generic.native_noise import NativeScheduledNoise


def _support(values: Sequence[int], count: int, name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(i) is not int or not 0 <= i < count for i in result) or len(set(result)) != len(result):
        msg = f"{name} must contain distinct in-range integer indices"
        raise ValueError(msg)
    return tuple(sorted(result))


@dataclass(frozen=True)
class FaultSignatureExplanation:
    """An immutable source explanation of an independent detector/logical factor."""

    _report_json: str = field(repr=False)

    def to_dict(self) -> dict:
        """Return a copy of named parity definitions, channels and their weights."""
        return json.loads(self._report_json)

    def describe(self, *, max_channels: int = 5) -> str:
        """Show source-local contributors without adding their physical probabilities."""
        if type(max_channels) is not int or max_channels < 0:
            msg = "max_channels must be a nonnegative integer"
            raise ValueError(msg)
        report = self.to_dict()
        support = [f"D{d['id']}:{d['label']}" for d in report["detectors"]]
        support.extend(f"L{o['id']}:{o['label']}" for o in report["observables"])
        lines = [
            "Signature flips: " + ", ".join(support),
            f"Recombined independent DEM probability: {report['dem_probability']:.12g}",
            f"{report['matching_factor_count']} independent factors from {len(report['channels'])} channels",
        ]
        for prefix, parities in (("D", report["detectors"]), ("L", report["observables"])):
            for parity in parities:
                terms = parity["measurement_names"] + (["1"] if parity["constant"] else [])
                lines.append(f"  {prefix}{parity['id']}:{parity['label']} = " + (" XOR ".join(terms) or "0"))
        for channel in report["channels"][:max_channels]:
            origin = channel["source_origin"]
            label = origin["id"] if origin else channel["source_operation"] or channel["channel_id"]
            lines.append(
                f"  {label}: {channel['reason']} on {', '.join(channel['roles'])}; "
                f"{', '.join(channel['matching_branches'])}; local physical mass "
                f"{channel['matching_physical_branch_mass']:.12g}; "
                f"independent q={channel['independent_probability_per_branch']:.12g}",
            )
        if len(report["channels"]) > max_channels:
            lines.append(f"  {len(report['channels']) - max_channels} further channels retained in to_dict()")
        lines.append("Local physical masses are not additive across channels; DEM factors combine by XOR")
        lines.append("The DEM factor weight is not the marginal probability of observing this signature")
        return "\n".join(lines)


def explain_fault_signature(
    native: NativeScheduledNoise,
    *,
    detectors: Sequence[int] = (),
    observables: Sequence[int] = (),
) -> FaultSignatureExplanation:
    """Reconstruct one independent DEM term from source-local converted branches.

    Only nonzero converted factors are listed, matching detector_error_model().
    Matching physical branch mass is the sum within one categorical channel,
    with other channels absent. The reconstructed DEM term is not the marginal
    probability of this signature: different DEM factors can XOR to it too.
    Affine constants cancel in flips but remain visible in parity definitions.
    """
    from pecos.qec.generic.native_noise import _independent_branch_probability  # noqa: PLC0415

    plan = native.noisy.ideal
    ds = _support(detectors, len(plan.detectors), "detectors")
    os = _support(observables, len(plan.observables), "observables")
    if not ds and not os:
        msg = "An explanation requires a nonempty detector or observable support"
        raise ValueError(msg)
    # This proves ideal parity determinism, including at p=0 with no channels.
    _ = native._propagation  # noqa: SLF001 - shared native DEM implementation
    nd = len(plan.detectors)
    wanted = sum(1 << d for d in ds) | sum(1 << (nd + o) for o in os)
    source, _ = _source_placements(plan)
    at = {(r["moment"], r["operation"]): key for key, r in source.items()}
    names = plan.measurement_names or tuple(f"measurement[{i}]" for i in range(plan.num_measurements))

    def definition(index: int, *, logical: bool) -> dict:
        parity = (plan.observables if logical else plan.detectors)[index]
        return {
            "id": index,
            "label": parity.label,
            "records": list(parity.records),
            "measurement_names": [names[i] for i in parity.records],
            "constant": parity.constant,
            "coords": parity.coords,
        }

    channels, probability, factors = [], 0.0, 0
    for channel in native.noisy.channels:
        q = _independent_branch_probability(channel)
        if not q:
            continue
        words = native._branches[channel.id]  # noqa: SLF001 - same categorical branches as native DEM
        matching = [word for word in words if native._branch_mask(channel, word) == wanted]  # noqa: SLF001
        if not matching:
            continue
        factors += len(matching)
        channel_factor = 0.0
        for _word in matching:
            probability = probability * (1 - q) + q * (1 - probability)
            channel_factor = channel_factor * (1 - q) + q * (1 - channel_factor)
        identity = None if channel.operation is None else at[channel.moment, channel.operation]
        op = None if channel.operation is None else plan.moments[channel.moment][channel.operation]
        physical_probability = channel.probability / len(words)
        channels.append(
            {
                "channel_id": channel.id,
                "moment": channel.moment,
                "operation": channel.operation,
                "reason": channel.reason,
                "kind": channel.kind,
                "placement": channel.placement,
                "qubits": list(channel.qubits),
                "roles": [plan.role_names[q] if plan.role_names else f"q[{q}]" for q in channel.qubits],
                "gate": None if op is None else op.name,
                "source_operation": identity,
                "source_origin": None if op is None or op.origin is None else {**asdict(op.origin), "id": op.origin.id},
                "channel_probability": channel.probability,
                "matching_branches": matching,
                "physical_probability_per_branch": physical_probability,
                "matching_physical_branch_mass": len(matching) * physical_probability,
                "independent_probability_per_branch": q,
                "matching_independent_factor_probability": channel_factor,
            },
        )
    report = {
        "p": native.noisy.noise.p,
        "schedule": plan.schedule.value,
        "detectors": [definition(d, logical=False) for d in ds],
        "observables": [definition(o, logical=True) for o in os],
        "dem_probability": probability,
        "matching_factor_count": factors,
        "channels": channels,
        "source_citations": list(plan.citations),
        "scope": "Nonzero independent factors producing this detector/logical support; flips from ideal",
        "physical_mass_scope": "Mutually exclusive matching branches of one channel, with other channels absent",
        "dem_probability_scope": "XOR-composed factor weight, not the marginal probability of this signature",
        "physical_masses_additive_across_channels": False,
        "decoded_logical_error_rate_verified": False,
        "guppy_noise_equivalence_verified": False,
    }
    return FaultSignatureExplanation(json.dumps(report))
