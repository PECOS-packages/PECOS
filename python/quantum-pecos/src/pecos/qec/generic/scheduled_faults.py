# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Physical witness replay and bounded exclusions for scheduled Pauli noise.

The small-weight exclusion follows the independently reproducible method in
Shutty's release, distance/check.py (https://doi.org/10.5281/zenodo.22820614).
Full undecomposed error supports form a relaxation of physical-location faults.
They can prove lower bounds; physical witnesses independently prove upper bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.qec.generic.scheduled_noise import NoisyScheduledCircuit


@dataclass(frozen=True)
class PauliFault:
    """A nonidentity Pauli branch at one identified physical noise channel."""

    channel_id: str
    paulis: str


@dataclass(frozen=True)
class LogicalFaultWitness:
    """A replayed, undetected physical fault set with a nonzero logical effect."""

    faults: tuple[PauliFault, ...]
    observable_flips: tuple[bool, ...]

    @property
    def weight(self) -> int:
        """Number of distinct physical noise locations, not affected qubits."""
        return len(self.faults)


@dataclass(frozen=True)
class FaultDistanceBounds:
    """Bounds for this finite memory under the supplied noise-location convention.

    Exactness requires matching a relaxed-support lower bound with a replayed
    physical upper bound. A short relaxed-support combination alone is never
    reported as a physical witness. No general code-family claim follows.
    """

    lower_bound: int
    upper_bound: int
    witness: LogicalFaultWitness
    excluded_through: int
    unique_error_supports: int | None

    @property
    def exact(self) -> bool:
        """Whether the independently obtained bounds coincide."""
        return self.lower_bound == self.upper_bound


def _validated_faults(noisy: NoisyScheduledCircuit, faults: Sequence[PauliFault]) -> dict[str, str]:
    """Validate physical branches shared by independent replay backends."""
    if not 0 < noisy.noise.p < 0.2:
        msg = "Fault analysis requires 0 < SI1000 p < 0.2"
        raise ValueError(msg)
    selected = tuple(faults)
    if not selected or any(not isinstance(f, PauliFault) for f in selected):
        msg = "Provide a nonempty sequence of PauliFault locations"
        raise ValueError(msg)
    channels = {c.id: c for c in noisy.channels}
    at = {}
    for fault in selected:
        if not isinstance(fault.channel_id, str) or fault.channel_id not in channels:
            msg = f"Unknown noise channel: {fault.channel_id}"
            raise ValueError(msg)
        if fault.channel_id in at:
            msg = "A physical noise location may be selected only once"
            raise ValueError(msg)
        channel = channels[fault.channel_id]
        branches = {"X_ERROR": 1, "DEPOLARIZE1": 3, "DEPOLARIZE2": 15}[channel.kind]
        if channel.probability / branches <= 0:
            msg = "Selected branch has zero probability at this floating-point noise rate"
            raise ValueError(msg)
        word = fault.paulis
        if (
            not isinstance(word, str)
            or len(word) != len(channel.qubits)
            or any(p not in "IXYZ" for p in word)
            or set(word) == {"I"}
            or (channel.kind == "X_ERROR" and word != "X")
        ):
            msg = f"Invalid nonidentity branch {word!r} for {channel.kind} at {channel.id}"
            raise ValueError(msg)
        at[fault.channel_id] = word
    return at


def verify_logical_fault(
    noisy: NoisyScheduledCircuit,
    faults: Sequence[PauliFault],
) -> LogicalFaultWitness:
    """Replay distinct positive-probability branches using Stim's Pauli frames.

    Only selected channels are replaced by deterministic Pauli faults; all other
    noise is removed. Ideal detector and observable determinism is checked first.
    The zero-probability and deterministic-readout endpoints are excluded from
    distance analysis, although they are valid SI1000 circuit export parameters.
    """
    import stim  # noqa: PLC0415 - optional independent propagation backend

    selected = tuple(faults)
    at = _validated_faults(noisy, selected)
    noisy.ideal.to_stim().detector_error_model()
    simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
    simulator.do(noisy._to_stim(at))  # noqa: SLF001 - shared internal noise emitter
    detectors = simulator.get_detector_flips(instance_index=0)
    observables = tuple(bool(value) for value in simulator.get_observable_flips(instance_index=0))
    if any(detectors) or not any(observables):
        msg = "Selected faults are detected or have no logical effect"
        raise ValueError(msg)
    return LogicalFaultWitness(selected, observables)


def remap_operation_faults(
    reference: NoisyScheduledCircuit,
    candidate: NoisyScheduledCircuit,
    faults: Sequence[PauliFault],
) -> tuple[PauliFault, ...]:
    """Transfer operation-attached faults across source-equivalent schedules.

    Source per-role occurrence identities disambiguate repeated gates and missing
    origin labels. Idle/wait locations have no assumed correspondence and are
    rejected. Identical SI1000 profiles, channel kinds, targets, and placements
    are required. The returned faults still need independent physical replay.
    """
    from pecos.qec.generic._schedule_analysis import _rescheduling_contract, _source_placements  # noqa: PLC0415

    selected = _validated_faults(reference, faults)
    if reference.noise != candidate.noise or _rescheduling_contract(reference.ideal) != _rescheduling_contract(
        candidate.ideal,
    ):
        msg = "Fault transfer requires the same noise profile and source rescheduling contract"
        raise ValueError(msg)
    old, _ = _source_placements(reference.ideal)
    new, _ = _source_placements(candidate.ideal)
    source_at = {(r["moment"], r["operation"]): key for key, r in old.items()}
    targets = {(c.moment, c.operation, c.placement, c.kind, c.qubits, c.reason): c for c in candidate.channels}
    sources = {c.id: c for c in reference.channels}
    mapped = []
    for channel_id, word in selected.items():
        channel = sources[channel_id]
        if channel.operation is None:
            msg = "Idle and collapse-wait faults cannot be transferred by source-operation identity"
            raise ValueError(msg)
        row = new[source_at[channel.moment, channel.operation]]
        key = (row["moment"], row["operation"], channel.placement, channel.kind, channel.qubits, channel.reason)
        target = targets.get(key)
        if target is None or target.probability != channel.probability:
            msg = "Rescheduled operation has no matching physical noise channel"
            raise ValueError(msg)
        mapped.append(PauliFault(target.id, word))
    _validated_faults(candidate, mapped)
    return tuple(mapped)


def _error_supports(noisy: NoisyScheduledCircuit) -> set[tuple[int, int]]:
    """Keep complete supports at a safe interior rate, independent of user weights.

    All interior SI1000 rates have the same possible Pauli branches. Tiny rates
    can disappear numerically when Stim converts depolarization to DEM errors.
    A fixed representative rate avoids an unsound lower bound from lost support.
    It also safely includes channels that underflow to zero in the supplied plan.
    """
    from pecos.qec.generic.scheduled_noise import SI1000  # noqa: PLC0415

    probe = SI1000(0.001).apply(noisy.ideal)
    dem = probe.to_stim().detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened()
    supports = set()
    for inst in dem:
        if inst.type != "error" or not inst.args_copy()[0]:
            continue
        detectors = observables = 0
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= 1 << target.val
            elif target.is_logical_observable_id():
                observables ^= 1 << target.val
            else:
                msg = "Distance exclusion requires undecomposed detector/logical supports"
                raise ValueError(msg)
        supports.add((detectors, observables))
    return supports


def _exclude(supports: set[tuple[int, int]], through: int, max_pair_checks: int) -> int:
    """Return the largest excluded weight, never an unverified upper bound."""
    if any(d == 0 and logical for d, logical in supports):
        return 0
    if through == 1:
        return 1
    by_detector = {0: 0}
    for detector, logical in supports:
        if detector in by_detector and by_detector[detector] != logical:
            return 1
        by_detector[detector] = logical
    if through == 2:
        return 2
    items = sorted(by_detector.items())
    pairs = len(items) * (len(items) - 1) // 2
    if pairs > max_pair_checks:
        msg = f"Three-branch exclusion needs {pairs} pair checks, exceeding budget {max_pair_checks}"
        raise ValueError(msg)
    for i, (d1, l1) in enumerate(items):
        for j in range(i):
            d2, l2 = items[j]
            other = by_detector.get(d1 ^ d2)
            if other is not None and other != l1 ^ l2:
                return 2
    return 3


def fault_distance_bounds(
    noisy: NoisyScheduledCircuit,
    faults: Sequence[PauliFault],
    *,
    exclude_through: int = 0,
    max_pair_checks: int = 10_000_000,
) -> FaultDistanceBounds:
    """Combine physical witness replay with exhaustive exclusion up to weight three.

    ``exclude_through=0`` performs only witness replay. The lower-bound search
    allows even mutually exclusive branches at one location, so excluding them
    is sound for physical faults. Probabilities and hyperedge decompositions are
    not used as weights. Supports are extracted at a fixed interior SI1000 rate
    to avoid numerical loss of tiny-probability branches. The pair budget fails explicitly rather than silently
    reporting a truncated search as complete. Stim is required for propagation.
    """
    if type(exclude_through) is not int or not 0 <= exclude_through <= 3:
        msg = "exclude_through must be an integer from 0 through 3"
        raise ValueError(msg)
    if type(max_pair_checks) is not int or max_pair_checks < 0:
        msg = "max_pair_checks must be a nonnegative integer"
        raise ValueError(msg)
    witness = verify_logical_fault(noisy, faults)
    supports = _error_supports(noisy) if exclude_through else None
    excluded = _exclude(supports, exclude_through, max_pair_checks) if supports is not None else 0
    if excluded >= witness.weight:
        msg = "Inconsistent lower bound and replayed witness"
        raise RuntimeError(msg)
    return FaultDistanceBounds(
        excluded + 1,
        witness.weight,
        witness,
        excluded,
        None if supports is None else len(supports),
    )
