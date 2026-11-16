# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Explicit noise channels attached to a selected physical QEC schedule.

SI1000 convention: Shutty, arXiv:2609.21376v3, simulation methods;
source release https://doi.org/10.5281/zenodo.22820614,
src/vendor/_triangle488/native.py::decorate_si1000.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

from pecos.qec.generic.scheduled import SchedulePolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import stim

    from pecos.qec.generic.native_noise import NativeScheduledNoise
    from pecos.qec.generic.noise_comparison import ScheduledNoiseComparison
    from pecos.qec.generic.scheduled import ScheduledCircuit
    from pecos.qec.generic.scheduled_faults import PauliFault
    from pecos.quantum import TickCircuit

_SINGLE_CLIFFORDS = frozenset({"H", "S", "S_DAG", "X", "Z", "C_XYZ", "C_ZYX", "SQRT_X"})


@dataclass(frozen=True)
class NoiseChannel:
    """One independent physical noise location, with mutually exclusive branches.

    Each single-qubit target is a distinct location. A two-qubit depolarizing
    channel is one location, even when its selected branch affects both qubits.
    IDs identify locations within this selected schedule, not across rescheduling.
    """

    id: str
    moment: int
    operation: int | None
    placement: str
    kind: str
    qubits: tuple[int, ...]
    probability: float
    reason: str


@dataclass(frozen=True)
class SI1000:
    """The released native-CZ SI1000 profile, including collapse wait noise.

    Every allocated site receives idle/wait noise where appropriate, including
    sites before preparation or after retirement. This matches the release.
    Guppy compilation may change timing: this profile describes the supplied
    physical schedule, not an unaudited Guppy execution trace.
    """

    p: float

    def __post_init__(self) -> None:
        """Validate the complete probability range, including 5p readout errors."""
        if type(self.p) not in (int, float) or not isfinite(self.p) or not 0 <= self.p <= 0.2:
            msg = "SI1000 p must be finite and in [0, 0.2]"
            raise ValueError(msg)
        object.__setattr__(self, "p", float(self.p))

    def apply(self, plan: ScheduledCircuit) -> NoisyScheduledCircuit:
        """Decorate the selected schedule without changing gates, ticks, or events."""
        return NoisyScheduledCircuit(plan, self)


@dataclass(frozen=True)
class NoisyScheduledCircuit:
    """An ideal schedule plus an inspectable noise profile.

    Stim, inline TickCircuit export, and ``compile_native()`` support the channels. ``ideal``
    provides Guppy and TickCircuit implementations; those do not acquire this
    noise model implicitly. Channel inspection and native execution need no Stim.
    """

    ideal: ScheduledCircuit
    noise: SI1000

    def __post_init__(self) -> None:
        """Require the explicit native gate alphabet rather than guess a lowering."""
        from pecos.qec.generic.scheduled import ScheduledCircuit  # noqa: PLC0415

        if not isinstance(self.ideal, ScheduledCircuit) or not isinstance(self.noise, SI1000):
            msg = "Expected a ScheduledCircuit and an SI1000 profile"
            raise TypeError(msg)
        unsupported = {op.name for moment in self.ideal.moments for op in moment} - _SINGLE_CLIFFORDS - {"R", "M", "CZ"}
        if unsupported:
            msg = f"SI1000 requires a physical native-CZ schedule; unsupported operations: {sorted(unsupported)}"
            raise ValueError(msg)

    @property
    def channels(self) -> tuple[NoiseChannel, ...]:
        """Enumerate independent locations, including idle and collapse wait channels."""
        p = self.noise.p
        if not p:
            return ()
        channels = []
        allocation = set(range(self.ideal.num_qubits))
        for m, moment in enumerate(self.ideal.moments):
            active, collapsed = set(), set()
            for j, op in enumerate(moment):
                active.update(op.qubits)
                if op.name in {"R", "M"}:
                    collapsed.update(op.qubits)
                if op.name == "M":
                    kind, rate, placement, reason = "X_ERROR", 5 * p, "before", "measurement"
                elif op.name == "R":
                    kind, rate, placement, reason = "X_ERROR", 2 * p, "after", "reset"
                elif op.name == "CZ":
                    kind, rate, placement, reason = "DEPOLARIZE2", p, "after", "two_qubit"
                else:
                    kind, rate, placement, reason = "DEPOLARIZE1", p / 10, "after", "one_qubit"
                channels.append(
                    NoiseChannel(
                        f"moment[{m}]/operation[{j}]/{reason}", m, j, placement, kind, op.qubits, rate, reason,
                    ),
                )
            for reason, qubits, rate in (
                ("idle", allocation - active, p / 10),
                ("collapse_wait", allocation - collapsed if collapsed else set(), 2 * p),
            ):
                channels.extend(
                    NoiseChannel(f"moment[{m}]/{reason}[{q}]", m, None, "end", "DEPOLARIZE1", (q,), rate, reason)
                    for q in sorted(qubits)
                )
        return tuple(channels)

    def describe(self) -> str:
        """Summarize rates and locations without claiming a runtime timing audit."""
        counts = Counter(c.reason for c in self.channels)
        return "\n".join(
            (
                f"SI1000 p={self.noise.p:g}; schedule={self.ideal.schedule.value}",
                f"{len(self.ideal.moments)} physical moments, {len(self.channels)} independent noise locations",
                *(f"  {reason}: {count}" for reason, count in sorted(counts.items())),
            ),
        )

    def _to_stim(self, faults: dict[str, str] | None = None) -> stim.Circuit:
        """Emit stochastic channels, or only the validated selected Pauli faults."""
        import stim  # noqa: PLC0415 - optional analysis backend

        at = defaultdict(list)
        for channel in self.channels:
            at[channel.moment, channel.operation, channel.placement].append(channel)
        circuit = stim.Circuit()

        def emit(channels: list[NoiseChannel]) -> None:
            for channel in channels:
                if faults is None:
                    circuit.append(channel.kind, channel.qubits, channel.probability)
                elif channel.id in faults:
                    for q, pauli in zip(channel.qubits, faults[channel.id], strict=True):
                        if pauli != "I":
                            # FlipSimulator tracks deviations from ideal gates;
                            # a deterministic error, unlike an ideal X/Y/Z gate,
                            # changes its Pauli frame relative to that reference.
                            circuit.append(f"{pauli}_ERROR", [q], 1)

        for m, moment in enumerate(self.ideal.moments):
            for j, op in enumerate(moment):
                emit(at[m, j, "before"])
                circuit.append(op.name, op.qubits)
                emit(at[m, j, "after"])
            emit(at[m, None, "end"])
            circuit.append("TICK")
        # Canonical event IDs, affine MPAD, and coordinates have one exporter.
        for inst in self.ideal.to_stim():
            if inst.name in {"MPAD", "DETECTOR", "OBSERVABLE_INCLUDE"}:
                circuit.append(inst)
        return circuit

    def to_stim(self) -> stim.Circuit:
        """Export the full noisy circuit; MPAD is synthetic and acquires no noise."""
        return self._to_stim()

    def compare_schedule(
        self, faults: Sequence[PauliFault], *, candidate_policy: SchedulePolicy = SchedulePolicy.ASAP_UNITARIES,
        exclude_through: int = 2, max_pair_checks: int = 10_000_000,
    ) -> ScheduledNoiseComparison:
        """Compare SI1000 fault behavior using an operation-attached reference witness."""
        from pecos.qec.generic.noise_comparison import compare_si1000_schedules  # noqa: PLC0415
        return compare_si1000_schedules(self, faults, candidate_policy=candidate_policy,
                                        exclude_through=exclude_through, max_pair_checks=max_pair_checks)

    def compile_native(self) -> NativeScheduledNoise:
        """Compile explicit channels for native PECOS execution and DEM analysis."""
        from pecos.qec.generic.native_noise import NativeScheduledNoise  # noqa: PLC0415

        return NativeScheduledNoise(self)

    def to_tick_circuit(self) -> TickCircuit:
        """Export executable inline channels, grouped into physical moments.

        Output ticks are execution steps; ``qec_moment`` retains the selected
        hardware schedule. Run using an explicit stabilizer backend and no
        additional noise model. Ideal export remains ``self.ideal.to_tick_circuit``.
        """
        from pecos.qec.generic.noisy_tick_lowering import lower_noisy_tick_circuit  # noqa: PLC0415

        return lower_noisy_tick_circuit(self)
