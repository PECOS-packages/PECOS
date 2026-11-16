# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Inline-channel TickCircuit lowering of an already selected physical schedule."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled_noise import NoiseChannel, NoisyScheduledCircuit
    from pecos.quantum import TickCircuit


def lower_noisy_tick_circuit(noisy: NoisyScheduledCircuit) -> TickCircuit:
    """Insert channel execution steps while retaining physical-moment provenance.

    TickCircuit forbids operations sharing a qubit in one tick. The output
    therefore uses ticks as execution steps, grouped by ``qec_moment``. Only
    ``qec_execution_phase='gates'`` corresponds to an authored physical moment;
    before/after/idle/collapse_wait steps serialize channels at its boundaries.
    These steps do not introduce extra hardware time or extra noise locations.

    The inline payloads, not metadata, drive native execution. Measurement IDs,
    record order, parity constants and coordinates come from the ideal lowering.
    As there, constant-only empty-support annotations are not yet supported.
    """
    circuit = noisy.ideal.to_tick_circuit()
    channels = defaultdict(list)
    for channel in noisy.channels:
        phase = channel.reason if channel.placement == "end" else channel.placement
        channels[channel.moment, phase].append(channel)

    def insert(index: int, moment: int, phase: str, entries: list[NoiseChannel]) -> None:
        if not entries:
            return
        tick = circuit.insert_tick(index)
        circuit.set_tick_meta(index, "qec_moment", moment)
        circuit.set_tick_meta(index, "qec_execution_phase", phase)
        for channel in entries:
            if channel.kind == "X_ERROR":
                tick.x_error(channel.probability, channel.qubits[0])
            elif channel.kind == "DEPOLARIZE1":
                tick.depolarize1(channel.probability, channel.qubits[0])
            else:  # SI1000's validated alphabet has only these three channels.
                tick.depolarize2(channel.probability, channel.qubits)
            tick.meta("qec_noise_channel", asdict(channel))

    # Reverse insertion retains every existing moment index until it is visited.
    # Within a physical moment, disjoint operations/channels commute, so all
    # pre-measurement channels can precede its gates, and all post-gate channels
    # can follow them. Idle and wait remain separate independent steps.
    for moment in reversed(range(len(noisy.ideal.moments))):
        circuit.set_tick_meta(moment, "qec_execution_phase", "gates")
        for phase in ("collapse_wait", "idle", "after"):
            insert(moment + 1, moment, phase, channels[moment, phase])
        insert(moment, moment, "before", channels[moment, "before"])
    circuit.metas(
        {
            "pecos_noise_source": "inline",
            "qec_tick_semantics": "execution_steps",
            "qec_physical_moments": len(noisy.ideal.moments),
            "qec_noise_model": "SI1000",
            "qec_noise_probability": noisy.noise.p,
            "qec_noise_channels": len(noisy.channels),
        },
    )
    return circuit
