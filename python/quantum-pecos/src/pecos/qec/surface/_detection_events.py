# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Definition-based detection-event extraction for surface memory circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pecos_rslib.quantum import TickCircuit


def extract_detection_events_and_observables(
    tick_circuit: TickCircuit,
    results: Iterable[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Extract fired definition ids, in id order, from flat measurement rows."""
    definitions = tick_circuit.circuit_definitions()
    num_meas = definitions["num_measurements"]
    detection_events_per_shot: list[list[int]] = []
    observable_flips_per_shot: list[list[int]] = []

    for row in results:
        if len(row) != num_meas:
            msg = f"result row has length {len(row)} but tick_circuit emits num_measurements={num_meas}"
            raise ValueError(msg)

        fired_detectors: list[int] = []
        for det in definitions["detectors"]:
            val = 0
            for position in det["measurements"]:
                val ^= int(row[position])
            if val:
                fired_detectors.append(det["id"])
        detection_events_per_shot.append(fired_detectors)

        flipped_observables: list[int] = []
        for obs in definitions["observables"]:
            val = 0
            for position in obs["measurements"]:
                val ^= int(row[position])
            if val:
                flipped_observables.append(obs["id"])
        observable_flips_per_shot.append(flipped_observables)

    return detection_events_per_shot, observable_flips_per_shot
