# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Definition-based detection-event extraction for surface memory circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pecos_rslib.quantum import TickCircuit


class MissingObservableReferenceError(ValueError):
    """An observable cannot be interpreted as a flip without its noiseless reference."""


def _validate_observable_references(
    observables: list[dict[str, object]],
) -> None:
    """Validate reader output before consuming any shots."""
    seen = set()
    for index, obs in enumerate(observables):
        entry = f"observable entry {index}"
        if "id" in obs:
            entry += f" (id={obs['id']})"
        if obs.get("reference") is None:
            msg = (
                f"{entry} is missing 'reference'; this metadata carries no reference, "
                "so a flip cannot be derived without one. "
                "Surface producers LogicalCircuitBuilder.to_tick_circuit and TickCircuitRenderer "
                "(via build_memory_circuit or generate_tick_circuit_from_patch) currently emit it."
            )
            raise MissingObservableReferenceError(msg)
        if "id" not in obs:
            msg = f"{entry} is missing 'id'"
            raise MissingObservableReferenceError(msg)
        observable_id = obs["id"]
        if isinstance(observable_id, bool) or not isinstance(observable_id, int):
            msg = f"{entry} has invalid 'id' {observable_id!r}; expected an integer"
            raise MissingObservableReferenceError(msg)
        if observable_id in seen:
            msg = f"observable entry {index} has duplicate 'id' {observable_id!r}"
            raise MissingObservableReferenceError(msg)
        seen.add(observable_id)
        reference = obs["reference"]
        if isinstance(reference, bool) or not isinstance(reference, int) or reference not in (0, 1):
            msg = f"{entry} has invalid 'reference' {reference!r}; expected integer 0 or 1"
            raise MissingObservableReferenceError(msg)


def extract_detection_events_and_observables(
    tick_circuit: TickCircuit,
    results: Iterable[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Extract fired detector ids and reference-relative observable ids in id order.

    XOR raw parity with each noiseless reference so a clean shot has no flips.
    ``pecos.testing.simulate_tick_circuit`` instead retains raw signed parity.
    Missing or invalid reader ids/references raise MissingObservableReferenceError.
    """
    definitions = tick_circuit.circuit_definitions()
    _validate_observable_references(definitions["observables"])
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
            val = obs["reference"]
            for position in obs["measurements"]:
                val ^= int(row[position])
            if val:
                flipped_observables.append(obs["id"])
        observable_flips_per_shot.append(flipped_observables)

    return detection_events_per_shot, observable_flips_per_shot
