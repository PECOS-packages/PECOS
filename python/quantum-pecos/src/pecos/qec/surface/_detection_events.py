# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Metadata-driven detection-event extraction for surface circuits."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class MissingObservableReferenceError(ValueError):
    """An observable cannot be interpreted as a flip without its noiseless reference."""


def _validate_observable_reference(obs: dict[str, object], entry_index: int) -> None:
    """Require an id and an integer reference bit before interpreting an observable."""
    entry = f"observable entry {entry_index}"
    if "id" in obs:
        entry += f" (id={obs['id']})"
    if "reference" not in obs:
        msg = (
            f"{entry} is missing 'reference'; this metadata predates the reference field. "
            "Surface producers LogicalCircuitBuilder.to_tick_circuit and TickCircuitRenderer "
            "(via build_memory_circuit or generate_tick_circuit_from_patch) emit it."
        )
        raise MissingObservableReferenceError(msg)
    if "id" not in obs:
        msg = f"{entry} is missing 'id'"
        raise MissingObservableReferenceError(msg)
    reference = obs["reference"]
    if isinstance(reference, bool) or not isinstance(reference, int) or reference not in (0, 1):
        msg = f"{entry} has invalid 'reference' {reference!r}; expected integer 0 or 1"
        raise MissingObservableReferenceError(msg)


class _TickCircuitLike(Protocol):
    def get_meta(self, key: str) -> str | None:
        """Return metadata stored under ``key`` when available."""
        ...


def _record_offsets(entry: dict[str, object], num_measurements: int) -> list[int]:
    records = entry.get("records")
    if records is not None:
        return [int(record) for record in records]  # type: ignore[union-attr]
    meas_ids = entry.get("meas_ids")
    if meas_ids is not None:
        return [int(meas_id) - num_measurements for meas_id in meas_ids]  # type: ignore[union-attr]
    msg = "detector/observable metadata entry must define either 'records' or 'meas_ids'"
    raise ValueError(msg)


def extract_detection_events_and_observables(
    tick_circuit: _TickCircuitLike,
    results: Iterable[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Extract fired detector positions and reference-relative observable ids.

    Each observable's raw record parity is XORed with its noiseless ``reference``
    bit. Unlike ``pecos.testing.simulate_tick_circuit``, a clean shot therefore
    has no observable flips, even when its signed raw parity is one.

    Raises:
        MissingObservableReferenceError: An observable entry lacks ``id`` or an integer ``reference`` bit (0 or 1).
        ValueError: Measurement count metadata is missing or a row has the wrong length.
    """
    detectors_json = tick_circuit.get_meta("detectors")
    detectors = json.loads(detectors_json) if detectors_json else []

    observables_json = tick_circuit.get_meta("observables")
    observables = json.loads(observables_json) if observables_json else []

    for entry_index, obs in enumerate(observables):
        _validate_observable_reference(obs, entry_index)

    num_meas_meta = tick_circuit.get_meta("num_measurements")
    if num_meas_meta is None or num_meas_meta == "":
        msg = "extract_detection_events_and_observables requires tick_circuit.get_meta('num_measurements') to be set"
        raise ValueError(msg)
    num_meas = int(num_meas_meta)

    detection_events_per_shot: list[list[int]] = []
    observable_flips_per_shot: list[list[int]] = []

    for row in results:
        if len(row) != num_meas:
            msg = f"result row has length {len(row)} but tick_circuit metadata declares num_measurements={num_meas}"
            raise ValueError(msg)

        fired_detectors: list[int] = []
        for det_idx, det in enumerate(detectors):
            val = 0
            for rec in _record_offsets(det, num_meas):
                idx = num_meas + rec
                if 0 <= idx < num_meas:
                    val ^= int(row[idx])
            if val:
                fired_detectors.append(det_idx)
        detection_events_per_shot.append(fired_detectors)

        flipped_observables: list[int] = []
        for obs in observables:
            val = obs["reference"]
            for rec in _record_offsets(obs, num_meas):
                idx = num_meas + rec
                if 0 <= idx < num_meas:
                    val ^= int(row[idx])
            if val:
                flipped_observables.append(obs["id"])
        observable_flips_per_shot.append(flipped_observables)

    return detection_events_per_shot, observable_flips_per_shot
