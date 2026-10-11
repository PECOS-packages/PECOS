# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Definition-based detection-event extraction for surface memory circuits."""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, overload

from pecos._traced_circuit import measurement_ids_in_execution_order

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos_rslib.quantum import TickCircuit


class MissingObservableReferenceError(ValueError):
    """An observable cannot be interpreted as a flip without its noiseless reference."""


def _validate_observable_references(
    observables: list[dict[str, object]],
) -> None:
    """Validate reader output when observable flips are requested."""
    seen = set()
    for index, obs in enumerate(observables):
        entry = f"observable entry {index}"
        if "id" in obs:
            entry += f" (id={obs['id']})"
        if obs.get("reference") is None:
            msg = (
                f"{entry} has no 'reference', so a flip cannot be derived from its raw parity. "
                "An observable carried only by a circuit annotation has no noiseless reference; "
                "neither does metadata written before the field existed or with "
                "TickCircuit.add_observable without reference=0 or reference=1. "
                "Surface producers LogicalCircuitBuilder.to_tick_circuit and TickCircuitRenderer "
                "(via build_memory_circuit or generate_tick_circuit_from_patch) emit it."
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


class _ObservableFlips(Sequence[list[int]]):
    """Cache flips on first access, allowing detector-only use without a reference."""

    __hash__ = None

    def __init__(self, observables: list[dict[str, object]], rows: list[tuple[int, ...]]) -> None:
        self._observables = observables
        self._rows = rows

    @cached_property
    def _values(self) -> list[list[int]]:
        _validate_observable_references(self._observables)
        flips = []
        for row in self._rows:
            flipped = []
            for obs in self._observables:
                val = obs["reference"]
                for position in obs["measurements"]:
                    val ^= int(row[position])
                if val:
                    flipped.append(obs["id"])
            flips.append(flipped)
        return flips

    @overload
    def __getitem__(self, index: int) -> list[int]: ...

    @overload
    def __getitem__(self, index: slice) -> list[list[int]]: ...

    def __getitem__(self, index: int | slice) -> list[int] | list[list[int]]:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ObservableFlips):
            return self._values == other._values
        return self._values == other


def _remove_executed_correction(observable: dict, readout: dict, positions: dict[int, int]) -> None:
    """Remove one crossed correction from a copied mask, preserving multiplicity."""
    if readout.get("correction") != "executed":
        msg = f"Crossed teleportation {readout['ancilla_patch']!r} must declare correction='executed'"
        raise ValueError(msg)
    _validate_observable_references([observable])
    for mid in (*readout["meas_ids"], *readout["resource_sign_meas_ids"]):
        observable["measurements"].remove(positions[mid])
    observable["reference"] ^= readout["resource_sign_reference"]


def extract_detection_events_and_observables(
    tick_circuit: TickCircuit,
    results: Iterable[Sequence[int]],
    *,
    executed_corrections: bool = False,
) -> tuple[list[list[int]], Sequence[list[int]]]:
    """Extract fired detector ids and reference-relative observable ids in id order.

    XOR raw parity with each noiseless reference so a clean shot has no flips.
    ``pecos.testing.simulate_tick_circuit`` instead retains raw signed parity.
    Observable flips are a cached, reusable sequence: accessing it validates
    reader ids/references and raises MissingObservableReferenceError when needed.
    Discarding it leaves detector-only callers unaffected by absent references.
    For program-corrected rows, ``executed_corrections=True`` removes crossed SZ
    correction records and XORs their resource-sign references. Every crossing
    must be declared ``correction="executed"``. Static builder rows always use
    the default mask extraction, even with an executed declaration.
    """
    definitions = tick_circuit.circuit_definitions()
    observables = definitions["observables"]
    if executed_corrections:
        metadata = {obs["id"]: obs for obs in json.loads(tick_circuit.get_meta("observables") or "[]")}
        readouts = {
            readout["ancilla_patch"]: readout
            for readout in json.loads(tick_circuit.get_meta("teleportation_readouts") or "[]")
        }
        positions = {mid: position for position, mid in enumerate(measurement_ids_in_execution_order(tick_circuit))}
        observables = [dict(obs, measurements=list(obs["measurements"])) for obs in observables]
        for obs in observables:
            for label in metadata.get(obs["id"], {}).get("crossed_teleportations", []):
                _remove_executed_correction(obs, readouts[label], positions)
    num_meas = definitions["num_measurements"]
    detection_events_per_shot: list[list[int]] = []
    rows: list[tuple[int, ...]] = []

    for row in results:
        if len(row) != num_meas:
            msg = f"result row has length {len(row)} but tick_circuit emits num_measurements={num_meas}"
            raise ValueError(msg)
        # Results may be a one-shot iterator reusing a mutable buffer. Freeze
        # each row now so later observable access sees the same shot as detectors.
        rows.append(tuple(row))
        fired_detectors: list[int] = []
        for det in definitions["detectors"]:
            val = 0
            for position in det["measurements"]:
                val ^= int(row[position])
            if val:
                fired_detectors.append(det["id"])
        detection_events_per_shot.append(fired_detectors)

    return detection_events_per_shot, _ObservableFlips(observables, rows)
