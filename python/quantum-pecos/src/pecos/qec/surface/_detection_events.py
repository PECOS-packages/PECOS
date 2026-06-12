# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Metadata-driven detection-event extraction for surface memory circuits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any


def extract_detection_events_and_observables(
    tick_circuit: Any,
    results: Iterable[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Extract fired detectors and observables from flat measurement rows."""
    detectors_json = tick_circuit.get_meta("detectors")
    detectors = json.loads(detectors_json) if detectors_json else []

    observables_json = tick_circuit.get_meta("observables")
    observables = json.loads(observables_json) if observables_json else []

    num_meas_meta = tick_circuit.get_meta("num_measurements")
    if num_meas_meta is None or num_meas_meta == "":
        msg = (
            "extract_detection_events_and_observables requires "
            "tick_circuit.get_meta('num_measurements') to be set"
        )
        raise ValueError(msg)
    num_meas = int(num_meas_meta)

    detection_events_per_shot: list[list[int]] = []
    observable_flips_per_shot: list[list[int]] = []

    for row in results:
        if len(row) != num_meas:
            msg = (
                f"result row has length {len(row)} but tick_circuit metadata "
                f"declares num_measurements={num_meas}"
            )
            raise ValueError(msg)

        fired_detectors: list[int] = []
        for det_idx, det in enumerate(detectors):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < num_meas:
                    val ^= int(row[idx])
            if val:
                fired_detectors.append(det_idx)
        detection_events_per_shot.append(fired_detectors)

        flipped_observables: list[int] = []
        for obs_idx, obs in enumerate(observables):
            val = 0
            for rec in obs["records"]:
                idx = num_meas + rec
                if 0 <= idx < num_meas:
                    val ^= int(row[idx])
            if val:
                flipped_observables.append(obs_idx)
        observable_flips_per_shot.append(flipped_observables)

    return detection_events_per_shot, observable_flips_per_shot
