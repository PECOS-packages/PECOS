#!/usr/bin/env python3
"""Original 27-moment compiler and generic exact retiming/resource utilities."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
from .rule import build_schedule

def gate(name: str, targets) -> dict:
    return {"gate": name, "targets": list(targets)}


def baseline(distance: int) -> dict:
    """The starter's exact stationary CX-to-CZ compiler, independently callable."""
    if type(distance) is not int or distance < 4 or distance % 4:
        raise ValueError("distance must be a positive multiple of four")
    schedule = build_schedule(distance)
    data = set(range(distance * distance))

    def compile_with_frame(frame: set[int]):
        moments = [[gate("R", schedule["measurement_order"])]]
        pending = set(schedule["h_prepare"]) ^ frame
        for layer in schedule["layers"]:
            targets = {b for _, b in layer}
            active = {q for pair in layer for q in pair}
            pending ^= targets
            hadamards = pending & active
            pending -= hadamards
            if hadamards:
                moments.append([gate("H", sorted(hadamards))])
            moments.append([gate("CZ", [q for pair in layer for q in pair])])
            pending ^= targets
        pending ^= set(schedule["h_measure"])
        hadamards = pending - data
        if hadamards:
            moments.append([gate("H", sorted(hadamards))])
        moments.append([gate("M", schedule["measurement_order"])])
        return moments, pending & data

    _, frame = compile_with_frame(set())
    moments, output_frame = compile_with_frame(frame)
    assert frame == output_frame
    assert len(moments) == 27
    return dict(schema="torus488.cz_cycle/1", distance=distance,
                measurement_order=schedule["measurement_order"],
                data_h_frame=sorted(frame), moments=moments)


def atoms(candidate: dict):
    """Atomic single-qubit gates and CZ pairs, in the given physical order."""
    for phase, moment in enumerate(candidate["moments"]):
        for operation in moment:
            name, qs = operation["gate"], operation["targets"]
            width = 2 if name == "CZ" else 1
            for start in range(0, len(qs), width):
                yield phase, name, tuple(qs[start:start + width])


def asap_events(candidate: dict):
    """ASAP topological schedule preserving the entire operation order on each wire."""
    last = [-1] * (2 * candidate["distance"] ** 2)
    events = []
    for phase, name, qs in atoms(candidate):
        if name == "M":
            continue
        time = 1 + max(last[q] for q in qs)
        for q in qs:
            last[q] = time
        events.append((phase, name, qs, time))
    # All ancillas are still read together; no asynchronous boundary is used.
    time = max(last) + 1
    for q in candidate["measurement_order"]:
        events.append((len(candidate["moments"]) - 1, "M", (q,), time))
    return events


def compact(candidate: dict) -> dict:
    events = asap_events(candidate)
    moments = [{} for _ in range(max(t for _, _, _, t in events) + 1)]
    for _, name, qs, time in events:
        moments[time].setdefault(name, []).extend(qs)
    return dict(candidate, moments=[
        [gate(name, qs) for name, qs in moment.items()] for moment in moments])


def periodic_key(n: int, phase: int, name: str, qs: tuple) -> str:
    local = [q % 16 + 16 * int(q >= n) for q in qs]
    return f"{phase}:{name}:" + ",".join(map(str, local))


def cycle_census(candidate: dict) -> dict:
    n = candidate["distance"] ** 2
    counts, gates = Counter(), Counter()
    for moment in candidate["moments"]:
        active, collapsed, names = set(), set(), set()
        for operation in moment:
            name, qs = operation["gate"], operation["targets"]
            assert len(set(qs)) == len(qs) and not active.intersection(qs)
            active.update(qs)
            names.add(name)
            if name in {"R", "M"}:
                collapsed.update(qs)
            gates[name] += len(qs) // (2 if name == "CZ" else 1)
        counts["physical_moments"] += 1
        counts["ordinary_idle_qubit_moments"] += 2 * n - len(active)
        counts["CZ_layers"] += int("CZ" in names)
        counts["H_moments"] += int("H" in names)
        if collapsed:
            counts["extra_collapse_wait_qubit_moments"] += 2 * n - len(collapsed)
    return dict(counts, gates=dict(gates))


