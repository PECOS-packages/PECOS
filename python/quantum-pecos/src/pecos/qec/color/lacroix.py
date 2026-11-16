# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Lacroix et al.'s readout-compatible native-CZ color-code memories.

Construction tables are extracted from Google Quantum AI's released dataset_0,
DOI 10.5281/zenodo.14238944, for arXiv:2412.14256v1, Supplement C.3/D.1.
This is a functional ideal construction: sweep bits are zero, spectator-only
resets and repeated resets of fresh |0> ancillas are omitted, and empty timing
slots are removed. It does not reproduce hardware timing or noise. X memory
has distinct first/last extraction gadgets with merged basis rotations.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from pecos.qec.generic import Gadget, GadgetProgram, Gate, NamedParity, SchedulePolicy

if TYPE_CHECKING:
    from pecos.qec.generic import ScheduledCircuit

_CITATIONS = (
    "N. Lacroix et al., Scaling and logic in the color code on a superconducting quantum processor, "
    "arXiv:2412.14256v1, Supplement C.3 and D.1; https://arxiv.org/abs/2412.14256v1",
    "Google Quantum AI, released memory circuits, dataset_0; https://doi.org/10.5281/zenodo.14238944",
)
_PATCHES = ("d3_west", "d3_east", "d3_north", "d5")


@cache
def _tables() -> dict:
    return json.loads(Path(__file__).with_name("lacroix_templates.json").read_text())


def lacroix_memory_program(patch: str = "d3_west", rounds: int = 3, basis: str = "Z") -> GadgetProgram:
    """Build the authors' square-grid memory as named boundary/recurring gadgets.

    ``patch`` is d3_west, d3_east, d3_north, or d5. At least two rounds are
    required; the first and last rounds have distinct boundary contracts.
    Three- and five-round circuits are checked against independent released
    files. Other lengths repeat the same recurring gadget and parity rules.
    Role suffixes retain the authors' qubit IDs and square-grid coordinates.
    This factory is separate from the canonical triangular 666 implementations.
    """
    if patch not in _PATCHES:
        msg = f"patch must be one of {_PATCHES}"
        raise ValueError(msg)
    if basis not in {"X", "Z"}:
        msg = "basis must be X or Z"
        raise ValueError(msg)
    if type(rounds) is not int or rounds < 2:
        msg = "rounds must be an integer of at least two"
        raise ValueError(msg)
    table = _tables()[f"{patch}_{basis}"]
    data = set(table["data"])
    names = {}
    for q in table["active"]:
        x, y = table["coords"][str(q)]
        role = "data" if q in data else "readout"
        names[q] = f"{role}[{q}]@({x:g},{y:g})"
    roles = tuple(names.values())
    data_roles = tuple(names[q] for q in table["data"])
    program = GadgetProgram(roles)
    program.append(
        Gadget(
            "Lacroix initial data",
            roles,
            (),
            data_roles,
            (tuple(Gate("R", names[q]) for q in table["data"]),),
            citations=_CITATIONS,
        ),
        label="prepare",
    )
    events: list[str] = []
    logical = NamedParity()
    sections = [("first", "round[0]")]
    sections.extend(("middle", f"round[{i}]") for i in range(1, rounds - 1))
    sections.extend((("last", f"round[{rounds - 1}]"), ("readout", "readout")))
    for section, label in sections:
        spec = table[section]
        moments = []
        outcomes = []
        for moment in spec["moments"]:
            gates = []
            for gate, qubits in moment:
                outcome = f"m[{qubits[0]}]" if gate == "M" else None
                gates.append(Gate(gate, tuple(names[q] for q in qubits), outcome))
                if outcome is not None:
                    outcomes.append(f"{label}/{outcome}")
            moments.append(tuple(gates))
        program.append(
            Gadget(
                f"Lacroix {patch} {basis} {section}",
                roles,
                data_roles,
                () if section == "readout" else data_roles,
                tuple(moments),
                citations=_CITATIONS,
            ),
            label=label,
        )
        start = len(events)
        events.extend(outcomes)

        def parity(records: list[int], start: int = start) -> NamedParity:
            indices = [start + i for i in records]
            if any(i < 0 or i >= len(events) for i in indices):
                msg = "Invalid measurement reference in Lacroix construction table"
                raise ValueError(msg)
            return NamedParity(tuple(events[i] for i in indices))

        for i, detector in enumerate(spec["detectors"]):
            x, y, _, kind = detector["coords"]
            program.detector(f"{label}:detector[{i}]@({x:g},{y:g};{kind:g})", parity(detector["records"]))
        logical ^= parity(spec["observable"])
    program.observable(f"Lacroix logical {basis}", logical)
    return program


def lacroix_memory(
    patch: str = "d3_west",
    rounds: int = 3,
    basis: str = "Z",
    *,
    schedule: SchedulePolicy = SchedulePolicy.REFERENCE,
) -> ScheduledCircuit:
    """Compile the named native-CZ memory for Guppy, TickCircuit, or Stim export."""
    return lacroix_memory_program(patch, rounds, basis).compile(schedule)
