# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Cited color-code specifications and selectable extraction implementations.

References:
    N. Shutty, Denser Planar Color Codes, arXiv:2609.21376v3.
    A. Lavasani et al., Quantum error correction well below the color code
    threshold, arXiv:2610.24077v1, Fig. 1 and Supplement B.

The 4.8.8 rule and instrument algebra are from the Apache-2.0 source release;
see _shutty488/NOTICE.md. The 6.6.6 construction expresses the paper's face
circuits and lattice convention also used by the local color-code experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

from pecos.qec.color._shutty488 import cx_rule, geometry, ideal_css
from pecos.qec.generic.gadgets import Gadget, GadgetFlow, NamedParity, NamedPauli
from pecos.qec.generic.scheduled import CircuitOp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos.qec.generic.gadget_flows import GadgetFlowAudit, PauliFlow

SHUTTY = "https://arxiv.org/abs/2609.21376v3"
LAVASANI = "https://arxiv.org/abs/2610.24077v1"
Moment = tuple[CircuitOp, ...]


class ColorExtraction(Enum):
    """Versioned construction choices; incompatible code families are rejected."""

    BRICKWORK_CX = "shutty488_cx_v1"
    BRICKWORK_CZ = "shutty488_native_cz_v1"
    SUPERDENSE = "color666_superdense_v1"
    SWAP_SUPERDENSE = "lavasani666_swap_superdense_v1"


@dataclass(frozen=True)
class CSSCodeSpec:
    """Code algebra and coordinates, independent of extraction ancillas."""

    family: str
    distance: int
    data_coordinates: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]
    logical_support: tuple[int, ...]
    citation: str
    # Oriented hexagon slots; -1 marks absent boundary vertices.
    face_slots: tuple[tuple[int, ...], ...] = ()

    @property
    def num_data(self) -> int:
        """Number of data positions in the code."""
        return len(self.data_coordinates)

    @property
    def data_roles(self) -> tuple[str, ...]:
        """Stable human-readable names for the data positions."""
        return tuple(f"data[{q}]" for q in range(self.num_data))

    @property
    def checks(self) -> tuple[NamedPauli, ...]:
        """Named X/Z stabilizers, independent of extraction and scheduling."""
        return tuple(
            NamedPauli(f"{axis}[{face}]", tuple((self.data_roles[q], axis) for q in support))
            for face, support in enumerate(self.faces)
            for axis in "XZ"
        )

    @property
    def logical_operators(self) -> tuple[NamedPauli, ...]:
        """Named representatives of the encoded X and Z operators."""
        return tuple(
            NamedPauli(f"logical:{axis}", tuple((self.data_roles[q], axis) for q in self.logical_support))
            for axis in "XZ"
        )

    def describe(self) -> str:
        """Describe the code using named data roles, checks, and logicals."""
        return "\n".join(
            (
                f"{self.family} distance {self.distance}: {self.num_data} data roles",
                *(str(check) for check in (*self.checks, *self.logical_operators)),
            ),
        )


@dataclass(frozen=True)
class CSSFlow:
    """Exact incoming/outgoing check and logical parities in raw local records."""

    input_x: tuple[int, ...]
    output_x: tuple[int, ...]
    input_z: tuple[int, ...]
    output_z: tuple[int, ...]
    logical_x: int
    logical_z: int
    constraints: tuple[int, ...]

    @classmethod
    def from_reference(cls, maps: dict) -> CSSFlow:
        """Translate the pinned reference's binary parity maps."""
        return cls(
            tuple(maps["input_face_record_maps"]["X"]),
            tuple(maps["output_face_record_maps"]["X"]),
            tuple(maps["input_face_record_maps"]["Z"]),
            tuple(maps["output_face_record_maps"]["Z"]),
            maps["logical_choi_signs"]["X"][0],
            maps["logical_choi_signs"]["Z"][0],
            tuple(maps["record_constraints"]),
        )


@dataclass(frozen=True)
class ExtractionCycle:
    """One concrete extraction phase, including its physical basis convention."""

    moments: tuple[Moment, ...]
    flow: CSSFlow
    input_h_frame: tuple[int, ...] = ()
    output_h_frame: tuple[int, ...] = ()


@dataclass(frozen=True)
class ColorCodeImplementation:
    """A code specification bound to a compatible extraction implementation."""

    code: CSSCodeSpec
    extraction: ColorExtraction
    num_ancillas: int
    cycles: tuple[ExtractionCycle, ...]

    @property
    def num_qubits(self) -> int:
        """Peak number of physical carriers in the reference allocation."""
        return self.code.num_data + self.num_ancillas

    @property
    def period(self) -> int:
        """Number of phases before the schedule counter repeats."""
        return len(self.cycles)

    @property
    def roles(self) -> tuple[str, ...]:
        """Named data and extraction ancilla roles in allocation order."""
        ancillas = (
            tuple(f"ancilla[{i}]" for i in range(self.num_ancillas))
            if self.code.family == "488"
            else tuple(f"ancilla_{axis}[{face}]" for face in range(len(self.code.faces)) for axis in "xz")
        )
        return (*self.code.data_roles, *ancillas)

    def gadget(self, phase: int = 0) -> Gadget:
        """Expose an extraction phase as a reusable named gadget."""
        return extraction_gadget(self, phase)

    def flow_contracts(self, phase: int = 0) -> tuple[PauliFlow, ...]:
        """State the published input/output checks, logical transport, and constraints.

        Operators are in the code's named data roles; the gadget audit accounts
        for its physical H frames and carrier relabeling. These are ideal
        extraction contracts, not a complete fault-tolerance certificate.
        """
        from pecos.qec.generic.gadget_flows import PauliFlow  # noqa: PLC0415

        gadget = self.gadget(phase)
        parities = {f.name: f.parity for f in gadget.flows}
        result = []
        for check in self.code.checks:
            result.append(PauliFlow(f"input:{check.name}", input=check, parity=parities[f"input:{check.name}"]))
            result.append(PauliFlow(f"output:{check.name}", output=check, parity=parities[f"output:{check.name}"]))
        result.extend(
            PauliFlow(logical.name, input=logical, output=logical, parity=parities[logical.name])
            for logical in self.code.logical_operators
        )
        result.extend(PauliFlow(f.name, parity=f.parity) for f in gadget.flows if f.name.startswith("constraint["))
        return tuple(result)

    def audit_flows(self, phase: int = 0) -> GadgetFlowAudit:
        """Independently check the signed extraction contracts for one phase."""
        return self.gadget(phase).audit_flows(self.flow_contracts(phase))


def extraction_gadget(implementation: ColorCodeImplementation, phase: int = 0) -> Gadget:
    """Adapt an exact extraction cycle into named ports, outcomes, and flows.

    Check relations refer to raw local measurements. A single ancilla bit need
    not equal a check; input and output relations remain separately inspectable.
    """
    if type(phase) is not int or not 0 <= phase < implementation.period:
        msg = "phase is outside this implementation's period"
        raise ValueError(msg)
    cycle = implementation.cycles[phase]
    roles = implementation.roles
    outcomes = tuple(roles[op.qubits[0]] for moment in cycle.moments for op in moment if op.name == "M")

    def parity(mask: int) -> NamedParity:
        return NamedParity(tuple(name for index, name in enumerate(outcomes) if (mask >> index) & 1))

    flows = []
    for kind, maps in (
        ("input:X", cycle.flow.input_x),
        ("output:X", cycle.flow.output_x),
        ("input:Z", cycle.flow.input_z),
        ("output:Z", cycle.flow.output_z),
    ):
        flows.extend(GadgetFlow(f"{kind}[{face}]", parity(mask)) for face, mask in enumerate(maps))
    flows.extend(
        (GadgetFlow("logical:X", parity(cycle.flow.logical_x)), GadgetFlow("logical:Z", parity(cycle.flow.logical_z))),
    )
    flows.extend(GadgetFlow(f"constraint[{i}]", parity(mask)) for i, mask in enumerate(cycle.flow.constraints))
    return Gadget.from_operations(
        f"{implementation.extraction.value}:phase[{phase}]",
        roles,
        cycle.moments,
        inputs=implementation.code.data_roles,
        outputs=implementation.code.data_roles,
        outcomes=outcomes,
        input_h_frame=tuple(roles[q] for q in cycle.input_h_frame),
        output_h_frame=tuple(roles[q] for q in cycle.output_h_frame),
        flows=flows,
        citations=(implementation.code.citation,),
    )


def _single(name: str, qubits: Iterable[int]) -> Moment:
    return tuple(CircuitOp(name, (q,)) for q in qubits)


def _pairs(name: str, pairs: Iterable[Iterable[int]]) -> Moment:
    return tuple(CircuitOp(name, tuple(pair)) for pair in pairs)


def _validate_distance(distance: int, modulus: int, remainder: int) -> None:
    if type(distance) is not int or distance < 3 or distance % modulus != remainder:
        msg = f"distance must be >= 3 and congruent to {remainder} modulo {modulus}"
        raise ValueError(msg)


@lru_cache(maxsize=32, typed=True)
def color_code_spec(family: str, distance: int) -> CSSCodeSpec:
    """Construct a planar 4.8.8 brickwork or triangular 6.6.6 code."""
    if family == "488":
        _validate_distance(distance, 4, 3)
        layout = geometry.generate_base_layout((distance + 1) // 4)
        return CSSCodeSpec(
            family,
            distance,
            tuple(map(tuple, layout["qubit_coords"])),
            tuple(tuple(face["qubits"]) for face in layout["plaquettes"]),
            tuple(range(len(layout["qubit_coords"]))),
            SHUTTY,
        )
    if family != "666":
        msg = "family must be '488' or '666'"
        raise ValueError(msg)
    _validate_distance(distance, 2, 1)
    radius = distance // 2
    sites = []
    left, width = -radius, 0
    for row in range(3 * radius, -1, -1):
        residue = row % 3
        if residue == 1:
            left += 1
        elif residue == 0:
            width += 1
        for column in range(left, left + width):
            if residue != 2 or column != left:
                sites.append((column, row, 0))
            if residue != 0 or column != left + width - 1:
                sites.append((column, row, 1))
    at = {site: i for i, site in enumerate(sites)}

    def face(column: int, row: int) -> tuple[int, ...]:
        return tuple(
            at.get(site, -1)
            for site in (
                (column, row, 0),
                (column, row, 1),
                (column, row + 1, 0),
                (column - 1, row + 2, 1),
                (column - 1, row + 2, 0),
                (column - 1, row + 1, 1),
            )
        )

    slots = []
    left, width = -radius, 0
    for row in range(3 * radius, -1, -1):
        if row % 3 == 1:
            left += 1
        elif row % 3 == 0:
            width += 1
        if row % 3 == 1:
            for column in range(left, left + width):
                slots.extend((face(column, row), face(column, row - 1), face(column + 1, row - 2)))
    # Integer embedding: x in half lattice units; y in sqrt(3)/2 units.
    coordinates = tuple((6 * column + 3 * row + 2 * sublattice, row) for column, row, sublattice in sites)
    return CSSCodeSpec(
        family,
        distance,
        coordinates,
        tuple(tuple(q for q in face if q >= 0) for face in slots),
        tuple(range(len(sites))),
        LAVASANI,
        tuple(slots),
    )


def _cx_moments(schedule: dict, n: int, total: int) -> tuple[Moment, ...]:
    return tuple(
        moment
        for moment in (
            _single("R", range(n, total)),
            _single("H", schedule["h_prepare"]),
            *(_pairs("CX", layer) for layer in schedule["layers"]),
            _single("H", schedule["h_measure"]),
            _single("M", schedule["measurement_order"]),
        )
        if moment
    )


def _native_cz(schedule: dict, n: int, total: int, incoming: tuple, outgoing: tuple) -> tuple[Moment, ...]:
    """Appendix E.2 Hadamard-frame lowering and per-wire ASAP packing."""
    operations = list(_single("R", range(n, total)))
    pending = set(schedule["h_prepare"]) ^ set(incoming)
    for layer in schedule["layers"]:
        targets = {b for _, b in layer}
        endpoints = {q for pair in layer for q in pair}
        operations.extend(_single("H", sorted((pending ^ targets) & endpoints)))
        operations.extend(_pairs("CZ", layer))
        pending = (pending - endpoints) | targets
    operations.extend(_single("H", sorted(pending ^ set(schedule["h_measure"]) ^ set(outgoing))))
    moments: list[list[CircuitOp]] = []
    last: dict[int, int] = {}
    for op in operations:
        depth = max((last.get(q, -1) for q in op.qubits), default=-1) + 1
        while len(moments) <= depth:
            moments.append([])
        moments[depth].append(op)
        for q in op.qubits:
            last[q] = depth
    return (*map(tuple, moments), _single("M", schedule["measurement_order"]))


def _brickwork(code: CSSCodeSpec, extraction: ColorExtraction) -> ColorCodeImplementation:
    layout = geometry.generate_base_layout((code.distance + 1) // 4)
    n, na = code.num_data, len(layout["ancilla_coords"])
    forward = cx_rule.build_cycle(layout, 0)
    if extraction == ColorExtraction.BRICKWORK_CZ:
        # Bell preparation polarity optimization from Appendix E.2.
        routing = set(layout["routing_ancillas"])
        feeds_routing = {a for a, b in forward["layers"][1] if b in routing}
        prepare = set(forward["h_prepare"])
        for edge in forward["layers"][0]:
            a, b = edge
            if b in feeds_routing:
                edge[:] = [b, a]
                prepare ^= {a, b}
        forward["h_prepare"] = sorted(prepare)
    reverse = dict(
        forward,
        layers=list(reversed(forward["layers"])),
        h_prepare=forward["h_measure"],
        h_measure=forward["h_prepare"],
    )
    schedules = (forward, reverse)
    cycles = []
    for phase, schedule in enumerate(schedules):
        flow = CSSFlow.from_reference(ideal_css.instrument(layout, schedule))
        if extraction == ColorExtraction.BRICKWORK_CZ:
            incoming = tuple(range(n)) if phase == 0 else ()
            outgoing = () if phase == 0 else tuple(range(n))
            cycles.append(
                ExtractionCycle(_native_cz(schedule, n, n + na, incoming, outgoing), flow, incoming, outgoing),
            )
        else:
            cycles.append(ExtractionCycle(_cx_moments(schedule, n, n + na), flow))
    return ColorCodeImplementation(code, extraction, na, tuple(cycles))


def _face_moment(code: CSSCodeSpec, name: str, pairs: Iterable[tuple[int, ...]]) -> Moment:
    operations = []
    for face_index, slots in enumerate(code.face_slots):
        roles = (*slots, code.num_data + 2 * face_index, code.num_data + 2 * face_index + 1)
        for pair in pairs:
            qubits = tuple(roles[q] for q in pair)
            if -1 not in qubits:
                operations.append(CircuitOp(name, qubits))
    return tuple(operations)


def bell_flag_projection(code: CSSCodeSpec, basis: str) -> tuple[Moment, ...]:
    """Measure checks opposite the memory basis, including decoder flag records.

    Implements arXiv:2610.24077v1 Supplement B. Data must already be prepared
    in the specified basis. Outcomes are face syndromes followed by flags.
    """
    if code.family != "666" or basis not in {"X", "Z"}:
        msg = "Bell-flag projection requires a 666 code and X/Z memory basis"
        raise ValueError(msg)
    n, nf = code.num_data, len(code.faces)
    a, b = range(n, n + 2 * nf, 2), range(n + 1, n + 2 * nf, 2)
    moments = []
    if basis == "Z":
        moments.append(_single("H", range(n)))
    moments.extend(
        (
            _single("R", range(n, n + 2 * nf)),
            _single("H", range(n, n + 2 * nf)),
            _face_moment(code, "CZ", ((6, 7),)),
            _single("H", a),
            *(_face_moment(code, "CZ", pairs) for pairs in (((6, 1), (7, 2)), ((6, 3), (7, 0)), ((6, 5), (7, 4)))),
            _single("H", b),
            _face_moment(code, "CZ", ((6, 7),)),
            _single("H", range(n, n + 2 * nf)),
            _single("M", (*a, *b)),
        ),
    )
    if basis == "Z":
        moments.append(_single("H", range(n)))
    return tuple(moment for moment in moments if moment)


def _hexagonal(code: CSSCodeSpec, extraction: ColorExtraction) -> ColorCodeImplementation:
    n, nf = code.num_data, len(code.faces)
    na = 2 * nf
    a, b = tuple(range(n, n + na, 2)), tuple(range(n + 1, n + na, 2))
    cycles = []
    phases = range(3) if extraction == ColorExtraction.SWAP_SUPERDENSE else range(1)
    for phase in phases:
        offset = 2 * phase
        s = tuple((q + offset) % 6 for q in range(6))
        body = [_face_moment(code, "CX", ((6, 7),))]
        if extraction == ColorExtraction.SWAP_SUPERDENSE:
            body.extend(_face_moment(code, "CX", ((s[i], 6), (s[i + 1], 7))) for i in (0, 2))
            body.append(_face_moment(code, "RELABEL", ((s[4], 6), (s[5], 7))))
            body.append(_face_moment(code, "CX", ((s[4], 6), (s[5], 7))))
            body.extend(_face_moment(code, "CX", ((6, s[i]), (7, s[i + 1]))) for i in (2, 0))
        else:
            body.extend(_face_moment(code, "CX", ((s[i], 6), (s[i + 1], 7))) for i in (0, 2, 4))
            body.extend(_face_moment(code, "CX", ((6, s[i]), (7, s[i + 1]))) for i in (0, 2, 4))
        body.append(_face_moment(code, "CX", ((6, 7),)))
        # Only the ideal verifier expands relabels into SWAP's three CNOTs.
        layers = []
        for moment in body:
            if moment and moment[0].name == "RELABEL":
                layers.extend(
                    [list(reversed(op.qubits)) if reverse else list(op.qubits) for op in moment]
                    for reverse in (False, True, False)
                )
            else:
                layers.append([list(op.qubits) for op in moment])
        schedule = {"layers": layers, "h_prepare": a, "h_measure": a, "measurement_order": (*a, *b)}
        layout = {
            "qubit_coords": code.data_coordinates,
            "ancilla_coords": [()] * na,
            "plaquettes": [{"qubits": face} for face in code.faces],
        }
        flow = CSSFlow.from_reference(ideal_css.instrument(layout, schedule))
        moments = (_single("R", range(n, n + na)), _single("H", a), *body, _single("H", a), _single("M", (*a, *b)))
        cycles.append(ExtractionCycle(tuple(moment for moment in moments if moment), flow))
    return ColorCodeImplementation(code, extraction, na, tuple(cycles))


@lru_cache(maxsize=32)
def color_code_implementation(code: CSSCodeSpec, extraction: ColorExtraction) -> ColorCodeImplementation:
    """Bind an explicit compatible construction; never silently substitute one."""
    if not isinstance(extraction, ColorExtraction):
        msg = "extraction must be a ColorExtraction member"
        raise TypeError(msg)
    if code != color_code_spec(code.family, code.distance):
        msg = "Paper implementations require the canonical code specification"
        raise ValueError(msg)
    if code.family == "488" and extraction in {ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ}:
        return _brickwork(code, extraction)
    if code.family == "666" and extraction in {ColorExtraction.SUPERDENSE, ColorExtraction.SWAP_SUPERDENSE}:
        return _hexagonal(code, extraction)
    msg = f"{extraction.value} does not implement the {code.family} family"
    raise ValueError(msg)
