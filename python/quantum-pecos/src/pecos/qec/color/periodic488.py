# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Four-logical-qubit periodic 4.8.8 construction from Shutty's source release.

The distance-4m torus uses d² data and d² ancillas. This adapter preserves the
released twelve-CNOT-layer and 25-moment native-CZ memory conventions,
including global color dependencies and independent intrinsic record constraints.
It does not assert a circuit-distance proof or reproduce the SI1000 benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from typing import TYPE_CHECKING

from pecos.qec.color._shutty_torus488 import geometry, instrument, make_memory, rule
from pecos.qec.color._shutty_torus488_cz import check as cz_check
from pecos.qec.color._shutty_torus488_cz import family as cz_family
from pecos.qec.color._shutty_torus488_cz import memory25
from pecos.qec.generic import CircuitOp, Gadget, GadgetFlow, GadgetProgram, NamedParity, NamedPauli, SchedulePolicy

if TYPE_CHECKING:
    from pecos.qec.generic import GadgetFlowAudit, PauliFlow, ScheduledCircuit

_CITATIONS = (
    "Noah Shutty, Denser Planar Color Codes, arXiv:2609.21376v3; https://arxiv.org/abs/2609.21376v3",
    "Periodic 4.8.8 source release; https://doi.org/10.5281/zenodo.22820614",
)


def _distance(distance: int) -> None:
    if type(distance) is not int or distance < 4 or distance % 4:
        msg = "distance must be a positive multiple of four"
        raise ValueError(msg)


@dataclass(frozen=True)
class Periodic488CodeSpec:
    """Periodic code geometry and the author's four canonical logical pairs."""

    distance: int
    data_coordinates: tuple[tuple[int, int], ...]
    faces: tuple[tuple[int, ...], ...]
    face_colors: tuple[str, ...]
    logical_x_supports: tuple[tuple[int, ...], ...]
    logical_z_supports: tuple[tuple[int, ...], ...]
    period_vectors: tuple[tuple[int, int], ...]

    @property
    def num_data(self) -> int:
        """Number of data roles in the periodic quotient."""
        return len(self.data_coordinates)

    @property
    def num_logicals(self) -> int:
        """Number of independently encoded qubits."""
        return len(self.logical_x_supports)

    @property
    def data_roles(self) -> tuple[str, ...]:
        """Stable names in the author's data-index order."""
        return tuple(f"data[{q}]" for q in range(self.num_data))

    @property
    def checks(self) -> tuple[NamedPauli, ...]:
        """All face checks, including the dependent global-color presentation."""
        return tuple(
            NamedPauli(f"{axis}[{i}]:{self.face_colors[i]}", tuple((f"data[{q}]", axis) for q in support))
            for i, support in enumerate(self.faces)
            for axis in "XZ"
        )

    @property
    def logical_operators(self) -> tuple[NamedPauli, ...]:
        """Canonical paired X/Z representatives; X[i] anticommutes only with Z[i]."""
        return tuple(
            NamedPauli(f"logical:{axis}[{i}]", tuple((f"data[{q}]", axis) for q in supports[i]))
            for i in range(self.num_logicals)
            for axis, supports in (("X", self.logical_x_supports), ("Z", self.logical_z_supports))
        )

    def describe(self) -> str:
        """Display geometry, topology, named checks, and four logical pairs."""
        return "\n".join(
            (
                f"Periodic 4.8.8 distance {self.distance}: {self.num_data} data roles, {self.num_logicals} logicals",
                f"Periods: {self.period_vectors}",
                *(str(p) for p in (*self.checks, *self.logical_operators)),
            ),
        )


@cache
def _geometry(distance: int) -> dict:
    return geometry.build_geometry(distance)


def periodic_488_code(distance: int) -> Periodic488CodeSpec:
    """Construct the author's canonical distance-4m torus and logical basis."""
    _distance(distance)
    g = _geometry(distance)
    return Periodic488CodeSpec(
        distance,
        tuple(tuple(row["coord"]) for row in g["data_qubits"]),
        tuple(tuple(f["data_support"]) for f in g["plaquettes"]),
        tuple(f["color"] for f in g["plaquettes"]),
        tuple(tuple(s) for s in g["canonical_logicals"]["x"]),
        tuple(tuple(s) for s in g["canonical_logicals"]["z"]),
        tuple(tuple(v) for v in g["period_vectors"]),
    )


@cache
def _components(distance: int) -> tuple[dict, dict]:
    g = _geometry(distance)
    schedule = rule.build_schedule(distance)
    geometry.verify_schedule(g, schedule)
    audit = instrument.instrument(schedule, [f["data_support"] for f in g["plaquettes"]], g["canonical_logicals"])
    if audit["status"] != "PASS":
        msg = "Periodic extraction failed the author's signed-instrument validation"
        raise ValueError(msg)
    return schedule, audit


@cache
def _native_components(distance: int) -> tuple[dict, dict]:
    cycle = cz_family.make_cycle(distance)
    return cycle, cz_check.validate(cycle, _geometry(distance))


@dataclass(frozen=True)
class Periodic488Implementation:
    """A periodic code bound to a released CNOT or native-CZ extraction."""

    code: Periodic488CodeSpec
    extraction: Gadget
    ancilla_coordinates: tuple[tuple[int, int], ...]
    hardware_edges: tuple[tuple[int, int], ...]
    gate_set: str = "cnot"

    @property
    def num_qubits(self) -> int:
        """Reference allocation includes one ancilla per data role."""
        return 2 * self.code.num_data

    @property
    def roles(self) -> tuple[str, ...]:
        """Named roles in physical allocation order."""
        return self.extraction.roles

    def gadget(self) -> Gadget:
        """Reusable single-phase extraction with named input/output check flows."""
        return self.extraction

    def flow_contracts(self) -> tuple[PauliFlow, ...]:
        """State every published face, logical-pair, and intrinsic record relation.

        Face names retain their color labels, while parity-map keys use the
        author's face indices. Dependent global checks are retained explicitly.
        """
        from pecos.qec.generic import PauliFlow  # noqa: PLC0415

        parities = {f.name: f.parity for f in self.extraction.flows}
        contracts = []
        for face, support in enumerate(self.code.faces):
            for axis in "XZ":
                check = NamedPauli(
                    f"{axis}[{face}]:{self.code.face_colors[face]}",
                    tuple((self.code.data_roles[q], axis) for q in support),
                )
                name = f"input:{axis}[{face}]"
                contracts.append(PauliFlow(name, input=check, parity=parities[name]))
                name = f"output:{axis}[{face}]"
                contracts.append(PauliFlow(name, output=check, parity=parities[name]))
        contracts.extend(
            PauliFlow(op.name, input=op, output=op, parity=parities[op.name]) for op in self.code.logical_operators
        )
        contracts.extend(
            PauliFlow(f.name, parity=f.parity) for f in self.extraction.flows if f.name.startswith("constraint[")
        )
        return tuple(contracts)

    def audit_flows(self) -> GadgetFlowAudit:
        """Check the signed ideal extraction maps in physical boundary frames."""
        return self.extraction.audit_flows(self.flow_contracts())


def periodic_488_implementation(code: Periodic488CodeSpec, *, gate_set: str = "cnot") -> Periodic488Implementation:
    """Bind canonical geometry to the CNOT or 25-moment native-CZ rule."""
    if not isinstance(code, Periodic488CodeSpec) or code != periodic_488_code(code.distance):
        msg = "Expected the canonical code returned by periodic_488_code"
        raise ValueError(msg)
    if gate_set not in {"cnot", "native-cz"}:
        msg = "gate_set must be cnot or native-cz"
        raise ValueError(msg)
    schedule, audit = (_components if gate_set == "cnot" else _native_components)(code.distance)
    n = code.num_data
    roles = (*code.data_roles, *(f"ancilla[{q}]" for q in range(n)))
    if gate_set == "cnot":
        moments = [tuple(CircuitOp("R", (q,)) for q in schedule["measurement_order"])]
        moments.append(tuple(CircuitOp("H", (q,)) for q in schedule["h_prepare"]))
        moments.extend(tuple(CircuitOp("CX", tuple(pair)) for pair in layer) for layer in schedule["layers"])
        moments.append(tuple(CircuitOp("H", (q,)) for q in schedule["h_measure"]))
        moments.append(tuple(CircuitOp("M", (q,)) for q in schedule["measurement_order"]))
    else:
        moments = [
            tuple(
                CircuitOp(op["gate"], tuple(op["targets"][i : i + (2 if op["gate"] == "CZ" else 1)]))
                for op in moment
                for i in range(0, len(op["targets"]), 2 if op["gate"] == "CZ" else 1)
            )
            for moment in schedule["moments"]
        ]
    outcomes = tuple(roles[q] for q in schedule["measurement_order"])

    def parity(mask: int) -> NamedParity:
        return NamedParity(tuple(outcomes[j] for j in range(n) if (mask >> j) & 1), bool((mask >> n) & 1))

    flows = []
    for kind, key in (
        ("input", "input_face_record_maps"),
        ("output", "output_face_record_maps"),
        ("logical", "logical_choi_signs"),
    ):
        for axis in "XZ":
            flows.extend(GadgetFlow(f"{kind}:{axis}[{i}]", parity(mask)) for i, mask in enumerate(audit[key][axis]))
    flows.extend(GadgetFlow(f"constraint[{i}]", parity(mask)) for i, mask in enumerate(audit["record_constraints"]))
    gadget = Gadget.from_operations(
        f"Shutty periodic 488 {gate_set} extraction",
        roles,
        tuple(moments),
        inputs=code.data_roles,
        outputs=code.data_roles,
        outcomes=outcomes,
        flows=tuple(flows),
        citations=_CITATIONS,
        input_h_frame=tuple(roles[q] for q in schedule.get("data_h_frame", [])),
        output_h_frame=tuple(roles[q] for q in schedule.get("data_h_frame", [])),
    )
    g = _geometry(code.distance)
    return Periodic488Implementation(
        code,
        gadget,
        tuple(tuple(a["coord"]) for a in g["ancillas"]),
        tuple(tuple(e) for e in g["complete_local_edges"]),
        gate_set,
    )


def _memory_parts(instructions: list) -> tuple[list, list, dict]:
    """Translate only the pinned author's instruction subset; discard its noise."""
    moments, pending, detectors, observables = [], [], [], {}
    count = 0
    for inst in instructions:
        name = inst.name
        if name == "TICK":
            moments.append(tuple(pending))
            pending = []
        elif name in {"QUBIT_COORDS", "X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}:
            continue
        elif name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            if any(kind != "r" for kind, _ in inst.targets):
                msg = "Unexpected author annotation target"
                raise ValueError(msg)
            records = tuple(count + offset for _, offset in inst.targets)
            if name == "DETECTOR":
                detectors.append((records, inst.args))
            else:
                index = int(inst.args[0])
                observables.setdefault(index, []).extend(records)
        elif name in {"R", "H", "CX", "CZ", "M", "S", "S_DAG", "C_XYZ", "C_ZYX", "SQRT_X"}:
            arity = 2 if name in {"CX", "CZ"} else 1
            pending.extend(
                CircuitOp(name, tuple(inst.targets[i : i + arity])) for i in range(0, len(inst.targets), arity)
            )
            if name == "M":
                count += len(inst.targets)
        else:
            msg = f"Unsupported author instruction: {name}"
            raise ValueError(msg)
    if pending:
        moments.append(tuple(pending))
    return moments, detectors, observables


def _native_decoder_coordinates(instructions: list, detectors: list) -> tuple[tuple[float, ...], ...]:
    """Apply the release's final torus decoder-coordinate convention.

    This is the torus branch of vendor/_coordinates.py::decoder_coordinates
    from the cited source release, expressed without a runtime Stim dependency.
    The public native-CZ generator removes the color axis and replaces (0, 0)
    placeholders with centroids of distinct measured physical qubits.
    """
    positions = {inst.targets[0]: inst.args[:2] for inst in instructions if inst.name == "QUBIT_COORDS"}
    measured = [q for inst in instructions if inst.name == "M" for q in inst.targets]
    result = []
    for records, coords in detectors:
        if len(coords) < 3:
            msg = "Author detector lacks a spacetime coordinate"
            raise ValueError(msg)
        x, y, time = coords[:3]
        if (x, y) == (0, 0):
            support = sorted({measured[r] for r in records})
            if not support:
                msg = "Cannot assign a detector centroid to empty support"
                raise ValueError(msg)
            x, y = (sum(positions[q][axis] for q in support) / len(support) for axis in (0, 1))
        result.append((x, y, time))
    return tuple(result)


def periodic_488_memory_program(
    implementation: Periodic488Implementation,
    rounds: int,
    basis: str = "Z",
) -> GadgetProgram:
    """Compose paper memory boundaries, four logical parities, and global checks.

    Boundary H/M operations overlap disjoint ancilla operations as in the author
    memory builder. The last round therefore closes the data ports. Detector
    definitions retain the author's global dependencies and add intrinsic
    constraints when independent of the face-derived span. No noise is imported.
    """
    if not isinstance(implementation, Periodic488Implementation):
        msg = "implementation must be a Periodic488Implementation"
        raise TypeError(msg)
    if type(rounds) is not int or rounds < 1:
        msg = "rounds must be a positive integer"
        raise ValueError(msg)
    if basis not in {"X", "Z"}:
        msg = "basis must be X or Z"
        raise ValueError(msg)
    code = implementation.code
    if implementation != periodic_488_implementation(code, gate_set=implementation.gate_set):
        msg = "Memory construction requires the canonical periodic implementation"
        raise ValueError(msg)
    native = implementation.gate_set == "native-cz"
    s, audit = (_native_components if native else _components)(code.distance)
    if native:
        instructions = memory25.build_memory(
            code.distance,
            basis,
            rounds,
            geometry=_geometry(code.distance),
            checked=audit,
        )["clean"]
        starts = [25 * i for i in range(rounds)]
    else:
        # Preserve the release emitter's bookkeeping, then remove its historical noise.
        instructions, metadata = make_memory.memory(
            _geometry(code.distance),
            [s] * rounds,
            basis,
            p=0.001,
            checked=[audit] * rounds,
        )
        starts = metadata["cycle_start_ticks"]
    moments, detectors, observables = _memory_parts(instructions)
    boundaries = [0, *starts[1:], len(moments)]
    program = GadgetProgram(implementation.roles)
    events = []
    for rd, (start, stop) in enumerate(pairwise(boundaries)):
        selected = tuple(moments[start:stop])
        outcomes = tuple(implementation.roles[op.qubits[0]] for m in selected for op in m if op.name == "M")
        label = f"round[{rd}]"
        if rounds == 1:
            kind = "single-round memory"
        elif rd == 0:
            kind = f"prepare {basis} and extract"
        elif rd == rounds - 1:
            kind = f"extract and read {basis}"
        else:
            kind = "recurring extraction"
        if 0 < rd < rounds - 1:
            gadget = implementation.gadget()
        else:
            gadget = Gadget.from_operations(
                f"Shutty periodic 488 {implementation.gate_set} {kind}",
                implementation.roles,
                selected,
                inputs=() if rd == 0 else code.data_roles,
                outputs=() if rd == rounds - 1 else code.data_roles,
                outcomes=outcomes,
                citations=_CITATIONS,
                input_h_frame=() if rd == 0 else implementation.gadget().input_h_frame,
                output_h_frame=() if rd == rounds - 1 else implementation.gadget().output_h_frame,
            )
        program.append(gadget, label=label)
        events.extend(f"{label}/{name}" for name in outcomes)

    signs = _memory_signs(code, audit, rounds, basis) if native else None

    def parity(records: list[int] | tuple[int, ...]) -> NamedParity:
        if any(i < 0 or i >= len(events) for i in records):
            msg = "Author memory references a nonexistent measurement"
            raise ValueError(msg)
        named = NamedParity(tuple(events[i] for i in records))
        if signs is not None:
            mask = 0
            for i in records:
                mask ^= 1 << i
            constant = 0
            while mask:
                pivot = mask.bit_length() - 1
                if pivot not in signs:
                    msg = "Author parity is not implied by the signed memory contract"
                    raise ValueError(msg)
                row, bit = signs[pivot]
                mask ^= row
                constant ^= bit
            named ^= constant
        return named

    decoder_coords = (
        _native_decoder_coordinates(instructions, detectors)
        if implementation.gate_set == "native-cz"
        else tuple(coords for _, coords in detectors)
    )
    for i, (records, coords) in enumerate(detectors):
        program.detector(f"detector[{i}]@{tuple(coords)}", parity(records), coords=decoder_coords[i])
    for i in range(code.num_logicals):
        program.observable(f"logical:{basis}[{i}]", parity(observables[i]))
    return program


def periodic_488_memory(
    implementation: Periodic488Implementation,
    rounds: int,
    basis: str = "Z",
    *,
    schedule: SchedulePolicy = SchedulePolicy.REFERENCE,
) -> ScheduledCircuit:
    """Compile the four-logical-qubit memory for Guppy, TickCircuit, or Stim."""
    return periodic_488_memory_program(implementation, rounds, basis).compile(schedule)


def _memory_signs(code: Periodic488CodeSpec, audit: dict, rounds: int, basis: str) -> dict[int, tuple[int, int]]:
    """Solve raw reference signs from signed cycle flows, without a simulator.

    Each row is an equation XOR(raw records) = sign. The initial product state,
    interround check continuity, final readout and logical flow fix the complete
    deterministic subspace. Querying the author's chosen parity support against
    this span handles its different representatives modulo record constraints.
    """
    n = code.num_data
    pivots: dict[int, tuple[int, int]] = {}

    def local(mask: int, rd: int) -> tuple[int, int]:
        return (mask & ((1 << n) - 1)) << (rd * n), (mask >> n) & 1

    def insert(row: int, sign: int) -> None:
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row, sign
                return
            previous, bit = pivots[pivot]
            row ^= previous
            sign ^= bit
        if sign:
            msg = "Inconsistent signed memory equations"
            raise ValueError(msg)

    incoming, outgoing = audit["input_face_record_maps"], audit["output_face_record_maps"]
    for rd in range(rounds):
        for mask in audit["record_constraints"]:
            insert(*local(mask, rd))
        if rd == 0:
            for mask in incoming[basis]:
                insert(*local(mask, rd))
            other = "Z" if basis == "X" else "X"
            for colors in (("blue", "green"), ("blue", "red")):
                combined = 0
                for color, mask in zip(code.face_colors, incoming[other], strict=True):
                    if color in colors:
                        combined ^= mask
                insert(*local(combined, rd))
        else:
            for axis in "XZ":
                for now, before in zip(incoming[axis], outgoing[axis], strict=True):
                    a, sa = local(now, rd)
                    b, sb = local(before, rd - 1)
                    insert(a ^ b, sa ^ sb)
    for face, mask in zip(code.faces, outgoing[basis], strict=True):
        row, sign = local(mask, rounds - 1)
        row ^= sum(1 << (rounds * n + q) for q in face)
        insert(row, sign)
    logicals = code.logical_x_supports if basis == "X" else code.logical_z_supports
    for support, mask in zip(logicals, audit["logical_choi_signs"][basis], strict=True):
        row = sum(1 << (rounds * n + q) for q in support)
        sign = 0
        for rd in range(rounds):
            bits, bit = local(mask, rd)
            row ^= bits
            sign ^= bit
        insert(row, sign)
    return pivots
