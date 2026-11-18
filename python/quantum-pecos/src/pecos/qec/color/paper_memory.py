# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Named memory composition for the two cited color-code constructions.

Shutty, arXiv:2609.21376v3, Sec. 2.2.1; Lavasani et al., arXiv:2610.24077v1,
Fig. 1 and Supplement B. Measurement-induced signs are tracked in named
parities, then lowered to raw events by the code-independent gadget compiler.
"""

from pecos.qec.color._shutty488 import geometry
from pecos.qec.color.paper_codes import ColorCodeImplementation, ColorExtraction, _single, bell_flag_projection
from pecos.qec.generic.gadgets import Gadget, GadgetProgram, NamedParity
from pecos.qec.generic.scheduled import ScheduledCircuit, SchedulePolicy


def color_memory_program(
    implementation: ColorCodeImplementation,
    rounds: int,
    basis: str = "Z",
    *,
    start_phase: int = 0,
) -> GadgetProgram:
    """Compose preparation, extraction phases, and readout as named gadgets.

    Inspect calls, ports, check relations and detector expressions before
    choosing a schedule with ``program.compile(schedule=...)``. The 666 family
    uses Bell-flag projection; 488 uses the reference product-state boundary.
    ``start_phase`` starts a fresh memory, not a continuation of an old frame.
    """
    if type(rounds) is not int or rounds < 1:
        msg = "rounds must be a positive integer"
        raise ValueError(msg)
    if basis not in {"X", "Z"}:
        msg = "basis must be X or Z"
        raise ValueError(msg)
    if type(start_phase) is not int or not 0 <= start_phase < implementation.period:
        msg = "start_phase is outside this implementation's period"
        raise ValueError(msg)
    code = implementation.code
    n, nf = code.num_data, len(code.faces)
    program = GadgetProgram(implementation.roles)
    # Shutty release vendor/_coordinates.py: native-CZ face detectors use data
    # centroids; constraint detectors use distinct measured physical supports.
    # CNOT retains the original face/index/color annotation convention.
    native_488 = implementation.extraction == ColorExtraction.BRICKWORK_CZ
    face_centers = (
        tuple(
            tuple(sum(code.data_coordinates[q][axis] for q in face) / len(face) for axis in (0, 1))
            for face in code.faces
        )
        if native_488
        else ()
    )
    carrier_coords = ()
    if native_488:
        layout = geometry.generate_base_layout((code.distance + 1) // 4)
        carrier_coords = (*layout["qubit_coords"], *layout["ancilla_coords"])

    def face_coordinates(face: int, axis: str, rd: int) -> tuple[float, ...] | None:
        if native_488:
            return (*face_centers[face], rd)
        if code.family == "488":
            return (face, 0, rd, 0 if axis == "X" else 3)
        return None

    initial_frame = implementation.cycles[start_phase].input_h_frame
    moments = [_single("R", range(n))]
    initial_h = (set(range(n)) if basis == "X" else set()) ^ set(initial_frame)
    if initial_h:
        moments.append(_single("H", sorted(initial_h)))
    program.append(
        Gadget.from_operations(
            f"prepare:{basis}",
            implementation.roles,
            moments,
            inputs=(),
            outputs=code.data_roles,
            outcomes=(),
            output_h_frame=tuple(implementation.roles[q] for q in initial_frame),
            citations=(code.citation,),
        ),
        label="prepare",
    )

    projection = None
    if code.family == "666":
        projection = program.append(
            Gadget.from_operations(
                f"bell_flag:{basis}",
                implementation.roles,
                bell_flag_projection(code, basis),
                inputs=code.data_roles,
                outputs=code.data_roles,
                outcomes=(*(f"syndrome[{f}]" for f in range(nf)), *(f"flag[{f}]" for f in range(nf))),
                citations=(code.citation,),
            ),
            label="initialize",
        )
        for face in range(nf):
            program.detector(f"init:flag:{face}", projection[f"flag[{face}]"])

    logical = NamedParity()
    previous = None
    for rd in range(rounds):
        current = program.append(
            implementation.gadget((start_phase + rd) % implementation.period),
            label=f"round[{rd}]",
        )
        for flow in current.gadget.flows:
            if flow.name.startswith("constraint["):
                index = flow.name[len("constraint[") : -1]
                coords = None
                if native_488:
                    cycle = implementation.cycles[(start_phase + rd) % implementation.period]
                    measured = [op.qubits[0] for moment in cycle.moments for op in moment if op.name == "M"]
                    mask = cycle.flow.constraints[int(index)]
                    support = {q for bit, q in enumerate(measured) if mask & (1 << bit)}
                    coords = (
                        *(sum(carrier_coords[q][axis] for q in sorted(support)) / len(support) for axis in (0, 1)),
                        rd,
                    )
                elif code.family == "488":
                    coords = (int(index), 1, rd)
                program.detector(f"round:{rd}:constraint:{index}", current.flow(flow.name), coords=coords)
        axes = "XZ" if previous is not None or projection is not None else basis
        for face in range(nf):
            for axis in axes:
                parity = current.flow(f"input:{axis}[{face}]")
                if previous is not None:
                    parity ^= previous.flow(f"output:{axis}[{face}]")
                elif projection is not None and axis != basis:
                    parity ^= projection[f"syndrome[{face}]"]
                program.detector(f"round:{rd}:{axis}:{face}", parity, coords=face_coordinates(face, axis, rd))
        logical ^= current.flow(f"logical:{basis}")
        previous = current

    output_frame = implementation.cycles[(start_phase + rounds - 1) % implementation.period].output_h_frame
    final_h = (set(range(n)) if basis == "X" else set()) ^ set(output_frame)
    final_moments = [_single("H", sorted(final_h))] if final_h else []
    final_moments.append(_single("M", range(n)))
    readout = program.append(
        Gadget.from_operations(
            f"readout:{basis}",
            implementation.roles,
            final_moments,
            inputs=code.data_roles,
            outputs=(),
            outcomes=code.data_roles,
            input_h_frame=tuple(implementation.roles[q] for q in output_frame),
            citations=(code.citation,),
        ),
        label="readout",
    )
    for face, support in enumerate(code.faces):
        parity = current.flow(f"output:{basis}[{face}]")
        for q in support:
            parity ^= readout[code.data_roles[q]]
        program.detector(f"final:{basis}:{face}", parity, coords=face_coordinates(face, basis, rounds))
    for q in code.logical_support:
        logical ^= readout[code.data_roles[q]]
    program.observable(f"logical:{basis}", logical)
    return program


def color_memory(
    implementation: ColorCodeImplementation,
    rounds: int,
    basis: str = "Z",
    *,
    start_phase: int = 0,
    schedule: SchedulePolicy = SchedulePolicy.REFERENCE,
) -> ScheduledCircuit:
    """Build a scheduled X/Z memory through the named gadget authoring layer."""
    return color_memory_program(implementation, rounds, basis, start_phase=start_phase).compile(schedule)
