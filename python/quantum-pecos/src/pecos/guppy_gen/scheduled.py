# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Functional Guppy lowering of static QEC circuit plans."""

from __future__ import annotations

import keyword
from typing import TYPE_CHECKING

from pecos.qec.generic.guppy_ordering import GuppyLifetimeFence, GuppyOrdering, plan_guppy_ordering

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


def render_scheduled_circuit(
    circuit: ScheduledCircuit,
    *,
    function_name: str = "memory",
    ordering: GuppyOrdering = GuppyOrdering.FUNCTIONAL,
    allocation_provenance: bool = False,
) -> str:
    """Emit scalar measurement events and preserve carrier relabel semantics.

    The default is functional. Opt-in ordering adds joined live-wire runtime
    barrier helpers at the cuts listed by plan_guppy_ordering(). These do not
    assign physical durations or certify the source noise model on lowered pulses.
    """
    if not function_name.isidentifier() or keyword.iskeyword(function_name):
        msg = "function_name must be a Python identifier"
        raise ValueError(msg)
    contract = plan_guppy_ordering(circuit, ordering)
    fences = {fence.after_moment: fence for fence in contract.fences}
    lifetime_ordered = ordering == GuppyOrdering.ALL_MOMENTS
    if type(allocation_provenance) is not bool or (allocation_provenance and not lifetime_ordered):
        msg = "allocation_provenance must be bool and requires ALL_MOMENTS when enabled"
        raise ValueError(msg)
    allocator = "pecos_qis_ordered_alloc_tagged_hugr" if allocation_provenance else "pecos_qis_ordered_alloc_hugr"
    reserved = {"guppy", "result", "qubit", "h", "s", "sdg", "v", "x", "z", "cx", "cz", "measure"}
    if fences:
        reserved.update(
            {"owned", "tuple", "pecos_qis_runtime_barrier_qubit_hugr", "pecos_qis_runtime_barrier_qubits2_hugr"},
        )
    if lifetime_ordered:
        reserved.update(
            {
                "owned",
                "int",
                "bool",
                "tuple",
                "pecos_qis_ordered_alloc_hugr",
                "pecos_qis_ordered_alloc_tagged_hugr",
                "pecos_qis_ordering_join_hugr",
                "pecos_qis_ordering_observe_hugr",
                "pecos_qis_ordering_barrier_hugr",
            },
        )
    if function_name in reserved:
        msg = "function_name must not shadow an imported Guppy primitive"
        raise ValueError(msg)
    description = (
        f"Generated static QEC experiment; {ordering.value} schedule contract.\nSchedule: {circuit.schedule.value}\n"
        + "\n".join(f"Reference: {citation}" for citation in circuit.citations)
    )
    lines = [
        repr(description),
        "from guppylang import guppy",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import qubit, h, s, sdg, v, x, z, cx, cz, measure",
    ]
    if lifetime_ordered:
        lines.extend(
            [
                "from guppylang.std.builtins import owned",
                "",
                "@guppy.declare",
                f"def {allocator}(token: int) -> qubit: ...",
                "",
                "@guppy.declare",
                "def pecos_qis_ordering_join_hugr(q: qubit @ owned, token: int) -> tuple[qubit, int]: ...",
                "",
                "@guppy.declare",
                "def pecos_qis_ordering_observe_hugr(token: int, outcome: bool) -> int: ...",
                "",
                "@guppy.declare",
                "def pecos_qis_ordering_barrier_hugr(token: int) -> int: ...",
            ],
        )
    elif fences:
        lines.extend(
            [
                "from guppylang.std.builtins import owned",
                "",
                "@guppy.declare",
                "def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...",
                "",
                "@guppy.declare",
                "def pecos_qis_runtime_barrier_qubits2_hugr(a: qubit @ owned, b: qubit @ owned) "
                "-> tuple[qubit, qubit]: ...",
            ],
        )
    lines.extend(["", "@guppy", f"def {function_name}() -> None:"])
    if lifetime_ordered:
        lines.append("    ordering_token = 0")
    # Rename source variables at compile time. A RELABEL emits no SWAP gate.
    names = [f"q{i}" for i in range(circuit.num_qubits)]
    measurement = 0
    for index, moment in enumerate(circuit.moments):
        lines.append(f"    # Reference moment {index}")
        for op in moment:
            if op.origin is not None:
                lines.append(f"    # Source: {op.origin.id!r}; gadget: {op.origin.gadget!r}")
            if circuit.role_names:
                lines.append(f"    # {op.name} roles: {tuple(circuit.role_names[q] for q in op.qubits)!r}")
            if op.name == "RELABEL":
                a, b = op.qubits
                names[a], names[b] = names[b], names[a]
            elif op.name == "R":
                allocation = f"{allocator}(ordering_token)" if lifetime_ordered else "qubit()"
                lines.append(f"    {names[op.qubits[0]]} = {allocation}")
            elif op.name == "M":
                if circuit.measurement_names:
                    lines.append(f"    # Outcome: {circuit.measurement_names[measurement]!r} -> qec:m:{measurement}")
                lines.append(f"    m{measurement} = measure({names[op.qubits[0]]}).read()")
                lines.append(f'    result("qec:m:{measurement}", m{measurement})')
                measurement += 1
            else:
                operands = ", ".join(names[q] for q in op.qubits)
                gates = {"C_XYZ": ("sdg", "h"), "C_ZYX": ("h", "s"), "SQRT_X": ("v",), "S_DAG": ("sdg",)}.get(
                    op.name,
                    (op.name.lower(),),
                )
                lines.extend(f"    {gate}({operands})" for gate in gates)
        if index in fences:
            fence = fences[index]
            lines.append(f"    # Ordering fence after reference moment {index}; no physical duration assigned")
            if isinstance(fence, GuppyLifetimeFence):
                joins = [
                    f"    {names[q]}, ordering_token = pecos_qis_ordering_join_hugr({names[q]}, ordering_token)"
                    for q in fence.qubits
                ]
                lines.extend(joins)
                lines.extend(
                    f"    ordering_token = pecos_qis_ordering_observe_hugr(ordering_token, m{m})"
                    for m in fence.measurements
                )
                lines.append("    ordering_token = pecos_qis_ordering_barrier_hugr(ordering_token)")
                lines.extend(joins)
            else:
                for join in fence.joins:
                    operands = ", ".join(names[q] for q in join)
                    helper = "qubit" if len(join) == 1 else "qubits2"
                    lines.append(f"    {operands} = pecos_qis_runtime_barrier_{helper}_hugr({operands})")
    lines.append("    return")
    return "\n".join(lines) + "\n"
