# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Render physical surface gadgets to reusable Guppy functions."""

from pecos.qec.surface._check_plan import cnot_round_order_for_check_plan, resolve_surface_check_plan
from pecos.qec.surface.circuit_builder import OpType
from pecos.qec.surface.gadgets import (
    Gadget,
    GadgetKind,
    default_allocation,
    init_syndrome_gadget,
    logical_pauli_gadget,
    measure_out_gadget,
    prep_gadget,
    syndrome_round_gadget,
)
from pecos.qec.surface.patch import SurfacePatch


def render_gadget_function(gadget: Gadget) -> list[str]:
    """Interpret one gadget's physical steps through its register allocation."""
    if (
        gadget.kind in {GadgetKind.TRANSVERSAL, GadgetKind.TWO_PATCH}
        or gadget.x_z_swapped
        or (gadget.kind == GadgetKind.PREP and gadget.basis == "Y")
    ):
        msg = f"Guppy rendering is deferred for {gadget.name}"
        raise NotImplementedError(msg)
    dx, dz = gadget.dimensions
    surface = f"SurfaceCode_{dx}x{dz}"
    syndrome = f"Syndrome_{dx}x{dz}"
    kind = gadget.kind
    data = gadget.allocations[0].data_qubits
    n = len(data)
    register = "data" if kind == GadgetKind.PREP else "surf.data"
    names = {q: f"{register}[{i}]" for i, q in enumerate(data)}
    names.update({q: f"ax{i}" for i, q in enumerate(gadget.allocations[0].x_ancilla_qubits)})
    names.update({q: f"az{i}" for i, q in enumerate(gadget.allocations[0].z_ancilla_qubits)})
    x_labels = [
        s.label
        for s in gadget.steps
        if s.op_type == OpType.MEASURE and s.qubits[0] in gadget.allocations[0].x_ancilla_qubits
    ]
    z_labels = [
        s.label
        for s in gadget.steps
        if s.op_type == OpType.MEASURE and s.qubits[0] in gadget.allocations[0].z_ancilla_qubits
    ]
    basis = gadget.basis
    argument = f"surf: {surface}"
    if kind == GadgetKind.PREP:
        argument = ""
        result = surface
        state = "+" if basis == "X" else "0"
        doc = f"Prepare logical |{state}_L> state."
    elif kind == GadgetKind.INIT_SYNDROME:
        family = "Z" if basis == "X" else "X"
        labels = x_labels if family == "X" else z_labels
        result = f"array[bool, {len(labels)}]"
        doc = f"Establish initial {family} stabilizer signs after {basis}-basis data prep."
    elif kind == GadgetKind.SYNDROME_ROUND:
        result = syndrome
        doc = "Extract full syndrome using 4-round parallel CNOT schedule."
    elif kind == GadgetKind.MEASURE_OUT:
        argument += " @ owned"
        result = f"array[bool, {n}]"
        doc = f"Destructively measure in {basis} basis."
    else:
        result = "None"
        edge = "left" if basis == "X" else "top"
        doc = f"Apply logical {basis} (string along {edge} edge)."
    lines = ["@guppy", f"def {gadget.name}({argument}) -> {result}:", f'    """{doc}"""']
    if kind == GadgetKind.SYNDROME_ROUND:
        lines.append("    # Allocate ancilla qubits (one per stabilizer)")
    index = 0
    ordinal = 0
    while index < len(gadget.steps):
        step = gadget.steps[index]
        op = step.op_type
        if op == OpType.COMMENT:
            comment = step.label
            if comment.startswith("CX round "):
                # The legacy init interface omits empty rounds; full rounds retain their comments.
                has_body = index + 1 < len(gadget.steps) and gadget.steps[index + 1].op_type not in {
                    OpType.COMMENT,
                    OpType.TICK,
                }
                if kind != GadgetKind.INIT_SYNDROME or has_body:
                    lines.extend(["", f"    # Round {comment.removeprefix('CX round ')}"])
            elif comment == "Hadamard on X ancillas":
                lines.extend(["", f"    # {comment}"])
            elif comment == "Measure ancillas":
                comment = "Measure init ancillas" if kind == GadgetKind.INIT_SYNDROME else comment
                lines.extend(["", f"    # {comment}"])
        elif op == OpType.TICK:
            pass
        elif step.qubits[0] in data and op in {OpType.ALLOC, OpType.H, OpType.MEASURE}:
            end = index
            while end < len(gadget.steps) and gadget.steps[end].op_type == op and gadget.steps[end].qubits[0] in data:
                end += 1
            run = gadget.steps[index:end]
            complete = [s.qubits for s in run] == [[q] for q in data]
            if op in {OpType.ALLOC, OpType.MEASURE} and not complete:
                msg = f"{gadget.name}: {op.name} must cover all data in register order"
                raise ValueError(msg)
            if op == OpType.ALLOC:
                if kind != GadgetKind.PREP:
                    msg = "Data allocation requires a preparation gadget"
                    raise ValueError(msg)
                lines.append(f"    data = array(qubit() for _ in range({len(run)}))")
            elif op == OpType.MEASURE:
                if any(s.op_type not in {OpType.COMMENT, OpType.TICK} for s in gadget.steps[end:]):
                    msg = f"{gadget.name}: physical operations cannot follow destructive data measurement"
                    raise ValueError(msg)
                lines.append("    return collect_measurements(measure_array(surf.data))")
            elif complete:
                lines.extend([f"    for i in range({len(run)}):", f"        h({register}[i])"])
            else:
                lines.extend(f"    h({names[s.qubits[0]]})" for s in run)
            index = end
            continue
        elif op == OpType.ALLOC:
            lines.append(f"    {names[step.qubits[0]]} = qubit()")
        elif op in {OpType.H, OpType.X, OpType.Z, OpType.CX}:
            operands = ", ".join(names[q] for q in step.qubits)
            lines.append(f"    {op.name.lower()}({operands})")
        elif op == OpType.MEASURE:
            label = step.label
            lines.append(f"    {label} = measure({names[step.qubits[0]]}).read()")
            tag = "init:meas" if kind == GadgetKind.INIT_SYNDROME else "meas"
            lines.append(f'    output("{label}:{tag}:{ordinal}", {label})')
            ordinal += 1
        else:
            msg = f"Unsupported gadget operation: {op.name}"
            raise ValueError(msg)
        index += 1
    if kind == GadgetKind.PREP:
        lines.append(f"    return {surface}(data)")
    elif kind == GadgetKind.INIT_SYNDROME:
        lines.extend(["", f"    return array({', '.join(labels)})"])
    elif kind == GadgetKind.SYNDROME_ROUND:
        x_results = ", ".join(x_labels)
        z_results = ", ".join(z_labels)
        lines.extend(
            [
                "",
                f"    synx = array({x_results})",
                f"    synz = array({z_results})",
                "",
                f"    return {syndrome}(synx, synz)",
            ],
        )
    return lines


def render_surface_gadget_module(patch: SurfacePatch) -> str:
    """Assemble the default surface module from independently rendered gadgets."""
    resolved_plan = resolve_surface_check_plan(interaction_basis="cx")
    round_order = cnot_round_order_for_check_plan(resolved_plan)
    geom = patch.geometry
    dx, dz = geom.dx, geom.dz
    n = geom.num_data
    nx, nz = len(geom.x_stabilizers), len(geom.z_stabilizers)
    allocation = default_allocation(patch)
    lines = [
        f'"""Surface code patch (dx={dx}, dz={dz}) implementation in Guppy.',
        "",
        "Auto-generated from SurfacePatch geometry.",
        "",
        f"Data qubits: {n}",
        f"X stabilizers: {nx}",
        f"Z stabilizers: {nz}",
        f"Ancilla qubits: {nx + nz} (one per stabilizer)",
        "Interaction basis: cx",
        f"Check plan: {resolved_plan.plan_id}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, output",
        "from guppylang.std.quantum import cx, discard, h, qubit, x, z",
        "from guppylang.std.quantum import collect_measurements, measure, measure_array",
        "",
        "",
        "@guppy.struct",
        f"class SurfaceCode_{dx}x{dz}:",
        f'    """Surface code patch with dx={dx}, dz={dz} ({n} data qubits)."""',
        "",
        f"    data: array[qubit, {n}]",
        "",
        "",
        "@guppy.struct",
        f"class Syndrome_{dx}x{dz}:",
        f'    """Syndrome for dx={dx}, dz={dz} patch."""',
        "",
        f"    synx: array[bool, {nx}]",
        f"    synz: array[bool, {nz}]",
        "",
        "",
        "# === State Preparation ===",
        "",
    ]
    for basis in ("Z", "X"):
        lines.extend(render_gadget_function(prep_gadget(patch, allocation, basis=basis)))
        lines.extend(["", ""])
    lines.extend(["# === Syndrome Extraction ===", ""])
    lines.extend(
        render_gadget_function(syndrome_round_gadget(patch, allocation, round_index=0, round_order=round_order)),
    )
    lines.extend(["", "", "", ""])
    lines.extend(render_gadget_function(init_syndrome_gadget(patch, allocation, basis="Z", round_order=round_order)))
    lines.extend(["", ""])
    lines.extend(render_gadget_function(init_syndrome_gadget(patch, allocation, basis="X", round_order=round_order)))
    lines.extend(["# === Measurement ===", ""])
    for basis in ("Z", "X"):
        lines.extend(render_gadget_function(measure_out_gadget(patch, allocation, basis=basis)))
        lines.extend(["", ""])
    lines.extend(["# === Logical Operators ===", ""])
    for pauli in ("X", "Z"):
        lines.extend(render_gadget_function(logical_pauli_gadget(patch, allocation, pauli=pauli)))
        lines.extend(["", ""])
    lines.extend(["# === Memory Experiments ===", ""])
    for basis, family in (("z", "x"), ("x", "z")):
        lines.extend(
            [
                f"def make_memory_{basis}(num_rounds: int):",
                f'    """Create {basis.upper()}-basis memory experiment."""',
                "    from guppylang.std.builtins import comptime",
                "",
                "    @guppy",
                f"    def memory_{basis}() -> None:",
                f'        """{basis.upper()}-basis memory experiment for dx={dx}, dz={dz}."""',
                f"        surf = prep_{basis}_basis()",
                f"        init_syn = init_{basis}_basis(surf)",
                f'        output("init_syn{family}", init_syn)',
                "",
                "        for _t in range(comptime(num_rounds)):",
                "            syn = syndrome_extraction(surf)",
                '            output("synx", syn.synx)',
                '            output("synz", syn.synz)',
                "",
                f"        final = measure_{basis}_basis(surf)",
                '        output("final", final)',
            ],
        )
        lines.extend(f'        output("final:meas:{q}", final[{q}])' for q in range(n))
        lines.extend(["", f"    return memory_{basis}", "", ""])
    return "\n".join(lines)
