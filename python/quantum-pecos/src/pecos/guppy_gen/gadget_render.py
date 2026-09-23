# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Render physical surface gadgets to reusable Guppy functions."""

import hashlib
from functools import cache

from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.surface import _certify_surface_measurement_layout, _validate_surface_memory_distance
from pecos.qec.surface._check_plan import (
    ancilla_schedule_for_check_plan,
    cnot_round_order_for_check_plan,
    require_current_surface_check_plan_renderer,
    resolve_surface_check_plan,
)
from pecos.qec.surface.circuit_builder import OpType, TickCircuitRenderer
from pecos.qec.surface.gadgets import (
    Gadget,
    GadgetKind,
    default_allocation,
    init_syndrome_gadget,
    logical_pauli_gadget,
    measure_out_gadget,
    memory_gadgets,
    prep_gadget,
    syndrome_round_gadget,
)
from pecos.qec.surface.patch import SurfacePatch


@cache
def _load_surface_gadget_source(source: str) -> dict:
    """Keep module identity tied to the entire rendered geometry and schedule."""
    key = f"surface_gadgets_{hashlib.sha256(source.encode()).hexdigest()}"
    return load_guppy_source(source, _get_temp_dir() / f"{key}.py", f"pecos._generated.{key}")


def make_surface_memory(
    distance_or_patch: int | SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    check_plan: str | None = None,
) -> object:
    """Compile gadget memory with a program-bound measurement-layout certificate.

    Args:
        distance_or_patch: Odd distance >= 3, or an existing SurfacePatch.
        num_rounds: Number of syndrome extraction rounds.
        basis: 'Z' or 'X', case insensitive.
        ancilla_budget: Optional cap on simultaneously live ancillas.
        check_plan: Named CX check-plan preset.

    Returns:
        Compiled Guppy definition accepted by the trusted Guppy-to-DEM routes.
    """
    basis = basis.upper()
    if basis not in ("Z", "X"):
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)
    if isinstance(distance_or_patch, SurfacePatch):
        patch = distance_or_patch
    else:
        _validate_surface_memory_distance(distance_or_patch)
        patch = SurfacePatch.create(distance=distance_or_patch)
    source = render_surface_gadget_module(patch, ancilla_budget=ancilla_budget, check_plan=check_plan)
    module = _load_surface_gadget_source(source)
    program = module[f"make_memory_{basis.lower()}"](num_rounds)

    resolved_plan = resolve_surface_check_plan(check_plan=check_plan)
    parts = memory_gadgets(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        ancilla_schedule=ancilla_schedule_for_check_plan(resolved_plan),
        round_order=cnot_round_order_for_check_plan(resolved_plan),
    )
    abstract_tc = TickCircuitRenderer(add_typed_annotations=False).render(
        [step for part in parts for step in part.steps],
        parts[0].allocations[0],
        patch,
        num_rounds,
        basis,
    )
    _certify_surface_measurement_layout(program, abstract_tc)
    return program


def _allocation_epochs(
    gadget: Gadget,
    registers: tuple[str, ...],
) -> tuple[dict[int, str], dict[int, str], list[str], list[str]]:
    """Resolve names and result slots from live allocations, not reusable physical IDs."""
    data_names = {}
    candidates: dict[int, list[tuple[str, str, int, str]]] = {}
    for register, allocation in zip(registers, gadget.allocations, strict=True):
        for index, qubit in enumerate(allocation.data_qubits):
            if qubit in data_names:
                msg = f"{gadget.name}: allocations must be disjoint"
                raise ValueError(msg)
            data_names[qubit] = f"{register}[{index}]"
        ancillas = set(allocation.x_ancilla_qubits + allocation.z_ancilla_qubits)
        # Two-register allocations must be globally disjoint regardless of use.
        if candidates.keys() & ancillas:
            msg = f"{gadget.name}: allocations must be disjoint across registers"
            raise ValueError(msg)
        prefix = f"{register.split('.')[0]}_" if len(registers) > 1 else ""
        for family, qubits in (("X", allocation.x_ancilla_qubits), ("Z", allocation.z_ancilla_qubits)):
            for index, qubit in enumerate(qubits):
                label = f"a{family.lower()}{index}"
                candidates.setdefault(qubit, []).append((f"{prefix}{label}", family, index, label))
    if data_names.keys() & candidates.keys():
        msg = f"{gadget.name}: allocations must be disjoint"
        raise ValueError(msg)
    active = {}
    epochs = {}
    results: dict[str, list[tuple[int, str]]] = {"X": [], "Z": []}
    for step_index, step in enumerate(gadget.steps):
        if not step.qubits or step.qubits[0] in data_names:
            continue
        qubit = step.qubits[0]
        if step.op_type == OpType.ALLOC:
            if qubit in active:
                msg = f"{gadget.name}: allocations must be disjoint within a live epoch"
                raise ValueError(msg)
            choices = candidates.get(qubit, [])
            if len(choices) != 1:
                choices = [choice for choice in choices if step.label in (choice[0], choice[3])]
            if len(choices) != 1:
                msg = f"{gadget.name}: ALLOC label {step.label!r} must identify an ancilla register slot"
                raise ValueError(msg)
            name, family, index, _label = choices[0]
            epochs[step_index] = name
            active[qubit] = (family, index)
        elif step.op_type == OpType.MEASURE:
            if qubit not in active:
                msg = f"{gadget.name}: measurement requires a live ancilla allocation"
                raise ValueError(msg)
            family, index = active.pop(qubit)
            results[family].append((index, step.label))
    x_family, z_family = ("Z", "X") if gadget.x_z_swapped else ("X", "Z")
    return (
        data_names,
        epochs,
        [label for _index, label in sorted(results[x_family], key=lambda item: item[0])],
        [label for _index, label in sorted(results[z_family], key=lambda item: item[0])],
    )


def render_gadget_function(gadget: Gadget, *, tag_scope: str | None = None) -> list[str]:
    """Interpret one gadget's physical steps through its register allocation."""
    if tag_scope is not None and not tag_scope.isidentifier():
        msg = f"Invalid gadget tag scope: {tag_scope!r}"
        raise ValueError(msg)
    function_name = gadget.name if tag_scope is None else f"{gadget.name}_{tag_scope}"
    tag_prefix = "" if tag_scope is None else f"{tag_scope}:"
    if gadget.x_z_swapped:
        tag_prefix += "swapped:"
    dx, dz = gadget.dimensions
    surface = f"SurfaceCode_{dx}x{dz}"
    syndrome = f"Syndrome_{dx}x{dz}"
    kind = gadget.kind
    registers = (
        ("ctrl.data", "tgt.data")
        if kind == GadgetKind.TWO_PATCH
        else ("data" if kind == GadgetKind.PREP else "surf.data",)
    )
    if len(gadget.allocations) != len(registers):
        msg = f"{gadget.name}: expected {len(registers)} allocations"
        raise ValueError(msg)
    names, epoch_names, x_labels, z_labels = _allocation_epochs(gadget, registers)

    def live_name(qubit: int) -> str:
        if qubit not in names:
            msg = f"{gadget.name}: ancilla {qubit} has no live allocation"
            raise ValueError(msg)
        return names[qubit]

    data_registers = [allocation.data_qubits for allocation in gadget.allocations]
    data = data_registers[0]
    register = registers[0]
    n = len(data)
    allocation = gadget.allocations[0]
    if gadget.x_z_swapped and dx == dz and len(allocation.x_ancilla_qubits) != len(allocation.z_ancilla_qubits):
        syndrome += "_swapped"
    basis = gadget.basis
    argument = f"surf: {surface}"
    if kind == GadgetKind.PREP:
        argument = ""
        result = surface
        state = {"X": "+", "Y": "+i", "Z": "0"}[basis]
        doc = f"Prepare logical |{state}_L> state."
    elif kind == GadgetKind.INIT_SYNDROME:
        family = "Z" if basis == "X" else "X"
        labels = x_labels if family == "X" else z_labels
        result = f"array[bool, {len(labels)}]"
        doc = f"Establish initial {family} stabilizer signs after {basis}-basis data prep."
    elif kind == GadgetKind.SYNDROME_ROUND:
        result = syndrome
        doc = "Extract full syndrome using 4-round parallel CNOT schedule."
        if gadget.fold == "SDG":
            doc = "Extract full syndrome with the fold-transversal logical S-dagger between CX layers 2 and 3."
        elif gadget.fold == "S":
            doc = "Extract full syndrome with the fold-transversal logical S between CX layers 2 and 3."
    elif kind == GadgetKind.MEASURE_OUT:
        argument += " @ owned"
        result = f"array[bool, {n}]"
        doc = f"Destructively measure in {basis} basis."
    elif kind in {GadgetKind.TRANSVERSAL, GadgetKind.TWO_PATCH}:
        result = "None"
        doc = f"Apply physical transversal {basis}."
        if kind == GadgetKind.TWO_PATCH:
            argument = f"ctrl: {surface}, tgt: {surface}"
    else:
        result = "None"
        edge = "left" if basis == "X" else "top"
        doc = f"Apply logical {basis} (string along {edge} edge)."
    lines = ["@guppy", f"def {function_name}({argument}) -> {result}:", f'    """{doc}"""']
    if kind == GadgetKind.SYNDROME_ROUND:
        ancillas = allocation.x_ancilla_qubits + allocation.z_ancilla_qubits
        if len(set(ancillas)) < len(ancillas):
            lines.append("    # Reuse ancillas in batches to respect the live-qubit budget")
        else:
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
            elif comment in {
                "Hadamard on X ancillas",
                "Hadamard on Z ancillas",
                "fold-transversal S layer",
                "fold-transversal S-dagger layer",
            }:
                lines.extend(["", f"    # {comment}"])
            elif comment == "Measure ancillas":
                comment = "Measure init ancillas" if kind == GadgetKind.INIT_SYNDROME else comment
                lines.extend(["", f"    # {comment}"])
        elif op == OpType.TICK:
            pass
        elif kind == GadgetKind.TWO_PATCH and op == OpType.CX:
            end = index
            while end < len(gadget.steps) and gadget.steps[end].op_type == OpType.CX:
                end += 1
            pairs = [list(pair) for pair in zip(*data_registers, strict=True)]
            if [s.qubits for s in gadget.steps[index:end]] == pairs:
                lines.extend([f"    for i in range({n}):", "        cx(ctrl.data[i], tgt.data[i])"])
            else:
                lines.extend(f"    cx({', '.join(live_name(q) for q in s.qubits)})" for s in gadget.steps[index:end])
            index = end
            continue
        elif step.qubits[0] in data and op in {OpType.ALLOC, OpType.H, OpType.SZ, OpType.SZDG, OpType.MEASURE}:
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
            else:
                gate = {OpType.H: "h", OpType.SZ: "s", OpType.SZDG: "sdg"}[op]
                if complete:
                    lines.extend([f"    for i in range({len(run)}):", f"        {gate}({register}[i])"])
                else:
                    lines.extend(f"    {gate}({names[s.qubits[0]]})" for s in run)
            index = end
            continue
        elif op == OpType.ALLOC:
            names[step.qubits[0]] = epoch_names[index]
            lines.append(f"    {names[step.qubits[0]]} = qubit()")
        elif op in {OpType.H, OpType.X, OpType.Z, OpType.CX, OpType.CZ, OpType.SZ, OpType.SZDG}:
            operands = ", ".join(live_name(q) for q in step.qubits)
            gate = {OpType.SZ: "s", OpType.SZDG: "sdg"}.get(op, op.name.lower())
            lines.append(f"    {gate}({operands})")
        elif op == OpType.MEASURE:
            label = step.label
            lines.append(f"    {label} = measure({names[step.qubits[0]]}).read()")
            tag = "init:meas" if kind == GadgetKind.INIT_SYNDROME else "meas"
            lines.append(f'    output("{tag_prefix}{label}:{tag}:{ordinal}", {label})')
            ordinal += 1
            del names[step.qubits[0]]
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


def render_surface_gadget_module(
    patch: SurfacePatch,
    *,
    ancilla_budget: int | None = None,
    check_plan: str | None = None,
) -> str:
    """Assemble the default surface module from independently rendered gadgets."""
    resolved_plan = resolve_surface_check_plan(check_plan=check_plan)
    if resolved_plan.interaction_basis != "cx":
        msg = f"check_plan {resolved_plan.plan_id!r} is unsupported by CX gadgets, including ancilla_budget"
        raise ValueError(msg)
    require_current_surface_check_plan_renderer(resolved_plan, context="surface gadget module")
    ancilla_schedule = ancilla_schedule_for_check_plan(resolved_plan)
    round_order = cnot_round_order_for_check_plan(resolved_plan)
    geom = patch.geometry
    dx, dz = geom.dx, geom.dz
    n = geom.num_data
    nx, nz = len(geom.x_stabilizers), len(geom.z_stabilizers)
    allocation = default_allocation(patch, ancilla_budget=ancilla_budget, ancilla_schedule=ancilla_schedule)
    effective_budget = allocation.total - n
    ancilla_description = (
        f"{nx + nz} (one per stabilizer)" if effective_budget == nx + nz else f"{effective_budget} (reused)"
    )
    lines = [
        f'"""Surface code patch (dx={dx}, dz={dz}) implementation in Guppy.',
        "",
        "Auto-generated from SurfacePatch geometry.",
        "",
        f"Data qubits: {n}",
        f"X stabilizers: {nx}",
        f"Z stabilizers: {nz}",
        f"Ancilla qubits: {ancilla_description}",
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
    ]
    if dx == dz and nx != nz:
        # Swapping an even-distance patch exchanges unequal syndrome register sizes.
        lines.extend(
            [
                "@guppy.struct",
                f"class Syndrome_{dx}x{dz}_swapped:",
                f'    """Syndrome for the swapped dx={dx}, dz={dz} patch."""',
                "",
                f"    synx: array[bool, {nz}]",
                f"    synz: array[bool, {nx}]",
                "",
                "",
            ],
        )
    lines.extend(["# === State Preparation ===", ""])
    for basis in ("Z", "X"):
        lines.extend(render_gadget_function(prep_gadget(patch, allocation, basis=basis)))
        lines.extend(["", ""])
    lines.extend(["# === Syndrome Extraction ===", ""])
    lines.extend(
        render_gadget_function(
            syndrome_round_gadget(
                patch,
                allocation,
                round_index=0,
                round_order=round_order,
                ancilla_budget=ancilla_budget,
                ancilla_schedule=ancilla_schedule,
            ),
        ),
    )
    lines.extend(["", "", "", ""])
    lines.extend(
        render_gadget_function(
            init_syndrome_gadget(
                patch,
                allocation,
                basis="Z",
                round_order=round_order,
                ancilla_budget=ancilla_budget,
                ancilla_schedule=ancilla_schedule,
            ),
        ),
    )
    lines.extend(["", ""])
    lines.extend(
        render_gadget_function(
            init_syndrome_gadget(
                patch,
                allocation,
                basis="X",
                round_order=round_order,
                ancilla_budget=ancilla_budget,
                ancilla_schedule=ancilla_schedule,
            ),
        ),
    )
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
