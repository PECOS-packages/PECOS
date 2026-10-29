# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Render C4-Helix memory from Berthusen et al., arXiv:2609.03194.

Rounds are unrolled calls to gadget functions. Each measurement emits a scalar
result with a round-free tag; its occurrence identifies the round. RELABEL uses
``mem_swap`` to exchange data and local ancilla bindings without a quantum gate.
Certification uses the step-ordered circuit to match source measurement order.
DEMs instead use typed result references from the ASAP circuit's discovered
parities, so measurement reordering cannot misattribute detector records.
"""

from pecos.guppy_gen._certificate import certify_measurement_layout
from pecos.guppy_gen._module_loader import load_cached_guppy_source
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix.gadgets import HelixAllocation, c4_round_steps, memory_gadgets, memory_steps
from pecos.qec.helix.lowering import helix_result_classifier, lower_steps

HELIX_MEMORY_NUM_QUBITS = 24


def _render_steps(steps: list[CircuitStep], *, readout: bool = False) -> list[str]:
    """Render gadget steps using data slots and function-local ancillas."""
    lines = []
    swapped = set()
    for step in steps:
        if step.op_type in {OpType.TICK, OpType.COMMENT}:
            continue
        if readout:
            slot = step.qubits[0]
            if slot not in swapped:
                lines.extend([f"    q{slot} = qubit()", f"    mem_swap(code.data[{slot}], q{slot})"])
                swapped.add(slot)
        names = [f"code.data[{q}]" if q < 20 and not readout else f"q{q}" for q in step.qubits]
        if step.op_type == OpType.PREP:
            lines.append(f"    {names[0]} = qubit()")
        elif step.op_type == OpType.H:
            lines.append(f"    h({names[0]})")
        elif step.op_type == OpType.CX:
            lines.append(f"    cx({names[0]}, {names[1]})")
        elif step.op_type == OpType.RELABEL:
            lines.append(f"    mem_swap({names[0]}, {names[1]})")
        elif step.op_type == OpType.MEASURE:
            lines.append(f'    result("{step.label}", measure({names[0]}).read())')
        else:
            msg = f"Unsupported Helix Guppy operation: {step.op_type.name} in step {step!r}"
            raise ValueError(msg)
    return lines


def render_helix_memory_source(basis: str, rounds: int) -> str:
    """Render a module with reusable gadgets and a fresh-program memory factory."""
    gadgets = memory_gadgets(basis, rounds)
    prepare_name, prepare_steps = gadgets[0]
    readout_name, readout_steps = gadgets[-1]
    lines = [
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, result",
        "from guppylang.std.mem import mem_swap",
        "from guppylang.std.quantum import cx, discard_array, h, measure, qubit",
        "",
        "",
        "@guppy.struct",
        "class HelixCode_20_2_6:",
        "    data: array[qubit, 20]",
        "",
        "",
        "@guppy",
        f"def {prepare_name}() -> HelixCode_20_2_6:",
    ]
    allocations = ", ".join("qubit()" for step in prepare_steps if step.op_type == OpType.PREP)
    lines.append(f"    code = HelixCode_20_2_6(array({allocations}))")
    lines.extend(_render_steps([step for step in prepare_steps if step.op_type != OpType.PREP]))
    lines.extend(["    return code", "", ""])
    definitions = {f"c4_{block}_{variant}": c4_round_steps(block, variant) for block in range(5) for variant in "ab"}
    definitions.update(gadgets[1:-1])
    for name, steps in definitions.items():
        lines.extend(["@guppy", f"def {name}(code: HelixCode_20_2_6) -> None:"])
        lines.extend(_render_steps(steps))
        lines.extend(["", ""])
    lines.extend(["@guppy", f"def {readout_name}(code: HelixCode_20_2_6 @ owned) -> None:"])
    lines.extend(_render_steps(readout_steps, readout=True))
    lines.extend(["    discard_array(code.data)", "", ""])
    name = f"memory_{basis.lower()}"
    lines.extend([f"def make_{name}():", "    @guppy", f"    def {name}() -> None:"])
    lines.append(f"        code = {prepare_name}()")
    lines.extend(f"        {gadget_name}(code)" for gadget_name, _ in gadgets[1:])
    lines.extend(["", f"    return {name}", ""])
    return "\n".join(lines)


def make_helix_memory(basis: str, rounds: int) -> object:
    """Compile memory and bind its scalar layout to the program's source order."""
    source = render_helix_memory_source(basis, rounds)
    module = load_cached_guppy_source("helix", source)
    program = module[f"make_memory_{basis.lower()}"]()
    circuit = lower_steps(memory_steps(basis, rounds), HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    certify_measurement_layout(program, circuit, helix_result_classifier, family="helix")
    return program
