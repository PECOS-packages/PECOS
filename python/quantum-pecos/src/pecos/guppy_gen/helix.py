# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Render C4-Helix memory, logic, and two-block CNOT programs from Berthusen et al., arXiv:2609.03194.

Rounds are unrolled calls to gadget functions. Each measurement emits a scalar
result with a round-free tag; its occurrence identifies the round. RELABEL uses
``mem_swap`` to exchange data and local ancilla bindings without a quantum gate.
Certification inventories output slots from the step-ordered circuit; layout
order is informational. Unique physical labels carry identity, resolved from
runtime provenance or compiled analysis. Per-tag occurrences assume result()
prints retain source order when measurements reorder.
DEMs instead use typed result references from the ASAP circuit's discovered
parities, so measurement reordering cannot misattribute detector records.
"""

from pecos.guppy_gen._certificate import certify_measurement_layout
from pecos.guppy_gen._module_loader import load_cached_guppy_source
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.helix.cnot import cnot_gadgets
from pecos.qec.helix.gadgets import HelixAllocation, c4_round_steps, memory_gadgets, memory_steps
from pecos.qec.helix.logic import logic_gadgets
from pecos.qec.helix.lowering import helix_result_classifier, lower_steps

# Logic fits too: at most 20 data + 4 weight-8 check ancillas; C4 and CZ flag gadgets use 2.
HELIX_MEMORY_NUM_QUBITS = 24
# Two blocks of 20 data plus the weight-8 gadget's 4 ancillas.
HELIX_CNOT_NUM_QUBITS = 44


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
        elif step.op_type == OpType.SZ:
            lines.append(f"    s({names[0]})")
        elif step.op_type == OpType.SZDG:
            lines.append(f"    sdg({names[0]})")
        elif step.op_type == OpType.CZ:
            lines.append(f"    cz({names[0]}, {names[1]})")
        elif step.op_type == OpType.X:
            lines.append(f"    x({names[0]})")
        elif step.op_type == OpType.Z:
            lines.append(f"    z({names[0]})")
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
    return _render_module(memory_gadgets(basis, rounds), f"memory_{basis.lower()}")


def render_helix_logic_source(basis: str, tokens: tuple[str, ...]) -> str:
    """Render checked logical words with one reusable function per gate token."""
    gadgets = [(f"gate_{name}" if name in tokens else name, steps) for name, steps in logic_gadgets(basis, tokens)]
    return _render_module(gadgets, f"logic_{basis.lower()}")


def _render_header(ops: set[OpType]) -> list[str]:
    """Render imports for the used operations and the shared code struct."""
    quantum_ops = {
        OpType.PREP: "qubit",
        OpType.H: "h",
        OpType.CX: "cx",
        OpType.SZ: "s",
        OpType.SZDG: "sdg",
        OpType.CZ: "cz",
        OpType.X: "x",
        OpType.Z: "z",
        OpType.MEASURE: "measure",
    }
    imports = sorted({"discard_array", "qubit"} | {quantum_ops[op] for op in ops if op in quantum_ops})
    return [
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, result",
        "from guppylang.std.mem import mem_swap",
        f"from guppylang.std.quantum import {', '.join(imports)}",
        "",
        "",
        "@guppy.struct",
        "class HelixCode_20_2_6:",
        "    data: array[qubit, 20]",
        "",
        "",
    ]


def _render_prepare(name: str, steps: list[CircuitStep]) -> list[str]:
    """Render preparation returning an owned code block."""
    lines = ["@guppy", f"def {name}() -> HelixCode_20_2_6:"]
    allocations = ", ".join("qubit()" for step in steps if step.op_type == OpType.PREP)
    lines.append(f"    code = HelixCode_20_2_6(array({allocations}))")
    lines.extend(_render_steps([step for step in steps if step.op_type != OpType.PREP]))
    return [*lines, "    return code", "", ""]


def _render_gadget(name: str, steps: list[CircuitStep]) -> list[str]:
    """Render a gadget borrowing one code block."""
    return ["@guppy", f"def {name}(code: HelixCode_20_2_6) -> None:", *_render_steps(steps), "", ""]


def _render_readout(name: str, steps: list[CircuitStep]) -> list[str]:
    """Render readout consuming an owned code block."""
    return [
        "@guppy",
        f"def {name}(code: HelixCode_20_2_6 @ owned) -> None:",
        *_render_steps(steps, readout=True),
        "    discard_array(code.data)",
        "",
        "",
    ]


def _render_module(gadgets: list[tuple[str, list[CircuitStep]]], program_name: str) -> str:
    """Build the shared struct, gadget definitions, and fresh-program factory."""
    prepare_name, prepare_steps = gadgets[0]
    readout_name, readout_steps = gadgets[-1]
    definitions = {f"c4_{block}_{variant}": c4_round_steps(block, variant) for block in range(5) for variant in "ab"}
    definitions.update(gadgets[1:-1])
    ops = {step.op_type for steps in [prepare_steps, *definitions.values(), readout_steps] for step in steps}
    lines = _render_header(ops)
    lines.extend(_render_prepare(prepare_name, prepare_steps))
    for name, steps in definitions.items():
        lines.extend(_render_gadget(name, steps))
    lines.extend(_render_readout(readout_name, readout_steps))
    lines.extend([f"def make_{program_name}():", "    @guppy", f"    def {program_name}() -> None:"])
    lines.append(f"        code = {prepare_name}()")
    lines.extend(f"        {gadget_name}(code)" for gadget_name, _ in gadgets[1:])
    lines.extend(["", f"    return {program_name}", ""])
    return "\n".join(lines)


def make_helix_memory(basis: str, rounds: int) -> object:
    """Compile memory and certify output slots whose identities are resolved from evidence."""
    source = render_helix_memory_source(basis, rounds)
    module = load_cached_guppy_source("helix", source)
    program = module[f"make_memory_{basis.lower()}"]()
    circuit = lower_steps(memory_steps(basis, rounds), HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    certify_measurement_layout(program, circuit, helix_result_classifier, family="helix")
    return program


def make_helix_logic(basis: str, tokens: tuple[str, ...]) -> object:
    """Compile logic and certify the output-slot inventory from step-order lowering."""
    source = render_helix_logic_source(basis, tokens)
    module = load_cached_guppy_source("helix", source)
    program = module[f"make_logic_{basis.lower()}"]()
    steps = [step for _, gadget in logic_gadgets(basis, tokens) for step in gadget]
    circuit = lower_steps(steps, HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    certify_measurement_layout(program, circuit, helix_result_classifier, family="helix")
    return program


def _render_cnot(steps: list[CircuitStep]) -> list[str]:
    """Render the transversal CX from block A data to the same block B data slots."""
    lines = ["@guppy", "def cnot(a: HelixCode_20_2_6, b: HelixCode_20_2_6) -> None:"]
    for step in steps:
        if (
            step.op_type != OpType.CX
            or len(step.qubits) != 2
            or not 0 <= step.qubits[0] < 20
            or step.qubits[1] != HelixAllocation.NUM_SLOTS + step.qubits[0]
        ):
            msg = f"Helix gadget cnot: not a transversal CX in step {step!r}"
            raise ValueError(msg)
        lines.append(f"    cx(a.data[{step.qubits[0]}], b.data[{step.qubits[0]}])")
    return [*lines, "", ""]


def render_helix_cnot_source(basis_a: str, basis_b: str) -> str:
    """Render both blocks with static result tags and a transversal CNOT."""
    gadgets = cnot_gadgets(basis_a, basis_b)
    definitions = {}
    for name, steps in gadgets:
        if name in definitions and steps != definitions[name]:
            msg = f"Helix gadget {name}: different steps from its first occurrence"
            raise ValueError(msg)
        definitions.setdefault(name, steps)
    lines = _render_header({step.op_type for steps in definitions.values() for step in steps})
    for name, steps in definitions.items():
        if name == "cnot":
            lines.extend(_render_cnot(steps))
            continue
        if not name.startswith(("A_", "B_")):
            msg = f"Helix gadget {name}: unknown gadget"
            raise ValueError(msg)
        offset = 0 if name.startswith("A_") else HelixAllocation.NUM_SLOTS
        local_steps = []
        for step in steps:
            if any(not offset <= slot < offset + HelixAllocation.NUM_SLOTS for slot in step.qubits):
                msg = f"Helix gadget {name}: outside block in step {step!r}"
                raise ValueError(msg)
            local_steps.append(CircuitStep(step.op_type, [slot - offset for slot in step.qubits], step.label))
        if "_prepare_" in name:
            lines.extend(_render_prepare(name, local_steps))
        elif "_readout_" in name:
            lines.extend(_render_readout(name, local_steps))
        else:
            lines.extend(_render_gadget(name, local_steps))
    program_name = f"cnot_{(basis_a + basis_b).lower()}"
    lines.extend([f"def make_{program_name}():", "    @guppy", f"    def {program_name}() -> None:"])
    for name, _ in gadgets:
        if name == "cnot":
            lines.append("        cnot(a, b)")
        else:
            block = name[0].lower()
            if "_prepare_" in name:
                lines.append(f"        {block} = {name}()")
            else:
                lines.append(f"        {name}({block})")
    lines.extend(["", f"    return {program_name}", ""])
    return "\n".join(lines)


def make_helix_cnot(basis_a: str, basis_b: str) -> object:
    """Compile two-block CNOT and certify output slots from step-order lowering."""
    source = render_helix_cnot_source(basis_a, basis_b)
    module = load_cached_guppy_source("helix", source)
    program = module[f"make_cnot_{(basis_a + basis_b).lower()}"]()
    steps = [step for _, gadget in cnot_gadgets(basis_a, basis_b) for step in gadget]
    circuit = lower_steps(steps, 2 * HelixAllocation.NUM_SLOTS, schedule="step").tick_circuit
    certify_measurement_layout(program, circuit, helix_result_classifier, family="helix")
    return program
