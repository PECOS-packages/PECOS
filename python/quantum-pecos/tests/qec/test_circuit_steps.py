# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Shared step identity and renderer handling of every step operation."""

from dataclasses import replace
from enum import Enum, auto

import pytest
from pecos import qec
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec import surface
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.surface import circuit_builder
from pecos.qec.surface.circuit_builder import StimRenderer, TickCircuitRenderer
from pecos.qec.surface.gadgets import default_allocation, prep_gadget

STIM_UNSUPPORTED = [OpType.PREP, OpType.F, OpType.FDG, OpType.SY, OpType.SYDG, OpType.X, OpType.Z]


def _single_qubit_steps(op_type: OpType) -> list[CircuitStep]:
    return [CircuitStep(OpType.ALLOC, [0]), CircuitStep(op_type, [0]), CircuitStep(OpType.MEASURE, [0], "m")]


def test_circuit_step_imports() -> None:
    assert surface.CircuitStep is qec.CircuitStep is CircuitStep
    assert surface.OpType is qec.OpType is OpType
    first = CircuitStep(OpType.TICK)
    second = CircuitStep(OpType.TICK)
    assert first.qubits == []
    assert first.qubits is not second.qubits
    assert first.label == ""


@pytest.mark.parametrize("op_type", STIM_UNSUPPORTED, ids=lambda op: op.name)
def test_stim_renderer_rejects_unsupported_steps(op_type: OpType) -> None:
    patch = surface.SurfacePatch.create(distance=3)
    renderer = StimRenderer(add_detectors=False)
    with pytest.raises(ValueError, match=rf"StimRenderer does not support {op_type.name} steps"):
        renderer.render(_single_qubit_steps(op_type), default_allocation(patch), patch, 0, "Z")


@pytest.mark.parametrize("op_type", STIM_UNSUPPORTED, ids=lambda op: op.name)
def test_tick_renderer_renders_steps_stim_rejects(op_type: OpType) -> None:
    patch = surface.SurfacePatch.create(distance=3)
    renderer = TickCircuitRenderer(add_detectors=False)
    circuit = renderer.render(_single_qubit_steps(op_type), default_allocation(patch), patch, 0, "Z")
    assert circuit.gate_count() == 3


class _UnhandledOp(Enum):
    """Stands in for an operation added to OpType without a renderer case."""

    NEW = auto()


def test_tick_renderer_rejects_an_operation_without_a_case() -> None:
    patch = surface.SurfacePatch.create(distance=3)
    steps = [CircuitStep(OpType.ALLOC, [0]), CircuitStep(_UnhandledOp.NEW, [0])]
    with pytest.raises(ValueError, match=r"^Unsupported TickCircuit operation: NEW$"):
        TickCircuitRenderer(add_detectors=False).render(steps, default_allocation(patch), patch, 0, "Z")


@pytest.mark.parametrize(
    "consumer",
    ["StimRenderer", "DagCircuitRenderer", "TickCircuitRenderer", "GuppyRenderer", "render_gadget_function"],
)
def test_readers_reject_relabel(consumer: str) -> None:
    patch = surface.SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    step = CircuitStep(OpType.RELABEL, [0, 1], "exchange data")
    if consumer == "render_gadget_function":
        gadget = replace(prep_gadget(patch, allocation, basis="Z"), steps=(step,))

        def consume() -> None:
            render_gadget_function(gadget)

    else:

        def consume() -> None:
            getattr(circuit_builder, consumer)().render([step], allocation, patch, 0, "Z")

    with pytest.raises(ValueError, match="does not support RELABEL step") as error:
        consume()
    assert repr(step) in str(error.value)
    assert "pecos.qec.helix.lower_steps" in str(error.value)
