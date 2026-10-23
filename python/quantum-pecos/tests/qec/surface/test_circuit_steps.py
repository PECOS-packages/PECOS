# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Shared step identity and renderer handling of every step operation."""

import pytest
from pecos import qec
from pecos.qec import surface
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.surface.circuit_builder import StimRenderer, TickCircuitRenderer
from pecos.qec.surface.gadgets import default_allocation

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
