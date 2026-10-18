# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Neutral step identity and explicit rejection at existing dispatch boundaries."""

from dataclasses import replace

import pytest
from pecos import qec
from pecos.guppy_gen import gadget_render
from pecos.guppy_gen.gadget_render import _allocation_epochs
from pecos.guppy_gen.surface import generate_guppy_source
from pecos.qec import surface
from pecos.qec.circuit_steps import CircuitStep, OpType
from pecos.qec.surface import circuit_builder as builder
from pecos.qec.surface.circuit_builder import (
    _SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING,
    _szz_flow_apply_gate_to_pauli,
    _szz_flow_compose_pending_gate,
)
from pecos.qec.surface.gadgets import default_allocation, prep_gadget
from pecos.qec.surface.logical_circuit import _CircuitGenerator


class GeneratorProbe(_CircuitGenerator):
    """Expose the internal step emission boundary for testing."""

    def emit_steps(self, step_lists):
        return self._emit_steps(step_lists)


def test_circuit_step_imports() -> None:
    assert surface.CircuitStep is qec.CircuitStep is CircuitStep
    assert surface.OpType is qec.OpType is OpType
    first = CircuitStep(OpType.TICK)
    second = CircuitStep(OpType.TICK)
    assert first.qubits == []
    assert first.qubits is not second.qubits
    assert first.label == ""


@pytest.mark.parametrize(
    "consumer",
    [
        "StimRenderer",
        "GuppyRenderer",
        "DagCircuitRenderer",
        "TickCircuitRenderer",
        "_analyze_szz_forward_flow",
        "_lower_szz_forward_flow_ops",
        "_szz_flow_apply_gate_to_pauli",
        "_szz_flow_compose_pending_gate",
        "_CircuitGenerator._emit_steps",
        "_allocation_epochs",
        "render_gadget_function",
        "_append_szz_flow_gate",
        "_append_szz_physical_prefix_gate",
    ],
)
def test_consumers_reject_relabel(consumer: str, monkeypatch: pytest.MonkeyPatch) -> None:
    patch = surface.SurfacePatch.create(distance=3)
    allocation = default_allocation(patch)
    steps = [CircuitStep(OpType.RELABEL, [0, 1])]
    gadget = replace(prep_gadget(patch, allocation, basis="Z"), steps=tuple(steps))

    def consume() -> None:
        if consumer.endswith("Renderer"):
            getattr(builder, consumer)().render(steps, allocation, patch, 0, "Z")
        elif consumer in {"_analyze_szz_forward_flow", "_lower_szz_forward_flow_ops"}:
            getattr(builder, consumer)(steps)
        elif consumer == "_szz_flow_apply_gate_to_pauli":
            _szz_flow_apply_gate_to_pauli(steps[0].op_type, 1)
        elif consumer == "_szz_flow_compose_pending_gate":
            _szz_flow_compose_pending_gate((1, 3), steps[0].op_type)
        elif consumer == "_CircuitGenerator._emit_steps":
            GeneratorProbe({}, []).emit_steps([tuple(steps)])
        elif consumer == "_allocation_epochs":
            _allocation_epochs(gadget, ("data",))
        elif consumer == "render_gadget_function":
            gadget_render.render_gadget_function(gadget)
        else:
            # Nested Guppy emitters consume gates from this lowering table.
            # Inject the supplied step's operation through their actual caller.
            for pending in _SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING:
                gates = (
                    (steps[0].op_type, OpType.H) if consumer == "_append_szz_flow_gate" else (None, steps[0].op_type)
                )
                monkeypatch.setitem(_SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING, pending, gates)
            generate_guppy_source(patch, interaction_basis="szz")

    with pytest.raises(ValueError, match=rf"{consumer} does not support RELABEL steps"):
        consume()
