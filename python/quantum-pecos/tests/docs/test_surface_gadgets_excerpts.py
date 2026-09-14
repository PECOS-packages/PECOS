# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Keep the guide's displayed Guppy source aligned with the public renderers."""

import re
from pathlib import Path

import pytest
from pecos.guppy_gen import render_surface_protocol_module
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface import SurfacePatch, gadgets
from pecos.qec.surface.circuit_builder import QubitAllocation

PAGE = Path(__file__).resolve().parents[4] / "docs/user-guide/surface-gadgets.md"


def _excerpts() -> list[tuple[str, str]]:
    """Associate each text fence with its enclosing H3 heading."""
    excerpts = []
    section = ""
    language = None
    lines = []
    for line in PAGE.read_text().splitlines():
        if line.startswith("```"):
            if language is None:
                language = line.removeprefix("```")
                lines = []
            else:
                if language == "text":
                    excerpts.append((section, "\n".join(lines)))
                language = None
        elif language is not None:
            lines.append(line)
        elif line.startswith("### "):
            section = line.removeprefix("### ")
        elif re.match(r"^#{1,2} ", line):
            section = ""
    assert excerpts, "The guide must contain rendered-source excerpts"
    return excerpts


@pytest.fixture(scope="module")
def rendered_sources() -> dict[str, list[str]]:
    """Render the representative gadget used by each guide section."""
    patch = SurfacePatch.create(distance=3)
    allocation = gadgets.default_allocation(patch)
    offset = patch.geometry.num_qubits
    target = QubitAllocation(
        [q + offset for q in allocation.data_qubits],
        [q + offset for q in allocation.x_ancilla_qubits],
        [q + offset for q in allocation.z_ancilla_qubits],
    )
    definitions = {
        **{f"Preparation in {basis}": gadgets.prep_gadget(patch, allocation, basis=basis) for basis in "ZXY"},
        "Initial syndrome projection": gadgets.init_syndrome_gadget(patch, allocation, basis="Z"),
        "Syndrome round": gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True),
        "Measure-out": gadgets.measure_out_gadget(patch, allocation, basis="X"),
        "Logical Pauli": gadgets.logical_pauli_gadget(patch, allocation, pauli="X"),
        "Transversal H": gadgets.transversal_layer_gadget(patch, allocation, gate="H"),
        "Transversal CX": gadgets.transversal_cx_gadget(patch, allocation, patch, target),
    }
    sources = {section: render_gadget_function(gadget) for section, gadget in definitions.items()}
    sources["Physical S and S-dagger layers"] = [
        line
        for gate in ("SZ", "SZDG")
        for line in render_gadget_function(gadgets.transversal_layer_gadget(patch, allocation, gate=gate))
    ]
    sources["H experiment"] = render_surface_protocol_module(patch).splitlines()
    return sources


@pytest.mark.parametrize(("section", "excerpt"), _excerpts(), ids=lambda value: value.splitlines()[0])
def test_excerpt_matches_rendered_source(section: str, excerpt: str, rendered_sources: dict[str, list[str]]) -> None:
    """Every displayed line, except omission markers, must come from its renderer."""
    assert section in rendered_sources, f"No renderer mapped for {section!r}"
    rendered = {line.strip() for line in rendered_sources[section]}
    for line in excerpt.splitlines():
        if line.strip() != "...":
            assert line.strip() in rendered, f"{section}: {line}"
