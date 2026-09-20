# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Keep the guide's displayed Guppy source aligned with the public renderers."""

import re
from pathlib import Path

import pytest
from pecos.guppy_gen import generate_surface_code_module, render_surface_protocol_module
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface import SurfacePatch, gadgets
from pecos.qec.surface.circuit_builder import QubitAllocation

GUIDES = Path(__file__).resolve().parents[4] / "docs/user-guide"


def _excerpts(page: Path) -> list[tuple[str, str, str]]:
    """Associate each text fence with its page and enclosing H2 or H3 heading."""
    excerpts = []
    section = ""
    language = None
    lines = []
    for line in page.read_text().splitlines():
        if line.startswith("```"):
            if language is None:
                language = line.removeprefix("```")
                lines = []
            else:
                if language == "text":
                    excerpts.append((page.name, section, "\n".join(lines)))
                language = None
        elif language is not None:
            lines.append(line)
        elif heading := re.match(r"^#{2,3} (.+)$", line):
            section = heading[1]
        elif line.startswith("# "):
            section = ""
    assert excerpts, "The guide must contain rendered-source excerpts"
    return excerpts


@pytest.fixture(scope="module")
def rendered_sources() -> dict[tuple[str, str], list[str]]:
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
        "Fold-transversal S": gadgets.fold_s_round_gadget(patch, allocation, round_index=0),
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
    return {
        **{("surface-gadgets.md", section): lines for section, lines in sources.items()},
        ("qec-guppy.md", "Generated Code Structure"): generate_surface_code_module(d=3).splitlines(),
    }


@pytest.mark.parametrize(
    ("page", "section", "excerpt"),
    [*_excerpts(GUIDES / "surface-gadgets.md"), *_excerpts(GUIDES / "qec-guppy.md")],
    ids=lambda value: value.splitlines()[0],
)
def test_excerpt_matches_rendered_source(
    page: str,
    section: str,
    excerpt: str,
    rendered_sources: dict[tuple[str, str], list[str]],
) -> None:
    """Displayed segments must be contiguous in the source and retain their order."""
    key = (page, section)
    assert key in rendered_sources, f"No renderer mapped for {key!r}"
    rendered = [line.strip() for line in rendered_sources[key]]
    segments: list[list[str]] = [[]]
    for line in excerpt.splitlines():
        if line.strip() == "...":
            segments.append([])
        else:
            segments[-1].append(line.strip())
    cursor = 0
    for segment in filter(None, segments):
        start = next(
            (
                index
                for index in range(cursor, len(rendered) - len(segment) + 1)
                if rendered[index : index + len(segment)] == segment
            ),
            None,
        )
        assert start is not None, f"{page}: {section}: no contiguous segment after line {cursor}: {segment!r}"
        cursor = start + len(segment)
