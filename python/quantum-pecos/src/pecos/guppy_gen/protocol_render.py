# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Render surface protocols separately from the byte-stable memory module."""

import hashlib

from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface import SurfacePatch, gadgets
from pecos.qec.surface.circuit_builder import QubitAllocation

_MODULE_CACHE: dict[str, dict] = {}


def _rounds(count: str, patches: tuple[str, ...], *, swapped: bool = False) -> list[str]:
    function = "syndrome_extraction_swapped" if swapped else "syndrome_extraction"
    lines = [f"        for _ in range(comptime({count})):"]
    for patch in patches:
        lines.extend(
            [
                f"            syn = {function}_{patch}({patch})",
                f'            output("synx_{patch}", syn.synx)',
                f'            output("synz_{patch}", syn.synz)',
            ],
        )
    return lines


def _readout(patch: str, basis: str = "z") -> list[str]:
    return [f"        final = measure_{basis}_basis({patch})", f'        output("final_{patch}", final)']


def render_surface_protocol_module(patch: SurfacePatch) -> str:
    """Render odd square rotated surface experiments with scoped measurement provenance."""
    geom = patch.geometry
    if patch.dx < 3 or patch.dx != patch.dz or patch.dx % 2 == 0 or not geom.rotated:
        msg = "Surface protocols require distance >= 3, odd dimensions, a square patch (dx == dz), and rotated=True"
        raise ValueError(msg)
    dx, dz = patch.dx, patch.dz
    allocation = gadgets.default_allocation(patch)
    offset = geom.num_qubits
    target = QubitAllocation(
        [q + offset for q in allocation.data_qubits],
        [q + offset for q in allocation.x_ancilla_qubits],
        [q + offset for q in allocation.z_ancilla_qubits],
    )
    lines = [
        f'"""Surface code protocols for dx={dx}, dz={dz}.',
        "",
        "Scoped scalar sidebands certify measurement provenance for each patch role.",
        "The surface DEM parser indexes scoped tags but never references them.",
        "The builder protocol has no init round. The certified-slot DEM route",
        "remains single-patch until the parser learns scopes.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, comptime, owned, output",
        "from guppylang.std.quantum import cx, h, qubit, s, x",
        "from guppylang.std.quantum import collect_measurements, measure, measure_array",
        "from pecos.guppy_gen.variant import variant_scoped",
        "",
        "",
        "@guppy.struct",
        f"class SurfaceCode_{dx}x{dz}:",
        f'    """Surface patch with {geom.num_data} data qubits."""',
        f"    data: array[qubit, {geom.num_data}]",
        "",
        "",
        "@guppy.struct",
        f"class Syndrome_{dx}x{dz}:",
        '    """Results in current X and Z family order."""',
        f"    synx: array[bool, {len(geom.x_stabilizers)}]",
        f"    synz: array[bool, {len(geom.z_stabilizers)}]",
        "",
        "",
    ]
    functions = [gadgets.prep_gadget(patch, allocation, basis=basis) for basis in ("Z", "X", "Y")]
    functions.extend(gadgets.measure_out_gadget(patch, allocation, basis=basis) for basis in ("Z", "X"))
    functions.append(gadgets.logical_pauli_gadget(patch, allocation, pauli="X"))
    functions.append(gadgets.transversal_layer_gadget(patch, allocation, gate="H"))
    functions.append(gadgets.transversal_cx_gadget(patch, allocation, patch, target))
    for gadget in functions:
        lines.extend(render_gadget_function(gadget))
        lines.extend(["", ""])
    for scope in ("a", "ctrl", "tgt", "data", "anc"):
        gadget = gadgets.syndrome_round_gadget(patch, allocation, round_index=0)
        lines.extend(render_gadget_function(gadget, tag_scope=scope))
        lines.extend(["", ""])
    swapped = gadgets.syndrome_round_gadget(patch, allocation, round_index=0, x_z_swapped=True)
    lines.extend(render_gadget_function(swapped, tag_scope="a"))
    lines.extend(["", ""])
    lines.extend(
        [
            "def make_h_experiment(num_rounds: int, *, logical_x: bool = False):",
            '    """Prepare Z, extract full rounds, apply H, extract swapped rounds, read X."""',
            "    def h_experiment() -> None:",
            '        """Logical H experiment."""',
            "        a = prep_z_basis()",
            "        if comptime(logical_x):",
            "            apply_logical_x(a)",
            *_rounds("num_rounds", ("a",)),
            "        transversal_h(a)",
            *_rounds("num_rounds", ("a",), swapped=True),
            *_readout("a", "x"),
            "    return guppy(variant_scoped(h_experiment, num_rounds, logical_x))",
            "",
            "",
            "def make_transversal_cx(num_rounds: int, *, control_x: bool = False):",
            '    """Prepare control and target, extract rounds on each side of CX, read Z."""',
            "    def transversal_cx_experiment() -> None:",
            '        """Logical CX experiment."""',
            "        ctrl = prep_z_basis()",
            "        if comptime(control_x):",
            "            apply_logical_x(ctrl)",
            "        tgt = prep_z_basis()",
            *_rounds("num_rounds", ("ctrl", "tgt")),
            "        transversal_cx(ctrl, tgt)",
            *_rounds("num_rounds", ("ctrl", "tgt")),
            *_readout("ctrl"),
            *_readout("tgt"),
            "    return guppy(variant_scoped(transversal_cx_experiment, num_rounds, control_x))",
            "",
            "",
            "def make_sz_teleportation(rounds_before: int, rounds_after: int, trailing_data_rounds: int):",
            '    """Y-ancilla teleportation followed by data-only memory and Z readout."""',
            "    def sz_teleportation() -> None:",
            '        """SZ teleportation experiment."""',
            "        data = prep_z_basis()",
            "        anc = prep_y_basis()",
            *_rounds("rounds_before", ("data", "anc")),
            "        transversal_cx(data, anc)",
            *_rounds("rounds_after", ("data", "anc")),
            *_readout("anc"),
            *_rounds("trailing_data_rounds", ("data",)),
            *_readout("data"),
            "    return guppy(variant_scoped(sz_teleportation, rounds_before, rounds_after, trailing_data_rounds))",
            "",
            "",
            "def make_t_injection(rounds_before: int, rounds_after: int):",
            '    """Clifford stand-in: ancilla prepared in |+>, no T, no conditional S."""',
            "    def t_injection() -> None:",
            '        """Clifford stand-in for T injection."""',
            "        data = prep_z_basis()",
            "        anc = prep_x_basis()",
            *_rounds("rounds_before", ("data", "anc")),
            "        transversal_cx(data, anc)",
            *_rounds("rounds_after", ("data", "anc")),
            *_readout("data"),
            *_readout("anc"),
            "    return guppy(variant_scoped(t_injection, rounds_before, rounds_after))",
            "",
        ],
    )
    return "\n".join(lines)


def load_surface_protocol_module(patch: SurfacePatch) -> dict:
    """Cache by patch identity and source, including custom stabilizer/logical supports."""
    source = render_surface_protocol_module(patch)
    geom = patch.geometry
    digest = hashlib.sha256(source.encode()).hexdigest()
    key = f"surface_protocol_{patch.dx}x{patch.dz}_{geom.orientation.name}_{geom.rotated}_{digest}"
    if key not in _MODULE_CACHE:
        _MODULE_CACHE[key] = load_guppy_source(source, _get_temp_dir() / f"{key}.py", f"pecos._generated.{key}")
    return _MODULE_CACHE[key]
