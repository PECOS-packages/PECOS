# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generic transversal operations for CSS codes.

This module provides transversal CNOT generation that works with any CSS code
(surface codes, color codes, etc.) by abstracting over the code-specific details.

Transversal CNOT between two CSS code blocks applies CX(ctrl[i], tgt[i]) for all
data qubits i. This preserves the CSS structure: X errors on control propagate
to X errors on target, Z errors on target propagate to Z errors on control.
"""

import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


class CSSCodeType(Enum):
    """Supported CSS code types."""

    SURFACE = "surface"
    COLOR = "color"


@runtime_checkable
class CSSCodeSpec(Protocol):
    """Protocol for CSS code specifications."""

    @property
    def num_data(self) -> int:
        """Number of data qubits."""
        ...

    @property
    def num_x_stabilizers(self) -> int:
        """Number of X stabilizers."""
        ...

    @property
    def num_z_stabilizers(self) -> int:
        """Number of Z stabilizers."""
        ...

    def get_logical_x(self) -> tuple[int, ...]:
        """Qubits for logical X operator."""
        ...

    def get_logical_z(self) -> tuple[int, ...]:
        """Qubits for logical Z operator."""
        ...


@dataclass
class TransversalConfig:
    """Configuration for transversal CNOT generation."""

    code_type: CSSCodeType
    distance: int
    num_rounds: int = 1
    ctrl_logical_x: bool = False  # Apply logical X to control before CNOT


# Module state container (avoids global statement)
class _ModuleState:
    """Container for module-level mutable state."""

    temp_dir: ClassVar[Path | None] = None
    css_transversal_cache: ClassVar[dict[str, dict]] = {}


_state = _ModuleState()


def _get_temp_dir() -> Path:
    """Get or create temporary directory for generated code."""
    if _state.temp_dir is None:
        _state.temp_dir = Path(tempfile.mkdtemp(prefix="pecos_guppy_css_trans_"))
    return _state.temp_dir


def _get_surface_code_info(d: int) -> dict:
    """Get surface code parameters."""
    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    geom = patch.geometry

    return {
        "num_data": d * d,
        "num_x_stab": len(geom.x_stabilizers),
        "num_z_stab": len(geom.z_stabilizers),
        "x_stabilizers": geom.x_stabilizers,
        "z_stabilizers": geom.z_stabilizers,
        "logical_x": [i * d for i in range(d)],  # Left column
        "logical_z": list(range(d)),  # Top row
        "struct_name": f"SurfaceCode_d{d}",
        "syndrome_name": f"SurfaceSyndrome_d{d}",
    }


def _get_color_code_info(d: int) -> dict:
    """Get color code parameters."""
    from pecos.qec.color import ColorCode488

    code = ColorCode488.create(distance=d)

    return {
        "num_data": code.num_data,
        "num_x_stab": code.num_stabilizers,  # Same stabilizers for X and Z
        "num_z_stab": code.num_stabilizers,
        "stabilizers": code.stabilizers,
        "logical_x": list(code.get_logical_x()),
        "logical_z": list(code.get_logical_z()),
        "struct_name": f"ColorCode_d{d}",
        "syndrome_name": f"ColorSyndrome_d{d}",
    }


def generate_surface_transversal_source(d: int) -> str:
    """Generate transversal CNOT source for surface codes."""
    info = _get_surface_code_info(d)
    num_data = info["num_data"]
    num_x_stab = info["num_x_stab"]
    num_z_stab = info["num_z_stab"]

    lines = [
        f'"""Transversal CNOT for distance-{d} surface codes.',
        "",
        "Auto-generated for logical CNOT between two patches.",
        '"""',
        "",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, result, comptime",
        "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x",
        "from guppylang.std.qsystem import measure_and_reset",
        "",
        "",
    ]

    # Struct definitions
    lines.extend(
        [
            "@guppy.struct",
            f"class {info['struct_name']}:",
            f'    """Surface code patch d={d}."""',
            "",
            f"    data: array[qubit, {num_data}]",
            "",
            "",
            "@guppy.struct",
            f"class {info['syndrome_name']}:",
            f'    """Syndrome for d={d} surface code."""',
            "",
            f"    synx: array[bool, {num_x_stab}]",
            f"    synz: array[bool, {num_z_stab}]",
            "",
            "",
        ],
    )

    # X stabilizer measurements
    lines.append("# === X Stabilizer Measurements ===")
    lines.append("")

    for stab in info["x_stabilizers"]:
        lines.extend(
            [
                "@guppy",
                f"def measure_x_stab_{stab.index}(ax: qubit, data: array[qubit, {num_data}]) -> bool:",
                f'    """Measure X stabilizer {stab.index}."""',
                "    h(ax)",
            ],
        )
        lines.extend(f"    cx(ax, data[{q}])" for q in stab.data_qubits)
        lines.extend(
            [
                "    h(ax)",
                "    return measure_and_reset(ax)",
                "",
                "",
            ],
        )

    # Z stabilizer measurements
    lines.append("# === Z Stabilizer Measurements ===")
    lines.append("")

    for stab in info["z_stabilizers"]:
        lines.extend(
            [
                "@guppy",
                f"def measure_z_stab_{stab.index}(az: qubit, data: array[qubit, {num_data}]) -> bool:",
                f'    """Measure Z stabilizer {stab.index}."""',
            ],
        )
        lines.extend(f"    cx(data[{q}], az)" for q in stab.data_qubits)
        lines.extend(
            [
                "    return measure_and_reset(az)",
                "",
                "",
            ],
        )

    # Syndrome extraction
    x_calls = ", ".join(f"sx{s.index}" for s in info["x_stabilizers"])
    z_calls = ", ".join(f"sz{s.index}" for s in info["z_stabilizers"])

    lines.extend(
        [
            "# === Syndrome Extraction ===",
            "",
            "@guppy",
            "def syndrome_extraction(",
            f"    surf: {info['struct_name']},",
            "    ax: qubit,",
            "    az: qubit,",
            f") -> {info['syndrome_name']}:",
            '    """Extract full syndrome."""',
            "    # Z stabilizers",
        ],
    )

    lines.extend(f"    sz{stab.index} = measure_z_stab_{stab.index}(az, surf.data)" for stab in info["z_stabilizers"])

    lines.append("")
    lines.append("    # X stabilizers")

    lines.extend(f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, surf.data)" for stab in info["x_stabilizers"])

    lines.extend(
        [
            "",
            f"    synx = array({x_calls})",
            f"    synz = array({z_calls})",
            "",
            f"    return {info['syndrome_name']}(synx, synz)",
            "",
            "",
        ],
    )

    # Initialization
    lines.extend(
        [
            "# === Initialization ===",
            "",
            "@guppy",
            f"def init_z_basis(surf: {info['struct_name']}, ax: qubit) -> array[bool, {num_x_stab}]:",
            '    """Initialize logical |0_L> and extract initial X syndrome."""',
        ],
    )

    lines.extend(f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, surf.data)" for stab in info["x_stabilizers"])

    lines.extend(
        [
            "",
            f"    return array({x_calls})",
            "",
            "",
        ],
    )

    # Apply logical X
    lines.extend(
        [
            "@guppy",
            f"def apply_logical_x(surf: {info['struct_name']}) -> None:",
            '    """Apply logical X operator."""',
        ],
    )
    lines.extend(f"    x(surf.data[{q}])" for q in info["logical_x"])
    lines.extend(
        [
            "",
            "",
        ],
    )

    # Transversal CNOT
    lines.extend(
        [
            "# === Transversal Operations ===",
            "",
            "@guppy",
            f"def transversal_cnot(ctrl: {info['struct_name']}, tgt: {info['struct_name']}) -> None:",
            '    """Apply transversal CNOT: ctrl[i] controls tgt[i]."""',
            f"    for i in range({num_data}):",
            "        cx(ctrl.data[i], tgt.data[i])",
            "",
            "",
        ],
    )

    # Measurement
    lines.extend(
        [
            "# === Measurement ===",
            "",
            "@guppy",
            f"def measure_z_basis(surf: {info['struct_name']} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in Z basis."""',
            "    return measure_array(surf.data)",
            "",
            "",
        ],
    )

    # Factory functions
    _add_transversal_factory_functions(lines, info, num_data, num_x_stab)

    return "\n".join(lines)


def generate_color_transversal_source(d: int) -> str:
    """Generate transversal CNOT source for color codes."""
    info = _get_color_code_info(d)
    num_data = info["num_data"]
    num_stab = info["num_x_stab"]  # Same for X and Z

    lines = [
        f'"""Transversal CNOT for distance-{d} color codes.',
        "",
        "Auto-generated for logical CNOT between two patches.",
        '"""',
        "",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, result, comptime",
        "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x",
        "from guppylang.std.qsystem import measure_and_reset",
        "",
        "",
    ]

    # Struct definitions
    lines.extend(
        [
            "@guppy.struct",
            f"class {info['struct_name']}:",
            f'    """Color code patch d={d}."""',
            "",
            f"    data: array[qubit, {num_data}]",
            "",
            "",
            "@guppy.struct",
            f"class {info['syndrome_name']}:",
            f'    """Syndrome for d={d} color code."""',
            "",
            f"    synx: array[bool, {num_stab}]",
            f"    synz: array[bool, {num_stab}]",
            "",
            "",
        ],
    )

    # X stabilizer measurements (H-CNOT-H pattern)
    lines.append("# === X Stabilizer Measurements ===")
    lines.append("")

    for stab in info["stabilizers"]:
        lines.extend(
            [
                "@guppy",
                f"def measure_x_stab_{stab.index}(ax: qubit, data: array[qubit, {num_data}]) -> bool:",
                f'    """Measure X stabilizer {stab.index} ({stab.color})."""',
                "    h(ax)",
            ],
        )
        lines.extend(f"    cx(ax, data[{q}])" for q in stab.qubits)
        lines.extend(
            [
                "    h(ax)",
                "    return measure_and_reset(ax)",
                "",
                "",
            ],
        )

    # Z stabilizer measurements
    lines.append("# === Z Stabilizer Measurements ===")
    lines.append("")

    for stab in info["stabilizers"]:
        lines.extend(
            [
                "@guppy",
                f"def measure_z_stab_{stab.index}(az: qubit, data: array[qubit, {num_data}]) -> bool:",
                f'    """Measure Z stabilizer {stab.index} ({stab.color})."""',
            ],
        )
        lines.extend(f"    cx(data[{q}], az)" for q in stab.qubits)
        lines.extend(
            [
                "    return measure_and_reset(az)",
                "",
                "",
            ],
        )

    # Syndrome extraction
    x_calls = ", ".join(f"sx{s.index}" for s in info["stabilizers"])
    z_calls = ", ".join(f"sz{s.index}" for s in info["stabilizers"])

    lines.extend(
        [
            "# === Syndrome Extraction ===",
            "",
            "@guppy",
            "def syndrome_extraction(",
            f"    code: {info['struct_name']},",
            "    ax: qubit,",
            "    az: qubit,",
            f") -> {info['syndrome_name']}:",
            '    """Extract full syndrome."""',
            "    # Z stabilizers first",
        ],
    )

    lines.extend(f"    sz{stab.index} = measure_z_stab_{stab.index}(az, code.data)" for stab in info["stabilizers"])

    lines.append("")
    lines.append("    # X stabilizers")

    lines.extend(f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, code.data)" for stab in info["stabilizers"])

    lines.extend(
        [
            "",
            f"    synx = array({x_calls})",
            f"    synz = array({z_calls})",
            "",
            f"    return {info['syndrome_name']}(synx, synz)",
            "",
            "",
        ],
    )

    # Initialization
    lines.extend(
        [
            "# === Initialization ===",
            "",
            "@guppy",
            f"def init_z_basis(code: {info['struct_name']}, ax: qubit) -> array[bool, {num_stab}]:",
            '    """Initialize logical |0_L> and extract initial X syndrome."""',
        ],
    )

    lines.extend(f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, code.data)" for stab in info["stabilizers"])

    lines.extend(
        [
            "",
            f"    return array({x_calls})",
            "",
            "",
        ],
    )

    # Apply logical X
    lines.extend(
        [
            "@guppy",
            f"def apply_logical_x(code: {info['struct_name']}) -> None:",
            '    """Apply logical X operator."""',
        ],
    )
    lines.extend(f"    x(code.data[{q}])" for q in info["logical_x"])
    lines.extend(
        [
            "",
            "",
        ],
    )

    # Transversal CNOT
    lines.extend(
        [
            "# === Transversal Operations ===",
            "",
            "@guppy",
            f"def transversal_cnot(ctrl: {info['struct_name']}, tgt: {info['struct_name']}) -> None:",
            '    """Apply transversal CNOT: ctrl[i] controls tgt[i]."""',
            f"    for i in range({num_data}):",
            "        cx(ctrl.data[i], tgt.data[i])",
            "",
            "",
        ],
    )

    # Measurement
    lines.extend(
        [
            "# === Measurement ===",
            "",
            "@guppy",
            f"def measure_z_basis(code: {info['struct_name']} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in Z basis."""',
            "    return measure_array(code.data)",
            "",
            "",
        ],
    )

    # Factory functions
    _add_color_transversal_factory_functions(lines, info, num_data, num_stab)

    return "\n".join(lines)


def _add_transversal_factory_functions(
    lines: list[str],
    info: dict,
    num_data: int,
    _num_stab: int,
) -> None:
    """Add factory functions for surface code transversal experiments."""
    struct_name = info["struct_name"]

    lines.extend(
        [
            "# === Transversal Experiments ===",
            "",
            "def make_transversal_cnot(num_rounds: int):",
            '    """Create transversal CNOT experiment: |0_L>|0_L> -> |0_L>|0_L>."""',
            "",
            "    @guppy",
            "    def transversal_cnot_exp() -> None:",
            '        """Transversal CNOT experiment."""',
            f"        ctrl_data = array(qubit() for _ in range({num_data}))",
            f"        tgt_data = array(qubit() for _ in range({num_data}))",
            "        ax_ctrl = qubit()",
            "        az_ctrl = qubit()",
            "        ax_tgt = qubit()",
            "        az_tgt = qubit()",
            "",
            f"        ctrl = {struct_name}(ctrl_data)",
            f"        tgt = {struct_name}(tgt_data)",
            "",
            "        # Initialize both patches in |0_L>",
            "        init_syn_ctrl = init_z_basis(ctrl, ax_ctrl)",
            "        init_syn_tgt = init_z_basis(tgt, ax_tgt)",
            '        result("init_synx_ctrl", init_syn_ctrl)',
            '        result("init_synx_tgt", init_syn_tgt)',
            "",
            "        # Apply transversal CNOT",
            "        transversal_cnot(ctrl, tgt)",
            "",
            "        # Syndrome extraction rounds",
            "        for _t in range(comptime(num_rounds)):",
            "            syn_ctrl = syndrome_extraction(ctrl, ax_ctrl, az_ctrl)",
            "            syn_tgt = syndrome_extraction(tgt, ax_tgt, az_tgt)",
            '            result("synx_ctrl", syn_ctrl.synx)',
            '            result("synz_ctrl", syn_ctrl.synz)',
            '            result("synx_tgt", syn_tgt.synx)',
            '            result("synz_tgt", syn_tgt.synz)',
            "",
            "        # Final measurement",
            "        final_ctrl = measure_z_basis(ctrl)",
            "        final_tgt = measure_z_basis(tgt)",
            '        result("final_ctrl", final_ctrl)',
            '        result("final_tgt", final_tgt)',
            "",
            "        discard(ax_ctrl)",
            "        discard(az_ctrl)",
            "        discard(ax_tgt)",
            "        discard(az_tgt)",
            "",
            "    return transversal_cnot_exp",
            "",
            "",
            "def make_transversal_cnot_with_x(num_rounds: int):",
            '    """Create transversal CNOT experiment: |1_L>|0_L> -> |1_L>|1_L>."""',
            "",
            "    @guppy",
            "    def transversal_cnot_with_x_exp() -> None:",
            '        """Transversal CNOT with X experiment."""',
            f"        ctrl_data = array(qubit() for _ in range({num_data}))",
            f"        tgt_data = array(qubit() for _ in range({num_data}))",
            "        ax_ctrl = qubit()",
            "        az_ctrl = qubit()",
            "        ax_tgt = qubit()",
            "        az_tgt = qubit()",
            "",
            f"        ctrl = {struct_name}(ctrl_data)",
            f"        tgt = {struct_name}(tgt_data)",
            "",
            "        # Initialize both patches in |0_L>",
            "        init_syn_ctrl = init_z_basis(ctrl, ax_ctrl)",
            "        init_syn_tgt = init_z_basis(tgt, ax_tgt)",
            '        result("init_synx_ctrl", init_syn_ctrl)',
            '        result("init_synx_tgt", init_syn_tgt)',
            "",
            "        # Apply logical X to control to get |1_L>",
            "        apply_logical_x(ctrl)",
            "",
            "        # Apply transversal CNOT",
            "        transversal_cnot(ctrl, tgt)",
            "",
            "        # Syndrome extraction rounds",
            "        for _t in range(comptime(num_rounds)):",
            "            syn_ctrl = syndrome_extraction(ctrl, ax_ctrl, az_ctrl)",
            "            syn_tgt = syndrome_extraction(tgt, ax_tgt, az_tgt)",
            '            result("synx_ctrl", syn_ctrl.synx)',
            '            result("synz_ctrl", syn_ctrl.synz)',
            '            result("synx_tgt", syn_tgt.synx)',
            '            result("synz_tgt", syn_tgt.synz)',
            "",
            "        # Final measurement",
            "        final_ctrl = measure_z_basis(ctrl)",
            "        final_tgt = measure_z_basis(tgt)",
            '        result("final_ctrl", final_ctrl)',
            '        result("final_tgt", final_tgt)',
            "",
            "        discard(ax_ctrl)",
            "        discard(az_ctrl)",
            "        discard(ax_tgt)",
            "        discard(az_tgt)",
            "",
            "    return transversal_cnot_with_x_exp",
            "",
        ],
    )


def _add_color_transversal_factory_functions(
    lines: list[str],
    info: dict,
    num_data: int,
    _num_stab: int,
) -> None:
    """Add factory functions for color code transversal experiments."""
    struct_name = info["struct_name"]

    lines.extend(
        [
            "# === Transversal Experiments ===",
            "",
            "def make_transversal_cnot(num_rounds: int):",
            '    """Create transversal CNOT experiment: |0_L>|0_L> -> |0_L>|0_L>."""',
            "",
            "    @guppy",
            "    def transversal_cnot_exp() -> None:",
            '        """Transversal CNOT experiment."""',
            f"        ctrl_data = array(qubit() for _ in range({num_data}))",
            f"        tgt_data = array(qubit() for _ in range({num_data}))",
            "        ax_ctrl = qubit()",
            "        az_ctrl = qubit()",
            "        ax_tgt = qubit()",
            "        az_tgt = qubit()",
            "",
            f"        ctrl = {struct_name}(ctrl_data)",
            f"        tgt = {struct_name}(tgt_data)",
            "",
            "        # Initialize both patches in |0_L>",
            "        init_syn_ctrl = init_z_basis(ctrl, ax_ctrl)",
            "        init_syn_tgt = init_z_basis(tgt, ax_tgt)",
            '        result("init_synx_ctrl", init_syn_ctrl)',
            '        result("init_synx_tgt", init_syn_tgt)',
            "",
            "        # Apply transversal CNOT",
            "        transversal_cnot(ctrl, tgt)",
            "",
            "        # Syndrome extraction rounds",
            "        for _t in range(comptime(num_rounds)):",
            "            syn_ctrl = syndrome_extraction(ctrl, ax_ctrl, az_ctrl)",
            "            syn_tgt = syndrome_extraction(tgt, ax_tgt, az_tgt)",
            '            result("synx_ctrl", syn_ctrl.synx)',
            '            result("synz_ctrl", syn_ctrl.synz)',
            '            result("synx_tgt", syn_tgt.synx)',
            '            result("synz_tgt", syn_tgt.synz)',
            "",
            "        # Final measurement",
            "        final_ctrl = measure_z_basis(ctrl)",
            "        final_tgt = measure_z_basis(tgt)",
            '        result("final_ctrl", final_ctrl)',
            '        result("final_tgt", final_tgt)',
            "",
            "        discard(ax_ctrl)",
            "        discard(az_ctrl)",
            "        discard(ax_tgt)",
            "        discard(az_tgt)",
            "",
            "    return transversal_cnot_exp",
            "",
            "",
            "def make_transversal_cnot_with_x(num_rounds: int):",
            '    """Create transversal CNOT experiment: |1_L>|0_L> -> |1_L>|1_L>."""',
            "",
            "    @guppy",
            "    def transversal_cnot_with_x_exp() -> None:",
            '        """Transversal CNOT with X experiment."""',
            f"        ctrl_data = array(qubit() for _ in range({num_data}))",
            f"        tgt_data = array(qubit() for _ in range({num_data}))",
            "        ax_ctrl = qubit()",
            "        az_ctrl = qubit()",
            "        ax_tgt = qubit()",
            "        az_tgt = qubit()",
            "",
            f"        ctrl = {struct_name}(ctrl_data)",
            f"        tgt = {struct_name}(tgt_data)",
            "",
            "        # Initialize both patches in |0_L>",
            "        init_syn_ctrl = init_z_basis(ctrl, ax_ctrl)",
            "        init_syn_tgt = init_z_basis(tgt, ax_tgt)",
            '        result("init_synx_ctrl", init_syn_ctrl)',
            '        result("init_synx_tgt", init_syn_tgt)',
            "",
            "        # Apply logical X to control to get |1_L>",
            "        apply_logical_x(ctrl)",
            "",
            "        # Apply transversal CNOT",
            "        transversal_cnot(ctrl, tgt)",
            "",
            "        # Syndrome extraction rounds",
            "        for _t in range(comptime(num_rounds)):",
            "            syn_ctrl = syndrome_extraction(ctrl, ax_ctrl, az_ctrl)",
            "            syn_tgt = syndrome_extraction(tgt, ax_tgt, az_tgt)",
            '            result("synx_ctrl", syn_ctrl.synx)',
            '            result("synz_ctrl", syn_ctrl.synz)',
            '            result("synx_tgt", syn_tgt.synx)',
            '            result("synz_tgt", syn_tgt.synz)',
            "",
            "        # Final measurement",
            "        final_ctrl = measure_z_basis(ctrl)",
            "        final_tgt = measure_z_basis(tgt)",
            '        result("final_ctrl", final_ctrl)',
            '        result("final_tgt", final_tgt)',
            "",
            "        discard(ax_ctrl)",
            "        discard(az_ctrl)",
            "        discard(ax_tgt)",
            "        discard(az_tgt)",
            "",
            "    return transversal_cnot_with_x_exp",
            "",
        ],
    )


def _load_css_transversal_module(code_type: CSSCodeType, d: int) -> dict:
    """Load a transversal module for the given code type and distance."""
    cache_key = f"{code_type.value}_d{d}"

    if cache_key in _state.css_transversal_cache:
        return _state.css_transversal_cache[cache_key]

    # Generate source based on code type
    if code_type == CSSCodeType.SURFACE:
        source = generate_surface_transversal_source(d)
    elif code_type == CSSCodeType.COLOR:
        source = generate_color_transversal_source(d)
    else:
        msg = f"Unsupported code type: {code_type}"
        raise ValueError(msg)

    # Write to temp file
    temp_dir = _get_temp_dir()
    temp_file = temp_dir / f"css_trans_{cache_key}.py"
    temp_file.write_text(source)

    # Load module
    module_name = f"pecos._generated.css_trans_{cache_key}"
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    if spec is None or spec.loader is None:
        msg = f"Failed to create module spec for {temp_file}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    _state.css_transversal_cache[cache_key] = vars(module)
    return _state.css_transversal_cache[cache_key]


# === Public API ===


def make_css_transversal_cnot(
    code_type: CSSCodeType | str,
    distance: int,
    num_rounds: int = 1,
) -> object:
    """Create a transversal CNOT experiment for any CSS code.

    Args:
        code_type: Type of CSS code ("surface" or "color")
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds

    Returns:
        Guppy function for the experiment

    Example:
        >>> prog = make_css_transversal_cnot("color", distance=3, num_rounds=1)
        >>> result = prog.emulator(n_qubits=18).stabilizer_sim().run()
    """
    if isinstance(code_type, str):
        code_type = CSSCodeType(code_type)

    if distance < 3 or distance % 2 == 0:
        msg = f"Distance must be odd >= 3, got {distance}"
        raise ValueError(msg)

    module = _load_css_transversal_module(code_type, distance)
    return module["make_transversal_cnot"](num_rounds)


def make_css_transversal_cnot_with_x(
    code_type: CSSCodeType | str,
    distance: int,
    num_rounds: int = 1,
) -> object:
    """Create a transversal CNOT experiment with logical X on control.

    This tests |1_L>|0_L> -> |1_L>|1_L>.

    Args:
        code_type: Type of CSS code ("surface" or "color")
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds

    Returns:
        Guppy function for the experiment
    """
    if isinstance(code_type, str):
        code_type = CSSCodeType(code_type)

    if distance < 3 or distance % 2 == 0:
        msg = f"Distance must be odd >= 3, got {distance}"
        raise ValueError(msg)

    module = _load_css_transversal_module(code_type, distance)
    return module["make_transversal_cnot_with_x"](num_rounds)


def get_transversal_num_qubits(code_type: CSSCodeType | str, distance: int) -> int:
    """Get total qubit count for transversal CNOT between two patches.

    Args:
        code_type: Type of CSS code
        distance: Code distance

    Returns:
        Total qubits needed (2 * num_data + 4 ancillas)
    """
    if isinstance(code_type, str):
        code_type = CSSCodeType(code_type)

    if code_type == CSSCodeType.SURFACE:
        num_data = distance * distance
    elif code_type == CSSCodeType.COLOR:
        from pecos.qec.color import ColorCode488

        code = ColorCode488.create(distance=distance)
        num_data = code.num_data
    else:
        msg = f"Unsupported code type: {code_type}"
        raise ValueError(msg)

    # Two patches + 4 ancillas (ax, az for each patch)
    return 2 * num_data + 4


# === Convenience functions for specific codes ===


def make_color_transversal_cnot(distance: int, num_rounds: int = 1) -> object:
    """Create transversal CNOT for color codes."""
    return make_css_transversal_cnot(CSSCodeType.COLOR, distance, num_rounds)


def make_color_transversal_cnot_with_x(distance: int, num_rounds: int = 1) -> object:
    """Create transversal CNOT with X for color codes."""
    return make_css_transversal_cnot_with_x(CSSCodeType.COLOR, distance, num_rounds)


def make_color_transversal_cnot_d3(num_rounds: int = 1) -> object:
    """Create d=3 transversal CNOT for color codes."""
    return make_color_transversal_cnot(3, num_rounds)


def make_color_transversal_cnot_with_x_d3(num_rounds: int = 1) -> object:
    """Create d=3 transversal CNOT with X for color codes."""
    return make_color_transversal_cnot_with_x(3, num_rounds)


def make_surface_transversal_cnot(distance: int, num_rounds: int = 1) -> object:
    """Create transversal CNOT for surface codes."""
    return make_css_transversal_cnot(CSSCodeType.SURFACE, distance, num_rounds)


def make_surface_transversal_cnot_with_x(distance: int, num_rounds: int = 1) -> object:
    """Create transversal CNOT with X for surface codes."""
    return make_css_transversal_cnot_with_x(CSSCodeType.SURFACE, distance, num_rounds)
