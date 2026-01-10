# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generate Guppy code for 4.8.8 triangular color codes.

The color code is a CSS code where each stabilizer measures both X and Z
on the same qubit support. Stabilizers are colored red, green, and blue.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.color import ColorCode488


# Module state container (avoids global statement)
class _ModuleState:
    """Container for module-level mutable state."""

    temp_dir: Path | None = None
    module_cache: dict[int, dict] = {}  # noqa: RUF012


_state = _ModuleState()


def _get_temp_dir() -> Path:
    """Get or create temporary directory for generated code."""
    if _state.temp_dir is None:
        _state.temp_dir = Path(tempfile.mkdtemp(prefix="pecos_guppy_color_"))
    return _state.temp_dir


def generate_color_code_source(code: "ColorCode488") -> str:
    """Generate Guppy source code for a color code.

    Args:
        code: ColorCode488 instance with geometry

    Returns:
        Python/Guppy source code as a string
    """
    d = code.distance
    num_data = code.num_data
    num_stab = code.num_stabilizers

    lines = [
        f'"""4.8.8 Color Code (d={d}) implementation in Guppy.',
        "",
        "Auto-generated from ColorCode488 geometry.",
        "",
        f"Data qubits: {num_data}",
        f"Stabilizers: {num_stab}",
        '"""',
        "",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, result",
        "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x",
        "from guppylang.std.qsystem import measure_and_reset",
        "",
        "",
    ]

    # Generate struct definitions
    lines.extend(
        [
            "@guppy.struct",
            f"class ColorCode_{d}:",
            f'    """Color code with d={d} ({num_data} data qubits)."""',
            "",
            f"    data: array[qubit, {num_data}]",
            "",
            "",
            "@guppy.struct",
            f"class ColorSyndrome_{d}:",
            f'    """Syndrome for d={d} color code."""',
            "",
            f"    synx: array[bool, {num_stab}]",
            f"    synz: array[bool, {num_stab}]",
            "",
            "",
        ],
    )

    # Generate X stabilizer measurement functions (H-CNOT-H pattern)
    lines.append("# === X Stabilizer Measurements ===")
    lines.append("")

    for stab in code.stabilizers:
        weight = f"w{stab.weight}, {stab.color}"
        lines.extend(
            [
                "@guppy",
                f"def measure_x_stab_{stab.index}(ax: qubit, data: array[qubit, {num_data}]) -> bool:",
                f'    """Measure X stabilizer {stab.index} ({weight})."""',
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

    # Generate Z stabilizer measurement functions (CNOT pattern)
    lines.append("# === Z Stabilizer Measurements ===")
    lines.append("")

    for stab in code.stabilizers:
        weight = f"w{stab.weight}, {stab.color}"
        lines.extend(
            [
                "@guppy",
                f"def measure_z_stab_{stab.index}(az: qubit, data: array[qubit, {num_data}]) -> bool:",
                f'    """Measure Z stabilizer {stab.index} ({weight})."""',
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

    # Generate syndrome extraction
    x_calls = ", ".join(f"sx{s.index}" for s in code.stabilizers)
    z_calls = ", ".join(f"sz{s.index}" for s in code.stabilizers)

    lines.extend(
        [
            "# === Syndrome Extraction ===",
            "",
            "@guppy",
            "def syndrome_extraction(",
            f"    code: ColorCode_{d},",
            "    ax: qubit,",
            "    az: qubit,",
            f") -> ColorSyndrome_{d}:",
            '    """Extract full X and Z syndrome."""',
            "    # Z stabilizers first",
        ],
    )

    lines.extend(
        f"    sz{stab.index} = measure_z_stab_{stab.index}(az, code.data)"
        for stab in code.stabilizers
    )

    lines.append("")
    lines.append("    # X stabilizers")

    lines.extend(
        f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, code.data)"
        for stab in code.stabilizers
    )

    lines.extend(
        [
            "",
            f"    synx = array({x_calls})",
            f"    synz = array({z_calls})",
            "",
            f"    return ColorSyndrome_{d}(synx, synz)",
            "",
            "",
        ],
    )

    # Generate initialization functions
    lines.extend(
        [
            "# === Initialization ===",
            "",
            "@guppy",
            f"def init_z_basis(code: ColorCode_{d}, ax: qubit) -> array[bool, {num_stab}]:",
            '    """Initialize logical |0_L> and extract initial X syndrome."""',
            "    # Qubits start in |0>, which is already a +1 eigenstate of Z stabilizers",
            "    # Measure X stabilizers to project into code space",
        ],
    )

    lines.extend(
        f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, code.data)"
        for stab in code.stabilizers
    )

    lines.extend(
        [
            "",
            f"    return array({x_calls})",
            "",
            "",
            "@guppy",
            f"def init_x_basis(code: ColorCode_{d}, az: qubit) -> array[bool, {num_stab}]:",
            '    """Initialize logical |+_L> and extract initial Z syndrome."""',
            f"    for i in range({num_data}):",
            "        h(code.data[i])",
            "",
            "    # Measure Z stabilizers to project into code space",
        ],
    )

    lines.extend(
        f"    sz{stab.index} = measure_z_stab_{stab.index}(az, code.data)"
        for stab in code.stabilizers
    )

    lines.extend(
        [
            "",
            f"    return array({z_calls})",
            "",
            "",
        ],
    )

    # Generate measurement functions
    lines.extend(
        [
            "# === Measurement ===",
            "",
            "@guppy",
            f"def measure_z_basis(code: ColorCode_{d} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in Z basis."""',
            "    return measure_array(code.data)",
            "",
            "",
            "@guppy",
            f"def measure_x_basis(code: ColorCode_{d} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in X basis."""',
            f"    for i in range({num_data}):",
            "        h(code.data[i])",
            "    return measure_array(code.data)",
            "",
            "",
        ],
    )

    # Generate logical operators
    logical_x_qubits = list(code.get_logical_x())
    logical_z_qubits = list(code.get_logical_z())

    lines.extend(
        [
            "# === Logical Operators ===",
            "",
            "@guppy",
            f"def apply_logical_x(code: ColorCode_{d}) -> None:",
            '    """Apply logical X operator."""',
        ],
    )
    lines.extend(f"    x(code.data[{q}])" for q in logical_x_qubits)

    lines.extend(
        [
            "",
            "",
            "@guppy",
            f"def apply_logical_z(code: ColorCode_{d}) -> None:",
            '    """Apply logical Z operator."""',
            "    from guppylang.std.quantum import z",
            "",
        ],
    )
    lines.extend(f"    z(code.data[{q}])" for q in logical_z_qubits)

    lines.extend(
        [
            "",
            "",
        ],
    )

    # Generate memory experiment factories
    lines.extend(
        [
            "# === Memory Experiments ===",
            "",
            "def make_memory_z(num_rounds: int):",
            '    """Create Z-basis memory experiment."""',
            "    from guppylang.std.builtins import comptime",
            "",
            "    @guppy",
            "    def memory_z() -> None:",
            f'        """Z-basis memory experiment for d={d} color code."""',
            f"        data = array(qubit() for _ in range({num_data}))",
            "        ax = qubit()",
            "        az = qubit()",
            "",
            f"        code = ColorCode_{d}(data)",
            "",
            "        init_syn = init_z_basis(code, ax)",
            '        result("init_synx", init_syn)',
            "",
            "        for _t in range(comptime(num_rounds)):",
            "            syn = syndrome_extraction(code, ax, az)",
            '            result("synx", syn.synx)',
            '            result("synz", syn.synz)',
            "",
            "        final = measure_z_basis(code)",
            '        result("final", final)',
            "",
            "        discard(ax)",
            "        discard(az)",
            "",
            "    return memory_z",
            "",
            "",
            "def make_memory_x(num_rounds: int):",
            '    """Create X-basis memory experiment."""',
            "    from guppylang.std.builtins import comptime",
            "",
            "    @guppy",
            "    def memory_x() -> None:",
            f'        """X-basis memory experiment for d={d} color code."""',
            f"        data = array(qubit() for _ in range({num_data}))",
            "        ax = qubit()",
            "        az = qubit()",
            "",
            f"        code = ColorCode_{d}(data)",
            "",
            "        init_syn = init_x_basis(code, az)",
            '        result("init_synz", init_syn)',
            "",
            "        for _t in range(comptime(num_rounds)):",
            "            syn = syndrome_extraction(code, ax, az)",
            '            result("synx", syn.synx)',
            '            result("synz", syn.synz)',
            "",
            "        final = measure_x_basis(code)",
            '        result("final", final)',
            "",
            "        discard(ax)",
            "        discard(az)",
            "",
            "    return memory_x",
            "",
        ],
    )

    return "\n".join(lines)


def _load_color_code_module(d: int) -> dict:
    """Load a color code module for distance d, using caching.

    Args:
        d: Code distance

    Returns:
        Module dictionary with generated functions
    """
    if d in _state.module_cache:
        return _state.module_cache[d]

    from pecos.qec.color import ColorCode488  # noqa: PLC0415

    code = ColorCode488.create(distance=d)
    source = generate_color_code_source(code)

    # Write to temp file
    temp_dir = _get_temp_dir()
    temp_file = temp_dir / f"color_d{d}.py"
    temp_file.write_text(source)

    # Load module
    module_name = f"pecos._generated.color_d{d}"
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    if spec is None or spec.loader is None:
        msg = f"Failed to create module spec for {temp_file}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    _state.module_cache[d] = vars(module)
    return _state.module_cache[d]


def get_color_code_module(d: int) -> dict:
    """Get a loaded color code module for distance d.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        Dictionary with module contents and metadata
    """
    from pecos.qec.color import ColorCode488  # noqa: PLC0415

    module = _load_color_code_module(d)

    # Add metadata if not present
    if "distance" not in module:
        code = ColorCode488.create(distance=d)
        module["distance"] = d
        module["num_data"] = code.num_data
        module["num_stab"] = code.num_stabilizers

    return module


def get_num_qubits_color(d: int) -> int:
    """Get total number of qubits for a distance-d color code.

    Args:
        d: Code distance

    Returns:
        Total qubits (num_data + 2 ancilla)
    """
    from pecos.qec.color import ColorCode488  # noqa: PLC0415

    code = ColorCode488.create(distance=d)
    return code.num_data + 2


def make_color_code(distance: int, num_rounds: int, basis: str) -> object:
    """Create a color code memory experiment.

    Args:
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' or 'X'

    Returns:
        Guppy function for the experiment
    """
    if basis.upper() not in ("Z", "X"):
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    module = get_color_code_module(distance)

    factory = (
        module["make_memory_z"] if basis.upper() == "Z" else module["make_memory_x"]
    )

    return factory(num_rounds)


def generate_color_code_module(d: int) -> str:
    """Generate source code for a distance-d color code module.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        Python/Guppy source code as a string
    """
    if d < 3 or d % 2 == 0:
        msg = f"Distance must be odd >= 3, got {d}"
        raise ValueError(msg)

    from pecos.qec.color import ColorCode488  # noqa: PLC0415

    code = ColorCode488.create(distance=d)
    return generate_color_code_source(code)
