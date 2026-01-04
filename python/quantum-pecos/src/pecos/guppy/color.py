# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generate Guppy code for 4.8.8 triangular color codes.

The color code is a CSS code where each stabilizer measures both X and Z
on the same qubit support. Stabilizers are colored red, green, and blue.
"""

from pathlib import Path
import importlib.util
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.color import ColorCode488

# Cache for generated modules
_module_cache: dict[int, dict] = {}
_temp_dir: Path | None = None


def _get_temp_dir() -> Path:
    """Get or create temporary directory for generated code."""
    global _temp_dir
    if _temp_dir is None:
        _temp_dir = Path(tempfile.mkdtemp(prefix="pecos_guppy_color_"))
    return _temp_dir


def generate_color_code_source(code: "ColorCode488") -> str:
    """Generate Guppy source code for a color code.

    Args:
        code: ColorCode488 instance with geometry

    Returns:
        Python/Guppy source code as a string
    """
    d = code.distance
    n_data = code.n_data
    n_stab = code.n_stabilizers

    lines = [
        f'"""4.8.8 Color Code (d={d}) implementation in Guppy.',
        "",
        "Auto-generated from ColorCode488 geometry.",
        "",
        f"Data qubits: {n_data}",
        f"Stabilizers: {n_stab}",
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
    lines.extend([
        "@guppy.struct",
        f"class ColorCode_{d}:",
        f'    """Color code with d={d} ({n_data} data qubits)."""',
        "",
        f"    data: array[qubit, {n_data}]",
        "",
        "",
        "@guppy.struct",
        f"class ColorSyndrome_{d}:",
        f'    """Syndrome for d={d} color code."""',
        "",
        f"    synx: array[bool, {n_stab}]",
        f"    synz: array[bool, {n_stab}]",
        "",
        "",
    ])

    # Generate X stabilizer measurement functions (H-CNOT-H pattern)
    lines.append("# === X Stabilizer Measurements ===")
    lines.append("")

    for stab in code.stabilizers:
        weight = f"w{stab.weight}, {stab.color}"
        lines.extend([
            "@guppy",
            f"def measure_x_stab_{stab.index}(ax: qubit, data: array[qubit, {n_data}]) -> bool:",
            f'    """Measure X stabilizer {stab.index} ({weight})."""',
            "    h(ax)",
        ])
        for q in stab.qubits:
            lines.append(f"    cx(ax, data[{q}])")
        lines.extend([
            "    h(ax)",
            "    return measure_and_reset(ax)",
            "",
            "",
        ])

    # Generate Z stabilizer measurement functions (CNOT pattern)
    lines.append("# === Z Stabilizer Measurements ===")
    lines.append("")

    for stab in code.stabilizers:
        weight = f"w{stab.weight}, {stab.color}"
        lines.extend([
            "@guppy",
            f"def measure_z_stab_{stab.index}(az: qubit, data: array[qubit, {n_data}]) -> bool:",
            f'    """Measure Z stabilizer {stab.index} ({weight})."""',
        ])
        for q in stab.qubits:
            lines.append(f"    cx(data[{q}], az)")
        lines.extend([
            "    return measure_and_reset(az)",
            "",
            "",
        ])

    # Generate syndrome extraction
    x_calls = ", ".join(f"sx{s.index}" for s in code.stabilizers)
    z_calls = ", ".join(f"sz{s.index}" for s in code.stabilizers)

    lines.extend([
        "# === Syndrome Extraction ===",
        "",
        "@guppy",
        f"def syndrome_extraction(",
        f"    code: ColorCode_{d},",
        "    ax: qubit,",
        "    az: qubit,",
        f") -> ColorSyndrome_{d}:",
        '    """Extract full X and Z syndrome."""',
        "    # Z stabilizers first",
    ])

    for stab in code.stabilizers:
        lines.append(f"    sz{stab.index} = measure_z_stab_{stab.index}(az, code.data)")

    lines.append("")
    lines.append("    # X stabilizers")

    for stab in code.stabilizers:
        lines.append(f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, code.data)")

    lines.extend([
        "",
        f"    synx = array({x_calls})",
        f"    synz = array({z_calls})",
        "",
        f"    return ColorSyndrome_{d}(synx, synz)",
        "",
        "",
    ])

    # Generate initialization functions
    lines.extend([
        "# === Initialization ===",
        "",
        "@guppy",
        f"def init_z_basis(code: ColorCode_{d}, ax: qubit) -> array[bool, {n_stab}]:",
        '    """Initialize logical |0_L> and extract initial X syndrome."""',
        "    # Qubits start in |0>, which is already a +1 eigenstate of Z stabilizers",
        "    # Measure X stabilizers to project into code space",
    ])

    for stab in code.stabilizers:
        lines.append(f"    sx{stab.index} = measure_x_stab_{stab.index}(ax, code.data)")

    lines.extend([
        "",
        f"    return array({x_calls})",
        "",
        "",
        "@guppy",
        f"def init_x_basis(code: ColorCode_{d}, az: qubit) -> array[bool, {n_stab}]:",
        '    """Initialize logical |+_L> and extract initial Z syndrome."""',
        f"    for i in range({n_data}):",
        "        h(code.data[i])",
        "",
        "    # Measure Z stabilizers to project into code space",
    ])

    for stab in code.stabilizers:
        lines.append(f"    sz{stab.index} = measure_z_stab_{stab.index}(az, code.data)")

    lines.extend([
        "",
        f"    return array({z_calls})",
        "",
        "",
    ])

    # Generate measurement functions
    lines.extend([
        "# === Measurement ===",
        "",
        "@guppy",
        f"def measure_z_basis(code: ColorCode_{d} @ owned) -> array[bool, {n_data}]:",
        '    """Destructively measure in Z basis."""',
        "    return measure_array(code.data)",
        "",
        "",
        "@guppy",
        f"def measure_x_basis(code: ColorCode_{d} @ owned) -> array[bool, {n_data}]:",
        '    """Destructively measure in X basis."""',
        f"    for i in range({n_data}):",
        "        h(code.data[i])",
        "    return measure_array(code.data)",
        "",
        "",
    ])

    # Generate logical operators
    logical_x_qubits = list(code.get_logical_x())
    logical_z_qubits = list(code.get_logical_z())

    lines.extend([
        "# === Logical Operators ===",
        "",
        "@guppy",
        f"def apply_logical_x(code: ColorCode_{d}) -> None:",
        '    """Apply logical X operator."""',
    ])
    for q in logical_x_qubits:
        lines.append(f"    x(code.data[{q}])")

    lines.extend([
        "",
        "",
        "@guppy",
        f"def apply_logical_z(code: ColorCode_{d}) -> None:",
        '    """Apply logical Z operator."""',
        "    from guppylang.std.quantum import z",
        "",
    ])
    for q in logical_z_qubits:
        lines.append(f"    z(code.data[{q}])")

    lines.extend([
        "",
        "",
    ])

    # Generate memory experiment factories
    lines.extend([
        "# === Memory Experiments ===",
        "",
        "def make_memory_z(num_rounds: int):",
        '    """Create Z-basis memory experiment."""',
        "    from guppylang.std.builtins import comptime",
        "",
        "    @guppy",
        "    def memory_z() -> None:",
        f'        """Z-basis memory experiment for d={d} color code."""',
        f"        data = array(qubit() for _ in range({n_data}))",
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
        f"        data = array(qubit() for _ in range({n_data}))",
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
    ])

    return "\n".join(lines)


def _load_color_code_module(d: int) -> dict:
    """Load a color code module for distance d, using caching.

    Args:
        d: Code distance

    Returns:
        Module dictionary with generated functions
    """
    if d in _module_cache:
        return _module_cache[d]

    from pecos.qec.color import ColorCode488

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
        raise RuntimeError(f"Failed to create module spec for {temp_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    _module_cache[d] = vars(module)
    return _module_cache[d]


def get_color_code_module(d: int) -> dict:
    """Get a loaded color code module for distance d.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        Dictionary with module contents and metadata
    """
    from pecos.qec.color import ColorCode488

    module = _load_color_code_module(d)

    # Add metadata if not present
    if "distance" not in module:
        code = ColorCode488.create(distance=d)
        module["distance"] = d
        module["n_data"] = code.n_data
        module["n_stab"] = code.n_stabilizers

    return module


def get_n_qubits_color(d: int) -> int:
    """Get total number of qubits for a distance-d color code.

    Args:
        d: Code distance

    Returns:
        Total qubits (n_data + 2 ancilla)
    """
    from pecos.qec.color import ColorCode488

    code = ColorCode488.create(distance=d)
    return code.n_data + 2


def make_color_code(distance: int, num_rounds: int, basis: str):
    """Create a color code memory experiment.

    Args:
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' or 'X'

    Returns:
        Guppy function for the experiment
    """
    if basis.upper() not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X', got {basis!r}")

    module = get_color_code_module(distance)

    if basis.upper() == "Z":
        factory = module["make_memory_z"]
    else:
        factory = module["make_memory_x"]

    return factory(num_rounds)


def generate_color_code_module(d: int) -> str:
    """Generate source code for a distance-d color code module.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        Python/Guppy source code as a string
    """
    if d < 3 or d % 2 == 0:
        raise ValueError(f"Distance must be odd >= 3, got {d}")

    from pecos.qec.color import ColorCode488

    code = ColorCode488.create(distance=d)
    return generate_color_code_source(code)
