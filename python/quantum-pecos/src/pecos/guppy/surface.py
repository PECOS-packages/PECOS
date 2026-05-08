# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generate Guppy code from SurfacePatch geometry.

This module generates Guppy quantum code from the geometry stored
in a SurfacePatch. The geometry is computed once and stored, then
used to generate code on demand.

The generated syndrome extraction uses a 4-round parallel CNOT
schedule (N/Z windmill pattern) with dedicated per-stabilizer ancillas.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pecos.qec.surface.schedule import compute_cnot_schedule

if TYPE_CHECKING:
    from pecos.qec.surface import SurfacePatch


# Module state container (avoids global statement)
class _ModuleState:
    """Container for module-level mutable state."""

    temp_dir: ClassVar[Path | None] = None
    module_cache: ClassVar[dict[str, object]] = {}
    distance_module_cache: ClassVar[dict[int, dict]] = {}


_state = _ModuleState()


def _get_temp_dir() -> Path:
    """Get or create temporary directory for generated code."""
    if _state.temp_dir is None:
        _state.temp_dir = Path(tempfile.mkdtemp(prefix="pecos_guppy_"))
    return _state.temp_dir


def generate_guppy_source(patch: "SurfacePatch") -> str:
    """Generate Guppy source code for a surface code patch.

    Uses a 4-round parallel CNOT schedule with dedicated per-stabilizer
    ancillas for syndrome extraction.

    Args:
        patch: SurfacePatch with geometry configuration

    Returns:
        Python/Guppy source code as a string
    """
    geom = patch.geometry
    num_data = geom.num_data
    num_x_stab = len(geom.x_stabilizers)
    num_z_stab = len(geom.z_stabilizers)
    dx, dz = geom.dx, geom.dz

    lines = [
        f'"""Surface code patch (dx={dx}, dz={dz}) implementation in Guppy.',
        "",
        "Auto-generated from SurfacePatch geometry.",
        "",
        f"Data qubits: {num_data}",
        f"X stabilizers: {num_x_stab}",
        f"Z stabilizers: {num_z_stab}",
        f"Ancilla qubits: {num_x_stab + num_z_stab} (one per stabilizer)",
        '"""',
        "",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, owned, result",
        "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x",
        "",
        "",
    ]

    # Generate struct definitions
    lines.extend(
        [
            "@guppy.struct",
            f"class SurfaceCode_{dx}x{dz}:",
            f'    """Surface code patch with dx={dx}, dz={dz} ({num_data} data qubits)."""',
            "",
            f"    data: array[qubit, {num_data}]",
            "",
            "",
            "@guppy.struct",
            f"class Syndrome_{dx}x{dz}:",
            f'    """Syndrome for dx={dx}, dz={dz} patch."""',
            "",
            f"    synx: array[bool, {num_x_stab}]",
            f"    synz: array[bool, {num_z_stab}]",
            "",
            "",
        ],
    )

    # Generate state preparation functions
    lines.extend(
        [
            "# === State Preparation ===",
            "",
            "@guppy",
            f"def prep_z_basis() -> SurfaceCode_{dx}x{dz}:",
            '    """Prepare logical |0_L> state."""',
            f"    data = array(qubit() for _ in range({num_data}))",
            f"    return SurfaceCode_{dx}x{dz}(data)",
            "",
            "",
            "@guppy",
            f"def prep_x_basis() -> SurfaceCode_{dx}x{dz}:",
            '    """Prepare logical |+_L> state."""',
            f"    data = array(qubit() for _ in range({num_data}))",
            f"    for i in range({num_data}):",
            "        h(data[i])",
            f"    return SurfaceCode_{dx}x{dz}(data)",
            "",
            "",
        ],
    )

    # Generate syndrome extraction with parallel CNOT schedule
    rounds = compute_cnot_schedule(patch)

    lines.extend(
        [
            "# === Syndrome Extraction ===",
            "",
            "@guppy",
            f"def syndrome_extraction(surf: SurfaceCode_{dx}x{dz}) -> Syndrome_{dx}x{dz}:",
            '    """Extract full syndrome using 4-round parallel CNOT schedule."""',
            "    # Allocate ancilla qubits (one per stabilizer)",
        ],
    )

    lines.extend(f"    ax{stab.index} = qubit()" for stab in geom.x_stabilizers)
    lines.extend(f"    az{stab.index} = qubit()" for stab in geom.z_stabilizers)

    lines.append("")
    lines.append("    # Hadamard on X ancillas")
    lines.extend(f"    h(ax{stab.index})" for stab in geom.x_stabilizers)

    # Emit 4 rounds of CX gates
    for rnd_idx, rnd_gates in enumerate(rounds):
        lines.append("")
        lines.append(f"    # Round {rnd_idx + 1}")
        for stab_type, stab_idx, data_q in rnd_gates:
            if stab_type == "X":
                lines.append(f"    cx(ax{stab_idx}, surf.data[{data_q}])")
            else:
                lines.append(f"    cx(surf.data[{data_q}], az{stab_idx})")

    lines.append("")
    lines.append("    # Hadamard on X ancillas")
    lines.extend(f"    h(ax{stab.index})" for stab in geom.x_stabilizers)

    # Measure ancillas (destructive)
    # Each measurement gets a per-measurement result() call that ties the
    # physical measurement to a MeasId. The result() names encode the
    # stabilizer type and index. The AllocateResult IDs generated by
    # these calls flow through the trace and become MeasIds on the TickCircuit.
    lines.append("")
    # Measure ancillas with per-measurement result() identity.
    # Tag format: "label:idx" where label is the stabilizer name and idx is the
    # round-local measurement index. The global MeasId is assigned by the runtime
    # via AllocateResult and flows through the trace automatically.
    lines.append("    # Measure ancillas")
    idx = 0
    for stab in geom.x_stabilizers:
        lines.append(f"    sx{stab.index} = measure(ax{stab.index})")
        lines.append(f'    result("sx{stab.index}:meas:{idx}", sx{stab.index})')
        idx += 1
    for stab in geom.z_stabilizers:
        lines.append(f"    sz{stab.index} = measure(az{stab.index})")
        lines.append(f'    result("sz{stab.index}:meas:{idx}", sz{stab.index})')
        idx += 1

    x_calls = ", ".join(f"sx{s.index}" for s in geom.x_stabilizers)
    z_calls = ", ".join(f"sz{s.index}" for s in geom.z_stabilizers)

    lines.extend(
        [
            "",
            f"    synx = array({x_calls})",
            f"    synz = array({z_calls})",
            "",
            f"    return Syndrome_{dx}x{dz}(synx, synz)",
            "",
            "",
        ],
    )

    # Generate measurement
    lines.extend(
        [
            "# === Measurement ===",
            "",
            "@guppy",
            f"def measure_z_basis(surf: SurfaceCode_{dx}x{dz} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in Z basis."""',
            "    return measure_array(surf.data)",
            "",
            "",
            "@guppy",
            f"def measure_x_basis(surf: SurfaceCode_{dx}x{dz} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in X basis."""',
            f"    for i in range({num_data}):",
            "        h(surf.data[i])",
            "    return measure_array(surf.data)",
            "",
            "",
        ],
    )

    # Generate logical operators
    logical_x_qubits = list(geom.logical_x.data_qubits) if geom.logical_x else []
    logical_z_qubits = list(geom.logical_z.data_qubits) if geom.logical_z else []

    lines.extend(
        [
            "# === Logical Operators ===",
            "",
            "@guppy",
            f"def apply_logical_x(surf: SurfaceCode_{dx}x{dz}) -> None:",
            '    """Apply logical X (string along left edge)."""',
        ],
    )
    lines.extend(f"    x(surf.data[{q}])" for q in logical_x_qubits)

    lines.extend(
        [
            "",
            "",
            "@guppy",
            f"def apply_logical_z(surf: SurfaceCode_{dx}x{dz}) -> None:",
            '    """Apply logical Z (string along top edge)."""',
            "    from guppylang.std.quantum import z",
            "",
        ],
    )
    lines.extend(f"    z(surf.data[{q}])" for q in logical_z_qubits)

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
            f'        """Z-basis memory experiment for dx={dx}, dz={dz}."""',
            "        surf = prep_z_basis()",
            "",
            "        for _t in range(comptime(num_rounds)):",
            "            syn = syndrome_extraction(surf)",
            '            result("synx", syn.synx)',
            '            result("synz", syn.synz)',
            "",
            "        final = measure_z_basis(surf)",
            '        result("final", final)',
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
            f'        """X-basis memory experiment for dx={dx}, dz={dz}."""',
            "        surf = prep_x_basis()",
            "",
            "        for _t in range(comptime(num_rounds)):",
            "            syn = syndrome_extraction(surf)",
            '            result("synx", syn.synx)',
            '            result("synz", syn.synz)',
            "",
            "        final = measure_x_basis(surf)",
            '        result("final", final)',
            "",
            "    return memory_x",
            "",
        ],
    )

    return "\n".join(lines)


def _load_guppy_module(patch: "SurfacePatch") -> dict:
    """Load a Guppy module for a patch, using caching.

    Args:
        patch: SurfacePatch with geometry

    Returns:
        Module dictionary with generated functions
    """
    cache_key = f"{patch.dx}x{patch.dz}"

    if cache_key in _state.module_cache:
        return _state.module_cache[cache_key]

    # Generate source
    source = generate_guppy_source(patch)

    # Write to temp file (required for Guppy introspection)
    temp_dir = _get_temp_dir()
    temp_file = temp_dir / f"patch_{cache_key}.py"
    temp_file.write_text(source)

    # Load module
    module_name = f"pecos._generated.patch_{cache_key}"
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    if spec is None or spec.loader is None:
        msg = f"Failed to create module spec for {temp_file}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    _state.module_cache[cache_key] = vars(module)
    return _state.module_cache[cache_key]


def generate_memory_experiment(
    patch: "SurfacePatch",
    num_rounds: int,
    basis: str,
) -> object:
    """Generate a memory experiment for a patch.

    Args:
        patch: SurfacePatch configuration
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'

    Returns:
        Guppy function for the experiment
    """
    module = _load_guppy_module(patch)

    if basis.upper() == "Z":
        factory = module["make_memory_z"]
    elif basis.upper() == "X":
        factory = module["make_memory_x"]
    else:
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    return factory(num_rounds)


def get_num_qubits(d: int) -> int:
    """Get total number of qubits for a distance-d surface code.

    Peak qubit count: d^2 data qubits + (d^2 - 1) ancilla qubits.

    Args:
        d: Code distance

    Returns:
        Total qubits (2 * d^2 - 1)
    """
    return 2 * d * d - 1


def generate_surface_code_module(d: int) -> str:
    """Generate source code for a distance-d surface code module.

    Args:
        d: Code distance (must be odd >= 3)

    Returns:
        Python/Guppy source code as a string
    """
    if d < 3 or d % 2 == 0:
        msg = f"Distance must be odd >= 3, got {d}"
        raise ValueError(msg)

    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    return generate_guppy_source(patch)


def get_surface_code_module(d: int) -> dict:
    """Get a loaded surface code module for distance d.

    Args:
        d: Code distance

    Returns:
        Dictionary with module contents and metadata
    """
    if d in _state.distance_module_cache:
        return _state.distance_module_cache[d]

    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    module = _load_guppy_module(patch)

    # Add metadata
    module["distance"] = d
    module["num_data"] = d * d
    module["num_stab"] = (d * d - 1) // 2

    _state.distance_module_cache[d] = module
    return module


def make_surface_code(distance: int, num_rounds: int, basis: str) -> object:
    """Create a surface code memory experiment.

    Args:
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' or 'X'

    Returns:
        Compiled Guppy program
    """
    if basis.upper() not in ("Z", "X"):
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    module = get_surface_code_module(distance)

    factory = module["make_memory_z"] if basis.upper() == "Z" else module["make_memory_x"]

    return factory(num_rounds)
