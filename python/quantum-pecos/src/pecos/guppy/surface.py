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
    distance_module_cache: ClassVar[dict[tuple[int, int], dict]] = {}


_state = _ModuleState()


def _get_temp_dir() -> Path:
    """Get or create temporary directory for generated code."""
    if _state.temp_dir is None:
        _state.temp_dir = Path(tempfile.mkdtemp(prefix="pecos_guppy_"))
    return _state.temp_dir


def generate_guppy_source(
    patch: "SurfacePatch",
    *,
    ancilla_budget: int | None = None,
) -> str:
    """Generate Guppy source code for a surface code patch.

    Uses a 4-round parallel CNOT schedule for syndrome extraction.

    ``ancilla_budget=None`` (default) emits the unconstrained shape:
    one ancilla per stabilizer, all measured in parallel at the end of
    one round. This matches the abstract circuit's unconstrained-path
    measurement order (X stabilizers first by index, then Z).

    A finite ``ancilla_budget`` emits a stabilizer-batched syndrome-
    extraction routine that mirrors the abstract circuit's
    ``_batched_stabilizers`` schedule (shared helper at
    ``pecos.qec.surface._ancilla_batching``): per batch, allocate
    ``min(ancilla_budget, total_ancilla)`` fresh ancillas, run the
    4-round CX schedule restricted to that batch's stabilizers,
    measure, then move to the next batch (which allocates fresh
    qubits whose physical slots are reused by Selene's lowering).
    The same per-stabilizer ``result("...:meas:N", …)`` calls fire
    in the abstract's batched measurement order, keeping
    detector record offsets transferable between abstract and traced
    paths.

    Args:
        patch: SurfacePatch with geometry configuration.
        ancilla_budget: Optional cap on simultaneously live ancillas.
            ``None`` or a value ``>= total_ancilla`` emits the
            unconstrained shape; ``< total_ancilla`` emits batched.

    Returns:
        Python/Guppy source code as a string.
    """
    from pecos.qec.surface._ancilla_batching import batched_stabilizers, normalize_ancilla_budget

    geom = patch.geometry
    num_data = geom.num_data
    num_x_stab = len(geom.x_stabilizers)
    num_z_stab = len(geom.z_stabilizers)
    total_ancilla = num_x_stab + num_z_stab
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    constrained = effective_budget < total_ancilla
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

    # Generate syndrome extraction with parallel CNOT schedule.
    rounds = compute_cnot_schedule(patch)

    lines.extend(
        [
            "# === Syndrome Extraction ===",
            "",
            "@guppy",
            f"def syndrome_extraction(surf: SurfaceCode_{dx}x{dz}) -> Syndrome_{dx}x{dz}:",
        ],
    )

    if not constrained:
        # Unconstrained: one ancilla per stabilizer, X-stabs first then
        # Z-stabs, measured in parallel at the end. Matches the
        # abstract circuit's unconstrained-path measurement order.
        lines.extend(
            [
                '    """Extract full syndrome using 4-round parallel CNOT schedule."""',
                "    # Allocate ancilla qubits (one per stabilizer)",
            ],
        )

        lines.extend(f"    ax{stab.index} = qubit()" for stab in geom.x_stabilizers)
        lines.extend(f"    az{stab.index} = qubit()" for stab in geom.z_stabilizers)

        lines.append("")
        lines.append("    # Hadamard on X ancillas")
        lines.extend(f"    h(ax{stab.index})" for stab in geom.x_stabilizers)

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

        lines.append("")
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
    else:
        # Constrained: stabilizer-batched. The batch sequence is the
        # shared `batched_stabilizers(patch, effective_budget)` so the
        # abstract circuit's measurement order matches by construction.
        batches = batched_stabilizers(patch, effective_budget)
        lines.append(
            f'    """Extract full syndrome in {len(batches)} ancilla-reuse batches (budget={effective_budget})."""',
        )
        idx = 0
        for batch_idx, batch in enumerate(batches):
            lines.append("")
            lines.append(f"    # Batch {batch_idx + 1}/{len(batches)} of stabilizers")

            # Per-batch ancilla variable names: _a_b{batch}_p{pos}. Each
            # `qubit()` call here allocates a fresh logical qubit that
            # Selene's lowering reuses the physical slot freed by the
            # previous batch's `measure()` calls (empirically verified
            # in the spike).
            batch_anc_var: dict[tuple[str, int], str] = {}
            for pos, (stab_type, stab_idx) in enumerate(batch):
                var = f"_a_b{batch_idx}_p{pos}"
                batch_anc_var[(stab_type, stab_idx)] = var
                lines.append(f"    {var} = qubit()")

            x_in_batch = [(t, i) for (t, i) in batch if t == "X"]
            if x_in_batch:
                lines.append("    # Hadamard on X ancillas in this batch")
                for stab_type, stab_idx in x_in_batch:
                    lines.append(f"    h({batch_anc_var[(stab_type, stab_idx)]})")

            # Filter the full CX schedule to just this batch's stabilizers.
            batch_keys = set(batch_anc_var.keys())
            for rnd_idx, rnd_gates in enumerate(rounds):
                rnd_in_batch = [
                    (stab_type, stab_idx, data_q)
                    for stab_type, stab_idx, data_q in rnd_gates
                    if (stab_type, stab_idx) in batch_keys
                ]
                if not rnd_in_batch:
                    continue
                lines.append("")
                lines.append(f"    # Batch {batch_idx + 1} round {rnd_idx + 1}")
                for stab_type, stab_idx, data_q in rnd_in_batch:
                    anc = batch_anc_var[(stab_type, stab_idx)]
                    if stab_type == "X":
                        lines.append(f"    cx({anc}, surf.data[{data_q}])")
                    else:
                        lines.append(f"    cx(surf.data[{data_q}], {anc})")

            if x_in_batch:
                lines.append("")
                lines.append("    # Hadamard on X ancillas in this batch")
                for stab_type, stab_idx in x_in_batch:
                    lines.append(f"    h({batch_anc_var[(stab_type, stab_idx)]})")

            lines.append("")
            lines.append(f"    # Measure batch {batch_idx + 1} ancillas")
            for stab_type, stab_idx in batch:
                anc = batch_anc_var[(stab_type, stab_idx)]
                syn_var = f"sx{stab_idx}" if stab_type == "X" else f"sz{stab_idx}"
                tag_prefix = syn_var
                lines.append(f"    {syn_var} = measure({anc})")
                lines.append(f'    result("{tag_prefix}:meas:{idx}", {syn_var})')
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


def _load_guppy_module(
    patch: "SurfacePatch",
    *,
    ancilla_budget: int | None = None,
) -> dict:
    """Load a Guppy module for a patch, using caching.

    The cache key is widened by the **effective** budget (after
    clamping via ``normalize_ancilla_budget``), so ``ancilla_budget=None``
    and ``ancilla_budget >= total_ancilla`` resolve to the same cache
    entry and don't produce two equivalent generated modules.

    Args:
        patch: SurfacePatch with geometry
        ancilla_budget: Optional cap on simultaneously live ancillas

    Returns:
        Module dictionary with generated functions
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    cache_key = f"{patch.dx}x{patch.dz}_b{effective_budget}"

    if cache_key in _state.module_cache:
        return _state.module_cache[cache_key]

    # Generate source for this (patch, effective_budget) combination.
    source = generate_guppy_source(patch, ancilla_budget=ancilla_budget)

    # Write to temp file (required for Guppy introspection).
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
    *,
    ancilla_budget: int | None = None,
) -> object:
    """Generate a memory experiment for a patch.

    Args:
        patch: SurfacePatch configuration
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas

    Returns:
        Guppy function for the experiment
    """
    module = _load_guppy_module(patch, ancilla_budget=ancilla_budget)

    if basis.upper() == "Z":
        factory = module["make_memory_z"]
    elif basis.upper() == "X":
        factory = module["make_memory_x"]
    else:
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    return factory(num_rounds)


def get_num_qubits(d: int, *, ancilla_budget: int | None = None) -> int:
    """Get total number of qubits for a distance-d surface code.

    Unconstrained (``ancilla_budget=None``): peak qubit count is
    ``d^2 data + (d^2 - 1) ancilla = 2 * d^2 - 1``.

    Constrained (``ancilla_budget`` provided): the program reuses
    ancilla slots across stabilizer-measurement batches, so only
    ``d^2 data + min(ancilla_budget, d^2 - 1) ancilla`` physical
    slots are simultaneously live. The clamping is the same as
    ``pecos.qec.surface._ancilla_batching.normalize_ancilla_budget``
    so the unconstrained-via-``None`` and unconstrained-via-large-int
    cases collapse to the same value.

    Args:
        d: Code distance
        ancilla_budget: Optional cap on simultaneously live ancillas.
            ``None`` (default) returns the peak count.

    Returns:
        Total qubits the traced program will simultaneously use.
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    total_ancilla = d * d - 1
    effective = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    return d * d + effective


def generate_surface_code_module(d: int, *, ancilla_budget: int | None = None) -> str:
    """Generate source code for a distance-d surface code module.

    Args:
        d: Code distance (must be odd >= 3)
        ancilla_budget: Optional cap on simultaneously live ancillas;
            forwarded to ``generate_guppy_source``.

    Returns:
        Python/Guppy source code as a string
    """
    if d < 3 or d % 2 == 0:
        msg = f"Distance must be odd >= 3, got {d}"
        raise ValueError(msg)

    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    return generate_guppy_source(patch, ancilla_budget=ancilla_budget)


def get_surface_code_module(d: int, *, ancilla_budget: int | None = None) -> dict:
    """Get a loaded surface code module for distance d.

    Cache key is widened to ``(d, effective_budget)`` so the
    unconstrained-via-``None`` and unconstrained-via-large-int cases
    collapse to one cached module.

    Args:
        d: Code distance
        ancilla_budget: Optional cap on simultaneously live ancillas

    Returns:
        Dictionary with module contents and metadata
    """
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    total_ancilla = d * d - 1
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    cache_key = (d, effective_budget)

    if cache_key in _state.distance_module_cache:
        return _state.distance_module_cache[cache_key]

    patch = SurfacePatch.create(distance=d)
    module = _load_guppy_module(patch, ancilla_budget=ancilla_budget)

    # Add metadata
    module["distance"] = d
    module["num_data"] = d * d
    module["num_stab"] = (d * d - 1) // 2
    module["ancilla_budget"] = effective_budget

    _state.distance_module_cache[cache_key] = module
    return module


def make_surface_code(
    distance: int,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
) -> object:
    """Create a surface code memory experiment.

    Args:
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas.
            ``None`` (default) emits the unconstrained Guppy program;
            a finite budget emits a stabilizer-batched program that
            matches the abstract circuit's
            ``batched_stabilizers(patch, effective_budget)`` schedule.

    Returns:
        Compiled Guppy program
    """
    if basis.upper() not in ("Z", "X"):
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    module = get_surface_code_module(distance, ancilla_budget=ancilla_budget)

    factory = module["make_memory_z"] if basis.upper() == "Z" else module["make_memory_x"]

    return factory(num_rounds)
