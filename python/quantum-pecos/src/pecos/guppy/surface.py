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
    from pecos.qec.surface import GuppyRngMaskConfig, SurfacePatch, TwirlConfig


# Module state container (avoids global statement)
class _ModuleState:
    """Container for module-level mutable state."""

    temp_dir: ClassVar[Path | None] = None
    module_cache: ClassVar[dict[str, object]] = {}
    # Keyed by full patch identity + effective budget (dx, dz, orientation,
    # rotated, effective_budget) so distinct patch geometries -- e.g. rotated
    # vs non-rotated at the same dx/dz -- never collide on a cached module.
    distance_module_cache: ClassVar[dict[tuple[int, int, str, bool, int], dict]] = {}


_state = _ModuleState()


def _get_temp_dir() -> Path:
    """Get or create temporary directory for generated code."""
    if _state.temp_dir is None:
        _state.temp_dir = Path(tempfile.mkdtemp(prefix="pecos_guppy_"))
    return _state.temp_dir


def _render_inline_pcg32() -> list[str]:
    """Render Guppy-local PCG32 helpers for runtime twirl masks.

    The user seed is a stream separator. Per-shot entropy comes from
    H/measure side-band qubits so the runtime twirl is not a fixed mask.
    """
    return [
        "@guppy",
        "@no_type_check",
        "def _pcg32_mask32(value: nat) -> nat:",
        "    uint32_mask: nat = 4294967295",
        "    return value & uint32_mask",
        "",
        "",
        "@guppy",
        "@no_type_check",
        "def _pcg32_advance(state: nat, inc: nat) -> nat:",
        "    pcg32_mult: nat = 6364136223846793005",
        "    return nat(state * pcg32_mult + inc)",
        "",
        "",
        "@guppy",
        "@no_type_check",
        "def _pcg32_next4(state: nat, inc: nat) -> tuple[nat, int]:",
        "    old_state = state",
        "    new_state = _pcg32_advance(state, inc)",
        "    xorshifted = _pcg32_mask32(((old_state >> nat(18)) ^ old_state) >> nat(27))",
        "    rot = _pcg32_mask32(old_state >> nat(59))",
        "    rot_inv = _pcg32_mask32((~rot + nat(1)) & nat(31))",
        "    output = _pcg32_mask32((xorshifted >> rot) | (xorshifted << rot_inv))",
        "    return new_state, int(output & nat(3))",
        "",
        "",
        "@guppy",
        "@no_type_check",
        "def seeded_pcg32_from_sequence(seed: int, sequence: nat) -> tuple[nat, nat]:",
        "    initstate = nat(42)",
        "    initseq = nat(seed) ^ sequence",
        "    inc = nat((initseq << nat(1)) | nat(1))",
        "    state = _pcg32_advance(nat(0), inc)",
        "    state += initstate",
        "    state = _pcg32_advance(state, inc)",
        "    return state, inc",
        "",
        "",
        "@guppy",
        "@no_type_check",
        "def seeded_pcg32_with_quantum_entropy(seed: int) -> tuple[nat, nat]:",
        "    entropy = nat(0)",
        "    for i in range(32):",
        "        entropy_q = qubit()",
        "        h(entropy_q)",
        "        if measure(entropy_q):",
        "            entropy = entropy | (nat(1) << nat(i))",
        "    return seeded_pcg32_from_sequence(seed, entropy)",
        "",
        "",
    ]


def generate_guppy_source(
    patch: "SurfacePatch",
    *,
    ancilla_budget: int | None = None,
    twirl: "TwirlConfig | None" = None,
    rng: "GuppyRngMaskConfig | None" = None,
    num_rounds: int | None = None,
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
    Per-stabilizer counted-round ``result("...:meas:N", …)`` calls
    and prep-boundary ``result("...:init:meas:N", …)`` calls fire
    in the abstract's batched measurement order, keeping
    detector record offsets transferable between abstract and traced
    paths.

    Args:
        patch: SurfacePatch with geometry configuration.
        ancilla_budget: Optional cap on simultaneously live ancillas.
            ``None`` or a value ``>= total_ancilla`` emits the
            unconstrained shape; ``< total_ancilla`` emits batched.
        twirl: When provided, emit Pauli-twirl-site mask draws between
            consecutive syndrome rounds and apply the sampled physical
            Pauli to each data qubit at runtime. Both ``twirl`` and
            ``rng`` must be supplied together. The encoding is
            ``"bool_array_v1"``: one
            ``result("pauli_mask:round:R", array(lo_q0, hi_q0, ...))``
            call per twirl site, with the per-round body Python-time
            unrolled at source-generation time so each tag fires exactly
            once per shot. ``twirl.frame_output="canonical"`` additionally
            emits measurement records in the canonical untwirled DEM frame.
        rng: Runtime mask source: a stream-separator seed mixed with
            per-shot quantum entropy when ``twirl`` is enabled.
        num_rounds: Number of syndrome rounds to render. Required when
            ``twirl`` is enabled because twirled source is unrolled per round.

    Returns:
        Python/Guppy source code as a string.

    Raises:
        ValueError: If exactly one of ``twirl`` / ``rng`` is supplied.
    """
    from pecos.qec.surface._ancilla_batching import batched_stabilizers, normalize_ancilla_budget

    if (twirl is None) != (rng is None):
        msg = f"twirl and rng must be supplied together; got twirl={twirl!r} rng={rng!r}"
        raise ValueError(msg)
    if twirl is not None:
        twirl.validate_runtime_supported()
        if num_rounds is None:
            msg = "num_rounds is required when twirl is supplied"
            raise ValueError(msg)
        if num_rounds < 1:
            msg = f"num_rounds must be >= 1, got {num_rounds}"
            raise ValueError(msg)
    canonical_frame_output = twirl is not None and twirl.frame_output == "canonical"

    geom = patch.geometry
    num_data = geom.num_data
    num_x_stab = len(geom.x_stabilizers)
    num_z_stab = len(geom.z_stabilizers)
    total_ancilla = num_x_stab + num_z_stab
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    constrained = effective_budget < total_ancilla
    if twirl is not None and constrained:
        msg = (
            f"twirl + constrained ancilla budget is not supported on "
            f"the Guppy runtime path "
            f"(ancilla_budget={ancilla_budget} < total_ancilla={total_ancilla}); "
            "the between_rounds twirl-site schedule assumes the "
            "unconstrained syndrome shape. Pass ancilla_budget=None or "
            ">= total_ancilla, or omit twirl."
        )
        raise ValueError(msg)
    dx, dz = geom.dx, geom.dz

    if twirl is not None:
        imports = [
            "from __future__ import annotations",
            "from typing import no_type_check",
            "",
            "from guppylang import guppy",
            "from guppylang.std.builtins import array, owned, result",
            "from guppylang.std.num import nat",
            "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x, y, z",
        ]
    else:
        imports = [
            "from __future__ import annotations",
            "",
            "from guppylang import guppy",
            "from guppylang.std.builtins import array, owned, result",
            "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x",
        ]

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
        *imports,
        "",
        "",
    ]

    if twirl is not None:
        lines.extend(_render_inline_pcg32())

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
        ],
    )
    if canonical_frame_output:
        lines.append(
            "def syndrome_extraction("
            f"surf: SurfaceCode_{dx}x{dz}, "
            f"frame_x: array[bool, {num_data}], "
            f"frame_z: array[bool, {num_data}]"
            f") -> Syndrome_{dx}x{dz}:",
        )
    else:
        lines.append(
            f"def syndrome_extraction(surf: SurfaceCode_{dx}x{dz}) -> Syndrome_{dx}x{dz}:",
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
            if canonical_frame_output:
                raw_var = f"sx{stab.index}_raw"
                flip_var = f"sx{stab.index}_flip"
                flip_expr = _xor_expr(f"frame_z[{q}]" for q in stab.data_qubits)
                lines.append(f"    {raw_var} = measure(ax{stab.index})")
                lines.append(f"    {flip_var} = {flip_expr}")
                lines.append(f"    sx{stab.index} = {raw_var} != {flip_var}")
                lines.append(f'    result("raw:sx{stab.index}:bit:{idx}", {raw_var})')
            else:
                lines.append(f"    sx{stab.index} = measure(ax{stab.index})")
            lines.append(f'    result("sx{stab.index}:meas:{idx}", sx{stab.index})')
            idx += 1
        for stab in geom.z_stabilizers:
            if canonical_frame_output:
                raw_var = f"sz{stab.index}_raw"
                flip_var = f"sz{stab.index}_flip"
                flip_expr = _xor_expr(f"frame_x[{q}]" for q in stab.data_qubits)
                lines.append(f"    {raw_var} = measure(az{stab.index})")
                lines.append(f"    {flip_var} = {flip_expr}")
                lines.append(f"    sz{stab.index} = {raw_var} != {flip_var}")
                lines.append(f'    result("raw:sz{stab.index}:bit:{idx}", {raw_var})')
            else:
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

    def append_init_syndrome_function(function_name: str, stab_type: str) -> None:
        """Append a basis-prep syndrome-establishment helper."""
        if stab_type == "X":
            stabs = list(geom.x_stabilizers)
            return_type = f"array[bool, {num_x_stab}]"
            return_calls = ", ".join(f"sx{s.index}" for s in stabs)
            doc = "Establish initial X stabilizer signs after Z-basis data prep."
        else:
            stabs = list(geom.z_stabilizers)
            return_type = f"array[bool, {num_z_stab}]"
            return_calls = ", ".join(f"sz{s.index}" for s in stabs)
            doc = "Establish initial Z stabilizer signs after X-basis data prep."

        lines.extend(
            [
                "",
                "",
                "@guppy",
                f"def {function_name}(surf: SurfaceCode_{dx}x{dz}) -> {return_type}:",
                f'    """{doc}"""',
            ],
        )

        if not constrained:
            if stab_type == "X":
                lines.extend(f"    ax{stab.index} = qubit()" for stab in stabs)
                lines.append("")
                lines.append("    # Hadamard on X ancillas")
                lines.extend(f"    h(ax{stab.index})" for stab in stabs)
            else:
                lines.extend(f"    az{stab.index} = qubit()" for stab in stabs)

            for rnd_idx, rnd_gates in enumerate(rounds):
                filtered = [(t, i, q) for t, i, q in rnd_gates if t == stab_type]
                if not filtered:
                    continue
                lines.append("")
                lines.append(f"    # Round {rnd_idx + 1}")
                for _stab_type, stab_idx, data_q in filtered:
                    if stab_type == "X":
                        lines.append(f"    cx(ax{stab_idx}, surf.data[{data_q}])")
                    else:
                        lines.append(f"    cx(surf.data[{data_q}], az{stab_idx})")

            if stab_type == "X":
                lines.append("")
                lines.append("    # Hadamard on X ancillas")
                lines.extend(f"    h(ax{stab.index})" for stab in stabs)

            lines.append("")
            lines.append("    # Measure init ancillas")
            for idx, stab in enumerate(stabs):
                if stab_type == "X":
                    lines.append(f"    sx{stab.index} = measure(ax{stab.index})")
                    lines.append(f'    result("sx{stab.index}:init:meas:{idx}", sx{stab.index})')
                else:
                    lines.append(f"    sz{stab.index} = measure(az{stab.index})")
                    lines.append(f'    result("sz{stab.index}:init:meas:{idx}", sz{stab.index})')
        else:
            batches = batched_stabilizers(patch, effective_budget)
            idx = 0
            for batch_idx, batch in enumerate(batches):
                init_batch = [(t, i) for t, i in batch if t == stab_type]
                if not init_batch:
                    continue
                lines.append("")
                lines.append(f"    # Batch {batch_idx + 1}/{len(batches)} of {stab_type} stabilizers")

                batch_anc_var: dict[tuple[str, int], str] = {}
                for pos, (selected_type, stab_idx) in enumerate(init_batch):
                    var = f"_init_a_b{batch_idx}_p{pos}"
                    batch_anc_var[(selected_type, stab_idx)] = var
                    lines.append(f"    {var} = qubit()")

                if stab_type == "X":
                    lines.append("    # Hadamard on X ancillas in this batch")
                    for selected_type, stab_idx in init_batch:
                        lines.append(f"    h({batch_anc_var[(selected_type, stab_idx)]})")

                batch_keys = set(batch_anc_var.keys())
                for rnd_idx, rnd_gates in enumerate(rounds):
                    rnd_in_batch = [
                        (selected_type, stab_idx, data_q)
                        for selected_type, stab_idx, data_q in rnd_gates
                        if (selected_type, stab_idx) in batch_keys
                    ]
                    if not rnd_in_batch:
                        continue
                    lines.append("")
                    lines.append(f"    # Batch {batch_idx + 1} round {rnd_idx + 1}")
                    for selected_type, stab_idx, data_q in rnd_in_batch:
                        anc = batch_anc_var[(selected_type, stab_idx)]
                        if selected_type == "X":
                            lines.append(f"    cx({anc}, surf.data[{data_q}])")
                        else:
                            lines.append(f"    cx(surf.data[{data_q}], {anc})")

                if stab_type == "X":
                    lines.append("")
                    lines.append("    # Hadamard on X ancillas in this batch")
                    for selected_type, stab_idx in init_batch:
                        lines.append(f"    h({batch_anc_var[(selected_type, stab_idx)]})")

                lines.append("")
                lines.append(f"    # Measure init batch {batch_idx + 1} ancillas")
                for selected_type, stab_idx in init_batch:
                    anc = batch_anc_var[(selected_type, stab_idx)]
                    syn_var = f"sx{stab_idx}" if selected_type == "X" else f"sz{stab_idx}"
                    lines.append(f"    {syn_var} = measure({anc})")
                    lines.append(f'    result("{syn_var}:init:meas:{idx}", {syn_var})')
                    idx += 1

        lines.extend(
            [
                "",
                f"    return array({return_calls})",
            ],
        )

    append_init_syndrome_function("init_z_basis", "X")
    append_init_syndrome_function("init_x_basis", "Z")

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
    lines.extend(_render_memory_experiments(dx, dz, num_data, twirl, rng, num_rounds))

    return "\n".join(lines)


def _xor_expr(terms: object) -> str:
    """Return a Guppy bool XOR expression for the given source terms."""
    parts = list(terms)
    if not parts:
        return "False"
    expr = parts[0]
    for part in parts[1:]:
        expr = f"({expr} != {part})"
    return str(expr)


def _render_memory_experiments(
    dx: int,
    dz: int,
    num_data: int,
    twirl: "TwirlConfig | None",
    rng: "GuppyRngMaskConfig | None",
    num_rounds: int | None,
) -> list[str]:
    """Render both memory factory functions."""
    lines = [
        "# === Memory Experiments ===",
        "",
    ]
    for basis, basis_upper in (("z", "Z"), ("x", "X")):
        if twirl is None:
            lines.extend(_render_plain_memory_block(basis, basis_upper, dx, dz))
        else:
            if rng is None or num_rounds is None:
                msg = "twirled memory rendering requires both rng and num_rounds"
                raise ValueError(msg)
            lines.extend(
                _render_twirled_memory_block(
                    basis,
                    basis_upper,
                    dx,
                    dz,
                    num_data,
                    twirl,
                    rng,
                    num_rounds,
                ),
            )
    return lines


def _render_plain_memory_block(
    basis: str,
    basis_upper: str,
    dx: int,
    dz: int,
) -> list[str]:
    """Render the vanilla handoff memory factory for one basis."""
    init_func = "init_z_basis" if basis == "z" else "init_x_basis"
    init_tag = "init_synx" if basis == "z" else "init_synz"
    return [
        f"def make_memory_{basis}(num_rounds: int):",
        f'    """Create {basis_upper}-basis memory experiment."""',
        "    from guppylang.std.builtins import comptime",
        "",
        "    @guppy",
        f"    def memory_{basis}() -> None:",
        f'        """{basis_upper}-basis memory experiment for dx={dx}, dz={dz}."""',
        f"        surf = prep_{basis}_basis()",
        f"        init_syn = {init_func}(surf)",
        f'        result("{init_tag}", init_syn)',
        "",
        "        for _t in range(comptime(num_rounds)):",
        "            syn = syndrome_extraction(surf)",
        '            result("synx", syn.synx)',
        '            result("synz", syn.synz)',
        "",
        f"        final = measure_{basis}_basis(surf)",
        '        result("final", final)',
        "",
        f"    return memory_{basis}",
        "",
        "",
    ]


def _render_twirled_memory_block(
    basis: str,
    basis_upper: str,
    dx: int,
    dz: int,
    num_data: int,
    twirl: "TwirlConfig",
    rng: "GuppyRngMaskConfig",
    num_rounds: int,
) -> list[str]:
    """Render a Python-time unrolled twirled memory factory."""
    from pecos.qec.surface._twirl_sites import num_twirl_sites, pauli_mask_round_tag

    seed = int(rng.seed)
    canonical_frame_output = twirl.frame_output == "canonical"
    init_func = "init_z_basis" if basis == "z" else "init_x_basis"
    init_tag = "init_synx" if basis == "z" else "init_synz"
    body: list[str] = [
        f'        """{basis_upper}-basis memory experiment for dx={dx}, dz={dz}, num_rounds={num_rounds} (twirled)."""',
        f"        surf = prep_{basis}_basis()",
        "        # RNG seed is structural -- changing it does not invalidate the",
        "        # abstract DEM / topology cache, only the per-shot mask buffer.",
        f"        rng_state, rng_inc = seeded_pcg32_with_quantum_entropy({seed})",
        f'        result("frame_mode:{twirl.frame_output}", True)',
        f"        init_syn = {init_func}(surf)",
        f'        result("{init_tag}", init_syn)',
        "",
    ]
    if canonical_frame_output:
        for q in range(num_data):
            body.append(f"        fx_{q} = False")
            body.append(f"        fz_{q} = False")
        body.append("")

    # Emit num_rounds - 1 twirled rounds, then one final untwirled round.
    n_twirl = num_twirl_sites(num_rounds)
    for r in range(n_twirl):
        body.append(f"        # === Round {r} (twirled) ===")
        if canonical_frame_output:
            frame_x = ", ".join(f"fx_{q}" for q in range(num_data))
            frame_z = ", ".join(f"fz_{q}" for q in range(num_data))
            body.append(
                f"        syn = syndrome_extraction(surf, array({frame_x}), array({frame_z}))",
            )
        else:
            body.append("        syn = syndrome_extraction(surf)")
        body.append('        result("synx", syn.synx)')
        body.append('        result("synz", syn.synz)')
        body.append("        # Pauli twirl site between this round and the next.")
        for q in range(num_data):
            body.append(f"        rng_state, m_{r}_{q} = _pcg32_next4(rng_state, rng_inc)")
            body.append(f"        if m_{r}_{q} == 1:")
            body.append(f"            x(surf.data[{q}])")
            body.append(f"        if m_{r}_{q} == 2:")
            body.append(f"            y(surf.data[{q}])")
            body.append(f"        if m_{r}_{q} == 3:")
            body.append(f"            z(surf.data[{q}])")
            body.append(f"        lo_{r}_{q} = (m_{r}_{q} == 1) | (m_{r}_{q} == 3)")
            body.append(f"        hi_{r}_{q} = (m_{r}_{q} == 2) | (m_{r}_{q} == 3)")
            if canonical_frame_output:
                body.append(
                    f"        twx_{r}_{q} = (m_{r}_{q} == 1) | (m_{r}_{q} == 2)",
                )
                body.append(
                    f"        twz_{r}_{q} = (m_{r}_{q} == 2) | (m_{r}_{q} == 3)",
                )
                body.append(f"        fx_{q} = fx_{q} != twx_{r}_{q}")
                body.append(f"        fz_{q} = fz_{q} != twz_{r}_{q}")
        elements = ", ".join(f"lo_{r}_{q}, hi_{r}_{q}" for q in range(num_data))
        tag = pauli_mask_round_tag(r)
        body.append(f'        result("{tag}", array({elements}))')
        body.append("")

    if num_rounds > 0:
        body.append(f"        # === Round {num_rounds - 1} (final, no twirl after) ===")
        if canonical_frame_output:
            frame_x = ", ".join(f"fx_{q}" for q in range(num_data))
            frame_z = ", ".join(f"fz_{q}" for q in range(num_data))
            body.append(
                f"        syn = syndrome_extraction(surf, array({frame_x}), array({frame_z}))",
            )
        else:
            body.append("        syn = syndrome_extraction(surf)")
        body.append('        result("synx", syn.synx)')
        body.append('        result("synz", syn.synz)')
        body.append("")

    if canonical_frame_output:
        body.append(f"        final_raw = measure_{basis}_basis(surf)")
        body.append('        result("raw:final", final_raw)')
        for q in range(num_data):
            flip_var = f"fx_{q}" if basis == "z" else f"fz_{q}"
            body.append(f"        final_{q} = final_raw[{q}] != {flip_var}")
        final_elements = ", ".join(f"final_{q}" for q in range(num_data))
        body.append(f'        result("final", array({final_elements}))')
    else:
        body.append(f"        final = measure_{basis}_basis(surf)")
        body.append('        result("final", final)')

    return [
        f"def make_memory_{basis}(num_rounds: int):",
        f'    """Create {basis_upper}-basis twirled memory experiment.',
        "",
        f"    num_rounds must equal {num_rounds} -- the body was unrolled at",
        "    source-generation time. Mismatched values raise ValueError.",
        '    """',
        f"    if num_rounds != {num_rounds}:",
        f'        msg = f"this generated module was unrolled for num_rounds={num_rounds}, got {{num_rounds!r}}"',
        "        raise ValueError(msg)",
        "",
        "    @guppy",
        f"    def memory_{basis}() -> None:",
        *body,
        "",
        f"    return memory_{basis}",
        "",
        "",
    ]


def _validate_surface_memory_distance(d: int) -> None:
    """Enforce the surface-memory Guppy entry-point distance contract.

    The distance-based public entry points (:func:`get_num_qubits`,
    :func:`get_surface_code_module`, :func:`make_surface_code`,
    :func:`generate_surface_code_module`) document and require an odd code
    distance ``>= 3``. Validate it in one place so they fail loud
    consistently rather than silently building an out-of-contract program
    (the patch-based entry points validate via ``SurfacePatch`` instead).
    """
    if d < 3 or d % 2 == 0:
        msg = f"Distance must be odd >= 3, got {d}"
        raise ValueError(msg)


def _guppy_module_cache_key(
    patch: "SurfacePatch",
    effective_budget: int,
    twirl: "TwirlConfig | None" = None,
    rng: "GuppyRngMaskConfig | None" = None,
    num_rounds: int | None = None,
) -> str:
    """Filesystem-safe cache key spanning full patch identity + budget + twirl.

    Mirrors the topology identity used by the native cache
    (``decode._surface_patch_cache_key``): dx, dz, orientation, and the
    rotated flag. Keying on distance/dx-dz alone would collide a rotated and
    a non-rotated patch of the same shape onto one generated module.

    Twirled source is Python-time unrolled, so the cache key includes
    ``num_rounds`` in addition to the structural twirl fields, runtime
    frame-output mode, and RNG seed.
    """
    geom = patch.geometry
    rotated = "rot" if geom.rotated else "unrot"
    base = f"{patch.dx}x{patch.dz}_{geom.orientation.name}_{rotated}_b{effective_budget}"
    if twirl is None:
        return base
    if rng is None or num_rounds is None:
        msg = "twirled Guppy module cache keys require both rng and num_rounds"
        raise ValueError(msg)
    twirl_part = (
        f"t-{twirl.scheme}-{twirl.site_schedule}-{twirl.result_encoding}"
        f"-frame-{twirl.frame_output}"
        f"-s{int(rng.seed)}-r{int(num_rounds)}"
    )
    return f"{base}_{twirl_part}"


def _load_guppy_module(
    patch: "SurfacePatch",
    *,
    ancilla_budget: int | None = None,
    twirl: "TwirlConfig | None" = None,
    rng: "GuppyRngMaskConfig | None" = None,
    num_rounds: int | None = None,
) -> dict:
    """Load a Guppy module for a patch, using caching.

    The cache key spans the full patch identity (dx, dz, orientation,
    rotated) and the **effective** budget (after clamping via
    ``normalize_ancilla_budget``), so ``ancilla_budget=None`` and
    ``ancilla_budget >= total_ancilla`` resolve to the same cache entry
    while distinct patch geometries never collide. Twirled source also
    keys on twirl fields, frame-output mode, RNG seed, and round count.

    Args:
        patch: SurfacePatch with geometry
        ancilla_budget: Optional cap on simultaneously live ancillas
        twirl: Pauli-twirl-site declaration (structural)
        rng: Runtime mask RNG seed (must be supplied with ``twirl``)
        num_rounds: Syndrome-round count for unrolled twirled source.

    Returns:
        Module dictionary with generated functions
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    cache_key = _guppy_module_cache_key(patch, effective_budget, twirl, rng, num_rounds)

    if cache_key in _state.module_cache:
        return _state.module_cache[cache_key]

    source = generate_guppy_source(
        patch,
        ancilla_budget=ancilla_budget,
        twirl=twirl,
        rng=rng,
        num_rounds=num_rounds,
    )

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
    twirl: "TwirlConfig | None" = None,
    rng: "GuppyRngMaskConfig | None" = None,
) -> object:
    """Generate a memory experiment for a patch.

    Args:
        patch: SurfacePatch configuration
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas
        twirl: Pauli-twirl-site declaration; must be supplied with ``rng``.
        rng: Runtime mask RNG seed; must be supplied with ``twirl``.

    Returns:
        Guppy function for the experiment
    """
    module = _load_guppy_module(
        patch,
        ancilla_budget=ancilla_budget,
        twirl=twirl,
        rng=rng,
        num_rounds=num_rounds if twirl is not None else None,
    )

    if basis.upper() == "Z":
        factory = module["make_memory_z"]
    elif basis.upper() == "X":
        factory = module["make_memory_x"]
    else:
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    return factory(num_rounds)


def get_num_qubits(
    d: int | None = None,
    *,
    patch: "SurfacePatch | None" = None,
    ancilla_budget: int | None = None,
    twirl: "TwirlConfig | None" = None,
) -> int:
    """Get the peak simultaneously-live qubit count for a surface-code program.

    Provide exactly one of ``d`` or ``patch``:

    - ``d`` (odd >= 3): the default symmetric rotated patch, with
      ``d^2`` data and ``d^2 - 1`` ancilla qubits.
    - ``patch``: any geometry (asymmetric / non-rotated included); counts
      are derived from ``patch.geometry`` so the result is faithful to the
      patch actually being traced -- not a scalar-distance approximation.

    Unconstrained (``ancilla_budget=None``): peak count is
    ``num_data + total_ancilla``. Constrained: the program reuses ancilla
    slots across stabilizer-measurement batches, so only
    ``num_data + min(ancilla_budget, total_ancilla)`` slots are live at once.
    Clamping matches ``normalize_ancilla_budget``, so the
    unconstrained-via-``None`` and unconstrained-via-large-int cases collapse.
    Twirled Guppy programs allocate one additional side-band entropy qubit at
    a time for per-shot mask seeding.

    Returns:
        Total qubits the traced program will simultaneously use.
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    if (d is None) == (patch is None):
        msg = "get_num_qubits requires exactly one of d=... or patch=..."
        raise ValueError(msg)

    if patch is not None:
        geom = patch.geometry
        num_data = geom.num_data
        total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    else:
        _validate_surface_memory_distance(d)
        num_data = d * d
        total_ancilla = d * d - 1

    twirl_entropy_qubits = 1 if twirl is not None else 0
    return num_data + normalize_ancilla_budget(total_ancilla, ancilla_budget) + twirl_entropy_qubits


def generate_surface_code_module(d: int, *, ancilla_budget: int | None = None) -> str:
    """Generate source code for a distance-d surface code module.

    Args:
        d: Code distance (must be odd >= 3)
        ancilla_budget: Optional cap on simultaneously live ancillas;
            forwarded to ``generate_guppy_source``.

    Returns:
        Python/Guppy source code as a string
    """
    _validate_surface_memory_distance(d)

    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    return generate_guppy_source(patch, ancilla_budget=ancilla_budget)


def _surface_code_module_for_patch(patch: "SurfacePatch", *, ancilla_budget: int | None = None) -> dict:
    """Load + cache a surface-code module for an arbitrary patch.

    Cache key spans full patch identity (dx, dz, orientation, rotated) plus
    the effective budget, so distinct geometries never collide and the
    unconstrained-via-``None`` / unconstrained-via-large-int cases share one
    entry. Module metadata is derived from the patch geometry (faithful for
    asymmetric / non-rotated patches), not from a scalar distance.
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    cache_key = (patch.dx, patch.dz, geom.orientation.name, geom.rotated, effective_budget)

    if cache_key in _state.distance_module_cache:
        return _state.distance_module_cache[cache_key]

    module = _load_guppy_module(patch, ancilla_budget=ancilla_budget)

    # Metadata derived from the actual patch geometry.
    module["distance"] = patch.distance
    module["num_data"] = geom.num_data
    module["num_stab"] = total_ancilla
    module["ancilla_budget"] = effective_budget

    _state.distance_module_cache[cache_key] = module
    return module


def get_surface_code_module(d: int, *, ancilla_budget: int | None = None) -> dict:
    """Get a loaded surface code module for distance d.

    Args:
        d: Code distance (must be odd >= 3)
        ancilla_budget: Optional cap on simultaneously live ancillas

    Returns:
        Dictionary with module contents and metadata
    """
    from pecos.qec.surface import SurfacePatch

    _validate_surface_memory_distance(d)
    patch = SurfacePatch.create(distance=d)
    return _surface_code_module_for_patch(patch, ancilla_budget=ancilla_budget)


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
