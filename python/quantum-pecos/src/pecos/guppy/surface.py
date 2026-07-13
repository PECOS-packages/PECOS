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
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pecos.qec.surface.schedule import compute_cnot_schedule

if TYPE_CHECKING:
    from pecos.qec.surface import GuppyRngMaskConfig, SurfacePatch, TwirlConfig
    from pecos.qec.surface._check_plan import ResolvedSurfaceCheckPlan


# Module state container (avoids global statement)
class _ModuleState:
    """Container for module-level mutable state."""

    temp_dir: ClassVar[Path | None] = None
    module_cache: ClassVar[dict[str, object]] = {}
    # Keyed by full patch identity + effective budget (dx, dz, orientation,
    # rotated, effective_budget) so distinct patch geometries -- e.g. rotated
    # vs non-rotated at the same dx/dz -- never collide on a cached module.
    distance_module_cache: ClassVar[
        dict[tuple[int, int, str, bool, int, str, str, str | None, str, bool, int | None], dict]
    ] = {}


_state = _ModuleState()


_SZZ_RUNTIME_BARRIER_POLICY_NONE = "none"
_SZZ_RUNTIME_BARRIER_POLICY_ALL = "all"
_SZZ_RUNTIME_BARRIER_POLICY_DATA_PREFIX = "data-prefix"
_SZZ_RUNTIME_BARRIER_POLICIES = frozenset(
    {
        _SZZ_RUNTIME_BARRIER_POLICY_NONE,
        _SZZ_RUNTIME_BARRIER_POLICY_ALL,
        _SZZ_RUNTIME_BARRIER_POLICY_DATA_PREFIX,
    },
)
_PACKED_TRACE_METADATA_JSON_KEY = "__pecos_trace_metadata_json_v1__"


def _normalize_szz_runtime_barrier_policy(value: bool | str) -> str:
    """Return the canonical SZZ runtime-barrier policy token."""
    if isinstance(value, bool):
        return _SZZ_RUNTIME_BARRIER_POLICY_ALL if value else _SZZ_RUNTIME_BARRIER_POLICY_NONE
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return _SZZ_RUNTIME_BARRIER_POLICY_ALL
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return _SZZ_RUNTIME_BARRIER_POLICY_NONE
    if normalized not in _SZZ_RUNTIME_BARRIER_POLICIES:
        msg = (
            "szz_runtime_barriers must be a boolean or one of "
            f"{sorted(_SZZ_RUNTIME_BARRIER_POLICIES)}, got {value!r}"
        )
        raise ValueError(msg)
    return normalized


def _normalize_surface_interaction_basis(interaction_basis: str) -> str:
    from pecos.qec.surface.circuit_builder import _normalize_interaction_basis

    return _normalize_interaction_basis(interaction_basis)


def _resolve_surface_check_plan(
    *,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
) -> "ResolvedSurfaceCheckPlan":
    from pecos.qec.surface._check_plan import resolve_surface_check_plan

    return resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )


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
        "def _pcg32_next32(state: nat, inc: nat) -> tuple[nat, nat]:",
        "    old_state = state",
        "    new_state = _pcg32_advance(state, inc)",
        "    xorshifted = _pcg32_mask32(((old_state >> nat(18)) ^ old_state) >> nat(27))",
        "    rot = _pcg32_mask32(old_state >> nat(59))",
        "    rot_inv = _pcg32_mask32((~rot + nat(1)) & nat(31))",
        "    output = _pcg32_mask32((xorshifted >> rot) | (xorshifted << rot_inv))",
        "    return new_state, output",
        "",
        "",
        "@guppy",
        "@no_type_check",
        "def _pcg32_next4(state: nat, inc: nat) -> tuple[nat, int]:",
        "    new_state, output = _pcg32_next32(state, inc)",
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
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
) -> str:
    """Generate Guppy source code for a surface code patch.

    Uses a 4-round parallel schedule for syndrome extraction. The default
    ``interaction_basis="cx"`` emits the CNOT template; ``"szz"`` emits the
    signed SZZ/SZZdg template. SZZ helpers forward-flow single-qubit data
    frames within each helper and then explicitly flush the frame before
    returning, preserving the reusable Guppy result-tag structure.

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
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset. This is the source of
            truth when supplied; ``interaction_basis`` must agree if also
            supplied. Current Guppy generation maps the resolved plan to the
            corresponding CX or SZZ/SZZdg concrete template.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy. Currently supported for SZZ global axis-cycle and
            checkerboard XZZX/ZXXZ deformed-check frames.
        szz_runtime_barriers: SZZ/SZZdg scheduling-barrier policy. ``False``
            or ``"none"`` emits no barriers; ``True`` or ``"all"`` emits a
            public Guppy ``barrier`` before every SZZ/SZZdg host region;
            ``"data-prefix"`` emits one only after a non-virtual local
            data-frame pulse is discharged and before its host. Barriers have
            no ideal-unitary effect, but give runtimes a principled scheduling
            boundary between selected local data-frame pulses and their
            entangling host.
        trace_metadata: Emit PECOS trace metadata helpers for SZZ/SZZdg
            hosted-operation diagnostics and strict DEM construction. Disable
            only for execution-only builds whose compiler/linker cannot resolve
            PECOS trace metadata helper symbols.

    Returns:
        Python/Guppy source code as a string.

    Raises:
        ValueError: If exactly one of ``twirl`` / ``rng`` is supplied.
    """
    from pecos.qec.surface._ancilla_batching import batched_stabilizers, normalize_ancilla_budget
    from pecos.qec.surface._check_plan import (
        ancilla_schedule_for_check_plan,
        cnot_round_order_for_check_plan,
        require_current_surface_check_plan_renderer,
    )
    from pecos.qec.surface.circuit_builder import (
        _SZZ_FLOW_IDENTITY,
        _SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING,
        OpType,
        _resolve_szz_clifford_frame_for_builder,
        _szz_flow_clifford_name,
        _szz_flow_compose_pending_gate,
        _szz_flow_is_virtual_z,
        _szz_memory_physical_axis_for_data,
        _szz_residual_plan_for_check_plan,
    )

    resolved_plan = _resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    require_current_surface_check_plan_renderer(
        resolved_plan,
        context="Guppy surface-code source generation",
    )
    ancilla_schedule = ancilla_schedule_for_check_plan(resolved_plan)
    cnot_round_order = cnot_round_order_for_check_plan(resolved_plan)
    interaction_basis = resolved_plan.interaction_basis
    szz_runtime_barrier_policy = _normalize_szz_runtime_barrier_policy(szz_runtime_barriers)
    if interaction_basis != "szz" and szz_runtime_barrier_policy != _SZZ_RUNTIME_BARRIER_POLICY_NONE:
        msg = "szz_runtime_barriers is only supported for interaction_basis='szz'"
        raise ValueError(msg)
    resolved_clifford_frame = _resolve_szz_clifford_frame_for_builder(
        patch,
        interaction_basis=interaction_basis,
        clifford_frame_policy=clifford_frame_policy,
    )
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
    if interaction_basis == "szz" and twirl is not None:
        msg = "interaction_basis='szz' Guppy runtime twirl integration is staged later"
        raise ValueError(msg)
    if twirl is not None and constrained:
        msg = (
            f"twirl + constrained ancilla budget is not supported on "
            f"the Guppy runtime path "
            f"(ancilla_budget={ancilla_budget} < total_ancilla={total_ancilla}); "
            "the runtime twirl-site schedules assume the "
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
    elif interaction_basis == "szz":
        imports = [
            "from __future__ import annotations",
            "",
            "from guppylang import guppy",
            "from guppylang.std.angles import angle",
            "from guppylang.std.builtins import array, owned, result",
            "from guppylang.std.qsystem.functional import phased_x, rz, zz_phase",
            "from guppylang.std.quantum import discard, h, measure, measure_array, qubit, s, sdg, v, vdg, x, y, z",
        ]
    else:
        imports = [
            "from __future__ import annotations",
            "",
            "from guppylang import guppy",
            "from guppylang.std.builtins import array, owned, result",
            "from guppylang.std.quantum import cx, discard, h, measure, measure_array, qubit, x, z",
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
        f"Interaction basis: {interaction_basis}",
        f"Check plan: {resolved_plan.plan_id}",
        '"""',
        "",
        *imports,
        "",
        "",
    ]

    if twirl is not None:
        lines.extend(_render_inline_pcg32())

    if interaction_basis == "szz":
        helper_declarations: list[str] = []
        if trace_metadata:
            helper_declarations.extend(
                [
                    "@guppy.declare",
                    (
                        "def pecos_qis_trace_metadata_qubit_hugr("
                        "q: qubit @ owned, key: str, value: str"
                        ") -> qubit: ..."
                    ),
                    "",
                ],
            )
        helper_declarations.extend(
            [
                "@guppy.declare",
                "def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...",
                "",
                "@guppy.declare",
                (
                    "def pecos_qis_runtime_barrier_qubits2_hugr("
                    "q0: qubit @ owned, q1: qubit @ owned"
                    ") -> tuple[qubit, qubit]: ..."
                ),
                "",
                "",
            ],
        )
        lines.extend(helper_declarations)

    # Generate struct definitions.
    lines.extend(
        [
            "@guppy.struct",
            f"class SurfaceCode_{dx}x{dz}:",
            f'    """Surface code patch with dx={dx}, dz={dz} ({num_data} data qubits)."""',
            "",
        ],
    )
    if interaction_basis == "szz":
        lines.extend(f"    d{i}: qubit" for i in range(num_data))
    else:
        lines.append(f"    data: array[qubit, {num_data}]")
    lines.extend(
        [
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

    szz_data_args = ", ".join(f"d{i}" for i in range(num_data))

    def _append_szz_data_unpack(target: list[str], indent: str) -> None:
        target.extend(f"{indent}d{i} = surf.d{i}" for i in range(num_data))

    def _szz_data_expr(data_q: int) -> str:
        return f"d{data_q}"

    szz_check_by_key = {}
    if resolved_clifford_frame is not None:
        szz_check_by_key.update({("X", check.stabilizer_index): check for check in resolved_clifford_frame.x_checks})
        szz_check_by_key.update({("Z", check.stabilizer_index): check for check in resolved_clifford_frame.z_checks})

    def _szz_physical_axis_for_touch(stabilizer_type: str, stab_idx: int, data_q: int) -> str:
        if resolved_clifford_frame is None:
            return "X" if stabilizer_type == "X" else "Z"
        check = szz_check_by_key[(stabilizer_type, stab_idx)]
        try:
            offset = check.data_qubits.index(data_q)
        except ValueError as exc:
            msg = f"data qubit {data_q} is not in resolved check {stabilizer_type}{stab_idx}"
            raise ValueError(msg) from exc
        return check.paulis[offset].axis

    def _szz_physical_axis_for_memory_data(source_basis: str, data_q: int) -> str:
        return _szz_memory_physical_axis_for_data(
            source_basis,
            resolved_clifford_frame,
            data_q,
        )

    def _szz_physical_axis_for_logical_data(source_logical: str, data_q: int) -> str:
        if resolved_clifford_frame is None:
            return source_logical
        logical = resolved_clifford_frame.logical_x if source_logical == "X" else resolved_clifford_frame.logical_z
        try:
            offset = logical.data_qubits.index(data_q)
        except ValueError as exc:
            msg = f"data qubit {data_q} is not in source logical {source_logical}"
            raise ValueError(msg) from exc
        return logical.paulis[offset].axis

    def _append_szz_axis_rotation_to_z(target: list[str], indent: str, axis: str, qubit_expr: str) -> None:
        if axis == "X":
            target.append(f"{indent}h({qubit_expr})")
        elif axis == "Y":
            target.append(f"{indent}vdg({qubit_expr})")
        elif axis != "Z":
            msg = f"unsupported Pauli axis {axis!r}"
            raise ValueError(msg)

    def _append_szz_axis_rotation_from_z(target: list[str], indent: str, axis: str, qubit_expr: str) -> None:
        if axis == "X":
            target.append(f"{indent}h({qubit_expr})")
        elif axis == "Y":
            target.append(f"{indent}v({qubit_expr})")
        elif axis != "Z":
            msg = f"unsupported Pauli axis {axis!r}"
            raise ValueError(msg)

    def _append_szz_y_compensation(target: list[str], indent: str, qubit_expr: str, *, dagger: bool) -> None:
        target.append(f"{indent}sdg({qubit_expr})")
        target.append(f"{indent}{'vdg' if dagger else 'v'}({qubit_expr})")
        target.append(f"{indent}s({qubit_expr})")

    def _append_szz_touch_compensation(target: list[str], indent: str, axis: str, sign: int, qubit_expr: str) -> None:
        if axis == "X":
            target.append(f"{indent}{'vdg' if sign > 0 else 'v'}({qubit_expr})")
        elif axis == "Y":
            _append_szz_y_compensation(target, indent, qubit_expr, dagger=sign > 0)
        elif axis == "Z":
            target.append(f"{indent}{'sdg' if sign > 0 else 's'}({qubit_expr})")
        else:
            msg = f"unsupported Pauli axis {axis!r}"
            raise ValueError(msg)

    def _szz_axis_rotation_to_z_gates(axis: str) -> tuple[OpType, ...]:
        if axis == "X":
            return (OpType.H,)
        if axis == "Y":
            return (OpType.SXDG,)
        if axis == "Z":
            return ()
        msg = f"unsupported Pauli axis {axis!r}"
        raise ValueError(msg)

    def _szz_axis_rotation_from_z_gates(axis: str) -> tuple[OpType, ...]:
        if axis == "X":
            return (OpType.H,)
        if axis == "Y":
            return (OpType.SX,)
        if axis == "Z":
            return ()
        msg = f"unsupported Pauli axis {axis!r}"
        raise ValueError(msg)

    def _szz_touch_compensation_gates(axis: str, sign: int) -> tuple[OpType, ...]:
        if axis == "X":
            return (OpType.SXDG if sign > 0 else OpType.SX,)
        if axis == "Y":
            return (
                OpType.SZDG,
                OpType.SXDG if sign > 0 else OpType.SX,
                OpType.SZ,
            )
        if axis == "Z":
            return (OpType.SZDG if sign > 0 else OpType.SZ,)
        msg = f"unsupported Pauli axis {axis!r}"
        raise ValueError(msg)

    def _append_szz_trace_metadata_payload(
        target: list[str],
        indent: str,
        metadata: dict[str, str],
        qubit_expr: str,
    ) -> None:
        if not trace_metadata or not metadata:
            return
        payload = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        target.append(
            f"{indent}{qubit_expr} = pecos_qis_trace_metadata_qubit_hugr("
            f"{qubit_expr}, "
            f"{json.dumps(_PACKED_TRACE_METADATA_JSON_KEY)}, "
            f"{json.dumps(payload)})",
        )

    def _append_szz_gate_trace_metadata(
        target: list[str],
        indent: str,
        *,
        source_kind: str,
        source_label: str,
        qubit_expr: str,
        host_label: str | None = None,
        local_role: str | None = None,
        gate: OpType | None = None,
        lowering_required: bool = False,
    ) -> None:
        metadata = {
            "source_kind": source_kind,
            "source_label": source_label,
        }
        if host_label is not None:
            metadata["szz_host_label"] = host_label
            metadata["host_id"] = host_label
        if local_role is not None:
            metadata["local_role"] = local_role
        if gate is not None:
            metadata["source_gate"] = gate.name
        if lowering_required:
            metadata["source_lowering_required"] = "true"
        _append_szz_trace_metadata_payload(target, indent, metadata, qubit_expr)

    def _append_szz_flow_gate(
        target: list[str],
        indent: str,
        op_type: OpType,
        qubit_expr: str,
        *,
        source_label: str | None = None,
        host_label: str | None = None,
    ) -> None:
        op_name = {
            OpType.H: "h",
            OpType.SX: "v",
            OpType.SXDG: "vdg",
            OpType.SZ: "s",
            OpType.SZDG: "sdg",
            OpType.X: "x",
            OpType.Z: "z",
        }.get(op_type)
        if op_name is None:
            msg = f"unsupported Guppy SZZ forward-flow gate {op_type.name}"
            raise ValueError(msg)
        if source_label is not None:
            _append_szz_gate_trace_metadata(
                target,
                indent,
                source_kind="szz_data_prefix",
                source_label=source_label,
                qubit_expr=qubit_expr,
                host_label=host_label,
                local_role="basis_prefix",
                gate=op_type,
            )
        target.append(f"{indent}{op_name}({qubit_expr})")

    def _append_szz_physical_prefix_gate(
        target: list[str],
        indent: str,
        op_type: OpType,
        qubit_expr: str,
        *,
        source_label: str,
        host_label: str,
    ) -> None:
        """Append one hosted physical SZZ prefix pulse.

        Guppy's public quantum stdlib does not expose PECOS ``F``/``SY``
        names directly.  The hardware-level SZZ forward-flow table still has
        a one-pulse interpretation: ``SY`` is a Y-axis sqrt pulse, and ``F``
        is an X-axis sqrt pulse plus a virtual Z-frame update.  We attach the
        hosted metadata only to the physical pulse so scheduling diagnostics
        track the operation that must remain adjacent to its SZZ/SZZdg host.
        """
        _append_szz_gate_trace_metadata(
            target,
            indent,
            source_kind="szz_data_prefix",
            source_label=source_label,
            qubit_expr=qubit_expr,
            host_label=host_label,
            local_role="basis_prefix",
            gate=op_type,
        )
        if op_type == OpType.H:
            target.append(f"{indent}h({qubit_expr})")
        elif op_type == OpType.SX:
            target.append(f"{indent}{qubit_expr} = phased_x({qubit_expr}, angle(0.5), angle(0.0))")
        elif op_type == OpType.SXDG:
            target.append(f"{indent}{qubit_expr} = phased_x({qubit_expr}, angle(-0.5), angle(0.0))")
        elif op_type == OpType.SY:
            target.append(f"{indent}{qubit_expr} = phased_x({qubit_expr}, angle(0.5), angle(0.5))")
        elif op_type == OpType.SYDG:
            target.append(f"{indent}{qubit_expr} = phased_x({qubit_expr}, angle(-0.5), angle(0.5))")
        elif op_type == OpType.F:
            target.append(f"{indent}{qubit_expr} = phased_x({qubit_expr}, angle(0.5), angle(0.0))")
            target.append(f"{indent}{qubit_expr} = rz({qubit_expr}, angle(0.5))")
        elif op_type == OpType.FDG:
            target.append(f"{indent}{qubit_expr} = phased_x({qubit_expr}, angle(0.5), angle(-0.5))")
            target.append(f"{indent}{qubit_expr} = rz({qubit_expr}, angle(-0.5))")
        else:
            msg = f"unsupported hosted SZZ physical prefix gate {op_type.name}"
            raise ValueError(msg)

    def _szz_guppy_physical_prefix_for_pending(
        pending: tuple[int, int],
    ) -> tuple[OpType | None, OpType]:
        """Return the source-level physical-prefix lowering for ``pending``."""
        try:
            return _SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING[pending]
        except KeyError as exc:
            msg = f"SZZ Guppy hosted-prefix lowering cannot lower pending Clifford {_szz_flow_clifford_name(pending)}"
            raise ValueError(msg) from exc

    _szz_guppy_prefix_cache: dict[tuple[int, int], tuple[OpType, ...]] = {_SZZ_FLOW_IDENTITY: ()}
    _szz_guppy_prefix_generators = (
        OpType.H,
        OpType.SX,
        OpType.SXDG,
        OpType.SZ,
        OpType.SZDG,
        OpType.X,
        OpType.Z,
    )

    def _szz_guppy_prefix_gates_for_pending(pending: tuple[int, int]) -> tuple[OpType, ...]:
        """Return an exact Guppy-supported 1q Clifford sequence for ``pending``."""
        cached = _szz_guppy_prefix_cache.get(pending)
        if cached is not None:
            return cached

        queue: list[tuple[tuple[int, int], tuple[OpType, ...]]] = [(_SZZ_FLOW_IDENTITY, ())]
        seen = {_SZZ_FLOW_IDENTITY}
        while queue:
            current, gates = queue.pop(0)
            for gate in _szz_guppy_prefix_generators:
                next_pending = _szz_flow_compose_pending_gate(current, gate)
                if next_pending in seen:
                    continue
                next_gates = (*gates, gate)
                _szz_guppy_prefix_cache[next_pending] = next_gates
                if next_pending == pending:
                    return next_gates
                seen.add(next_pending)
                queue.append((next_pending, next_gates))

        msg = f"cannot synthesize Guppy SZZ forward-flow Clifford prefix for {pending!r}"
        raise ValueError(msg)

    def _append_szz_flush_data_frame(
        target: list[str],
        indent: str,
        pending_by_data: dict[int, tuple[int, int]],
        *,
        reason: str,
    ) -> None:
        """Materialize pending data Cliffords before returning a frame-free helper."""
        emitted_comment = False
        for data_q, pending in sorted(pending_by_data.items()):
            if pending == _SZZ_FLOW_IDENTITY:
                continue
            prefix = _szz_guppy_prefix_gates_for_pending(pending)
            if prefix and not emitted_comment:
                target.append("")
                target.append(f"{indent}# Flush SZZ data frame before {reason}")
                emitted_comment = True
            for gate in prefix:
                _append_szz_flow_gate(target, indent, gate, f"d{data_q}")
            pending_by_data[data_q] = _SZZ_FLOW_IDENTITY

    def _append_szz_logical_pauli(target: list[str], indent: str, axis: str, qubit_expr: str) -> None:
        if axis == "X":
            target.append(f"{indent}x({qubit_expr})")
        elif axis == "Y":
            target.append(f"{indent}y({qubit_expr})")
        elif axis == "Z":
            target.append(f"{indent}z({qubit_expr})")
        else:
            msg = f"unsupported Pauli axis {axis!r}"
            raise ValueError(msg)

    # Generate state preparation functions
    lines.extend(["# === State Preparation ===", "", "@guppy", f"def prep_z_basis() -> SurfaceCode_{dx}x{dz}:"])
    lines.append('    """Prepare logical |0_L> state."""')
    if interaction_basis == "szz":
        lines.extend(f"    d{i} = qubit()" for i in range(num_data))
        for i in range(num_data):
            _append_szz_axis_rotation_to_z(
                lines,
                "    ",
                _szz_physical_axis_for_memory_data("Z", i),
                f"d{i}",
            )
        lines.append(f"    return SurfaceCode_{dx}x{dz}({szz_data_args})")
    else:
        lines.append(f"    data = array(qubit() for _ in range({num_data}))")
        lines.append(f"    return SurfaceCode_{dx}x{dz}(data)")
    lines.extend(["", "", "@guppy", f"def prep_x_basis() -> SurfaceCode_{dx}x{dz}:"])
    lines.append('    """Prepare logical |+_L> state."""')
    if interaction_basis == "szz":
        lines.extend(f"    d{i} = qubit()" for i in range(num_data))
        for i in range(num_data):
            _append_szz_axis_rotation_to_z(
                lines,
                "    ",
                _szz_physical_axis_for_memory_data("X", i),
                f"d{i}",
            )
        lines.append(f"    return SurfaceCode_{dx}x{dz}({szz_data_args})")
    else:
        lines.append(f"    data = array(qubit() for _ in range({num_data}))")
        lines.append(f"    for i in range({num_data}):")
        lines.append("        h(data[i])")
        lines.append(f"    return SurfaceCode_{dx}x{dz}(data)")
    lines.extend(["", ""])

    # Generate syndrome extraction with the selected parallel interaction schedule.
    rounds = compute_cnot_schedule(patch, round_order=cnot_round_order)
    szz_sign_by_touch: dict[tuple[str, int, int], int] = {}
    if interaction_basis == "szz":
        residual_plan = _szz_residual_plan_for_check_plan(patch, resolved_plan)
        szz_sign_by_touch = {
            (entry.stabilizer_type, entry.stabilizer_index, entry.data_qubit): entry.sign
            for entry in residual_plan.signs
        }

    def _szz_ancilla_expr(stab_type: str, stab_idx: int) -> str:
        return f"ax{stab_idx}" if stab_type == "X" else f"az{stab_idx}"

    def _append_szz_layer(
        target: list[str],
        indent: str,
        rnd_idx: int,
        layer_gates: list[tuple[str, int, int]],
        ancilla_expr: Callable[[str, int], str],
        data_expr: Callable[[int], str],
        pending_by_data: dict[int, tuple[int, int]] | None = None,
        host_label_scope: str | None = None,
    ) -> None:
        def compose_data(data_q: int, gates: tuple[OpType, ...]) -> None:
            if pending_by_data is None:
                for gate in gates:
                    _append_szz_flow_gate(target, indent, gate, data_expr(data_q))
                return
            pending = pending_by_data.setdefault(data_q, _SZZ_FLOW_IDENTITY)
            for gate in gates:
                pending = _szz_flow_compose_pending_gate(pending, gate)
            pending_by_data[data_q] = pending

        def discharge_data_for_szz(
            data_q: int,
            *,
            host_label: str,
        ) -> None:
            if pending_by_data is None:
                return
            pending = pending_by_data.setdefault(data_q, _SZZ_FLOW_IDENTITY)
            if pending == _SZZ_FLOW_IDENTITY or _szz_flow_is_virtual_z(pending):
                return
            virtual_gate, physical_gate = _szz_guppy_physical_prefix_for_pending(pending)
            if virtual_gate is not None:
                _append_szz_flow_gate(target, indent, virtual_gate, data_expr(data_q))
            _append_szz_physical_prefix_gate(
                target,
                indent,
                physical_gate,
                data_expr(data_q),
                source_label=f"{host_label}:prefix:0:{physical_gate.name}",
                host_label=host_label,
            )
            pending_by_data[data_q] = _SZZ_FLOW_IDENTITY

        target.append("")
        target.append(f"{indent}# SZZ round {rnd_idx + 1}")
        for stab_type, stab_idx, data_q in layer_gates:
            axis = _szz_physical_axis_for_touch(stab_type, stab_idx, data_q)
            compose_data(data_q, _szz_axis_rotation_to_z_gates(axis))

        for stab_type, stab_idx, data_q in layer_gates:
            pending = (
                _SZZ_FLOW_IDENTITY
                if pending_by_data is None
                else pending_by_data.setdefault(data_q, _SZZ_FLOW_IDENTITY)
            )
            has_data_prefix = pending != _SZZ_FLOW_IDENTITY and not _szz_flow_is_virtual_z(pending)
            sign = szz_sign_by_touch[(stab_type, stab_idx, data_q)]
            host_gate = OpType.SZZ if sign > 0 else OpType.SZZDG
            host_label_core = f"r{rnd_idx + 1}:{stab_type}{stab_idx}:d{data_q}:{host_gate.name}"
            host_label = (
                f"szz:{host_label_core}" if host_label_scope is None else f"szz:{host_label_scope}:{host_label_core}"
            )
            if szz_runtime_barrier_policy == _SZZ_RUNTIME_BARRIER_POLICY_ALL or (
                szz_runtime_barrier_policy == _SZZ_RUNTIME_BARRIER_POLICY_DATA_PREFIX and has_data_prefix
            ):
                target.append(
                    f"{indent}{ancilla_expr(stab_type, stab_idx)}, {data_expr(data_q)} = "
                    "pecos_qis_runtime_barrier_qubits2_hugr("
                    f"{ancilla_expr(stab_type, stab_idx)}, {data_expr(data_q)})",
                )
            discharge_data_for_szz(data_q, host_label=host_label)
            half_turns = "0.5" if sign > 0 else "-0.5"
            _append_szz_gate_trace_metadata(
                target,
                indent,
                source_kind="szz_host",
                source_label=host_label,
                qubit_expr=data_expr(data_q),
                host_label=host_label,
                gate=host_gate,
                lowering_required=True,
            )
            target.append(
                f"{indent}{ancilla_expr(stab_type, stab_idx)}, {data_expr(data_q)} = "
                f"zz_phase({ancilla_expr(stab_type, stab_idx)}, {data_expr(data_q)}, angle({half_turns}))",
            )

        for stab_type, stab_idx, data_q in layer_gates:
            axis = _szz_physical_axis_for_touch(stab_type, stab_idx, data_q)
            compose_data(data_q, _szz_axis_rotation_from_z_gates(axis))

        for stab_type, stab_idx, data_q in layer_gates:
            sign = szz_sign_by_touch[(stab_type, stab_idx, data_q)]
            axis = _szz_physical_axis_for_touch(stab_type, stab_idx, data_q)
            compose_data(data_q, _szz_touch_compensation_gates(axis, sign))

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
        if interaction_basis == "szz":
            lines.append(
                f"def syndrome_extraction(surf: SurfaceCode_{dx}x{dz} @ owned) "
                f"-> tuple[SurfaceCode_{dx}x{dz}, Syndrome_{dx}x{dz}]:",
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
                (
                    '    """Extract full syndrome using 4-round parallel SZZ/SZZdg schedule."""'
                    if interaction_basis == "szz"
                    else '    """Extract full syndrome using 4-round parallel CNOT schedule."""'
                ),
            ],
        )
        if interaction_basis == "szz":
            lines.append("    # Unpack data qubits")
            _append_szz_data_unpack(lines, "    ")

        szz_syndrome_pending_by_data = (
            dict.fromkeys(range(num_data), _SZZ_FLOW_IDENTITY) if interaction_basis == "szz" else None
        )

        lines.append("    # Allocate ancilla qubits (one per stabilizer)")
        lines.extend(f"    ax{stab.index} = qubit()" for stab in geom.x_stabilizers)
        lines.extend(f"    az{stab.index} = qubit()" for stab in geom.z_stabilizers)

        if interaction_basis == "cx":
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
        else:
            lines.append("")
            lines.append("    # Hadamard on SZZ ancillas")
            lines.extend(f"    h(ax{stab.index})" for stab in geom.x_stabilizers)
            lines.extend(f"    h(az{stab.index})" for stab in geom.z_stabilizers)

            for rnd_idx, rnd_gates in enumerate(rounds):
                _append_szz_layer(
                    lines,
                    "    ",
                    rnd_idx,
                    list(rnd_gates),
                    _szz_ancilla_expr,
                    _szz_data_expr,
                    pending_by_data=szz_syndrome_pending_by_data,
                    host_label_scope="syndrome_extraction",
                )

            lines.append("")
            lines.append("    # Hadamard on SZZ ancillas")
            lines.extend(f"    h(ax{stab.index})" for stab in geom.x_stabilizers)
            lines.extend(f"    h(az{stab.index})" for stab in geom.z_stabilizers)

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
        # shared `batched_stabilizers(patch, effective_budget, schedule=...)` so the
        # abstract circuit's measurement order matches by construction.
        batches = batched_stabilizers(
            patch,
            effective_budget,
            ancilla_schedule=ancilla_schedule,
        )
        lines.append(
            f'    """Extract full syndrome in {len(batches)} ancilla-reuse batches (budget={effective_budget})."""',
        )
        if interaction_basis == "szz":
            lines.append("    # Unpack data qubits")
            _append_szz_data_unpack(lines, "    ")
        szz_syndrome_pending_by_data = (
            dict.fromkeys(range(num_data), _SZZ_FLOW_IDENTITY) if interaction_basis == "szz" else None
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

            if interaction_basis == "cx":
                x_in_batch = [(t, i) for (t, i) in batch if t == "X"]
                if x_in_batch:
                    lines.append("    # Hadamard on X ancillas in this batch")
                    for stab_type, stab_idx in x_in_batch:
                        lines.append(f"    h({batch_anc_var[(stab_type, stab_idx)]})")
            else:
                lines.append("    # Hadamard on SZZ ancillas in this batch")
                for stab_type, stab_idx in batch:
                    lines.append(f"    h({batch_anc_var[(stab_type, stab_idx)]})")

            # Filter the full interaction schedule to just this batch's stabilizers.
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
                if interaction_basis == "cx":
                    for stab_type, stab_idx, data_q in rnd_in_batch:
                        anc = batch_anc_var[(stab_type, stab_idx)]
                        if stab_type == "X":
                            lines.append(f"    cx({anc}, surf.data[{data_q}])")
                        else:
                            lines.append(f"    cx(surf.data[{data_q}], {anc})")
                else:
                    _append_szz_layer(
                        lines,
                        "    ",
                        rnd_idx,
                        rnd_in_batch,
                        lambda stab_type, stab_idx, batch_anc_var=batch_anc_var: batch_anc_var[(stab_type, stab_idx)],
                        _szz_data_expr,
                        pending_by_data=szz_syndrome_pending_by_data,
                        host_label_scope="syndrome_extraction",
                    )

            if interaction_basis == "cx":
                x_in_batch = [(t, i) for (t, i) in batch if t == "X"]
                if x_in_batch:
                    lines.append("")
                    lines.append("    # Hadamard on X ancillas in this batch")
                    for stab_type, stab_idx in x_in_batch:
                        lines.append(f"    h({batch_anc_var[(stab_type, stab_idx)]})")
            else:
                lines.append("")
                lines.append("    # Hadamard on SZZ ancillas in this batch")
                for stab_type, stab_idx in batch:
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

    lines.extend(["", f"    synx = array({x_calls})", f"    synz = array({z_calls})", ""])
    if interaction_basis == "szz":
        if szz_syndrome_pending_by_data is None:
            msg = "internal error: SZZ syndrome extraction did not initialize data-frame state"
            raise ValueError(msg)
        _append_szz_flush_data_frame(lines, "    ", szz_syndrome_pending_by_data, reason="syndrome return")
        lines.append(f"    surf = SurfaceCode_{dx}x{dz}({szz_data_args})")
        lines.append(f"    return surf, Syndrome_{dx}x{dz}(synx, synz)")
    else:
        lines.append(f"    return Syndrome_{dx}x{dz}(synx, synz)")
    lines.extend(["", ""])

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
                (
                    f"def {function_name}(surf: SurfaceCode_{dx}x{dz} @ owned) "
                    f"-> tuple[SurfaceCode_{dx}x{dz}, {return_type}]:"
                    if interaction_basis == "szz"
                    else f"def {function_name}(surf: SurfaceCode_{dx}x{dz}) -> {return_type}:"
                ),
                f'    """{doc}"""',
            ],
        )
        if interaction_basis == "szz":
            _append_szz_data_unpack(lines, "    ")
        szz_init_pending_by_data = (
            dict.fromkeys(range(num_data), _SZZ_FLOW_IDENTITY) if interaction_basis == "szz" else None
        )

        if not constrained:
            if stab_type == "X":
                lines.extend(f"    ax{stab.index} = qubit()" for stab in stabs)
            else:
                lines.extend(f"    az{stab.index} = qubit()" for stab in stabs)

            if interaction_basis == "cx":
                if stab_type == "X":
                    lines.append("")
                    lines.append("    # Hadamard on X ancillas")
                    lines.extend(f"    h(ax{stab.index})" for stab in stabs)

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
            else:
                lines.append("")
                lines.append("    # Hadamard on SZZ ancillas")
                if stab_type == "X":
                    lines.extend(f"    h(ax{stab.index})" for stab in stabs)
                else:
                    lines.extend(f"    h(az{stab.index})" for stab in stabs)

                for rnd_idx, rnd_gates in enumerate(rounds):
                    filtered = [(t, i, q) for t, i, q in rnd_gates if t == stab_type]
                    if not filtered:
                        continue
                    _append_szz_layer(
                        lines,
                        "    ",
                        rnd_idx,
                        filtered,
                        _szz_ancilla_expr,
                        _szz_data_expr,
                        pending_by_data=szz_init_pending_by_data,
                        host_label_scope=function_name,
                    )

                lines.append("")
                lines.append("    # Hadamard on SZZ ancillas")
                if stab_type == "X":
                    lines.extend(f"    h(ax{stab.index})" for stab in stabs)
                else:
                    lines.extend(f"    h(az{stab.index})" for stab in stabs)

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
            batches = batched_stabilizers(
                patch,
                effective_budget,
                ancilla_schedule=ancilla_schedule,
            )
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

                if interaction_basis == "cx":
                    if stab_type == "X":
                        lines.append("    # Hadamard on X ancillas in this batch")
                        for selected_type, stab_idx in init_batch:
                            lines.append(f"    h({batch_anc_var[(selected_type, stab_idx)]})")
                else:
                    lines.append("    # Hadamard on SZZ ancillas in this batch")
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
                    if interaction_basis == "cx":
                        for selected_type, stab_idx, data_q in rnd_in_batch:
                            anc = batch_anc_var[(selected_type, stab_idx)]
                            if selected_type == "X":
                                lines.append(f"    cx({anc}, surf.data[{data_q}])")
                            else:
                                lines.append(f"    cx(surf.data[{data_q}], {anc})")
                    else:
                        _append_szz_layer(
                            lines,
                            "    ",
                            rnd_idx,
                            rnd_in_batch,
                            lambda selected_type, stab_idx, batch_anc_var=batch_anc_var: batch_anc_var[
                                (selected_type, stab_idx)
                            ],
                            _szz_data_expr,
                            pending_by_data=szz_init_pending_by_data,
                            host_label_scope=function_name,
                        )

                if interaction_basis == "cx":
                    if stab_type == "X":
                        lines.append("")
                        lines.append("    # Hadamard on X ancillas in this batch")
                        for selected_type, stab_idx in init_batch:
                            lines.append(f"    h({batch_anc_var[(selected_type, stab_idx)]})")
                else:
                    lines.append("")
                    lines.append("    # Hadamard on SZZ ancillas in this batch")
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

        lines.append("")
        if interaction_basis == "szz":
            if szz_init_pending_by_data is None:
                msg = "internal error: SZZ init helper did not initialize data-frame state"
                raise ValueError(msg)
            _append_szz_flush_data_frame(lines, "    ", szz_init_pending_by_data, reason=f"{function_name} return")
            lines.append(f"    surf = SurfaceCode_{dx}x{dz}({szz_data_args})")
            lines.append(f"    return surf, array({return_calls})")
        else:
            lines.append(f"    return array({return_calls})")

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
        ],
    )
    if interaction_basis == "szz":
        _append_szz_data_unpack(lines, "    ")
        for i in range(num_data):
            _append_szz_axis_rotation_from_z(
                lines,
                "    ",
                _szz_physical_axis_for_memory_data("Z", i),
                f"d{i}",
            )
        z_meas = ", ".join(f"measure(d{i})" for i in range(num_data))
        lines.append(f"    return array({z_meas})")
    else:
        lines.append("    return measure_array(surf.data)")
    lines.extend(
        [
            "",
            "",
            "@guppy",
            f"def measure_x_basis(surf: SurfaceCode_{dx}x{dz} @ owned) -> array[bool, {num_data}]:",
            '    """Destructively measure in X basis."""',
        ],
    )
    if interaction_basis == "szz":
        _append_szz_data_unpack(lines, "    ")
        for i in range(num_data):
            _append_szz_axis_rotation_from_z(
                lines,
                "    ",
                _szz_physical_axis_for_memory_data("X", i),
                f"d{i}",
            )
        x_meas = ", ".join(f"measure(d{i})" for i in range(num_data))
        lines.append(f"    return array({x_meas})")
    else:
        lines.append(f"    for i in range({num_data}):")
        lines.append("        h(surf.data[i])")
        lines.append("    return measure_array(surf.data)")
    lines.extend(["", ""])

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
    if interaction_basis == "szz":
        for q in logical_x_qubits:
            logical_x_axis = _szz_physical_axis_for_logical_data("X", q)
            _append_szz_logical_pauli(lines, "    ", logical_x_axis, f"surf.d{q}")
    else:
        lines.extend(f"    x(surf.data[{q}])" for q in logical_x_qubits)

    lines.extend(
        [
            "",
            "",
            "@guppy",
            f"def apply_logical_z(surf: SurfaceCode_{dx}x{dz}) -> None:",
            '    """Apply logical Z (string along top edge)."""',
        ],
    )
    if interaction_basis == "szz":
        for q in logical_z_qubits:
            logical_z_axis = _szz_physical_axis_for_logical_data("Z", q)
            _append_szz_logical_pauli(lines, "    ", logical_z_axis, f"surf.d{q}")
    else:
        lines.extend(f"    z(surf.data[{q}])" for q in logical_z_qubits)

    lines.extend(
        [
            "",
            "",
        ],
    )

    def _append_inline_szz_syndrome_extraction(
        target: list[str],
        indent: str,
        *,
        host_label_scope: str,
    ) -> None:
        """Append one SZZ syndrome-extraction body with unique hosted ids."""
        if interaction_basis != "szz":
            msg = "inline SZZ syndrome extraction requires interaction_basis='szz'"
            raise ValueError(msg)
        target.append(f"{indent}# Inline SZZ syndrome extraction ({host_label_scope})")
        target.append(f"{indent}# Unpack data qubits")
        _append_szz_data_unpack(target, indent)
        pending_by_data = dict.fromkeys(range(num_data), _SZZ_FLOW_IDENTITY)

        if not constrained:
            target.append(f"{indent}# Allocate ancilla qubits (one per stabilizer)")
            target.extend(f"{indent}ax{stab.index} = qubit()" for stab in geom.x_stabilizers)
            target.extend(f"{indent}az{stab.index} = qubit()" for stab in geom.z_stabilizers)

            target.append("")
            target.append(f"{indent}# Hadamard on SZZ ancillas")
            target.extend(f"{indent}h(ax{stab.index})" for stab in geom.x_stabilizers)
            target.extend(f"{indent}h(az{stab.index})" for stab in geom.z_stabilizers)

            for rnd_idx, rnd_gates in enumerate(rounds):
                _append_szz_layer(
                    target,
                    indent,
                    rnd_idx,
                    list(rnd_gates),
                    _szz_ancilla_expr,
                    _szz_data_expr,
                    pending_by_data=pending_by_data,
                    host_label_scope=host_label_scope,
                )

            target.append("")
            target.append(f"{indent}# Hadamard on SZZ ancillas")
            target.extend(f"{indent}h(ax{stab.index})" for stab in geom.x_stabilizers)
            target.extend(f"{indent}h(az{stab.index})" for stab in geom.z_stabilizers)

            target.append("")
            target.append(f"{indent}# Measure ancillas")
            idx = 0
            for stab in geom.x_stabilizers:
                target.append(f"{indent}sx{stab.index} = measure(ax{stab.index})")
                target.append(f'{indent}result("sx{stab.index}:meas:{idx}", sx{stab.index})')
                idx += 1
            for stab in geom.z_stabilizers:
                target.append(f"{indent}sz{stab.index} = measure(az{stab.index})")
                target.append(f'{indent}result("sz{stab.index}:meas:{idx}", sz{stab.index})')
                idx += 1
        else:
            batches = batched_stabilizers(
                patch,
                effective_budget,
                ancilla_schedule=ancilla_schedule,
            )
            idx = 0
            for batch_idx, batch in enumerate(batches):
                target.append("")
                target.append(f"{indent}# Batch {batch_idx + 1}/{len(batches)} of stabilizers")
                batch_anc_var: dict[tuple[str, int], str] = {}
                for pos, (stab_type, stab_idx) in enumerate(batch):
                    var = f"_a_b{batch_idx}_p{pos}"
                    batch_anc_var[(stab_type, stab_idx)] = var
                    target.append(f"{indent}{var} = qubit()")

                target.append(f"{indent}# Hadamard on SZZ ancillas in this batch")
                for stab_type, stab_idx in batch:
                    target.append(f"{indent}h({batch_anc_var[(stab_type, stab_idx)]})")

                batch_keys = set(batch_anc_var.keys())
                for rnd_idx, rnd_gates in enumerate(rounds):
                    rnd_in_batch = [
                        (stab_type, stab_idx, data_q)
                        for stab_type, stab_idx, data_q in rnd_gates
                        if (stab_type, stab_idx) in batch_keys
                    ]
                    if not rnd_in_batch:
                        continue
                    target.append("")
                    target.append(f"{indent}# Batch {batch_idx + 1} round {rnd_idx + 1}")
                    _append_szz_layer(
                        target,
                        indent,
                        rnd_idx,
                        rnd_in_batch,
                        lambda stab_type, stab_idx, batch_anc_var=batch_anc_var: batch_anc_var[(stab_type, stab_idx)],
                        _szz_data_expr,
                        pending_by_data=pending_by_data,
                        host_label_scope=host_label_scope,
                    )

                target.append("")
                target.append(f"{indent}# Hadamard on SZZ ancillas in this batch")
                for stab_type, stab_idx in batch:
                    target.append(f"{indent}h({batch_anc_var[(stab_type, stab_idx)]})")

                target.append("")
                target.append(f"{indent}# Measure batch {batch_idx + 1} ancillas")
                for stab_type, stab_idx in batch:
                    anc = batch_anc_var[(stab_type, stab_idx)]
                    syn_var = f"sx{stab_idx}" if stab_type == "X" else f"sz{stab_idx}"
                    target.append(f"{indent}{syn_var} = measure({anc})")
                    target.append(f'{indent}result("{syn_var}:meas:{idx}", {syn_var})')
                    idx += 1

        x_calls = ", ".join(f"sx{s.index}" for s in geom.x_stabilizers)
        z_calls = ", ".join(f"sz{s.index}" for s in geom.z_stabilizers)
        target.extend(["", f"{indent}synx = array({x_calls})", f"{indent}synz = array({z_calls})", ""])
        _append_szz_flush_data_frame(
            target,
            indent,
            pending_by_data,
            reason=f"{host_label_scope} inline syndrome",
        )
        target.append(f"{indent}surf = SurfaceCode_{dx}x{dz}({szz_data_args})")

    def _render_plain_szz_round_helper(round_idx: int) -> list[str]:
        helper_name = f"syndrome_extraction_memory_r{round_idx}"
        body = [
            "",
            "",
            "@guppy",
            (
                f"def {helper_name}(surf: SurfaceCode_{dx}x{dz} @ owned) "
                f"-> tuple[SurfaceCode_{dx}x{dz}, Syndrome_{dx}x{dz}]:"
            ),
            f'    """Extract counted SZZ syndrome round {round_idx} with round-scoped hosted metadata."""',
        ]
        _append_inline_szz_syndrome_extraction(
            body,
            "    ",
            host_label_scope=f"memory_r{round_idx}",
        )
        body.append(f"    return surf, Syndrome_{dx}x{dz}(synx, synz)")
        return body

    def _render_plain_szz_memory_block(
        basis: str,
        basis_upper: str,
        rendered_num_rounds: int,
    ) -> list[str]:
        init_func = "init_z_basis" if basis == "z" else "init_x_basis"
        init_tag = "init_synx" if basis == "z" else "init_synz"
        body: list[str] = [
            (
                f'        """{basis_upper}-basis SZZ memory experiment for '
                f"dx={dx}, dz={dz}, num_rounds={rendered_num_rounds}."
                '"""'
            ),
            f"        surf = prep_{basis}_basis()",
            f"        surf, init_syn = {init_func}(surf)",
            f'        result("{init_tag}", init_syn)',
            "",
        ]
        for round_idx in range(rendered_num_rounds):
            body.append(f"        # === Counted syndrome round {round_idx} ===")
            body.append(f"        surf, syn = syndrome_extraction_memory_r{round_idx}(surf)")
            body.append('        result("synx", syn.synx)')
            body.append('        result("synz", syn.synz)')
            body.append("")

        body.extend(
            [
                f"        final = measure_{basis}_basis(surf)",
                '        result("final", final)',
            ],
        )
        return [
            f"def make_memory_{basis}(num_rounds: int):",
            f'    """Create {basis_upper}-basis SZZ memory experiment.',
            "",
            f"    num_rounds must equal {rendered_num_rounds} -- the body was unrolled at",
            "    source-generation time so hosted-operation metadata is unique per",
            "    counted syndrome round. Mismatched values raise ValueError.",
            '    """',
            f"    if num_rounds != {rendered_num_rounds}:",
            (
                f'        msg = f"this generated module was unrolled for '
                f'num_rounds={rendered_num_rounds}, got {{num_rounds!r}}"'
            ),
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

    # Generate memory experiment factories.  Plain SZZ memory programs are
    # unrolled when the round count is available so hosted metadata identifies
    # the concrete counted syndrome round instead of the reusable helper body.
    if twirl is None and interaction_basis == "szz" and num_rounds is not None:
        lines.extend(["# === Counted SZZ Syndrome Helpers ==="])
        for round_idx in range(num_rounds):
            lines.extend(_render_plain_szz_round_helper(round_idx))
        lines.extend(["# === Memory Experiments ===", ""])
        for basis, basis_upper in (("z", "Z"), ("x", "X")):
            lines.extend(_render_plain_szz_memory_block(basis, basis_upper, num_rounds))
    else:
        lines.extend(_render_memory_experiments(patch, dx, dz, num_data, twirl, rng, num_rounds, interaction_basis))

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
    patch: "SurfacePatch",
    dx: int,
    dz: int,
    num_data: int,
    twirl: "TwirlConfig | None",
    rng: "GuppyRngMaskConfig | None",
    num_rounds: int | None,
    interaction_basis: str,
) -> list[str]:
    """Render both memory factory functions."""
    lines = [
        "# === Memory Experiments ===",
        "",
    ]
    for basis, basis_upper in (("z", "Z"), ("x", "X")):
        if twirl is None:
            lines.extend(_render_plain_memory_block(basis, basis_upper, dx, dz, interaction_basis))
        else:
            if rng is None or num_rounds is None:
                msg = "twirled memory rendering requires both rng and num_rounds"
                raise ValueError(msg)
            if twirl.site_schedule == "before_two_qubit_gate":
                lines.extend(
                    _render_gate_local_twirled_memory_block(
                        patch,
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
            else:
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
    interaction_basis: str,
) -> list[str]:
    """Render the vanilla handoff memory factory for one basis."""
    init_func = "init_z_basis" if basis == "z" else "init_x_basis"
    init_tag = "init_synx" if basis == "z" else "init_synz"
    if interaction_basis == "szz":
        init_line = f"        surf, init_syn = {init_func}(surf)"
        syndrome_line = "            surf, syn = syndrome_extraction(surf)"
    else:
        init_line = f"        init_syn = {init_func}(surf)"
        syndrome_line = "            syn = syndrome_extraction(surf)"
    return [
        f"def make_memory_{basis}(num_rounds: int):",
        f'    """Create {basis_upper}-basis memory experiment."""',
        "    from guppylang.std.builtins import comptime",
        "",
        "    @guppy",
        f"    def memory_{basis}() -> None:",
        f'        """{basis_upper}-basis memory experiment for dx={dx}, dz={dz}."""',
        f"        surf = prep_{basis}_basis()",
        init_line,
        f'        result("{init_tag}", init_syn)',
        "",
        "        for _t in range(comptime(num_rounds)):",
        syndrome_line,
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


def _twirl_activation_threshold(twirl: "TwirlConfig") -> int:
    """Return the full-width PCG threshold for a twirl activation probability."""
    probability = float(twirl.twirl_probability)
    # Validation lives on TwirlConfig; clamp the mathematically exact endpoint
    # after float multiplication so f=1.0 always activates every 32-bit draw.
    threshold = int(probability * (1 << 32))
    return min(max(threshold, 0), 1 << 32)


def _emit_scaled_activation_tags(twirl: "TwirlConfig") -> bool:
    """Whether generated source needs explicit activation side-band tags."""
    return float(twirl.twirl_probability) != 1.0


def _append_pauli_draw(
    lines: list[str],
    indent: str,
    *,
    active_var: str,
    draw_var: str,
    m_var: str,
    qubit_expr: str,
    threshold: int,
) -> None:
    """Emit fixed-consumption activation + Pauli draw code for one twirl operand."""
    lines.append(f"{indent}rng_state, active_draw_{m_var} = _pcg32_next32(rng_state, rng_inc)")
    lines.append(f"{indent}{active_var} = active_draw_{m_var} < nat({threshold})")
    lines.append(f"{indent}rng_state, {draw_var} = _pcg32_next4(rng_state, rng_inc)")
    lines.append(f"{indent}{m_var} = 0")
    lines.append(f"{indent}if {active_var}:")
    lines.append(f"{indent}    {m_var} = {draw_var}")
    lines.append(f"{indent}if {m_var} == 1:")
    lines.append(f"{indent}    x({qubit_expr})")
    lines.append(f"{indent}if {m_var} == 2:")
    lines.append(f"{indent}    y({qubit_expr})")
    lines.append(f"{indent}if {m_var} == 3:")
    lines.append(f"{indent}    z({qubit_expr})")


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
    from pecos.qec.surface._twirl_sites import num_twirl_sites, pauli_active_round_tag, pauli_mask_round_tag

    seed = int(rng.seed)
    canonical_frame_output = twirl.frame_output == "canonical"
    activation_threshold = _twirl_activation_threshold(twirl)
    emit_activation_tags = _emit_scaled_activation_tags(twirl)
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
            _append_pauli_draw(
                body,
                "        ",
                active_var=f"active_{r}_{q}",
                draw_var=f"m_draw_{r}_{q}",
                m_var=f"m_{r}_{q}",
                qubit_expr=f"surf.data[{q}]",
                threshold=activation_threshold,
            )
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
        if emit_activation_tags:
            active_elements = ", ".join(f"active_{r}_{q}" for q in range(num_data))
            active_tag = pauli_active_round_tag(r)
            body.append(f'        result("{active_tag}", array({active_elements}))')
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


def _frame_vars(prefix: str, idx: int) -> tuple[str, str]:
    return f"frame_x_{prefix}{idx}", f"frame_z_{prefix}{idx}"


def _append_frame_swap(lines: list[str], indent: str, x_var: str, z_var: str, tmp_var: str) -> None:
    lines.append(f"{indent}{tmp_var} = {x_var}")
    lines.append(f"{indent}{x_var} = {z_var}")
    lines.append(f"{indent}{z_var} = {tmp_var}")


def _append_gate_local_draw(
    lines: list[str],
    indent: str,
    *,
    site_idx: int,
    operand_idx: int,
    qubit_expr: str,
    frame_vars: tuple[str, str] | None,
    threshold: int,
) -> tuple[str, str, str]:
    m_var = f"m_g{site_idx}_o{operand_idx}"
    active_var = f"active_g{site_idx}_o{operand_idx}"
    draw_var = f"m_draw_g{site_idx}_o{operand_idx}"
    lo_var = f"lo_g{site_idx}_o{operand_idx}"
    hi_var = f"hi_g{site_idx}_o{operand_idx}"
    _append_pauli_draw(
        lines,
        indent,
        active_var=active_var,
        draw_var=draw_var,
        m_var=m_var,
        qubit_expr=qubit_expr,
        threshold=threshold,
    )
    lines.append(f"{indent}{lo_var} = ({m_var} == 1) | ({m_var} == 3)")
    lines.append(f"{indent}{hi_var} = ({m_var} == 2) | ({m_var} == 3)")
    if frame_vars is not None:
        x_frame, z_frame = frame_vars
        twx_var = f"twx_g{site_idx}_o{operand_idx}"
        twz_var = f"twz_g{site_idx}_o{operand_idx}"
        lines.append(f"{indent}{twx_var} = ({m_var} == 1) | ({m_var} == 2)")
        lines.append(f"{indent}{twz_var} = ({m_var} == 2) | ({m_var} == 3)")
        lines.append(f"{indent}{x_frame} = {x_frame} != {twx_var}")
        lines.append(f"{indent}{z_frame} = {z_frame} != {twz_var}")
    return lo_var, hi_var, active_var


def _append_gate_local_layer(
    lines: list[str],
    indent: str,
    *,
    site_idx: int,
    cx_ops: list[tuple[str, str, tuple[str, str] | None, tuple[str, str] | None]],
    threshold: int,
    emit_activation_tags: bool,
) -> int:
    """Emit all twirl draws before a parallel CX layer, then the CX layer."""
    from pecos.qec.surface._twirl_sites import pauli_active_gate_tag, pauli_mask_gate_tag

    for control_expr, target_expr, control_frame, target_frame in cx_ops:
        lo0, hi0, active0 = _append_gate_local_draw(
            lines,
            indent,
            site_idx=site_idx,
            operand_idx=0,
            qubit_expr=control_expr,
            frame_vars=control_frame,
            threshold=threshold,
        )
        lo1, hi1, active1 = _append_gate_local_draw(
            lines,
            indent,
            site_idx=site_idx,
            operand_idx=1,
            qubit_expr=target_expr,
            frame_vars=target_frame,
            threshold=threshold,
        )
        tag = pauli_mask_gate_tag(site_idx)
        lines.append(f'{indent}result("{tag}", array({lo0}, {hi0}, {lo1}, {hi1}))')
        if emit_activation_tags:
            active_tag = pauli_active_gate_tag(site_idx)
            lines.append(f'{indent}result("{active_tag}", array({active0}, {active1}))')
        site_idx += 1

    for control_expr, target_expr, control_frame, target_frame in cx_ops:
        lines.append(f"{indent}cx({control_expr}, {target_expr})")
        if control_frame is not None and target_frame is not None:
            control_x, control_z = control_frame
            target_x, target_z = target_frame
            lines.append(f"{indent}{target_x} = {target_x} != {control_x}")
            lines.append(f"{indent}{control_z} = {control_z} != {target_z}")
    return site_idx


def _append_gate_local_measure(
    lines: list[str],
    indent: str,
    *,
    bit_var: str,
    qubit_expr: str,
    result_tag: str,
    raw_tag: str,
    frame_vars: tuple[str, str] | None,
) -> None:
    if frame_vars is None:
        lines.append(f"{indent}{bit_var} = measure({qubit_expr})")
    else:
        raw_var = f"{bit_var}_raw"
        frame_x, _frame_z = frame_vars
        lines.append(f"{indent}{raw_var} = measure({qubit_expr})")
        lines.append(f"{indent}{bit_var} = {raw_var} != {frame_x}")
        lines.append(f'{indent}result("{raw_tag}", {raw_var})')
    lines.append(f'{indent}result("{result_tag}", {bit_var})')


def _render_gate_local_twirled_memory_block(
    patch: "SurfacePatch",
    basis: str,
    basis_upper: str,
    dx: int,
    dz: int,
    num_data: int,
    twirl: "TwirlConfig",
    rng: "GuppyRngMaskConfig",
    num_rounds: int,
) -> list[str]:
    """Render a gate-local twirled memory factory."""
    seed = int(rng.seed)
    canonical_frame_output = twirl.frame_output == "canonical"
    activation_threshold = _twirl_activation_threshold(twirl)
    emit_activation_tags = _emit_scaled_activation_tags(twirl)
    geom = patch.geometry
    rounds = compute_cnot_schedule(patch)
    init_stab_type = "X" if basis == "z" else "Z"
    init_tag = "init_synx" if basis == "z" else "init_synz"
    indent = "        "
    site_idx = 0

    data_frames = [_frame_vars("d", q) for q in range(num_data)]
    x_anc_frames = [_frame_vars("ax", stab.index) for stab in geom.x_stabilizers]
    z_anc_frames = [_frame_vars("az", stab.index) for stab in geom.z_stabilizers]

    def data_expr(q: int) -> str:
        return f"surf.data[{q}]"

    def anc_expr(stab_type: str, stab_idx: int) -> str:
        return f"ax{stab_idx}" if stab_type == "X" else f"az{stab_idx}"

    def anc_frame(stab_type: str, stab_idx: int) -> tuple[str, str] | None:
        if not canonical_frame_output:
            return None
        return x_anc_frames[stab_idx] if stab_type == "X" else z_anc_frames[stab_idx]

    def data_frame(q: int) -> tuple[str, str] | None:
        return data_frames[q] if canonical_frame_output else None

    def cx_tuple(
        stab_type: str,
        stab_idx: int,
        data_q: int,
    ) -> tuple[str, str, tuple[str, str] | None, tuple[str, str] | None]:
        if stab_type == "X":
            return (
                anc_expr(stab_type, stab_idx),
                data_expr(data_q),
                anc_frame(stab_type, stab_idx),
                data_frame(data_q),
            )
        return (
            data_expr(data_q),
            anc_expr(stab_type, stab_idx),
            data_frame(data_q),
            anc_frame(stab_type, stab_idx),
        )

    body: list[str] = [
        f'        """{basis_upper}-basis memory experiment for dx={dx}, dz={dz}, '
        f'num_rounds={num_rounds} (gate-local twirled)."""',
        f"        surf = prep_{basis}_basis()",
        "        # RNG seed is structural -- changing it does not invalidate the",
        "        # abstract DEM / topology cache, only the per-shot mask buffer.",
        f"        rng_state, rng_inc = seeded_pcg32_with_quantum_entropy({seed})",
        f'        result("frame_mode:{twirl.frame_output}", True)',
    ]
    if canonical_frame_output:
        for q in range(num_data):
            frame_x, frame_z = data_frames[q]
            body.append(f"{indent}{frame_x} = False")
            body.append(f"{indent}{frame_z} = False")
        body.append("")

    # Initial syndrome establishment for the complementary stabilizer family.
    init_stabs = list(geom.x_stabilizers if init_stab_type == "X" else geom.z_stabilizers)
    for stab in init_stabs:
        body.append(f"{indent}{anc_expr(init_stab_type, stab.index)} = qubit()")
        if canonical_frame_output:
            frame_x, frame_z = anc_frame(init_stab_type, stab.index) or ("", "")
            body.append(f"{indent}{frame_x} = False")
            body.append(f"{indent}{frame_z} = False")
    if init_stab_type == "X":
        body.append("")
        for stab in init_stabs:
            body.append(f"{indent}h(ax{stab.index})")
            if canonical_frame_output:
                frame_x, frame_z = anc_frame("X", stab.index) or ("", "")
                _append_frame_swap(body, indent, frame_x, frame_z, f"tmp_h_init_ax{stab.index}")

    for rnd_idx, rnd_gates in enumerate(rounds):
        filtered = [(t, i, q) for t, i, q in rnd_gates if t == init_stab_type]
        if not filtered:
            continue
        body.append("")
        body.append(f"{indent}# Init CX round {rnd_idx + 1}")
        cx_ops = [cx_tuple(stab_type, stab_idx, data_q) for stab_type, stab_idx, data_q in filtered]
        site_idx = _append_gate_local_layer(
            body,
            indent,
            site_idx=site_idx,
            cx_ops=cx_ops,
            threshold=activation_threshold,
            emit_activation_tags=emit_activation_tags,
        )

    if init_stab_type == "X":
        body.append("")
        for stab in init_stabs:
            body.append(f"{indent}h(ax{stab.index})")
            if canonical_frame_output:
                frame_x, frame_z = anc_frame("X", stab.index) or ("", "")
                _append_frame_swap(body, indent, frame_x, frame_z, f"tmp_h_init2_ax{stab.index}")

    body.append("")
    init_bits: list[str] = []
    for idx, stab in enumerate(init_stabs):
        bit_var = f"s{init_stab_type.lower()}{stab.index}_init"
        init_bits.append(bit_var)
        _append_gate_local_measure(
            body,
            indent,
            bit_var=bit_var,
            qubit_expr=anc_expr(init_stab_type, stab.index),
            result_tag=f"s{init_stab_type.lower()}{stab.index}:init:meas:{idx}",
            raw_tag=f"raw:s{init_stab_type.lower()}{stab.index}:init:bit:{idx}",
            frame_vars=anc_frame(init_stab_type, stab.index),
        )
    body.append(f'{indent}result("{init_tag}", array({", ".join(init_bits)}))')
    body.append("")

    for round_idx in range(num_rounds):
        body.append(f"{indent}# === Round {round_idx} (gate-local twirled) ===")
        for stab in geom.x_stabilizers:
            body.append(f"{indent}ax{stab.index} = qubit()")
            if canonical_frame_output:
                frame_x, frame_z = anc_frame("X", stab.index) or ("", "")
                body.append(f"{indent}{frame_x} = False")
                body.append(f"{indent}{frame_z} = False")
        for stab in geom.z_stabilizers:
            body.append(f"{indent}az{stab.index} = qubit()")
            if canonical_frame_output:
                frame_x, frame_z = anc_frame("Z", stab.index) or ("", "")
                body.append(f"{indent}{frame_x} = False")
                body.append(f"{indent}{frame_z} = False")
        for stab in geom.x_stabilizers:
            body.append(f"{indent}h(ax{stab.index})")
            if canonical_frame_output:
                frame_x, frame_z = anc_frame("X", stab.index) or ("", "")
                _append_frame_swap(body, indent, frame_x, frame_z, f"tmp_h_r{round_idx}_ax{stab.index}")

        for rnd_idx, rnd_gates in enumerate(rounds):
            body.append("")
            body.append(f"{indent}# Round {round_idx} CX layer {rnd_idx + 1}")
            cx_ops = [cx_tuple(stab_type, stab_idx, data_q) for stab_type, stab_idx, data_q in rnd_gates]
            site_idx = _append_gate_local_layer(
                body,
                indent,
                site_idx=site_idx,
                cx_ops=cx_ops,
                threshold=activation_threshold,
                emit_activation_tags=emit_activation_tags,
            )

        for stab in geom.x_stabilizers:
            body.append(f"{indent}h(ax{stab.index})")
            if canonical_frame_output:
                frame_x, frame_z = anc_frame("X", stab.index) or ("", "")
                _append_frame_swap(body, indent, frame_x, frame_z, f"tmp_h2_r{round_idx}_ax{stab.index}")

        sx_bits: list[str] = []
        sz_bits: list[str] = []
        meas_idx = 0
        for stab in geom.x_stabilizers:
            bit_var = f"sx{stab.index}_r{round_idx}"
            sx_bits.append(bit_var)
            _append_gate_local_measure(
                body,
                indent,
                bit_var=bit_var,
                qubit_expr=f"ax{stab.index}",
                result_tag=f"sx{stab.index}:meas:{meas_idx}",
                raw_tag=f"raw:sx{stab.index}:bit:{meas_idx}",
                frame_vars=anc_frame("X", stab.index),
            )
            meas_idx += 1
        for stab in geom.z_stabilizers:
            bit_var = f"sz{stab.index}_r{round_idx}"
            sz_bits.append(bit_var)
            _append_gate_local_measure(
                body,
                indent,
                bit_var=bit_var,
                qubit_expr=f"az{stab.index}",
                result_tag=f"sz{stab.index}:meas:{meas_idx}",
                raw_tag=f"raw:sz{stab.index}:bit:{meas_idx}",
                frame_vars=anc_frame("Z", stab.index),
            )
            meas_idx += 1
        body.append(f'{indent}result("synx", array({", ".join(sx_bits)}))')
        body.append(f'{indent}result("synz", array({", ".join(sz_bits)}))')
        body.append("")

    if canonical_frame_output:
        if basis == "x":
            for q in range(num_data):
                body.append(f"{indent}h(surf.data[{q}])")
                frame_x, frame_z = data_frames[q]
                _append_frame_swap(body, indent, frame_x, frame_z, f"tmp_h_final_d{q}")
        body.append(f"{indent}final_raw = measure_array(surf.data)")
        body.append(f'{indent}result("raw:final", final_raw)')
        for q in range(num_data):
            frame_x, _frame_z = data_frames[q]
            body.append(f"{indent}final_{q} = final_raw[{q}] != {frame_x}")
        body.append(f'{indent}result("final", array({", ".join(f"final_{q}" for q in range(num_data))}))')
    else:
        body.append(f"{indent}final = measure_{basis}_basis(surf)")
        body.append(f'{indent}result("final", final)')

    return [
        f"def make_memory_{basis}(num_rounds: int):",
        f'    """Create {basis_upper}-basis gate-local twirled memory experiment.',
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
    *,
    twirl: "TwirlConfig | None" = None,
    rng: "GuppyRngMaskConfig | None" = None,
    num_rounds: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
) -> str:
    """Filesystem-safe cache key spanning full patch identity + budget + twirl.

    Mirrors the topology identity used by the native cache
    (``decode._surface_patch_cache_key``): dx, dz, orientation, and the
    rotated flag. Keying on distance/dx-dz alone would collide a rotated and
    a non-rotated patch of the same shape onto one generated module.

    Memory factories close over ``num_rounds`` as a Guppy comptime value, so
    the cache key includes it whenever a memory program is requested. This
    keeps Guppy's module-qualified function names round-specific instead of
    reusing a stale compiled body from an earlier factory call. Twirled source
    and plain SZZ source also Python-time unroll the round body, so they require
    this keying for source identity as well.
    """
    geom = patch.geometry
    rotated = "rot" if geom.rotated else "unrot"
    resolved_plan = _resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    interaction_basis = resolved_plan.interaction_basis
    szz_runtime_barrier_policy = _normalize_szz_runtime_barrier_policy(szz_runtime_barriers)
    if interaction_basis != "szz" and szz_runtime_barrier_policy != _SZZ_RUNTIME_BARRIER_POLICY_NONE:
        msg = "szz_runtime_barriers is only supported for interaction_basis='szz'"
        raise ValueError(msg)
    interaction_part = "" if interaction_basis == "cx" else f"_ib{interaction_basis}"
    check_plan_part = "" if check_plan is None else f"_cp{resolved_plan.plan_id}"
    frame_part = "" if clifford_frame_policy is None else f"_cf{str(clifford_frame_policy).lower().replace('-', '_')}"
    runtime_barrier_part = (
        "" if szz_runtime_barrier_policy == _SZZ_RUNTIME_BARRIER_POLICY_NONE else f"_szzrb-{szz_runtime_barrier_policy}"
    )
    trace_metadata_part = "" if trace_metadata else "_trace-metadata-off"
    base = (
        f"{patch.dx}x{patch.dz}_{geom.orientation.name}_{rotated}"
        f"_b{effective_budget}{interaction_part}{check_plan_part}{frame_part}"
        f"{runtime_barrier_part}{trace_metadata_part}"
    )
    if twirl is None:
        if num_rounds is not None:
            return f"{base}_r{int(num_rounds)}"
        return base
    if rng is None or num_rounds is None:
        msg = "twirled Guppy module cache keys require both rng and num_rounds"
        raise ValueError(msg)
    twirl_part = (
        f"t-{twirl.scheme}-{twirl.site_schedule}-{twirl.result_encoding}"
        f"-frame-{twirl.frame_output}"
        f"-p{_twirl_activation_threshold(twirl)}"
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
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
) -> dict:
    """Load a Guppy module for a patch, using caching.

    The cache key spans the full patch identity (dx, dz, orientation,
    rotated) and the **effective** budget (after clamping via
    ``normalize_ancilla_budget``), so ``ancilla_budget=None`` and
    ``ancilla_budget >= total_ancilla`` resolve to the same cache entry
    while distinct patch geometries never collide. Twirled source also
    keys on twirl fields, frame-output mode, activation-probability threshold,
    RNG seed, round count, and SZZ runtime-barrier policy.

    Args:
        patch: SurfacePatch with geometry
        ancilla_budget: Optional cap on simultaneously live ancillas
        twirl: Pauli-twirl-site declaration (structural)
        rng: Runtime mask RNG seed (must be supplied with ``twirl``)
        num_rounds: Syndrome-round count for unrolled twirled source.
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ/SZZdg surface-code generation.
        szz_runtime_barriers: SZZ/SZZdg scheduling-barrier policy.
        trace_metadata: Emit PECOS trace metadata helpers in generated SZZ
            source. Keep enabled for traced-QIS/DEM paths; disable for
            execution-only builds whose compiler/linker cannot resolve the
            metadata helper symbols.

    Returns:
        Module dictionary with generated functions
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    resolved_plan = _resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    interaction_basis = resolved_plan.interaction_basis
    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    cache_key = _guppy_module_cache_key(
        patch,
        effective_budget,
        twirl=twirl,
        rng=rng,
        num_rounds=num_rounds,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id if check_plan is not None else None,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        trace_metadata=trace_metadata,
    )

    if cache_key in _state.module_cache:
        return _state.module_cache[cache_key]

    source = generate_guppy_source(
        patch,
        ancilla_budget=ancilla_budget,
        twirl=twirl,
        rng=rng,
        num_rounds=num_rounds,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        trace_metadata=trace_metadata,
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
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
) -> object:
    """Generate a memory experiment for a patch.

    Args:
        patch: SurfacePatch configuration
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas
        twirl: Pauli-twirl-site declaration; must be supplied with ``rng``.
        rng: Runtime mask RNG seed; must be supplied with ``twirl``.
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ/SZZdg surface-code generation.
        szz_runtime_barriers: SZZ/SZZdg scheduling-barrier policy.
        trace_metadata: Emit PECOS trace metadata helpers in generated SZZ
            source. Keep enabled for traced-QIS/DEM paths; disable for
            execution-only builds whose compiler/linker cannot resolve the
            metadata helper symbols.

    Returns:
        Guppy function for the experiment
    """
    resolved_plan = _resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    module = _load_guppy_module(
        patch,
        ancilla_budget=ancilla_budget,
        twirl=twirl,
        rng=rng,
        num_rounds=num_rounds,
        interaction_basis=resolved_plan.interaction_basis,
        check_plan=resolved_plan.plan_id if check_plan is not None else None,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        trace_metadata=trace_metadata,
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
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
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

    resolved_plan = _resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    interaction_basis = resolved_plan.interaction_basis
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

    if clifford_frame_policy is not None:
        if patch is None:
            from pecos.qec.surface import SurfacePatch

            patch = SurfacePatch.create(distance=d)
        from pecos.qec.surface.circuit_builder import _resolve_szz_clifford_frame_for_builder

        _resolve_szz_clifford_frame_for_builder(
            patch,
            interaction_basis=interaction_basis,
            clifford_frame_policy=clifford_frame_policy,
        )

    if interaction_basis == "szz" and twirl is not None:
        msg = "interaction_basis='szz' Guppy runtime twirl integration is staged later"
        raise ValueError(msg)

    twirl_entropy_qubits = 1 if twirl is not None else 0
    return num_data + normalize_ancilla_budget(total_ancilla, ancilla_budget) + twirl_entropy_qubits


def generate_surface_code_module(
    d: int,
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
) -> str:
    """Generate source code for a distance-d surface code module.

    Args:
        d: Code distance (must be odd >= 3)
        ancilla_budget: Optional cap on simultaneously live ancillas;
            forwarded to ``generate_guppy_source``.
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ/SZZdg surface-code generation.
        szz_runtime_barriers: SZZ/SZZdg scheduling-barrier policy.
        trace_metadata: Emit PECOS trace metadata helpers in generated SZZ
            source.

    Returns:
        Python/Guppy source code as a string
    """
    _validate_surface_memory_distance(d)

    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    return generate_guppy_source(
        patch,
        ancilla_budget=ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        trace_metadata=trace_metadata,
    )


def _round_scoped_surface_memory_factory(
    patch: "SurfacePatch",
    basis: str,
    *,
    ancilla_budget: int | None,
    interaction_basis: str | None,
    check_plan: str | None,
    clifford_frame_policy: str | None,
    szz_runtime_barriers: bool | str,
    trace_metadata: bool,
) -> Callable[[int], object]:
    """Memory factory that scopes each call to a round-specific Guppy module.

    Backs :func:`get_surface_code_module`: each ``factory(n)`` re-enters
    :func:`_surface_code_module_for_patch` with the concrete ``num_rounds`` so it
    lands in a distinct ``pecos._generated.patch_..._r{n}`` module. Without this,
    a caller building memory experiments at more than one round count on one
    round-agnostic module would collide on a single guppy module-qualified name.
    """

    def factory(num_rounds: int) -> object:
        scoped = _surface_code_module_for_patch(
            patch,
            ancilla_budget=ancilla_budget,
            num_rounds=int(num_rounds),
            interaction_basis=interaction_basis,
            check_plan=check_plan,
            clifford_frame_policy=clifford_frame_policy,
            szz_runtime_barriers=szz_runtime_barriers,
            trace_metadata=trace_metadata,
        )
        return scoped[f"make_memory_{basis}"](num_rounds)

    return factory


def _surface_code_module_for_patch(
    patch: "SurfacePatch",
    *,
    ancilla_budget: int | None = None,
    num_rounds: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
) -> dict:
    """Load + cache a surface-code module for an arbitrary patch.

    Cache key spans full patch identity (dx, dz, orientation, rotated) plus
    the effective budget, so distinct geometries never collide and the
    unconstrained-via-``None`` / unconstrained-via-large-int cases share one
    entry. Module metadata is derived from the patch geometry (faithful for
    asymmetric / non-rotated patches), not from a scalar distance.

    ``num_rounds`` is threaded into both the cache key and the generated module
    identity so that callers building memory experiments at different round
    counts in one process get distinct Guppy modules (see
    :func:`_guppy_module_cache_key`). When ``num_rounds`` is ``None`` the
    returned ``make_memory_*`` factories are replaced with round-scoping
    wrappers that re-enter this function with the concrete count, so the
    round-agnostic getter path stays isolated across round counts too.
    """
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    original_interaction_basis = interaction_basis
    resolved_plan = _resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    interaction_basis = resolved_plan.interaction_basis
    szz_runtime_barrier_policy = _normalize_szz_runtime_barrier_policy(szz_runtime_barriers)
    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    cache_key = (
        patch.dx,
        patch.dz,
        geom.orientation.name,
        geom.rotated,
        effective_budget,
        interaction_basis,
        resolved_plan.plan_id,
        None if clifford_frame_policy is None else str(clifford_frame_policy).lower().replace("-", "_"),
        szz_runtime_barrier_policy,
        trace_metadata,
        None if num_rounds is None else int(num_rounds),
    )

    if cache_key in _state.distance_module_cache:
        return _state.distance_module_cache[cache_key]

    module = _load_guppy_module(
        patch,
        ancilla_budget=ancilla_budget,
        num_rounds=num_rounds,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barrier_policy,
        trace_metadata=trace_metadata,
    )

    # Metadata derived from the actual patch geometry.
    module["distance"] = patch.distance
    module["num_data"] = geom.num_data
    module["num_stab"] = total_ancilla
    module["ancilla_budget"] = effective_budget
    module["interaction_basis"] = interaction_basis
    module["check_plan"] = resolved_plan.plan_id
    module["clifford_frame_policy"] = clifford_frame_policy
    module["szz_runtime_barriers"] = szz_runtime_barrier_policy != _SZZ_RUNTIME_BARRIER_POLICY_NONE
    module["szz_runtime_barrier_policy"] = szz_runtime_barrier_policy
    module["trace_metadata"] = trace_metadata
    module["resolved_check_plan"] = resolved_plan.resolved_metadata
    module["resolved_check_plan_hash"] = resolved_plan.resolved_hash

    if num_rounds is None:
        module = dict(module)
        for basis in ("z", "x"):
            key = f"make_memory_{basis}"
            if key in module:
                module[key] = _round_scoped_surface_memory_factory(
                    patch,
                    basis,
                    ancilla_budget=ancilla_budget,
                    interaction_basis=original_interaction_basis,
                    check_plan=check_plan,
                    clifford_frame_policy=clifford_frame_policy,
                    szz_runtime_barriers=szz_runtime_barrier_policy,
                    trace_metadata=trace_metadata,
                )

    _state.distance_module_cache[cache_key] = module
    return module


def get_surface_code_module(
    d: int,
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
) -> dict:
    """Get a loaded surface code module for distance d.

    Args:
        d: Code distance (must be odd >= 3)
        ancilla_budget: Optional cap on simultaneously live ancillas
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ/SZZdg surface-code generation.
        szz_runtime_barriers: SZZ/SZZdg scheduling-barrier policy.

    Returns:
        Dictionary with module contents and metadata
    """
    from pecos.qec.surface import SurfacePatch

    _validate_surface_memory_distance(d)
    patch = SurfacePatch.create(distance=d)
    return _surface_code_module_for_patch(
        patch,
        ancilla_budget=ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
    )


def make_surface_code(
    distance: int,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    trace_metadata: bool = True,
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
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ/SZZdg surface-code generation.
        szz_runtime_barriers: SZZ/SZZdg scheduling-barrier policy.
        trace_metadata: Emit PECOS trace metadata helpers in generated SZZ
            source. Keep enabled for traced-QIS/DEM paths; disable for
            execution-only builds whose compiler/linker cannot resolve the
            metadata helper symbols.

    Returns:
        Compiled Guppy program
    """
    if basis.upper() not in ("Z", "X"):
        msg = f"basis must be 'Z' or 'X', got {basis!r}"
        raise ValueError(msg)

    from pecos.qec.surface import SurfacePatch

    _validate_surface_memory_distance(distance)
    patch = SurfacePatch.create(distance=distance)
    return generate_memory_experiment(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        trace_metadata=trace_metadata,
    )
