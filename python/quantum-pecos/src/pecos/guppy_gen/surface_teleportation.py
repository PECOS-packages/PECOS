# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Executable surface-code T teleportation with raw resource preparation.

The feed-forward uses a physical fold-S round, not a Pauli-frame annotation.
Noisy execution requires decoded measurement outcomes and a sufficiently good
resource; the convenience factory uses raw parities and is a noiseless
reference experiment, not a fault-tolerant magic-state factory.
"""

import hashlib
from functools import cache

from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface.circuit_builder import QubitAllocation
from pecos.qec.surface.gadgets import (
    default_allocation,
    fold_sz_round_gadget,
    measure_out_gadget,
    syndrome_round_gadget,
    transversal_cx_gadget,
)
from pecos.qec.surface.injection import state_injection
from pecos.qec.surface.patch import SurfacePatch


def render_surface_t_teleportation_module(patch: SurfacePatch) -> str:
    """Render a standalone module for raw state injection and T/TDG teleportation."""
    return _render_surface_t_teleportation_module(patch, include_common=True)


def _render_surface_t_teleportation_module(patch: SurfacePatch, *, include_common: bool) -> str:
    """Render raw injection and reusable resource-consumption/correction functions.

    ``consume_t_resource(data, resource)`` consumes an encoded T (or T-dagger)
    resource and returns its raw logical-Z parity. ``correct_t_teleportation``
    takes the decoded parity and performs S (or S-dagger) when it is true.
    Both patches must be in the canonical unswapped orientation with +1
    stabilizers and no pending logical Pauli frame.
    The square restriction comes from the existing fold-S correction gadget.
    """
    if not patch.rotated or patch.dx != patch.dz or patch.dx < 3 or patch.dx % 2 == 0:
        msg = "Surface T teleportation requires an odd square rotated patch of distance >= 3"
        raise ValueError(msg)
    allocation = default_allocation(patch)
    n = patch.geometry.num_data
    nx = len(patch.geometry.x_stabilizers)
    nz = len(patch.geometry.z_stabilizers)
    surface = f"SurfaceCode_{patch.dx}x{patch.dz}"
    syndrome = f"Syndrome_{patch.dx}x{patch.dz}"
    lines = [
        '"""Raw surface injection and corrected T/TDG teleportation; no noisy decoder or distillation."""',
        "from __future__ import annotations",
        "from guppylang import guppy",
        "from guppylang.std.builtins import array, comptime, owned, output",
        "from guppylang.std.quantum import cx, cz, h, s, sdg, t, tdg, x, z, qubit",
        "from guppylang.std.quantum import collect_measurements, measure, measure_array",
        "from pecos.guppy_gen.variant import variant_scoped",
        "",
        "@guppy.struct",
        f"class {surface}:",
        f"    data: array[qubit, {n}]",
        "",
        "@guppy.struct",
        f"class {syndrome}:",
        f"    synx: array[bool, {nx}]",
        f"    synz: array[bool, {nz}]",
        "",
    ]
    if not include_common:
        lines = ["from guppylang.std.quantum import t, tdg, z", ""]
    round_name = "syndrome_extraction" if include_common else "syndrome_extraction_a"
    fold_s = "syndrome_extraction_fold_sz" if include_common else "syndrome_extraction_fold_sz_a"
    fold_sdg = "syndrome_extraction_fold_szdg" if include_common else "syndrome_extraction_fold_szdg_a"
    injections = [state_injection(patch, state=state) for state in ("Z", "-Z", "X", "-X", "Y", "-Y", "T", "TDG")]
    offset = patch.geometry.num_qubits
    target = QubitAllocation(
        [q + offset for q in allocation.data_qubits],
        [q + offset for q in allocation.x_ancilla_qubits],
        [q + offset for q in allocation.z_ancilla_qubits],
    )
    functions = [injection.seed for injection in injections]
    functions.extend(
        [
            syndrome_round_gadget(patch, allocation, round_index=0),
            fold_sz_round_gadget(patch, allocation, round_index=0),
            fold_sz_round_gadget(patch, allocation, round_index=0, dagger=True),
            transversal_cx_gadget(patch, allocation, patch, target),
            measure_out_gadget(patch, allocation, basis="Z"),
            measure_out_gadget(patch, allocation, basis="X"),
        ],
    )
    if not include_common:
        functions = [injection.seed for injection in injections]
    for gadget in functions:
        lines.extend(render_gadget_function(gadget))
        lines.extend(["", ""])
    lines.extend(
        [
            "@guppy",
            f"def fix_injection_signs(surf: {surface}, syn: {syndrome}) -> None:",
            '    """Remove ideal encoding byproducts; this is not noisy decoding."""',
        ],
    )
    for i, support in enumerate(injections[0].correction_supports):
        lines.append(f"    if syn.synx[{i}]:")
        lines.extend(f"        z(surf.data[{q}])" for q in support)
    lines.extend(["", ""])
    for injection in injections:
        name = injection.seed.name.removeprefix("prep_injection_").removesuffix("_seed")
        lines.extend(
            [
                "@guppy",
                f"def prepare_injected_{name}() -> {surface}:",
                f"    surf = {injection.seed.name}()",
                f"    syn = {round_name}(surf)",
                '    output("injection_synx", syn.synx)',
                '    output("injection_synz", syn.synz)',
                "    fix_injection_signs(surf, syn)",
                "    return surf",
                "",
                "",
            ],
        )
    parity = " ^ ".join(f"bits[{q}]" for q in patch.geometry.logical_z.data_qubits)
    lines.extend(
        [
            "@guppy",
            f"def consume_t_resource(data: {surface}, resource: {surface} @ owned) -> bool:",
            '    """Consume canonical T/TDG resource; return RAW logical Z, pending decoding."""',
            "    transversal_cx(data, resource)",
            "    bits = measure_z_basis(resource)",
            '    output("teleportation_resource_readout", bits)',
            f"    return {parity}",
            "",
            "",
            "@guppy",
            f"def correct_t_teleportation(data: {surface}, decoded_outcome: bool, dagger: bool) -> None:",
            '    """Apply physical logical S/SDG for the decoded teleportation outcome."""',
            '    output("t_correction", decoded_outcome)',
            "    if decoded_outcome:",
            "        if dagger:",
            f"            syn = {fold_sdg}(data)",
            "        else:",
            f"            syn = {fold_s}(data)",
            "    else:",
            f"        syn = {round_name}(data)",
            '    output("correction_synx", syn.synx)',
            '    output("correction_synz", syn.synz)',
            "",
            "",
            "@guppy",
            f"def apply_t_teleportation(data: {surface}, resource: {surface} @ owned, dagger: bool) -> None:",
            '    """Noiseless convenience: use raw parity as the correction decision."""',
            "    outcome = consume_t_resource(data, resource)",
            "    correct_t_teleportation(data, outcome, dagger)",
            "",
            "",
            "def make_t_teleportation(rounds_before: int = 0, rounds_after: int = 0, *,",
            '                     input_state: str = "X", readout_basis: str = "X", dagger: bool = False):',
            '    """Raw resource + corrected logical T/TDG. Syndrome rounds are recorded, not decoded."""',
            "    if (type(rounds_before) is not int or type(rounds_after) is not int",
            "            or min(rounds_before, rounds_after) < 0):",
            '        raise ValueError("Teleportation round counts must be nonnegative integers")',
            '    if input_state not in ("Z", "-Z", "X", "-X", "Y", "-Y"):',
            '        raise ValueError("input_state must be a signed X/Y/Z eigenstate")',
            '    if readout_basis not in ("X", "Y", "Z"):',
            '        raise ValueError("readout_basis must be X, Y, or Z")',
            "    if type(dagger) is not bool:",
            '        raise ValueError("dagger must be a boolean")',
            "    def experiment() -> None:",
        ],
    )
    for i, state in enumerate(("Z", "-Z", "X", "-X", "Y", "-Y")):
        name = state.lower().replace("-", "minus_")
        lines.extend(
            [
                f'        {"if" if i == 0 else "elif"} comptime(input_state == "{state}"):',
                f"            data = prepare_injected_{name}()",
            ],
        )
    # A final else lets Guppy prove the linear data variable is always defined.
    lines.extend(
        [
            "        else:",
            "            data = prepare_injected_x()",
            "        if comptime(dagger):",
            "            resource = prepare_injected_tdg()",
            "        else:",
            "            resource = prepare_injected_t()",
            "        for _ in range(comptime(rounds_before)):",
            f"            syn_data = {round_name}(data)",
            f"            syn_resource = {round_name}(resource)",
            '            output("data_synx", syn_data.synx)',
            '            output("data_synz", syn_data.synz)',
            '            output("resource_synx", syn_resource.synx)',
            '            output("resource_synz", syn_resource.synz)',
            "        apply_t_teleportation(data, resource, comptime(dagger))",
            "        for _ in range(comptime(rounds_after)):",
            f"            syn = {round_name}(data)",
            '            output("data_synx", syn.synx)',
            '            output("data_synz", syn.synz)',
            '        if comptime(readout_basis == "Y"):',
            f"            syn = {fold_sdg}(data)",
            '            output("readout_synx", syn.synx)',
            '            output("readout_synz", syn.synz)',
            '        if comptime(readout_basis == "Z"):',
            "            final = measure_z_basis(data)",
            "        else:",
            "            final = measure_x_basis(data)",
            '        output("final_data", final)',
            "    return guppy(variant_scoped(",
            "        experiment, rounds_before, rounds_after, input_state, readout_basis, dagger))",
            "",
        ],
    )
    return "\n".join(lines)


@cache
def _load_source(source: str) -> dict:
    key = f"surface_t_teleportation_{hashlib.sha256(source.encode()).hexdigest()}"
    return load_guppy_source(source, _get_temp_dir() / f"{key}.py", f"pecos._generated.{key}")


def load_surface_t_teleportation_module(patch: SurfacePatch) -> dict:
    """Load the generated state-preparation and gate-teleportation functions, caching by their complete source."""
    return _load_source(render_surface_t_teleportation_module(patch))


def make_surface_t_teleportation(
    patch: SurfacePatch,
    rounds_before: int = 0,
    rounds_after: int = 0,
    *,
    input_state: str = "X",
    readout_basis: str = "X",
    dagger: bool = False,
    resource_preparation: str = "raw",
    verification_rounds: int = 2,
) -> object:
    """Build a T/TDG gate-teleportation experiment with physical feed-forward.

    Use a non-Clifford simulator (e.g. ``pecos.stab_vec()``). Peak live qubits
    are ``patch.geometry.num_data + patch.geometry.num_qubits``. No trusted
    DEM certificate is attached: non-Clifford adaptive execution cannot be
    represented by the existing static Clifford detector-model pipeline.

    Set ``resource_preparation="hook"`` for a heralded hook resource with
    ``verification_rounds`` extra checks. Rejected attempts emit
    ``hook_accepted=False`` and are discarded before allocating the data patch.
    Rejected shots have zero placeholders in ``final_data`` and downstream
    records; filter by ``hook_accepted``. ``teleportation_performed`` also
    reports whether the data patch was used. Verification applies only to
    resource preparation; teleportation still uses raw parity.
    """
    if resource_preparation == "hook":
        from pecos.guppy_gen.surface_hook_injection import (  # noqa: PLC0415 - mutually dependent renderers
            load_surface_hook_injection_module,
        )

        if type(dagger) is not bool:
            msg = "dagger must be a boolean"
            raise ValueError(msg)
        module = load_surface_hook_injection_module(patch, verification_rounds)
        return module["make_hook_experiment"](
            state="TDG" if dagger else "T",
            readout_basis=readout_basis,
            consume=True,
            input_state=input_state,
            rounds_before=rounds_before,
            rounds_after=rounds_after,
        )
    if resource_preparation != "raw":
        msg = "resource_preparation must be raw or hook"
        raise ValueError(msg)
    module = load_surface_t_teleportation_module(patch)
    return module["make_t_teleportation"](
        rounds_before,
        rounds_after,
        input_state=input_state,
        readout_basis=readout_basis,
        dagger=dagger,
    )
