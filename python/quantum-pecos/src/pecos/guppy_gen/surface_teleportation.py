# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Executable surface-code raw injection, heralded hook injection, and T teleportation.

The feed-forward uses a physical fold-S round, not a Pauli-frame annotation.
Noisy execution requires decoded measurement outcomes and a sufficiently good
resource; the convenience factory uses raw parities and is a noiseless
reference experiment, not a fault-tolerant magic-state factory.
"""

from pecos.guppy_gen._module_loader import load_cached_guppy_source
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.qec.surface.circuit_builder import QubitAllocation
from pecos.qec.surface.gadgets import (
    default_allocation,
    fold_sz_round_gadget,
    measure_out_gadget,
    syndrome_round_gadget,
    transversal_cx_gadget,
)
from pecos.qec.surface.hook_injection import hook_injection
from pecos.qec.surface.injection import state_injection
from pecos.qec.surface.patch import SurfacePatch

_INPUT_STATES = ("Z", "-Z", "X", "-X", "Y", "-Y")
_RESOURCE_STATES = ("T", "TDG")
_HOOK_STATES = (*_RESOURCE_STATES, "X", "-X", "Y", "-Y")
_READOUT_BASES = ("X", "Y", "Z")


def _state_suffix(state: str) -> str:
    return state.lower().replace("-", "minus_")


def _prepare_input(indent: str) -> list[str]:
    lines = []
    for i, state in enumerate(_INPUT_STATES):
        lines.extend(
            [
                f'{indent}{"if" if i == 0 else "elif"} comptime(input_state == "{state}"):',
                f"{indent}    data = prepare_injected_{_state_suffix(state)}()",
            ],
        )
    # A final else lets Guppy prove the linear data variable is always defined.
    lines.extend([f"{indent}else:", f"{indent}    data = prepare_injected_x()"])
    return lines


def _teleportation_body(indent: str, dagger: str) -> list[str]:
    return [
        f"{indent}for _ in range(comptime(rounds_before)):",
        f"{indent}    syn_data = syndrome_extraction_data(data)",
        f"{indent}    syn_resource = syndrome_extraction_anc(resource)",
        f'{indent}    output("data_synx", syn_data.synx)',
        f'{indent}    output("data_synz", syn_data.synz)',
        f'{indent}    output("resource_synx", syn_resource.synx)',
        f'{indent}    output("resource_synz", syn_resource.synz)',
        f"{indent}apply_t_teleportation(data, resource, comptime({dagger}))",
    ]


def _readout_body(indent: str) -> list[str]:
    return [
        f"{indent}for _ in range(comptime(rounds_after)):",
        f"{indent}    syn = syndrome_extraction_data(data)",
        f'{indent}    output("data_synx", syn.synx)',
        f'{indent}    output("data_synz", syn.synz)',
        f'{indent}if comptime(readout_basis == "Y"):',
        f"{indent}    syn = syndrome_extraction_fold_szdg_data(data)",
        f'{indent}    output("readout_synx", syn.synx)',
        f'{indent}    output("readout_synz", syn.synz)',
        f'{indent}if comptime(readout_basis == "Z"):',
        f"{indent}    final = measure_z_basis(data)",
        f"{indent}else:",
        f"{indent}    final = measure_x_basis(data)",
        f'{indent}output("final_data", final)',
    ]


def render_surface_t_teleportation_module(patch: SurfacePatch) -> str:
    """Render a standalone module for raw state injection and T/TDG teleportation."""
    return _render_surface_t_teleportation_module(patch, include_common=True)


def _render_surface_t_teleportation_module(
    patch: SurfacePatch,
    *,
    include_common: bool,
    sidebands: bool = True,
) -> str:
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
    injections = [state_injection(patch, state=state) for state in (*_INPUT_STATES, *_RESOURCE_STATES)]
    if include_common:
        lines = [
            '"""Raw injection and corrected T/TDG teleportation; no noisy decoder or distillation.',
            "Injected signed X/Y/Z inputs project in the data role; T/TDG resources project in the anc role.",
            '"""',
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
        offset = patch.geometry.num_qubits
        target = QubitAllocation(
            [q + offset for q in allocation.data_qubits],
            [q + offset for q in allocation.x_ancilla_qubits],
            [q + offset for q in allocation.z_ancilla_qubits],
        )
        functions = [(syndrome_round_gadget(patch, allocation, round_index=0), scope) for scope in ("data", "anc")]
        functions.extend(
            (fold_sz_round_gadget(patch, allocation, round_index=0, dagger=dagger), "data") for dagger in (False, True)
        )
        functions.extend(
            [
                (transversal_cx_gadget(patch, allocation, patch, target), None),
                (measure_out_gadget(patch, allocation, basis="Z"), None),
                (measure_out_gadget(patch, allocation, basis="X"), None),
            ],
        )
    else:
        lines = ["from guppylang.std.quantum import t, tdg, z", ""]
        functions = []
    functions.extend((injection.seed, None) for injection in injections)
    for gadget, scope in functions:
        lines.extend(render_gadget_function(gadget, tag_scope=scope, sidebands=sidebands))
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
        name = _state_suffix(injection.seed.basis)
        scope = "anc" if injection.seed.basis in _RESOURCE_STATES else "data"
        lines.extend(
            [
                "@guppy",
                f"def prepare_injected_{name}() -> {surface}:",
                f"    surf = {injection.seed.name}()",
                f"    syn = syndrome_extraction_{scope}(surf)",
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
            "            syn = syndrome_extraction_fold_szdg_data(data)",
            "        else:",
            "            syn = syndrome_extraction_fold_sz_data(data)",
            "    else:",
            "        syn = syndrome_extraction_data(data)",
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
            "    input_state = input_state.upper() if isinstance(input_state, str) else input_state",
            "    readout_basis = readout_basis.upper() if isinstance(readout_basis, str) else readout_basis",
            f"    if input_state not in {_INPUT_STATES!r}:",
            '        raise ValueError("input_state must be a signed X/Y/Z eigenstate")',
            f"    if readout_basis not in {_READOUT_BASES!r}:",
            '        raise ValueError("readout_basis must be X, Y, or Z")',
            "    if type(dagger) is not bool:",
            '        raise ValueError("dagger must be a boolean")',
            "    def experiment() -> None:",
        ],
    )
    lines.extend(_prepare_input("        "))
    lines.extend(
        [
            "        if comptime(dagger):",
            "            resource = prepare_injected_tdg()",
            "        else:",
            "            resource = prepare_injected_t()",
            *_teleportation_body("        ", "dagger"),
            *_readout_body("        "),
            "    return guppy(variant_scoped(",
            "        experiment, rounds_before, rounds_after, input_state, readout_basis, dagger))",
            "",
        ],
    )
    return "\n".join(lines)


def load_surface_t_teleportation_module(patch: SurfacePatch) -> dict:
    """Load the generated state-preparation and gate-teleportation functions, caching by their complete source."""
    return load_cached_guppy_source("surface_t_teleportation", render_surface_t_teleportation_module(patch))


def make_surface_t_teleportation(
    patch: SurfacePatch,
    rounds_before: int = 0,
    rounds_after: int = 0,
    *,
    input_state: str = "X",
    readout_basis: str = "X",
    dagger: bool = False,
    resource_preparation: str = "raw",
    verification_rounds: int | None = None,
) -> object:
    """Build a T/TDG gate-teleportation experiment with physical feed-forward.

    Use a non-Clifford simulator (e.g. ``pecos.stab_vec()``). Peak live qubits
    are ``patch.geometry.num_data + patch.geometry.num_qubits``. No trusted
    DEM certificate is attached: non-Clifford adaptive execution cannot be
    represented by the existing static Clifford detector-model pipeline.

    Set ``resource_preparation="hook"`` for a heralded hook resource with
    ``verification_rounds`` extra checks (None uses the hook renderer default). Rejected attempts emit
    ``hook_accepted=False`` and are discarded before allocating the data patch.
    Rejected shots have zero placeholders in ``final_data`` and downstream
    records; filter by ``hook_accepted``. ``teleportation_performed`` also
    reports whether the data patch was used. Verification applies only to
    resource preparation; teleportation still uses raw parity.
    """
    if resource_preparation == "hook":
        if type(dagger) is not bool:
            msg = "dagger must be a boolean"
            raise ValueError(msg)
        module = (
            load_surface_hook_injection_module(patch)
            if verification_rounds is None
            else load_surface_hook_injection_module(patch, verification_rounds)
        )
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
    if verification_rounds is not None:
        msg = "verification_rounds is only supported with resource_preparation=hook"
        raise ValueError(msg)
    module = load_surface_t_teleportation_module(patch)
    return module["make_t_teleportation"](
        rounds_before,
        rounds_after,
        input_state=input_state,
        readout_basis=readout_basis,
        dagger=dagger,
    )


def render_surface_hook_injection_module(patch: SurfacePatch, verification_rounds: int = 2) -> str:
    """Render hook attempts, with a fixed positive number of verification rounds.

    The generated ``attempt_hook_t()`` (also tdg/x/minus_x/y/minus_y) returns
    ``(resource, accepted)``. Callers MUST discard rejected resources. Accepted
    resources have their ideal encoding signs corrected physically; this does
    not decode noise. The module includes the reusable teleportation helpers.
    Their fold-S implementation restricts this module to odd squares >= 3;
    the abstract ``hook_injection`` also supports even square distances.
    """
    if type(verification_rounds) is not int or verification_rounds < 1:
        msg = "verification_rounds must be a positive integer (after the injection round)"
        raise ValueError(msg)
    # Aggregate syndrome records below are sufficient; per-ancilla debug
    # outputs inside borrowed helpers would differ between accept/reject paths.
    common = _render_surface_t_teleportation_module(patch, include_common=True, sidebands=False)
    lines = common.splitlines()
    lines.append("")
    surface = f"SurfaceCode_{patch.dx}x{patch.dz}"
    syndrome = f"Syndrome_{patch.dx}x{patch.dz}"
    n = patch.geometry.num_data
    nx = len(patch.geometry.x_stabilizers)
    nz = len(patch.geometry.z_stabilizers)
    injections = [hook_injection(patch, state=state) for state in _HOOK_STATES]
    hook = injections[0]
    for gadget in [hook.seed, hook.verification, *(inj.injection for inj in injections)]:
        lines.extend(render_gadget_function(gadget, tag_scope="hook"))
        lines.extend(["", ""])
    lines.extend(
        [
            "@guppy",
            f"def hook_initial_is_valid(syn: {syndrome}) -> bool:",
            '    """Check only the first-round outcomes predictable from product preparation."""',
            "    accepted = True",
        ],
    )
    for family, indices in (("x", hook.predictable_x), ("z", hook.predictable_z)):
        lines.extend(f"    accepted = accepted & (not syn.syn{family}[{i}])" for i in indices)
    lines.extend(
        [
            "    return accepted",
            "",
            "",
            "@guppy",
            f"def hook_syndromes_match(first: {syndrome}, later: {syndrome}) -> bool:",
            "    accepted = True",
        ],
    )
    for family, supports in (("x", hook.z_corrections), ("z", hook.x_corrections)):
        lines.extend(
            f"    accepted = accepted & (first.syn{family}[{i}] == later.syn{family}[{i}])"
            for i in range(len(supports))
        )
    lines.extend(
        [
            "    return accepted",
            "",
            "",
            "@guppy",
            f"def fix_hook_signs(surf: {surface}, syn: {syndrome}) -> None:",
            '    """Canonicalize ideal byproducts; physical corrections are not a noisy decoder."""',
        ],
    )
    for family, supports, gate in (("x", hook.z_corrections, "z"), ("z", hook.x_corrections, "x")):
        for i, support in enumerate(supports):
            lines.append(f"    if syn.syn{family}[{i}]:")
            lines.extend(f"        {gate}(surf.data[{q}])" for q in support)
    lines.extend(["", ""])
    for inj in injections:
        name = inj.injection.name.removeprefix("hook_injection_")
        lines.extend(
            [
                "@guppy",
                f"def attempt_hook_{name}() -> tuple[{surface}, bool]:",
                '    """One attempt; caller must discard the resource when accepted is false."""',
                "    surf = prep_injection_hook_seed_hook()",
                f"    first = {inj.injection.name}_hook(surf)",
                '    output("hook_synx", first.synx)',
                '    output("hook_synz", first.synz)',
                "    accepted = hook_initial_is_valid(first)",
                f"    for _ in range({verification_rounds}):",
                "        later = hook_verification_hook(surf)",
                '        output("hook_synx", later.synx)',
                '        output("hook_synz", later.synz)',
                "        accepted = accepted & hook_syndromes_match(first, later)",
                "    if accepted:",
                "        fix_hook_signs(surf, first)",
                '    output("hook_accepted", accepted)',
                "    return surf, accepted",
                "",
                "",
            ],
        )
    lines.extend(
        [
            'def make_hook_experiment(*, state: str = "T", readout_basis: str = "X",',
            '                         consume: bool = False, input_state: str = "X",',
            "                         rounds_before: int = 0, rounds_after: int = 0):",
            '    """One attempt; rejected shots have zero placeholders and hook_accepted=False."""',
            "    state = state.upper() if isinstance(state, str) else state",
            "    input_state = input_state.upper() if isinstance(input_state, str) else input_state",
            "    readout_basis = readout_basis.upper() if isinstance(readout_basis, str) else readout_basis",
            f"    if state not in {_HOOK_STATES!r}:",
            '        raise ValueError("Unsupported hook state")',
            f"    if readout_basis not in {_READOUT_BASES!r}:",
            '        raise ValueError("readout_basis must be X, Y, or Z")',
            f"    if type(consume) is not bool or (consume and state not in {_RESOURCE_STATES!r}):",
            '        raise ValueError("Teleportation requires a T or TDG resource")',
            f"    if input_state not in {_INPUT_STATES!r}:",
            '        raise ValueError("input_state must be a signed X/Y/Z eigenstate")',
            "    if (type(rounds_before) is not int or type(rounds_after) is not int",
            "            or min(rounds_before, rounds_after) < 0):",
            '        raise ValueError("Teleportation round counts must be nonnegative integers")',
            "    def experiment() -> None:",
        ],
    )
    for i, state in enumerate(_HOOK_STATES):
        name = _state_suffix(state)
        lines.extend(
            [
                f'        {"if" if i == 0 else "elif"} comptime(state == "{state}"):',
                f"            resource, accepted = attempt_hook_{name}()",
            ],
        )
    lines.extend(
        [
            "        else:",
            "            resource, accepted = attempt_hook_t()",
            "        if accepted:",
            "            if comptime(consume):",
        ],
    )
    lines.extend(_prepare_input("                "))
    lines.extend(
        [
            *_teleportation_body("                ", 'state == "TDG"'),
            '                output("teleportation_performed", True)',
            "            else:",
            "                data = resource",
            *_readout_body("            "),
            "        else:",
            "            discarded = measure_z_basis(resource)",
            "            # No data patch or teleportation on failure. Fixed-shape",
            "            # placeholders are INVALID unless hook_accepted is true.",
            f"            empty_x = array(False for _ in range({nx}))",
            f"            empty_z = array(False for _ in range({nz}))",
            f"            empty_data = array(False for _ in range({n}))",
            '            output("final_data", empty_data)',
            '            if comptime(readout_basis == "Y"):',
            '                output("readout_synx", empty_x)',
            '                output("readout_synz", empty_z)',
            "            if comptime(consume):",
            '                output("teleportation_performed", False)',
            '                output("injection_synx", empty_x)',
            '                output("injection_synz", empty_z)',
            '                output("teleportation_resource_readout", empty_data)',
            '                output("t_correction", False)',
            '                output("correction_synx", empty_x)',
            '                output("correction_synz", empty_z)',
            "                for _ in range(comptime(rounds_before)):",
            '                    output("data_synx", empty_x)',
            '                    output("data_synz", empty_z)',
            '                    output("resource_synx", empty_x)',
            '                    output("resource_synz", empty_z)',
            "            for _ in range(comptime(rounds_after)):",
            '                output("data_synx", empty_x)',
            '                output("data_synz", empty_z)',
            "    return guppy(variant_scoped(",
            "        experiment, state, readout_basis, consume, input_state, rounds_before, rounds_after))",
            "",
        ],
    )
    return "\n".join(lines)


def load_surface_hook_injection_module(patch: SurfacePatch, verification_rounds: int = 2) -> dict:
    """Load reusable hook attempts and the heralded experiment factory."""
    return load_cached_guppy_source(
        "surface_hook_injection",
        render_surface_hook_injection_module(patch, verification_rounds),
    )


def make_surface_hook_injection(
    patch: SurfacePatch,
    verification_rounds: int = 2,
    *,
    state: str = "T",
    readout_basis: str = "X",
) -> object:
    """Prepare and measure one hook resource per shot, recording acceptance.

    ``hook_accepted`` reports every attempt. Failed shots have zero placeholders
    in ``final_data`` and readout fields: always filter by ``hook_accepted``
    when computing conditional statistics. No retry loop or static detector
    certificate is attached. Physical sign corrections and final readout can
    themselves be noisy.
    """
    module = load_surface_hook_injection_module(patch, verification_rounds)
    return module["make_hook_experiment"](state=state, readout_basis=readout_basis)
