# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Executable, heralded hook resources and their use in T teleportation."""

import hashlib
from functools import cache

from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.gadget_render import render_gadget_function
from pecos.guppy_gen.surface_teleportation import render_surface_t_teleportation_module
from pecos.qec.surface.hook_injection import hook_injection
from pecos.qec.surface.patch import SurfacePatch


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
    common = render_surface_t_teleportation_module(patch)
    lines = [line for line in common.splitlines() if ":meas:" not in line]
    lines.append("")
    surface = f"SurfaceCode_{patch.dx}x{patch.dz}"
    syndrome = f"Syndrome_{patch.dx}x{patch.dz}"
    n = patch.geometry.num_data
    nx = len(patch.geometry.x_stabilizers)
    nz = len(patch.geometry.z_stabilizers)
    injections = [hook_injection(patch, state=state) for state in ("T", "TDG", "X", "-X", "Y", "-Y")]
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
            '    if state not in ("T", "TDG", "X", "-X", "Y", "-Y"):',
            '        raise ValueError("Unsupported hook state")',
            '    if readout_basis not in ("X", "Y", "Z"):',
            '        raise ValueError("readout_basis must be X, Y, or Z")',
            '    if type(consume) is not bool or (consume and state not in ("T", "TDG")):',
            '        raise ValueError("Teleportation requires a T or TDG resource")',
            '    if input_state not in ("Z", "-Z", "X", "-X", "Y", "-Y"):',
            '        raise ValueError("input_state must be a signed X/Y/Z eigenstate")',
            "    if (type(rounds_before) is not int or type(rounds_after) is not int",
            "            or min(rounds_before, rounds_after) < 0):",
            '        raise ValueError("Teleportation round counts must be nonnegative integers")',
            "    def experiment() -> None:",
        ],
    )
    for i, state in enumerate(("T", "TDG", "X", "-X", "Y", "-Y")):
        name = state.lower().replace("-", "minus_")
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
    for i, state in enumerate(("Z", "-Z", "X", "-X", "Y", "-Y")):
        name = state.lower().replace("-", "minus_")
        lines.extend(
            [
                f'                {"if" if i == 0 else "elif"} comptime(input_state == "{state}"):',
                f"                    data = prepare_injected_{name}()",
            ],
        )
    lines.extend(
        [
            "                else:",
            "                    data = prepare_injected_x()",
            "                for _ in range(comptime(rounds_before)):",
            "                    syn_data = syndrome_extraction(data)",
            "                    syn_resource = syndrome_extraction(resource)",
            '                    output("data_synx", syn_data.synx)',
            '                    output("data_synz", syn_data.synz)',
            '                    output("resource_synx", syn_resource.synx)',
            '                    output("resource_synz", syn_resource.synz)',
            '                apply_t_teleportation(data, resource, comptime(state == "TDG"))',
            '                output("teleportation_performed", True)',
            "            else:",
            "                data = resource",
            "            for _ in range(comptime(rounds_after)):",
            "                syn = syndrome_extraction(data)",
            '                output("data_synx", syn.synx)',
            '                output("data_synz", syn.synz)',
            '            if comptime(readout_basis == "Y"):',
            "                syn = syndrome_extraction_fold_szdg(data)",
            '                output("readout_synx", syn.synx)',
            '                output("readout_synz", syn.synz)',
            '            if comptime(readout_basis == "Z"):',
            "                final = measure_z_basis(data)",
            "            else:",
            "                final = measure_x_basis(data)",
            '            output("final_data", final)',
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


@cache
def _load_source(source: str) -> dict:
    key = f"surface_hook_injection_{hashlib.sha256(source.encode()).hexdigest()}"
    return load_guppy_source(source, _get_temp_dir() / f"{key}.py", f"pecos._generated.{key}")


def load_surface_hook_injection_module(patch: SurfacePatch, verification_rounds: int = 2) -> dict:
    """Load reusable hook attempts and the heralded experiment factory."""
    return _load_source(render_surface_hook_injection_module(patch, verification_rounds))


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
