# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Render logical Pauli frame structs and helpers for surface-code Guppy modules."""

from pecos.qec.surface.patch import SurfacePatch


def render_frame_primitives(patch: SurfacePatch) -> list[str]:
    """Return the Guppy source lines of the logical Pauli frame block."""
    geom = patch.geometry
    lines = [
        "@guppy.struct",
        "class PauliFrame:",
        '    """Pending logical Pauli X^x Z^z on one patch, sign-free."""',
        "    x: bool",
        "    z: bool",
        "",
        "",
        "@guppy",
        "def frame_h(frame: PauliFrame) -> None:",
        "    x = frame.x",
        "    frame.x = frame.z",
        "    frame.z = x",
        "",
        "",
        "@guppy",
        "def frame_s(frame: PauliFrame) -> None:",
        "    frame.z = frame.z != frame.x",
        "",
        "",
        "@guppy",
        "def frame_cx(ctrl: PauliFrame, tgt: PauliFrame) -> None:",
        "    tgt.x = tgt.x != ctrl.x",
        "    ctrl.z = ctrl.z != tgt.z",
        "",
        "",
        "@guppy",
        "def frame_sz_teleportation(frame: PauliFrame, decision: bool) -> None:",
        '    """Net S on the frame, then the deferred Z byproduct."""',
        "    frame_s(frame)",
        "    frame.z = frame.z != decision",
        "",
        "",
    ]
    for basis, component, support, suffix in (
        ("z", "x", geom.logical_x.data_qubits, ""),
        ("x", "z", geom.logical_z.data_qubits, ""),
        ("z", "x", geom.logical_z.data_qubits, "_swapped"),
        ("x", "z", geom.logical_x.data_qubits, "_swapped"),
    ):
        orientation = "swapped" if suffix else "canonical"
        lines.extend(
            [
                "@guppy",
                f"def frame_correct_{basis}_readout{suffix}"
                f"(frame: PauliFrame, bits: array[bool, {geom.num_data}]) -> None:",
                f'    """Correct {basis.upper()} readout in the {orientation} orientation."""',
                f"    if frame.{component}:",
                *(f"        bits[{q}] = not bits[{q}]" for q in support),
                "",
                "",
            ],
        )
    return lines
