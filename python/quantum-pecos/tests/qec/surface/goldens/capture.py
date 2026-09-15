# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Capture both surface golden suites using the PECOS version in the environment.

Run from the repository root; see README.md for provenance and revision choice.
No git operations are performed. Existing files require explicit --force.
"""

import argparse
import json
from pathlib import Path

from pecos.guppy_gen.protocol_render import render_surface_protocol_module
from pecos.guppy_gen.surface import generate_guppy_source
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch, TwirlConfig
from pecos.qec.surface.circuit_builder import (
    QubitAllocation,
    build_surface_code_circuit,
    generate_tick_circuit_from_patch,
    tick_circuit_to_stim,
)

BASE_REVISION = "8568727d5"
POST_FIX_SHAPES = (
    "dx3dz1_mem_Z",
    "dx1dz3_mem_Z",
    "dx1dz3_mem_X",
    "d3_sz_teleport_first_op",
    "d3_sz_teleport_memory_first",
    "d3_t_inject_first_op",
    "d3_t_inject_data_memory_first",
)


META_KEYS = ("detectors", "observables", "num_measurements", "num_detectors", "basis")

PHASE_KEYS = ("phase", "syndrome_round", "cx_round")

OPS_NAMES = (
    "ops_d3_X_r0.json",
    "ops_d3_X_r1.json",
    "ops_d3_X_r3.json",
    "ops_d3_Z_r0.json",
    "ops_d3_Z_r1.json",
    "ops_d3_Z_r2_balanced.json",
    "ops_d3_Z_r2_budget2.json",
    "ops_d3_Z_r2_szz.json",
    "ops_d3_Z_r2_twirl_between.json",
    "ops_d3_Z_r2_twirl_gate.json",
    "ops_d3_Z_r3.json",
    "ops_d3_x_lower_r1.json",
    "ops_d3_z_lower_r1.json",
    "ops_d3nonrot_X_r1.json",
    "ops_d5_X_r1.json",
    "ops_d5_X_r3.json",
    "ops_d5_Z_r1.json",
    "ops_d5_Z_r3.json",
    "ops_dx1dz3_Z_r1.json",
    "ops_dx3dz5_Z_r1.json",
)

STIM_NAMES = (
    "stim_d3_X_r0.txt",
    "stim_d3_X_r1.txt",
    "stim_d3_X_r3.txt",
    "stim_d3_Z_r0.txt",
    "stim_d3_Z_r1.txt",
    "stim_d3_Z_r3.txt",
    "stim_d3nonrot_X_r1.txt",
    "stim_d5_X_r1.txt",
    "stim_d5_X_r3.txt",
    "stim_d5_Z_r1.txt",
    "stim_d5_Z_r3.txt",
    "stim_dx1dz3_Z_r1.txt",
    "stim_dx3dz5_Z_r1.txt",
)

GUPPY_NAMES = (
    "guppy_d3.py.txt",
    "guppy_d3nonrot.py.txt",
    "guppy_d5.py.txt",
    "guppy_d7.py.txt",
    "guppy_dx1dz3.py.txt",
    "guppy_dx3dz5.py.txt",
    "guppy_dx5dz3.py.txt",
)


def _patch(name: str) -> SurfacePatch:
    if name.startswith("dx"):
        dx, dz = name.removeprefix("dx").split("dz")
        return SurfacePatch.create(dx=int(dx), dz=int(dz))
    if name == "d3nonrot":
        return SurfacePatch.create(distance=3, rotated=False)
    return SurfacePatch.create(distance=int(name[1:]))


def _case(name: str) -> tuple:
    geometry, basis, *parts = name.split("_")
    rounds = int(next(part[1:] for part in parts if part.startswith("r")))
    kwargs = {}
    if "balanced" in parts:
        kwargs["check_plan"] = "cx_balanced_data_v1"
    if "budget2" in parts:
        kwargs["ancilla_budget"] = 2
    if "szz" in parts:
        kwargs["interaction_basis"] = "szz"
    if "twirl" in parts:
        kwargs["twirl"] = TwirlConfig(site_schedule="before_two_qubit_gate") if "gate" in parts else TwirlConfig()
    return _patch(geometry), rounds, basis, kwargs


def _ops(steps: list, allocation: QubitAllocation) -> dict:
    return {
        "allocation": {
            "data": allocation.data_qubits,
            "x_anc": allocation.x_ancilla_qubits,
            "z_anc": allocation.z_ancilla_qubits,
        },
        "ops": [{"op": s.op_type.name, "qubits": s.qubits, "label": s.label} for s in steps],
    }


SHAPES = (
    "dx3dz1_mem_Z",
    "dx1dz3_mem_Z",
    "dx1dz3_mem_X",
    "d3_mem_Z",
    "d3_mem_X",
    "d3_h_z_to_x",
    "d3_h_x_to_z",
    "d3_hh",
    "d3_cx_zz",
    "d3_cx_xx",
    "d3_cx_xz",
    "d3_cx_zx",
    "d3_h_cx_h",
    "d3_sz_teleport_first_op",
    "d3_sz_teleport_memory_first",
    "d3_t_inject_first_op",
    "d3_t_inject_data_memory_first",
    "d5_mem_Z",
    "d5_h",
    "d5_cx_zz",
)


def make_builder(name: str) -> LogicalCircuitBuilder:
    """Reproduce the orchestrator's captured recipes exactly."""
    if name in {"dx3dz1_mem_Z", "dx1dz3_mem_Z", "dx1dz3_mem_X"}:
        dx, dz = (3, 1) if name == "dx3dz1_mem_Z" else (1, 3)
        builder = LogicalCircuitBuilder()
        builder.add_patch(SurfacePatch.create(dx=dx, dz=dz), "A")
        builder.add_memory("A", 2, name[-1])
        return builder
    patch = SurfacePatch.create(distance=int(name[1]))
    shape = name[3:]
    if shape.startswith("sz_"):
        labels = ["D", "Y"]
    elif shape.startswith("t_"):
        labels = ["D", "A"]
    elif shape.startswith("cx_"):
        labels = ["C", "T"]
    elif shape == "h_cx_h":
        labels = ["A", "B"]
    else:
        labels = ["A"]
    builder = LogicalCircuitBuilder()
    for i, label in enumerate(labels):
        builder.add_patch(patch, label, qubit_offset=i * (patch.geometry.num_data + patch.geometry.num_ancilla))
    if shape.startswith("mem_"):
        builder.add_memory("A", 3 if name.startswith("d5") else 2, shape[-1])
    elif shape in {"h", "h_z_to_x", "h_x_to_z", "hh"}:
        before, after = ("X", "Z") if shape == "h_x_to_z" else ("Z", "X")
        builder.add_memory("A", 2, before)
        builder.add_transversal_h("A")
        builder.add_memory("A", 2, after)
        if shape == "hh":
            builder.add_transversal_h("A")
            builder.add_memory("A", 2, "Z")
    elif shape.startswith("cx_"):
        basis = {"C": shape[-2].upper(), "T": shape[-1].upper()}
        builder.add_memory(labels, 2, basis)
        builder.add_transversal_cx("C", "T")
        builder.add_memory(labels, 2, basis)
    elif shape == "h_cx_h":
        builder.add_memory(labels, 2, "Z")
        builder.add_transversal_h("A")
        builder.add_memory(labels, 2, {"A": "X", "B": "Z"})
        builder.add_transversal_h("A")
        builder.add_memory(labels, 2, "Z")
        builder.add_transversal_cx("A", "B")
        builder.add_memory(labels, 2, "Z")
    else:
        if "memory_first" in shape:
            builder.add_memory("D", 2, "Z")
        if shape.startswith("sz_"):
            builder.add_sz_via_teleportation("D", "Y", 2, 2)
            builder.add_memory("D", 2, "Z")
        else:
            builder.add_t_via_injection("D", "A", 2, 2)
    return builder


def golden_outputs(builder: LogicalCircuitBuilder) -> dict[str, str]:
    """Serialize raw metadata, including nested strings and nulls."""
    tc = builder.to_tick_circuit()
    return {
        "stim": builder.to_stim(),
        "noisy.stim": builder.to_stim(p1=0.001, p2=0.01, p_meas=0.005, p_prep=0.002),
        "tickmeta.json": json.dumps(
            {k: tc.get_meta(k) for k in ("detectors", "observables", "num_measurements", "num_detectors", "basis")}
            | {"num_ticks": tc.num_ticks()},
            indent=1,
        ),
    }


def capture_outputs() -> dict[str, str]:
    """Evaluate the explicit recipes for both golden directories."""
    outputs = {"gadget_parity/protocol_d3.py.txt": render_surface_protocol_module(SurfacePatch.create(distance=3))}
    for name in OPS_NAMES:
        patch, rounds, basis, kwargs = _case(name.removeprefix("ops_").removesuffix(".json"))
        steps, allocation = build_surface_code_circuit(patch, rounds, basis, **kwargs)
        outputs[f"gadget_parity/{name}"] = json.dumps(_ops(steps, allocation), indent=1)
    for name in STIM_NAMES:
        case = name.removeprefix("stim_").removesuffix(".txt")
        patch, rounds, basis, kwargs = _case(case)
        circuit = generate_tick_circuit_from_patch(patch, rounds, basis, **kwargs)
        outputs[f"gadget_parity/{name}"] = tick_circuit_to_stim(circuit)
        outputs[f"gadget_parity/tickmeta_{case}.json"] = json.dumps(
            {key: circuit.get_meta(key) for key in META_KEYS},
            indent=1,
        )
        outputs[f"gadget_parity/tickphases_{case}.json"] = json.dumps(
            [{key: circuit.get_tick_meta(i, key) for key in PHASE_KEYS} for i in range(circuit.num_ticks())],
            indent=1,
        )
    for name in GUPPY_NAMES:
        patch = _patch(name.removeprefix("guppy_").removesuffix(".py.txt"))
        outputs[f"gadget_parity/{name}"] = generate_guppy_source(patch)
    for shape in SHAPES:
        for suffix, contents in golden_outputs(make_builder(shape)).items():
            outputs[f"logical_builder/{shape}.{suffix}"] = contents
    return outputs


def main() -> None:
    """Write captures only after checking all selected output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--force", action="store_true", help="Allow replacement of existing golden files")
    parser.add_argument(
        "--post-fix-only",
        action="store_true",
        help="Capture only the protocol, injection and repetition post-change outputs",
    )
    args = parser.parse_args()
    outputs = capture_outputs()
    if args.post_fix_only:
        outputs = {
            name: contents
            for name, contents in outputs.items()
            if name == "gadget_parity/protocol_d3.py.txt"
            or any(name.startswith(f"logical_builder/{shape}.") for shape in POST_FIX_SHAPES)
        }
    if not args.force:
        existing = [name for name in outputs if (args.output_dir / name).exists()]
        if existing:
            parser.error(f"Refusing to overwrite {existing[0]}; use --force to replace existing files")
    for name, contents in outputs.items():
        path = args.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w" if args.force else "x", encoding="utf-8", newline="") as output:
            output.write(contents)


if __name__ == "__main__":
    main()
