# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Compiled adaptive injection: branch-resolved tomography and composability."""

import math

import pecos_rslib as prs
import pytest
from pecos import sim, stab_vec
from pecos.guppy_gen import (
    load_surface_protocol_module,
    load_surface_t_teleportation_module,
    make_surface_t_teleportation,
)
from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.surface_teleportation import render_surface_t_teleportation_module
from pecos.qec import build_dem_from_guppy
from pecos.qec.surface import SurfacePatch


def _run(program, patch, shots=512):
    # Explicitly choose both runtime and interface, including on builds without
    # a default QIS interface.
    engine = prs.qis_engine().selene_runtime().interface(prs.qis_helios_interface())
    return (
        sim(program)
        .classical(engine)
        .quantum(stab_vec())
        .qubits(patch.geometry.num_data + patch.geometry.num_qubits)
        .seed(712)
        .run(shots)
        .to_dict()
    )


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize(("state", "basis"), [("X", "X"), ("X", "Y"), ("Y", "X"), ("Z", "Z")])
def test_tomography_in_both_feedforward_branches(state, basis, dagger):
    patch = SurfacePatch.create(distance=3)
    program = make_surface_t_teleportation(patch, 1, 1, input_state=state, readout_basis=basis, dagger=dagger)
    results = _run(program, patch)
    support = patch.geometry.logical_z.data_qubits if basis == "Z" else patch.geometry.logical_x.data_qubits
    rotation_sign = -1 if dagger else 1
    expected = {
        ("X", "X"): 1 / math.sqrt(2),
        ("X", "Y"): rotation_sign / math.sqrt(2),
        ("Y", "X"): -rotation_sign / math.sqrt(2),
        ("Z", "Z"): 1,
    }[state, basis]
    for branch in (0, 1):
        values = [
            (-1) ** (sum(bits[q] for q in support) % 2)
            for bits, correction in zip(results["final_data"], results["t_correction"], strict=True)
            if correction == branch
        ]
        assert len(values) > 150
        assert sum(values) / len(values) == pytest.approx(expected, abs=0.16)
    for tag in ("correction_synx", "correction_synz", "data_synx", "data_synz", "resource_synx", "resource_synz"):
        assert not any(any(row) for row in results[tag])


@pytest.mark.parametrize("distance", [3, 5])
def test_reusable_gadget_preserves_data_through_t_then_tdg(distance):
    patch = SurfacePatch.create(distance=distance)
    source = render_surface_t_teleportation_module(patch) + """
@guppy
def round_trip() -> None:
    data = prepare_injected_y()
    resource = prepare_injected_t()
    apply_t_teleportation(data, resource, False)
    resource = prepare_injected_tdg()
    apply_t_teleportation(data, resource, True)
    syn = syndrome_extraction_fold_szdg(data)
    output("readout_synx", syn.synx)
    output("readout_synz", syn.synz)
    final = measure_x_basis(data)
    output("final_data", final)
"""
    name = f"test_t_teleportation_round_trip_d{distance}"
    module = load_guppy_source(source, _get_temp_dir() / f"{name}.py", name)
    results = _run(module["round_trip"], patch, shots=64)
    support = patch.geometry.logical_x.data_qubits
    assert all(sum(bits[q] for q in support) % 2 == 0 for bits in results["final_data"])
    assert {tuple(row) for row in results["t_correction"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_compile_and_protocol_entry_point(distance):
    patch = SurfacePatch.create(distance=distance)
    module = load_surface_t_teleportation_module(patch)
    assert module["make_t_teleportation"](0, 0).compile() is not None
    protocol = load_surface_protocol_module(patch)
    assert protocol["make_t_teleportation"](0, 0, dagger=True).compile() is not None
    assert protocol["make_t_teleportation_placeholder"](1, 1).compile() is not None


@pytest.mark.parametrize(
    "kwargs",
    [{"rounds_before": -1}, {"rounds_after": 1.5}, {"input_state": "T"}, {"readout_basis": "A"}, {"dagger": 1}],
)
def test_invalid_factory_arguments(kwargs):
    with pytest.raises(ValueError, match=r"(round counts|input_state|readout_basis|dagger)"):
        make_surface_t_teleportation(SurfacePatch.create(distance=3), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"distance": 1}, {"distance": 2}, {"dx": 3, "dz": 5}, {"distance": 3, "rotated": False}],
)
def test_correction_geometry_restrictions(kwargs):
    with pytest.raises(ValueError, match="odd square rotated"):
        make_surface_t_teleportation(SurfacePatch.create(**kwargs))


def test_adaptive_injection_rejects_static_dem_route():
    program = make_surface_t_teleportation(SurfacePatch.create(distance=3))
    with pytest.raises(ValueError, match="statically straight-line"):
        build_dem_from_guppy(program, num_qubits=26, detectors=[], observables=[])
