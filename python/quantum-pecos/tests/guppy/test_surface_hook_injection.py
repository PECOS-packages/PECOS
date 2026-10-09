# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Compiled hook injection: accepted states, noisy flags, and safe failure paths."""

import math

import pecos_rslib as prs
import pytest
from pecos import sim, stab_vec
from pecos.guppy_gen import (
    load_surface_hook_injection_module,
    make_surface_hook_injection,
    make_surface_t_teleportation,
    render_surface_hook_injection_module,
)
from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.qec.surface import SurfacePatch, hook_injection


def _run(program, patch, *, shots=64, noisy=False):
    engine = prs.qis_engine().selene_runtime().interface(prs.qis_helios_interface())
    builder = (
        sim(program)
        .classical(engine)
        .quantum(stab_vec())
        .qubits(
            patch.geometry.num_qubits + patch.geometry.num_data,
        )
        .seed(917)
    )
    if noisy:
        builder = builder.noise(prs.depolarizing_noise().with_uniform_probability(0.005))
    return builder.run(shots).to_dict()


@pytest.mark.parametrize("state", ["T", "TDG"])
@pytest.mark.parametrize("basis", ["X", "Y"])
def test_true_magic_resource_tomography(state, basis):
    patch = SurfacePatch.create(distance=3)
    result = _run(make_surface_hook_injection(patch, state=state, readout_basis=basis), patch, shots=512)
    assert result["hook_accepted"] == [1] * 512
    support = patch.geometry.logical_x.data_qubits
    signs = [(-1) ** (sum(row[q] for q in support) % 2) for row in result["final_data"]]
    expected = (-1 if state == "TDG" and basis == "Y" else 1) / math.sqrt(2)
    assert sum(signs) / len(signs) == pytest.approx(expected, abs=0.12)
    assert {tuple(row[:4]) for row in result["hook_synx"]} == {(0, 0, 0, 0), (0, 1, 0, 0)}
    assert len({tuple(x[:4] + z[:4]) for x, z in zip(result["hook_synx"], result["hook_synz"], strict=True)}) == 16
    if basis == "Y":
        assert not any(any(row) for row in result["readout_synx"] + result["readout_synz"])


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_compile_at_generated_distances(distance):
    patch = SurfacePatch.create(distance=distance)
    assert make_surface_hook_injection(patch).compile() is not None
    assert make_surface_t_teleportation(patch, resource_preparation="hook", dagger=True).compile() is not None


@pytest.mark.parametrize("dagger", [False, True])
def test_hook_resource_teleportation_in_both_correction_branches(dagger):
    patch = SurfacePatch.create(distance=3)
    program = make_surface_t_teleportation(
        patch,
        1,
        1,
        input_state="X",
        readout_basis="Y",
        dagger=dagger,
        resource_preparation="hook",
    )
    result = _run(program, patch, shots=512)
    assert all(result["hook_accepted"])
    support = patch.geometry.logical_x.data_qubits
    for branch in (0, 1):
        signs = [
            (-1) ** (sum(bits[q] for q in support) % 2)
            for bits, correction in zip(result["final_data"], result["t_correction"], strict=True)
            if correction == branch
        ]
        assert len(signs) > 150
        assert sum(signs) / len(signs) == pytest.approx((-1 if dagger else 1) / math.sqrt(2), abs=0.17)
    for tag in ("data_synx", "data_synz", "correction_synx", "correction_synz", "resource_synx", "resource_synz"):
        assert not any(any(row) for row in result[tag])


def test_resources_compose_at_distance_five():
    patch = SurfacePatch.create(distance=5)
    source = render_surface_hook_injection_module(patch) + """
@guppy
def round_trip() -> None:
    data = prepare_injected_y()
    resource, first_ok = attempt_hook_t()
    if first_ok:
        apply_t_teleportation(data, resource, False)
        resource, second_ok = attempt_hook_tdg()
        if second_ok:
            apply_t_teleportation(data, resource, True)
        else:
            discarded = measure_z_basis(resource)
            output("discarded", discarded)
    else:
        discarded = measure_z_basis(resource)
        output("discarded", discarded)
    syn = syndrome_extraction_fold_szdg(data)
    output("readout_synx", syn.synx)
    output("readout_synz", syn.synz)
    final = measure_x_basis(data)
    output("final_data", final)
"""
    name = "test_hook_round_trip_d5"
    module = load_guppy_source(source, _get_temp_dir() / f"{name}.py", name)
    result = _run(module["round_trip"], patch)
    assert result["hook_accepted"] == [[1, 1]] * 64
    assert not any(sum(row[q] for q in patch.geometry.logical_x.data_qubits) % 2 for row in result["final_data"])
    assert not any(any(row) for row in result["readout_synx"] + result["readout_synz"])
    assert {tuple(row) for row in result["t_correction"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize("fault", ["X", "Y", "Z"])
def test_injected_fault_drives_the_actual_accept_or_discard_branch(fault):
    patch = SurfacePatch.create(distance=3)
    source = render_surface_hook_injection_module(patch)
    # Inject a definite Pauli immediately after the Y-state hook rotation.
    # Z must survive (as the wrong logical state); X and Y must be discarded.
    marker = "    s(az1)"
    assert source.count(marker) == 1
    gates = {"X": "    x(az1)", "Y": "    x(az1)\n    z(az1)", "Z": "    z(az1)"}
    source = source.replace(marker, marker + "\n" + gates[fault])
    name = f"test_hook_fault_{fault}"
    module = load_guppy_source(source, _get_temp_dir() / f"{name}.py", name)
    result = _run(module["make_hook_experiment"](state="Y", readout_basis="Y"), patch, shots=16)
    if fault == "Z":
        assert result["hook_accepted"] == [1] * 16
        assert all(sum(row[q] for q in patch.geometry.logical_x.data_qubits) % 2 for row in result["final_data"])
    else:
        assert result["hook_accepted"] == [0] * 16
        assert result["final_data"] == [[0] * 9] * 16


def test_rejected_resource_never_reaches_teleportation():
    patch = SurfacePatch.create(distance=3)
    source = render_surface_hook_injection_module(patch)
    marker = "    t(az1)"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + "\n    x(az1)")
    name = "test_hook_teleportation_rejection"
    module = load_guppy_source(source, _get_temp_dir() / f"{name}.py", name)
    result = _run(module["make_hook_experiment"](consume=True), patch, shots=16)
    assert result["hook_accepted"] == [0] * 16
    assert result["teleportation_performed"] == [0] * 16
    assert result["final_data"] == [[0] * 9] * 16
    assert result["injection_synx"] == [[0] * 4] * 16


@pytest.mark.parametrize("consume", [False, True])
def test_full_circuit_depolarizing_noise_acceptance_matches_records(consume):
    patch = SurfacePatch.create(distance=3)
    hook = hook_injection(patch)
    program = (
        make_surface_t_teleportation(patch, 1, 1, readout_basis="Y", resource_preparation="hook")
        if consume
        else make_surface_hook_injection(patch)
    )
    result = _run(program, patch, shots=256, noisy=True)
    assert len(result["final_data"]) == 256
    if consume:
        assert result["teleportation_performed"] == result["hook_accepted"]
    for accepted, bits in zip(result["hook_accepted"], result["final_data"], strict=True):
        if not accepted:
            assert bits == [0] * 9
    assert set(result["hook_accepted"]) == {0, 1}
    for xs, zs, accepted in zip(result["hook_synx"], result["hook_synz"], result["hook_accepted"], strict=True):
        records = tuple(
            (tuple(bool(bit) for bit in xs[start : start + 4]), tuple(bool(bit) for bit in zs[start : start + 4]))
            for start in range(0, 12, 4)
        )
        assert hook.accepts(records) == bool(accepted)


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_invalid_verification_count(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        load_surface_hook_injection_module(SurfacePatch.create(distance=3), rounds)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state": "Z"},
        {"readout_basis": "A"},
        {"consume": True, "state": "Y"},
        {"rounds_before": -1},
        {"rounds_after": True},
        {"input_state": "T"},
    ],
)
def test_invalid_experiment_arguments(kwargs):
    module = load_surface_hook_injection_module(SurfacePatch.create(distance=3))
    with pytest.raises(ValueError, match=r"Unsupported|must be|requires"):
        module["make_hook_experiment"](**kwargs)


def test_invalid_resource_preparation():
    with pytest.raises(ValueError, match="resource_preparation"):
        make_surface_t_teleportation(SurfacePatch.create(distance=3), resource_preparation="cultivation")
    with pytest.raises(ValueError, match="boolean"):
        make_surface_t_teleportation(SurfacePatch.create(distance=3), resource_preparation="hook", dagger=1)
