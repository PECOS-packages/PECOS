from __future__ import annotations

import pytest
from pecos.qec.surface import (
    LocalCliffordFrame,
    NoiseModel,
    OpType,
    SignedPauli,
    SurfacePatch,
    build_memory_circuit,
    build_surface_code_circuit,
    generate_tick_circuit_from_patch,
    global_surface_frame,
    resolve_surface_clifford_frame,
)
from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder


def test_identity_frame_resolves_to_css_surface_checks() -> None:
    patch = SurfacePatch.create(distance=3)

    resolved = resolve_surface_clifford_frame(patch, policy="identity")

    assert not resolved.requires_deformed_check_synthesis
    assert {check.uniform_axis for check in resolved.x_checks} == {"X"}
    assert {check.uniform_axis for check in resolved.z_checks} == {"Z"}
    assert resolved.logical_x.uniform_axis == "X"
    assert resolved.logical_z.uniform_axis == "Z"
    assert resolved.css_physical_memory_basis("X") == "X"
    assert resolved.css_physical_memory_basis("Z") == "Z"


def test_global_h_frame_resolves_to_css_basis_swap() -> None:
    patch = SurfacePatch.create(distance=3)

    resolved = resolve_surface_clifford_frame(patch, policy="global-h")

    assert not resolved.requires_deformed_check_synthesis
    assert {check.uniform_axis for check in resolved.x_checks} == {"Z"}
    assert {check.uniform_axis for check in resolved.z_checks} == {"X"}
    assert resolved.logical_x.uniform_axis == "Z"
    assert resolved.logical_z.uniform_axis == "X"
    assert resolved.css_physical_memory_basis("X") == "Z"
    assert resolved.css_physical_memory_basis("Z") == "X"


def test_axis_cycle_frames_resolve_but_require_deformed_checks() -> None:
    patch = SurfacePatch.create(distance=3)

    resolved_f = resolve_surface_clifford_frame(patch, policy="global_axis_cycle_f")
    resolved_f2 = resolve_surface_clifford_frame(patch, policy="global_axis_cycle_f2")

    assert resolved_f.requires_deformed_check_synthesis
    assert {check.uniform_axis for check in resolved_f.x_checks} == {"Y"}
    assert {check.uniform_axis for check in resolved_f.z_checks} == {"X"}
    assert resolved_f.logical_x.uniform_axis == "Y"
    assert resolved_f.logical_z.uniform_axis == "X"

    assert resolved_f2.requires_deformed_check_synthesis
    assert {check.uniform_axis for check in resolved_f2.x_checks} == {"Z"}
    assert {check.uniform_axis for check in resolved_f2.z_checks} == {"Y"}
    assert resolved_f2.logical_x.uniform_axis == "Z"
    assert resolved_f2.logical_z.uniform_axis == "Y"

    with pytest.raises(NotImplementedError, match="deformed check synthesis"):
        resolved_f.css_physical_memory_basis("Z")
    with pytest.raises(NotImplementedError, match="deformed check synthesis"):
        resolved_f2.css_physical_memory_basis("X")


def test_explicit_mixed_local_frame_marks_only_mixed_checks_deformed() -> None:
    patch = SurfacePatch.create(distance=3)
    frames = list(global_surface_frame("identity", patch.num_data))
    frames[0] = LocalCliffordFrame(SignedPauli("Z"), SignedPauli("X"))

    resolved = resolve_surface_clifford_frame(
        patch,
        policy="identity",
        data_frames=frames,
    )

    assert resolved.requires_deformed_check_synthesis
    assert any(check.requires_deformed_check_synthesis for check in resolved.checks)
    assert any(not check.is_uniform_axis for check in resolved.checks)
    assert any(check.axes == ("Z", "X") for check in resolved.x_checks if len(check.axes) == 2)


def test_rejects_frame_map_with_wrong_length() -> None:
    patch = SurfacePatch.create(distance=3)
    frames = global_surface_frame("identity", patch.num_data - 1)

    with pytest.raises(ValueError, match=r"does not match patch\.num_data"):
        resolve_surface_clifford_frame(patch, data_frames=frames)


def test_global_axis_cycle_f_emits_uniform_y_szz_check_scaffold() -> None:
    patch = SurfacePatch.create(distance=3)

    ops, _allocation = build_surface_code_circuit(
        patch,
        num_rounds=1,
        basis="Z",
        interaction_basis="szz",
        clifford_frame_policy="global_axis_cycle_f",
    )

    assert any(op.op_type == OpType.SXDG and "szz_y_touch_pre:X" in op.label for op in ops)
    assert any(op.op_type == OpType.SX and "szz_y_touch_post:X" in op.label for op in ops)
    assert any(op.op_type == OpType.SYDG and "szz_touch_comp:SYDG:Y:X" in op.label for op in ops)
    assert any(op.op_type == OpType.H and op.label == "prep_x_basis_d0:to_z" for op in ops)
    assert any(op.op_type == OpType.H and op.label == "measure_x_basis_d0:from_z" for op in ops)
    assert any(op.op_type == OpType.MEASURE and op.label.startswith("sx") for op in ops)
    assert any(op.op_type == OpType.MEASURE and op.label.startswith("sz") for op in ops)


def test_global_axis_cycle_f_tick_circuit_keeps_source_detector_metadata() -> None:
    patch = SurfacePatch.create(distance=3)

    tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        basis="Z",
        interaction_basis="szz",
        clifford_frame_policy="global_axis_cycle_f",
    )

    detectors = tick_circuit.get_meta("detectors")
    observables = tick_circuit.get_meta("observables")
    assert detectors
    assert observables
    assert tick_circuit.get_meta("basis") == "Z"


def test_global_axis_cycle_f_native_abstract_dem_path_accepts_frame_policy() -> None:
    patch = SurfacePatch.create(distance=3)

    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=NoiseModel(p1=0.0, p2=0.0, p_meas=0.0, p_prep=0.0),
        basis="Z",
        circuit_source="abstract",
        interaction_basis="szz",
        clifford_frame_policy="global_axis_cycle_f",
    )

    assert isinstance(dem, str)


def test_clifford_frame_policy_rejects_traced_qis_until_guppy_matches_frame() -> None:
    with pytest.raises(NotImplementedError, match="requires circuit_source='abstract'"):
        build_memory_circuit(
            distance=3,
            rounds=1,
            basis="Z",
            circuit_source="traced_qis",
            interaction_basis="szz",
            clifford_frame_policy="global_axis_cycle_f",
        )
