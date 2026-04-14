# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Invariant checks for PECOS native decomposed DEM output.

These tests focus on algorithmic correctness of the decomposition itself:
- every decomposed component emitted for MWPM is graphlike
- XOR of decomposed components maps back to an effect present in the full DEM
- representative graphlike L0 singleton edges match Stim's decomposition
"""

import json
import re
from functools import cache, lru_cache

import pytest

stim = pytest.importorskip("stim")

DIRECT_SOURCE_TYPES = {"Direct", "DirectOneSidedComponent"}
FAST_DISTANCE = pytest.param(3, id="3")
SLOW_DISTANCE = pytest.param(9, marks=pytest.mark.slow, id="9")


def parse_dem_with_decomposed(
    dem_str: str,
) -> tuple[set[tuple[tuple[int, ...], tuple[int, ...]]], list[list[tuple[tuple[int, ...], tuple[int, ...]]]]]:
    """Parse direct and decomposed error targets from a DEM string."""
    direct_targets: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    decomposed_targets: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = []

    for raw_line in dem_str.strip().split("\n"):
        line = raw_line.strip()
        if not line.startswith("error("):
            continue

        match = re.match(r"error\(([^)]+)\)\s+(.*)", line)
        if not match:
            continue

        rest = match.group(2)
        parts = [part.strip() for part in rest.split("^")]
        parsed_parts = []
        for part in parts:
            dets = tuple(sorted(int(m.group(1)) for m in re.finditer(r"D(\d+)", part)))
            logs = tuple(sorted(int(m.group(1)) for m in re.finditer(r"L(\d+)", part)))
            parsed_parts.append((dets, logs))

        if len(parsed_parts) == 1:
            direct_targets.add(parsed_parts[0])
        else:
            decomposed_targets.append(parsed_parts)

    return direct_targets, decomposed_targets


def xor_targets(parts: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """XOR a decomposed list of detector/logical targets into one combined effect."""
    dets: set[int] = set()
    logs: set[int] = set()
    for part_dets, part_logs in parts:
        for det in part_dets:
            if det in dets:
                dets.remove(det)
            else:
                dets.add(det)
        for log in part_logs:
            if log in logs:
                logs.remove(log)
            else:
                logs.add(log)
    return tuple(sorted(dets)), tuple(sorted(logs))


def detector_union(parts: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> set[int]:
    """Return the union of detector ids touched by a decomposed effect."""
    out: set[int] = set()
    for dets, _logs in parts:
        out.update(dets)
    return out


def singleton_l0_edges(direct_targets: set[tuple[tuple[int, ...], tuple[int, ...]]]) -> set[int]:
    """Collect singleton detector edges that also flip logical L0."""
    return {dets[0] for dets, logs in direct_targets if len(dets) == 1 and len(logs) == 1}


def xor_lists(left: list[int], right: list[int]) -> list[int]:
    """XOR two integer lists interpreted as parity sets."""
    out = set(left)
    for value in right:
        if value in out:
            out.remove(value)
        else:
            out.add(value)
    return sorted(out)


def xor_effect_rows(left: dict[str, list[int]], right: dict[str, list[int]]) -> tuple[list[int], list[int]]:
    """XOR two structured detector/logical rows."""
    return (
        xor_lists(left["detectors"], right["detectors"]),
        xor_lists(left["logicals"], right["logicals"]),
    )


def parse_dem_error_probabilities(dem_str: str) -> dict[str, float]:
    """Map DEM target strings to their stated error probabilities."""
    out: dict[str, float] = {}
    for raw_line in dem_str.strip().split("\n"):
        line = raw_line.strip()
        if not line.startswith("error("):
            continue
        match = re.match(r"error\(([^)]+)\)\s+(.*)", line)
        if not match:
            continue
        out[match.group(2).strip()] = float(match.group(1))
    return out


def combine_independent_probs(left: float, right: float) -> float:
    """Combine independent error probabilities landing on the same rendered term."""
    return left + right - left * right


def combine_xor_probs(left: float, right: float) -> float:
    """Combine probabilities for XOR-composed contributions."""
    return left * (1.0 - right) + right * (1.0 - left)


@cache
def build_source_tracked_dem(distance: int, basis: str, rounds: int = 20) -> object:
    """Build and cache a source-tracked native DEM for one surface-code shape."""
    from pecos.qec import DagFaultAnalyzer, DemBuilder
    from pecos.qec.surface import (
        NoiseModel,
        SurfacePatch,
        generate_tick_circuit_from_patch,
        get_measurement_order_from_tick_circuit,
    )

    patch = SurfacePatch.create(distance=distance)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=rounds, basis=basis)
    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()
    noise = NoiseModel(p1=0.01, p2=0.01, p_meas=0.01, p_init=0.01)

    builder = DemBuilder(influence_map)
    builder.with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_init)
    builder.with_num_measurements(int(tc.get_meta("num_measurements") or "0"))
    builder.with_measurement_order(get_measurement_order_from_tick_circuit(tc))
    builder.with_detectors_json(tc.get_meta("detectors"))
    observables_json = tc.get_meta("observables")
    if observables_json:
        builder.with_observables_json(observables_json)
    return builder.build()


def test_dem_builder_accepts_public_surface_descriptor_json() -> None:
    """Public surface descriptor JSON should reproduce the legacy builder output."""
    from pecos.qec import DagFaultAnalyzer, DemBuilder
    from pecos.qec.surface import (
        NoiseModel,
        SurfacePatch,
        generate_tick_circuit_from_patch,
        get_detector_descriptors_from_tick_circuit,
        get_measurement_order_from_tick_circuit,
        get_observable_descriptors_from_tick_circuit,
    )

    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=4, basis="X")
    dag = tc.to_dag_circuit()
    influence_map = DagFaultAnalyzer(dag).build_influence_map()
    noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001)

    def _build(detectors_json: str, observables_json: str | None) -> object:
        """Build one source-tracked DEM from serialized detector metadata."""
        builder = DemBuilder(influence_map)
        builder.with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_init)
        builder.with_num_measurements(int(tc.get_meta("num_measurements") or "0"))
        builder.with_measurement_order(get_measurement_order_from_tick_circuit(tc))
        builder.with_detectors_json(detectors_json)
        if observables_json:
            builder.with_observables_json(observables_json)
        return builder.build_with_source_tracking()

    legacy_dem = _build(tc.get_meta("detectors"), tc.get_meta("observables"))
    public_dem = _build(
        json.dumps(get_detector_descriptors_from_tick_circuit(tc, patch)),
        json.dumps(get_observable_descriptors_from_tick_circuit(tc, patch)),
    )

    assert public_dem.to_string() == legacy_dem.to_string()
    assert public_dem.num_contributions == legacy_dem.num_contributions
    assert public_dem.all_contribution_effects() == legacy_dem.all_contribution_effects()


def _find_gate_attrs(
    dag: object,
    gate_type: str,
    *,
    phase: str | None = None,
    label_prefix: str | None = None,
    stabilizer: str | None = None,
) -> dict[str, object]:
    """Find the first DAG gate attribute record matching the requested filters."""
    for node in sorted(dag.nodes()):
        gate = dag.gate(node)
        attrs = dag.gate_attrs(node) or {}
        if gate is None or gate.gate_type.name != gate_type:
            continue
        if phase is not None and attrs.get("phase") != phase:
            continue
        if label_prefix is not None and not str(attrs.get("label", "")).startswith(label_prefix):
            continue
        if stabilizer is not None and attrs.get("stabilizer") != stabilizer:
            continue
        return attrs
    msg = (
        f"no gate attrs found for gate_type={gate_type!r}, phase={phase!r}, "
        f"label_prefix={label_prefix!r}, stabilizer={stabilizer!r}"
    )
    raise AssertionError(msg)


def test_surface_tick_gate_metadata_preserves_phase_round_context_in_dag() -> None:
    """Surface DAG metadata should retain round and stabilizer context."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch

    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")
    dag = tc.to_dag_circuit()

    h_pre = _find_gate_attrs(dag, "H", phase="syndrome_h_pre")
    assert h_pre["phase"] == "syndrome_h_pre"
    assert h_pre["syndrome_round"] == 0
    assert "cx_round" not in h_pre

    ancilla_reset = _find_gate_attrs(dag, "PZ", phase="syndrome_prep", label_prefix="ax")
    assert ancilla_reset["phase"] == "syndrome_prep"
    assert ancilla_reset["syndrome_round"] == 1
    assert "cx_round" not in ancilla_reset
    assert ancilla_reset["stabilizer"] == "X0"
    assert ancilla_reset["stabilizer_kind"] == "X"
    assert ancilla_reset["stabilizer_index"] == 0
    assert ancilla_reset["stabilizer_is_boundary"] is True
    assert ancilla_reset["stabilizer_region"]
    assert ancilla_reset["ancilla_qubit"] >= patch.num_data

    cx = _find_gate_attrs(dag, "CX", phase="cx_round_1")
    assert cx["phase"] == "cx_round_1"
    assert cx["syndrome_round"] == 0
    assert cx["cx_round"] == 1
    assert str(cx["stabilizer"]).startswith(("X", "Z"))
    assert cx["stabilizer_kind"] in {"X", "Z"}
    assert isinstance(cx["stabilizer_index"], int)
    assert isinstance(cx["stabilizer_is_boundary"], bool)
    assert cx["stabilizer_region"]
    assert cx["touch_label"] in {"TL", "TR", "BL", "BR", "top", "bottom", "left", "right"}
    assert cx["cx_round_0based"] == 0
    assert cx["ancilla_qubit"] >= patch.num_data
    assert cx["data_qubit"] < patch.num_data

    ancilla_measure = _find_gate_attrs(dag, "MZ", phase="measure_ancilla", label_prefix="sx")
    assert ancilla_measure["phase"] == "measure_ancilla"
    assert ancilla_measure["syndrome_round"] == 0
    assert ancilla_measure["cx_round"] == 4
    assert ancilla_measure["stabilizer"] == "X0"


def test_surface_tick_gate_metadata_tracks_reused_ancillas_by_label() -> None:
    """Ancilla labels should keep metadata stable even when qubits are reused."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch

    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        basis="X",
        ancilla_budget=2,
    )
    dag = tc.to_dag_circuit()

    first_alloc = _find_gate_attrs(dag, "QAlloc", phase="syndrome_prep", label_prefix="ax0")
    reused_reset = _find_gate_attrs(dag, "PZ", phase="syndrome_prep", label_prefix="ax1")
    reused_cx = _find_gate_attrs(dag, "CX", phase="cx_round_1", stabilizer="X1")

    assert first_alloc["stabilizer"] == "X0"
    assert reused_reset["stabilizer"] == "X1"
    assert reused_reset["ancilla_qubit"] == first_alloc["ancilla_qubit"]
    assert reused_reset["ancilla_qubit"] == patch.num_data
    assert reused_cx["stabilizer"] == "X1"
    assert reused_cx["ancilla_qubit"] == patch.num_data


@pytest.mark.parametrize("distance", [FAST_DISTANCE, SLOW_DISTANCE])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_decomposed_components_are_graphlike_and_map_back_to_full_dem(distance: int, basis: str) -> None:
    """Native decomposed graphlike pieces should reconstruct a full DEM effect."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

    patch = SurfacePatch.create(distance=distance)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=20, basis=basis)
    params = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

    full_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=False)
    native_decomp_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=True)

    full_targets, _ = parse_dem_with_decomposed(full_dem)
    _direct_targets, decomposed_targets = parse_dem_with_decomposed(native_decomp_dem)

    assert decomposed_targets, "expected representative circuit to contain decomposed terms"

    saw_l0_decomposition = False
    for parts in decomposed_targets:
        for dets, logs in parts:
            assert len(dets) <= 2, f"component is not graphlike by detector count: {parts!r}"
            assert len(logs) <= 1, f"component is not graphlike by logical count: {parts!r}"

        combined = xor_targets(parts)
        msg = f"decomposed components must XOR back to an effect present in the full DEM: {parts!r} -> {combined!r}"
        assert combined in full_targets, msg

        if combined[1]:
            saw_l0_decomposition = True

    assert saw_l0_decomposition, "expected representative circuit to include decomposed L0 terms"


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_decomposed_matches_stim_singleton_l0_edges_for_representative_circuit(basis: str) -> None:
    """Native decomposition should preserve Stim's singleton logical-observable edges."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit,
        generate_dem_from_tick_circuit_via_stim,
    )

    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=20, basis=basis)
    params = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

    native_decomp_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=True)
    stim_dem = generate_dem_from_tick_circuit_via_stim(tc, **params)

    native_direct, _ = parse_dem_with_decomposed(native_decomp_dem)
    stim_direct, _ = parse_dem_with_decomposed(stim_dem)

    assert singleton_l0_edges(native_direct) == singleton_l0_edges(stim_direct)


@pytest.mark.parametrize("distance", [FAST_DISTANCE, SLOW_DISTANCE])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_decomposed_preserves_all_stim_direct_observable_targets(distance: int, basis: str) -> None:
    """Native decomposition should include every direct observable target Stim exposes."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit,
        generate_dem_from_tick_circuit_via_stim,
    )

    patch = SurfacePatch.create(distance=distance)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=20, basis=basis)
    params = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

    native_decomp_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=True)
    stim_dem = generate_dem_from_tick_circuit_via_stim(tc, **params)

    native_direct, _ = parse_dem_with_decomposed(native_decomp_dem)
    stim_direct, _ = parse_dem_with_decomposed(stim_dem)

    native_observable_direct = {target for target in native_direct if target[1]}
    stim_observable_direct = {target for target in stim_direct if target[1]}

    assert stim_observable_direct.issubset(native_observable_direct)
    assert all(len(dets) == 2 and len(logs) == 1 for dets, logs in native_observable_direct - stim_observable_direct)


@pytest.mark.parametrize("distance", [FAST_DISTANCE, SLOW_DISTANCE])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_native_full_matches_stim_full_graph_summary_for_representative_circuit(distance: int, basis: str) -> None:
    """Full native and Stim DEMs should produce matching PyMatching graph summaries."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit,
        generate_dem_from_tick_circuit_via_stim,
    )
    from pecos_rslib.decoders import PyMatchingDecoder

    patch = SurfacePatch.create(distance=distance)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=20, basis=basis)
    params = {"p1": 0.01, "p2": 0.01, "p_meas": 0.01, "p_init": 0.01}

    native_full_dem = generate_dem_from_tick_circuit(tc, **params, decompose_errors=False)
    stim_full_dem = generate_dem_from_tick_circuit_via_stim(tc, **params, decompose_errors=False)

    native_graph = PyMatchingDecoder.from_dem(native_full_dem)
    stim_graph = PyMatchingDecoder.from_dem(stim_full_dem)

    assert native_graph.num_detectors == stim_graph.num_detectors
    assert native_graph.num_edges == stim_graph.num_edges
    assert native_graph.num_observables == stim_graph.num_observables


def test_generate_dem_from_tick_circuit_via_stim_can_skip_decomposition() -> None:
    """Stim DEM generation should honor the explicit non-decomposed option."""
    from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_stim

    patch = SurfacePatch.create(distance=3)
    tc = generate_tick_circuit_from_patch(patch, num_rounds=4, basis="X")

    dem = generate_dem_from_tick_circuit_via_stim(
        tc,
        p1=0.01,
        p2=0.01,
        p_meas=0.01,
        p_init=0.01,
        decompose_errors=False,
    )

    assert "error(" in dem
    assert "^" not in dem


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_source_tracking_summaries_include_graphlike_decomposable_count(basis: str) -> None:
    """Structured summaries should expose graphlike decomposition counts."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    summaries = dem.contribution_effect_summaries()
    assert summaries

    pair_summaries = [summary for summary in summaries if len(summary["detectors"]) == 2 and not summary["logicals"]]
    assert pair_summaries
    assert all("graphlike_decomposable_count" in summary for summary in pair_summaries)
    assert all(summary["graphlike_decomposable_count"] >= 0 for summary in pair_summaries)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_source_tracking_bindings_are_self_consistent(basis: str) -> None:
    """Structured contribution rows should sum back to their effect summaries."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    summaries = dem.contribution_effect_summaries()
    assert summaries

    observable_summary = next(row for row in summaries if row["logicals"])
    contributions = dem.contributions_for_effect(
        observable_summary["detectors"],
        observable_summary["logicals"],
    )

    assert contributions
    assert len(contributions) == observable_summary["num_contributions"]

    total_probability = sum(float(row["probability"]) for row in contributions)
    direct_rows = [row for row in contributions if row["source_type"] in DIRECT_SOURCE_TYPES]
    y_rows = [row for row in contributions if row["source_type"] == "YDecomposed"]
    assert all(row["location_indices"] for row in contributions)
    assert all(row["pauli_labels"] for row in contributions)
    assert all("gate_type_labels" in row for row in contributions)
    assert all("before_flags" in row for row in contributions)
    assert all(len(row["location_indices"]) == len(row["pauli_labels"]) for row in contributions)
    assert all(len(row["location_indices"]) == len(row["gate_type_labels"]) for row in contributions)
    assert all(len(row["location_indices"]) == len(row["before_flags"]) for row in contributions)
    assert all(all(label in {"I", "X", "Y", "Z"} for label in row["pauli_labels"]) for row in contributions)
    assert all(all(label for label in row["gate_type_labels"]) for row in contributions)
    assert total_probability == pytest.approx(observable_summary["total_probability"])
    assert len(direct_rows) == observable_summary["direct_count"]
    assert sum(float(row["probability"]) for row in direct_rows) == pytest.approx(
        observable_summary["direct_probability"],
    )
    assert len(y_rows) == observable_summary["y_decomposed_count"]
    assert sum(float(row["probability"]) for row in y_rows) == pytest.approx(
        observable_summary["y_decomposed_probability"],
    )


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_source_tracking_y_decomposed_rows_xor_back_to_effect(basis: str) -> None:
    """Y-decomposed structured rows should XOR back to their parent effect."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    summaries = [row for row in dem.contribution_effect_summaries() if row["y_decomposed_count"] > 0]
    assert summaries

    for summary in summaries[:20]:
        contributions = dem.contributions_for_effect(summary["detectors"], summary["logicals"])
        y_rows = [row for row in contributions if row["source_type"] == "YDecomposed"]
        assert y_rows
        for row in y_rows:
            assert xor_lists(row["x_detectors"], row["z_detectors"]) == summary["detectors"]
            assert xor_lists(row["x_logicals"], row["z_logicals"]) == summary["logicals"]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_direct_component_rows_xor_back_to_effect(basis: str) -> None:
    """Stored direct components should reconstruct the parent effect via XOR."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    rows = []
    for summary in dem.contribution_effect_summaries():
        for row in dem.contributions_for_effect(summary["detectors"], summary["logicals"]):
            if row["source_type"] not in DIRECT_SOURCE_TYPES:
                continue
            if "component_1_detectors" not in row or "component_2_detectors" not in row:
                continue
            rows.append((summary, row))

    assert rows

    for summary, row in rows[:100]:
        left = {
            "detectors": row["component_1_detectors"],
            "logicals": row["component_1_logicals"],
        }
        right = {
            "detectors": row["component_2_detectors"],
            "logicals": row["component_2_logicals"],
        }
        dets, logs = xor_effect_rows(left, right)
        assert dets == summary["detectors"]
        assert logs == summary["logicals"]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_one_sided_direct_component_rows_are_exposed(basis: str) -> None:
    """One-sided direct components should remain visible in the structured bindings."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    rows = []
    for summary in dem.contribution_effect_summaries():
        for row in dem.contributions_for_effect(summary["detectors"], summary["logicals"]):
            if row["source_type"] != "DirectOneSidedComponent":
                continue
            rows.append((summary, row))

    assert rows

    for summary, row in rows[:100]:
        left_non_empty = bool(row["component_1_detectors"] or row["component_1_logicals"])
        right_non_empty = bool(row["component_2_detectors"] or row["component_2_logicals"])
        assert left_non_empty != right_non_empty
        assert row["direct_source_family"] == "TwoLocationOneSidedComponent"
        direct_dets, direct_logs = xor_effect_rows(
            {
                "detectors": row["component_1_detectors"],
                "logicals": row["component_1_logicals"],
            },
            {
                "detectors": row["component_2_detectors"],
                "logicals": row["component_2_logicals"],
            },
        )
        assert direct_dets == summary["detectors"]
        assert direct_logs == summary["logicals"]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_direct_source_families_are_exposed_for_direct_rows(basis: str) -> None:
    """Direct structured rows should advertise their underlying source family."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    rows = []
    for summary in dem.contribution_effect_summaries():
        rows.extend(
            row
            for row in dem.contributions_for_effect(summary["detectors"], summary["logicals"])
            if row["source_type"] in DIRECT_SOURCE_TYPES
        )

    assert rows
    assert all("direct_source_family" in row for row in rows)
    assert any(row["direct_source_family"] == "SingleLocationY" for row in rows)
    assert any(row["direct_source_family"] == "TwoLocationOneSidedComponent" for row in rows)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_source_tracking_summaries_partition_all_contributions(basis: str) -> None:
    """Effect summaries should form a lossless partition of all contributions."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    summaries = dem.contribution_effect_summaries()
    assert summaries

    effect_keys = {(tuple(summary["detectors"]), tuple(summary["logicals"])) for summary in summaries}
    assert len(effect_keys) == len(summaries)

    total_count = 0
    total_probability = 0.0
    for summary in summaries:
        contributions = dem.contributions_for_effect(summary["detectors"], summary["logicals"])
        total_count += len(contributions)
        total_probability += sum(float(row["probability"]) for row in contributions)
        assert all(row["detectors"] == summary["detectors"] for row in contributions)
        assert all(row["logicals"] == summary["logicals"] for row in contributions)

    assert total_count == dem.num_contributions
    assert total_count == sum(int(summary["num_contributions"]) for summary in summaries)
    assert total_probability == pytest.approx(sum(float(summary["total_probability"]) for summary in summaries))


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_render_summaries_reproduce_decomposed_regrouping(basis: str) -> None:
    """Render summaries should regroup into the same decomposed DEM probabilities."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    render_summaries = dem.contribution_render_summaries()
    assert render_summaries
    assert sum(int(row["num_contributions"]) for row in render_summaries) == dem.num_contributions

    decomposed_by_targets = parse_dem_error_probabilities(dem.to_string_decomposed())
    regrouped_from_summaries: dict[str, float] = {}
    for row in render_summaries:
        regrouped_from_summaries[row["rendered_targets"]] = combine_independent_probs(
            regrouped_from_summaries.get(row["rendered_targets"], 0.0),
            float(row["combined_probability"]),
        )

    assert set(regrouped_from_summaries) == set(decomposed_by_targets)
    for targets, probability in regrouped_from_summaries.items():
        assert probability == pytest.approx(decomposed_by_targets[targets], abs=5e-7)

    assert all("source_type_counts" in row for row in render_summaries)
    assert any("YDecomposed" in row["source_type_counts"] for row in render_summaries)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_render_records_reproduce_render_summaries(basis: str) -> None:
    """Per-contribution render records should rebuild the grouped render summaries."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    render_records = dem.contribution_render_records()
    render_summaries = dem.contribution_render_summaries()

    assert render_records
    assert len(render_records) == dem.num_contributions
    assert all("rendered_targets" in row for row in render_records)
    assert all("render_strategy" in row for row in render_records)
    assert any("recorded_component_targets" in row for row in render_records)

    regrouped: dict[tuple[tuple[int, ...], tuple[int, ...], str], dict[str, object]] = {}
    for row in render_records:
        key = (
            tuple(row["detectors"]),
            tuple(row["logicals"]),
            str(row["rendered_targets"]),
        )
        bucket = regrouped.setdefault(
            key,
            {
                "num_contributions": 0,
                "total_probability": 0.0,
                "combined_probability": 0.0,
                "source_type_counts": {},
                "source_type_probabilities": {},
                "direct_source_family_counts": {},
                "direct_source_family_probabilities": {},
            },
        )
        bucket["num_contributions"] += 1
        bucket["total_probability"] += float(row["probability"])
        bucket["combined_probability"] = combine_xor_probs(
            float(bucket["combined_probability"]),
            float(row["probability"]),
        )

        source_type = str(row["source_type"])
        bucket["source_type_counts"][source_type] = bucket["source_type_counts"].get(source_type, 0) + 1
        bucket["source_type_probabilities"][source_type] = bucket["source_type_probabilities"].get(
            source_type,
            0.0,
        ) + float(
            row["probability"],
        )

        direct_family = row.get("direct_source_family")
        if direct_family is not None:
            direct_family = str(direct_family)
            bucket["direct_source_family_counts"][direct_family] = (
                bucket["direct_source_family_counts"].get(direct_family, 0) + 1
            )
            bucket["direct_source_family_probabilities"][direct_family] = bucket[
                "direct_source_family_probabilities"
            ].get(direct_family, 0.0) + float(row["probability"])

    assert len(regrouped) == len(render_summaries)
    for summary in render_summaries:
        key = (
            tuple(summary["detectors"]),
            tuple(summary["logicals"]),
            str(summary["rendered_targets"]),
        )
        bucket = regrouped[key]
        assert int(bucket["num_contributions"]) == int(summary["num_contributions"])
        assert float(bucket["total_probability"]) == pytest.approx(float(summary["total_probability"]))
        assert float(bucket["combined_probability"]) == pytest.approx(float(summary["combined_probability"]))
        assert bucket["source_type_counts"] == summary["source_type_counts"]
        assert bucket["direct_source_family_counts"] == summary["direct_source_family_counts"]
        assert bucket["source_type_probabilities"] == pytest.approx(summary["source_type_probabilities"])
        assert bucket["direct_source_family_probabilities"] == pytest.approx(
            summary["direct_source_family_probabilities"],
        )


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_keep_direct_policy_matches_default_render_outputs(basis: str) -> None:
    """The explicit KeepDirect policy should match the default rendering behavior."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    assert dem.to_string_decomposed_with_two_detector_direct_policy("KeepDirect") == dem.to_string_decomposed()
    assert (
        dem.contribution_render_summaries_with_two_detector_direct_policy("KeepDirect")
        == dem.contribution_render_summaries()
    )
    assert (
        dem.contribution_render_records_with_two_detector_direct_policy("KeepDirect")
        == dem.contribution_render_records()
    )


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_structured_recorded_component_policy_exposes_alternative_records(basis: str) -> None:
    """Recorded-component policy should expose alternate render strategies and targets."""
    dem = build_source_tracked_dem(distance=3, basis=basis, rounds=20)

    default_records = dem.contribution_render_records()
    policy_records = dem.contribution_render_records_with_two_detector_direct_policy(
        "PreferRecordedComponents",
    )

    assert len(policy_records) == len(default_records)
    assert any(row["render_strategy"] == "RecordedComponents" for row in policy_records)
    assert any(
        policy_row["rendered_targets"] != default_row["rendered_targets"]
        for default_row, policy_row in zip(default_records, policy_records, strict=False)
    )
