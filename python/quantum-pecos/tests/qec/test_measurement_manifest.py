"""Portable raw-event identities and independent affine-parity reconstruction."""

import json
from dataclasses import replace
from itertools import product

import pytest
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    lacroix_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.generic import CircuitOp, MeasurementParity, OperationOrigin, ScheduledCircuit, SchedulePolicy


def _plan():
    op = CircuitOp
    return ScheduledCircuit(
        2,
        ((op("R", (0,)), op("R", (1,))), (op("H", (0,)),), (op("H", (1,)),),
         (op("RELABEL", (0, 1)),),
         (op("M", (0,), OperationOrigin("first", "readout", 0, 0)), op("M", (1,))),
         (op("R", (0,)),), (op("M", (0,), OperationOrigin("last", "readout", 0, 0)),)),
        (MeasurementParity((0, 2), "reuse", True, (1, 2)), MeasurementParity((), "one", True)),
        (MeasurementParity((1,), "logical"), MeasurementParity((), "zero")),
        citations=("published fixture",), role_names=("data", "ancilla"),
        measurement_names=("first/data", "first/ancilla", "last/data"),
    )


def _evaluate(manifest, record):
    return tuple(
        tuple(bool((sum(record[tag] for tag in parity["measurement_tags"]) + parity["constant"]) % 2)
              for parity in manifest["derived_parities"][kind])
        for kind in ("detectors", "observables")
    )


@pytest.mark.parametrize("policy", list(SchedulePolicy))
def test_exported_definitions_reconstruct_all_raw_assignments(policy):
    plan = _plan().rescheduled(policy)
    manifest = json.loads(json.dumps(plan.measurement_manifest(), allow_nan=False))
    assert manifest["format"] == "pecos.scheduled-measurements"
    assert manifest["version"] == 1
    assert manifest["schedule"] == policy.value
    assert manifest["measurement_count"] == 3
    assert not manifest["compiled_measurement_binding_verified"]
    assert not manifest["physical_schedule_verified"]
    assert not manifest["noise_equivalence_verified"]
    for bits in product((0, 1), repeat=3):
        record = {f"qec:m:{i}": bit for i, bit in enumerate(bits)}
        expected = ((bool(bits[0] ^ bits[2] ^ 1), True), (bool(bits[1]), False))
        assert _evaluate(manifest, record) == expected == plan.evaluate_named(record)


def test_relabel_origins_reused_roles_and_detached_export():
    plan = _plan()
    manifest = plan.measurement_manifest()
    events = manifest["raw_events"]
    assert [e["index"] for e in events] == [0, 1, 2]
    assert [e["tag"] for e in events] == ["qec:m:0", "qec:m:1", "qec:m:2"]
    assert [e["name"] for e in events] == ["first/data", "first/ancilla", "last/data"]
    assert [e["role"] for e in events] == ["data", "ancilla", "data"]
    assert [e["role_index"] for e in events] == [0, 1, 0]
    assert [e["reference_carrier"] for e in events] == [1, 0, 1]
    assert [e["source_moment"] for e in events] == [4, 4, 6]
    assert events[0]["origin"] == {
        "call": "first", "gadget": "readout", "moment": 0, "operation": 0, "id": "first/moment[0]/op[0]",
    }
    assert events[0]["source_operation"] == "first/moment[0]/op[0]"
    assert events[1]["origin"] is None
    assert events[1]["source_operation"] == "moment[4]/op[1]"
    assert all(e["value_type"] == "bool" for e in events)
    expected = json.loads(json.dumps(manifest))
    manifest["raw_events"][0]["origin"]["call"] = "changed"
    manifest["source_citations"].clear()
    manifest["derived_parities"]["detectors"][0]["measurement_tags"].clear()
    manifest["derived_parities"]["detectors"][0]["coords"].clear()
    assert plan.measurement_manifest() == expected


def test_schedule_changes_locations_but_preserves_events_and_authored_origins():
    plan = _plan()
    before = plan.measurement_manifest()["raw_events"]
    after = plan.rescheduled(SchedulePolicy.COMPACT_UNITARIES).measurement_manifest()["raw_events"]
    assert before[0]["source_moment"] != after[0]["source_moment"]
    for left, right in zip(before, after, strict=True):
        for key in ("index", "tag", "name", "role", "role_index", "reference_carrier", "origin"):
            assert left[key] == right[key]


def test_parity_changes_do_not_modify_raw_layout_or_generated_guppy():
    plan = _plan()
    other = replace(plan, detectors=(MeasurementParity((1,), "new", False, (9,)),), observables=())
    assert plan.measurement_manifest()["raw_events"] == other.measurement_manifest()["raw_events"]
    assert plan.guppy_source() == other.guppy_source()
    unnamed = replace(plan, role_names=(), measurement_names=()).measurement_manifest()
    assert unnamed["raw_events"][0]["name"] == "measurement[0]"
    assert unnamed["raw_events"][0]["role"] == "q[0]"


def test_empty_measurement_layout_with_constant_only_definitions():
    plan = ScheduledCircuit(1, ((),), (MeasurementParity((), "one", True),), ())
    manifest = json.loads(json.dumps(plan.measurement_manifest()))
    assert manifest["raw_events"] == []
    assert manifest["measurement_count"] == 0
    assert _evaluate(manifest, {}) == ((True,), ())
    assert _evaluate(replace(plan, detectors=()).measurement_manifest(), {}) == ((), ())


@pytest.mark.parametrize("case", [*(c.value for c in ColorExtraction), "periodic-cnot", "periodic-native-cz",
                                  "lacroix-d3_west", "lacroix-d3_east", "lacroix-d3_north", "lacroix-d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_published_families_keep_named_raw_events_and_all_parities(case, basis):
    if case.startswith("periodic-"):
        impl = periodic_488_implementation(periodic_488_code(4), gate_set=case.removeprefix("periodic-"))
        plan = periodic_488_memory(impl, 3, basis)
        assert len(plan.observables) == 4
    elif case.startswith("lacroix-"):
        plan = lacroix_memory(case.removeprefix("lacroix-"), 3, basis)
    else:
        choice = ColorExtraction(case)
        family = "488" if choice in {ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ} else "666"
        plan = color_memory(color_code_implementation(color_code_spec(family, 3), choice), 3, basis)
    manifest = json.loads(json.dumps(plan.measurement_manifest(), allow_nan=False))
    assert manifest["source_citations"] == list(plan.citations)
    assert manifest["source_citations"]
    assert [e["name"] for e in manifest["raw_events"]] == list(plan.measurement_names)
    assert len({e["tag"] for e in manifest["raw_events"]}) == plan.num_measurements
    # Deterministic mixed bit patterns exercise parity reconstruction without assuming ideal outcomes.
    for stride in (1, 3, 7):
        record = {e["tag"]: (e["index"] // stride) % 2 for e in manifest["raw_events"]}
        assert _evaluate(manifest, record) == plan.evaluate_named(record)
