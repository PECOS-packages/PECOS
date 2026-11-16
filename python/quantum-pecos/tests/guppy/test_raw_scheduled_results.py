"""Scheduled Guppy emits raw records, independently of downstream QEC parities."""

import json
from dataclasses import replace

import pytest
from pecos import sim, stabilizer
from pecos.qec.generic import CircuitOp, GuppyOrdering, MeasurementParity, ScheduledCircuit, SchedulePolicy


def _memory():
    op = CircuitOp
    return ScheduledCircuit(
        3,
        (
            (op("R", (0,)), op("R", (1,)), op("R", (2,))),
            (op("H", (0,)), op("X", (2,))),
            (op("CX", (0, 1)),),
            (op("M", (0,)), op("M", (1,)), op("M", (2,))),
        ),
        (MeasurementParity((0, 1), "Bell parity"), MeasurementParity((2,), "signed", True)),
        (MeasurementParity((0, 1), "logical"),),
        role_names=("left", "right", "one"),
        measurement_names=("readout/left", "readout/right", "readout/one"),
    )


@pytest.mark.parametrize("policy", list(SchedulePolicy))
@pytest.mark.parametrize("ordering", list(GuppyOrdering))
def test_detector_definitions_do_not_change_generated_guppy_or_raw_tags(policy, ordering):
    plan = _memory().rescheduled(policy)
    changed = replace(
        plan,
        detectors=(MeasurementParity((0, 1), "different detector", True, (1, 2)),),
        observables=(MeasurementParity((2,), "different logical"),),
    )
    assert plan.guppy_source(ordering=ordering) == changed.guppy_source(ordering=ordering)
    assert plan.guppy_source(ordering=ordering) == replace(plan, detectors=(), observables=()).guppy_source(
        ordering=ordering,
    )
    assert (
        plan.measurement_aliases
        == changed.measurement_aliases
        == {
            "readout/left": "qec:m:0",
            "readout/right": "qec:m:1",
            "readout/one": "qec:m:2",
        }
    )


@pytest.mark.parametrize("ordering", [GuppyOrdering.FUNCTIONAL, GuppyOrdering.ALL_MOMENTS])
def test_compiled_raw_bits_are_tagged_and_reinterpreted_without_rerunning(ordering):
    plan = _memory()
    audit = plan.audit_guppy(ordering=ordering, seed=7)
    assert audit.build.named_result_binding == "compiler_direct_scalar_complete"
    assert len(audit.build.measurement_ledger) == 3
    columns = sim(plan.guppy_program(ordering=ordering)).qubits(3).quantum(stabilizer()).seed(7).run(16).to_dict()
    # Simulate persistence: raw columns and definitions have independent JSON representations.
    columns = json.loads(json.dumps(columns))
    manifest = json.loads(json.dumps(plan.measurement_manifest()))
    assert {event["tag"] for event in manifest["raw_events"]} == set(columns)
    assert set(columns) == {"qec:m:0", "qec:m:1", "qec:m:2"}
    assert list(columns["qec:m:0"]) == list(columns["qec:m:1"])
    assert all(columns["qec:m:2"])  # Raw prepared-one readout, although every QEC parity is zero.
    changed = replace(
        plan,
        detectors=(MeasurementParity((0, 1), "different", True),),
        observables=(MeasurementParity((2,), "different"),),
    )
    for shot in range(16):
        record = {tag: values[shot] for tag, values in columns.items()}
        exported = tuple(
            tuple(
                bool((sum(record[tag] for tag in p["measurement_tags"]) + p["constant"]) % 2)
                for p in manifest["derived_parities"][kind]
            )
            for kind in ("detectors", "observables")
        )
        assert exported == plan.evaluate_named(record)
        assert plan.evaluate_named(record) == ((False, False), (False,))
        assert changed.evaluate_named(record) == ((True,), (True,))
