"""Ideal closed parity classification and published memory coverage."""

from dataclasses import replace

import pytest
from pecos.qec.color import lacroix_memory, periodic_488_code, periodic_488_implementation, periodic_488_memory
from pecos.qec.generic import CircuitOp, MeasurementParity, OperationOrigin, ScheduledCircuit, SchedulePolicy


def test_bell_records_distinguish_zero_one_and_random_affine_parities():
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        ((op("R", (0,)), op("R", (1,))), (op("H", (0,)),), (op("CX", (0, 1)),), (op("M", (0,)), op("M", (1,)))),
        (
            MeasurementParity((0, 1), "same", coords=(1, 2)),
            MeasurementParity((0, 1), "wrong sign", True),
            MeasurementParity((0,), "random"),
        ),
        (MeasurementParity((0, 1), "same"),),
        measurement_names=("first/bit", "second/bit"),
    )
    audit = plan.audit_parities()
    report = audit.to_dict()
    assert [p["ideal_value"] for p in report["parities"]] == [0, 1, None, 0]
    assert [p["deterministic"] for p in report["parities"]] == [True, True, False, True]
    assert [p["valid"] for p in report["parities"]] == [True, False, False, True]
    assert report["parities"][0]["coords"] == [1.0, 2.0]
    assert report["parities"][1]["expression"] == "first/bit ^ second/bit ^ 1"
    assert "always one" in audit.describe()
    assert "nondeterministic" in audit.describe()
    with pytest.raises(ValueError, match=r"detector\[1\].*wrong sign.*detector\[2\].*random"):
        audit.require_valid()


@pytest.mark.parametrize("policy", list(SchedulePolicy))
def test_names_origins_relabel_and_reused_lifetimes_survive_audit(policy):
    op = CircuitOp
    origin = OperationOrigin("prepare", "one", 1, 0)
    plan = ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("X", (0,), origin),),
            (op("RELABEL", (0, 1)),),
            (op("M", (0,)), op("M", (1,))),
            (op("R", (0,)),),
            (op("M", (0,)),),
        ),
        (MeasurementParity((0, 2), "reuse"),),
        (MeasurementParity((1,), "signed", True),),
        citations=("published fixture",),
        role_names=("block/data", "role[0]"),
        measurement_names=("first/bit", "event[0]", "last/bit"),
    )
    plan = plan.rescheduled(policy)
    before = plan.to_stim()
    audit = plan.audit_parities()
    audit.require_valid()
    report = audit.to_dict()
    assert report["schedule"] == policy.value
    assert report["source_citations"] == ["published fixture"]
    assert [b["authored"] for b in report["measurement_bindings"]] == list(plan.measurement_names)
    assert [b["authored"] for b in report["role_bindings"]] == list(plan.role_names)
    assert report["source_operations"][2]["source"]["call"] == "prepare"
    assert plan.to_stim() == before
    report["parities"][0]["valid"] = False
    assert audit.all_valid


def test_constant_only_checks_and_no_checks_are_distinct():
    plan = ScheduledCircuit(1, ((),), (MeasurementParity((), "zero"), MeasurementParity((), "one", True)), ())
    assert [r["ideal_value"] for r in plan.audit_parities().to_dict()["parities"]] == [0, 1]
    with pytest.raises(ValueError, match="at least one"):
        replace(plan, detectors=()).audit_parities()


def test_prepared_one_requires_explicit_constant_and_observable_only_is_supported():
    op = CircuitOp
    plan = ScheduledCircuit(
        1,
        ((op("R", (0,)),), (op("X", (0,)),), (op("M", (0,)),)),
        (),
        (MeasurementParity((0,), "one"),),
    )
    audit = plan.audit_parities()
    assert not audit.all_valid
    assert audit.to_dict()["parities"][0]["ideal_value"] == 1
    assert replace(plan, observables=(plan.observables[0] ^ 1,)).audit_parities().all_valid


@pytest.mark.parametrize("patch", ["d3_west", "d3_east", "d3_north", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [2, 3, 5])
@pytest.mark.parametrize("policy", list(SchedulePolicy))
def test_lacroix_closed_memories_have_signed_zero_parities(patch, basis, rounds, policy):
    plan = lacroix_memory(patch, rounds, basis, schedule=policy)
    audit = plan.audit_parities()
    assert audit.all_valid, audit.describe()
    report = audit.to_dict()
    assert len(report["parities"]) == len(plan.detectors) + len(plan.observables)
    assert not report["open_gadget_action_verified"]
    assert not report["compiled_guppy_verified"]
    assert not report["fault_tolerance_verified"]
    assert not report["noise_equivalence_verified"]


@pytest.mark.parametrize("gate_set", ["cnot", "native-cz"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("policy", list(SchedulePolicy))
def test_periodic_memory_boundaries_and_four_observables(gate_set, basis, policy):
    impl = periodic_488_implementation(periodic_488_code(4), gate_set=gate_set)
    plan = periodic_488_memory(impl, 3, basis, schedule=policy)
    audit = plan.audit_parities()
    assert audit.all_valid, audit.describe()
    assert sum(r["kind"] == "observable" for r in audit.to_dict()["parities"]) == 4
    if gate_set == "native-cz":
        assert any(r["constant"] for r in audit.to_dict()["parities"])
    broken = replace(plan, detectors=(plan.detectors[0] ^ 1, *plan.detectors[1:]))
    assert broken.audit_parities().to_dict()["parities"][0]["ideal_value"] == 1
