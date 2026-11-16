"""Physical fault transfer and an independently derived affine-memory noise oracle."""

from dataclasses import replace

import pytest
from pecos.qec.generic import (
    SI1000,
    CircuitOp,
    MeasurementParity,
    PauliFault,
    ScheduledCircuit,
    SchedulePolicy,
    remap_operation_faults,
)


def _memory(p=0.001):
    op = CircuitOp
    repeated = op("X", (0,))
    plan = ScheduledCircuit(2, ((op("R", (0,)), op("R", (1,))), (repeated,), (repeated,),
                                (op("X", (1,)),), (op("M", (0,)), op("M", (1,)))),
                            (MeasurementParity((1,), "syndrome", True),),
                            (MeasurementParity((0,), "logical", True),))
    return SI1000(p).apply(plan)


def _readout_fault(noisy):
    return PauliFault(next(c.id for c in noisy.channels if c.reason == "measurement" and c.qubits == (0,)), "X")


@pytest.mark.parametrize("policy", [SchedulePolicy.ASAP_UNITARIES, SchedulePolicy.COMPACT_UNITARIES])
def test_affine_memory_probabilities_follow_independent_xor_oracle(policy):
    p = 0.001
    noisy = _memory(p)
    comparison = noisy.compare_schedule([_readout_fault(noisy)], candidate_policy=policy)
    report = comparison.to_dict()
    assert report["reference"]["bounds"]["exact"]
    assert report["candidate"]["bounds"]["exact"]
    assert report["reference"]["bounds"]["lower_bound"] == report["candidate"]["bounds"]["upper_bound"] == 1
    mapping = report["witness_mapping"][0]
    assert mapping["reference"]["channel_id"] != mapping["candidate"]["channel_id"]
    # Each wire has reset X(2p), readout X(5p), and three versus two depolarizing
    # channels p/10. Each depolarizing channel flips Z readout with 2p/30.
    for key, count in [("reference", 3), ("candidate", 2)]:
        expected = (1 - (1 - 4*p)*(1 - 10*p)*(1 - 2*p/15)**count) / 2
        mechanisms = report[key]["independent_dem"]
        assert len(mechanisms) == 2
        assert all(row["probability"] == pytest.approx(expected, rel=1e-11) for row in mechanisms)
        assert report[key]["native_propagation_verified"]
        assert report[key]["native_trajectory_replay_verified"]
    assert report["reference"]["channels_by_reason"]["idle"] - report["candidate"]["channels_by_reason"]["idle"] == 2
    assert not report["independent_dem_comparison"]["matches_at_p"]
    assert [c["status"] for c in report["independent_dem_comparison"]["changes"]] == ["probability_changed"] * 2
    assert not report["logical_error_rates_verified"]
    assert not report["guppy_noise_equivalence_verified"]
    assert "physical distance [1, 1]" in comparison.describe()
    report["witness_mapping"].clear()
    assert comparison.to_dict()["witness_mapping"]


def test_repeated_operations_and_overtaking_remap_by_occurrence_not_object_or_old_id():
    noisy = _memory()
    candidate = noisy.noise.apply(noisy.ideal.rescheduled(SchedulePolicy.ASAP_UNITARIES))
    sources = [c for c in noisy.channels if c.reason == "one_qubit"]
    faults = [PauliFault(c.id, "X") for c in sources]
    mapped = remap_operation_faults(noisy, candidate, faults)
    assert len({f.channel_id for f in mapped}) == 3
    target_channels = {c.id: c for c in candidate.channels}
    assert [(target_channels[f.channel_id].moment, target_channels[f.channel_id].operation) for f in mapped] == [
        (1, 0), (2, 0), (1, 1),
    ]
    assert candidate.compile_native().fault_effect(mapped) == noisy.compile_native().fault_effect(faults)


@pytest.mark.parametrize("reason", ["idle", "collapse_wait"])
def test_nonoperation_fault_transfer_requires_a_new_physical_choice(reason):
    op = CircuitOp
    plan = ScheduledCircuit(2, ((op("R", (0,)),), (op("R", (1,)),),
                                (op("M", (0,)), op("M", (1,)))), (), ())
    noisy = SI1000(0.001).apply(plan)
    selected = next(c for c in noisy.channels if c.reason == reason)
    with pytest.raises(ValueError, match="Idle and collapse-wait"):
        remap_operation_faults(noisy, noisy, [PauliFault(selected.id, "X")])


def test_changed_profile_or_event_contract_and_detected_witness_fail():
    noisy = _memory()
    fault = _readout_fault(noisy)
    changed = SI1000(0.002).apply(noisy.ideal)
    with pytest.raises(ValueError, match="same noise profile"):
        remap_operation_faults(noisy, changed, [fault])
    changed = noisy.noise.apply(replace(noisy.ideal, observables=(MeasurementParity((0,), "different"),)))
    with pytest.raises(ValueError, match="source rescheduling contract"):
        remap_operation_faults(noisy, changed, [fault])
    measured = next(c for c in noisy.channels if c.reason == "measurement" and c.qubits == (1,))
    with pytest.raises(ValueError, match="detected"):
        noisy.compare_schedule([PauliFault(measured.id, "X")])


@pytest.mark.parametrize("p", [0, 0.2])
def test_distance_comparison_rejects_export_only_probability_endpoints(p):
    noisy = _memory(p)
    with pytest.raises(ValueError, match=r"0 < SI1000 p < 0\.2"):
        noisy.compare_schedule([PauliFault("unused", "X")])


def test_unsupported_policy_and_exclusion_fail_explicitly():
    noisy = _memory()
    with pytest.raises(ValueError, match="candidate_policy"):
        noisy.compare_schedule([_readout_fault(noisy)], candidate_policy=SchedulePolicy.REFERENCE)
    with pytest.raises(ValueError, match="original REFERENCE"):
        noisy.noise.apply(noisy.ideal.rescheduled(SchedulePolicy.COMPACT_UNITARIES)).compare_schedule([_readout_fault(noisy)])
    with pytest.raises(ValueError, match="exclude_through"):
        noisy.compare_schedule([_readout_fault(noisy)], exclude_through=4)


def test_two_qubit_pauli_branch_remains_one_location():
    op = CircuitOp
    plan = ScheduledCircuit(2, ((op("R", (0,)), op("R", (1,))), (op("CZ", (0, 1)),),
                                (op("M", (0,)), op("M", (1,)))), (), (MeasurementParity((0,), "L"),))
    noisy = SI1000(0.001).apply(plan)
    channel = next(c for c in noisy.channels if c.kind == "DEPOLARIZE2")
    comparison = noisy.compare_schedule([PauliFault(channel.id, "XY")])
    report = comparison.to_dict()
    assert report["reference"]["bounds"]["upper_bound"] == report["candidate"]["bounds"]["upper_bound"] == 1
    assert report["independent_dem_comparison"]["matches_at_p"]


def test_nonzero_detector_and_observable_baselines_are_not_fault_flips():
    noisy = _memory()
    noisy = noisy.noise.apply(replace(noisy.ideal, detectors=(MeasurementParity((1,), "syndrome"),)))
    ideal = SI1000(0).apply(noisy.ideal).compile_native().sample_records(1, seed=0)[0]
    assert noisy.ideal.evaluate(ideal) == ((True,), (True,))
    report = noisy.compare_schedule([_readout_fault(noisy)]).to_dict()
    assert report["reference"]["native_trajectory_replay_verified"]
    assert report["candidate"]["native_trajectory_replay_verified"]


@pytest.mark.parametrize("distance", [3, 4])
@pytest.mark.parametrize("policy", [SchedulePolicy.ASAP_UNITARIES, SchedulePolicy.COMPACT_UNITARIES])
def test_released_small_memory_witnesses_retain_exact_bounds(distance, policy):
    import hashlib
    import json
    from pathlib import Path

    from pecos.qec.color import (
        ColorExtraction,
        color_code_implementation,
        color_code_spec,
        color_memory,
        periodic_488_code,
        periodic_488_implementation,
        periodic_488_memory,
    )

    root = Path(__file__).parent / "fixtures/si1000"
    topology = "planar" if distance == 3 else "torus"
    name = f"{topology}_488_d{distance}_r{distance}_Z_phase0_p0p001.stim"
    case = next(c for c in json.loads((root / "witnesses.json").read_text())["cases"] if c["fixture"] == name)
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == case["sha256"]
    if distance == 3:
        impl = color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ)
        plan = color_memory(impl, 3, "Z")
    else:
        impl = periodic_488_implementation(periodic_488_code(4), gate_set="native-cz")
        plan = periodic_488_memory(impl, 4, "Z")
    noisy = SI1000(0.001).apply(plan)
    faults = [PauliFault(c.id, "X") for c in noisy.channels if c.kind == "X_ERROR"
              and c.moment == case["tick"] and c.qubits[0] in case["x_error_qubits"]]
    assert len(faults) == distance
    report = noisy.compare_schedule(faults, candidate_policy=policy, exclude_through=distance - 1).to_dict()
    assert report["reference"]["bounds"]["lower_bound"] == report["reference"]["bounds"]["upper_bound"] == distance
    assert report["candidate"]["bounds"]["lower_bound"] == report["candidate"]["bounds"]["upper_bound"] == distance
    assert report["independent_dem_comparison"]["matches_at_p"]
    if distance == 4 and policy == SchedulePolicy.COMPACT_UNITARIES:
        with pytest.raises(ValueError, match="exceeding budget"):
            noisy.compare_schedule(faults, candidate_policy=policy, exclude_through=3, max_pair_checks=0)
