"""Independent probability identities and source provenance for DEM explanations."""

import json
from dataclasses import replace
from math import sqrt

import pytest
from pecos.qec.generic import (
    SI1000,
    CircuitOp,
    MeasurementParity,
    OperationOrigin,
    PauliFault,
    ScheduledCircuit,
    ScheduledNoiseComparison,
    SchedulePolicy,
)


def _cz(p=0.1):
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("CZ", (0, 1), OperationOrigin("round0", "check", 1, 0)),),
            (op("M", (0,)), op("M", (1,))),
        ),
        (MeasurementParity((0,), "check", True, (1, 2)),),
        (MeasurementParity((1,), "logical", True),),
        role_names=("data", "ancilla"),
        measurement_names=("data.final", "ancilla.final"),
        citations=("test construction",),
    )
    return SI1000(p).apply(plan).compile_native()


def test_two_qubit_joint_factor_has_distinct_physical_and_independent_weights():
    p = 0.1
    native = _cz(p)
    explanation = native.explain_signature(detectors=[0], observables=[0])
    report = explanation.to_dict()
    assert len(report["channels"]) == 1
    channel = report["channels"][0]
    assert channel["matching_branches"] == ["XX", "XY", "YX", "YY"]
    assert channel["matching_physical_branch_mass"] == pytest.approx(4 * p / 15)
    # Four equal independent factors on this support. Derived from the uniform
    # 2-qubit depolarizing channel's nontrivial Fourier eigenvalue 1-16p/15.
    expected = (1 - sqrt(1 - 16 * p / 15)) / 2
    assert report["dem_probability"] == pytest.approx(expected)
    assert channel["matching_independent_factor_probability"] == pytest.approx(expected)
    assert expected != pytest.approx(4 * p / 15)
    assert channel["source_operation"] == "q[0]/op[1]"
    assert channel["source_origin"] == {
        "call": "round0",
        "gadget": "check",
        "moment": 1,
        "operation": 0,
        "id": "round0/moment[1]/op[0]",
    }
    assert channel["roles"] == ["data", "ancilla"]
    assert report["detectors"][0] == {
        "id": 0,
        "label": "check",
        "records": [0],
        "measurement_names": ["data.final"],
        "constant": True,
        "coords": [1.0, 2.0],
    }
    assert report["observables"][0]["constant"]
    assert report["source_citations"] == ["test construction"]
    assert not report["physical_masses_additive_across_channels"]
    assert "not the marginal" in report["dem_probability_scope"]
    assert "round0/moment[1]/op[0]" in explanation.describe()
    assert "D0:check = data.final XOR 1" in explanation.describe()
    assert "L0:logical = ancilla.final XOR 1" in explanation.describe()


@pytest.mark.parametrize("p", [0.001, 0.1, 0.2])
def test_every_dem_term_recombines_and_affine_constants_cancel(p):
    native = _cz(p)
    plain = (
        SI1000(p)
        .apply(
            replace(
                native.noisy.ideal,
                detectors=(MeasurementParity((0,), "check"),),
                observables=(MeasurementParity((1,), "logical"),),
            ),
        )
        .compile_native()
    )
    replay = SI1000(0.001).apply(native.noisy.ideal).compile_native()
    baseline = SI1000(0).apply(native.noisy.ideal).compile_native().sample_records(1, seed=1)[0]
    for probability, detectors, observables in native.detector_error_model().independent_mechanisms():
        report = native.explain_signature(detectors=detectors, observables=observables).to_dict()
        assert report["dem_probability"] == pytest.approx(probability, rel=1e-13, abs=0)
        assert (
            plain.explain_signature(detectors=detectors, observables=observables).to_dict()["dem_probability"]
            == report["dem_probability"]
        )
        for channel in report["channels"]:
            for word in channel["matching_branches"]:
                # Independent sparse-stabilizer trajectory replay of each contributor.
                record = replay.replay_faults([PauliFault(channel["channel_id"], word)], seed=1)
                old, new = native.noisy.ideal.evaluate(baseline), native.noisy.ideal.evaluate(record)
                flips = [
                    tuple(i for i, (a, b) in enumerate(zip(x, y, strict=True)) if a != b)
                    for x, y in zip(old, new, strict=True)
                ]
                assert flips == [tuple(detectors), tuple(observables)]


def test_zero_rate_and_absent_joint_support_have_no_factors():
    native = _cz(0)
    report = native.explain_signature(detectors=[0]).to_dict()
    assert report["dem_probability"] == 0
    assert report["matching_factor_count"] == 0
    assert report["channels"] == []
    plan = native.noisy.ideal
    # Remove CZ: independent one-wire channels cannot directly flip both parities.
    native = SI1000(0.1).apply(replace(plan, moments=(plan.moments[0], plan.moments[2]))).compile_native()
    report = native.explain_signature(detectors=[0], observables=[0]).to_dict()
    assert report["channels"] == []
    assert report["dem_probability"] == 0


def test_idle_and_collapse_wait_keep_moments_without_invented_operation_origins():
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        ((op("R", (0,)),), (op("R", (1,)),), (op("M", (0,)), op("M", (1,)))),
        (MeasurementParity((0,), "check"),),
        (),
    )
    report = SI1000(0.001).apply(plan).compile_native().explain_signature(detectors=[0]).to_dict()
    waits = [c for c in report["channels"] if c["reason"] in {"idle", "collapse_wait"}]
    assert {c["reason"] for c in waits} == {"idle", "collapse_wait"}
    for channel in waits:
        assert channel["moment"] == 1
        assert channel["operation"] is None
        assert channel["source_operation"] is None
        assert channel["source_origin"] is None
        assert channel["roles"] == ["q[0]"]


def test_deterministic_readout_factors_cancel_at_probability_endpoint():
    op = CircuitOp
    plan = ScheduledCircuit(
        1,
        ((op("R", (0,)),), (op("M", (0,)),), (op("R", (0,)),), (op("M", (0,)),)),
        (MeasurementParity((0, 1), "repeat"),),
        (),
    )
    report = SI1000(0.2).apply(plan).compile_native().explain_signature(detectors=[0]).to_dict()
    # Both deterministic readout flips cancel. Only two independent reset X(0.4)
    # locations remain in the parity probability: 2 * 0.4 * 0.6.
    assert report["dem_probability"] == pytest.approx(0.48)
    assert report["matching_factor_count"] == 4
    assert [c["independent_probability_per_branch"] for c in report["channels"]] == [0.4, 1, 0.4, 1]


@pytest.mark.parametrize(
    "support",
    [
        {},
        {"detectors": [-1]},
        {"detectors": [1]},
        {"detectors": [True]},
        {"detectors": [0, 0]},
        {"observables": [0.0]},
        {"observables": [1]},
    ],
)
def test_invalid_support_is_rejected(support):
    with pytest.raises(ValueError, match=r"nonempty|distinct in-range"):
        _cz().explain_signature(**support)


def test_random_ideal_parity_rejected_even_without_noise():
    op = CircuitOp
    plan = ScheduledCircuit(
        1,
        ((op("R", (0,)),), (op("H", (0,)),), (op("M", (0,)),)),
        (MeasurementParity((0,), "random"),),
        (),
    )
    with pytest.raises(ValueError, match="deterministic"):
        SI1000(0).apply(plan).compile_native().explain_signature(detectors=[0])


def _changed_comparison():
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("X", (0,)),),
            (op("X", (0,)),),
            (op("X", (1,)),),
            (op("M", (0,)), op("M", (1,))),
        ),
        (MeasurementParity((1,), "syndrome", True),),
        (MeasurementParity((0,), "logical", True),),
    )
    noisy = SI1000(0.001).apply(plan)
    fault = PauliFault(next(c.id for c in noisy.channels if c.reason == "measurement" and c.qubits == (0,)), "X")
    return noisy.compare_schedule([fault], candidate_policy=SchedulePolicy.COMPACT_UNITARIES)


def test_changes_explain_removed_idle_factors_and_check_stored_probabilities():
    comparison = _changed_comparison()
    for index in range(2):
        report = comparison.explain_change(index)
        assert report["reference"]["dem_probability"] > report["candidate"]["dem_probability"]
        for side in ("reference", "candidate"):
            assert report[side]["dem_probability"] == pytest.approx(report["change"][f"{side}_probability"])
        counts = [sum(c["reason"] == "idle" for c in report[side]["channels"]) for side in ("reference", "candidate")]
        assert counts[0] - counts[1] == 1
        for c in report["reference"]["channels"]:
            if c["reason"] == "idle":
                assert c["source_operation"] is None
                assert c["source_origin"] is None
                assert c["gate"] is None
    assert "D0:syndrome" in comparison.explain()
    assert "L0:logical" in comparison.explain()
    damaged = comparison.to_dict()
    damaged["independent_dem_comparison"]["changes"][0]["reference_probability"] = 0.3
    with pytest.raises(ValueError, match="disagrees"):
        replace(comparison, _report_json=json.dumps(damaged)).explain_change(0)


def test_unchanged_comparison_needs_no_fault_reanalysis():
    native = _cz(0.001)
    channel = next(c for c in native.noisy.channels if c.kind == "DEPOLARIZE2")
    comparison = native.noisy.compare_schedule([PauliFault(channel.id, "IX")])
    assert "No changed independent DEM terms" in comparison.explain()
    assert "_native_pair" not in comparison.__dict__


def test_copy_and_display_limits_preserve_full_evidence():
    explanation = _cz().explain_signature(detectors=[0])
    report = explanation.to_dict()
    report["channels"].clear()
    assert explanation.to_dict()["channels"]
    assert "further channels retained" in explanation.describe(max_channels=0)
    for limit in (-1, True, 1.5):
        with pytest.raises(ValueError, match="max_channels"):
            explanation.describe(max_channels=limit)
    comparison = _changed_comparison()
    assert "2 further changes" in comparison.explain(top_changes=0)
    for index in (-1, 2, True, 0.5):
        with pytest.raises(ValueError, match="index"):
            comparison.explain_change(index)
    for kwargs in ({"top_changes": -1}, {"top_changes": True}, {"max_channels": 0.5}):
        with pytest.raises(ValueError, match="nonnegative integers"):
            comparison.explain(**kwargs)
