"""Paper SI1000 circuits, physical witnesses, and independent small exclusions."""

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest
import stim
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.generic import (
    SI1000,
    CircuitOp,
    MeasurementParity,
    PauliFault,
    ScheduledCircuit,
    SchedulePolicy,
    fault_distance_bounds,
    verify_logical_fault,
)
from pecos.qec.generic.scheduled_faults import _exclude

FIXTURES = Path(__file__).parent / "fixtures/si1000"
CASES = json.loads((FIXTURES / "witnesses.json").read_text())["cases"]


def _case_plan(case):
    topology, _, distance, rounds, basis, phase, rate = Path(case["fixture"]).stem.split("_")
    d, r = int(distance[1:]), int(rounds[1:])
    if topology == "planar":
        impl = color_code_implementation(color_code_spec("488", d), ColorExtraction.BRICKWORK_CZ)
        plan = color_memory(impl, r, basis, start_phase=int(phase[5:]))
    else:
        impl = periodic_488_implementation(periodic_488_code(d), gate_set="native-cz")
        plan = periodic_488_memory(impl, r, basis)
    return SI1000(float(rate[1:].replace("p", "."))).apply(plan)


def _wire_moments(circuit):
    """Preserve operation/noise order per physical wire, ignoring disjoint ordering."""
    result, tick = defaultdict(list), 0
    for inst in circuit.flattened():
        if inst.name == "TICK":
            tick += 1
        elif inst.name not in {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"}:
            qs = [t.value for t in inst.targets_copy()]
            arity = 2 if inst.name in {"CZ", "DEPOLARIZE2"} else 1
            for start in range(0, len(qs), arity):
                pair = tuple(qs[start : start + arity])
                for q in pair:
                    result[tick, q].append((inst.name, pair, tuple(inst.gate_args_copy())))
    return dict(result), tick


def _mechanisms(circuit):
    result = {}
    for inst in circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened():
        if inst.type == "error":
            key = tuple(str(t) for t in inst.targets_copy())
            p = inst.args_copy()[0]
            result[key] = p + result.get(key, 0) - 2 * p * result.get(key, 0)
    return result


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["fixture"])
def test_full_noise_matches_unchanged_published_circuit(case):
    raw = (FIXTURES / case["fixture"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["sha256"]
    reference = stim.Circuit(raw.decode())
    noisy = _case_plan(case)
    actual = noisy.to_stim()
    assert _wire_moments(actual) == _wire_moments(reference)
    assert actual.num_ticks == len(noisy.ideal.moments)
    assert actual.num_measurements == noisy.ideal.num_measurements + any(
        p.constant for p in (*noisy.ideal.detectors, *noisy.ideal.observables)
    )
    expected = _mechanisms(reference)
    mechanisms = _mechanisms(actual)
    assert mechanisms.keys() == expected.keys()
    assert mechanisms == pytest.approx(expected, rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["fixture"])
def test_released_physical_witnesses_and_exact_small_bounds(case):
    noisy = _case_plan(case)
    selected = [
        c for c in noisy.channels
        if c.kind == "X_ERROR" and c.moment == case["tick"] and c.qubits[0] in case["x_error_qubits"]
    ]
    assert sorted(c.qubits[0] for c in selected) == sorted(case["x_error_qubits"])
    faults = [PauliFault(c.id, "X") for c in selected]
    through = case["distance"] - 1 if case["distance"] <= 4 else 0
    bounds = fault_distance_bounds(noisy, faults, exclude_through=through)
    assert bounds.upper_bound == case["distance"]
    assert any(bounds.witness.observable_flips)
    assert bounds.exact == (case["distance"] <= 4)
    assert bounds.lower_bound == (case["distance"] if through else 1)


def _two_qubit_plan():
    op = CircuitOp
    return ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("CZ", (0, 1)),),
            (op("M", (0,)), op("M", (1,))),
        ),
        (),
        (MeasurementParity((0,), "left"), MeasurementParity((1,), "right")),
    )


def test_two_qubit_branch_costs_one_location_and_deterministic_errors_replay():
    noisy = SI1000(0.001).apply(_two_qubit_plan())
    channel = next(c for c in noisy.channels if c.kind == "DEPOLARIZE2")
    witness = verify_logical_fault(noisy, [PauliFault(channel.id, "XY")])
    assert witness.weight == 1
    assert witness.observable_flips == (True, True)
    # Exhaustive lower-bound search finds weight-one supports rather than claiming >1.
    result = fault_distance_bounds(noisy, witness.faults, exclude_through=3)
    assert result.exact
    assert result.lower_bound == result.upper_bound == 1
    with pytest.raises(ValueError, match="only once"):
        verify_logical_fault(noisy, [*witness.faults, *witness.faults])
    for word in ("II", "X", "xyz", "A!"):
        with pytest.raises(ValueError, match="branch"):
            verify_logical_fault(noisy, [PauliFault(channel.id, word)])
    reset = next(c for c in noisy.channels if c.reason == "reset")
    with pytest.raises(ValueError, match="branch"):
        verify_logical_fault(noisy, [PauliFault(reset.id, "Y")])
    with pytest.raises(ValueError, match="Unknown"):
        verify_logical_fault(noisy, [PauliFault("missing", "X")])
    with pytest.raises(ValueError, match="nonempty"):
        verify_logical_fault(noisy, [])
    with pytest.raises(ValueError, match="no logical effect"):
        verify_logical_fault(noisy, [PauliFault(channel.id, "ZI")])
    checked = SI1000(0.001).apply(replace(noisy.ideal, detectors=(MeasurementParity((0,), "check"),)))
    with pytest.raises(ValueError, match="detected"):
        verify_logical_fault(checked, witness.faults)


def test_nondeterministic_ideal_detector_is_not_accepted_for_replay():
    op = CircuitOp
    plan = ScheduledCircuit(
        1, ((op("R", (0,)),), (op("H", (0,)),), (op("M", (0,)),)),
        (MeasurementParity((0,), "random check"),), (MeasurementParity((0,), "random logical"),),
    )
    noisy = SI1000(0.001).apply(plan)
    channel = next(c for c in noisy.channels if c.reason == "measurement")
    with pytest.raises(ValueError, match="non-deterministic"):
        verify_logical_fault(noisy, [PauliFault(channel.id, "X")])


def test_idle_and_wait_are_separate_and_include_active_and_retired_sites():
    op = CircuitOp
    plan = ScheduledCircuit(
        3,
        (
            (op("R", (0,)),),
            (op("H", (0,)), op("R", (1,))),
            (),
            (op("M", (0,)), op("M", (1,))),
            (op("R", (2,)),),
            (op("M", (2,)),),
        ), (), (),
    )
    noisy = SI1000(0.001).apply(plan)
    at = [c for c in noisy.channels if c.moment == 1]
    assert {(c.qubits, c.reason, c.probability) for c in at} == {
        ((0,), "one_qubit", 0.0001), ((1,), "reset", 0.002),
        ((2,), "idle", 0.0001), ((0,), "collapse_wait", 0.002), ((2,), "collapse_wait", 0.002),
    }
    assert len([c for c in noisy.channels if c.moment == 2]) == 3
    # Already measured sites still receive the release's allocated-site idle and wait channels.
    assert {(c.reason, c.qubits) for c in noisy.channels if c.moment == 4 and c.qubits == (0,)} == {
        ("idle", (0,)), ("collapse_wait", (0,)),
    }
    assert noisy.to_stim().num_ticks == 6


@pytest.mark.parametrize("p", [-0.001, 0.201, float("nan"), float("inf"), True, "0.01"])
def test_invalid_si1000_rate(p):
    with pytest.raises(ValueError, match="SI1000 p"):
        SI1000(p)


@pytest.mark.parametrize("p", [0, 0.2])
def test_export_endpoints_do_not_enable_distance_claims(p):
    noisy = SI1000(p).apply(_two_qubit_plan())
    assert noisy.to_stim().num_ticks == 3
    if not p:
        assert noisy.channels == ()
        assert noisy.to_stim() == noisy.ideal.to_stim()
    with pytest.raises(ValueError, match="0 < SI1000"):
        verify_logical_fault(noisy, [PauliFault("unused", "X")])


def test_cnot_is_not_silently_given_a_native_cz_profile():
    impl = periodic_488_implementation(periodic_488_code(4))
    with pytest.raises(ValueError, match="native-CZ"):
        SI1000(0.001).apply(periodic_488_memory(impl, 1, "Z"))


def test_rescheduling_requires_recomputing_noise_locations():
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("H", (0,)),), (op("H", (1,)),),
            (op("M", (0,)), op("M", (1,))),
        ), (), (),
    )
    reference = SI1000(0.001).apply(plan)
    alternative = reference.noise.apply(reference.ideal.rescheduled(SchedulePolicy.ASAP_UNITARIES))
    assert alternative.ideal.schedule == SchedulePolicy.ASAP_UNITARIES
    assert len(alternative.channels) < len(reference.channels)
    assert sum(c.reason == "idle" for c in reference.channels) == 2
    assert all(c.reason != "idle" for c in alternative.channels)
    assert alternative.to_stim().num_ticks == len(alternative.ideal.moments)


def test_small_exclusion_against_independent_subset_enumeration():
    # Includes hyperedges and two separate logical bits. Same detector supports
    # with different logical masks must remain distinct, even if both are nonzero.
    universes = [
        {(3, 0), (5, 0), (6, 1)}, {(0, 2)}, {(7, 1), (7, 2)},
        {(1, 0), (2, 0), (4, 0), (7, 3)}, {(1, 1), (2, 2), (4, 0)},
    ]
    for supports in universes:
        for through in (1, 2, 3):
            expected = through
            for weight in range(1, through + 1):
                found = False
                for subset in itertools.combinations(supports, weight):
                    detectors = logical = 0
                    for d, value in subset:
                        detectors ^= d
                        logical ^= value
                    found |= detectors == 0 and logical != 0
                if found:
                    expected = weight - 1
                    break
            assert _exclude(supports, through, 100) == expected
    with pytest.raises(ValueError, match="budget"):
        _exclude({(1, 0), (2, 0), (4, 0), (7, 1)}, 3, 0)


@pytest.mark.parametrize("rate", [0.001, 1e-20])
def test_tiny_dem_probabilities_cannot_hide_a_weight_one_physical_fault(rate):
    op = CircuitOp
    moments = [(op("R", (0,)), op("R", (1,)), op("R", (2,)))]
    for target in (1, 2):
        moments.extend(((op("H", (target,)),), (op("CZ", (0, target)),), (op("H", (target,)),)))
    moments.append(())
    for target in (2, 1):
        moments.extend(((op("H", (target,)),), (op("CZ", (0, target)),), (op("H", (target,)),)))
    moments.append((op("M", (0,)), op("M", (1,)), op("M", (2,))))
    plan = ScheduledCircuit(
        3, tuple(moments),
        (MeasurementParity((0, 1), "left"), MeasurementParity((1, 2), "right")),
        (MeasurementParity((0,), "logical"),),
    )
    noisy = SI1000(rate).apply(plan)
    reset_faults = [PauliFault(c.id, "X") for c in noisy.channels if c.reason == "reset"]
    bounds = fault_distance_bounds(noisy, reset_faults, exclude_through=3)
    # The long witness is valid, but the empty middle moment admits a one-location logical.
    # At 1e-20 Stim's direct numerical DEM can lose that depolarizing branch.
    assert bounds.lower_bound == 1
    assert bounds.upper_bound == 3
    assert not bounds.exact
    middle = next(c for c in noisy.channels if c.moment == 7 and c.qubits == (0,))
    short = fault_distance_bounds(noisy, [PauliFault(middle.id, "X")], exclude_through=3)
    assert short.exact
    assert short.lower_bound == short.upper_bound == 1


def test_underflowed_channel_is_not_a_positive_probability_witness():
    op = CircuitOp
    plan = ScheduledCircuit(
        1,
        ((op("R", (0,)),), (op("H", (0,)),), (op("H", (0,)),), (op("M", (0,)),)),
        (), (MeasurementParity((0,), "logical"),),
    )
    noisy = SI1000(5e-324).apply(plan)
    channel = next(c for c in noisy.channels if c.moment == 2)
    assert channel.probability == 0
    with pytest.raises(ValueError, match="zero probability"):
        verify_logical_fault(noisy, [PauliFault(channel.id, "X")])
