"""Independent native execution/propagation versus the published Stim oracle."""

import builtins
import itertools
import random
from dataclasses import FrozenInstanceError

import pytest
import stim
from pecos.qec.generic import SI1000, CircuitOp, MeasurementParity, PauliFault, ScheduledCircuit
from pecos_rslib.qec import DetectorErrorModel

from qec.test_scheduled_si1000 import CASES, FIXTURES, _case_plan, _mechanisms


def _native_mechanisms(dem):
    return {(*(f"D{i}" for i in ds), *(f"L{i}" for i in obs)): p for p, ds, obs in dem.independent_mechanisms()}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["fixture"])
def test_native_published_dem_and_physical_witness(case):
    noisy = _case_plan(case)
    native = noisy.compile_native()
    dem = native.detector_error_model()
    expected = _mechanisms(stim.Circuit((FIXTURES / case["fixture"]).read_text()))
    assert _native_mechanisms(dem) == pytest.approx(expected, rel=1e-10, abs=1e-14)
    assert dem.num_detectors == len(noisy.ideal.detectors)
    assert dem.num_observables == len(noisy.ideal.observables)
    assert dem.detector_coordinates() == [
        (i, None if p.coords is None else list(p.coords)) for i, p in enumerate(noisy.ideal.detectors)
    ]
    faults = [
        PauliFault(c.id, "X")
        for c in noisy.channels
        if c.kind == "X_ERROR" and c.moment == case["tick"] and c.qubits[0] in case["x_error_qubits"]
    ]
    det, obs = native.fault_effect(faults)
    assert not any(det)
    assert any(obs)
    assert noisy.ideal.evaluate(native.replay_faults(faults, seed=32)) == (det, obs)


def _pair_plan():
    op, parity = CircuitOp, MeasurementParity
    return ScheduledCircuit(
        3,
        (
            (),
            (op("R", (0,)), op("R", (1,))),
            (op("CZ", (0, 1)),),
            (op("M", (1,)), op("X", (0,))),
            (),
            (op("R", (1,)),),
            (op("M", (0,)), op("M", (1,))),
            (),
        ),
        (parity((0, 2), "check", coords=(1, 2, 3, 4)),),
        (parity((1,), "signed", constant=True),),
    )


def test_every_branch_native_propagation_trajectory_and_stim_frames_agree():
    noisy = SI1000(0.01).apply(_pair_plan())
    native = noisy.compile_native()
    for c in noisy.channels:
        words = (
            ["X"]
            if c.kind == "X_ERROR"
            else ["".join(w) for w in itertools.product("IXYZ", repeat=len(c.qubits)) if set(w) != {"I"}]
        )
        for word in words:
            faults = [PauliFault(c.id, word)]
            expected = native.fault_effect(faults)
            assert noisy.ideal.evaluate(native.replay_faults(faults)) == expected
            simulator = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
            simulator.do(noisy._to_stim({c.id: word}))  # noqa: SLF001 - independent replay oracle
            actual = (
                tuple(simulator.get_detector_flips(instance_index=0)),
                tuple(
                    simulator.get_observable_flips(instance_index=0),
                ),
            )
            assert actual == expected


@pytest.mark.parametrize("p", [0, 0.001, 0.2, 1e-20])
def test_noise_endpoints_empty_moments_and_unused_sites(p):
    noisy = SI1000(p).apply(_pair_plan())
    dem = noisy.compile_native().detector_error_model()
    assert dem.num_detectors == 1
    assert dem.num_observables == 1
    if p == 1e-20:
        assert all(prob > 0 for prob, _, _ in dem.independent_mechanisms())
        assert dem.independent_mechanisms()
    else:
        assert _native_mechanisms(dem) == pytest.approx(_mechanisms(noisy.to_stim()), abs=1e-14)


def test_native_path_needs_no_stim_and_retains_affine_constants():
    op, parity = CircuitOp, MeasurementParity
    plan = ScheduledCircuit(
        1,
        ((op("R", (0,)),), (op("X", (0,)),), (op("M", (0,)),)),
        (parity((0,), "minus", constant=True), parity((), "constant", constant=True)),
        (parity((0,), "raw one"),),
    )
    native = SI1000(0).apply(plan).compile_native()
    original_import = builtins.__import__

    def no_stim(name, *args, **kwargs):
        if name == "stim" or name.startswith("stim."):
            pytest.fail("Native execution/analysis imported Stim")
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, "__import__", no_stim)
        assert native.sample_records(8, seed=77) == ((True,),) * 8
        assert plan.evaluate((True,)) == ((False, True), (True,))
        model = native.detector_error_model()
        assert model.independent_mechanisms() == []
        assert model.num_detectors == 2
        assert model.num_observables == 1
        assert not any(model.to_sampler().sample_batch(8, seed=77).observable_flips()[0])
    with pytest.raises(FrozenInstanceError):
        native.noisy = SI1000(0.1).apply(plan)


def test_random_clifford_determinism_and_native_dem_against_stim():
    rng = random.Random(5231)
    op, parity = CircuitOp, MeasurementParity
    for _ in range(160):
        moments = [(op("R", (q,)),) for q in range(3)]
        live = {0, 1, 2}
        measurements = 0
        for _ in range(16):
            q = rng.randrange(3)
            if q not in live:
                moments.append((op("R", (q,)),))
                live.add(q)
                continue
            gate = rng.choice(["H", "S", "S_DAG", "C_XYZ", "C_ZYX", "SQRT_X", "M", "CZ", "X", "Z"])
            if gate == "CZ":
                others = sorted(live - {q})
                if not others:
                    continue
                moments.append((op(gate, (q, rng.choice(others))),))
            else:
                moments.append((op(gate, (q,)),))
            if gate == "M":
                live.remove(q)
                measurements += 1
        for q in sorted(live):
            moments.append((op("M", (q,)),))
            measurements += 1
        support = tuple(i for i in range(measurements) if rng.randrange(2))
        plan = ScheduledCircuit(3, tuple(moments), (parity(support, "test"),), ())
        noisy = SI1000(0.01).apply(plan)
        native = noisy.compile_native()
        try:
            expected = _mechanisms(noisy.to_stim())
        except ValueError:
            with pytest.raises(ValueError, match="deterministic"):
                native.detector_error_model()
        else:
            assert _native_mechanisms(native.detector_error_model()) == pytest.approx(expected, abs=1e-14)


def test_categorical_sampling_matches_independently_enumerated_distribution():
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("CZ", (0, 1)),),
            (op("M", (0,)), op("M", (1,))),
        ),
        (),
        (),
    )
    p = 0.16
    # CZ on computational states preserves measured bits. Enumerate reset,
    # one categorical two-qubit depolarizing branch, then readout flips.
    expected = dict.fromkeys(itertools.product((False, True), repeat=2), 0.0)
    for r0, r1, m0, m1 in itertools.product((0, 1), repeat=4):
        rate = (2 * p if r0 else 1 - 2 * p) * (2 * p if r1 else 1 - 2 * p)
        rate *= (5 * p if m0 else 1 - 5 * p) * (5 * p if m1 else 1 - 5 * p)
        for a, b in itertools.product("IXYZ", repeat=2):
            branch = 1 - p if a == b == "I" else p / 15
            expected[bool(r0 ^ m0 ^ (a in "XY")), bool(r1 ^ m1 ^ (b in "XY"))] += rate * branch
    native = SI1000(p).apply(plan).compile_native()
    shots = native.sample_records(20_000, seed=614)
    assert {key: shots.count(key) / len(shots) for key in expected} == pytest.approx(expected, abs=0.012)
    assert native.sample_records(40, seed=11) == native.sample_records(40, seed=11)
    assert native.sample_records(0, seed=11) == ()


@pytest.mark.parametrize("shots", [-1, 1.5, True])
def test_invalid_shot_counts(shots):
    with pytest.raises(ValueError, match="shots"):
        SI1000(0).apply(_pair_plan()).compile_native().sample_records(shots)


def test_independent_model_constructor_preserves_probabilities_dimensions_and_coordinates():
    coords = [None, [], [1.5, -2, 3, 9]]
    model = DetectorErrorModel.from_independent_mechanisms(
        [(1e-25, [0], [1]), (0.2, [2, 1], []), (0.3, [1, 2], []), (1.0, [], [0])],
        num_detectors=3,
        num_observables=3,
        detector_coordinates=coords,
    )
    assert model.num_detectors == model.num_observables == 3
    assert model.detector_coordinates() == list(enumerate(coords))
    mechanisms = _native_mechanisms(model)
    assert mechanisms[("D0", "L1")] == 1e-25
    assert mechanisms[("D1", "D2")] == pytest.approx(0.38)
    assert mechanisms[("L0",)] == 1
    sampler = model.to_sampler()
    assert sampler.num_detectors == sampler.num_observables == 3
    assert all(obs[0] and not obs[2] for obs in sampler.sample_batch(10, seed=4).observable_flips())


@pytest.mark.parametrize(
    "mechanisms",
    [
        [(float("nan"), [0], [])],
        [(float("inf"), [0], [])],
        [(-0.1, [0], [])],
        [(1.1, [0], [])],
        [(0.1, [2], [])],
        [(0.1, [], [1])],
        [(0.1, [0, 0], [])],
        [(0.1, [], [0, 0])],
    ],
)
def test_independent_model_rejects_invalid_probabilities_or_targets(mechanisms):
    with pytest.raises(ValueError, match="mechanism"):
        DetectorErrorModel.from_independent_mechanisms(mechanisms, num_detectors=2, num_observables=1)


@pytest.mark.parametrize("coords", [[], [[1]], [None, [float("nan")]], [None, [float("inf")]]])
def test_independent_model_rejects_invalid_coordinates(coords):
    with pytest.raises(ValueError, match="coordinates"):
        DetectorErrorModel.from_independent_mechanisms(
            [], num_detectors=2, num_observables=0, detector_coordinates=coords,
        )
