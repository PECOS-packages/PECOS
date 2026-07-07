# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Generality tests for meas_sampling() stochastic raw measurement backend.

These test the core fault propagation and measurement sampling logic
on minimal hand-built circuits — not surface-code-specific.
"""

import json

import pytest
from pecos.quantum import TickCircuit
from pecos_rslib_exp import depolarizing, meas_sampling, monte_carlo, sim_neo, stabilizer, statevec


def build_two_round_x_check():
    """Minimal 2-round X-check: ancilla q0, data q1 q2."""
    tc = TickCircuit()
    # Round 1
    tc.tick().h([0])
    tc.tick().cx([(0, 1)])
    tc.tick().cx([(0, 2)])
    tc.tick().h([0])
    tick = tc.tick()
    tick.mz([0])
    tc.tick().pz([0])
    # Round 2
    tc.tick().h([0])
    tc.tick().cx([(0, 1)])
    tc.tick().cx([(0, 2)])
    tc.tick().h([0])
    tick = tc.tick()
    tick.mz([0])

    # Detector: m0 XOR m1 should be 0 in noiseless case
    tc.set_meta("num_measurements", "2")
    tc.set_meta("detectors", json.dumps([{"records": [-1, -2]}]))
    tc.set_meta("observables", "[]")
    return tc


def build_three_round_z_check():
    """3-round Z-check on single data qubit: ancilla q0, data q1."""
    tc = TickCircuit()
    for _ in range(3):
        tc.tick().pz([0])
        tc.tick().cx([(1, 0)])  # data is control for Z-check
        tick = tc.tick()
        tick.mz([0])

    tc.set_meta("num_measurements", "3")
    tc.set_meta(
        "detectors",
        json.dumps(
            [
                {"records": [-2, -3]},  # m0 XOR m1
                {"records": [-1, -2]},  # m1 XOR m2
            ],
        ),
    )
    tc.set_meta("observables", "[]")
    return tc


class TestMeasurementFaultIndependence:
    """Measurement faults must not cancel through Copy chains."""

    def test_two_round_meas_fault_both_fire(self):
        """A detector comparing two Copy-linked measurements should see
        faults from BOTH measurements independently.
        """
        tc = build_two_round_x_check()
        # Measurement-only noise: each meas flips with p=0.01
        depol = depolarizing().p1(0).p2(0).p_meas(0.01).p_prep(0)
        shots = 50000

        meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()
        stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()

        # Extract detector rate
        def det_rate(results):
            return sum(s[0] ^ s[1] for s in results) / len(results)

        meas_rate = det_rate(meas_r)
        stab_rate = det_rate(stab_r)

        # Expected: ~2*p_meas = 0.02 (two independent flips)
        assert (
            abs(meas_rate - stab_rate) / max(stab_rate, 1e-10) < 0.15
        ), f"Meas fault rate mismatch: dem={meas_rate:.4f} stab={stab_rate:.4f}"


class TestPrepFaultAbsorption:
    """PZ faults propagate forward but get absorbed at PZ/MZ."""

    def test_prep_fault_reaches_next_measurement(self):
        """A prep fault on PZ(ancilla) should flip the next ancilla MZ."""
        tc = build_two_round_x_check()
        # PZ-only noise
        depol = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0.01)
        shots = 50000

        meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()
        stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()

        def det_rate(results):
            return sum(s[0] ^ s[1] for s in results) / len(results)

        meas_rate = det_rate(meas_r)
        stab_rate = det_rate(stab_r)

        # PZ faults fire the detector (X error → detected at MZ)
        assert stab_rate > 0.005, f"Stabilizer should see prep faults: {stab_rate}"
        assert (
            abs(meas_rate - stab_rate) / stab_rate < 0.15
        ), f"PZ fault rate mismatch: dem={meas_rate:.4f} stab={stab_rate:.4f}"

    def test_prep_fault_does_not_cross_reset(self):
        """A prep fault should NOT propagate past a subsequent PZ on the same qubit."""
        tc = build_three_round_z_check()
        depol = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0.01)
        shots = 50000

        meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()
        stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()

        # Extract detector 0 (m0 XOR m1) rate
        def det_rate(results, d):
            num_meas = 3
            recs = [{"records": [-2, -3]}, {"records": [-1, -2]}][d]["records"]
            fired = sum(1 for s in results if sum(s[num_meas + r] for r in recs) % 2 == 1)
            return fired / len(results)

        for d in [0, 1]:
            meas_rate = det_rate(meas_r, d)
            stab_rate = det_rate(stab_r, d)
            assert (
                abs(meas_rate - stab_rate) / max(stab_rate, 1e-10) < 0.20
            ), f"Det {d} prep fault mismatch: dem={meas_rate:.4f} stab={stab_rate:.4f}"


class TestMultiRoundNonSurface:
    """Multi-round circuits that are NOT surface codes."""

    def test_three_round_z_check_all_noise(self):
        """3-round Z-check with full depolarizing noise matches stabilizer."""
        tc = build_three_round_z_check()
        depol = depolarizing().p1(0.001).p2(0.005).p_meas(0.005).p_prep(0.005)
        shots = 50000

        meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()
        stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).sampling(monte_carlo(shots)).seed(42).run()

        def det_rate(results, d):
            num_meas = 3
            recs = [{"records": [-2, -3]}, {"records": [-1, -2]}][d]["records"]
            fired = sum(1 for s in results if sum(s[num_meas + r] for r in recs) % 2 == 1)
            return fired / len(results)

        for d in [0, 1]:
            meas_rate = det_rate(meas_r, d)
            stab_rate = det_rate(stab_r, d)
            assert (
                abs(meas_rate - stab_rate) / max(stab_rate, 1e-10) < 0.15
            ), f"Det {d} mismatch: dem={meas_rate:.4f} stab={stab_rate:.4f}"


class TestZeroNoise:
    """With zero noise, all detectors must fire at rate 0."""

    def test_two_round_x_check_zero_noise(self):
        tc = build_two_round_x_check()
        depol = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0)
        r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(1000)).seed(42).run()
        det_fires = sum(s[0] ^ s[1] for s in r)
        assert det_fires == 0, f"Zero-noise detector fired {det_fires}/1000 times"

    def test_three_round_z_check_zero_noise(self):
        tc = build_three_round_z_check()
        depol = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0)
        r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(1000)).seed(42).run()
        for d in [0, 1]:
            num_meas = 3
            recs = [{"records": [-2, -3]}, {"records": [-1, -2]}][d]["records"]
            fired = sum(1 for s in r if sum(s[num_meas + r_] for r_ in recs) % 2 == 1)
            assert fired == 0, f"Zero-noise det {d} fired {fired}/1000"

    def test_new_clifford_gates_match_stabilizer_zero_noise(self):
        """meas_sampling and stabilizer agree exactly on a noiseless new-gate circuit."""
        tc = TickCircuit()
        tc.tick().pz([0, 1, 2])
        tc.tick().x([0])
        tc.tick().cy([(0, 1)])
        tc.tick().szz([(1, 2)])
        tc.tick().swap([(1, 2)])
        tc.tick().sx([2])
        tc.tick().sxdg([2])
        tick = tc.tick()
        tick.mz([0])
        tick.mz([1])
        tick.mz([2])
        tc.set_meta("num_measurements", "3")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        depol = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0)
        shots = 32
        meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(shots)).seed(11).run()
        stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).sampling(monte_carlo(shots)).seed(99).run()

        assert len(meas_r) == len(stab_r) == shots
        for shot in range(shots):
            assert list(meas_r[shot]) == list(stab_r[shot]) == [1, 0, 1]


class TestCYGateSupport:
    """CY gate should work through the meas_sampling public path."""

    def test_cy_sign_has_circuit_level_measurement_effect(self):
        """CY maps XX to -YZ, so measuring YZ after CY gives odd parity."""
        tc = TickCircuit()
        tc.tick().pz([0, 1])
        tc.tick().h([0])
        tc.tick().h([1])
        tc.tick().cy([(0, 1)])
        tc.tick().f([0])
        tick = tc.tick()
        tick.mz([0])
        tick.mz([1])
        tc.set_meta("num_measurements", "2")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        depol = depolarizing().p1(0).p2(0).p_meas(0).p_prep(0)
        for backend in (stabilizer(), statevec()):
            result = sim_neo(tc).quantum(backend).noise(depol).sampling(monte_carlo(32)).seed(123).run()
            for shot in range(result.num_shots):
                row = list(result[shot])
                assert row[0] ^ row[1] == 1

    def test_cy_circuit_shape_and_values(self):
        """H(0) CY(0,1) MZ(0) MZ(1): 2 measurements, no unsupported-gate error."""
        tc = TickCircuit()
        tc.tick().h([0])
        tc.tick().cy([(0, 1)])
        tick = tc.tick()
        tick.mz([0])
        tick = tc.tick()
        tick.mz([1])
        tc.set_meta("num_measurements", "2")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        depol = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)
        result = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(100)).seed(42).run()

        assert len(result) == 100
        assert len(result[0]) == 2
        for shot in range(100):
            for meas in range(2):
                assert result.get(shot, meas) in (0, 1)

    def test_cy_matches_stabilizer_protocol(self):
        """CY circuit: meas_sampling and stabilizer produce same output shape."""
        tc = TickCircuit()
        tc.tick().h([0])
        tc.tick().cy([(0, 1)])
        tick = tc.tick()
        tick.mz([0])
        tick = tc.tick()
        tick.mz([1])
        tc.set_meta("num_measurements", "2")
        tc.set_meta("detectors", "[]")
        tc.set_meta("observables", "[]")

        depol = depolarizing().p1(0.005).p2(0.005).p_meas(0.005).p_prep(0.005)

        meas_r = sim_neo(tc).quantum(meas_sampling()).noise(depol).sampling(monte_carlo(100)).seed(42).run()
        stab_r = sim_neo(tc).quantum(stabilizer()).noise(depol).sampling(monte_carlo(100)).seed(42).run()

        assert len(meas_r) == len(stab_r) == 100
        assert meas_r.num_measurements == stab_r.num_measurements == 2
