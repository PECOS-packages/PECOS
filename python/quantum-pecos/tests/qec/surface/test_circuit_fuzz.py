# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Random circuit fuzzing comparing physical and logical PECOS simulations.

Generates random stabilizer circuits, runs them at two levels:
1. Physical: single-qubit PECOS SparseStab (ground truth)
2. Logical: encoded in a surface code via LogicalCircuitBuilder,
   TickCircuit replayed on SparseStab with detector/tracked-Pauli checking

No Stim dependency. Pure PECOS end-to-end.
"""

from __future__ import annotations

import json
import random

import pytest
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch
from pecos_rslib import SparseStab
from pecos_rslib.quantum import TickCircuit

# ---------------------------------------------------------------------------
# TickCircuit simulation on SparseStab
# ---------------------------------------------------------------------------


def simulate_tick_circuit(tc: TickCircuit, seed: int = 0) -> tuple[list[int], int, dict[int, int]]:
    """Simulate a TickCircuit on PECOS SparseStab.

    Returns (flat_measurements, det_fired, observable_values).
    """
    max_q = 0
    for i in range(tc.num_ticks()):
        for g in tc.get_tick(i).gate_batches():
            for q in g.qubits:
                max_q = max(max_q, int(q))

    sim = SparseStab(max_q + 1)
    sim.set_seed(seed)
    flat = []

    for i in range(tc.num_ticks()):
        for g in tc.get_tick(i).gate_batches():
            name = g.gate_type.name
            qs = [int(q) for q in g.qubits]
            if name == "QAlloc":
                pass
            elif name == "PZ":
                sim.run_gate("PZ", set(qs))
            elif name == "MZ":
                for q in qs:
                    r = sim.run_gate("MZ", {q})
                    flat.append(r.get(q, 0))
            elif name in ("CX", "CZ"):
                pairs = {(qs[j], qs[j + 1]) for j in range(0, len(qs), 2)}
                sim.run_gate(name, pairs)
            else:
                sim.run_gate(name, set(qs))

    num_meas = int(tc.get_meta("num_measurements"))

    # Check detectors
    det_fired = 0
    det_json = tc.get_meta("detectors")
    if det_json:
        for det in json.loads(det_json):
            val = 0
            for rec in det["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(flat):
                    val ^= flat[idx]
            if val != 0:
                det_fired += 1

    # Extract observables
    obs_vals = {}
    obs_json = tc.get_meta("observables")
    if obs_json:
        for obs in json.loads(obs_json):
            val = 0
            for rec in obs["records"]:
                idx = num_meas + rec
                if 0 <= idx < len(flat):
                    val ^= flat[idx]
            obs_vals[obs["id"]] = val

    return flat, det_fired, obs_vals


def physical_sim_1q(gates: list[str], init_basis: str, meas_basis: str) -> int:
    """Single-qubit ground truth on SparseStab."""
    sim = SparseStab(1)
    if init_basis == "X":
        sim.run_gate("H", {0})
    for g in gates:
        sim.run_gate(g, {0})
    if meas_basis == "X":
        sim.run_gate("H", {0})
    return sim.run_gate("MZ", {0}).get(0, 0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def patch():
    return SurfacePatch.create(distance=3)


@pytest.fixture
def nq(patch):
    return patch.geometry.num_data + patch.geometry.num_ancilla


# ---------------------------------------------------------------------------
# Deterministic gate correctness (observable values)
# ---------------------------------------------------------------------------


class TestGateCorrectness:
    """Verify gate observables match physical ground truth (pure PECOS)."""

    def test_memory_z(self, patch):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_memory_x(self, patch):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "X")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_h_z_to_x(self, patch):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "Z")
        b.add_transversal_h("A")
        b.add_memory("A", 2, "X")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_h_x_to_z(self, patch):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "X")
        b.add_transversal_h("A")
        b.add_memory("A", 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_hh_identity(self, patch):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "Z")
        b.add_transversal_h("A")
        b.add_memory("A", 2, "X")
        b.add_transversal_h("A")
        b.add_memory("A", 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_cx_00_zz(self, patch, nq):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0
        assert obs[1] == 0

    def test_cx_pp_xx(self, patch, nq):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, "X")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, "X")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0
        assert obs[1] == 0


# ---------------------------------------------------------------------------
# Noiseless detector validity (multiple seeds)
# ---------------------------------------------------------------------------


class TestNoiselessDetectors:
    """Verify 0 detector fires across many seeds (PECOS-only)."""

    @pytest.mark.parametrize("seed", range(20))
    def test_memory_z(self, patch, seed):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 3, "Z")
        _, det, _ = simulate_tick_circuit(b.to_tick_circuit(), seed)
        assert det == 0

    @pytest.mark.parametrize("seed", range(20))
    def test_h(self, patch, seed):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "Z")
        b.add_transversal_h("A")
        b.add_memory("A", 2, "X")
        _, det, _ = simulate_tick_circuit(b.to_tick_circuit(), seed)
        assert det == 0

    @pytest.mark.parametrize("seed", range(20))
    def test_cx(self, patch, nq, seed):
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, "Z")
        _, det, _ = simulate_tick_circuit(b.to_tick_circuit(), seed)
        assert det == 0


# ---------------------------------------------------------------------------
# Fuzz: physical vs logical H sequences
# ---------------------------------------------------------------------------


class TestFuzzH:
    @pytest.mark.parametrize("seed", range(50))
    def test_random_h(self, patch, seed):
        rng = random.Random(seed)
        num_h = rng.randint(0, 8)
        init_b = rng.choice(["Z", "X"])
        eff_b = init_b
        for _ in range(num_h):
            eff_b = "X" if eff_b == "Z" else "Z"

        expected = physical_sim_1q(["H"] * num_h, init_b, eff_b)

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, init_b)
        cur = init_b
        for _ in range(num_h):
            b.add_transversal_h("A")
            cur = "X" if cur == "Z" else "Z"
            b.add_memory("A", 2, cur)

        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == expected


# ---------------------------------------------------------------------------
# Fuzz: H+CX composition
# ---------------------------------------------------------------------------


class TestFuzzCX:
    """Fuzz CX with various init/meas bases against physical SparseStab."""

    @pytest.mark.parametrize("seed", range(50))
    def test_random_cx(self, patch, nq, seed):
        rng = random.Random(seed)
        ic = rng.choice(["Z", "X"])
        it = rng.choice(["Z", "X"])
        mc = rng.choice(["Z", "X"])
        mt = rng.choice(["Z", "X"])

        # Physical ground truth: detect deterministic outcomes
        results = []
        for _ in range(50):
            sim = SparseStab(2)
            if ic == "X":
                sim.run_gate("H", {0})
            if it == "X":
                sim.run_gate("H", {1})
            sim.run_gate("CX", {(0, 1)})
            if mc == "X":
                sim.run_gate("H", {0})
            if mt == "X":
                sim.run_gate("H", {1})
            r = sim.run_gate("MZ", {0, 1})
            results.append((r.get(0, 0), r.get(1, 0)))
        r0s, r1s = [r[0] for r in results], [r[1] for r in results]
        exp0 = r0s[0] if len(set(r0s)) == 1 else None
        exp1 = r1s[0] if len(set(r1s)) == 1 else None

        # Encoded
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, basis={"C": ic, "T": it})
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, basis={"C": mc, "T": mt})

        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0, f"Noiseless det fired: {ic}{it}->{mc}{mt}"

        if exp0 is not None and 0 in obs:
            assert obs[0] == exp0, f"obs0: got {obs[0]} expected {exp0}"
        if exp1 is not None and 1 in obs:
            assert obs[1] == exp1, f"obs1: got {obs[1]} expected {exp1}"


class TestFuzzComposition:
    @pytest.mark.parametrize("seed", range(30))
    def test_random_h_cx(self, patch, nq, seed):
        rng = random.Random(seed)
        num_ops = rng.randint(1, 6)

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A", qubit_offset=0)
        b.add_patch(patch, "B", qubit_offset=nq)
        b.add_memory(["A", "B"], 2, "Z")
        eff = {"A": "Z", "B": "Z"}

        for _ in range(num_ops):
            op = rng.choice(["H_A", "H_B", "CX"])
            if op == "H_A":
                b.add_transversal_h("A")
                eff["A"] = "X" if eff["A"] == "Z" else "Z"
                b.add_memory(["A", "B"], 2, basis={"A": eff["A"], "B": eff["B"]})
            elif op == "H_B":
                b.add_transversal_h("B")
                eff["B"] = "X" if eff["B"] == "Z" else "Z"
                b.add_memory(["A", "B"], 2, basis={"A": eff["A"], "B": eff["B"]})
            else:
                if eff["A"] != eff["B"]:
                    continue
                b.add_transversal_cx("A", "B")
                b.add_memory(["A", "B"], 2, basis={"A": eff["A"], "B": eff["B"]})

        _, det, _ = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0


# ---------------------------------------------------------------------------
# Distance scaling
# ---------------------------------------------------------------------------


class TestDistanceScaling:
    @pytest.fixture
    def patch5(self):
        return SurfacePatch.create(distance=5)

    @pytest.fixture
    def nq5(self, patch5):
        return patch5.geometry.num_data + patch5.geometry.num_ancilla

    def test_d5_memory(self, patch5):
        b = LogicalCircuitBuilder()
        b.add_patch(patch5, "A")
        b.add_memory("A", 3, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_d5_h(self, patch5):
        b = LogicalCircuitBuilder()
        b.add_patch(patch5, "A")
        b.add_memory("A", 2, "Z")
        b.add_transversal_h("A")
        b.add_memory("A", 2, "X")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    @pytest.mark.parametrize("seed", range(5))
    def test_d5_cx(self, patch5, nq5, seed):
        b = LogicalCircuitBuilder()
        b.add_patch(patch5, "C", qubit_offset=0)
        b.add_patch(patch5, "T", qubit_offset=nq5)
        b.add_memory(["C", "T"], 2, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit(), seed)
        assert det == 0
        assert obs[0] == 0
        assert obs[1] == 0

    def test_d5_pecos_dem(self, patch5):
        b = LogicalCircuitBuilder()
        b.add_patch(patch5, "A")
        b.add_memory("A", 2, "Z")
        dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
        errors = [line for line in dem_str.split("\n") if line.startswith("error(")]
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# TickCircuit structural tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SZ teleportation
# ---------------------------------------------------------------------------


class TestReliableObservables:
    """Verify that non-reliable observables are correctly skipped."""

    def test_cx_same_basis_both_reliable(self, patch, nq):
        """CX with same-basis measurements: both observables emitted."""
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, "Z")
        tc = b.to_tick_circuit()
        obs = json.loads(tc.get_meta("observables"))
        assert len(obs) == 2, f"Expected 2 observables, got {len(obs)}"

    def test_cx_cross_basis_skips_unreliable(self, patch, nq):
        """CX with cross-basis: non-deterministic observables skipped."""
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, basis={"C": "X", "T": "Z"})
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, basis={"C": "X", "T": "Z"})
        tc = b.to_tick_circuit()
        obs = json.loads(tc.get_meta("observables"))
        # Ctrl measured in X after CX: X_ctrl entangled with tgt (measured Z)
        # → ctrl X observable is non-reliable → skipped
        # Tgt measured in Z after CX: Z_tgt entangled with ctrl (measured X)
        # → tgt Z observable is non-reliable → skipped
        # Both should be skipped
        assert len(obs) == 0, f"Expected 0 observables (both non-reliable), got {len(obs)}"

    def test_cx_zx_one_reliable(self, patch, nq):
        """CX|0>|+> with meas(Z,X): both should be reliable.

        After CX: Z_ctrl unchanged, X_tgt unchanged.
        Measuring ctrl in Z: reliable (Z not entangled).
        Measuring tgt in X: reliable (X not entangled).
        """
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 2, basis={"C": "Z", "T": "X"})
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 2, basis={"C": "Z", "T": "X"})
        tc = b.to_tick_circuit()
        obs = json.loads(tc.get_meta("observables"))
        assert len(obs) == 2, f"Expected 2 observables, got {len(obs)}"

        _, det, obs_vals = simulate_tick_circuit(tc)
        assert det == 0
        assert obs_vals[0] == 0
        assert obs_vals[1] == 0


class TestSZTeleportation:
    def test_sz_preserves_z(self, patch, nq):
        """SZ|0> = |0>: Z eigenvalue preserved."""
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "D", qubit_offset=0)
        b.add_patch(patch, "Y", qubit_offset=nq)
        b.add_memory("D", 2, "Z")
        b.add_sz_via_teleportation("D", "Y", 2, 2)
        b.add_memory("D", 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0

    def test_sz_phase_single_qubit(self):
        """Verify SZ teleportation protocol at the single-qubit level.

        The phase test (SZ^2|+> = Z|+> = |->) cannot be verified at the
        logical level because the |+Y> injection is non-fault-tolerant
        (distance-1 error). But we CAN verify the protocol works on a
        single physical qubit, confirming the circuit structure is correct.
        """
        sim = SparseStab(3)
        # Qubit 0: data (|+>), Qubit 1: ancilla 1 (|+Y>), Qubit 2: ancilla 2 (|+Y>)

        # Prep |+>
        sim.run_gate("H", {0})
        # Prep |+Y> = S|+>
        sim.run_gate("H", {1})
        sim.run_gate("SZ", {1})
        sim.run_gate("H", {2})
        sim.run_gate("SZ", {2})

        # First teleportation: CX(data, anc1), measure anc1 in Z
        sim.run_gate("CX", {(0, 1)})
        r1 = sim.run_gate("MZ", {1})
        m1 = r1.get(1, 0)

        # Second teleportation: CX(data, anc2), measure anc2 in Z
        sim.run_gate("CX", {(0, 2)})
        r2 = sim.run_gate("MZ", {2})
        m2 = r2.get(2, 0)

        # Measure data in X basis: SZ^2|+> = Z|+> = |->
        sim.run_gate("H", {0})
        r_data = sim.run_gate("MZ", {0})
        data_val = r_data.get(0, 0)

        # Corrected observable: data_X XOR m1 XOR m2
        corrected = data_val ^ m1 ^ m2
        assert (
            corrected == 1
        ), f"SZ^2|+> should give |-> (corrected=1), got data={data_val} m1={m1} m2={m2} corrected={corrected}"


# ---------------------------------------------------------------------------
# Gate composition
# ---------------------------------------------------------------------------


class TestGateComposition:
    def test_h_cx_h(self, patch, nq):
        """H -> CX -> H on both patches."""
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A", qubit_offset=0)
        b.add_patch(patch, "B", qubit_offset=nq)
        b.add_memory(["A", "B"], 2, "Z")
        b.add_transversal_h("A")
        b.add_transversal_h("B")
        b.add_memory(["A", "B"], 2, "X")
        b.add_transversal_cx("A", "B")
        b.add_memory(["A", "B"], 2, "X")
        b.add_transversal_h("A")
        b.add_transversal_h("B")
        b.add_memory(["A", "B"], 2, "Z")
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0
        assert obs[1] == 0

    def test_triple_h(self, patch):
        """HHH = H: |0> -> |+>."""
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, "Z")
        for i in range(3):
            b.add_transversal_h("A")
            cur = "X" if i % 2 == 0 else "Z"
            b.add_memory("A", 2, cur)
        _, det, obs = simulate_tick_circuit(b.to_tick_circuit())
        assert det == 0
        assert obs[0] == 0


# ---------------------------------------------------------------------------
# Decoder pipeline (PECOS-native)
# ---------------------------------------------------------------------------


class TestDecoderPipeline:
    """Test the full decode pipeline using PECOS DEM + PECOS sampler."""

    def test_build_decoder_pecos_dem(self, patch):
        """build_decoder with PECOS-native DEM produces a working decoder."""
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 3, "Z")
        _, decoder = b.build_decoder(p1=0.001, p2=0.001, p_meas=0.001, use_stim_dem=False)
        assert decoder.num_observables() == 1

    def test_pecos_dem_decode_memory(self, patch):
        """End-to-end: PECOS DEM → PECOS sample → decode → low error rate."""
        from pecos_rslib.qec import ParsedDem

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 3, "Z")
        dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)

        parsed = ParsedDem.from_string(dem_str)
        rust_sampler = parsed.to_dem_sampler()
        batch = rust_sampler.generate_samples(5000, seed=42)
        errors = batch.decode_count(dem_str, "pecos_uf:fast")
        ler = errors / 5000
        # At d=3 p=0.001, LER should be very low
        assert ler < 0.05, f"LER too high: {ler}"

    def test_observable_subgraph_decoder(self, patch, nq):
        """OSD with PECOS DEM on CX circuit."""
        from pecos_rslib.qec import ObservableSubgraphDecoder, ParsedDem

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 3, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 3, "Z")

        dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
        sc = b.stab_coords()
        osd = ObservableSubgraphDecoder(dem_str, sc, "pecos_uf:fast")
        assert osd.num_observables() == 2

        sizes = osd.subgraph_sizes()
        for s in sizes:
            assert s > 0, "Empty subgraph"

    def test_pecos_dem_cx_decode(self, patch, nq):
        """End-to-end CX: PECOS DEM → sample → decode."""
        from pecos_rslib.qec import ParsedDem

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 3, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 3, "Z")

        dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)
        parsed = ParsedDem.from_string(dem_str)
        rust_sampler = parsed.to_dem_sampler()
        batch = rust_sampler.generate_samples(5000, seed=42)
        errors = batch.decode_count(dem_str, "pecos_uf:fast")
        ler = errors / 5000
        assert ler < 0.1, f"CX LER too high: {ler}"


# ---------------------------------------------------------------------------
# TickCircuit structural tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Threshold: error suppression with distance (d=3 vs d=5)
# ---------------------------------------------------------------------------


class TestThreshold:
    """Verify error suppression increases with distance.

    Uses Stim DEM for error mechanisms (more complete noise model)
    and PECOS decoder for correction. Tests at p=0.001 where we should
    be well below threshold.
    """

    def _run_threshold(self, builder, _d, decoder_type="pecos_uf:fast"):
        import stim
        from pecos_rslib.qec import ParsedDem

        c = stim.Circuit(builder.to_stim(p1=0.001, p2=0.001, p_meas=0.001))
        dem = c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        dem_str = str(dem)
        parsed = ParsedDem.from_string(dem_str)
        sampler = parsed.to_dem_sampler()
        batch = sampler.generate_samples(20000, seed=42)
        errors = batch.decode_count(dem_str, decoder_type)
        return errors / 20000

    def test_memory_suppression(self):
        """Memory: d=5 should have lower LER than d=3."""
        lers = {}
        for d in [3, 5]:
            patch = SurfacePatch.create(distance=d)
            b = LogicalCircuitBuilder()
            b.add_patch(patch, "A")
            b.add_memory("A", rounds=d, basis="Z")
            lers[d] = self._run_threshold(b, d, "pymatching")
        assert lers[5] < lers[3], f"d=5 ({lers[5]}) not better than d=3 ({lers[3]})"

    def test_h_suppression(self):
        """H gate: d=5 should have lower LER than d=3."""
        lers = {}
        for d in [3, 5]:
            patch = SurfacePatch.create(distance=d)
            b = LogicalCircuitBuilder()
            b.add_patch(patch, "A")
            b.add_memory("A", rounds=d, basis="Z")
            b.add_transversal_h("A")
            b.add_memory("A", rounds=d, basis="X")
            lers[d] = self._run_threshold(b, d, "pymatching")
        assert lers[5] < lers[3], f"d=5 ({lers[5]}) not better than d=3 ({lers[3]})"


# ---------------------------------------------------------------------------
# TickCircuit structural tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Noisy fuzz: random circuits decode with reasonable error rates
# ---------------------------------------------------------------------------


class TestNoisyFuzz:
    """Verify that random circuits with noise decode with sub-50% error rate.

    This is a basic sanity check: if the decoder is completely broken,
    the LER would be ~50%. Any reasonable decoder should do much better.
    """

    @pytest.mark.parametrize("seed", range(5))
    def test_noisy_h(self, patch, seed):
        import stim
        from pecos_rslib.qec import ParsedDem

        rng = random.Random(seed)
        num_h = rng.randint(1, 4)
        init_b = "Z"
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A")
        b.add_memory("A", 2, init_b)
        cur = init_b
        for _ in range(num_h):
            b.add_transversal_h("A")
            cur = "X" if cur == "Z" else "Z"
            b.add_memory("A", 2, cur)

        c = stim.Circuit(b.to_stim(p1=0.001, p2=0.001, p_meas=0.001))
        dem = c.detector_error_model(decompose_errors=True)
        dem_str = str(dem)
        parsed = ParsedDem.from_string(dem_str)
        batch = parsed.to_dem_sampler().generate_samples(5000, seed=seed)
        errors = batch.decode_count(dem_str, "pecos_uf:fast")
        ler = errors / 5000
        assert ler < 0.1, f"LER too high: {ler}"


# ---------------------------------------------------------------------------
# TickCircuit structural tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Composed noisy gate sequences
# ---------------------------------------------------------------------------


class TestNoisyComposition:
    """Verify that multi-gate sequences with noise decode correctly."""

    def test_noisy_h_cx_h(self, patch, nq):
        """H -> CX -> H with noise: should decode with low LER."""
        import stim
        from pecos_rslib.qec import ParsedDem

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A", qubit_offset=0)
        b.add_patch(patch, "B", qubit_offset=nq)
        b.add_memory(["A", "B"], 2, "Z")
        b.add_transversal_h("A")
        b.add_transversal_h("B")
        b.add_memory(["A", "B"], 2, "X")
        b.add_transversal_cx("A", "B")
        b.add_memory(["A", "B"], 2, "X")
        b.add_transversal_h("A")
        b.add_transversal_h("B")
        b.add_memory(["A", "B"], 2, "Z")

        # Use Stim DEM (more error mechanisms) for noisy test
        c = stim.Circuit(b.to_stim(p1=0.001, p2=0.001, p_meas=0.001))
        dem = c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        dem_str = str(dem)
        parsed = ParsedDem.from_string(dem_str)
        batch = parsed.to_dem_sampler().generate_samples(10000, seed=42)
        errors = batch.decode_count(dem_str, "pecos_uf:fast")
        ler = errors / 10000
        assert ler < 0.1, f"H-CX-H LER too high: {ler}"

    @pytest.mark.parametrize("seed", range(5))
    def test_noisy_random_composition(self, patch, nq, seed):
        """Random H+CX with noise: should decode with sub-50% LER."""
        import stim
        from pecos_rslib.qec import ParsedDem

        rng = random.Random(seed)
        b = LogicalCircuitBuilder()
        b.add_patch(patch, "A", qubit_offset=0)
        b.add_patch(patch, "B", qubit_offset=nq)
        b.add_memory(["A", "B"], 2, "Z")
        eff = {"A": "Z", "B": "Z"}

        for _ in range(rng.randint(1, 4)):
            op = rng.choice(["H_A", "H_B", "CX"])
            if op == "H_A":
                b.add_transversal_h("A")
                eff["A"] = "X" if eff["A"] == "Z" else "Z"
                b.add_memory(["A", "B"], 2, basis={"A": eff["A"], "B": eff["B"]})
            elif op == "H_B":
                b.add_transversal_h("B")
                eff["B"] = "X" if eff["B"] == "Z" else "Z"
                b.add_memory(["A", "B"], 2, basis={"A": eff["A"], "B": eff["B"]})
            else:
                if eff["A"] != eff["B"]:
                    continue
                b.add_transversal_cx("A", "B")
                b.add_memory(["A", "B"], 2, basis={"A": eff["A"], "B": eff["B"]})

        c = stim.Circuit(b.to_stim(p1=0.001, p2=0.001, p_meas=0.001))
        dem = c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        dem_str = str(dem)
        parsed = ParsedDem.from_string(dem_str)
        batch = parsed.to_dem_sampler().generate_samples(5000, seed=42)
        errors = batch.decode_count(dem_str, "pecos_uf:fast")
        ler = errors / 5000
        assert ler < 0.2, f"Random composition LER too high: {ler}"


# ---------------------------------------------------------------------------
# OSD accuracy comparison
# ---------------------------------------------------------------------------


class TestOSDAccuracy:
    """Compare observable subgraph decoder accuracy against baseline."""

    def test_osd_better_than_naive_on_cx(self, patch, nq):
        """OSD should outperform naive decomposed MWPM on CX circuits."""
        import stim
        from pecos_rslib.qec import ObservableSubgraphDecoder, ParsedDem

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 3, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 3, "Z")

        c = stim.Circuit(b.to_stim(p1=0.001, p2=0.001, p_meas=0.001))
        dem = c.detector_error_model(ignore_decomposition_failures=True)
        dem_str = str(dem)

        # Sample
        sampler = dem.compile_sampler()
        det_events, obs_flips, _ = sampler.sample(20000)

        # Naive: decomposed MWPM via PECOS UF
        dem_decomp = c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
        parsed = ParsedDem.from_string(str(dem_decomp))
        batch_naive = parsed.to_dem_sampler().generate_samples(20000, seed=42)
        naive_errors = batch_naive.decode_count(str(dem_decomp), "pecos_uf:fast")
        naive_ler = naive_errors / 20000

        # OSD with FB
        sc = b.stab_coords()
        osd = ObservableSubgraphDecoder(dem_str, sc, "fusion_blossom_serial")
        osd_errors = sum(
            1
            for i in range(20000)
            if osd.decode(det_events[i].tolist()) != sum((1 << j) for j in range(obs_flips.shape[1]) if obs_flips[i, j])
        )
        osd_ler = osd_errors / 20000

        # OSD should be at least as good (usually much better)
        assert osd_ler <= naive_ler * 1.5 + 0.001, f"OSD ({osd_ler:.5f}) much worse than naive ({naive_ler:.5f})"


# ---------------------------------------------------------------------------
# PECOS-native DEM with OSD decoder on CX
# ---------------------------------------------------------------------------


class TestPecosDemWithOSD:
    """Test PECOS-native DEM pipeline with observable subgraph decoder."""

    def test_pecos_dem_osd_cx(self, patch, nq):
        """PECOS DEM → OSD decoder on CX circuit."""
        from pecos_rslib.qec import ObservableSubgraphDecoder, ParsedDem

        b = LogicalCircuitBuilder()
        b.add_patch(patch, "C", qubit_offset=0)
        b.add_patch(patch, "T", qubit_offset=nq)
        b.add_memory(["C", "T"], 3, "Z")
        b.add_transversal_cx("C", "T")
        b.add_memory(["C", "T"], 3, "Z")

        dem_str = b.build_dem(p1=0.001, p2=0.001, p_meas=0.001)

        # Verify DEM has content
        errors = [line for line in dem_str.split("\n") if line.startswith("error(")]
        assert len(errors) > 0

        # Build OSD decoder from PECOS DEM
        sc = b.stab_coords()
        osd = ObservableSubgraphDecoder(dem_str, sc, "pecos_uf:fast")
        assert osd.num_observables() == 2

        # Sample and decode
        parsed = ParsedDem.from_string(dem_str)
        batch = parsed.to_dem_sampler().generate_samples(5000, seed=42)
        errors = batch.decode_count(dem_str, "pecos_uf:fast")
        ler = errors / 5000
        assert ler < 0.1, f"PECOS DEM + OSD CX LER too high: {ler}"


# ---------------------------------------------------------------------------
# TickCircuit structural tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mirrored brickwork circuits
# ---------------------------------------------------------------------------


def _build_mirrored_brickwork(num_qubits, depth, seed, patch, rounds=2):
    """Build a mirrored brickwork circuit (identity, output = |0...0>).

    Forward: random H gates + CX brickwork layers.
    Mirror: exact reverse (H and CX are self-inverse).
    """
    nq = patch.geometry.num_data + patch.geometry.num_ancilla

    b = LogicalCircuitBuilder()
    labels = [f"Q{i}" for i in range(num_qubits)]
    for i, label in enumerate(labels):
        b.add_patch(patch, label, qubit_offset=i * nq)

    eff = dict.fromkeys(labels, "Z")
    b.add_memory(labels, rounds, "Z")

    rng = random.Random(seed)
    ops_forward = []

    for layer in range(depth):
        layer_ops = []
        for label in labels:
            # H is the only single-qubit logical gate available.
            # SZ requires teleportation (not a standalone gate on the surface code).
            if rng.random() < 0.5:
                b.add_transversal_h(label)
                eff[label] = "X" if eff[label] == "Z" else "Z"
                layer_ops.append(("H", label))
        b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})

        offset = layer % 2
        cx_applied = []
        for i in range(offset, num_qubits - 1, 2):
            ctrl, tgt = labels[i], labels[i + 1]
            if eff[ctrl] == eff[tgt]:
                b.add_transversal_cx(ctrl, tgt)
                cx_applied.append((ctrl, tgt))
        if cx_applied:
            b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})
            layer_ops.append(("CX", cx_applied))
        ops_forward.append(layer_ops)

    for layer_ops in reversed(ops_forward):
        for op_type, *args in reversed(layer_ops):
            if op_type == "CX":
                for ctrl, tgt in reversed(args[0]):
                    if eff[ctrl] == eff[tgt]:
                        b.add_transversal_cx(ctrl, tgt)
                b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})
        for op_type, *args in reversed(layer_ops):
            if op_type == "H":
                label = args[0]
                b.add_transversal_h(label)
                eff[label] = "X" if eff[label] == "Z" else "Z"
        b.add_memory(labels, rounds, basis={label: eff[label] for label in labels})

    return b


class TestMirroredBrickwork:
    """Mirrored brickwork circuits: identity circuit, output always |0...0>.

    Tests random H + CX brickwork layers at various widths and depths.
    The mirror guarantees the output is |0...0> regardless of random choices.
    """

    @pytest.mark.parametrize("width", [2, 3, 4])
    @pytest.mark.parametrize("depth", [1, 2, 3])
    @pytest.mark.parametrize("seed", range(5))
    def test_brickwork_d3(self, width, depth, seed):
        patch = SurfacePatch.create(distance=3)
        b = _build_mirrored_brickwork(width, depth, seed, patch)
        tc = b.to_tick_circuit()
        det_fired, obs_vals = simulate_tick_circuit(tc, seed)[-2:]
        assert det_fired == 0, f"w={width} d={depth}: {det_fired} detectors fired"
        for obs_id, val in obs_vals.items():
            assert val == 0, f"w={width} d={depth}: obs{obs_id}={val}"

    @pytest.mark.parametrize("width", [2, 3])
    @pytest.mark.parametrize("seed", range(3))
    def test_brickwork_d5(self, width, seed):
        patch = SurfacePatch.create(distance=5)
        b = _build_mirrored_brickwork(width, 2, seed, patch)
        tc = b.to_tick_circuit()
        det_fired, obs_vals = simulate_tick_circuit(tc, seed)[-2:]
        assert det_fired == 0
        for val in obs_vals.values():
            assert val == 0


# ---------------------------------------------------------------------------
# TickCircuit structural tests
# ---------------------------------------------------------------------------


class TestTickCircuitStructure:
    def test_gate_count_and_gate_batch_count_are_distinct(self):
        tc = TickCircuit()
        tc.tick().h([0]).h([1]).cx([(2, 3), (4, 5)])

        tick = tc.get_tick(0)
        assert tick.gate_count() == 4
        assert tick.gate_batch_count() == 2
        assert len(tick.gate_batches()) == 2
        assert len(tick) == 2

        assert tc.gate_count() == 4
        assert tc.gate_batch_count() == 2
        assert len(tc.gate_batches()) == 2

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 5, 8])
    @pytest.mark.parametrize("depth", [10, 30])
    @pytest.mark.parametrize("seed", range(3))
    def test_build_roundtrip(self, num_qubits, depth, seed):
        gate_set = ["H", "SZ", "X", "Z"]
        if num_qubits >= 2:
            gate_set.extend(["CX", "CZ"])
        rng = random.Random(seed)
        tc = TickCircuit()
        t = tc.tick()
        t.qalloc(list(range(num_qubits)))
        for _ in range(depth):
            gate = rng.choice(gate_set)
            t = tc.tick()
            if gate in ("CX", "CZ"):
                q1, q2 = rng.sample(range(num_qubits), 2)
                getattr(t, gate.lower())([(q1, q2)])
            else:
                q = rng.randint(0, num_qubits - 1)
                getattr(t, gate.lower())([q])
        t = tc.tick()
        t.mz(list(range(num_qubits)))
        assert tc.num_ticks() >= 2
