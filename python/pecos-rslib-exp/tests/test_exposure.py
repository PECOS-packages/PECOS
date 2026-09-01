"""Smoke tests for the approved experimental Python exposure set.

REQUIRES a built ``pecos_rslib_exp`` extension. The downstream harness venv
builds the extension before running this file; this test is not wired into Cargo.
"""

import math

import pecos_rslib_exp as exp
import pytest
from pecos.simulators import StateVec

STATS_KEYS = {
    "total_nonclifford",
    "single_site",
    "multi_disent",
    "deferred_disent_bypass",
    "numerical_redetect",
    "multi_std",
    "multi_std_add",
    "multi_std_cascade",
    "signed_eigenstate_candidates",
    "stabilizer",
    "ofd_in_span",
    "ofd_new_dim",
    "ofd_in_span_std",
    "ofd_in_span_single",
    "ofd_in_span_disent",
}


def assert_complex_tuple(value):
    assert isinstance(value, tuple)
    assert len(value) == 2
    assert all(isinstance(component, float) for component in value)


def assert_state_vector(actual, expected):
    assert len(actual) == len(expected)
    for actual_amplitude, expected_amplitude in zip(actual, expected, strict=True):
        assert math.isclose(actual_amplitude[0], expected_amplitude[0], abs_tol=1e-12)
        assert math.isclose(actual_amplitude[1], expected_amplitude[1], abs_tol=1e-12)


def test_sim_neo_stab_mps_measurement_and_boolean_builder_options():
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().x([0])
    circuit.tick().mz([0])

    backends = [
        exp.stab_mps().measurement("exact").merge_rz(),
        exp.stab_mps().measurement("lazy").merge_rz(True),
        exp.stab_mps().measurement("pragmatic").merge_rz(False),
    ]
    for backend in backends:
        result = exp.sim_neo(circuit).quantum(backend).sampling(exp.monte_carlo(1)).seed(7).run()
        assert [list(row) for row in result] == [[1]]

    with pytest.raises(ValueError, match="measurement must be one of"):
        exp.stab_mps().measurement("eager")


def test_sim_neo_rejects_rotation_with_wrong_angle_arity():
    from pecos_rslib.quantum import Gate, GateType

    class Tick:
        def __init__(self):
            self.calls = 0

        def gate_batches(self):
            self.calls += 1
            if self.calls == 1:
                return [Gate(GateType.RZ, params=[0.5], qubits=[0])]
            return [Gate(GateType.RZ, qubits=[0])]

    class Circuit:
        def __init__(self):
            self.tick = Tick()

        def num_ticks(self):
            return 1

        def get_tick(self, _index):
            return self.tick

        def annotations(self):
            return []

    with pytest.raises(ValueError, match="Gate RZ expected 1 angle parameters, got 0"):
        exp.sim_neo(Circuit())


def test_sim_neo_python_fallback_crz_preserves_full_matrix():
    class GateType:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"GateType.{self.name}"

    class Gate:
        def __init__(self, name, qubits, angles=()):
            self.gate_type = GateType(name)
            self.qubits = list(qubits)
            self.angles = list(angles)

    class Tick:
        def __init__(self, gates):
            self.gates = gates

        def gate_batches(self):
            return self.gates

    class Circuit:
        def __init__(self, basis, theta):
            prep = []
            if basis & 1:
                prep.append(Gate("X", [0]))
            if basis & 2:
                prep.append(Gate("X", [1]))
            self.ticks = [Tick(prep), Tick([Gate("CRZ", [1, 0], [theta])])]

        def num_ticks(self):
            return len(self.ticks)

        def get_tick(self, index):
            return self.ticks[index]

        def annotations(self):
            return []

    for theta in (-math.pi, math.pi / 3, math.pi, math.tau, 3 * math.pi):
        columns = []
        for basis in range(4):
            native = exp.neo_fallback_native_gates(Circuit(basis, theta))
            simulator = StateVec(2)
            for _, name, qubits, angles in native:
                if name in {"X", "RZ"}:
                    params = {"angle": angles[0]} if name == "RZ" else None
                    for qubit in qubits:
                        simulator.backend.run_1q_gate(name, qubit, params)
                elif name == "RZZ":
                    for offset in range(0, len(qubits), 2):
                        simulator.backend.run_2q_gate(
                            name,
                            (qubits[offset], qubits[offset + 1]),
                            {"angle": angles[0]},
                        )
                else:
                    msg = f"unexpected neo fallback gate {name}"
                    raise AssertionError(msg)
            columns.append([complex(value) for value in simulator.backend.vector])

        half = theta / 2
        reference = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, complex(math.cos(half), -math.sin(half)), 0],
            [0, 0, 0, complex(math.cos(half), math.sin(half))],
        ]
        phase = columns[0][0] / reference[0][0]
        assert abs(abs(phase) - 1) < 1e-12
        if theta in {-math.pi, math.pi / 3, math.pi}:
            assert abs(phase - 1) < 1e-12
        else:
            assert min(abs(phase - 1), abs(phase + 1)) < 1e-12
        for column in range(4):
            for row in range(4):
                assert abs(columns[column][row] / phase - reference[row][column]) < 1e-12


def test_stab_mps_measurement_selection_precedence_and_reset_retention():
    assert exp.StabMps(1).measurement == "exact"
    assert exp.StabMps(1, measurement="pragmatic").measurement == "pragmatic"
    assert exp.StabMps(1, measurement="lazy").measurement == "lazy"
    assert exp.StabMps(1, for_qec=True).measurement == "exact"
    assert exp.StabMps(1, for_qec=True, measurement="exact").measurement == "exact"
    assert exp.StabMps(1, for_qec=True, measurement="pragmatic").measurement == "pragmatic"
    lazy = exp.StabMps(1, for_qec=True, measurement="lazy")
    assert lazy.measurement == "lazy"
    assert lazy.reset().measurement == "lazy"

    with pytest.raises(ValueError, match="measurement must be one of"):
        exp.StabMps(1, measurement="eager")


def test_stab_mps_analysis_and_noise_exposure():
    telemetry_enabled = exp.StabMps(1, saturation_telemetry=True)
    assert telemetry_enabled.stats()["signed_eigenstate_candidates"] == 0

    bell = exp.StabMps(2, seed=7)
    bell.run_1q_gate("H", 0)
    bell.run_2q_gate("CX", (0, 1))

    amplitude = bell.amplitude([False, False])
    assert_complex_tuple(amplitude)
    assert math.isclose(amplitude[0], 1 / math.sqrt(2), abs_tol=1e-12)
    assert math.isclose(amplitude[1], 0.0, abs_tol=1e-12)

    iterative = bell.amplitude_iterative([False, False])
    assert_complex_tuple(iterative)
    assert math.isclose(iterative[0], amplitude[0], abs_tol=1e-12)
    assert math.isclose(iterative[1], amplitude[1], abs_tol=1e-12)

    overlap = bell.overlap_with_stabilizer(
        [[(0, "X"), (1, "X")], [(0, "Z"), (1, "Z")]],
        num_samples=32,
        rng_seed=11,
    )
    assert_complex_tuple(overlap)
    assert math.isclose(overlap[0], 1.0, abs_tol=1e-12)
    assert math.isclose(overlap[1], 0.0, abs_tol=1e-12)

    product = exp.StabMps(2)
    assert math.isclose(product.renyi_s2(1), 0.0, abs_tol=1e-12)
    assert math.isclose(product.s2_pce(1), 0.0, abs_tol=1e-12)
    assert math.isclose(product.s2_pcmps(1), 0.0, abs_tol=1e-12)
    assert isinstance(product.disentangle(1), int)
    assert isinstance(product.bond_cap_hits, int)
    assert isinstance(product.ofd_nullity(), int)
    assert isinstance(product.theoretical_min_bond_dim(), int)
    assert isinstance(product.ofd_disentangled_count(), int)
    assert isinstance(product.ofd_total_absorbed(), int)

    product.run_1q_gate("T", 0)
    stats = product.stats()
    assert isinstance(stats, dict)
    assert set(stats) == STATS_KEYS
    assert all(isinstance(value, int) for value in stats.values())

    bit_flip = exp.StabMps(1, seed=1)
    assert bit_flip.apply_bit_flip(0, 1.0) is True
    assert math.isclose(bit_flip.prob_bitstring([True]), 1.0, abs_tol=1e-12)

    phase_flip = exp.StabMps(1, seed=1)
    phase_flip.run_1q_gate("H", 0)
    assert phase_flip.apply_phase_flip(0, 1.0) is True
    assert math.isclose(
        phase_flip.pauli_expectation([(0, "X")]),
        -1.0,
        abs_tol=1e-12,
    )


def test_stab_mps_named_t_has_conventional_exact_amplitudes():
    inv_sqrt_2 = 1 / math.sqrt(2)

    ht = exp.StabMps(1, merge_rz=False)
    ht.run_1q_gate("H", 0)
    ht.run_1q_gate("T", 0)
    assert_state_vector(ht.state_vector(), [(inv_sqrt_2, 0.0), (0.5, 0.5)])

    t_squared = exp.StabMps(1, merge_rz=False)
    t_squared.run_1q_gate("H", 0)
    t_squared.run_1q_gate("T", 0)
    t_squared.run_1q_gate("T", 0)

    s = exp.StabMps(1, merge_rz=False)
    s.run_1q_gate("H", 0)
    s.run_1q_gate("S", 0)
    expected = [(inv_sqrt_2, 0.0), (0.0, inv_sqrt_2)]
    assert_state_vector(t_squared.state_vector(), expected)
    assert_state_vector(s.state_vector(), expected)


def test_stab_mps_bitstring_convention_auto_flush_and_validation():
    q0_one = exp.StabMps(2, seed=13)
    q0_one.run_1q_gate("X", 0)
    assert not hasattr(q0_one, "sample_bit" + "string")
    assert hasattr(q0_one, "sample_bit" + "strings")
    assert q0_one.sample_bitstrings(4) == [[True, False]] * 4
    assert q0_one.state_vector() == [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert q0_one.amplitude([True, False]) == (1.0, 0.0)
    assert q0_one.amplitude_iterative([True, False]) == (1.0, 0.0)
    assert math.isclose(q0_one.prob_bitstring([True, False]), 1.0)
    assert q0_one.prob_bitstrings(
        [[True, False], [False, False], [True, False]],
    ) == [1.0, 0.0, 1.0]
    assert q0_one.prob_bitstrings([]) == []

    merged = exp.StabMps(2, seed=17, merge_rz=True)
    merged.run_1q_gate("H", 0)
    merged.run_1q_gate("T", 0)
    assert merged.is_state_exact() is False
    merged_amplitude = merged.amplitude([True, False])
    assert_complex_tuple(merged_amplitude)
    assert merged.is_state_exact() is True

    lazy = exp.StabMps(2, seed=19, measurement="lazy")
    lazy.run_1q_gate("H", 1)
    lazy.run_1q_gate("T", 1)
    lazy.run_1q_gate("S", 0)
    lazy.run_1q_gate("H", 0)
    lazy.run_2q_gate("CX", (0, 1))
    lazy.run_1q_gate("MZ", 0)
    assert lazy.is_state_exact() is False
    assert len(lazy.state_vector()) == 4
    assert lazy.is_state_exact() is False
    assert lazy.uncompensated_pre_reduction_count == 0
    assert lazy.summed_discarded_weight == 0.0
    assert lazy.branch_vanish_retry_count == 0
    assert lazy.deferred_branch_lost_count == 0
    assert isinstance(lazy.lifetime_peak_bond, int)

    with pytest.raises(ValueError, match="bitstring length 1, expected 2"):
        q0_one.amplitude([True])
    with pytest.raises(ValueError, match="bitstring item 0 must be bool"):
        q0_one.prob_bitstring([1, False])
    with pytest.raises(
        ValueError,
        match="prob_bitstrings: query 0: bitstring item 0 must be bool",
    ):
        q0_one.prob_bitstrings([[1, False]])
    with pytest.raises(
        ValueError,
        match="prob_bitstrings: query 1: bitstring length 1, expected 2",
    ):
        q0_one.prob_bitstrings([[True, False], [True]])
    with pytest.raises(
        ValueError,
        match="prob_bitstrings: queries must be an iterable of bitstrings",
    ):
        q0_one.prob_bitstrings(1)
    with pytest.raises(IndexError):
        q0_one.frame_x_bit(2)
    with pytest.raises(IndexError):
        q0_one.frame_x_bit(-1)
    with pytest.raises(IndexError):
        q0_one.pauli_expectation([(2, "Z")])
    with pytest.raises(ValueError, match="Unknown Pauli: A"):
        q0_one.pauli_expectation([(0, "A")])
    with pytest.raises(
        ValueError,
        match=r"probability must be finite and in \[0, 1\]",
    ):
        q0_one.apply_depolarizing(0, math.nan)
    with pytest.raises(
        ValueError,
        match="max_truncation_error must be finite and non-negative",
    ):
        exp.StabMps(1, max_truncation_error=math.nan)
    with pytest.raises(
        ValueError,
        match="max_truncation_error must be finite and non-negative",
    ):
        exp.StabMps(1, max_truncation_error=-1.0)
    with pytest.raises(
        ValueError,
        match="max_truncation_error must be finite and non-negative",
    ):
        exp.stab_mps().max_truncation_error(math.nan)


def test_seeded_reset_continues_and_unseeded_reset_smoke():
    def run_stab_mps(sim):
        sim.run_1q_gate("H", 0)
        sim.run_1q_gate("T", 0)
        return sim.run_1q_gate("MZ", 0)

    stn = exp.StabMps(1, seed=0x5EED, merge_rz=False)
    stn_outcomes = []
    for _ in range(200):
        assert stn.reset() is stn
        assert all(value == 0 for value in stn.stats().values())
        stn_outcomes.append(run_stab_mps(stn))
        assert stn.stats()["total_nonclifford"] > 0
    assert set(stn_outcomes) == {0, 1}

    unseeded_stn = exp.StabMps(1)
    unseeded_stn.run_1q_gate("H", 0)
    unseeded_stn.run_1q_gate("MZ", 0)
    assert unseeded_stn.reset() is unseeded_stn

    def run_mast(sim):
        sim.run_1q_gate("H", 0)
        sim.run_1q_gate("T", 0)
        return sim.run_1q_gate("MZ", 0)

    mast = exp.Mast(1, 2, seed=0x5EED, merge_rz=False)
    mast_outcomes = []
    for _ in range(200):
        assert mast.reset() is mast
        assert mast.projection_records() == []
        assert mast.projection_peak_bond == 0
        assert all(value == 0 for value in mast.stats().values())
        mast_outcomes.append(run_mast(mast))
        assert len(mast.projection_records()) == 1
    assert set(mast_outcomes) == {0, 1}

    unseeded_mast = exp.Mast(1, 1)
    unseeded_mast.run_1q_gate("H", 0)
    unseeded_mast.run_1q_gate("MZ", 0)
    assert unseeded_mast.reset() is unseeded_mast


def test_mast_configuration_projection_diagnostics_and_stats():
    with pytest.raises(TypeError):
        exp.Mast(1, 1, lazy_measure=True)

    mast = exp.Mast(
        1,
        1,
        seed=5,
        max_bond_dim=1,
        max_truncation_error=0.0,
        merge_rz=False,
        numerical_flag_redetection=True,
        projection_order="input",
    )
    mast.run_1q_gate("H", 0)
    mast.run_1q_gate("T", 0)
    assert mast.remaining_injections == 0
    with pytest.raises(IndexError):
        mast.run_1q_gate("H", -1)
    mast.project_all()

    records = mast.projection_records()
    assert isinstance(records, list)
    assert len(records) == 1
    assert set(records[0]) == {
        "ancilla",
        "support_size",
        "mps_span",
        "bond_before",
        "bond_after",
    }
    assert all(isinstance(value, int) for value in records[0].values())
    assert isinstance(mast.projection_peak_bond, int)
    assert isinstance(mast.truncation_error, float)
    assert isinstance(mast.bond_cap_hits, int)

    stats = mast.stats()
    assert isinstance(stats, dict)
    assert set(stats) == STATS_KEYS
    assert all(isinstance(value, int) for value in stats.values())

    with pytest.raises(
        ValueError,
        match="max_truncation_error must be finite and non-negative",
    ):
        exp.Mast(1, 1, max_truncation_error=math.nan)


def test_mast_diagnostic_getters_do_not_materialize_pending_rotations():
    mast = exp.Mast(1, 2, seed=23, merge_rz=True)
    mast.run_1q_gate("H", 0)
    mast.run_1q_gate("T", 0)

    assert mast.num_ancillas_used == 0
    assert mast.remaining_injections == 2
    assert isinstance(mast.max_bond_dim, int)
    assert mast.truncation_error == 0.0
    assert mast.bond_cap_hits == 0
    assert mast.projection_records() == []
    assert mast.projection_peak_bond == 0
    assert mast.stats()["total_nonclifford"] == 0
    assert mast.remaining_injections == 2

    mast.run_1q_gate("T", 0)
    mast.project_all()
    assert mast.num_ancillas_used == 0
    assert mast.remaining_injections == 2
    assert mast.projection_records() == []


def test_stab_mps_compile_dispatch_accessors_and_advice():
    compile_only = exp.StabMpsCompile(2)
    assert compile_only.num_qubits == 2

    assert compile_only.run_1q_gate("H", 0) is None
    assert compile_only.run_2q_gate("CX", (0, 1)) is None
    assert compile_only.run_gate("Z", {1}) == {}
    with pytest.raises(IndexError):
        compile_only.run_1q_gate("H", -1)

    recommendation = compile_only.recommend()
    assert recommendation["simulator"] == "ch_form"
    assert isinstance(recommendation["reason"], str)

    advice = compile_only.advise()
    assert set(advice) == {
        "simulator",
        "injection",
        "injectable_count",
        "deferred_ancillas_required",
        "deferred_feasible",
        "warnings",
        "reason",
    }
    assert advice["injection"] == "direct"
    assert advice["deferred_feasible"] is None
    assert isinstance(advice["warnings"], list)

    compile_only.run_1q_gate("T", 0)
    for name in (
        "absorbed",
        "grown",
        "stabilizer",
        "total_nonclifford",
        "nonclifford_rz_total",
        "injectable_clifford_correction",
        "nullity",
        "rank",
        "bond_dim_bound",
    ):
        assert isinstance(getattr(compile_only, name), int)

    assert compile_only.reset() is compile_only
    assert compile_only.total_nonclifford == 0
