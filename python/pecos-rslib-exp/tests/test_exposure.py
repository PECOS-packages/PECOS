"""Smoke tests for the approved experimental Python exposure set.

REQUIRES a built ``pecos_rslib_exp`` extension. The downstream harness venv
builds the extension before running this file; this test is not wired into Cargo.
"""

import math

import pecos_rslib_exp as exp


STATS_KEYS = {
    "total_nonclifford",
    "single_site",
    "multi_disent",
    "numerical_redetect",
    "multi_std",
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


def test_stab_mps_analysis_and_noise_exposure():
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
        phase_flip.pauli_expectation([(0, "X")]), -1.0, abs_tol=1e-12
    )


def test_mast_configuration_projection_diagnostics_and_stats():
    mast = exp.Mast(
        1,
        1,
        seed=5,
        lazy_measure=False,
        merge_rz=False,
        numerical_flag_redetection=True,
        projection_order="input",
    )
    mast.run_1q_gate("H", 0)
    mast.run_1q_gate("T", 0)
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

    stats = mast.stats()
    assert isinstance(stats, dict)
    assert set(stats) == STATS_KEYS
    assert all(isinstance(value, int) for value in stats.values())


def test_stab_mps_compile_dispatch_accessors_and_advice():
    compile_only = exp.StabMpsCompile(2)
    assert compile_only.num_qubits == 2

    assert compile_only.run_1q_gate("H", 0) is None
    assert compile_only.run_2q_gate("CX", (0, 1)) is None
    assert compile_only.run_gate("Z", {1}) == {}

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
