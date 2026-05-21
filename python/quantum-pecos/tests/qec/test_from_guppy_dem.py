# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regression tests for the Guppy-to-DEM convenience path."""

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import h, measure, qubit, x
from pecos.guppy import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import (
    _build_surface_tick_circuit_for_native_model,
    _replay_lowered_qis_trace_into_tick_circuit,
    _replay_qis_trace_into_tick_circuit,
)


@guppy
def _single_measurement() -> None:
    q = qubit()
    b = measure(q)
    result("m", b)


@guppy
def _measurement_feedback() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    b0 = measure(q0)
    if b0:
        x(q1)
    b1 = measure(q1)
    result("b0", b0)
    result("b1", b1)


def _dem_text(*, detectors_json: str = "[]", observables_json: str = "[]") -> str:
    dem = DetectorErrorModel.from_guppy(
        _single_measurement,
        num_qubits=1,
        detectors_json=detectors_json,
        observables_json=observables_json,
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
        seed=0,
    )
    return dem.to_string()


def _flat_mz_ids(tc) -> list[int]:
    dag = tc.to_dag_circuit()
    ids: list[int] = []
    for node_id in dag.nodes():
        gate = dag.gate(node_id)
        if gate is not None and gate.gate_type.name == "MZ":
            ids.extend(int(mid) for mid in gate.meas_ids)
    return ids


def test_from_guppy_meas_ids_are_normalized_to_records() -> None:
    assert _dem_text(detectors_json='[{"id":0,"meas_ids":[0]}]') == _dem_text(
        detectors_json='[{"id":0,"records":[-1]}]',
    )

    assert _dem_text(observables_json='[{"id":0,"meas_ids":[0]}]') == _dem_text(
        observables_json='[{"id":0,"records":[-1]}]',
    )


@pytest.mark.parametrize(
    "detectors_json",
    [
        "{}",
        '[{"id":0,"records":["-1"]}]',
        '[{"id":0,"records":[-1.2]}]',
        '[{"id":0,"meas_ids":["0"]}]',
    ],
)
def test_from_guppy_rejects_malformed_detector_metadata(detectors_json: str) -> None:
    with pytest.raises(ValueError, match=r"JSON list|integer|record offset|meas_id"):
        _dem_text(detectors_json=detectors_json)


def test_from_guppy_rejects_json_tracked_pauli_observables() -> None:
    with pytest.raises(ValueError, match="tracked_pauli"):
        _dem_text(observables_json='[{"kind":"tracked_pauli","label":"x","pauli":"X0"}]')


def test_from_guppy_dynamic_control_is_unsupported_and_unguarded() -> None:
    """Measurement-dependent control flow is unsupported/undefined.

    A prior runtime-trace guard false-positived on the standard surface code
    (statically-scheduled post-measurement gates look the same in the trace),
    so it was reverted. This test pins that NO guard rejects programs here --
    from_guppy must not raise on either a dynamic program or, by extension,
    the surface code. The DEM for a dynamic program is undefined/seed-dependent
    and callers must not rely on it (see from_guppy docstring / proposal 001).
    """
    for s in (0, 2, 5):
        dem = DetectorErrorModel.from_guppy(
            _measurement_feedback,
            num_qubits=2,
            detectors_json='[{"id":0,"records":[-2,-1]}]',
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
            seed=s,
        )
        assert dem.num_detectors == 1  # builds (undefined content; do not rely)


def test_lowered_replay_uses_measure_result_ids_directly() -> None:
    chunks = [
        {
            "operations": [
                {"AllocateResult": {"id": 42}},
                {"AllocateResult": {"id": 99}},
                {"Quantum": {"Measure": [0, 99]}},
                {"Quantum": {"Measure": [1, 42]}},
            ],
            "lowered_quantum_ops": [
                {"gate_type": "MZ", "qubits": [0], "angles": []},
                {"gate_type": "MZ", "qubits": [1], "angles": []},
            ],
        },
    ]

    tc = _replay_lowered_qis_trace_into_tick_circuit(chunks)

    assert _flat_mz_ids(tc) == [99, 42]


def test_lowered_replay_fails_on_measurement_count_mismatch() -> None:
    chunks = [
        {
            "operations": [{"Quantum": {"Measure": [0, 7]}}],
            "lowered_quantum_ops": [{"gate_type": "MZ", "qubits": [0, 1], "angles": []}],
        },
    ]

    with pytest.raises(ValueError, match="More measured qubits"):
        _replay_lowered_qis_trace_into_tick_circuit(chunks)


def test_non_lowered_replay_preserves_non_sequential_result_ids() -> None:
    operations = [
        {"AllocateQubit": {"id": 10}},
        {"AllocateQubit": {"id": 20}},
        {"Quantum": {"Measure": [10, 77]}},
        {"Quantum": {"Measure": [20, 3]}},
    ]

    tc = _replay_qis_trace_into_tick_circuit(operations)

    assert _flat_mz_ids(tc) == [77, 3]


def test_from_guppy_surface_code_is_byte_identical_to_reference() -> None:
    """Regression: from_guppy(make_surface_code(...)) must work and match the
    traced_qis reference DEM. A reverted dynamic-control guard had broken this
    exact path (it false-positived on surface's post-measurement gates)."""
    p = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    for basis in ("Z", "X"):
        patch = SurfacePatch.create(distance=3)
        ref = _build_surface_tick_circuit_for_native_model(
            patch,
            3,
            basis,
            circuit_source="traced_qis",
        )
        ref.lower_clifford_rotations()
        ref.assign_missing_meas_ids()
        ref_dem = DetectorErrorModel.from_circuit(ref, **p).to_string()
        got = DetectorErrorModel.from_guppy(
            make_surface_code(distance=3, num_rounds=3, basis=basis),
            num_qubits=get_num_qubits(3),
            detectors_json=ref.get_meta("detectors"),
            observables_json=ref.get_meta("observables"),
            num_measurements=int(ref.get_meta("num_measurements")),
            **p,
        ).to_string()
        assert got == ref_dem, f"surface from_guppy not byte-identical ({basis})"


def test_from_guppy_out_of_range_record_fails_loud() -> None:
    with pytest.raises(ValueError, match=r"out of range|record offset"):
        _dem_text(detectors_json='[{"id":0,"records":[-2]}]')  # only 1 measurement


def test_from_guppy_out_of_range_meas_id_fails_loud() -> None:
    with pytest.raises(ValueError, match=r"meas_id|not present"):
        _dem_text(detectors_json='[{"id":0,"meas_ids":[999]}]')


def test_from_guppy_accepts_dem_label_id_forms() -> None:
    """The "D0"/"L0" id convenience form is now normalized in the Rust
    builder (single source of truth), equivalent to the bare integer."""
    assert _dem_text(detectors_json='[{"id":"D0","records":[-1]}]') == _dem_text(
        detectors_json='[{"id":0,"records":[-1]}]',
    )
    assert _dem_text(observables_json='[{"id":"L0","records":[-1]}]') == _dem_text(
        observables_json='[{"id":0,"records":[-1]}]',
    )


def test_from_guppy_rejects_bad_string_id() -> None:
    with pytest.raises(ValueError, match=r"not a valid identifier"):
        _dem_text(detectors_json='[{"id":"X0","records":[-1]}]')


def test_from_guppy_rejects_detector_tracked_pauli() -> None:
    with pytest.raises(ValueError, match="tracked_pauli"):
        _dem_text(detectors_json='[{"kind":"tracked_pauli","label":"x","pauli":"X0"}]')


def test_from_guppy_rejects_entry_without_records_or_meas_ids() -> None:
    with pytest.raises(ValueError, match=r"records|meas_ids|neither"):
        _dem_text(detectors_json='[{"id":0}]')


def test_from_guppy_redundant_records_and_meas_ids_are_accepted() -> None:
    """Co-present records + meas_ids that name the SAME measurement are
    tolerated (the surface logical_circuit path emits both redundantly) and
    produce the same DEM as either form alone. (Non-redundant co-presence is
    rejected fail-loud; that precise semantics is pinned by the deterministic
    Rust unit test ``test_try_build_mixed_records_meas_ids_must_be_redundant``,
    since stamped MeasId values are not predictable from Python here.)"""
    both = _dem_text(detectors_json='[{"id":0,"records":[-1],"meas_ids":[0]}]')
    assert both == _dem_text(detectors_json='[{"id":0,"records":[-1]}]')


# ---------------------------------------------------------------------------
# Constrained-ancilla surface support
# ---------------------------------------------------------------------------


def _constrained_surface_via_guppy(*, d, basis, rounds, budget, noise):
    """Build the constrained-surface DEM through `from_guppy`."""
    patch = SurfacePatch.create(distance=d)
    ref = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=rounds,
        basis=basis,
        ancilla_budget=budget,
        circuit_source="traced_qis",
    )
    ref.lower_clifford_rotations()
    ref.assign_missing_meas_ids()
    ref_dem = DetectorErrorModel.from_circuit(ref, **noise).to_string()

    got = DetectorErrorModel.from_guppy(
        make_surface_code(distance=d, num_rounds=rounds, basis=basis, ancilla_budget=budget),
        num_qubits=get_num_qubits(d, ancilla_budget=budget),
        detectors_json=ref.get_meta("detectors"),
        observables_json=ref.get_meta("observables"),
        num_measurements=int(ref.get_meta("num_measurements")),
        **noise,
    ).to_string()
    return ref_dem, got, ref


def test_from_guppy_constrained_surface_dem_byte_identical() -> None:
    """`from_guppy(make_surface_code(..., ancilla_budget=b))` must produce a
    DEM byte-identical to the reference DEM built through the
    `_build_surface_tick_circuit_for_native_model(circuit_source="traced_qis",
    ancilla_budget=b)` path. Covers (d=3, budget=1, Z), (d=3, budget=2, X), and
    (d=9, budget=17, Z) -- small-and-fast + asymmetric basis + canonical
    high-distance stress."""
    noise = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    for d, basis, rounds, budget in [(3, "Z", 2, 1), (3, "X", 2, 2), (9, "Z", 3, 17)]:
        ref_dem, got, _ = _constrained_surface_via_guppy(
            d=d,
            basis=basis,
            rounds=rounds,
            budget=budget,
            noise=noise,
        )
        assert got == ref_dem, (
            f"constrained surface from_guppy not byte-identical for "
            f"d={d}, budget={budget}, basis={basis}, rounds={rounds}"
        )


def test_constrained_surface_traced_metadata_matches_abstract() -> None:
    """The traced TickCircuit's surface metadata is copied verbatim from the
    abstract reference. Specifically pins that
    ``_copy_surface_tick_circuit_metadata`` propagates ``ancilla_budget``
    (the new key added when the constrained codegen landed) alongside the
    existing detectors/observables/counts."""
    patch = SurfacePatch.create(distance=3)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
    )
    traced_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="traced_qis",
    )
    for key in (
        "basis",
        "detectors",
        "observables",
        "num_measurements",
        "num_detectors",
        "ancilla_budget",
    ):
        a = abstract_tc.get_meta(key)
        b = traced_tc.get_meta(key)
        assert a == b, f"metadata mismatch on key {key!r}: abstract={a!r}, traced={b!r}"
    # ancilla_budget specifically must be the requested budget (stored as a string by set_meta).
    assert traced_tc.get_meta("ancilla_budget") == "2"


def test_constrained_surface_lowered_qubit_stream_within_budget() -> None:
    """The lowered-trace physical qubit IDs must stay within the budgeted
    pool, and ancilla slots must be empirically reused (more measurements
    than physical ancilla qubits). Pins the load-bearing assumption the
    spike validated."""
    import pecos

    d, budget = 3, 2
    program = make_surface_code(distance=d, num_rounds=2, basis="Z", ancilla_budget=budget)
    n_q = get_num_qubits(d, ancilla_budget=budget)
    chunks = list(
        pecos.sim(program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(n_q)
        .seed(0)
        .capture_operation_trace(),
    )

    all_qubits: set[int] = set()
    mz_qubits: list[int] = []
    for chunk in chunks:
        for gate in chunk.get("lowered_quantum_ops") or []:
            qs = [int(q) for q in gate.get("qubits", [])]
            all_qubits.update(qs)
            if str(gate.get("gate_type")) == "MZ":
                mz_qubits.extend(qs)

    max_q = max(all_qubits) if all_qubits else -1
    # Budget enforcement: total physical qubits used must fit in d^2 + budget.
    over_budget_msg = f"max physical qubit id {max_q} exceeds budgeted pool size {n_q}"
    assert max_q < n_q, over_budget_msg
    # Reuse demonstrated: some physical qubit appears in multiple MZ ops.
    reuse = any(mz_qubits.count(q) > 1 for q in set(mz_qubits))
    assert reuse, "no physical qubit appears in more than one MZ op"


def test_constrained_from_guppy_dem_is_consumable_by_pecos_native_decoder() -> None:
    """PECOS-native decoder smoke for the constrained-ancilla DEM: the DEM
    returned by ``from_guppy(...)`` must be consumable by both the PECOS
    sampler (``dem.to_sampler()``) and the PECOS Rust-backed
    ``PyMatchingDecoder.from_dem(...)`` -- the actual downstream surfaces
    callers use, not an external ``pymatching`` install.

    Also asserts ``stim.DetectorErrorModel(dem.to_string_decomposed())``
    parses as a lightweight syntax-compatibility smoke (optional reference,
    not the correctness oracle).
    """
    from pecos_rslib.decoders import PyMatchingDecoder

    p = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    patch = SurfacePatch.create(distance=3)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
    )
    dem = DetectorErrorModel.from_guppy(
        make_surface_code(distance=3, num_rounds=2, basis="Z", ancilla_budget=2),
        num_qubits=get_num_qubits(3, ancilla_budget=2),
        detectors_json=abstract_tc.get_meta("detectors"),
        observables_json=abstract_tc.get_meta("observables"),
        num_measurements=int(abstract_tc.get_meta("num_measurements")),
        **p,
    )

    # PECOS-native sampler path: DEM is well-formed for sampling.
    sampler = dem.to_sampler()
    assert sampler.num_dem_outputs >= 0
    assert sampler.num_observables >= 0

    # PECOS-native Rust-backed matching decoder: DEM is consumable by
    # the actual downstream decoder surface.
    decomp = dem.to_string_decomposed()
    decoder = PyMatchingDecoder.from_dem(decomp)
    assert decoder is not None

    # Lightweight format-compatibility smoke (optional reference coverage,
    # not the correctness oracle). Stim should parse the decomposed DEM.
    import stim

    parsed = stim.DetectorErrorModel(decomp)
    assert parsed.num_detectors >= 0


def test_constrained_from_guppy_fails_loud_on_mismatched_num_measurements() -> None:
    """The constrained-ancilla surface program must flow through the same
    Rust metadata-validation fail-loud path as any other Guppy program.
    No surface-specific bypass: passing a wrong ``num_measurements`` (here,
    the unconstrained count, which differs from the constrained traced
    program's count) is rejected by the generic builder, not by anything
    surface-aware in ``from_guppy``."""
    p = {"p1": 0.005, "p2": 0.005, "p_meas": 0.005, "p_prep": 0.005}
    patch = SurfacePatch.create(distance=3)
    abstract_tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds=2,
        basis="Z",
        ancilla_budget=2,
        circuit_source="abstract",
    )
    actual = int(abstract_tc.get_meta("num_measurements"))
    wrong = actual + 1

    with pytest.raises(ValueError, match=r"num_measurements"):
        DetectorErrorModel.from_guppy(
            make_surface_code(distance=3, num_rounds=2, basis="Z", ancilla_budget=2),
            num_qubits=get_num_qubits(3, ancilla_budget=2),
            detectors_json=abstract_tc.get_meta("detectors"),
            observables_json=abstract_tc.get_meta("observables"),
            num_measurements=wrong,
            **p,
        )
