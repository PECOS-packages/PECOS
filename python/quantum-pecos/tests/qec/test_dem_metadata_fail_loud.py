# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Fail-loud regression tests for circuit-ingested DEM metadata.

Out-of-range record offsets / meas_ids, and a declared ``num_measurements``
that disagrees with the circuit, must be rejected on every circuit-ingest
path -- ``DetectorErrorModel.from_circuit``, ``DemSampler.from_circuit``,
and the public ``DemBuilder.build`` -- not silently dropped.
"""

import pytest
from pecos_rslib import DagCircuit
from pecos_rslib.qec import (
    DagFaultAnalyzer,
    DemBuilder,
    DemSampler,
    DemSamplerBuilder,
    DetectorErrorModel,
)
from pecos_rslib.quantum import Gate, GateType


def _one_measurement_dag(*, num_measurements: str = "1") -> DagCircuit:
    """A circuit performing exactly one Z measurement."""
    dag = DagCircuit()
    dag.pz([0])
    dag.mz([0])
    dag.set_attr("num_measurements", num_measurements)
    return dag


_NOISE = {"p1": 0.0, "p2": 0.0, "p_meas": 0.1, "p_prep": 0.0}


# --- positive controls: valid metadata still builds on every path ----------


def test_valid_metadata_builds_on_all_paths() -> None:
    dag = _one_measurement_dag()
    dag.set_attr("detectors", '[{"id": 0, "records": [-1]}]')

    assert DetectorErrorModel.from_circuit(dag, **_NOISE).num_detectors == 1
    assert DemSampler.from_circuit(dag, **_NOISE).num_detectors == 1

    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = DemBuilder(im)
    builder.with_noise(**_NOISE)
    builder.with_num_measurements(1)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    assert builder.build().num_detectors == 1


def test_exact_measurement_crosstalk_payload_emits_python_source_record() -> None:
    dag = DagCircuit()
    prep = dag.add_gate(Gate(GateType.Prep, qubits=[0]))
    payload = dag.add_gate(Gate(GateType.MeasCrosstalkLocalPayload, qubits=[0]))
    meas = dag.add_gate(Gate(GateType.Measure, qubits=[0]))
    dag.connect(prep, payload, 0)
    dag.connect(payload, meas, 0)
    dag.set_attr("num_measurements", "1")
    dag.set_attr("detectors", '[{"id": 0, "records": [-1]}]')

    im = DagFaultAnalyzer(dag).build_influence_map()
    builder = DemBuilder(im)
    builder.with_noise(
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
        p_meas_crosstalk_local=0.25,
        p_meas_crosstalk_model={"0->1": 0.4},
        measurement_crosstalk_dem_mode="exact_deterministic",
    )
    builder.with_num_measurements(1)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    builder.with_exact_branch_replay_circuit(dag)
    dem = builder.build_with_source_tracking()

    records = dem.contribution_render_records()
    assert len(records) == 1
    assert records[0]["direct_source_family"] == "MeasurementCrosstalk"
    assert records[0]["gate_type_labels"] == ["MeasCrosstalkLocalPayload"]


def test_exact_branch_replay_circuit_rejects_unsupported_gate() -> None:
    """The independently supplied Python replay circuit is validated too."""
    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()

    replay = DagCircuit()
    replay.pz([0])
    replay.t([0])
    replay.mz([0])

    builder = DemBuilder(im)
    builder.with_noise(**_NOISE)
    builder.with_exact_branch_replay_circuit(replay)
    with pytest.raises(ValueError, match=r"unsupported gate T at DAG node 1"):
        builder.build()


# --- out-of-range record offsets -------------------------------------------


def test_from_circuit_out_of_range_record_fails_loud() -> None:
    dag = _one_measurement_dag()
    dag.set_attr("detectors", '[{"id": 0, "records": [-2]}]')
    with pytest.raises(ValueError, match=r"out of range|record offset"):
        DetectorErrorModel.from_circuit(dag, **_NOISE)


def test_dem_sampler_out_of_range_record_fails_loud() -> None:
    dag = _one_measurement_dag()
    dag.set_attr("detectors", '[{"id": 0, "records": [-2]}]')
    with pytest.raises(ValueError, match=r"out of range|record offset"):
        DemSampler.from_circuit(dag, **_NOISE)


def test_public_dem_builder_out_of_range_record_fails_loud() -> None:
    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = DemBuilder(im)
    builder.with_noise(**_NOISE)
    builder.with_num_measurements(1)
    builder.with_detectors_json('[{"id": 0, "records": [-2]}]')
    with pytest.raises(ValueError, match=r"out of range|record offset"):
        builder.build()


# --- out-of-range meas_ids -------------------------------------------------


def test_from_circuit_out_of_range_meas_id_fails_loud() -> None:
    dag = _one_measurement_dag()
    dag.set_attr("detectors", '[{"id": 0, "meas_ids": [999]}]')
    with pytest.raises(ValueError, match="meas_id"):
        DetectorErrorModel.from_circuit(dag, **_NOISE)


def test_dem_sampler_out_of_range_meas_id_fails_loud() -> None:
    dag = _one_measurement_dag()
    dag.set_attr("detectors", '[{"id": 0, "meas_ids": [999]}]')
    with pytest.raises(ValueError, match="meas_id"):
        DemSampler.from_circuit(dag, **_NOISE)


def test_public_dem_builder_out_of_range_meas_id_fails_loud() -> None:
    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = DemBuilder(im)
    builder.with_noise(**_NOISE)
    builder.with_num_measurements(1)
    builder.with_detectors_json('[{"id": 0, "meas_ids": [999]}]')
    with pytest.raises(ValueError, match="meas_id"):
        builder.build()


# --- bogus declared num_measurements ---------------------------------------


def test_from_circuit_inconsistent_num_measurements_fails_loud() -> None:
    """Declaring 2 measurements on a 1-measurement circuit must be rejected;
    otherwise a record offset of -2 would falsely validate and misbind."""
    dag = _one_measurement_dag(num_measurements="2")
    dag.set_attr("detectors", '[{"id": 0, "records": [-2]}]')
    with pytest.raises(ValueError, match="num_measurements"):
        DetectorErrorModel.from_circuit(dag, **_NOISE)


def test_dem_sampler_inconsistent_num_measurements_fails_loud() -> None:
    dag = _one_measurement_dag(num_measurements="2")
    dag.set_attr("detectors", '[{"id": 0, "records": [-2]}]')
    with pytest.raises(ValueError, match="num_measurements"):
        DemSampler.from_circuit(dag, **_NOISE)


def test_public_dem_builder_inconsistent_num_measurements_fails_loud() -> None:
    """Public builder with a real (non-empty) influence map must reject a
    with_num_measurements() that disagrees with the circuit; otherwise an
    out-of-range record (e.g. -2 against 1 measurement) silently misbinds."""
    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = DemBuilder(im)
    builder.with_noise(**_NOISE)
    builder.with_num_measurements(2)  # circuit performs only 1 measurement
    builder.with_detectors_json('[{"id": 0, "records": [-2]}]')
    with pytest.raises(ValueError, match="num_measurements"):
        builder.build()


def test_public_dem_builder_consistent_num_measurements_still_builds() -> None:
    """The matching-count case (and the empty-influence-map escape hatch)
    must keep working -- the count check only fires on a genuine mismatch."""
    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = DemBuilder(im)
    builder.with_noise(**_NOISE)
    builder.with_num_measurements(1)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    assert builder.build().num_detectors == 1


# --- DemSamplerBuilder JSON path (M-E): context-aware fail-loud -------------
# The public sampler builder previously parsed detector/observable JSON with a
# hand-rolled string scanner that silently dropped out-of-range refs. It now
# resolves refs against the circuit's measurement count, like DemBuilder.


def test_dem_sampler_builder_out_of_range_record_fails_loud() -> None:
    from pecos_rslib.qec import DemSamplerBuilder

    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_detectors_json(
            '[{"id": 0, "records": [-1, -2]}]',  # -2 out of range for 1 measurement
        )
    )
    with pytest.raises(ValueError, match=r"out of range"):
        builder.build()


def test_dem_sampler_builder_out_of_range_observable_fails_loud() -> None:
    from pecos_rslib.qec import DemSamplerBuilder

    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_observables_json(
            '[{"id": 0, "records": [-1, -2]}]',
        )
    )
    with pytest.raises(ValueError, match=r"out of range"):
        builder.build()


def test_dem_sampler_builder_out_of_range_meas_id_fails_loud() -> None:
    from pecos_rslib.qec import DemSamplerBuilder

    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    builder = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_detectors_json(
            '[{"id": 0, "meas_ids": [0, 999]}]',  # 999 absent / out of range
        )
    )
    with pytest.raises(ValueError, match=r"not present|out of range"):
        builder.build()


def test_dem_sampler_builder_valid_metadata_still_builds() -> None:
    """Positive control: an in-range record still builds."""
    from pecos_rslib.qec import DemSamplerBuilder

    im = DagFaultAnalyzer(_one_measurement_dag()).build_influence_map()
    sampler = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_detectors_json(
            '[{"id": 0, "records": [-1]}]',
        )
        .build()
    )
    assert sampler is not None


def test_dem_sampler_builder_resolves_stamped_meas_ids() -> None:
    """meas_ids are stamped MeasIds resolved via the influence map (matching
    DemBuilder), not positional indices. A stamped id present in the circuit
    resolves; a value absent from the stamped set fails loud. Previously the
    sampler treated meas_ids positionally, so a stamped id raised 'out of range'
    and an absent id silently misbound."""
    from pecos_rslib.qec import DemSamplerBuilder
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    tc.tick().mz_with_ids([0, 1], [10, 5])  # non-positional stamped ids
    im = DagFaultAnalyzer(tc.to_dag_circuit()).build_influence_map()

    # Stamped id 10 is present -> resolves and builds.
    DemSamplerBuilder(im).with_noise(**_NOISE).with_detectors_json(
        '[{"id": 0, "meas_ids": [10]}]',
    ).build()

    # Stamped id 0 is absent -> fail loud (positional would have accepted index 0).
    builder = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_detectors_json(
            '[{"id": 0, "meas_ids": [0]}]',
        )
    )
    with pytest.raises(ValueError, match=r"not present|out of range"):
        builder.build()


def test_dem_sampler_builder_rejects_inconsistent_measurement_order() -> None:
    """A measurement_order must cover every measurement; a shorter order would
    let validated record offsets resolve in a different frame and silently
    misbind (the count-frame hole)."""
    from pecos_rslib.qec import DemSamplerBuilder

    dag = DagCircuit()
    for q in range(3):
        dag.pz([q])
        dag.mz([q])
    dag.set_attr("num_measurements", "3")
    im = DagFaultAnalyzer(dag).build_influence_map()

    builder = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_detectors_json('[{"id": 0, "records": [-3]}]')
        .with_measurement_order([0, 1])  # only 2 of 3 measurements
    )
    with pytest.raises(ValueError, match=r"measurement_order|cover every measurement"):
        builder.build()


def test_mz_with_ids_rejects_a_repeated_id_in_one_call() -> None:
    """Duplicate stable MeasIds make stamped-id resolution ambiguous (it binds
    to the first occurrence). A repeat within one call is caught where the ids
    are supplied, so it never reaches the circuit."""
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    with pytest.raises(ValueError, match=r"repeats MeasId\(7\)"):
        tc.tick().mz_with_ids([0, 1], [7, 7])


def test_measuring_nothing_reserves_no_record() -> None:
    """An empty measurement produces no results, so it must not consume a record.
    Treating the empty id list as id 0 reserved one for a measurement that does
    not exist."""
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().mz_with_ids([], [])
    assert tc.num_measurements() == 0


def test_a_rejected_measurement_does_not_consume_records() -> None:
    """`mz` can fail on a qubit conflict. A caller that catches that must not be
    left with records consumed by a measurement the circuit never got, or every
    later measurement is misnumbered."""
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0])
    tick = tc.tick()
    with pytest.raises(ValueError, match=r"distinct qubits"):
        tick.mz([0, 0])
    assert tc.num_measurements() == 0

    tick.mz([0])
    assert tc.num_measurements() == 1


def test_mz_with_ids_rejects_an_id_with_no_room_for_a_successor() -> None:
    """The largest representable id leaves the record counter nowhere to go. It
    is refused as a ValueError rather than overflowing."""
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0])
    with pytest.raises(ValueError, match=r"no room for a later id"):
        tc.tick().mz_with_ids([0], [2**64 - 1])


def test_a_high_supplied_id_does_not_let_a_later_measurement_overflow() -> None:
    """A legal id just below the ceiling pushes the record counter onto it, so the
    next measurement has nowhere to go. It is refused as a ValueError rather than
    overflowing the counter into an uncatchable panic.

    The last representable id is never handed out: the counter must always have a
    valid successor.
    """
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    tc.tick().mz_with_ids([0], [2**64 - 2])
    with pytest.raises(ValueError, match=r"remain below usize::MAX"):
        tc.tick().mz([1])


def test_dag_circuit_measurement_reports_exhaustion_as_a_value_error() -> None:
    """`DagCircuit.mz` mints ids, so it can run out. That has to arrive as a
    catchable ValueError, not a panic Python cannot handle as an exception."""
    from pecos_rslib.quantum import DagCircuit, TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    tc.tick().mz_with_ids([0], [2**64 - 2])
    dag = tc.to_dag_circuit()
    assert isinstance(dag, DagCircuit)

    with pytest.raises(ValueError, match=r"remain below usize::MAX"):
        dag.mz([1])


def test_an_idless_and_a_stamped_measurement_collide_loudly() -> None:
    """The generic ``add_gate`` reserves no measurement record, so its MZ is
    id-less until conversion mints one -- which then collides with a measurement
    that already holds that id.

    A pre-conversion scan cannot catch this: it sees only *supplied* ids, and the
    colliding id does not exist until the conversion runs. Only the conversion
    itself can report it.
    """
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().add_gate("MZ", [0])
    tc.tick().mz([1])

    with pytest.raises(ValueError, match=r"reuses MeasId"):
        tc.to_dag_circuit()


def test_duplicate_ids_across_calls_fail_at_dag_conversion() -> None:
    """A duplicate spread across two calls is invisible to either call, so it is
    caught when the circuit is converted."""
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    tc.tick().mz_with_ids([0], [7])
    tc.tick().mz_with_ids([1], [7])
    with pytest.raises(ValueError, match=r"reuses MeasId\(7\)"):
        tc.to_dag_circuit()


@pytest.mark.parametrize("kind", ["detectors", "observables"])
@pytest.mark.parametrize("record", [-3, 2, 5])
def test_map_only_record_bounds(kind: str, record: int) -> None:
    """Two measurements admit positions 0 and 1, never N or N+3."""
    dag = DagCircuit()
    dag.pz([0, 1])
    dag.mz([0, 1])
    im = DagFaultAnalyzer(dag).build_influence_map()
    for builder_type in (DemBuilder, DemSamplerBuilder):
        builder = builder_type(im)
        getattr(builder, f"with_{kind}_json")(f'[{{"id":0,"records":[{record}]}}]')
        with pytest.raises(ValueError, match=rf"record offset {record}.*out of range"):
            builder.build()


@pytest.mark.parametrize("kind", ["detectors", "observables"])
@pytest.mark.parametrize(
    ("references", "accepted"),
    [
        ('"records":[0]', True),
        ('"records":[-2]', True),
        ('"meas_ids":[17]', True),
        ('"records":[0],"meas_ids":[17]', True),
        ('"records":[0,1,0],"meas_ids":[17,17,9]', True),
        ('"records":[0],"meas_ids":[9]', False),
        ('"records":[0,0,1],"meas_ids":[17,9,9]', False),
    ],
)
def test_map_only_stamped_reference_positions(kind: str, references: str, *, accepted: bool) -> None:
    """Map position 0 has stamp 17; redundant forms compare multisets."""
    from pecos.quantum import TickCircuit

    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().mz_with_ids([0, 1], [17, 9])
    im = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    builder = DemBuilder(im)
    getattr(builder, f"with_{kind}_json")(f'[{{"id":0,{references}}}]')
    if accepted:
        builder.build()
    else:
        with pytest.raises(
            ValueError,
            match="has both 'records' and 'meas_ids' but they reference different measurements",
        ):
            builder.build()


def _two_measurement_map():
    dag = DagCircuit()
    dag.pz([0, 1])
    dag.x([0])  # p1 noise only affects measurement 0.
    dag.mz([0, 1])
    return DagFaultAnalyzer(dag).build_influence_map()


@pytest.mark.parametrize("kind", ["detectors", "observables"])
@pytest.mark.parametrize(
    ("ids", "message"),
    [([0, 0], "duplicate.*id 0"), ([0, 2], r"ids \[0, 2\]"), ([1], r"ids \[1\]")],
)
def test_sampler_json_requires_dense_unique_ids(kind: str, ids: list[int], message: str) -> None:
    import json

    builder = DemSamplerBuilder(_two_measurement_map())
    getattr(builder, f"with_{kind}_json")(json.dumps([{"id": idx, "records": [0]} for idx in ids]))
    with pytest.raises(ValueError, match=message):
        builder.build()


def test_sampler_json_output_indices_are_declared_ids() -> None:
    definitions = '[{"id":1,"records":[0]},{"id":0,"records":[1]}]'
    sampler = (
        DemSamplerBuilder(_two_measurement_map())
        .with_noise(p1=0.3, p2=0.0, p_meas=0.0, p_prep=0.0)
        .with_detectors_json(definitions)
        .with_observables_json(definitions)
        .build()
    )
    text = sampler.to_detector_error_model().to_string()
    assert "D1 L1" in text
    assert "D0" not in text
    assert "L0" not in text
    empty = DemSamplerBuilder(_two_measurement_map()).with_detectors_json("[]").with_observables_json("[]").build()
    assert empty.num_detectors == 0


@pytest.mark.parametrize("order", [[0], [0, 0]])
def test_map_only_incomplete_measurement_order_is_rejected(order: list[int]) -> None:
    for builder_type in (DemBuilder, DemSamplerBuilder):
        builder = (
            builder_type(_two_measurement_map())
            .with_measurement_order(order)
            .with_detectors_json('[{"id":0,"records":[-1]}]')
        )
        with pytest.raises(ValueError, match=r"1 of 2 measurement\(s\) uncovered"):
            builder.build()


@pytest.mark.parametrize("references", ['"meas_ids":[0]', '"records":[1],"meas_ids":[0]'])
def test_sampler_json_resolves_stamps_after_measurement_order(references: str) -> None:
    definitions = f'[{{"id":0,{references}}}]'
    sampler = (
        DemSamplerBuilder(_two_measurement_map())
        .with_noise(p1=0.3, p2=0.0, p_meas=0.0, p_prep=0.0)
        .with_detectors_json(definitions)
        .with_observables_json(definitions)
        .with_measurement_order([1, 0])
        .build()
    )
    assert "D0 L0" in sampler.to_detector_error_model().to_string()
