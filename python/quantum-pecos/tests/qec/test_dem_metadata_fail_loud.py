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
    DetectorErrorModel,
)


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


def test_dem_sampler_builder_rejects_duplicate_stamped_meas_ids() -> None:
    """Duplicate stable MeasIds make stamped-id resolution ambiguous (bind to
    the first occurrence). DemBuilder rejects them; the sampler JSON path must
    too, rather than silently binding."""
    from pecos_rslib.qec import DemSamplerBuilder
    from pecos_rslib.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().pz([0, 1])
    tc.tick().mz_with_ids([0, 1], [7, 7])  # duplicate stamped id 7
    im = DagFaultAnalyzer(tc.to_dag_circuit()).build_influence_map()

    builder = (
        DemSamplerBuilder(im)
        .with_noise(**_NOISE)
        .with_detectors_json(
            '[{"id": 0, "meas_ids": [7]}]',
        )
    )
    with pytest.raises(ValueError, match=r"duplicate stable MeasId"):
        builder.build()
