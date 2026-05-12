# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for DemSampler Python bindings.

These tests verify that the Rust DemSampler is correctly exposed to Python.
The bulk of algorithmic testing is done in Rust (crates/pecos-qec/tests/dem_sampler_tests.rs).
"""

import pytest


def test_dem_sampler_builder_basic() -> None:
    """Test basic DemSamplerBuilder usage."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    # Build a simple parity check circuit
    dag = DagCircuit()
    dag.pz([2])  # Ancilla
    dag.cx([(0, 2)])
    dag.cx([(1, 2)])
    dag.mz([2])

    # Build influence map
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    # Build sampler
    detectors_json = '[{"id": 0, "records": [-1]}]'

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.01, 0.01, 0.01, 0.01)
    builder.with_detectors_json(detectors_json)

    sampler = builder.build()

    assert sampler.num_detectors == 1
    assert sampler.num_observables == 0
    assert sampler.num_mechanisms > 0


def test_dem_sampler_sampling() -> None:
    """Test DemSampler sampling methods."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    dag = DagCircuit()
    dag.pz([2])
    dag.cx([(0, 2)])
    dag.cx([(1, 2)])
    dag.mz([2])

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.1, 0.1, 0.1, 0.1)  # Higher noise for visible effects
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    builder.with_observables_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

    assert sampler.num_dem_outputs == 1
    assert sampler.num_observables == 1
    assert sampler.num_tracked_paulis == 0

    # Single sample
    det_events, obs_flips = sampler.sample(seed=42)
    assert isinstance(det_events, list)
    assert isinstance(obs_flips, list)
    assert len(det_events) == 1
    assert len(obs_flips) == 1

    # Batch sample
    det_batch, obs_batch = sampler.sample_batch(100, seed=42)
    assert len(det_batch) == 100
    assert len(obs_batch) == 100


def test_dem_sampler_determinism() -> None:
    """Test that sampling is deterministic with same seed."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    dag = DagCircuit()
    dag.pz([2])
    dag.cx([(0, 2)])
    dag.cx([(1, 2)])
    dag.mz([2])

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.1, 0.1, 0.1, 0.1)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

    # Same seed should produce same results
    det1, obs1 = sampler.sample_batch(50, seed=12345)
    det2, obs2 = sampler.sample_batch(50, seed=12345)

    assert det1 == det2
    assert obs1 == obs2


def test_dem_sampler_statistics() -> None:
    """Test DemSampler.sample_statistics method."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    dag = DagCircuit()
    dag.pz([2])
    dag.cx([(0, 2)])
    dag.cx([(1, 2)])
    dag.mz([2])

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.01, 0.01, 0.01, 0.01)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    builder.with_observables_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

    stats = sampler.sample_statistics(10000, seed=42)

    assert "total_shots" in stats
    assert "logical_error_count" in stats
    assert "syndrome_count" in stats
    assert "undetectable_count" in stats
    assert "logical_error_rate" in stats
    assert "syndrome_rate" in stats
    assert "undetectable_rate" in stats
    assert "per_dem_output" in stats
    assert "dem_output_rates" in stats
    assert "observable_error_count" not in stats
    assert "observable_error_rate" not in stats
    assert "per_tracked_op" not in stats
    assert "tracked_op_statistics_supported" not in stats
    assert stats["per_dem_output"] == stats["per_observable"]
    assert stats["dem_output_rates"] == stats["logical_rates"]
    assert stats["tracked_pauli_statistics_supported"] is True
    assert "tracked_pauli_statistics_error" not in stats
    assert sampler.sample_tracked_paulis(seed=42) == []
    assert sampler.sample_tracked_pauli_batch(2, seed=42) == [[], []]

    assert stats["total_shots"] == 10000
    assert 0.0 <= stats["logical_error_rate"] <= 1.0
    assert 0.0 <= stats["syndrome_rate"] <= 1.0


def test_dem_sampler_tracked_pauli_labels() -> None:
    """Test sampler labels expose PECOS tracked-Pauli terminology."""
    from pecos_rslib import DagCircuit, PauliString
    from pecos_rslib.qec import DemSampler

    dag = DagCircuit()
    dag.pz([0])
    dag.h([0])
    dag.tracked_pauli(PauliString.from_str("X"), label="x_check")

    sampler = DemSampler.from_circuit(dag, p1=0.03, p2=0.0, p_meas=0.0, p_prep=0.0)

    labels = sampler.labels()

    assert sampler.num_tracked_paulis == 1
    assert sampler.num_dem_outputs == 0
    assert sampler.num_observables == 0
    assert "dem_outputs" in labels
    assert "tracked_paulis" in labels
    assert labels["dem_outputs"] == []
    assert labels["tracked_paulis"] == ["x_check"]

    stats = sampler.sample_statistics(2000, seed=7)
    assert stats["logical_error_count"] == 0
    assert stats["per_observable"] == []
    assert stats["per_tracked_pauli"] == []
    assert stats["per_dem_output"] == []
    assert stats["tracked_pauli_statistics_supported"] is False
    assert "cannot directly sample tracked Pauli flips" in stats["tracked_pauli_statistics_error"]

    with pytest.raises(RuntimeError, match="cannot directly sample tracked Pauli flips"):
        sampler.sample_tracked_paulis(seed=7)
    with pytest.raises(RuntimeError, match="cannot directly sample tracked Pauli flips"):
        sampler.sample_tracked_pauli_batch(4, seed=7)


def test_detector_error_model_rejects_legacy_tracked_metadata_json() -> None:
    """Python metadata import should fail fast on legacy tracked-op fields."""
    from pecos_rslib.qec import DetectorErrorModel

    old_json = """
    {
      "format": "pecos.dem.metadata",
      "version": 1,
      "observables": [],
      "tracked_paulis": [],
      "tracked_ops": [
        {
          "id": 0,
          "kind": "tracked_op",
          "label": "old_name",
          "pauli": "+X0",
          "records": []
        }
      ]
    }
    """

    with pytest.raises(ValueError, match="unsupported legacy metadata field: tracked_ops"):
        DetectorErrorModel.from_pecos_metadata_json(old_json)


def test_dem_events_split_observables_and_tracked_paulis() -> None:
    """DEM summaries report detector, observable, and tracked-Pauli effects separately."""
    from pecos_rslib import DagCircuit, PauliString
    from pecos_rslib.qec import DetectorErrorModel

    dag = DagCircuit()
    dag.pz([0])
    dag.h([0])
    dag.tracked_pauli(PauliString.from_str("X"), label="x_check")
    dag.mz([0])
    dag.set_attr("num_measurements", "1")
    dag.set_attr("observables", '[{"id": 0, "records": [-1]}]')

    dem = DetectorErrorModel.from_circuit(
        dag,
        p1=0.03,
        p2=0.0,
        p_meas=0.02,
        p_prep=0.0,
    )
    sampler = dem.to_sampler()

    assert dem.num_dem_outputs == 1
    assert dem.num_observables == 1
    assert dem.num_tracked_paulis == 1
    assert sampler.num_dem_outputs == 1
    assert sampler.num_observables == 1
    assert sampler.num_tracked_paulis == 1
    assert sampler.labels()["tracked_paulis"] == ["x_check"]

    summaries = dem.contribution_effect_summaries()
    assert summaries
    assert all("dem_outputs" in row for row in summaries)
    assert all("observables" in row for row in summaries)
    assert all("tracked_paulis" in row for row in summaries)

    observable_hits = {idx for row in summaries for idx in row["observables"]}
    tracked_hits = {idx for row in summaries for idx in row["tracked_paulis"]}
    assert 0 in observable_hits
    assert tracked_hits == set()


def test_sample_decode_count_ignores_tracked_paulis() -> None:
    """Decoder error counting uses observables, not tracked Paulis."""
    from pecos_rslib import DagCircuit, PauliString
    from pecos_rslib.qec import DemSampler, DetectorErrorModel

    dag = DagCircuit()
    dag.pz([0])
    dag.pz([1])
    dag.h([1])
    dag.tracked_pauli(PauliString.from_str("IZ"), label="tracked_z")
    dag.mz([0])
    dag.set_attr("num_measurements", "1")
    dag.set_attr("detectors", '[{"id": 0, "records": [-1]}]')
    dag.set_attr("observables", '[{"id": 0, "records": [-1]}]')

    sampler = DemSampler.from_circuit(
        dag,
        p1=0.4,
        p2=0.0,
        p_meas=0.15,
        p_prep=0.0,
    )
    dem = DetectorErrorModel.from_circuit(
        dag,
        p1=0.4,
        p2=0.0,
        p_meas=0.15,
        p_prep=0.0,
    )

    assert sampler.num_dem_outputs == 1
    assert sampler.num_observables == 1
    assert sampler.num_tracked_paulis == 1
    assert "logical_observable L0" in dem.to_string()
    assert "logical_observable L1" not in dem.to_string()

    errors = sampler.sample_decode_count(dem.to_string(), 2000, seed=17)
    assert errors == 0


def test_influence_map_tracks_dem_outputs_and_tracked_paulis_separately() -> None:
    """Test influence maps expose DEM outputs and filtered tracked Paulis."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import InfluenceBuilder

    dag = DagCircuit()
    dag.pz([0])
    dag.h([0])

    builder = InfluenceBuilder(dag)
    builder.with_tracked_pauli([(0, "X")])
    influence_map = builder.build()

    assert influence_map.num_tracked_paulis > 0
    assert influence_map.num_observables == 0
    assert influence_map.num_dem_outputs == 0

    csr = influence_map.export_csr()
    assert csr["num_dem_outputs"] == influence_map.num_dem_outputs
    assert csr["num_internal_dem_outputs"] == influence_map.num_tracked_paulis
    assert csr["num_observables"] == 0
    assert csr["num_tracked_paulis"] == influence_map.num_tracked_paulis
    assert "dem_output_offsets_x" in csr
    assert "dem_output_data_x" in csr

    for loc_idx in range(influence_map.num_locations):
        tracked = influence_map.get_tracked_pauli_indices(loc_idx, 1)
        dem_outputs = influence_map.get_dem_output_indices(loc_idx, 1)
        internal_dem_outputs = influence_map.get_internal_dem_output_indices(loc_idx, 1)
        assert dem_outputs == []
        assert tracked == internal_dem_outputs
        assert not influence_map.has_dem_output_flips(loc_idx, 1)
        assert influence_map.has_tracked_pauli_flips(loc_idx, 1) == bool(tracked)


def test_influence_builder_does_not_add_empty_tracked_paulis() -> None:
    """An unconfigured Python InfluenceBuilder should not create identity tracked Paulis."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import InfluenceBuilder

    dag = DagCircuit()
    dag.pz([0])
    dag.h([0])
    dag.mz([0])

    influence_map = InfluenceBuilder(dag).build()

    assert influence_map.num_dem_outputs == 0
    assert influence_map.num_observables == 0
    assert influence_map.num_tracked_paulis == 0


def test_influence_builder_tracked_x_z_are_dem_outputs() -> None:
    """Tracked X/Z helpers create tracked Paulis, not observables."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import InfluenceBuilder

    dag = DagCircuit()
    dag.pz([0])
    dag.h([0])

    builder = InfluenceBuilder(dag)
    builder.with_tracked_x([0])
    builder.with_tracked_z([0])
    influence_map = builder.build()

    assert influence_map.num_dem_outputs == 0
    assert influence_map.num_observables == 0
    assert influence_map.num_tracked_paulis == 2


def test_dem_sampler_zero_noise() -> None:
    """Test that zero noise produces no errors."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    dag = DagCircuit()
    dag.pz([2])
    dag.cx([(0, 2)])
    dag.cx([(1, 2)])
    dag.mz([2])

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.0, 0.0, 0.0, 0.0)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

    assert sampler.num_mechanisms == 0

    stats = sampler.sample_statistics(1000, seed=42)
    assert stats["syndrome_count"] == 0
    assert stats["logical_error_count"] == 0


def test_dem_sampler_repr() -> None:
    """Test DemSampler __repr__."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    dag = DagCircuit()
    dag.pz([0])
    dag.mz([0])

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.01, 0.01, 0.01, 0.01)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

    repr_str = repr(sampler)
    assert "DemSampler" in repr_str
    assert "mechanisms" in repr_str
    assert "dem_outputs" in repr_str
    assert "tracked_paulis" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
