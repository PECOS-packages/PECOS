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
    dag.pz(2)  # Ancilla
    dag.cx(0, 2)
    dag.cx(1, 2)
    dag.mz(2)

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
    dag.pz(2)
    dag.cx(0, 2)
    dag.cx(1, 2)
    dag.mz(2)

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.1, 0.1, 0.1, 0.1)  # Higher noise for visible effects
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    builder.with_observables_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

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
    dag.pz(2)
    dag.cx(0, 2)
    dag.cx(1, 2)
    dag.mz(2)

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
    dag.pz(2)
    dag.cx(0, 2)
    dag.cx(1, 2)
    dag.mz(2)

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

    assert stats["total_shots"] == 10000
    assert 0.0 <= stats["logical_error_rate"] <= 1.0
    assert 0.0 <= stats["syndrome_rate"] <= 1.0


def test_dem_sampler_zero_noise() -> None:
    """Test that zero noise produces no errors."""
    from pecos_rslib import DagCircuit
    from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder

    dag = DagCircuit()
    dag.pz(2)
    dag.cx(0, 2)
    dag.cx(1, 2)
    dag.mz(2)

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
    dag.pz(0)
    dag.mz(0)

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    builder = DemSamplerBuilder(influence_map)
    builder.with_noise(0.01, 0.01, 0.01, 0.01)
    builder.with_detectors_json('[{"id": 0, "records": [-1]}]')
    sampler = builder.build()

    repr_str = repr(sampler)
    assert "DemSampler" in repr_str
    assert "mechanisms" in repr_str
    assert "detectors" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
