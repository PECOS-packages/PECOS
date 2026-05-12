# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for DemSampler Python bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pecos.qec import DagFaultAnalyzer, DemSampler, DemSamplerBuilder
from pecos_rslib import DagCircuit

if TYPE_CHECKING:
    from pecos.qec import DagFaultInfluenceMap


def _build_repetition_code_circuit(num_rounds: int = 3) -> DagCircuit:
    """Build a simple repetition code circuit."""
    dag = DagCircuit()
    for _ in range(num_rounds):
        dag.pz([3])
        dag.pz([4])
        dag.cx([(0, 3)])
        dag.cx([(1, 3)])
        dag.cx([(1, 4)])
        dag.cx([(2, 4)])
        dag.mz([3])
        dag.mz([4])
    return dag


def _build_influence_map(dag: DagCircuit) -> DagFaultInfluenceMap:
    """Build influence map with logical Z."""
    analyzer = DagFaultAnalyzer(dag)
    return analyzer.build_influence_map()


class TestDemSamplerRawMode:
    """Test raw measurement output mode."""

    def test_raw_uniform_creates_sampler(self) -> None:
        """Test that raw_uniform constructor creates a valid sampler."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw_uniform(im, 0.01)
        assert sampler.num_mechanisms > 0

    def test_raw_circuit_noise_creates_sampler(self) -> None:
        """Test that raw constructor with per-gate noise creates a valid sampler."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw(im, 0.001, 0.01, 0.005, 0.001)
        assert sampler.num_mechanisms > 0

    def test_raw_sample_returns_correct_shape(self) -> None:
        """Test that raw sample returns non-empty output and observable lists."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw_uniform(im, 0.01)
        outputs, obs = sampler.sample(seed=42)
        assert isinstance(outputs, list)
        assert isinstance(obs, list)
        assert len(outputs) > 0

    def test_raw_sample_batch(self) -> None:
        """Test that sample_batch returns the requested number of shots."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw_uniform(im, 0.01)
        all_outputs, _all_obs = sampler.sample_batch(100, seed=42)
        assert len(all_outputs) == 100

    def test_raw_zero_noise_statistics(self) -> None:
        """Test that zero noise produces no syndromes or logical errors."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw_uniform(im, 0.0)
        stats = sampler.sample_statistics(1000, seed=42)
        assert stats["syndrome_count"] == 0
        assert stats["logical_error_count"] == 0

    def test_raw_high_noise_produces_syndromes(self) -> None:
        """Test that high noise produces a non-trivial syndrome rate."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw_uniform(im, 0.1)
        stats = sampler.sample_statistics(5000, seed=42)
        assert stats["syndrome_rate"] > 0.05


class TestDemSamplerDetectorMode:
    """Test detector-event output mode."""

    def test_detector_mode_creates_sampler(self) -> None:
        """Test that detector mode creates a sampler with correct output counts."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.with_detectors(
            im,
            detectors=[[-1], [-2]],
            observables=[],
            p1=0.001,
            p2=0.01,
            p_meas=0.005,
            p_prep=0.001,
        )
        assert sampler.num_outputs == 2
        assert sampler.num_observables == 0

    def test_detector_mode_sample_shape(self) -> None:
        """Test detector mode sample returns detector and observable counts."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.with_detectors(
            im,
            detectors=[[-1], [-2]],
            observables=[[-1]],
            p1=0.001,
            p2=0.01,
            p_meas=0.005,
            p_prep=0.001,
        )
        det_events, obs_flips = sampler.sample(seed=42)
        assert len(det_events) == 2
        assert len(obs_flips) == 1
        assert sampler.num_dem_outputs == 1
        assert sampler.num_observables == 1
        assert sampler.num_tracked_paulis == 0

    def test_detector_mode_matches_dem_sampler_builder(self) -> None:
        """DemSampler detector mode should match DemSamplerBuilder exactly."""
        dag = _build_repetition_code_circuit(3)
        im = _build_influence_map(dag)

        p1, p2, p_meas, p_prep = 0.001, 0.01, 0.005, 0.001
        det_records = [[-1], [-2]]
        obs_records = []
        num_shots = 20_000
        seed = 42

        # DemSamplerBuilder path
        import json

        det_json = json.dumps([{"id": i, "records": r} for i, r in enumerate(det_records)])
        obs_json = json.dumps([{"id": i, "records": r} for i, r in enumerate(obs_records)])
        dem_sampler = (
            DemSamplerBuilder(im)
            .with_noise(p1, p2, p_meas, p_prep)
            .with_detectors_json(det_json)
            .with_observables_json(obs_json)
            .build()
        )
        dem_stats = dem_sampler.sample_statistics(num_shots, seed)

        # DemSampler detector mode
        unified_sampler = DemSampler.with_detectors(
            im,
            detectors=det_records,
            observables=obs_records,
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        unified_stats = unified_sampler.sample_statistics(num_shots, seed)

        # Should match exactly (same seed → same internal DemSamplerBuilder path)
        assert dem_stats["syndrome_count"] == unified_stats["syndrome_count"]
        assert dem_stats["logical_error_count"] == unified_stats["logical_error_count"]


class TestDemSamplerValidation:
    """Test detector definition validation."""

    def test_linearly_dependent_detectors_rejected(self) -> None:
        """Test that linearly dependent detector definitions are rejected."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)

        # D2 = D0 XOR D1 → linearly dependent
        with pytest.raises(ValueError, match="linearly independent"):
            DemSampler.with_detectors(
                im,
                detectors=[[0], [1], [0, 1]],
                observables=[],
                p1=0.001,
                p2=0.01,
                p_meas=0.005,
                p_prep=0.001,
            )

    def test_independent_detectors_accepted(self) -> None:
        """Test that linearly independent detector definitions are accepted."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)

        sampler = DemSampler.with_detectors(
            im,
            detectors=[[0], [1]],
            observables=[],
            p1=0.001,
            p2=0.01,
            p_meas=0.005,
            p_prep=0.001,
        )
        assert sampler.num_outputs == 2


class TestDemSamplerRepr:
    """Test string representation."""

    def test_repr_raw_mode(self) -> None:
        """Test repr output for raw mode sampler."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.raw_uniform(im, 0.01)
        r = repr(sampler)
        assert "DemSampler" in r
        assert "DemSampler" in r

    def test_repr_detector_mode(self) -> None:
        """Test repr output for detector mode sampler."""
        dag = _build_repetition_code_circuit(2)
        im = _build_influence_map(dag)
        sampler = DemSampler.with_detectors(
            im,
            detectors=[[-1]],
            observables=[],
            p1=0.01,
            p2=0.01,
            p_meas=0.01,
            p_prep=0.01,
        )
        r = repr(sampler)
        assert "DemSampler" in r
        assert "DemSampler" in r
