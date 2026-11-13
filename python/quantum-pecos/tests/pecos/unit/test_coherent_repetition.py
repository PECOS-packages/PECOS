# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent physical state-vector oracles for coherent repetition instruments."""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pytest
from pecos.experimental.detector_models.process import compile_instrument
from pecos.experimental.detector_models.repetition import (
    RepetitionConfig,
    RepetitionModel,
    base_angles,
    build_model,
    detector_history,
)


def _tensor(operators):
    result = np.ones((1, 1), complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def _dense_reference(config, checks, *, retain_states=False):
    """Physical d-qubit state vectors; no compact two-component transfer logic."""
    distance = config.distance
    dimension = 1 << distance
    bits = (np.arange(dimension)[:, None] >> np.arange(distance - 1, -1, -1)) & 1
    parities = bits[:, :-1] ^ bits[:, 1:]
    hadamard = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    basis_change = _tensor([hadamard] * distance) if config.basis == "phase" else np.eye(dimension)
    if config.basis == "bit":
        local = [
            np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
            for theta in config.angles
        ]
    else:
        local = [np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]) for theta in config.angles]
    unitary = _tensor(local)
    flip_probabilities = np.sin(np.asarray(config.angles) / 2) ** 2
    event_probability = np.prod(np.where(bits, flip_probabilities, 1 - flip_probabilities), axis=1)
    transition = event_probability[np.bitwise_xor(np.arange(dimension)[:, None], np.arange(dimension)[None, :])]
    enhanced, canonical, log_joint, canonical_log_joint, evidence = [], [], [], [], []
    logical_states = []
    for history in checks:
        state = basis_change[:, 0].copy()
        classical = np.zeros(dimension)
        classical[0] = 1
        for syndrome in history:
            keep = np.all(parities == syndrome, axis=1)
            projector = basis_change @ np.diag(keep) @ basis_change.conj().T
            state = projector @ unitary @ state
            classical = (classical @ transition) * keep
        measured_state = basis_change.conj().T @ state
        probability = abs(measured_state) ** 2
        if retain_states:
            indices = [
                np.flatnonzero((bits[:, 0] == logical) & np.all(parities == history[-1], axis=1))[0]
                for logical in (0, 1)
            ]
            logical_vector = measured_state[indices]
            logical_states.append(np.outer(logical_vector, logical_vector.conj()))
        joint = np.array([probability[bits[:, 0] == logical].sum() for logical in (0, 1)])
        binary_joint = np.array([classical[bits[:, 0] == logical].sum() for logical in (0, 1)])
        enhanced.append(joint / joint.sum())
        canonical.append(binary_joint / binary_joint.sum())
        with np.errstate(divide="ignore"):
            log_joint.append(np.log(joint))
            canonical_log_joint.append(np.log(binary_joint))
            evidence.append(np.log(joint.sum()))
    result = {
        "posterior": np.array(enhanced),
        "canonical_posterior": np.array(canonical),
        "log_joint": np.array(log_joint),
        "canonical_log_joint": np.array(canonical_log_joint),
        "log_evidence": np.array(evidence),
    }
    if retain_states:
        result["logical_states"] = np.array(logical_states)
    return result


@pytest.mark.parametrize("basis", ["bit", "phase"])
def test_five_qubit_two_round_queries_match_independent_physical_statevectors(basis):
    config = RepetitionConfig(distance=5, rounds=2, basis=basis, angles=(0.6, 0.75, 0.9, 0.92, 0.94))
    checks = np.array(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [1, 1, 0, 1]],
            [[0, 1, 1, 0], [1, 0, 1, 1]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    actual = build_model(config).query(checks)
    expected = _dense_reference(config, checks)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=0)


@pytest.mark.parametrize("basis", ["bit", "phase"])
def test_all_three_qubit_three_round_histories_match_dense_statevectors(basis):
    config = RepetitionConfig(rounds=3, basis=basis)
    checks = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8).reshape(-1, 3, 2)
    queried = build_model(config).query(checks)
    expected = _dense_reference(config, checks)
    for key in expected:
        np.testing.assert_allclose(queried[key], expected[key], atol=2e-12, rtol=0)
    assert np.max(np.abs(np.asarray(queried["posterior"]) - np.asarray(queried["canonical_posterior"]))) > 0.1


@pytest.mark.parametrize("basis", ["bit", "phase"])
def test_seven_qubit_ten_round_queries_match_physical_statevectors(basis):
    config = RepetitionConfig(distance=7, rounds=10, basis=basis, angles=base_angles(7))
    checks = np.random.default_rng(928).integers(0, 2, size=(3, 10, 6), dtype=np.uint8)
    checks[0] = 0
    queried = build_model(config).query(checks)
    expected = _dense_reference(config, checks)
    for key in expected:
        np.testing.assert_allclose(queried[key], expected[key], atol=3e-12, rtol=0)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_cached_quantum_detector_maps_form_a_complete_instrument(distance):
    angles = tuple(np.interp(np.linspace(0, 1, distance), [0, 0.5, 1], [0.6, 0.9, 0.94]))
    model = build_model(RepetitionConfig(distance=distance, rounds=3, basis="bit", angles=angles))
    assert model.delta_operators.shape == (1 << (distance - 1), 2, 2)
    effects = np.array([operator.conj().T @ operator for operator in np.asarray(model.delta_operators)])
    np.testing.assert_allclose(effects.sum(axis=0), np.eye(2), atol=2e-12)
    np.testing.assert_allclose(effects, np.asarray(model.syndrome_probabilities)[:, None, None] * np.eye(2), atol=2e-12)


@pytest.mark.parametrize("basis", ["bit", "phase"])
def test_generic_schema_export_retains_conditional_logical_coherence(basis):
    config = RepetitionConfig(distance=3, rounds=3, basis=basis, angles=(0.6, 0.9, 0.94))
    model = build_model(config)
    instrument = compile_instrument(model.as_detector_process())
    checks = np.array(
        [
            [[0, 0], [0, 0], [0, 0]],
            [[1, 0], [1, 1], [0, 1]],
            [[0, 1], [1, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )
    expected = _dense_reference(config, checks, retain_states=True)["logical_states"]
    detectors = model.query(checks)["detectors"]
    rho_zero = np.diag([1, 0]).astype(complex)
    actual = np.array([instrument.branches[tuple(int(bit) for bit in row)].apply(rho_zero) for row in detectors])
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    assert np.max(np.abs(actual[:, 0, 1])) > 1e-6


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_one_round_is_a_null_for_all_test_distances_and_scales(distance, scale):
    angles = tuple(scale * np.interp(np.linspace(0, 1, distance), [0, 0.5, 1], [0.6, 0.9, 0.94]))
    config = RepetitionConfig(distance=distance, rounds=1, basis="bit", angles=angles)
    checks = np.array(list(itertools.product((0, 1), repeat=distance - 1)), dtype=np.uint8)[:, None, :]
    queried = build_model(config).query(checks)
    np.testing.assert_allclose(queried["posterior"], queried["canonical_posterior"], atol=2e-12)
    np.testing.assert_allclose(np.exp(queried["log_evidence"]).sum(), 1, atol=2e-12)


def test_bit_and_phase_models_are_exact_hadamard_duals():
    angles = (0.6, 0.75, 0.9, 0.92, 0.94)
    checks = np.random.default_rng(1491).integers(0, 2, size=(30, 5, 4), dtype=np.uint8)
    outputs = [
        build_model(RepetitionConfig(distance=5, rounds=5, basis=basis, angles=angles)).query(checks)
        for basis in ("bit", "phase")
    ]
    for key in ("posterior", "canonical_posterior", "log_joint", "canonical_log_joint", "log_evidence", "detectors"):
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], atol=2e-12)


def test_final_readout_complement_cannot_change_model_posterior():
    config = RepetitionConfig(distance=7, rounds=10, basis="bit", angles=(0.6, 0.7, 0.8, 0.9, 0.913, 0.926, 0.94))
    checks = np.random.default_rng(120).integers(0, 2, size=(20, 10, 6), dtype=np.uint8)
    # Reconstruct any final string with the last observed parity and then its
    # global complement. Those have opposite raw logical readout labels.
    final_data = np.zeros((len(checks), 7), dtype=np.uint8)
    for site in range(6):
        final_data[:, site + 1] = final_data[:, site] ^ checks[:, -1, site]
    model = build_model(config)
    left = model.query(checks, final_data=final_data)
    right = model.query(checks, final_data=final_data ^ 1)
    assert left["detectors"].shape == (20, 66)
    assert np.asarray(left["detectors"]).dtype == np.uint8
    assert not np.any(left["detectors"][:, -6:])
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
    # For odd distance and single local Rx/Rz pulses the two models predict
    # exactly the same syndrome law. Their difference is conditional logical
    # coherence, not a fit to different detector frequencies.
    np.testing.assert_allclose(np.logaddexp.reduce(left["log_joint"], axis=1), left["log_evidence"], atol=2e-12)
    np.testing.assert_allclose(
        np.logaddexp.reduce(left["canonical_log_joint"], axis=1),
        left["log_evidence"],
        atol=2e-12,
    )
    np.testing.assert_allclose(np.asarray(left["posterior"]).sum(axis=1), 1, atol=2e-12)


def test_inconsistent_final_parity_is_rejected_instead_of_used_as_target_information():
    config = RepetitionConfig(distance=3, rounds=3, basis="bit", angles=(0.6, 0.9, 0.94))
    checks = np.zeros((1, 3, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="final parity"):
        build_model(config).query(checks, final_data=np.array([[0, 1, 0]], dtype=np.uint8))


def test_compact_model_serialization_preserves_complex_queries(tmp_path):
    config = RepetitionConfig(distance=7, rounds=10, basis="phase", angles=(0.6, 0.7, 0.8, 0.9, 0.913, 0.926, 0.94))
    model = build_model(config)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = RepetitionModel.load(path)
    checks = np.random.default_rng(913).integers(0, 2, size=(20, 10, 6), dtype=np.uint8)
    left, right = model.query(checks), loaded.query(checks)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])
    manifest = json.loads(path.read_text())
    assert manifest["schema_version"] >= 1
    assert "amplitudes_sha256" in manifest


@pytest.mark.parametrize(
    "fields",
    [
        {"distance": 2, "angles": (0.1, 0.2)},
        {"distance": 4, "angles": (0.1,) * 4},
        {"distance": 17, "angles": (0.1,) * 17},
        {"rounds": 0},
        {"rounds": 1.5},
        {"basis": "both"},
        {"angles": (0.1, 0.2)},
        {"angles": (0.1, np.nan, 0.3)},
        {"angles": (0.1, True, 0.3)},
        {"angles": (0.1, "0.2", 0.3)},
        {"angles": (0.1, 0.2j, 0.3)},
    ],
)
def test_unsupported_configurations_are_rejected(fields):
    with pytest.raises(ValueError, match=r"distance|rounds|basis|angle"):
        RepetitionConfig(**fields)


def test_mutating_caller_angles_or_array_views_cannot_change_a_fixed_model():
    angles = [0.6, 0.9, 0.94]
    config = RepetitionConfig(angles=angles)
    model = build_model(config)
    checks = np.zeros((1, 3, 2), dtype=np.uint8)
    expected = np.asarray(model.query(checks)["posterior"]).copy()
    angles[0] = 2.5
    exposed = model.delta_operators
    exposed[0, 0, 0] = 5
    np.testing.assert_array_equal(model.query(checks)["posterior"], expected)


def test_empty_queries_keep_output_shapes():
    model = build_model(RepetitionConfig())
    queried = model.query(np.zeros((0, 3, 2), dtype=np.uint8))
    assert queried["detectors"].shape == (0, 8)
    assert queried["log_evidence"].shape == (0,)
    for key in ("posterior", "canonical_posterior", "log_joint", "canonical_log_joint"):
        assert queried[key].shape == (0, 2)


def test_noiseless_model_keeps_zero_probability_logical_branches_explicit():
    model = build_model(RepetitionConfig(angles=(0.0,) * 3))
    queried = model.query(np.zeros((1, 3, 2), dtype=np.uint8))
    np.testing.assert_array_equal(queried["posterior"], [[1.0, 0.0]])
    np.testing.assert_array_equal(queried["log_joint"], [[0.0, -np.inf]])
    np.testing.assert_array_equal(queried["log_evidence"], [0.0])
    with pytest.raises(ValueError, match="zero probability"):
        model.query(np.ones((1, 3, 2), dtype=np.uint8))


def test_binary_records_and_round_count_are_validated():
    model = build_model(RepetitionConfig())
    with pytest.raises(ValueError, match="binary"):
        model.query(np.full((1, 3, 2), 0.5))
    with pytest.raises(ValueError, match="shape"):
        model.query(np.zeros((1, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="at least one round"):
        detector_history(np.zeros((1, 0, 2), dtype=np.uint8))


def test_snapshot_integrity_and_declared_channel_are_both_checked(tmp_path):
    model = build_model(RepetitionConfig())
    path = tmp_path / "model.json"
    model.save(path)
    with pytest.raises(FileExistsError):
        model.save(path)
    manifest = json.loads(path.read_text())
    manifest["round_amplitudes"][0][0][0] += 0.01
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        RepetitionModel.load(path)
    encoded = json.dumps(manifest["round_amplitudes"], separators=(",", ":"), allow_nan=False)
    manifest["amplitudes_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="declared parameters"):
        RepetitionModel.load(path)
