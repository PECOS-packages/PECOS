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

"""Tests for pickle serialization of Rust-backed simulators."""

import copy
import pickle

import numpy as np
import pytest

from pecos_rslib import CoinToss, PauliProp, Qulacs, SparseSim, StateVec


def _state_vec_to_numpy(sim):
    """Convert a StateVec's state to a numpy array."""
    return np.array(sim.vector)


class TestStateVecPickle:
    """Test pickle support for StateVec."""

    def test_roundtrip_default_state(self):
        sim = StateVec(3)
        data = pickle.dumps(sim)
        restored = pickle.loads(data)
        assert restored.num_qubits == 3
        np.testing.assert_array_equal(_state_vec_to_numpy(restored), _state_vec_to_numpy(sim))

    def test_roundtrip_after_gates(self):
        sim = StateVec(2, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        original_state = _state_vec_to_numpy(sim)

        restored = pickle.loads(pickle.dumps(sim))
        assert restored.num_qubits == 2
        np.testing.assert_allclose(_state_vec_to_numpy(restored), original_state, atol=1e-15)

    def test_deepcopy(self):
        sim = StateVec(2, seed=42)
        sim.run_1q_gate("H", 0)
        original_state = _state_vec_to_numpy(sim)

        copied = copy.deepcopy(sim)
        np.testing.assert_allclose(_state_vec_to_numpy(copied), original_state, atol=1e-15)

    def test_unpickled_sim_is_functional(self):
        """Ensure the restored sim can continue running gates."""
        sim = StateVec(2, seed=42)
        sim.run_1q_gate("H", 0)

        restored = pickle.loads(pickle.dumps(sim))
        # Should be able to apply more gates without error
        restored.run_2q_gate("CX", (0, 1), None)
        result = restored.run_1q_gate("MZ", 0)
        assert result in (0, 1)


class TestStateVecProbabilities:
    """Test the probabilities property on StateVec."""

    def test_default_state(self):
        """All probability should be on |00...0>."""
        sim = StateVec(3)
        probs = np.array(sim.probabilities)
        assert probs.shape == (8,)
        np.testing.assert_allclose(probs[0], 1.0, atol=1e-15)
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-15)

    def test_bell_state(self):
        """Bell state should have 50/50 on |00> and |11>."""
        sim = StateVec(2, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        probs = np.array(sim.probabilities)
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-15)
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-15)
        np.testing.assert_allclose(probs[1], 0.0, atol=1e-15)
        np.testing.assert_allclose(probs[2], 0.0, atol=1e-15)

    def test_matches_abs_squared(self):
        """Probabilities should equal |amplitude|^2."""
        sim = StateVec(2, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_1q_gate("H", 1)
        probs = np.array(sim.probabilities)
        amplitudes = np.array(sim.vector)
        np.testing.assert_allclose(probs, np.abs(amplitudes) ** 2, atol=1e-15)

    def test_sums_to_one(self):
        """Probabilities should always sum to 1."""
        sim = StateVec(4, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_1q_gate("H", 2)
        sim.run_2q_gate("CX", (0, 1), None)
        probs = np.array(sim.probabilities)
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-15)

    def test_probability_single_basis_state(self):
        """probability(i) should match probabilities[i]."""
        sim = StateVec(2, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        probs = np.array(sim.probabilities)
        for i in range(4):
            assert sim.probability(i) == pytest.approx(probs[i])

    def test_probability_out_of_range(self):
        """Out-of-range basis_state should raise IndexError."""
        sim = StateVec(2)
        with pytest.raises(IndexError):
            sim.probability(4)


class TestQulacsProbabilities:
    """Test the probabilities property on Qulacs."""

    def test_default_state(self):
        """All probability should be on |00...0>."""
        sim = Qulacs(3, seed=42)
        probs = sim.probabilities
        assert len(probs) == 8
        assert probs[0] == pytest.approx(1.0)
        assert sum(probs) == pytest.approx(1.0)

    def test_bell_state(self):
        """Bell state should have 50/50 on |00> and |11>."""
        sim = Qulacs(2, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        probs = sim.probabilities
        assert probs[0] == pytest.approx(0.5)
        assert probs[3] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.0, abs=1e-15)
        assert probs[2] == pytest.approx(0.0, abs=1e-15)

    def test_matches_probability_method(self):
        """probabilities[i] should match probability(i)."""
        sim = Qulacs(2, seed=42)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        probs = sim.probabilities
        for i in range(4):
            assert sim.probability(i) == pytest.approx(probs[i])


class TestSparseSimPickle:
    """Test pickle support for SparseSim."""

    def test_roundtrip_default_state(self):
        sim = SparseSim(4)
        data = pickle.dumps(sim)
        restored = pickle.loads(data)
        assert restored.num_qubits == 4
        assert restored.stab_tableau() == sim.stab_tableau()
        assert restored.destab_tableau() == sim.destab_tableau()

    def test_roundtrip_after_gates(self):
        sim = SparseSim(3)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        sim.run_1q_gate("S", 2)
        original_stab = sim.stab_tableau()
        original_destab = sim.destab_tableau()

        restored = pickle.loads(pickle.dumps(sim))
        assert restored.num_qubits == 3
        assert restored.stab_tableau() == original_stab
        assert restored.destab_tableau() == original_destab

    def test_deepcopy(self):
        sim = SparseSim(3)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)
        original_stab = sim.stab_tableau()

        copied = copy.deepcopy(sim)
        assert copied.stab_tableau() == original_stab

    def test_unpickled_sim_is_functional(self):
        sim = SparseSim(3)
        sim.run_1q_gate("H", 0)

        restored = pickle.loads(pickle.dumps(sim))
        restored.run_2q_gate("CX", (0, 1), None)
        result = restored.run_1q_gate("MZ", 0)
        assert result in (0, 1)


class TestSparseSimGens:
    """Test SparseSim .gens property."""

    def test_gens_returns_tuple(self):
        sim = SparseSim(3)
        gens = sim.gens
        assert isinstance(gens, tuple)
        assert len(gens) == 2

    def test_gens_matches_stabs_destabs(self):
        sim = SparseSim(3)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)

        stabs, destabs = sim.gens
        assert type(stabs) is type(sim.stabs)
        assert type(destabs) is type(sim.destabs)
        # Both should render the same tableau content
        assert stabs.print_tableau() == sim.stabs.print_tableau()
        assert destabs.print_tableau() == sim.destabs.print_tableau()

    def test_gens_after_gates(self):
        sim = SparseSim(2)
        sim.run_1q_gate("H", 0)
        sim.run_2q_gate("CX", (0, 1), None)

        stabs, destabs = sim.gens
        # Should be a Bell state - stabs and destabs should be non-trivial
        assert stabs is not None
        assert destabs is not None


class TestCoinTossPickle:
    """Test pickle support for CoinToss."""

    def test_roundtrip(self):
        sim = CoinToss(5, prob=0.3)
        data = pickle.dumps(sim)
        restored = pickle.loads(data)
        assert restored.num_qubits == 5
        assert restored.prob == pytest.approx(0.3)

    def test_deepcopy(self):
        sim = CoinToss(3, prob=0.7)
        copied = copy.deepcopy(sim)
        assert copied.num_qubits == 3
        assert copied.prob == pytest.approx(0.7)

    def test_unpickled_sim_is_functional(self):
        sim = CoinToss(2, prob=0.5)
        restored = pickle.loads(pickle.dumps(sim))
        result = restored.run_measure(0)
        assert isinstance(result, dict)


class TestPauliPropPickle:
    """Test pickle support for PauliProp."""

    def test_roundtrip_empty(self):
        sim = PauliProp(num_qubits=4, track_sign=True)
        data = pickle.dumps(sim)
        restored = pickle.loads(data)
        assert restored.num_qubits == 4
        assert restored.track_sign is True
        assert restored.is_identity()

    def test_roundtrip_with_faults(self):
        sim = PauliProp(num_qubits=4, track_sign=True)
        sim.add_x(0)
        sim.add_z(2)
        sim.add_y(3)

        restored = pickle.loads(pickle.dumps(sim))
        assert restored.contains_x(0)
        assert not restored.contains_z(0)
        assert restored.contains_z(2)
        assert not restored.contains_x(2)
        assert restored.contains_y(3)

    def test_roundtrip_preserves_sign(self):
        sim = PauliProp(num_qubits=2, track_sign=True)
        sim.add_x(0)
        sim.flip_sign()  # sign is now negative

        restored = pickle.loads(pickle.dumps(sim))
        assert restored.get_sign() is True  # True means negative

    def test_roundtrip_preserves_img(self):
        sim = PauliProp(num_qubits=2, track_sign=True)
        sim.add_x(0)
        sim.flip_img(1)  # imaginary component

        restored = pickle.loads(pickle.dumps(sim))
        assert restored.get_img() == 1

    def test_roundtrip_no_sign_tracking(self):
        sim = PauliProp()
        sim.add_x(0)
        sim.add_z(1)

        restored = pickle.loads(pickle.dumps(sim))
        assert restored.track_sign is False
        assert restored.contains_x(0)
        assert restored.contains_z(1)

    def test_deepcopy(self):
        sim = PauliProp(num_qubits=3, track_sign=True)
        sim.add_x(0)
        sim.add_z(1)

        copied = copy.deepcopy(sim)
        assert copied.contains_x(0)
        assert copied.contains_z(1)
        assert copied.num_qubits == 3
