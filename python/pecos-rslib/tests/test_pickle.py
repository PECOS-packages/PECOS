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

from pecos_rslib import CoinToss, PauliProp, SparseSim, StateVec


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
