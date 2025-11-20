# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""NumPy compatibility tests for Qulacs simulator."""

import pytest

# Skip entire module if numpy not available
pytest.importorskip("numpy")

import numpy as np
import pecos as pc

pytest.importorskip("pecos_rslib", reason="pecos_rslib required for qulacs tests")

from pecos.simulators.qulacs import Qulacs

# Mark all tests in this module as requiring numpy
pytestmark = pytest.mark.numpy


class TestQulacsNumpyCompatibility:
    """Test compatibility with NumPy array operations."""

    def test_numpy_array_conversion(self) -> None:
        """Test that PECOS arrays can be converted to NumPy arrays."""
        sim = Qulacs(2)

        state = sim.vector

        # Should be numpy-compatible (Array implements buffer protocol)
        # Can convert to numpy array via np.asarray
        state_np = np.asarray(state)
        assert isinstance(state_np, np.ndarray)

        # Should have complex dtype
        assert np.iscomplexobj(state_np)

        # Should be normalized
        norm = np.sum(abs(state_np) ** 2)
        assert pc.isclose(norm, 1.0, rtol=1e-5, atol=1e-8)

        # Should support numpy operations
        probabilities = abs(state_np) ** 2
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.dtype == float

    def test_numpy_sum_with_pecos_arrays(self) -> None:
        """Test that np.sum works on PECOS arrays."""
        sim = Qulacs(2)

        # Prepare |10⟩ and swap to |01⟩
        sim.bindings["X"](sim, 0)  # |10⟩
        sim.bindings["SWAP"](sim, 0, 1)  # Should become |01⟩

        # Check that exactly one basis state has probability 1
        probs = pc.abs(sim.vector) ** 2
        assert np.sum(probs > 0.5) == 1  # Exactly one state should be populated

    def test_numpy_operations_preserve_normalization(self) -> None:
        """Test that state normalization is preserved after NumPy operations."""
        sim = Qulacs(3)

        # Apply various gates
        sim.bindings["H"](sim, 0)
        sim.bindings["CX"](sim, 0, 1)
        sim.bindings["RY"](sim, 2, angle=pc.f64.frac_pi_4)
        sim.bindings["CZ"](sim, 1, 2)
        sim.bindings["T"](sim, 0)

        # Check normalization using NumPy
        state = sim.vector
        norm_squared = np.sum(abs(state) ** 2)
        assert pc.isclose(norm_squared, 1.0, rtol=0.0, atol=1e-10)
