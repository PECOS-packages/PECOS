#!/usr/bin/env python

# copyright 2022 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Initial author: Tyson Lawrence

import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

sys.path.append("./build/bin")
sys.path.append("../build/bin")
import cuquantum_wrapper as cq


class TestPythonBindings(unittest.TestCase):
    def test_bell_state(self):
        """Test Bell state"""

        # Create a workspace
        ws = cq.CuStatevecWorkspace()

        # Initialize a zero state on th device
        sv = cq.StateVector(2)
        sv.init_on_device()

        # Since we initialized on the device, we need to read back to check
        sv.read_from_device()
        v = sv.get()
        v_expected = np.array([1, 0, 0, 0], dtype=complex)
        assert_array_equal(v, v_expected)

        # Create gates, copy to device, apply to target qubits, and free
        h = cq.Hadamard()
        cnot = cq.CX()
        h.copy_to_device()
        cnot.copy_to_device()
        h.apply(sv, ws, [], [0], False)
        cnot.apply(sv, ws, [0], [1], False)
        h.free_on_device()
        cnot.free_on_device()

        # Read final state
        sv.read_from_device()
        v = sv.get()
        S2 = 1 / np.sqrt(2)
        v_expected = np.array([S2, 0, 0, S2], dtype=complex)
        assert_allclose(v, v_expected)

        # Reset back to zero state
        sv.reset(ws)
        sv.read_from_device()
        v = sv.get()
        v_expected = np.array([1, 0, 0, 0], dtype=complex)
        assert_array_equal(v, v_expected)

    def test_tq_gates(self):
        """Test TQ gates not already tested work."""
        # Create a workspace
        ws = cq.CuStatevecWorkspace()

        # Initialize a zero state on th device
        sv = cq.StateVector(2)
        sv.init_on_device()

        # test some other TQ gates
        sqrtzz = cq.SZZ()
        rzz = cq.RZZ(0.234)
        rzz.copy_to_device()
        sqrtzz.copy_to_device()
        rzz.apply(sv, ws, [0], [1], False)
        sqrtzz.apply(sv, ws, [0], [1], False)
        rzz.free_on_device()
        sqrtzz.free_on_device()

    def test_measure_channels(self):
        """Test batch measure"""
        # Create a workspace
        ws = cq.CuStatevecWorkspace()

        # Create the input (to be measured) state vector
        v = np.array(
            [0, 0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j, 0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j], dtype=complex
        )
        sv = cq.StateVector(v)

        sv.copy_to_device()

        # Batch measure the qubits (all of them)
        bit_order = [2, 1, 0]
        randnum = 0.5
        collapse = True

        res = sv.batch_measure(ws, bit_order, randnum, collapse)

        sv.read_from_device()
        sv.free_on_device()

        # Get the final state as a numpy array
        v = sv.get()

        # Expected state vector and results
        res_expected = [1, 1, 0]
        v_expected = np.array([0, 0, 0, 0, 0, 0, 0.6 + 0.8j, 0], dtype=complex)

        # Check
        assert_array_equal(v, v_expected)
        self.assertEqual(res, res_expected)

    def test_quantum_volume(self):
        """Test quantum volume... not a real test"""

        num_qubits = 4

        # Create the workspace
        ws = cq.CuStatevecWorkspace()

        # Initialize the state vector to the zero state on the device
        sv = cq.StateVector(num_qubits)
        sv.init_on_device()

        # Create and run the QV sim
        qv = cq.QuantumVolume(num_qubits)
        qv.copy_to_device()
        qv.apply(sv, ws)
        qv.free_on_device()

        # Get probabilities out
        probs = sv.get_probabilities(ws)

        # Read the final state and free
        sv.read_from_device()
        sv.free_on_device()


if __name__ == "__main__":
    unittest.main()
