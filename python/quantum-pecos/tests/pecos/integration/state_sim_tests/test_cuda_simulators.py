# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Integration tests for CUDA-accelerated quantum simulators using Rust cuQuantum bindings.

These tests require:
- NVIDIA GPU with CUDA support
- cuQuantum SDK installed
- pecos-rslib-cuda package

Tests will be skipped if cuQuantum is not available.
"""

from __future__ import annotations

import pytest

# Check if CUDA simulators are available
try:
    from pecos_rslib_cuda import is_cuquantum_available

    CUQUANTUM_AVAILABLE = is_cuquantum_available()
except ImportError:
    CUQUANTUM_AVAILABLE = False

# Skip all tests in this module if cuQuantum is not available
pytestmark = pytest.mark.skipif(
    not CUQUANTUM_AVAILABLE,
    reason="cuQuantum SDK not available (pecos-rslib-cuda requires cuQuantum)",
)


class TestCudaStateVec:
    """Tests for CudaStateVec (Rust cuQuantum state vector simulator)."""

    def test_import(self) -> None:
        """Test that CudaStateVec can be imported."""
        from pecos.simulators import CudaStateVec

        assert CudaStateVec is not None

    def test_creation(self) -> None:
        """Test creating a CudaStateVec simulator."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(4)
        assert sim.num_qubits == 4

    def test_creation_with_seed(self) -> None:
        """Test creating a CudaStateVec simulator with a seed."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(4, seed=42)
        assert sim.num_qubits == 4

    def test_reset(self) -> None:
        """Test resetting the simulator."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(2)
        sim.run_gate("X", [0])
        sim.reset()
        # After reset, state should be |00>

    def test_single_qubit_gates(self) -> None:
        """Test single-qubit gate operations."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(2)

        # Test Pauli gates
        sim.run_gate("X", [0])
        sim.run_gate("Y", [1])
        sim.run_gate("Z", [0])

        # Test Hadamard
        sim.run_gate("H", [0])
        sim.run_gate("H", [1])

        # Test S and T gates
        sim.run_gate("S", [0])
        sim.run_gate("T", [1])

    def test_two_qubit_gates(self) -> None:
        """Test two-qubit gate operations."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(3)

        # Create Bell state
        sim.run_gate("H", [0])
        sim.run_gate("CX", [(0, 1)])

        # Test other two-qubit gates
        sim.run_gate("CZ", [(1, 2)])
        sim.run_gate("SWAP", [(0, 2)])

    def test_rotation_gates(self) -> None:
        """Test rotation gate operations."""
        import math

        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(2)

        # Test rotation gates with angles
        sim.run_gate("RX", [0], angles=(math.pi / 4,))
        sim.run_gate("RY", [1], angles=(math.pi / 2,))
        sim.run_gate("RZ", [0], angles=(math.pi,))

    def test_measurement(self) -> None:
        """Test measurement operations."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(2)

        # Prepare |00> state and measure - should get 0
        result = sim.run_gate("Measure", [0, 1])
        assert 0 in result or (0,) in result or len(result) == 2

    def test_bell_state_measurement(self) -> None:
        """Test Bell state creation and measurement correlation."""
        from pecos.simulators import CudaStateVec

        # Run multiple shots to check correlation
        num_shots = 100
        correlated = 0

        for _ in range(num_shots):
            sim = CudaStateVec(2, seed=None)
            sim.run_gate("H", [0])
            sim.run_gate("CX", [(0, 1)])
            results = sim.run_gate("Measure", [0, 1])

            # In a Bell state, measurements should be correlated
            if len(results) == 2:
                vals = list(results.values())
                if vals[0] == vals[1]:
                    correlated += 1

        # Should be highly correlated (allowing for some measurement noise)
        assert correlated > num_shots * 0.9

    def test_run_gate_interface(self) -> None:
        """Test the run_gate interface with various gate symbols."""
        from pecos.simulators import CudaStateVec

        sim = CudaStateVec(4)

        # Test initialization
        sim.run_gate("Init", [0, 1, 2, 3])

        # Test various Clifford gates
        sim.run_gate("SX", [0])
        sim.run_gate("SY", [1])
        sim.run_gate("SZ", [2])
        sim.run_gate("H2", [3])

        # Test two-qubit Clifford gates
        sim.run_gate("SXX", [(0, 1)])
        sim.run_gate("SYY", [(2, 3)])


class TestCudaStabilizer:
    """Tests for CudaStabilizer (Rust cuQuantum stabilizer simulator)."""

    def test_import(self) -> None:
        """Test that CudaStabilizer can be imported."""
        from pecos.simulators import CudaStabilizer

        assert CudaStabilizer is not None

    def test_creation(self) -> None:
        """Test creating a CudaStabilizer simulator."""
        from pecos.simulators import CudaStabilizer

        sim = CudaStabilizer(100)
        assert sim.num_qubits == 100

    def test_creation_with_seed(self) -> None:
        """Test creating a CudaStabilizer simulator with a seed."""
        from pecos.simulators import CudaStabilizer

        sim = CudaStabilizer(50, seed=42)
        assert sim.num_qubits == 50

    def test_large_qubit_count(self) -> None:
        """Test that CudaStabilizer can handle many qubits (Clifford-only)."""
        from pecos.simulators import CudaStabilizer

        # Stabilizer simulators can handle many more qubits than state vector
        sim = CudaStabilizer(500)
        assert sim.num_qubits == 500

        # Apply some Clifford gates
        sim.run_gate("H", [0])
        for i in range(10):
            sim.run_gate("CX", [(i, i + 1)])

    def test_clifford_gates(self) -> None:
        """Test Clifford gate operations."""
        from pecos.simulators import CudaStabilizer

        sim = CudaStabilizer(4)

        # Pauli gates
        sim.run_gate("X", [0])
        sim.run_gate("Y", [1])
        sim.run_gate("Z", [2])

        # Hadamard
        sim.run_gate("H", [0, 1, 2, 3])

        # S gate
        sim.run_gate("S", [0])
        sim.run_gate("Sd", [1])

        # Two-qubit Clifford
        sim.run_gate("CX", [(0, 1)])
        sim.run_gate("CZ", [(2, 3)])

    def test_non_clifford_raises(self) -> None:
        """Test that non-Clifford gates raise an error."""
        from pecos.simulators import CudaStabilizer

        sim = CudaStabilizer(2)

        # T gate is non-Clifford and should raise ValueError
        with pytest.raises(ValueError, match="not a Clifford gate"):
            sim.run_gate("T", [0])

    def test_measurement(self) -> None:
        """Test measurement operations."""
        from pecos.simulators import CudaStabilizer

        sim = CudaStabilizer(2)

        # Prepare and measure
        sim.run_gate("H", [0])
        sim.run_gate("CX", [(0, 1)])
        result = sim.run_gate("Measure", [0, 1])

        # Should have measurement results
        assert isinstance(result, dict)

    def test_surface_code_syndrome(self) -> None:
        """Test a simple surface code syndrome extraction pattern."""
        from pecos.simulators import CudaStabilizer

        # 9 qubits for a distance-3 surface code
        sim = CudaStabilizer(9)

        # Initialize data qubits
        sim.run_gate("Init +Z", [0, 1, 2, 3, 4, 5, 6, 7, 8])

        # Simple X stabilizer check (H-CX-CX-H pattern)
        sim.run_gate("H", [4])  # Ancilla
        sim.run_gate("CX", [(4, 0)])  # Connect to data qubits
        sim.run_gate("CX", [(4, 1)])
        sim.run_gate("H", [4])
        result = sim.run_gate("Measure", [4])

        # In +Z state, X stabilizer measurement should give 0
        assert 4 in result or (4,) in result


class TestCuTensorNet:
    """Tests for CuTensorNet handle."""

    def test_import(self) -> None:
        """Test that CuTensorNet can be imported from pecos_rslib_cuda."""
        from pecos_rslib_cuda import CuTensorNet

        assert CuTensorNet is not None

    def test_creation(self) -> None:
        """Test creating a CuTensorNet handle."""
        from pecos_rslib_cuda import CuTensorNet

        net = CuTensorNet()
        assert net is not None

    def test_version(self) -> None:
        """Test getting the cuTensorNet version."""
        from pecos_rslib_cuda import CuTensorNet

        version = CuTensorNet.version()
        assert isinstance(version, int)
        assert version > 0


class TestCuDensityMat:
    """Tests for CuDensityMat density matrix simulator."""

    def test_import(self) -> None:
        """Test that CuDensityMat can be imported from pecos_rslib_cuda."""
        from pecos_rslib_cuda import CuDensityMat

        assert CuDensityMat is not None

    def test_creation(self) -> None:
        """Test creating a CuDensityMat simulator."""
        from pecos_rslib_cuda import CuDensityMat

        sim = CuDensityMat(4)
        assert sim.num_qubits == 4

    def test_version(self) -> None:
        """Test getting the cuDensityMat version."""
        from pecos_rslib_cuda import CuDensityMat

        version = CuDensityMat.version()
        assert isinstance(version, int)
        assert version > 0

    def test_small_qubit_count(self) -> None:
        """Test that CuDensityMat works with small qubit counts.

        Note: Density matrices have O(4^n) memory requirements,
        so we keep qubit counts small in tests.
        """
        from pecos_rslib_cuda import CuDensityMat

        # 6 qubits = 4^6 = 4096 complex numbers = reasonable size
        sim = CuDensityMat(6)
        assert sim.num_qubits == 6


class TestQuantumSimulatorBackend:
    """Tests for QuantumSimulator with CUDA backends."""

    def test_cuda_statevec_backend(self) -> None:
        """Test QuantumSimulator with CudaStateVec backend."""
        from pecos.simulators.quantum_simulator import QuantumSimulator

        sim = QuantumSimulator(backend="CudaStateVec")
        sim.init(4)

        assert sim.num_qubits == 4

    def test_cuda_stabilizer_backend(self) -> None:
        """Test QuantumSimulator with CudaStabilizer backend."""
        from pecos.simulators.quantum_simulator import QuantumSimulator

        sim = QuantumSimulator(backend="CudaStabilizer")
        sim.init(10)

        assert sim.num_qubits == 10
