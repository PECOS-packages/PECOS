"""Test complex number operations against NumPy."""

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("pecos_rslib") is None:
    pytest.skip("pecos_rslib not available", allow_module_level=True)


class TestComplexScalars:
    """Test complex number operations on scalars."""

    def test_abs_pure_real(self):
        """Test abs on purely real complex number."""
        z = 3.0 + 0j
        np_result = np.abs(z)
        # TODO: Add pecos equivalent when available
        assert np_result == 3.0

    def test_abs_pure_imaginary(self):
        """Test abs on purely imaginary complex number."""
        z = 0 + 4.0j
        np_result = np.abs(z)
        # TODO: Add pecos equivalent when available
        assert np_result == 4.0

    def test_abs_general_complex(self):
        """Test abs on general complex number."""
        z = 3.0 + 4.0j
        np_result = np.abs(z)
        # |3+4i| = sqrt(9+16) = 5
        assert np_result == 5.0

    def test_abs_squared_vs_magnitude_squared(self):
        """Test that |z|² = z * z*."""
        z = 3.0 + 4.0j
        mag_squared = np.abs(z) ** 2
        z_conj_product = z * np.conj(z)
        assert np.isclose(mag_squared, z_conj_product.real)


class TestComplexArrays:
    """Test complex number operations on arrays."""

    def test_abs_array_pure_real(self):
        """Test abs on array of purely real complex numbers."""
        arr = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype=np.complex64)
        np_result = np.abs(arr)
        np.testing.assert_allclose(np_result, [1.0, 2.0, 3.0])

    def test_abs_array_pure_imaginary(self):
        """Test abs on array of purely imaginary complex numbers."""
        arr = np.array([0 + 1.0j, 0 + 2.0j, 0 + 3.0j], dtype=np.complex64)
        np_result = np.abs(arr)
        np.testing.assert_allclose(np_result, [1.0, 2.0, 3.0])

    def test_abs_array_mixed(self):
        """Test abs on array of mixed complex numbers."""
        arr = np.array([3.0 + 4.0j, 5.0 + 12.0j, 0 + 1.0j], dtype=np.complex64)
        np_result = np.abs(arr)
        # |3+4i| = 5, |5+12i| = 13, |i| = 1
        np.testing.assert_allclose(np_result, [5.0, 13.0, 1.0])

    def test_norm_squared_quantum_state(self):
        """Test normalization of quantum state vector."""
        # Normalized state: (|0⟩ + |1⟩)/√2
        state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex64)
        norm_squared = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm_squared, 1.0, atol=1e-7)

    def test_norm_squared_with_phase(self):
        """Test normalization with complex phases."""
        # State with phase: (|0⟩ + i|1⟩)/√2
        state = np.array([1 / np.sqrt(2) + 0j, 0 + 1j / np.sqrt(2)], dtype=np.complex64)
        norm_squared = np.sum(np.abs(state) ** 2)
        assert np.isclose(norm_squared, 1.0, atol=1e-7)

    def test_abs_squared_vs_conj_product(self):
        """Test that |z|² = z * z* element-wise for arrays."""
        arr = np.array([3.0 + 4.0j, 1.0 + 1.0j, 0 + 2.0j], dtype=np.complex64)
        mag_squared = np.abs(arr) ** 2
        conj_product = (arr * np.conj(arr)).real
        np.testing.assert_allclose(mag_squared, conj_product)


class TestComplexArithmetic:
    """Test complex arithmetic operations."""

    def test_power_operation(self):
        """Test power operation on complex numbers."""
        z = 3.0 + 4.0j
        result = z**2
        expected = (3.0 + 4.0j) * (3.0 + 4.0j)
        assert np.isclose(result, expected)

    def test_abs_squared_formula(self):
        """Test that |z|² = a² + b² for z = a + bi."""
        arr = np.array([3.0 + 4.0j, 1.0 + 1.0j, 0 + 2.0j], dtype=np.complex64)
        abs_squared = np.abs(arr) ** 2
        manual = arr.real**2 + arr.imag**2
        np.testing.assert_allclose(abs_squared, manual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
