"""Comparison tests for diag() and expm() against numpy/scipy references.

These tests verify that our Rust implementations produce results that match
numpy.diag and scipy.linalg.expm within reasonable numerical tolerances.
"""

import pytest

pytest.importorskip("numpy")

import numpy as np

from pecos_rslib import Array, num

pytestmark = pytest.mark.numpy


# ---------------------------------------------------------------------------
# diag() -- compare against numpy.diag
# ---------------------------------------------------------------------------


class TestDiagNumpyComparison:
    """Compare pecos_rslib diag() against numpy.diag()."""

    def test_1d_to_2d_f64(self) -> None:
        """1D -> diagonal matrix, f64."""
        v = np.array([1.0, 2.0, 3.0])
        np_result = np.diag(v)
        pecos_result = np.asarray(num.diag(Array(v)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_1d_to_2d_complex(self) -> None:
        """1D -> diagonal matrix, complex128."""
        v = np.array([1 + 2j, 3 - 4j, 5 + 0j], dtype=np.complex128)
        np_result = np.diag(v)
        pecos_result = np.asarray(num.diag(Array(v)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_2d_to_1d_f64(self) -> None:
        """2D -> extract diagonal, f64."""
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np_result = np.diag(m)
        pecos_result = np.asarray(num.diag(Array(m)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_2d_to_1d_complex(self) -> None:
        """2D -> extract diagonal, complex128."""
        m = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=np.complex128)
        np_result = np.diag(m)
        pecos_result = np.asarray(num.diag(Array(m)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_rectangular_more_rows(self) -> None:
        """Extract diagonal from tall rectangular matrix."""
        m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np_result = np.diag(m)
        pecos_result = np.asarray(num.diag(Array(m)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_rectangular_more_cols(self) -> None:
        """Extract diagonal from wide rectangular matrix."""
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np_result = np.diag(m)
        pecos_result = np.asarray(num.diag(Array(m)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_round_trip_f64(self) -> None:
        """diag(diag(v)) should match numpy's round-trip."""
        v = np.array([7.0, -3.0, 0.0, 2.5])
        np_result = np.diag(np.diag(v))
        pecos_result = np.asarray(num.diag(num.diag(Array(v))))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_round_trip_complex(self) -> None:
        """diag(diag(v)) round-trip for complex."""
        v = np.array([1 + 2j, 0 + 0j, -1 - 1j], dtype=np.complex128)
        np_result = np.diag(np.diag(v))
        pecos_result = np.asarray(num.diag(num.diag(Array(v))))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_single_element(self) -> None:
        """1-element vector produces 1x1 matrix."""
        v = np.array([42.0])
        np_result = np.diag(v)
        pecos_result = np.asarray(num.diag(Array(v)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_identity_diagonal(self) -> None:
        """Diagonal of identity matrix should be all ones."""
        m = np.eye(4)
        np_result = np.diag(m)
        pecos_result = np.asarray(num.diag(Array(m)))
        np.testing.assert_array_equal(pecos_result, np_result)

    def test_1d_to_2d_off_diagonal_zeros(self) -> None:
        """Verify all off-diagonal elements are exactly zero."""
        v = np.array([10.0, 20.0, 30.0])
        np_result = np.diag(v)
        pecos_result = np.asarray(num.diag(Array(v)))
        np.testing.assert_array_equal(pecos_result, np_result)


# ---------------------------------------------------------------------------
# expm() -- compare against scipy.linalg.expm
# ---------------------------------------------------------------------------


class TestExpmScipyComparison:
    """Compare pecos_rslib expm() against scipy.linalg.expm()."""

    @pytest.fixture(autouse=True)
    def _import_scipy(self):
        scipy_linalg = pytest.importorskip("scipy.linalg")
        self.scipy_expm = scipy_linalg.expm

    def test_zero_matrix(self) -> None:
        """expm(0) = I."""
        m = np.zeros((3, 3), dtype=np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(m))),
            self.scipy_expm(m),
            atol=1e-12,
        )

    def test_identity_matrix(self) -> None:
        """expm(I) = e*I (diagonal)."""
        m = np.eye(3, dtype=np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(m))),
            self.scipy_expm(m),
            atol=1e-10,
        )

    def test_diagonal_matrix(self) -> None:
        """expm(diag(a,b,c)) = diag(exp(a), exp(b), exp(c))."""
        m = np.diag([1.0 + 0j, 2.0 + 0j, -1.0 + 0j]).astype(np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(m))),
            self.scipy_expm(m),
            atol=1e-10,
        )

    def test_nilpotent_matrix(self) -> None:
        """Nilpotent: [[0,1],[0,0]] -> exp = [[1,1],[0,1]]."""
        m = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(m))),
            self.scipy_expm(m),
            atol=1e-10,
        )

    def test_pauli_x(self) -> None:
        """expm of Pauli X gate."""
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(X))),
            self.scipy_expm(X),
            atol=1e-10,
        )

    def test_pauli_z(self) -> None:
        """expm of Pauli Z gate."""
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(Z))),
            self.scipy_expm(Z),
            atol=1e-10,
        )

    def test_imaginary_diagonal(self) -> None:
        """expm(i*diag(theta1, theta2)) -- typical quantum phase gate."""
        m = np.diag([0.5j, -0.5j]).astype(np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(m))),
            self.scipy_expm(m),
            atol=1e-10,
        )

    def test_hermitian_matrix(self) -> None:
        """expm of a Hermitian matrix (common in quantum)."""
        H = np.array([[2.0, 1 + 1j], [1 - 1j, 3.0]], dtype=np.complex128)
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(H))),
            self.scipy_expm(H),
            atol=1e-8,
        )

    def test_skew_hermitian(self) -> None:
        """expm(i*H) where H is Hermitian -- produces unitary matrix."""
        H = np.array([[1.0, 0.5 + 0.5j], [0.5 - 0.5j, 2.0]], dtype=np.complex128)
        m = 1j * H
        result = np.asarray(num.linalg.expm(Array(m)))
        expected = self.scipy_expm(m)
        np.testing.assert_allclose(result, expected, atol=1e-8)
        # Verify result is unitary: U @ U^dagger = I
        product = result @ result.conj().T
        np.testing.assert_allclose(product, np.eye(2), atol=1e-8)

    def test_f64_promotion(self) -> None:
        """F64 input should produce the same result as complex input."""
        m_f64 = np.array([[0.0, 1.0], [0.0, 0.0]])
        m_c128 = m_f64.astype(np.complex128)
        result_f64 = np.asarray(num.linalg.expm(Array(m_f64)))
        result_c128 = np.asarray(num.linalg.expm(Array(m_c128)))
        np.testing.assert_allclose(result_f64, result_c128, atol=1e-14)

    def test_larger_matrix(self) -> None:
        """4x4 matrix comparison."""
        rng = np.random.default_rng(42)
        m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        # Scale down for numerical stability
        m = m * 0.5
        np.testing.assert_allclose(
            np.asarray(num.linalg.expm(Array(m))),
            self.scipy_expm(m),
            atol=1e-6,
        )
