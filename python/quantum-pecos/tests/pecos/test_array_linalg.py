"""Tests for Array linear algebra operations: conj, T, dot/matmul, kron, expm, matrix_power, diag, iter, eq."""

from __future__ import annotations

import pecos as pc
import pytest

# ---------------------------------------------------------------------------
# conj()
# ---------------------------------------------------------------------------


class TestConj:
    """Test complex conjugation operations."""

    def test_complex128(self) -> None:
        """Test conjugation of complex128 array."""
        a = pc.array([1 + 2j, 3 - 4j], dtype=pc.dtypes.complex128)
        c = a.conj()
        assert pc.isclose(c[0], 1 - 2j)
        assert pc.isclose(c[1], 3 + 4j)

    def test_real_is_identity(self) -> None:
        """Test that conjugation of real array is identity."""
        a = pc.array([1.0, 2.0, 3.0])
        c = a.conj()
        assert (c == a).all()

    def test_2d(self) -> None:
        """Test conjugation of a 2D complex array."""
        a = pc.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=pc.dtypes.complex128)
        c = a.conj()
        assert pc.isclose(c[0][0], 1 - 1j)
        assert pc.isclose(c[1][1], 4 - 4j)


# ---------------------------------------------------------------------------
# .T (transpose)
# ---------------------------------------------------------------------------


class TestTranspose:
    """Test transpose operations on arrays."""

    def test_2x3(self) -> None:
        """Test transpose of a 2x3 matrix."""
        a = pc.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = a.T
        assert t.shape == (3, 2)
        assert pc.isclose(t[0][1], 4.0)
        assert pc.isclose(t[2][0], 3.0)

    def test_complex(self) -> None:
        """Test transpose of a complex matrix."""
        a = pc.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=pc.dtypes.complex128)
        t = a.T
        assert pc.isclose(t[0][1], 3 + 3j)
        assert pc.isclose(t[1][0], 2 + 2j)

    def test_hermitian_conjugate(self) -> None:
        """conj().T should give the Hermitian conjugate (adjoint)."""
        a = pc.array([[0j, 1j], [-1j, 0j]], dtype=pc.dtypes.complex128)
        adj = a.conj().T
        assert pc.isclose(adj[0][1], 1j)
        assert pc.isclose(adj[1][0], -1j)


# ---------------------------------------------------------------------------
# dot() / * (matrix multiplication)
# ---------------------------------------------------------------------------


class TestMatmul:
    """Test matrix multiplication operations."""

    def test_2x2_identity(self) -> None:
        """Test multiplication by identity matrix."""
        I = pc.array([[1.0, 0.0], [0.0, 1.0]])
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        result = I * a
        assert pc.isclose(result[0][0], 1.0)
        assert pc.isclose(result[1][1], 4.0)

    def test_2x2_known(self) -> None:
        """Test 2x2 matrix multiplication with known result."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        b = pc.array([[5.0, 6.0], [7.0, 8.0]])
        c = a.dot(b)
        assert pc.isclose(c[0][0], 19.0)
        assert pc.isclose(c[0][1], 22.0)
        assert pc.isclose(c[1][0], 43.0)
        assert pc.isclose(c[1][1], 50.0)

    def test_complex(self) -> None:
        """Test matrix multiplication of complex matrices."""
        a = pc.array([[1 + 0j, 0j], [0j, 1j]], dtype=pc.dtypes.complex128)
        b = pc.array([[0j, 1 + 0j], [1 + 0j, 0j]], dtype=pc.dtypes.complex128)
        c = a * b
        assert pc.isclose(c[0][0], 0.0)
        assert pc.isclose(c[0][1], 1.0 + 0j)
        assert pc.isclose(c[1][0], 1j)
        assert pc.isclose(c[1][1], 0.0)

    def test_type_promotion_f64_complex(self) -> None:
        """Test type promotion from F64 to Complex128 in matmul."""
        a = pc.array([[1.0, 0.0], [0.0, 1.0]])
        b = pc.array([[1 + 1j, 0j], [0j, 1 - 1j]], dtype=pc.dtypes.complex128)
        c = a * b
        assert pc.isclose(c[0][0], 1 + 1j)
        assert pc.isclose(c[1][1], 1 - 1j)

    def test_mul_vs_dot(self) -> None:
        """Test that * operator and dot method produce identical results."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        b = pc.array([[5.0, 6.0], [7.0, 8.0]])
        c1 = a * b
        c2 = a.dot(b)
        assert (c1 == c2).all()

    def test_non_square(self) -> None:
        """Test matrix multiplication of non-square matrices."""
        a = pc.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        b = pc.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
        c = a * b  # 2x2
        assert c.shape == (2, 2)
        assert pc.isclose(c[0][0], 22.0)  # 1*1 + 2*3 + 3*5

    def test_mul_is_matmul_not_elementwise(self) -> None:
        """Verify * does matrix multiply, not element-wise multiply."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        b = pc.array([[5.0, 6.0], [7.0, 8.0]])
        result = a * b
        # matmul: [[19,22],[43,50]], element-wise would be [[5,12],[21,32]]
        assert pc.isclose(result[0][0], 19.0)
        assert pc.isclose(result[1][1], 50.0)

    def test_scalar_mul_still_scales(self) -> None:
        """Verify scalar * array is still element-wise scaling."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        result = 2.0 * a
        assert pc.isclose(result[0][0], 2.0)
        assert pc.isclose(result[0][1], 4.0)
        assert pc.isclose(result[1][1], 8.0)

    def test_matmul_operator_raises(self) -> None:
        """Verify @ operator gives a helpful error message."""
        a = pc.array([[1.0, 0.0], [0.0, 1.0]])
        b = pc.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(TypeError, match=r"Use \* for matrix multiplication"):
            _ = a @ b


# ---------------------------------------------------------------------------
# elemwise_mul
# ---------------------------------------------------------------------------


class TestElemwiseMul:
    """Test element-wise multiplication operations."""

    def test_basic(self) -> None:
        """Test basic element-wise multiplication of real matrices."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        b = pc.array([[5.0, 6.0], [7.0, 8.0]])
        result = a.elemwise_mul(b)
        assert pc.isclose(result[0][0], 5.0)
        assert pc.isclose(result[0][1], 12.0)
        assert pc.isclose(result[1][0], 21.0)
        assert pc.isclose(result[1][1], 32.0)

    def test_complex(self) -> None:
        """Test element-wise multiplication of complex matrices."""
        a = pc.array([[1 + 1j, 2 + 0j], [0j, 3 - 1j]], dtype=pc.dtypes.complex128)
        b = pc.array([[2 + 0j, 0j], [1j, 1 + 1j]], dtype=pc.dtypes.complex128)
        result = a.elemwise_mul(b)
        assert pc.isclose(result[0][0], 2 + 2j)
        assert pc.isclose(result[0][1], 0j)

    def test_scalar(self) -> None:
        """Test element-wise multiplication by a scalar."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        result = a.elemwise_mul(3.0)
        assert pc.isclose(result[0][0], 3.0)
        assert pc.isclose(result[1][1], 12.0)


# ---------------------------------------------------------------------------
# Complex arithmetic correctness
# ---------------------------------------------------------------------------


class TestComplexArithmetic:
    """Verify complex array operations use proper complex math, not component-wise f64 ops."""

    def test_complex_array_div_complex_array(self) -> None:
        """Test division of two complex arrays."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex128)
        b = pc.array([2 + 0j, 0 + 1j], dtype=pc.dtypes.complex128)
        result = a / b
        assert pc.isclose(result[0], (1 + 2j) / (2 + 0j))
        assert pc.isclose(result[1], (3 + 4j) / (0 + 1j))

    def test_scalar_sub_complex_array(self) -> None:
        """Test real scalar minus complex array."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex128)
        result = 10.0 - a
        assert pc.isclose(result[0], 10 - (1 + 2j))
        assert pc.isclose(result[1], 10 - (3 + 4j))

    def test_scalar_div_complex_array(self) -> None:
        """Test real scalar divided by complex array."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex128)
        result = 1.0 / a
        assert pc.isclose(result[0], 1 / (1 + 2j))
        assert pc.isclose(result[1], 1 / (3 + 4j))

    def test_complex_array_pow_complex_array(self) -> None:
        """Test complex array raised to the power of another complex array."""
        a = pc.array([1 + 2j, 2 + 0j], dtype=pc.dtypes.complex128)
        b = pc.array([2 + 1j, 3 + 0j], dtype=pc.dtypes.complex128)
        result = a**b
        assert pc.isclose(result[0], (1 + 2j) ** (2 + 1j))
        assert pc.isclose(result[1], (2 + 0j) ** (3 + 0j))

    def test_complex_array_pow_scalar(self) -> None:
        """Test complex array raised to a real scalar power."""
        a = pc.array([1 + 2j, 0 + 1j], dtype=pc.dtypes.complex128)
        result = a**2.0
        assert pc.isclose(result[0], (1 + 2j) ** 2)
        assert pc.isclose(result[1], (0 + 1j) ** 2, atol=1e-10)

    def test_neg_complex(self) -> None:
        """Test negation of a complex array."""
        a = pc.array([1 + 2j, -3 + 4j], dtype=pc.dtypes.complex128)
        result = -a
        assert pc.isclose(result[0], -(1 + 2j))
        assert pc.isclose(result[1], -(-3 + 4j))

    def test_neg_real(self) -> None:
        """Test negation of a real array."""
        a = pc.array([1.0, -2.0, 3.0])
        result = -a
        assert pc.isclose(result[0], -1.0)
        assert pc.isclose(result[1], 2.0)
        assert pc.isclose(result[2], -3.0)

    # --- complex scalar reverse ops ---

    def test_complex_scalar_sub_complex_array(self) -> None:
        """Test complex scalar minus complex array."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex128)
        result = (1 + 2j) - a
        assert pc.isclose(result[0], (1 + 2j) - (1 + 2j))
        assert pc.isclose(result[1], (1 + 2j) - (3 + 4j))

    def test_complex_scalar_div_complex_array(self) -> None:
        """Test complex scalar divided by complex array."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex128)
        result = (1 + 0j) / a
        assert pc.isclose(result[0], (1 + 0j) / (1 + 2j))
        assert pc.isclose(result[1], (1 + 0j) / (3 + 4j))

    def test_complex_scalar_add_complex_array(self) -> None:
        """Test complex scalar plus complex array."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex128)
        result = (10 + 5j) + a
        assert pc.isclose(result[0], (10 + 5j) + (1 + 2j))
        assert pc.isclose(result[1], (10 + 5j) + (3 + 4j))

    def test_complex_scalar_mul_f64_array(self) -> None:
        """Complex scalar * F64 array should promote to Complex128."""
        a = pc.array([1.0, 2.0, 3.0])
        result = (1 + 2j) * a
        assert pc.isclose(result[0], (1 + 2j) * 1.0)
        assert pc.isclose(result[1], (1 + 2j) * 2.0)

    def test_f64_array_add_complex_scalar(self) -> None:
        """F64 array + complex scalar should promote to Complex128."""
        a = pc.array([1.0, 2.0, 3.0])
        result = a + (1 + 2j)
        assert pc.isclose(result[0], 1.0 + (1 + 2j))
        assert pc.isclose(result[1], 2.0 + (1 + 2j))

    def test_complex_scalar_sub_f64_array(self) -> None:
        """Complex scalar - F64 array should promote to Complex128."""
        a = pc.array([1.0, 2.0])
        result = (5 + 3j) - a
        assert pc.isclose(result[0], (5 + 3j) - 1.0)
        assert pc.isclose(result[1], (5 + 3j) - 2.0)

    # --- cross-type array-array ops ---

    def test_f64_array_add_complex_array(self) -> None:
        """F64 + Complex128 array should promote to Complex128."""
        a = pc.array([1.0, 2.0])
        b = pc.array([1 + 1j, 2 + 2j], dtype=pc.dtypes.complex128)
        result = a + b
        assert pc.isclose(result[0], 1.0 + (1 + 1j))
        assert pc.isclose(result[1], 2.0 + (2 + 2j))

    def test_complex_array_sub_f64_array(self) -> None:
        """Complex128 - F64 array should work."""
        a = pc.array([10 + 5j, 20 + 10j], dtype=pc.dtypes.complex128)
        b = pc.array([1.0, 2.0])
        result = a - b
        assert pc.isclose(result[0], (10 + 5j) - 1.0)
        assert pc.isclose(result[1], (20 + 10j) - 2.0)

    def test_f64_array_mul_complex_array(self) -> None:
        """F64 .elemwise_mul Complex128 should promote to Complex128."""
        a = pc.array([2.0, 3.0])
        b = pc.array([1 + 1j, 2 + 2j], dtype=pc.dtypes.complex128)
        result = a.elemwise_mul(b)
        assert pc.isclose(result[0], 2.0 * (1 + 1j))
        assert pc.isclose(result[1], 3.0 * (2 + 2j))


# ---------------------------------------------------------------------------
# __iter__
# ---------------------------------------------------------------------------


class TestIter:
    """Test array iteration operations."""

    def test_iterate_1d(self) -> None:
        """Test iterating over a 1D array."""
        a = pc.array([10.0, 20.0, 30.0])
        vals = list(a)
        assert len(vals) == 3
        assert pc.isclose(vals[1], 20.0)

    def test_unpack(self) -> None:
        """Test unpacking a 1D array into variables."""
        a = pc.array([1.0, 2.0, 3.0])
        x, y, z = a
        assert pc.isclose(x, 1.0)
        assert pc.isclose(y, 2.0)
        assert pc.isclose(z, 3.0)

    def test_iterate_2d_rows(self) -> None:
        """Test iterating over rows of a 2D array."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        rows = list(a)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# __eq__ / __ne__ array-vs-array
# ---------------------------------------------------------------------------


class TestArrayEquality:
    """Test array equality and inequality comparisons."""

    def test_equal_f64(self) -> None:
        """Test equality of identical F64 arrays."""
        a = pc.array([1.0, 2.0, 3.0])
        b = pc.array([1.0, 2.0, 3.0])
        assert (a == b).all()

    def test_not_equal_f64(self) -> None:
        """Test that differing F64 arrays are not equal."""
        a = pc.array([1.0, 2.0, 3.0])
        b = pc.array([1.0, 0.0, 3.0])
        eq = a == b
        assert not eq.all()

    def test_equal_complex(self) -> None:
        """Test equality of identical complex arrays."""
        a = pc.array([1 + 1j, 2 + 2j], dtype=pc.dtypes.complex128)
        b = pc.array([1 + 1j, 2 + 2j], dtype=pc.dtypes.complex128)
        assert (a == b).all()

    def test_ne_complex(self) -> None:
        """Test inequality operator on complex arrays."""
        a = pc.array([1 + 1j, 2 + 2j], dtype=pc.dtypes.complex128)
        b = pc.array([1 + 1j, 0j], dtype=pc.dtypes.complex128)
        ne = a != b
        # First element should be False (they're equal), second True
        assert not ne[0]
        assert ne[1]


# ---------------------------------------------------------------------------
# kron()
# ---------------------------------------------------------------------------


class TestKron:
    """Test Kronecker product operations."""

    def test_identity_kron(self) -> None:
        """Test Kronecker product of two identity matrices."""
        I = pc.array([[1.0, 0.0], [0.0, 1.0]])
        result = pc.kron(I, I)
        assert result.shape == (4, 4)
        assert pc.isclose(result[0][0], 1.0)
        assert pc.isclose(result[0][1], 0.0)
        assert pc.isclose(result[3][3], 1.0)

    def test_known_result(self) -> None:
        """kron([[1,0],[0,0]], [[0,1],[1,0]]) = [[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]."""
        a = pc.array([[1.0, 0.0], [0.0, 0.0]])
        b = pc.array([[0.0, 1.0], [1.0, 0.0]])
        result = pc.kron(a, b)
        assert pc.isclose(result[0][1], 1.0)
        assert pc.isclose(result[1][0], 1.0)
        assert pc.isclose(result[2][2], 0.0)

    def test_complex(self) -> None:
        """Test Kronecker product of complex matrices."""
        I = pc.array([[1 + 0j, 0j], [0j, 1 + 0j]], dtype=pc.dtypes.complex128)
        X = pc.array([[0j, 1 + 0j], [1 + 0j, 0j]], dtype=pc.dtypes.complex128)
        IX = pc.kron(I, X)
        assert IX.shape == (4, 4)
        assert pc.isclose(IX[0][1], 1 + 0j)
        assert pc.isclose(IX[2][3], 1 + 0j)

    def test_non_square(self) -> None:
        """Test Kronecker product of non-square matrices."""
        a = pc.array([[1.0, 2.0, 3.0]])  # 1x3
        b = pc.array([[1.0], [2.0]])  # 2x1
        result = pc.kron(a, b)
        assert result.shape == (2, 3)

    def test_and_operator(self) -> None:
        """A & b should give the same result as pc.kron(a, b)."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        b = pc.array([[5.0, 6.0], [7.0, 8.0]])
        assert (a & b == pc.kron(a, b)).all()

    def test_and_operator_complex(self) -> None:
        """Test & operator for Kronecker product with complex matrices."""
        I = pc.array([[1 + 0j, 0j], [0j, 1 + 0j]], dtype=pc.dtypes.complex128)
        X = pc.array([[0j, 1 + 0j], [1 + 0j, 0j]], dtype=pc.dtypes.complex128)
        assert (pc.kron(I, X) == I & X).all()

    def test_and_not_commutative(self) -> None:
        """A & b != b & a for non-trivial inputs."""
        a = pc.array([[1.0, 0.0], [0.0, 0.0]])
        b = pc.array([[0.0, 1.0], [1.0, 0.0]])
        ab = a & b
        ba = b & a
        assert not (ab == ba).all()


# ---------------------------------------------------------------------------
# expm()
# ---------------------------------------------------------------------------


class TestExpm:
    """Test matrix exponential operations."""

    def test_zero_matrix(self) -> None:
        """expm(0) = I."""
        Z = pc.array([[0j, 0j], [0j, 0j]], dtype=pc.dtypes.complex128)
        result = pc.linalg.expm(Z)
        assert pc.isclose(result[0][0], 1 + 0j)
        assert pc.isclose(result[0][1], 0j)
        assert pc.isclose(result[1][0], 0j)
        assert pc.isclose(result[1][1], 1 + 0j)

    def test_diagonal(self) -> None:
        """expm(diag(a,b)) = diag(exp(a), exp(b))."""
        a = pc.array([[1 + 0j, 0j], [0j, 2 + 0j]], dtype=pc.dtypes.complex128)
        result = pc.linalg.expm(a)
        assert pc.isclose(result[0][0], pc.exp(1.0 + 0j))
        assert pc.isclose(result[1][1], pc.exp(2.0 + 0j))
        assert pc.isclose(result[0][1], 0j, atol=1e-10)

    def test_f64_promotion(self) -> None:
        """F64 input should be auto-promoted to Complex128."""
        a = pc.array([[0.0, 0.0], [0.0, 0.0]])
        result = pc.linalg.expm(a)
        assert pc.isclose(result[0][0], 1 + 0j)


# ---------------------------------------------------------------------------
# matrix_power()
# ---------------------------------------------------------------------------


class TestMatrixPower:
    """Test matrix power operations."""

    def test_power_zero_is_identity(self) -> None:
        """Test that matrix to the power of zero gives identity."""
        a = pc.array([[2.0, 1.0], [0.0, 3.0]])
        result = pc.linalg.matrix_power(a, 0)
        assert pc.isclose(result[0][0], 1.0)
        assert pc.isclose(result[0][1], 0.0)
        assert pc.isclose(result[1][0], 0.0)
        assert pc.isclose(result[1][1], 1.0)

    def test_power_one(self) -> None:
        """Test that matrix to the power of one returns the original matrix."""
        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        result = pc.linalg.matrix_power(a, 1)
        assert (result == a).all()

    def test_power_two(self) -> None:
        """Test matrix squared with a known result."""
        a = pc.array([[1.0, 1.0], [0.0, 1.0]])
        result = pc.linalg.matrix_power(a, 2)
        # [[1,1],[0,1]]^2 = [[1,2],[0,1]]
        assert pc.isclose(result[0][0], 1.0)
        assert pc.isclose(result[0][1], 2.0)
        assert pc.isclose(result[1][1], 1.0)

    def test_complex(self) -> None:
        """Test matrix power with complex identity matrix."""
        I = pc.array([[1 + 0j, 0j], [0j, 1 + 0j]], dtype=pc.dtypes.complex128)
        result = pc.linalg.matrix_power(I, 5)
        assert pc.isclose(result[0][0], 1 + 0j)
        assert pc.isclose(result[1][1], 1 + 0j)

    def test_pauli_x_squared(self) -> None:
        """Test that Pauli X squared equals the identity matrix."""
        X = pc.array([[0j, 1 + 0j], [1 + 0j, 0j]], dtype=pc.dtypes.complex128)
        I = pc.array([[1 + 0j, 0j], [0j, 1 + 0j]], dtype=pc.dtypes.complex128)
        result = pc.linalg.matrix_power(X, 2)
        assert pc.isclose(result, I).all()


# ---------------------------------------------------------------------------
# diag() - bidirectional
# ---------------------------------------------------------------------------


class TestDiag:
    """Test diagonal matrix creation and extraction."""

    def test_1d_to_matrix_f64(self) -> None:
        """Test creating a diagonal matrix from a 1D F64 array."""
        v = pc.array([1.0, 2.0, 3.0])
        m = pc.diag(v)
        assert m.shape == (3, 3)
        assert pc.isclose(m[0][0], 1.0)
        assert pc.isclose(m[1][1], 2.0)
        assert pc.isclose(m[2][2], 3.0)
        assert pc.isclose(m[0][1], 0.0)

    def test_1d_to_matrix_complex(self) -> None:
        """Test creating a diagonal matrix from a 1D complex array."""
        v = pc.array([1 + 0j, 0 + 1j], dtype=pc.dtypes.complex128)
        m = pc.diag(v)
        assert m.shape == (2, 2)
        assert pc.isclose(m[0][0], 1 + 0j)
        assert pc.isclose(m[1][1], 1j)
        assert pc.isclose(m[0][1], 0j)

    def test_2d_to_diagonal_f64(self) -> None:
        """Test extracting the diagonal from a 2D F64 matrix."""
        m = pc.array([[1.0, 2.0], [3.0, 4.0]])
        d = pc.diag(m)
        assert d.shape == (2,)
        assert pc.isclose(d[0], 1.0)
        assert pc.isclose(d[1], 4.0)

    def test_2d_to_diagonal_complex(self) -> None:
        """Test extracting the diagonal from a 2D complex matrix."""
        m = pc.array([[1 + 1j, 2 + 0j], [0j, 3 - 1j]], dtype=pc.dtypes.complex128)
        d = pc.diag(m)
        assert pc.isclose(d[0], 1 + 1j)
        assert pc.isclose(d[1], 3 - 1j)

    def test_round_trip_f64(self) -> None:
        """diag(diag(v)) should give back the original values on the diagonal."""
        v = pc.array([5.0, 10.0, 15.0])
        d = pc.diag(pc.diag(v))
        assert pc.isclose(d[0], 5.0)
        assert pc.isclose(d[1], 10.0)
        assert pc.isclose(d[2], 15.0)


# ---------------------------------------------------------------------------
# random() with optional size
# ---------------------------------------------------------------------------


class TestRandom:
    """Test random number generation."""

    def test_no_args_returns_scalar(self) -> None:
        """Test that random() with no arguments returns a float scalar."""
        val = pc.random.random()
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_with_size_returns_array(self) -> None:
        """Test that random() with a size argument returns an array."""
        arr = pc.random.random(5)
        assert arr.shape == (5,)

    def test_with_size_1(self) -> None:
        """Test that random() with size 1 returns a single-element array."""
        arr = pc.random.random(1)
        assert arr.shape == (1,)


# ---------------------------------------------------------------------------
# linalg.norm()
# ---------------------------------------------------------------------------


class TestLinalgNorm:
    """Test linear algebra norm operations."""

    def test_f64_vector_l2(self) -> None:
        """Default L2 norm of a real vector."""
        a = pc.array([3.0, 4.0])
        result = pc.linalg.norm(a)
        assert abs(result - 5.0) < 1e-12

    def test_f64_vector_l1(self) -> None:
        """L1 norm (sum of absolute values)."""
        a = pc.array([3.0, -4.0])
        result = pc.linalg.norm(a, ord=1.0)
        assert abs(result - 7.0) < 1e-12

    def test_f64_vector_linf(self) -> None:
        """L-inf norm (max absolute value)."""
        a = pc.array([3.0, -4.0, 2.0])
        # ord=inf maps to float('inf')
        result = pc.linalg.norm(a, ord=float("inf"))
        assert abs(result - 4.0) < 1e-12

    def test_complex128_vector(self) -> None:
        """L2 norm of a complex vector: sqrt(|z1|^2 + |z2|^2)."""
        a = pc.array([3 + 4j, 0 + 0j], dtype=pc.dtypes.complex128)
        result = pc.linalg.norm(a)
        assert abs(result - 5.0) < 1e-12

    def test_complex128_bell_state(self) -> None:
        """Norm of a Bell state should be 1."""
        import math

        s = 1.0 / math.sqrt(2)
        bell = pc.array([s + 0j, 0j, 0j, s + 0j], dtype=pc.dtypes.complex128)
        result = pc.linalg.norm(bell)
        assert abs(result - 1.0) < 1e-12

    def test_complex64_vector(self) -> None:
        """Norm with Complex64 dtype."""
        a = pc.array([3 + 4j], dtype=pc.dtypes.complex64)
        result = pc.linalg.norm(a)
        assert abs(result - 5.0) < 1e-5  # f32 precision

    def test_f64_matrix_frobenius(self) -> None:
        """Frobenius norm of a 2x2 matrix."""
        import math

        a = pc.array([[1.0, 2.0], [3.0, 4.0]])
        result = pc.linalg.norm(a)
        expected = math.sqrt(1 + 4 + 9 + 16)
        assert abs(result - expected) < 1e-12

    def test_i64_vector(self) -> None:
        """Norm works on integer arrays."""
        a = pc.array([3, 4], dtype=pc.dtypes.int64)
        result = pc.linalg.norm(a)
        assert abs(result - 5.0) < 1e-12

    def test_unit_vector(self) -> None:
        """Norm of a unit vector is 1."""
        a = pc.array([0.0, 1.0, 0.0])
        assert abs(pc.linalg.norm(a) - 1.0) < 1e-12

    def test_zero_vector(self) -> None:
        """Norm of a zero vector is 0."""
        a = pc.array([0.0, 0.0, 0.0])
        assert abs(pc.linalg.norm(a)) < 1e-12


# ---------------------------------------------------------------------------
# Complex64 array-array operations
# ---------------------------------------------------------------------------


class TestComplex64ArrayArray:
    """Verify Complex64-Complex64 array-array operations work correctly."""

    def test_add(self) -> None:
        """Test addition of two Complex64 arrays."""
        a = pc.array([1 + 2j, 3 + 4j], dtype=pc.dtypes.complex64)
        b = pc.array([10 + 0j, 0 + 10j], dtype=pc.dtypes.complex64)
        result = a + b
        assert pc.isclose(result[0], (1 + 2j) + (10 + 0j))
        assert pc.isclose(result[1], (3 + 4j) + (0 + 10j))

    def test_subtract(self) -> None:
        """Test subtraction of two Complex64 arrays."""
        a = pc.array([10 + 5j, 20 + 10j], dtype=pc.dtypes.complex64)
        b = pc.array([1 + 1j, 2 + 2j], dtype=pc.dtypes.complex64)
        result = a - b
        assert pc.isclose(result[0], (10 + 5j) - (1 + 1j))
        assert pc.isclose(result[1], (20 + 10j) - (2 + 2j))

    def test_elemwise_mul(self) -> None:
        """Test element-wise multiplication of two Complex64 arrays."""
        a = pc.array([1 + 1j, 2 + 0j], dtype=pc.dtypes.complex64)
        b = pc.array([2 + 0j, 0 + 3j], dtype=pc.dtypes.complex64)
        result = a.elemwise_mul(b)
        assert pc.isclose(result[0], (1 + 1j) * (2 + 0j))
        assert pc.isclose(result[1], (2 + 0j) * (0 + 3j))

    def test_divide(self) -> None:
        """Test division of two Complex64 arrays."""
        a = pc.array([4 + 2j, 6 + 0j], dtype=pc.dtypes.complex64)
        b = pc.array([2 + 0j, 0 + 3j], dtype=pc.dtypes.complex64)
        result = a / b
        assert pc.isclose(result[0], (4 + 2j) / (2 + 0j))
        assert pc.isclose(result[1], (6 + 0j) / (0 + 3j))

    def test_power(self) -> None:
        """Test exponentiation of two Complex64 arrays."""
        a = pc.array([2 + 0j, 0 + 1j], dtype=pc.dtypes.complex64)
        b = pc.array([3 + 0j, 2 + 0j], dtype=pc.dtypes.complex64)
        result = a**b
        assert pc.isclose(result[0], (2 + 0j) ** (3 + 0j))
        assert pc.isclose(result[1], (0 + 1j) ** (2 + 0j), atol=1e-6)

    def test_preserves_dtype(self) -> None:
        """Result should still be complex64."""
        a = pc.array([1 + 0j, 2 + 0j], dtype=pc.dtypes.complex64)
        b = pc.array([3 + 0j, 4 + 0j], dtype=pc.dtypes.complex64)
        result = a + b
        assert result.dtype == pc.dtypes.complex64


# ---------------------------------------------------------------------------
# Cross-type array-array: division and power
# ---------------------------------------------------------------------------


class TestCrossTypeArrayOps:
    """Test F64 / Complex128 cross-type operations beyond +, -, elemwise_mul."""

    def test_f64_div_complex128(self) -> None:
        """Test F64 array divided by Complex128 array."""
        a = pc.array([10.0, 20.0])
        b = pc.array([2 + 0j, 0 + 4j], dtype=pc.dtypes.complex128)
        result = a / b
        assert pc.isclose(result[0], 10.0 / (2 + 0j))
        assert pc.isclose(result[1], 20.0 / (0 + 4j))

    def test_complex128_div_f64(self) -> None:
        """Test Complex128 array divided by F64 array."""
        a = pc.array([10 + 5j, 20 + 10j], dtype=pc.dtypes.complex128)
        b = pc.array([2.0, 4.0])
        result = a / b
        assert pc.isclose(result[0], (10 + 5j) / 2.0)
        assert pc.isclose(result[1], (20 + 10j) / 4.0)

    def test_f64_pow_complex128(self) -> None:
        """Test F64 array raised to Complex128 array power."""
        a = pc.array([2.0, 3.0])
        b = pc.array([1 + 0j, 2 + 0j], dtype=pc.dtypes.complex128)
        result = a**b
        assert pc.isclose(result[0], 2.0 ** (1 + 0j))
        assert pc.isclose(result[1], 3.0 ** (2 + 0j))

    def test_complex128_pow_f64(self) -> None:
        """Test Complex128 array raised to F64 array power."""
        a = pc.array([1 + 1j, 2 + 0j], dtype=pc.dtypes.complex128)
        b = pc.array([2.0, 3.0])
        result = a**b
        assert pc.isclose(result[0], (1 + 1j) ** (2 + 0j))
        assert pc.isclose(result[1], (2 + 0j) ** (3 + 0j))

    def test_cross_type_result_is_complex128(self) -> None:
        """F64 op Complex128 should always produce Complex128."""
        a = pc.array([1.0, 2.0])
        b = pc.array([1 + 0j, 2 + 0j], dtype=pc.dtypes.complex128)
        result = a + b
        assert result.dtype == pc.dtypes.complex128


# ---------------------------------------------------------------------------
# Broadcasting with cross-type arrays
# ---------------------------------------------------------------------------


class TestCrossTypeBroadcasting:
    """Test broadcasting between F64 and Complex128 arrays of different shapes."""

    def test_f64_column_plus_complex_row(self) -> None:
        """(3,1) F64 + (4,) Complex128 -> (3,4) Complex128."""
        a = pc.array([[1.0], [2.0], [3.0]])  # (3,1)
        b = pc.array([10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j], dtype=pc.dtypes.complex128)  # (4,)
        result = a + b
        assert result.shape == (3, 4)
        assert pc.isclose(result[0][0], 1.0 + (10 + 1j))
        assert pc.isclose(result[2][3], 3.0 + (40 + 4j))

    def test_complex_matrix_minus_f64_row(self) -> None:
        """(2,3) Complex128 - (3,) F64 -> (2,3) Complex128."""
        a = pc.array([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]], dtype=pc.dtypes.complex128)
        b = pc.array([1.0, 2.0, 3.0])
        result = a - b
        assert result.shape == (2, 3)
        assert pc.isclose(result[0][0], (1 + 1j) - 1.0)
        assert pc.isclose(result[1][2], (6 + 6j) - 3.0)

    def test_f64_scalar_broadcast_to_complex(self) -> None:
        """(1,1) F64 + (2,2) Complex128 -> (2,2) Complex128."""
        a = pc.array([[5.0]])
        b = pc.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=pc.dtypes.complex128)
        result = a + b
        assert result.shape == (2, 2)
        assert pc.isclose(result[0][0], 5.0 + (1 + 1j))
        assert pc.isclose(result[1][1], 5.0 + (4 + 4j))

    def test_cross_type_elemwise_mul_broadcast(self) -> None:
        """(3,1) F64 .elemwise_mul (1,3) Complex128 -> (3,3)."""
        a = pc.array([[2.0], [3.0], [4.0]])  # (3,1)
        b = pc.array([[1 + 1j, 2 + 0j, 0 + 3j]], dtype=pc.dtypes.complex128)  # (1,3)
        result = a.elemwise_mul(b)
        assert result.shape == (3, 3)
        assert pc.isclose(result[0][0], 2.0 * (1 + 1j))
        assert pc.isclose(result[2][2], 4.0 * (0 + 3j))


# ---------------------------------------------------------------------------
# Complex scalar reverse: * and **
# ---------------------------------------------------------------------------


class TestComplexScalarReverseOps:
    """Test complex_scalar * array and complex_scalar ** array reverse ops."""

    def test_complex_scalar_mul_complex_array(self) -> None:
        """Test complex scalar multiplied by complex array."""
        a = pc.array([1 + 1j, 2 + 0j], dtype=pc.dtypes.complex128)
        result = (2 + 3j) * a
        assert pc.isclose(result[0], (2 + 3j) * (1 + 1j))
        assert pc.isclose(result[1], (2 + 3j) * (2 + 0j))

    def test_complex_scalar_pow_complex_array(self) -> None:
        """Test complex scalar raised to complex array power."""
        a = pc.array([2 + 0j, 1 + 0j], dtype=pc.dtypes.complex128)
        result = (2 + 0j) ** a
        assert pc.isclose(result[0], (2 + 0j) ** (2 + 0j))
        assert pc.isclose(result[1], (2 + 0j) ** (1 + 0j))

    def test_complex_scalar_mul_f64_array_reverse(self) -> None:
        """(1+2j) * f64_array via __rmul__."""
        a = pc.array([1.0, 2.0, 3.0])
        result = (1 + 2j) * a
        assert pc.isclose(result[0], (1 + 2j) * 1.0)
        assert pc.isclose(result[1], (1 + 2j) * 2.0)
        assert pc.isclose(result[2], (1 + 2j) * 3.0)

    def test_complex_scalar_pow_f64_array(self) -> None:
        """(2+1j) ** f64_array via __rpow__."""
        a = pc.array([2.0, 3.0])
        result = (2 + 1j) ** a
        assert pc.isclose(result[0], (2 + 1j) ** 2.0)
        assert pc.isclose(result[1], (2 + 1j) ** 3.0)

    def test_complex_scalar_div_f64_array(self) -> None:
        """(1+0j) / f64_array via __rtruediv__."""
        a = pc.array([2.0, 4.0])
        result = (1 + 0j) / a
        assert pc.isclose(result[0], (1 + 0j) / 2.0)
        assert pc.isclose(result[1], (1 + 0j) / 4.0)
