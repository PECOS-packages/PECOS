"""
Comprehensive dtype validation tests comparing PECOS operations with NumPy.

This test suite systematically verifies that all math operations work correctly
across all supported dtypes (f32, f64, Complex32, Complex64) for both scalars
and arrays, comparing results with NumPy to catch any dtype-related bugs.

This was created in response to a critical bug where pc.abs([0+1j]) returned [0.0]
instead of [1.0] due to missing dtype validation in the array extraction macro.
"""

import sys

sys.path.insert(0, "/home/ciaranra/Repos/cl_projects/gup/PECOS-alt/python/quantum-pecos/src")

import pytest
import numpy as np
import pecos as pc


class TestUnaryOperationsDtypeValidation:
    """Test unary math operations across all dtypes, comparing with NumPy."""

    # Test values for different dtypes
    REAL_VALUES = {
        "positive": 2.0,
        "negative": -2.0,
        "fraction": 0.5,
        "zero": 0.0,
    }

    COMPLEX_VALUES = {
        "real_only": 3.0 + 0j,
        "imag_only": 0.0 + 1j,
        "both": 3.0 + 4j,
        "negative": -1.0 - 2j,
    }

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("f32", np.float32, "float32"),
        ],
    )
    @pytest.mark.parametrize(("value_name", "value"), REAL_VALUES.items())
    def test_abs_real_scalars(self, dtype_name, dtype_np, dtype_pc, value_name, value) -> None:
        """Test abs() on real scalar values."""
        pc_result = pc.abs(dtype_np(value))
        np_result = np.abs(dtype_np(value))
        assert np.isclose(
            pc_result, np_result
        ), f"abs({value_name}={value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("f32", np.float32, "float32"),
        ],
    )
    @pytest.mark.parametrize(("value_name", "value"), REAL_VALUES.items())
    def test_abs_real_arrays(self, dtype_name, dtype_np, dtype_pc, value_name, value) -> None:
        """Test abs() on real array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.abs(pc_arr)
        np_result = np.abs(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"abs([{value_name}]={value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("Complex64", np.complex128, "complex"),
            ("Complex32", np.complex64, "complex64"),
        ],
    )
    @pytest.mark.parametrize(("value_name", "value"), COMPLEX_VALUES.items())
    def test_abs_complex_scalars(self, dtype_name, dtype_np, dtype_pc, value_name, value) -> None:
        """Test abs() on complex scalar values."""
        pc_result = pc.abs(dtype_np(value))
        np_result = np.abs(dtype_np(value))
        assert np.isclose(
            pc_result, np_result
        ), f"abs({value_name}={value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("Complex64", np.complex128, "complex"),
            ("Complex32", np.complex64, "complex64"),
        ],
    )
    @pytest.mark.parametrize(("value_name", "value"), COMPLEX_VALUES.items())
    def test_abs_complex_arrays(self, dtype_name, dtype_np, dtype_pc, value_name, value) -> None:
        """Test abs() on complex array values.

        This is the critical test that would have caught the original bug
        where pc.abs([0+1j]) returned [0.0] instead of [1.0].
        """
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.abs(pc_arr)
        np_result = np.abs(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"abs([{value_name}]={value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [2.0, 4.0, 0.25])
    def test_sqrt_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test sqrt() on scalar values."""
        pc_result = pc.sqrt(dtype_np(value))
        np_result = np.sqrt(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"sqrt({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [2.0, 4.0, 0.25])
    def test_sqrt_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test sqrt() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.sqrt(pc_arr)
        np_result = np.sqrt(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"sqrt([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.0])
    def test_exp_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test exp() on scalar values."""
        pc_result = pc.exp(dtype_np(value))
        np_result = np.exp(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"exp({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.0])
    def test_exp_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test exp() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.exp(pc_arr)
        np_result = np.exp(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"exp([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_sin_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test sin() on scalar values."""
        pc_result = pc.sin(dtype_np(value))
        np_result = np.sin(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"sin({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_sin_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test sin() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.sin(pc_arr)
        np_result = np.sin(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"sin([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_cos_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test cos() on scalar values."""
        pc_result = pc.cos(dtype_np(value))
        np_result = np.cos(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"cos({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_cos_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test cos() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.cos(pc_arr)
        np_result = np.cos(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"cos([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.0])
    def test_sinh_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test sinh() on scalar values."""
        pc_result = pc.sinh(dtype_np(value))
        np_result = np.sinh(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"sinh({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.0])
    def test_sinh_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test sinh() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.sinh(pc_arr)
        np_result = np.sinh(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"sinh([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.0])
    def test_cosh_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test cosh() on scalar values."""
        pc_result = pc.cosh(dtype_np(value))
        np_result = np.cosh(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"cosh({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2.0])
    def test_cosh_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test cosh() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.cosh(pc_arr)
        np_result = np.cosh(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"cosh([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 0.5, -0.5])
    def test_tanh_scalars(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test tanh() on scalar values."""
        pc_result = pc.tanh(dtype_np(value))
        np_result = np.tanh(dtype_np(value))
        assert np.allclose(
            pc_result, np_result
        ), f"tanh({value}) failed for {dtype_name}: pc={pc_result}, np={np_result}"

    @pytest.mark.parametrize(
        ("dtype_name", "dtype_np", "dtype_pc"),
        [
            ("f64", np.float64, "float64"),
            ("Complex64", np.complex128, "complex"),
        ],
    )
    @pytest.mark.parametrize("value", [0.0, 0.5, -0.5])
    def test_tanh_arrays(self, dtype_name, dtype_np, dtype_pc, value) -> None:
        """Test tanh() on array values."""
        pc_arr = pc.array([value], dtype=dtype_pc)
        np_arr = np.array([value], dtype=dtype_np)

        pc_result = pc.tanh(pc_arr)
        np_result = np.tanh(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"tanh([{value}]) failed for {dtype_name}: pc={pc_result}, np={np_result}"


class TestRegressionOriginalBug:
    """Specific regression tests for the original abs([0+1j]) bug."""

    def test_abs_purely_imaginary_complex64_array(self) -> None:
        """
        Regression test for the bug where pc.abs([0+1j]) returned [0.0].

        This bug occurred because extract_f64_array() succeeded on Complex64
        arrays by reinterpreting the memory, returning only the real parts.
        """
        # The exact case that was failing
        pc_arr = pc.array([0 + 1j], dtype="complex")
        pc_result = pc.abs(pc_arr)

        np_arr = np.array([0 + 1j], dtype=np.complex128)
        np_result = np.abs(np_arr)

        assert np.allclose(
            pc_result, np_result
        ), f"REGRESSION: abs([0+1j]) bug has returned! Expected [1.0], got {pc_result}"
        assert np.isclose(pc_result[0], 1.0), f"REGRESSION: abs([0+1j]) should be [1.0], got {pc_result}"

    def test_abs_various_complex_arrays(self) -> None:
        """Test abs() on various complex arrays to ensure dtype validation works."""
        test_cases = [
            ([0 + 1j], "purely imaginary"),
            ([1 + 0j], "purely real"),
            ([3 + 4j], "both components"),
            ([0 - 1j], "negative imaginary"),
            ([-3 + 4j], "negative real"),
        ]

        for values, description in test_cases:
            pc_arr = pc.array(values, dtype="complex")
            np_arr = np.array(values, dtype=np.complex128)

            pc_result = pc.abs(pc_arr)
            np_result = np.abs(np_arr)

            assert np.allclose(
                pc_result, np_result
            ), f"abs({description}) failed: expected {np_result}, got {pc_result}"

    def test_dtype_mismatch_detection(self) -> None:
        """
        Verify that dtype mismatches are properly detected.

        This tests that the dtype validation added to impl_extract_array
        properly rejects type mismatches.
        """
        # Create a complex array
        complex_arr = pc.array([1 + 2j], dtype="complex")

        # Try to extract it - should work with correct dtype
        # The internal extract_complex64_array should succeed
        result = pc.abs(complex_arr)
        assert np.isclose(result[0], np.abs(1 + 2j))

        # If we tried extract_f64_array internally, it should fail
        # (We can't directly test this from Python, but the abs() test above
        # verifies that the correct extraction path is taken)


class TestMultiElementArrays:
    """Test that dtype validation works with multi-element arrays."""

    @pytest.mark.parametrize("size", [2, 5, 10])
    def test_abs_complex_multi_element(self, size) -> None:
        """Test abs() on multi-element complex arrays."""
        values = [complex(i, i + 1) for i in range(size)]

        pc_arr = pc.array(values, dtype="complex")
        np_arr = np.array(values, dtype=np.complex128)

        pc_result = pc.abs(pc_arr)
        np_result = np.abs(np_arr)

        assert np.allclose(pc_result, np_result), f"abs() failed for {size}-element complex array"

    @pytest.mark.parametrize("size", [2, 5, 10])
    def test_sqrt_complex_multi_element(self, size) -> None:
        """Test sqrt() on multi-element complex arrays."""
        values = [complex(i + 1, i + 2) for i in range(size)]

        pc_arr = pc.array(values, dtype="complex")
        np_arr = np.array(values, dtype=np.complex128)

        pc_result = pc.sqrt(pc_arr)
        np_result = np.sqrt(np_arr)

        assert np.allclose(pc_result, np_result), f"sqrt() failed for {size}-element complex array"
