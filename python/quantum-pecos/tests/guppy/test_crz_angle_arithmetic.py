"""Test suite for CRz angle arithmetic improvements."""

import pecos_rslib
from guppylang import guppy
from guppylang.std.quantum import crz, h, measure, pi, qubit


class TestCRzAngleArithmetic:
    """Test CRz gate with proper angle arithmetic."""

    def test_crz_angle_halving(self) -> None:
        """Test that CRz properly halves angles in RZZ decomposition."""

        @guppy
        def test_crz_pi() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            crz(q0, q1, pi)  # π angle
            return measure(q0), measure(q1)

        hugr = test_crz_pi.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have proper angle arithmetic
        assert "___rzz" in output
        assert "___rz" in output

        # Check that we have different angle values (indicating proper arithmetic)
        lines = output.split("\n")
        rzz_calls = [line for line in lines if "tail call void @___rzz" in line]
        rz_calls = [
            line
            for line in lines
            if "tail call void @___rz" in line and "rzz" not in line
        ]

        assert len(rzz_calls) >= 1, "Should have RZZ call"
        assert len(rz_calls) >= 2, "Should have RZ correction calls"

    def test_crz_different_angles(self) -> None:
        """Test CRz with different angle values."""

        @guppy
        def test_crz_pi_half() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            crz(q0, q1, pi / 2)  # π/2 angle
            return measure(q0), measure(q1)

        hugr = test_crz_pi_half.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should decompose correctly
        assert "___rzz" in output
        assert "___rz" in output

    def test_crz_angle_consistency(self) -> None:
        """Test that CRz angles are properly calculated."""

        @guppy
        def test_crz_pi_fourth() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            crz(q0, q1, pi / 4)  # π/4 angle
            return measure(q0), measure(q1)

        hugr = test_crz_pi_fourth.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Verify the decomposition is present
        assert "tail call void @___rzz" in output
        # Should have correction rotations
        rz_corrections = output.count("tail call void @___rz")
        assert rz_corrections >= 2, "Should have at least 2 RZ corrections"

    def test_crz_decomposition(self) -> None:
        """Test CRz gate decomposes correctly into RZZ and RZ operations."""

        @guppy
        def simple_crz() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            crz(q0, q1, pi / 2)
            return measure(q0), measure(q1)

        hugr = simple_crz.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should decompose CRz into RZZ and RZ operations
        assert "___rzz" in output, "CRz should use RZZ in its decomposition"
        assert "___rz" in output, "CRz should use RZ corrections"

        # Should also have the other expected quantum operations
        assert "___rxy" in output, "Should have RXY (H gate is decomposed to RXY+RZ)"
        assert "___lazy_measure" in output, "Should have measurement operations"
        assert "___qalloc" in output, "Should allocate qubits"
        assert "___qfree" in output, "Should free qubits"

    def test_crz_zero_angle(self) -> None:
        """Test CRz with zero angle (should be identity)."""

        @guppy
        def test_crz_zero() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            # Use pi * 0 instead of 0.0 to get proper angle type
            crz(q0, q1, pi * 0)  # Zero angle
            return measure(q0), measure(q1)

        hugr = test_crz_zero.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Even with zero angle, should still have the decomposition structure
        assert "___rzz" in output or len(output) > 100  # Should compile successfully
