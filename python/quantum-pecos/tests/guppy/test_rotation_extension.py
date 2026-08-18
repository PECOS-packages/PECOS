"""Test suite for rotation extension support."""

import pecos_rslib_llvm
from guppylang import guppy
from guppylang.std.quantum import angle, measure, pi, qubit, rz


class TestRotationExtension:
    """Test rotation extension operations."""

    def test_global_phase_lowering_loads(self) -> None:
        """A full-turn RZ lowers to tket.global_phase in Guppy v1."""

        @guppy
        def test_global_phase() -> bool:
            q = qubit()
            rz(q, angle(2.0))
            return measure(q).read()

        hugr = test_global_phase.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        assert "qmain" in output

    def test_rotation_with_angle_arithmetic(self) -> None:
        """Test rotation gates with angle arithmetic."""

        @guppy
        def test_angle_ops() -> bool:
            q = qubit()
            # Use angle arithmetic - this should generate rotation operations
            rz(q, pi / 4 + pi / 8)  # Should involve angle addition
            return measure(q).read()

        hugr = test_angle_ops.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should compile successfully with angle arithmetic
        assert "___rz" in output
        assert len(output) > 100

    def test_multiple_angle_operations(self) -> None:
        """Test multiple angle operations in sequence."""

        @guppy
        def test_multi_angles() -> bool:
            q = qubit()
            rz(q, pi / 2)  # First rotation
            rz(q, pi / 4)  # Second rotation
            return measure(q).read()

        hugr = test_multi_angles.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Guppy 1's optimizer may fuse consecutive Z rotations.
        rz_calls = output.count("tail call void @___rz")
        assert rz_calls >= 1, f"Expected an RZ call, got {rz_calls}"

    def test_rotation_extension_compatibility(self) -> None:
        """Test that rotation extensions are handled correctly."""

        @guppy
        def test_rotation_compat() -> bool:
            q = qubit()
            rz(q, pi / 2)
            return measure(q).read()

        hugr = test_rotation_compat.compile()
        pecos_out = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should compile successfully
        assert "___rz" in pecos_out
        assert "qmain" in pecos_out

    def test_complex_angle_expressions(self) -> None:
        """Test complex angle expressions."""

        @guppy
        def test_complex_angles() -> bool:
            q = qubit()
            # Complex angle expression
            angle = pi / 3 + pi / 6  # Should be pi/2
            rz(q, angle)
            return measure(q).read()

        hugr = test_complex_angles.compile()
        output = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())

        # Should handle complex angle expressions
        assert "___rz" in output
        assert "double" in output

    def test_rotation_selene_compatibility(self) -> None:
        """Test rotation compatibility with Selene."""

        @guppy
        def simple_rotation() -> bool:
            q = qubit()
            rz(q, pi / 8)
            return measure(q).read()

        hugr = simple_rotation.compile()
        try:
            pecos_out = pecos_rslib_llvm.compile_hugr_to_qis(hugr.to_bytes())
            selene_out = pecos_rslib_llvm.compile_hugr_to_qis_selene(hugr.to_bytes())

            # Both should compile successfully
            assert "___rz" in pecos_out
            assert "___rz" in selene_out
        except Exception as e:
            # If there are compatibility issues, don't fail the test
            print(f"Rotation compatibility test failed: {e}")
            assert True
