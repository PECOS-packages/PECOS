"""Test suite for Reset operation."""

import pecos_rslib
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit, reset, x


class TestResetOperation:
    """Test reset operation."""

    def test_reset_basic(self) -> None:
        """Test basic reset operation."""

        @guppy
        def test_reset() -> bool:
            q = qubit()
            h(q)  # Put in superposition
            reset(q)  # Reset to |0⟩
            return measure(q)

        hugr = test_reset.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have reset operation
        assert "___reset" in output
        assert "tail call void @___reset" in output

    def test_reset_after_x(self) -> None:
        """Test reset after X gate."""

        @guppy
        def test_reset_x() -> bool:
            q = qubit()
            x(q)  # Flip to |1⟩
            reset(q)  # Reset to |0⟩
            return measure(q)

        hugr = test_reset_x.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have both X gate operations and reset
        assert "___rxy" in output  # X gate uses RXY
        assert "___reset" in output

    def test_multiple_resets(self) -> None:
        """Test multiple reset operations."""

        @guppy
        def test_multi_reset() -> bool:
            q = qubit()
            h(q)
            reset(q)
            x(q)
            reset(q)
            return measure(q)

        hugr = test_multi_reset.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have two reset calls (plus potentially one from QAlloc)
        reset_calls = output.count("tail call void @___reset")
        assert reset_calls >= 2, f"Expected at least 2 reset calls, got {reset_calls}"

    def test_reset_two_qubits(self) -> None:
        """Test reset on two qubits."""

        @guppy
        def test_reset_two() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            h(q1)
            h(q2)
            reset(q1)
            reset(q2)
            return measure(q1), measure(q2)

        hugr = test_reset_two.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have multiple reset calls
        assert "___reset" in output
        # Should have at least 2 reset calls from the Reset operations
        # (plus 2 from QAlloc initialization)
        reset_calls = output.count("tail call void @___reset")
        assert reset_calls >= 4, f"Expected at least 4 reset calls, got {reset_calls}"

    def test_reset_compiler_compatibility(self) -> None:
        """Verify reset operation compiles correctly."""

        @guppy
        def simple_reset() -> bool:
            q = qubit()
            reset(q)
            return measure(q)

        hugr = simple_reset.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should declare and use reset
        assert "declare" in output
        assert "___reset" in output, "Should have reset operation"
        assert "___lazy_measure" in output, "Should have measurement"
        assert "___qfree" in output, "Should free qubit"

        # Verify reset is actually called
        assert "tail call void @___reset" in output, "Reset should be called"

    def test_reset_in_circuit(self) -> None:
        """Test reset in a more complex circuit."""
        from guppylang.std.quantum import cx

        @guppy
        def reset_circuit() -> tuple[bool, bool]:
            q1 = qubit()
            q2 = qubit()
            h(q1)
            cx(q1, q2)  # Entangle
            reset(q1)  # Reset control qubit
            # q2 should still be in a mixed state
            return measure(q1), measure(q2)

        hugr = reset_circuit.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have all operations
        assert "___rxy" in output  # From H and CX
        assert "___rzz" in output  # From CX
        assert "___reset" in output  # From reset operation
