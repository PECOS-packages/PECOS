"""Test suite for V and Vdg gates."""

import pecos_rslib
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit, v, vdg


class TestVGates:
    """Test V and Vdg gates."""

    def test_v_gate(self) -> None:
        """Test V gate (sqrt(X))."""

        @guppy
        def test_v() -> bool:
            q = qubit()
            h(q)
            v(q)
            return measure(q)

        hugr = test_v.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # V gate should be decomposed to RXY(0, π/2)
        assert "___rxy" in output
        assert "double 0.0" in output  # First angle should be 0
        assert "0x3FF921FB54442D18" in output  # π/2 in hex

    def test_vdg_gate(self) -> None:
        """Test Vdg gate (V†, sqrt(X)†)."""

        @guppy
        def test_vdg() -> bool:
            q = qubit()
            h(q)
            vdg(q)
            return measure(q)

        hugr = test_vdg.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Vdg gate should be decomposed to RXY(0, -π/2)
        assert "___rxy" in output
        assert "double 0.0" in output  # First angle should be 0
        assert "0xBFF921FB54442D18" in output  # -π/2 in hex

    def test_v_vdg_sequence(self) -> None:
        """Test V followed by Vdg (should cancel)."""

        @guppy
        def test_v_vdg() -> bool:
            q = qubit()
            h(q)
            v(q)
            vdg(q)
            return measure(q)

        hugr = test_v_vdg.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have two RXY calls (V and Vdg)
        assert output.count("___rxy") >= 2

    def test_double_v(self) -> None:
        """Test V applied twice (equals X)."""

        @guppy
        def test_double_v() -> bool:
            q = qubit()
            v(q)
            v(q)
            return measure(q)

        hugr = test_double_v.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have two RXY calls for the two V gates (plus one declaration)
        rxy_calls = output.count("tail call void @___rxy")
        assert rxy_calls == 2, f"Expected 2 RXY calls, got {rxy_calls}"
        assert output.count("double 0.0") >= 2

    def test_compiler_compatibility_v_gates(self) -> None:
        """Verify V gates compile correctly."""

        @guppy
        def simple_v() -> bool:
            q = qubit()
            v(q)
            return measure(q)

        hugr = simple_v.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # V gate should be decomposed into RXY
        assert "declare" in output
        assert "___rxy" in output, "V gate should use RXY"
        assert "___lazy_measure" in output, "Should have measurement"
        assert "___qfree" in output, "Should free qubit"

        # Verify RXY is actually called
        assert "tail call void @___rxy" in output, "RXY should be called for V gate"
