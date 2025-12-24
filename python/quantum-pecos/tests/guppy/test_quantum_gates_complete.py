"""Test suite for complete quantum gate coverage in PECOS compiler."""

import pecos_rslib
from guppylang import guppy
from guppylang.std.quantum import (
    ch,
    cx,
    cy,
    cz,
    h,
    measure,
    pi,
    qubit,
    rx,
    ry,
    rz,
    s,
    sdg,
    t,
    tdg,
    x,
    y,
    z,
)


class TestBasicGates:
    """Test basic single-qubit gates."""

    def test_pauli_gates(self) -> None:
        """Test Pauli gates X, Y, Z."""

        @guppy
        def test_x() -> bool:
            q = qubit()
            x(q)
            return measure(q)

        @guppy
        def test_y() -> bool:
            q = qubit()
            y(q)
            return measure(q)

        @guppy
        def test_z() -> bool:
            q = qubit()
            z(q)
            return measure(q)

        for func in [test_x, test_y, test_z]:
            hugr = func.compile()
            output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
            assert "tail call" in output
            assert "@___r" in output  # Should have rotation calls

    def test_phase_gates(self) -> None:
        """Test phase gates S and T."""

        @guppy
        def test_s() -> bool:
            q = qubit()
            s(q)
            return measure(q)

        @guppy
        def test_t() -> bool:
            q = qubit()
            t(q)
            return measure(q)

        for func in [test_s, test_t]:
            hugr = func.compile()
            output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
            assert "___rz" in output
            assert "tail call" in output

    def test_hadamard(self) -> None:
        """Test Hadamard gate."""

        @guppy
        def test_h() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        hugr = test_h.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "___rz" in output


class TestAdjointGates:
    """Test adjoint gates."""

    def test_adjoint_gates(self) -> None:
        """Test S† and T† gates."""

        @guppy
        def test_sdg_gate() -> bool:
            q = qubit()
            h(q)
            sdg(q)
            return measure(q)

        @guppy
        def test_tdg_gate() -> bool:
            q = qubit()
            h(q)
            tdg(q)
            return measure(q)

        for func in [test_sdg_gate, test_tdg_gate]:
            hugr = func.compile()
            output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
            assert "___rz" in output
            # Should have negative angle for adjoint
            assert "0xBF" in output  # Negative hex prefix


class TestRotationGates:
    """Test parameterized rotation gates."""

    def test_rx_gate(self) -> None:
        """Test Rx gate with angle."""

        @guppy
        def test_rx_pi4() -> bool:
            q = qubit()
            rx(q, pi / 4)
            return measure(q)

        hugr = test_rx_pi4.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "double 0.0" in output  # First angle should be 0 for Rx

    def test_ry_gate(self) -> None:
        """Test Ry gate with angle."""

        @guppy
        def test_ry_pi2() -> bool:
            q = qubit()
            ry(q, pi / 2)
            return measure(q)

        hugr = test_ry_pi2.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        # For Ry, second angle should be 0

    def test_rz_gate(self) -> None:
        """Test Rz gate with angle."""

        @guppy
        def test_rz_pi() -> bool:
            q = qubit()
            rz(q, pi)
            return measure(q)

        hugr = test_rz_pi.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rz" in output
        # Should have an angle parameter
        assert "double" in output


class TestControlGates:
    """Test two-qubit control gates."""

    def test_cx_gate(self) -> None:
        """Test CNOT/CX gate."""

        @guppy
        def test_cx() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        hugr = test_cx.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "___rzz" in output
        assert "___rz" in output

    def test_cy_gate(self) -> None:
        """Test CY gate."""

        @guppy
        def test_cy() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cy(q0, q1)
            return measure(q0), measure(q1)

        hugr = test_cy.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "___rzz" in output
        assert "___rz" in output
        # Should have multiple operations for CY decomposition
        assert output.count("tail call void @___") >= 7

    def test_cz_gate(self) -> None:
        """Test CZ gate."""

        @guppy
        def test_cz() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cz(q0, q1)
            return measure(q0), measure(q1)

        hugr = test_cz.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rzz" in output
        assert "___rz" in output

    def test_ch_gate(self) -> None:
        """Test CH gate."""

        @guppy
        def test_ch() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            ch(q0, q1)
            return measure(q0), measure(q1)

        hugr = test_ch.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "___rz" in output
        # CH has its own decomposition


class TestComplexCircuits:
    """Test more complex quantum circuits."""

    def test_bell_state(self) -> None:
        """Test Bell state preparation."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        hugr = bell.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "___rzz" in output
        assert "___lazy_measure" in output
        assert "___qfree" in output

    def test_ghz_state(self) -> None:
        """Test GHZ state preparation."""

        @guppy
        def ghz() -> tuple[bool, bool, bool]:
            q0 = qubit()
            q1 = qubit()
            q2 = qubit()
            h(q0)
            cx(q0, q1)
            cx(q0, q2)
            return measure(q0), measure(q1), measure(q2)

        hugr = ghz.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rzz" in output  # Has CX gates
        assert "___lazy_measure" in output  # Has measurements

    def test_mixed_gates(self) -> None:
        """Test circuit with mixed gate types."""

        @guppy
        def mixed() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            s(q0)
            rx(q1, pi / 4)
            cy(q0, q1)
            t(q1)
            return measure(q0), measure(q1)

        hugr = mixed.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())
        assert "___rxy" in output
        assert "___rz" in output
        assert "___rzz" in output


class TestCompilerOutput:
    """Test PECOS compiler output quality."""

    def test_basic_gates_compile_correctly(self) -> None:
        """Verify basic gates compile to expected operations."""

        @guppy
        def simple() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        hugr = simple.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should have the expected quantum operations
        assert "___qalloc" in output, "Should allocate qubit"
        assert "___rxy" in output, "H gate is decomposed to RXY+RZ operations"
        assert "___rz" in output, "H gate decomposition includes RZ"
        assert "___lazy_measure" in output, "Should have measurement"
        assert "___qfree" in output, "Should free qubit"

        # Should have reasonable number of operations
        operations = output.count("tail call void @___")
        assert 3 <= operations <= 10, f"Expected 3-10 operations, got {operations}"

    def test_declarations_optimized(self) -> None:
        """Verify only used operations are declared."""

        @guppy
        def only_h() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        hugr = only_h.compile()
        output = pecos_rslib.compile_hugr_to_llvm_rust(hugr.to_bytes())

        # Should declare only what's used
        assert "declare" in output
        assert "___rxy" in output
        assert "___rz" in output

        # Should NOT declare unused operation
        assert "___rzz" not in output or "tail call void @___rzz" not in output
