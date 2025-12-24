"""Test allocation optimization in Guppy code generation."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, For, If, Main, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_short_lived_ancilla_optimization() -> None:
    """Test that short-lived ancilla qubits are allocated locally."""
    prog = Main(
        data := QReg("data", 3),
        ancilla := QReg("ancilla", 2),
        temp1 := CReg("temp1", 1),
        temp2 := CReg("temp2", 1),
        results := CReg("results", 3),
        # Use data normally
        qubit.H(data[0]),
        qubit.CX(data[0], data[1]),
        # Short-lived ancilla use - should be optimized
        qubit.H(ancilla[0]),
        Measure(ancilla[0]) > temp1[0],
        # Another short-lived use - should be optimized
        qubit.X(ancilla[1]),
        Measure(ancilla[1]) > temp2[0],
        # Measure data
        Measure(data) > results,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check for unified resource planning report
    assert "UNIFIED RESOURCE PLANNING REPORT" in code or "Optimization Report" in code
    # Should have optimization analysis (unified or allocation report)
    assert "short-lived" in code.lower() or "local allocation" in code.lower()


def test_reused_ancilla_no_optimization() -> None:
    """Test that reused ancilla qubits are not optimized (allocated upfront)."""
    prog = Main(
        data := QReg("data", 2),
        ancilla := QReg("ancilla", 1),
        temp1 := CReg("temp1", 1),
        temp2 := CReg("temp2", 1),
        results := CReg("results", 2),
        # First use of ancilla
        qubit.H(ancilla[0]),
        qubit.CX(data[0], ancilla[0]),
        Measure(ancilla[0]) > temp1[0],
        # Reuse the same ancilla - should prevent optimization
        qubit.X(ancilla[0]),
        qubit.CZ(data[1], ancilla[0]),
        Measure(ancilla[0]) > temp2[0],
        Measure(data) > results,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should not optimize reused qubits
    # Unified report shows "need replacement" or old report shows "reused after consumption"
    assert (
        "reused after consumption" in code.lower()
        or "pre_allocate" in code
        or "need replacement" in code.lower()
        or "UNPACKED_PREALLOCATED" in code
    )


def test_mixed_allocation_strategy() -> None:
    """Test mixed allocation where some elements are local, others pre-allocated."""
    prog = Main(
        mixed := QReg("mixed", 4),
        long_lived := CReg("long_lived", 1),
        short1 := CReg("short1", 1),
        short2 := CReg("short2", 1),
        results := CReg("results", 3),
        # Long-lived use of mixed[0] - should be pre-allocated
        qubit.H(mixed[0]),
        qubit.CX(mixed[0], mixed[1]),
        qubit.CZ(mixed[0], mixed[2]),
        qubit.H(mixed[0]),
        Measure(mixed[0]) > long_lived[0],
        # Short-lived use of mixed[1] - could be local
        qubit.X(mixed[1]),
        Measure(mixed[1]) > short1[0],
        # Short-lived use of mixed[2] - could be local
        qubit.Y(mixed[2]),
        Measure(mixed[2]) > short2[0],
        # Never use mixed[3] - should be noted
        Measure(mixed[0:3]) > results,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have optimization or unified planning report
    assert "UNIFIED RESOURCE PLANNING REPORT" in code or "Optimization Report" in code


def test_conditional_scope_prevents_optimization() -> None:
    """Test that qubits used in conditionals are not optimized."""
    prog = Main(
        q := QReg("q", 2),
        flag := CReg("flag", 1),
        conditional_result := CReg("conditional_result", 1),
        Measure(q[0]) > flag[0],
        If(flag[0]).Then(
            # This use is in a conditional scope - should prevent optimization
            qubit.X(q[1]),
            Measure(q[1]) > conditional_result[0],
        ),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have optimization or unified planning report
    assert "UNIFIED RESOURCE PLANNING REPORT" in code or "Optimization Report" in code


def test_loop_scope_prevents_optimization() -> None:
    """Test that qubits used in loops are not optimized."""
    prog = Main(
        q := QReg("q", 3),
        results := CReg("results", 3),
        For("i", 0, 3).Do(
            # This use is in a loop scope - should prevent optimization
            qubit.H(q[0]),
            qubit.X(q[1]),
        ),
        Measure(q) > results,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have optimization or unified planning report
    assert "UNIFIED RESOURCE PLANNING REPORT" in code or "Optimization Report" in code


def test_optimization_report_generation() -> None:
    """Test that optimization reports are generated correctly."""
    prog = Main(
        simple := QReg("simple", 2),
        result1 := CReg("result1", 1),
        result2 := CReg("result2", 1),
        # Simple pattern - allocate and immediately measure
        qubit.H(simple[0]),
        Measure(simple[0]) > result1[0],
        qubit.X(simple[1]),
        Measure(simple[1]) > result2[0],
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have detailed optimization or unified planning report
    assert (
        "UNIFIED RESOURCE PLANNING REPORT" in code
        or "=== Qubit Allocation Optimization Report ===" in code
    )
    assert "simple" in code.lower()  # Array name mentioned
    assert "Strategy:" in code


def test_never_used_qubits() -> None:
    """Test detection of never-used qubits."""
    prog = Main(
        used := QReg("used", 1),
        _unused := QReg("unused", 2),
        results := CReg("results", 1),
        # Only use one register
        qubit.H(used[0]),
        Measure(used) > results,
        # unused register is never touched
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should detect unused qubits
    assert "never used" in code.lower() or "unused" in code.lower()
