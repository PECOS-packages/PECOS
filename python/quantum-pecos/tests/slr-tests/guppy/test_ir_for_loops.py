"""Test For loop implementation in IR generator."""

import pytest
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, For, Main, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator
from pecos.slr.misc import Comment


def test_for_loop_range_basic() -> None:
    """Test basic for loop with range."""
    prog = Main(
        q := QReg("q", 5),
        # Apply H gate to each qubit using For loop
        For("i", 0, 5).Do(
            Comment("Apply H to qubit i"),
            # In real implementation, we'd use: qubit.H(q[i])
            # For now, just apply to q[0] as placeholder
            qubit.H(q[0]),
        ),
        Measure(q) > CReg("results", 5),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check that for loop is generated
    assert "for i in range(0, 5):" in code
    assert "# Apply H to qubit i" in code


def test_for_loop_range_with_step() -> None:
    """Test for loop with custom step."""
    prog = Main(
        q := QReg("q", 10),
        # Apply X to every other qubit
        For("i", 0, 10, 2).Do(
            Comment("Apply X to even-indexed qubits"),
            qubit.X(q[0]),  # Placeholder
        ),
        Measure(q) > CReg("results", 10),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check step parameter
    assert "for i in range(0, 10, 2):" in code


def test_for_loop_iterable() -> None:
    """Test for loop over an iterable."""
    prog = Main(
        q := QReg("q", 3),
        _indices := CReg("indices", 3),
        # For loop over a collection (conceptual)
        For("idx", "indices").Do(
            Comment("Process index from collection"),
            qubit.Y(q[0]),
        ),
        Measure(q) > CReg("results", 3),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check iterable pattern
    assert "for idx in indices:" in code


def test_nested_for_loops() -> None:
    """Test nested for loops."""
    prog = Main(
        q := QReg("q", 9),  # 3x3 grid
        # Nested loops for 2D pattern
        For("i", 0, 3).Do(
            Comment("Outer loop"),
            For("j", 0, 3).Do(
                Comment("Inner loop"),
                Comment("Would apply operation to q[i*3 + j]"),
                qubit.H(q[0]),  # Placeholder
            ),
        ),
        Measure(q) > CReg("results", 9),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check nested structure
    assert "for i in range(0, 3):" in code
    assert "for j in range(0, 3):" in code
    assert "# Outer loop" in code
    assert "# Inner loop" in code


def test_for_loop_with_quantum_operations() -> None:
    """Test for loop with quantum operations inside."""
    prog = Main(
        data := QReg("data", 4),
        ancilla := QReg("ancilla", 1),
        # Initialize data qubits
        For("i", 0, 4).Do(
            Comment("Initialize data qubit"),
            qubit.H(data[0]),  # Would be data[i]
        ),
        # Entangle with ancilla
        For("i", 0, 4).Do(
            Comment("Entangle with ancilla"),
            qubit.CX(data[0], ancilla[0]),  # Would be data[i]
        ),
        Measure(data) > CReg("data_results", 4),
        Measure(ancilla) > CReg("ancilla_result", 1),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check multiple for loops
    assert code.count("for i in range(0, 4):") >= 2
    assert "quantum.h" in code
    assert "quantum.cx" in code


def test_for_loop_limitations() -> None:
    """Test current limitations of For loop implementation."""
    # The main limitation is that we can't use the loop variable
    # to index into quantum registers yet

    prog = Main(
        q := QReg("q", 5),
        For("i", 0, 5).Do(
            Comment("TODO: Need to support q[i] indexing"),
            Comment("Currently would need to unpack array first"),
        ),
        Measure(q) > CReg("c", 5),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Document the limitation
    assert "for i in range(0, 5):" in code
    assert "TODO" in code


def test_for_error_in_qasm() -> None:
    """Test that For loops raise error in QASM generator."""
    from pecos.slr.gen_codes.gen_qasm import QASMGenerator

    prog = Main(
        q := QReg("q", 3),
        For("i", 0, 3).Do(
            qubit.H(q[0]),
        ),
        Measure(q) > CReg("c", 3),
    )

    gen = QASMGenerator()

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        gen.generate_block(prog)

    assert "For loops are not supported in QASM" in str(exc_info.value)
