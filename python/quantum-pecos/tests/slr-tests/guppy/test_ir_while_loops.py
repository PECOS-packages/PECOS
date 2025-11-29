"""Test While loop support in IR generator."""

import pytest
from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, For, Main, QReg, While
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator
from pecos.slr.misc import Comment


def test_ir_while_loop_basic() -> None:
    """Test basic while loop generation."""
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        # While loop with a simple condition
        # In real use, condition would come from measurement or input
        While(c[0]).Do(
            Comment("Apply operation in loop"),
            qubit.H(q[0]),
        ),
        Measure(q[0]) > c[0],
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check that while loop is generated
    assert "while " in code
    assert "# Apply operation in loop" in code


def test_ir_while_loop_with_quantum() -> None:
    """Test while loop with quantum operations."""
    prog = Main(
        q := QReg("q", 2),
        control := CReg("control", 1),
        result := CReg("result", 2),
        # Initialize with H on first qubit
        qubit.H(q[0]),
        # While loop controlled by classical bit
        While(control[0]).Do(
            Comment("Quantum operations in loop"),
            qubit.CX(q[0], q[1]),
            qubit.H(q[1]),
            Comment("In practice, control bit would be updated here"),
        ),
        # Measure both qubits
        Measure(q) > result,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check structure
    assert "while " in code
    assert "quantum.cx" in code
    assert "quantum.h" in code
    assert "quantum.measure_array(q)" in code


def test_ir_nested_while_loops() -> None:
    """Test nested while loops."""
    prog = Main(
        q := QReg("q", 1),
        outer_cond := CReg("outer_cond", 1),
        inner_cond := CReg("inner_cond", 1),
        result := CReg("result", 1),
        # Outer while loop
        While(outer_cond[0]).Do(
            Comment("Outer loop"),
            # Inner while loop
            While(inner_cond[0]).Do(
                Comment("Inner loop"),
                qubit.X(q[0]),
            ),
            qubit.Z(q[0]),
        ),
        Measure(q[0]) > result[0],
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check for nested structure
    assert code.count("while ") >= 2  # At least 2 while statements
    assert "# Outer loop" in code
    assert "# Inner loop" in code


def test_ir_while_error_in_qasm() -> None:
    """Test that While loops raise error in QASM generator."""
    from pecos.slr.gen_codes.gen_qasm import QASMGenerator

    prog = Main(
        c := CReg("c", 1),
        While(c[0]).Do(
            Comment("This should fail in QASM"),
        ),
    )

    gen = QASMGenerator()

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        gen.generate_block(prog)

    assert "While loops are not supported in QASM" in str(exc_info.value)


def test_ir_for_loop_placeholder() -> None:
    """Test For loop generates placeholder (not fully implemented yet)."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        # For loop (generates TODO for now)
        For("i", 0, 3).Do(
            Comment("Apply H to each qubit"),
            qubit.H(q[0]),  # Would use i to index in full implementation
        ),
        # Measure all
        Measure(q) > c,
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # For loops should now be properly implemented
    assert "for i in range(0, 3):" in code


def test_while_loop_quantum_resource_handling() -> None:
    """Test that quantum resources are properly tracked in while loops."""
    prog = Main(
        q := QReg("q", 3),
        ancilla := QReg("ancilla", 1),
        cond := CReg("cond", 1),
        results := CReg("results", 4),
        # Initialize
        qubit.H(q[0]),
        qubit.H(q[1]),
        # While loop that uses quantum resources
        While(cond[0]).Do(
            Comment("Use ancilla in loop"),
            qubit.H(ancilla[0]),
            qubit.CX(q[0], ancilla[0]),
            Measure(ancilla[0]) > results[0],
            Comment("Ancilla is consumed each iteration"),
        ),
        # Measure remaining qubits
        Measure(q) > results[1:4],
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check that measurements are properly handled
    assert "while " in code
    # With dynamic allocation, ancilla is allocated locally in the loop
    assert "ancilla_0_local = quantum.qubit()" in code
    assert "quantum.measure(ancilla_0_local)" in code
    assert "quantum.measure_array(q)" in code
