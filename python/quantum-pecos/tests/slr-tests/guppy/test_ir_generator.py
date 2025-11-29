"""Test the IR-based Guppy generator."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_ir_simple_measurement() -> None:
    """Test simple measurement with IR generator."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        Measure(q[0]) > c[0],
        Measure(q[1]) > c[1],
    )
    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check basic structure
    assert "@guppy" in code
    assert "def main() -> None:" in code
    # With optimization, q might be dynamically allocated instead of pre-allocated
    assert (
        "q = array(quantum.qubit() for _ in range(2))" in code
        or "q_0 = quantum.qubit()" in code
    )
    assert "c = array(False for _ in range(2))" in code

    # Should have measurements - format depends on allocation strategy
    # Array indexing for pre-allocated, unpacked variables for dynamic
    if "q_0 = quantum.qubit()" in code:
        # Dynamic allocation uses unpacked variables
        assert "c_0 = quantum.measure(q_0)" in code
        assert "c_1 = quantum.measure(q_1)" in code
    else:
        # Pre-allocation uses array indexing
        assert "c[0] = quantum.measure(q[0])" in code
        assert "c[1] = quantum.measure(q[1])" in code
    assert 'result("c", c)' in code


def test_ir_full_array_measurement() -> None:
    """Test full array measurement with IR generator."""
    prog = Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        Measure(q) > c,
    )
    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should use measure_array for full measurement
    assert "c = quantum.measure_array(q)" in code
    # Should NOT unpack
    assert "=" not in code or "= q" not in code or ", q_" not in code


def test_ir_quantum_gates() -> None:
    """Test quantum gates with IR generator."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qubit.H(q[0]),
        qubit.CX(q[0], q[1]),
        Measure(q) > c,
    )
    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should use array indexing
    assert "quantum.h(q[0])" in code
    assert "quantum.cx(q[0], q[1])" in code


def test_ir_conditional_resources() -> None:
    """Test conditional resource consumption with IR generator."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        flag := CReg("flag", 1),
        Measure(q[0]) > flag[0],
        If(flag[0]).Then(
            Measure(q[1]) > c[1],
        ),
    )
    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should have conditional structure (with unpacking, flag[0] becomes flag_0)
    assert "if flag[0]:" in code or "if flag_0:" in code
    # With unpacking, q[1] becomes q_1
    assert "quantum.measure(q[1])" in code or "quantum.measure(q_1)" in code
    # Should generate valid code
    assert "result(" in code


def test_ir_variable_renaming() -> None:
    """Test variable renaming to avoid conflicts."""
    prog = Main(
        result := QReg("result", 2),  # Conflicts with result() function
        array := CReg("array", 2),  # Conflicts with array() function
        Measure(result) > array,
    )
    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Should rename conflicting variables
    assert "result_reg" in code
    assert "array_reg" in code
    # Should use renamed variables correctly
    assert "quantum.measure_array" in code
