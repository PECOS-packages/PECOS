"""Basic tests for IR generator functionality."""

from pecos.qeclib import qubit
from pecos.qeclib.qubit.measures import Measure
from pecos.slr import CReg, If, Main, QReg
from pecos.slr.gen_codes.guppy.ir_generator import IRGuppyGenerator


def test_ir_generates_valid_guppy() -> None:
    """Test that IR generator produces valid Guppy code structure."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        Measure(q[0]) > c[0],
        Measure(q[1]) > c[1],
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check imports
    assert "from __future__ import annotations" in code
    assert "from guppylang.decorator import guppy" in code
    assert "from guppylang.std import quantum" in code
    assert "from guppylang.std.builtins import array, owned, result" in code

    # Check function structure
    assert "@guppy" in code
    assert "def main() -> None:" in code

    # Check variable declarations
    assert "c = array(False for _ in range(2))" in code
    # q might be pre-allocated or locally allocated

    # Check measurements - with dynamic allocation uses unpacked variables
    assert "quantum.measure" in code
    # Could be c[0]/c[1] or c_0/c_1 depending on allocation strategy
    assert ("c[0]" in code and "c[1]" in code) or ("c_0" in code and "c_1" in code)

    # Check result
    assert 'result("c", c)' in code


def test_ir_handles_quantum_gates() -> None:
    """Test that IR generator handles quantum gates."""
    prog = Main(
        q := QReg("q", 2),
        qubit.H(q[0]),
        qubit.CX(q[0], q[1]),
        Measure(q) > CReg("c", 2),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check gates
    assert "quantum.h(q[0])" in code
    assert "quantum.cx(q[0], q[1])" in code

    # Check full array measurement
    assert "quantum.measure_array(q)" in code


def test_ir_handles_conditionals() -> None:
    """Test that IR generator handles conditional statements."""
    prog = Main(
        q := QReg("q", 2),
        flag := CReg("flag", 1),
        Measure(q[0]) > flag[0],
        If(flag[0]).Then(
            qubit.X(q[1]),
        ),
    )

    gen = IRGuppyGenerator()
    gen.generate_block(prog)
    code = gen.get_output()

    # Check conditional structure (with unpacking, flag[0] becomes flag_0)
    assert "if flag[0]:" in code or "if flag_0:" in code
    assert "quantum.x(q_1)" in code
