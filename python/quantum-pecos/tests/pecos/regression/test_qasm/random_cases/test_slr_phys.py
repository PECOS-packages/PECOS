"""Test SLR to physical quantum circuit compilation for various cases."""

import pytest
from pecos.slr import (
    Barrier,
    Bit,
    Block,
    Comment,
    CReg,
    If,
    Main,
    Permute,
    QReg,
    Qubit,
    Repeat,
    Return,
    SlrConverter,
    rad,
)
from pecos.slr.qeclib import qubit as p
from pecos.slr.qeclib.steane.steane_class import Steane

# TODO: Remove reference to hqslib1.inc... better yet, don't have tests on qasm


def test_bell() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.H(q[0]),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg m[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q[0] -> m[0];\n"
        "measure q[1] -> m[1];"
    )

    assert SlrConverter(prog).qasm() == qasm


@pytest.mark.optional_dependency
def test_bell_qir() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog: Main = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.H(q[0]),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir


@pytest.mark.optional_dependency
def test_bell_qreg_qir() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog: Main = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.H(q),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir


def test_if_bell() -> None:
    """Test that a more complex Bell prep and measure circuit with if statements can be created."""

    class Bell(Block):
        def __init__(self, q0: Qubit, q1: Qubit, m0: Bit, m1: Bit) -> None:
            super().__init__()
            self.extend(
                p.PZ(q0),
                p.PZ(q1),
                p.H(q0),
                p.CX(q0, q1),
                p.Measure(q0) > m0,
                p.Measure(q1) > m1,
            )

    prog = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        c := CReg("c", 4),
        If(c == 1).Then(Bell(q0=q[0], q1=q[1], m0=m[0], m1=m[1])),
        Return(m, c),
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg m[2];\n"
        "creg c[4];\n"
        "if(c == 1) reset q[0];\n"
        "if(c == 1) reset q[1];\n"
        "if(c == 1) h q[0];\n"
        "if(c == 1) cx q[0], q[1];\n"
        "if(c == 1) measure q[0] -> m[0];\n"
        "if(c == 1) measure q[1] -> m[1];"
    )

    assert SlrConverter(prog).qasm() == qasm


def test_strange_program() -> None:
    """Test a weird program to verify we get what is expected for various other SLR objects."""
    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 4),
        b := CReg("b", 4),
        Repeat(3).block(
            c.set(3),
        ),
        Comment("Here is some injected QASM:"),
        c.set(b & 1),
        Permute([q[0], q[1]], [q[1], q[0]]),
        p.H(q[0]),
        Return(c, b),
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "qreg q[2];\n"
        "creg c[4];\n"
        "creg b[4];\n"
        "// Repeat 3 times (unrolled)\n"
        "c = 3;\n"
        "c = 3;\n"
        "c = 3;\n"
        "// Here is some injected QASM:\n"
        "c = (b & 1);\n"
        "// Permutation: q[0] -> q[1], q[1] -> q[0]\n"
        "h q[1];"
    )

    # TODO: Weird things can happen with Permute... if you run a program twice

    assert SlrConverter(prog).qasm() == qasm


@pytest.mark.optional_dependency
def test_control_flow_qir() -> None:
    """Test a program with control flow into QIR."""
    prog = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        m_hidden := CReg("m_hidden", 2),
        Repeat(3).block(
            p.H(q[0]),
        ),
        Comment("Comments go here"),
        If(m == 0)
        .Then(
            p.H(q[0]),
            Block(
                p.H(q[1]),
            ),
        )
        .Else(
            p.RX(rad(0.3), q[0]),
        ),
        If(m < m_hidden).Then(
            p.H(q[0]),
        ),
        Barrier(q[0], q[1]),
        p.F4dg(q[1]),
        p.SZdg(q[0]),
        p.CX(q[0], q[1]),
        Barrier(q[1], q[0]),
        p.RX(rad(0.3), q[0]),
        p.Measure(q) > m,
        Return(m),
    )
    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir


@pytest.mark.optional_dependency
def test_plus_qir() -> None:
    """Test a program with addition compiling into QIR."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),
        m.set(2),
        n.set(2),
        o.set(m + n),
        Return(m, n, o),
    )
    qir = SlrConverter(prog).qir()
    assert "add" in qir


@pytest.mark.optional_dependency
def test_nested_xor_qir() -> None:
    """Test a program with addition compiling into QIR."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),
        p := CReg("p", 2),
        m.set(2),
        n.set(2),
        o.set(2),
        p[0].set((m[0] ^ n[0]) ^ o[0]),
        Return(m, n, o, p),
    )
    qir = SlrConverter(prog).qir()
    assert "xor" in qir


@pytest.mark.optional_dependency
def test_minus_qir() -> None:
    """Test a program with addition compiling into QIR."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),
        m.set(2),
        n.set(2),
        o.set(m - n),
        Return(m, n, o),
    )
    qir = SlrConverter(prog).qir()
    assert "sub" in qir
