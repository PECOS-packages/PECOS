"""Test SLR to physical quantum circuit compilation for various cases."""

import pytest
from pecos.qeclib import qubit as p
from pecos.qeclib.steane.steane_class import Steane
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
    SlrConverter,
)

# TODO: Remove reference to hqslib1.inc... better yet, don't have tests on qasm


def telep(prep_basis: str, meas_basis: str) -> str:
    """A simple example of creating a logical teleportation circuit.

    Args:
        prep_basis (str): A string indicating what Pauli basis to prepare the state in. Acceptable inputs include:
            "+X"/"X", "-X", "+Y"/"Y", "-Y", "+Z"/"Z", and "-Z".
        meas_basis (str): A string indicating what Pauli basis the measure out the logical qubit in. Acceptable inputs
            include: "X", "Y", and "Z".

    Returns:
        A logical program written in extended OpenQASM 2.0
    """
    return Main(
        m_bell := CReg("m_bell", size=2),
        m_out := CReg("m_out", size=1),
        # Input state:
        sin := Steane("sin", default_rus_limit=2),
        smid := Steane("smid"),
        sout := Steane("sout"),
        # Create Bell state
        smid.pz(),  # prep logical qubit in |0>/|+Z> state with repeat-until-success initialization
        sout.pz(),
        Barrier(smid.d, sout.d),
        smid.h(),
        smid.cx(sout),  # CX with control on smid and target on sout
        smid.qec(),
        sout.qec(),
        # prepare input state in some Pauli basis state
        sin.p(prep_basis, rus_limit=3),
        sin.qec(),
        # entangle input with one of the logical qubits of the Bell pair
        sin.cx(smid),
        sin.h(),
        # Bell measurement
        sin.mz(m_bell[0]),
        smid.mz(m_bell[1]),
        # Corrections
        If(m_bell[1] == 0).Then(sout.x()),
        If(m_bell[0] == 0).Then(sout.z()),
        # Final output stored in `m_out[0]`
        sout.m(meas_basis, m_out[0]),
    )


def test_bell() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.H(q[0]),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "hqslib1.inc";\n'
        "qreg q[2];\n"
        "creg m[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q -> m;"
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
    )

    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir


def test_if_bell() -> None:
    """Test that a more complex Bell prep and measure circuit with if statements can be created."""

    class Bell(Block):
        def __init__(self, q0: Qubit, q1: Qubit, m0: Bit, m1: Bit) -> None:
            super().__init__()
            self.extend(
                p.Prep(q0),
                p.Prep(q1),
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
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "hqslib1.inc";\n'
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
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "hqslib1.inc";\n'
        "qreg q[2];\n"
        "creg c[4];\n"
        "creg b[4];\n"
        "c = 3;\n"
        "c = 3;\n"
        "c = 3;\n"
        "// Here is some injected QASM:\n"
        "c = b & 1;\n"
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
        m_hidden := CReg("m_hidden", 2, result=False),
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
            p.RX[0.3](q[0]),
        ),
        If(m < m_hidden).Then(
            p.H(q[0]),
        ),
        Barrier(q[0], q[1]),
        p.F4dg(q[1]),
        p.SZdg(q[0]),
        p.CX(q[0], q[1]),
        Barrier(q[1], q[0]),
        p.RX[0.3](q[0]),
        p.Measure(q) > m,
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
    )
    qir = SlrConverter(prog).qir()
    assert "sub" in qir


@pytest.mark.optional_dependency
def test_steane_qir() -> None:
    """Test the teleportation program using the Steane code."""
    # print(SlrConverter(telep("X", "X")).qir())
