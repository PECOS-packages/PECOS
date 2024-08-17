from pecos import __version__
from pecos.qeclib import qubit as p
from pecos.slr import QASM, Assign, Bit, Block, Comment, CReg, If, Main, Permute, QReg, Qubit, Repeat

# TODO: Remove reference to hqslib1.inc... better yet, don't have tests on qasm


def test_bell():
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
        f"// Generated using: PECOS version {__version__}\n"
        "qreg q[2];\n"
        "creg m[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q -> m;"
    )

    assert prog.qasm() == qasm


def test_if_bell():
    """Test that a more complex Bell prep and measure circuit with if statemenscan be created."""

    class Bell(Block):

        def __init__(self, q0: Qubit, q1: Qubit, m0: Bit, m1: Bit):
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
        f"// Generated using: PECOS version {__version__}\n"
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

    assert prog.qasm() == qasm


def test_strange_program():
    """Test a weird program to verify we get what is expected for various other SLR objects."""

    prog = Main(
        q := QReg("q", 2),
        c := CReg("c", 4),
        Repeat(3).block(
            Assign(c, 3),
        ),
        Comment("Here is some injected QASM:"),
        QASM("c = 0 & 1;"),
        Permute([q[0], q[1]], [q[1], q[0]], comment=False),
        p.H(q[0]),
    )

    qasm = (
        "OPENQASM 2.0;\n"
        'include "hqslib1.inc";\n'
        f"// Generated using: PECOS version {__version__}\n"
        "qreg q[2];\n"
        "creg c[4];\n"
        "c = 3;\n"
        "c = 3;\n"
        "c = 3;\n"
        "// Here is some injected QASM:\n"
        "c = 0 & 1;\n"
        "h q[1];"
    )

    # TODO: Weird things can happen with Permute... if you run a program twice

    assert prog.qasm() == qasm
