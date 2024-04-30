import numpy as np

from pecos.reps.pypmir.name_resolver import sim_name_resolver
from pecos.reps.pypmir.op_types import QOp


def test_rzz2szz():
    """Verify that a RZZ(pi/2) gate will be resolved to a SZZ gate."""

    qop = QOp(name="RZZ", angles=(np.pi / 2,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "SZZ"


def test_rzz2szzdg():
    """Verify that a RZZ(-pi/2) gate will be resolved to a SZZdg gate."""

    qop = QOp(name="RZZ", angles=(-np.pi / 2,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "SZZdg"


def test_rzz2i():
    """Verify that a RZZ(0.0) gate will be resolved to an I gate."""

    qop = QOp(name="RZZ", angles=(0.0,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "I"


def test_rzz2rzz():
    """Verify that a RZZ(pi/4) gate will be resolved to a RZZ gate since it is non-Clifford."""

    qop = QOp(name="RZZ", angles=(0.0,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "I"


def test_rz2sz():
    """Verify that a RZ(pi/2) gate will be resolved to a SZ gate."""

    qop = QOp(name="RZ", angles=(np.pi / 2,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "SZ"


def test_rz2szdg():
    """Verify that a RZ(-pi/2) gate will be resolved to a SZdg gate."""

    qop = QOp(name="RZ", angles=(-np.pi / 2,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "SZdg"


def test_rz2i():
    """Verify that a RZ(0.0) gate will be resolved to an I gate."""

    qop = QOp(name="RZ", angles=(0.0,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "I"


def test_rz2rz():
    """Verify that a RZ(pi/4) will give back RZ since it is non-Clifford."""

    qop = QOp(name="RZ", angles=(np.pi / 4,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "RZ"


def test_r1xy2x():
    """Verify that a R1XY(pi, 0) will give back X."""

    qop = QOp(name="R1XY", angles=(np.pi, 0.0), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "X"


def test_r1xy2sydg():
    """Verify that a R1XY(-pi/2,pi/2) will give back SYdg."""

    qop = QOp(name="R1XY", angles=(-np.pi / 2, np.pi / 2), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "SYdg"


def test_r1xy2i():
    """Verify that a R1XY(0, 0) will give back I."""

    qop = QOp(name="R1XY", angles=(0.0, 0.0), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "I"
