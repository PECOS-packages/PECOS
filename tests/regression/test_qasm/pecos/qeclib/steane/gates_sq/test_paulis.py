from pecos.qeclib.steane.gates_sq.paulis import X, Y, Z
from pecos.slr import QReg


def test_X(compare_qasm):
    q = QReg("q_test", 7)

    block = X(q)
    compare_qasm(block)


def test_Y(compare_qasm):
    q = QReg("q_test", 7)

    block = Y(q)
    compare_qasm(block)


def test_Z(compare_qasm):
    q = QReg("q_test", 7)

    block = Z(q)
    compare_qasm(block)
