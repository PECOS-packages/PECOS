from pecos.qeclib import qubit
from pecos.slr import QReg


def test_SX(compare_qasm):
    q = QReg("q_test", 2)
    prog = qubit.SX(q[1])
    compare_qasm(prog)


def test_SXdg(compare_qasm):
    q = QReg("q_test", 2)
    prog = qubit.SXdg(q[1])
    compare_qasm(prog)


def test_SY(compare_qasm):
    q = QReg("q_test", 2)
    prog = qubit.SY(q[1])
    compare_qasm(prog)


def test_SYdg(compare_qasm):
    q = QReg("q_test", 2)
    prog = qubit.SYdg(q[1])
    compare_qasm(prog)


def test_SZ(compare_qasm):
    q = QReg("q_test", 2)
    prog = qubit.SZ(q[1])
    compare_qasm(prog)


def test_SZdg(compare_qasm):
    q = QReg("q_test", 2)
    prog = qubit.SZdg(q[1])
    compare_qasm(prog)
