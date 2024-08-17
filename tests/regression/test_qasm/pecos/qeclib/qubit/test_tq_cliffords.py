from pecos.qeclib import qubit
from pecos.slr import QReg


def test_CX(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.CX(q[1], q[3])
    compare_qasm(prog)

def test_CY(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.CY(q[1], q[3])
    compare_qasm(prog)

def test_CZ(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.CZ(q[1], q[3])
    compare_qasm(prog)

def test_SXX(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.SXX(q[1], q[3])
    compare_qasm(prog)

def test_SXXdg(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.SXXdg(q[1], q[3])
    compare_qasm(prog)

def test_SYY(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.SYY(q[1], q[3])
    compare_qasm(prog)

def test_SYYdg(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.SYYdg(q[1], q[3])
    compare_qasm(prog)

def test_SZZ(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.SZZ(q[1], q[3])
    compare_qasm(prog)

def test_SZZdg(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.SZZdg(q[1], q[3])
    compare_qasm(prog)

