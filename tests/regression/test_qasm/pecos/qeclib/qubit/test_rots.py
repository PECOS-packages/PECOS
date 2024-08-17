from numpy import pi
from pecos.qeclib import qubit
from pecos.slr import QReg


def test_RX(compare_qasm):
    q = QReg("q_test", 1)
    prog = qubit.RX[pi/3](q[0])
    compare_qasm(prog)

def test_RY(compare_qasm):
    q = QReg("q_test", 1)
    prog = qubit.RY[pi/3](q[0])
    compare_qasm(prog)

def test_RZ(compare_qasm):
    q = QReg("q_test", 1)
    prog = qubit.RZ[pi/3](q[0])
    compare_qasm(prog)

def test_RZZ(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.RZZ[pi/3](q[1], q[3])
    compare_qasm(prog)
