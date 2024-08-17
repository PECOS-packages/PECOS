from pecos.qeclib import qubit
from pecos.slr import QReg


def test_T(compare_qasm):
    q = QReg("q_test", 2)

    prog = qubit.T(q[1])
    compare_qasm(prog)


def test_Tdg(compare_qasm):
    q = QReg("q_test", 2)

    prog = qubit.Tdg(q[1])
    compare_qasm(prog)
