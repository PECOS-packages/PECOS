from pecos.qeclib import qubit
from pecos.slr import QReg


def test_H(compare_qasm):
    q = QReg("q_test", 2)

    prog = qubit.H(q[1])
    compare_qasm(prog)
