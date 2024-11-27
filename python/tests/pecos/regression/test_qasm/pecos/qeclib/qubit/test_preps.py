from pecos.qeclib import qubit
from pecos.slr import QReg


def test_Prep(compare_qasm):
    q = QReg("q_test", 1)

    prog = qubit.Prep(q[0])
    compare_qasm(prog)
