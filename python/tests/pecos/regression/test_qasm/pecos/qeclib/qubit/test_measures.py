from pecos.qeclib import qubit
from pecos.slr import CReg, QReg


def test_Measure(compare_qasm):
    q = QReg("q_test", 1)
    m = CReg("m_test", 1)

    prog = qubit.Measure(q[0]) > m[0]
    compare_qasm(prog)
