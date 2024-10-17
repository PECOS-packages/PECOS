from pecos.qeclib import qubit
from pecos.slr import QReg


def test_CH(compare_qasm):
    q = QReg("q_test", 4)
    prog = qubit.CH(q[1], q[3])
    compare_qasm(prog)
