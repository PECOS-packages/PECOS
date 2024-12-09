from pecos.qeclib.steane.gates_sq.hadamards import H
from pecos.slr import QReg


def test_H(compare_qasm):
    q = QReg("q_test", 7)

    block = H(q)
    compare_qasm(block)
