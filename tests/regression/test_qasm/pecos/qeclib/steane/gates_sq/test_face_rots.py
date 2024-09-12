from pecos.qeclib.steane.gates_sq.face_rots import F, Fdg
from pecos.slr import QReg


def test_F(compare_qasm):
    q = QReg("q_test", 7)

    block = F(q)
    compare_qasm(block)


def test_Fdg(compare_qasm):
    q = QReg("q_test", 7)

    block = Fdg(q)
    compare_qasm(block)
