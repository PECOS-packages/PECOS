from pecos.qeclib.steane.gates_tq.transversal_tq import CX, CY, CZ, SZZ
from pecos.slr import QReg


def test_CX(compare_qasm):
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    for barrier in [True, False]:
        block = CX(q1, q2, barrier=barrier)
        compare_qasm(block, barrier)


def test_CY(compare_qasm):
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    block = CY(q1, q2)
    compare_qasm(block)


def test_CZ(compare_qasm):
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    block = CZ(q1, q2)
    compare_qasm(block)


def test_SZZ(compare_qasm):
    q1 = QReg("q1_test", 7)
    q2 = QReg("q2_test", 7)

    block = SZZ(q1, q2)
    compare_qasm(block)
