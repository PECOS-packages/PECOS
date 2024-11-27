from pecos.qeclib.steane.meas.measure_x import NoFlagMeasureX
from pecos.slr import CReg, QReg


def test_MeasureX(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    out = CReg("out_test", 1)

    block = NoFlagMeasureX(q, a, out)
    compare_qasm(block)
