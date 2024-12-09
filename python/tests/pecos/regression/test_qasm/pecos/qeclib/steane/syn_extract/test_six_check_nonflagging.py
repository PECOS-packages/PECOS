from pecos.qeclib.steane.syn_extract.six_check_nonflagging import SixUnflaggedSyn
from pecos.slr import CReg, QReg


def test_SixUnflaggedSyn(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    syn_x = CReg("syn_x_test", 3)
    syn_z = CReg("syn_z_test", 3)

    block = SixUnflaggedSyn(q, a, syn_x, syn_z)
    compare_qasm(block)
