from pecos.qeclib.steane.preps.plus_h_state import PrepHStateFT
from pecos.slr import CReg, QReg


def test_PrepHStateFT(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    out = CReg("out_test", 2)
    reject = CReg("reject_test", 1)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)
    block = PrepHStateFT(q, a, out, reject[0], flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z)
    compare_qasm(block)
