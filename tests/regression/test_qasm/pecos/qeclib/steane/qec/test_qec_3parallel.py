from pecos.qeclib.steane.qec.qec_3parallel import ParallelFlagQECActiveCorrection
from pecos.slr import CReg, QReg


def test_ParallelFlagQECActiveCorrection(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    syn_x = CReg("syn_x_test", 3)
    syn_z = CReg("syn_z_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)
    syndromes = CReg("syndromes_test", 3)
    pf = CReg("pf_test", 2)
    scratch = CReg("scratch_test", 32)

    block = ParallelFlagQECActiveCorrection(q, a, flag_x, flag_z, flags, syn_x, syn_z, last_raw_syn_x, last_raw_syn_z,
                                            syndromes, pf[0], pf[1], scratch)
    compare_qasm(block)
