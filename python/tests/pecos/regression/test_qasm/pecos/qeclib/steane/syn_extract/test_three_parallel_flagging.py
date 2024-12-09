from pecos.qeclib.steane.syn_extract.three_parallel_flagging import (
    ThreeParallelFlaggingXZZ,
    ThreeParallelFlaggingZXX,
)
from pecos.slr import CReg, QReg


def test_ThreeParallelFlaggingXZZ(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    block = ThreeParallelFlaggingXZZ(
        q,
        a,
        flag_x,
        flag_z,
        flags,
        last_raw_syn_x,
        last_raw_syn_z,
    )
    compare_qasm(block)


def test_ThreeParallelFlaggingZXX(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    block = ThreeParallelFlaggingZXX(
        q,
        a,
        flag_x,
        flag_z,
        flags,
        last_raw_syn_x,
        last_raw_syn_z,
    )
    compare_qasm(block)
