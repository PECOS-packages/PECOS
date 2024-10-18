from pecos.qeclib.steane.preps.t_plus_state import (
    PrepEncodeTDagPlusNonFT,
    PrepEncodeTPlusFT,
    PrepEncodeTPlusFTRUS,
    PrepEncodeTPlusNonFT,
)
from pecos.slr import CReg, QReg


def test_PrepEncodeTPlusNonFT(compare_qasm):
    q = QReg("q_test", 7)
    block = PrepEncodeTPlusNonFT(q)
    compare_qasm(block)


def test_PrepEncodeTDagPlusNonFT(compare_qasm):
    q = QReg("q_test", 7)
    block = PrepEncodeTDagPlusNonFT(q)
    compare_qasm(block)


def test_PrepEncodeTPlusFT(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    out = CReg("out_test", 2)
    reject = CReg("reject_test", 1)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)
    block = PrepEncodeTPlusFT(
        q,
        a,
        out,
        reject[0],
        flag_x,
        flag_z,
        flags,
        last_raw_syn_x,
        last_raw_syn_z,
    )
    compare_qasm(block)


def test_PrepEncodeTPlusFTRUS(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 3)
    out = CReg("out_test", 2)
    reject = CReg("reject_test", 1)
    flag_x = CReg("flag_x_test", 3)
    flag_z = CReg("flag_z_test", 3)
    flags = CReg("flags_test", 3)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    for limit in [1, 2, 3]:
        block = PrepEncodeTPlusFTRUS(
            q,
            a,
            out,
            reject[0],
            flag_x,
            flag_z,
            flags,
            last_raw_syn_x,
            last_raw_syn_z,
            limit,
        )
        compare_qasm(block, limit)
