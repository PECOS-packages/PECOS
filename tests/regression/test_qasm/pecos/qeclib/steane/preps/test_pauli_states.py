from pecos.qeclib.steane.preps.pauli_states import (
    LogZeroRot,
    PrepEncodingFTZero,
    PrepEncodingNonFTZero,
    PrepRUS,
    PrepZeroVerify,
)
from pecos.slr import CReg, QReg


def test_PrepEncodingNonFTZero(compare_qasm):
    q = QReg("q_test", 7)
    block = PrepEncodingNonFTZero(q)
    compare_qasm(block)


def test_PrepZeroVerify(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    init_bit = CReg("init_bit", 1)
    for reset_ancilla in [True, False]:
        block = PrepZeroVerify(q, a[0], init_bit[0], reset_ancilla=reset_ancilla)
        compare_qasm(block, reset_ancilla)


def test_PrepEncodingFTZero(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    init_bit = CReg("init_bit_test", 1)

    for reset in [True, False]:
        block = PrepEncodingFTZero(q, a[0], init_bit[0], reset=reset)
        compare_qasm(block, reset)


def test_PrepRUS(compare_qasm):
    q = QReg("q_test", 7)
    a = QReg("a_test", 1)
    init = CReg("init_test", 1)

    for limit in [1, 2, 3]:
        for state in ["-Z", "+Z", "+X", "-X", "+Y", "-Y"]:
            for first_round_reset in [True, False]:
                block = PrepRUS(q, a[0], init[0], limit, state, first_round_reset=first_round_reset)
                compare_qasm(block, limit, state, first_round_reset)


def test_LogZeroRot(compare_qasm):
    q = QReg("q_test", 7)

    for state in ["-Z", "+Z", "+X", "-X", "+Y", "-Y"]:
        block = LogZeroRot(q, state)
        compare_qasm(block, state)
