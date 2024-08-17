from pecos.qeclib.steane.decoders.lookup import (
    FlagLookupQASM,
    FlagLookupQASMActiveCorrectionX,
    FlagLookupQASMActiveCorrectionZ,
)
from pecos.slr import CReg, QReg


def test_FlagLookupQASM(compare_qasm):
    syn = CReg("syn_test", 3)
    syndromes = CReg("syndromes_test", 3)
    raw_syn = CReg("raw_syn_test", 3)
    pf = CReg("pf_test", 2)
    flag = CReg("flag_test", 1)
    flags = CReg("flags_test", 3)
    scratch = CReg("scratch_test", 32)

    for basis in ["X", "Y", "Z"]:
        block = FlagLookupQASM(basis, syn, syndromes, raw_syn, pf[0], flag, flags, scratch)
        compare_qasm(block, basis)

def test_FlagLookupQASMActiveCorrectionX(compare_qasm):
    q = QReg("q_test", 7)
    syn = CReg("syn_test", 3)
    syndromes = CReg("syndromes_test", 3)
    raw_syn = CReg("raw_syn_test", 3)
    pf = CReg("pf_test", 2)
    flag = CReg("flag_test", 1)
    flags = CReg("flags_test", 3)
    scratch = CReg("scratch_test", 32)

    for pf_bit_copy in [True, False]:
        block = FlagLookupQASMActiveCorrectionX(q, syn, syndromes, raw_syn, pf[0], flag, flags, scratch, pf_bit_copy)
        compare_qasm(block, pf_bit_copy)

def test_FlagLookupQASMActiveCorrectionZ(compare_qasm):
    q = QReg("q_test", 7)
    syn = CReg("syn_test", 3)
    syndromes = CReg("syndromes_test", 3)
    raw_syn = CReg("raw_syn_test", 3)
    pf = CReg("pf_test", 2)
    flag = CReg("flag_test", 1)
    flags = CReg("flags_test", 3)
    scratch = CReg("scratch_test", 32)

    for pf_bit_copy in [True, False]:
        block = FlagLookupQASMActiveCorrectionZ(q, syn, syndromes, raw_syn, pf[0], flag, flags, scratch, pf_bit_copy)
        compare_qasm(block, pf_bit_copy)

