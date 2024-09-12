from pecos.qeclib.steane.preps.encoding_circ import EncodingCircuit
from pecos.slr import QReg


def test_EncodingCircuit(compare_qasm):
    q = QReg("q_test", 7)

    block = EncodingCircuit(q)
    compare_qasm(block)
