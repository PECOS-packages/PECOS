from pecos.qeclib.steane.preps.encoding_circ import EncodingCircuit, EncodingCircuit2
from pecos.slr import QReg


def test_EncodingCircuit(compare_qasm):
    q = QReg("q_test", 7)

    block = EncodingCircuit(q)
    compare_qasm(block)

def test_EncodingCircuit2(compare_qasm):
    q = QReg("q_test", 7)

    block = EncodingCircuit2(q)
    compare_qasm(block)
