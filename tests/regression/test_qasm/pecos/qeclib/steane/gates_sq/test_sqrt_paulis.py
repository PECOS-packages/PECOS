from pecos.qeclib.steane.gates_sq.sqrt_paulis import SX, SY, SZ, SXdg, SYdg, SZdg
from pecos.slr import QReg


def test_SX(compare_qasm):
    q = QReg("q_test", 7)

    block = SX(q)
    compare_qasm(block)

def test_SXdg(compare_qasm):
    q = QReg("q_test", 7)

    block = SXdg(q)
    compare_qasm(block)

def test_SY(compare_qasm):
    q = QReg("q_test", 7)

    block = SY(q)
    compare_qasm(block)

def test_SYdg(compare_qasm):
    q = QReg("q_test", 7)

    block = SYdg(q)
    compare_qasm(block)

def test_SZ(compare_qasm):
    q = QReg("q_test", 7)

    block = SZ(q)
    compare_qasm(block)

def test_SZdg(compare_qasm):
    q = QReg("q_test", 7)

    block = SZdg(q)
    compare_qasm(block)
