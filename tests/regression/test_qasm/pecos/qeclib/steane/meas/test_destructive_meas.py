from pecos.qeclib.steane.meas.destructive_meas import MeasDecode, Measure, MeasureX, MeasureY, MeasureZ, ProcessMeas
from pecos.slr import CReg, QReg


def test_MeasureX(compare_qasm):
    q = QReg("q_test", 7)
    meas_creg = CReg("meas_creg_test", 7)
    log_raw = CReg("log_raw_test", 1)

    for barrier in [True, False]:
        block = MeasureX(q, meas_creg, log_raw, barrier=barrier)
        compare_qasm(block, barrier)

def test_MeasureY(compare_qasm):
    q = QReg("q_test", 7)
    meas_creg = CReg("meas_creg_test", 7)
    log_raw = CReg("log_raw_test", 1)

    for barrier in [True, False]:
        block = MeasureY(q, meas_creg, log_raw, barrier=barrier)
        compare_qasm(block, barrier)

def test_MeasureZ(compare_qasm):
    q = QReg("q_test", 7)
    meas_creg = CReg("meas_creg_test", 7)
    log_raw = CReg("log_raw_test", 1)

    for barrier in [True, False]:
        block = MeasureZ(q, meas_creg, log_raw, barrier=barrier)
        compare_qasm(block, barrier)

def test_Measure(compare_qasm):
    q = QReg("q_test", 7)
    meas_creg = CReg("meas_creg_test", 7)
    log_raw = CReg("log_raw_test", 1)

    for meas_basis in ["X", "Y", "Z"]:
        block = Measure(q, meas_creg, log_raw, meas_basis=meas_basis)
        compare_qasm(block, meas_basis)

def test_ProcessMeas(compare_qasm):
    meas = CReg("meas_test", 7)
    log = CReg("log_test", 2)
    syn_meas = CReg("syn_meas_test", 3)
    pf = CReg("pf_test", 2)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_y = CReg("last_raw_syn_y_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    for basis in ["X", "Y", "Z"]:
        for check_type in ["xy", "xz", "yz"]:
            block = ProcessMeas(basis, meas, log[0], log[1], syn_meas, pf[0], pf[1], check_type, last_raw_syn_x,
                                last_raw_syn_y, last_raw_syn_z)
            compare_qasm(block, basis, check_type)

def test_MeasDecode(compare_qasm):
    q = QReg("q_test", 7)
    meas = CReg("meas_test", 7)
    log = CReg("log_test", 2)
    syn_meas = CReg("syn_meas_test", 3)
    pf = CReg("pf_test", 2)
    last_raw_syn_x = CReg("last_raw_syn_x_test", 3)
    last_raw_syn_z = CReg("last_raw_syn_z_test", 3)

    for meas_basis in ["X", "Y", "Z"]:
        block = MeasDecode(q, meas_basis, meas, log[0], log[1], syn_meas, pf[0], pf[1], last_raw_syn_x, last_raw_syn_z)
        compare_qasm(block, meas_basis)
