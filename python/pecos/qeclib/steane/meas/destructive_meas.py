# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.qeclib.steane.gates_sq.sqrt_paulis import SX, SYdg
from pecos.slr import Block, Comment, CReg, QReg, util


class MeasureX(Block):
    """Measure in the logical X basis.

    Need: X_L -> Z_L

    SYdg: X->Z, Y->Y, Z->-X
    """

    def __init__(self,
                 qubits: QReg,
                 meas_creg: CReg,
                 log_raw: CReg,
                 *,
                 barrier: bool = True):
        super().__init__()

        self.extend(
            SYdg(qubits),
            MeasureZ(qubits, meas_creg, log_raw, barrier=barrier),
        )


class MeasureY(Block):
    """Measure in the logical Y basis.

    Need: Y_L -> Z_L

    SX: X->X, Y->Z, Z->-Y
    """

    def __init__(self, qubits: QReg, meas_creg: CReg, log_raw: CReg, *, barrier: bool = True):
        super().__init__()

        self.extend(
            SX(qubits),  # logical SX == physical SXdg gates
            MeasureZ(qubits, meas_creg, log_raw, barrier=barrier),
        )


class MeasureZ:
    """Measure in the logical Z basis."""

    def __init__(self, qubits: QReg, meas: CReg, log_raw: CReg, *, barrier: bool = True):
        self.qubits = qubits
        self.meas = meas
        self.log_raw = log_raw
        self.barrier = barrier

    def qasm(self):
        q = self.qubits
        m = self.meas
        log_raw = self.log_raw

        qasm = ""

        if self.barrier:
            qasm += f"\nbarrier {q};\n"

        qasm += f"""
        measure {q[0]} -> {m[0]};
        measure {q[1]} -> {m[1]};
        measure {q[2]} -> {m[2]};
        measure {q[3]} -> {m[3]};
        measure {q[4]} -> {m[4]};
        measure {q[5]} -> {m[5]};
        measure {q[6]} -> {m[6]};

        // determine raw logical output
        // ============================
        {log_raw} = {m[4]} ^ {m[5]} ^ {m[6]};

        """
        return util.rm_white_space(qasm)


class Measure(Block):
    def __init__(self, q: QReg, meas_creg: CReg, log_raw: CReg, meas_basis: str):
        super().__init__()

        if meas_basis == "X":
            self.extend(
                Comment("Destructive logical X measurement"),
                MeasureX(q, meas_creg, log_raw),
            )
        elif meas_basis == "Y":
            self.extend(
                Comment("Destructive logical Y measurement"),
                MeasureY(q, meas_creg, log_raw),
            )
        elif meas_basis == "Z":
            self.extend(
                Comment("Destructive logical Z measurement"),
                MeasureZ(q, meas_creg, log_raw),
            )
        else:
            msg = f"Logical measurement in '{meas_basis}' basis is not supported."
            raise Exception(msg)


class ProcessMeas:
    """Process measurement results to determine additional corrections. Applies these and previous corrections to
    logical measurement."""

    def __init__(
        self,
        basis,
        meas,
        log_raw_bit,
        log_bit,
        syn_meas,
        pf_x,
        pf_z,
        check_type="xz",
        last_raw_syn_x=None,
        last_raw_syn_y=None,
        last_raw_syn_z=None,
        *,
        ft_meas=True,
    ):
        self.meas = meas
        self.log_raw = log_raw_bit
        self.log = log_bit
        self.syn_meas = syn_meas
        self.basis = basis
        self.check_type = check_type
        self.last_raw_syn_x = last_raw_syn_x
        self.last_raw_syn_y = last_raw_syn_y
        self.last_raw_syn_z = last_raw_syn_z
        self.pf_x = pf_x
        self.pf_z = pf_z
        self.ft_meas = ft_meas

    def qasm(self):
        meas = self.meas
        log_raw = self.log_raw
        log = self.log
        syn_meas = self.syn_meas
        basis = self.basis
        last_raw_syn_x = self.last_raw_syn_x
        last_raw_syn_y = self.last_raw_syn_y
        last_raw_syn_z = self.last_raw_syn_z

        qasm = f"""
        // =================== //
        // PROCESS MEASUREMENT //
        // =================== //

        // Determine correction to get logical output
        // ==========================================
        {syn_meas[0]} = {meas[0]} ^ {meas[1]} ^ {meas[2]} ^ {meas[3]};
        {syn_meas[1]} = {meas[1]} ^ {meas[2]} ^ {meas[4]} ^ {meas[5]};
        {syn_meas[2]} = {meas[2]} ^ {meas[3]} ^ {meas[5]} ^ {meas[6]};
        """

        qasm_list = util.str2list(qasm)

        qasm_list.append("// XOR syndromes")

        if self.check_type not in {"xy", "xz", "yz"}:
            msg = "QEC type not recognized!"
            raise Exception(msg)

        if basis == "X":
            if "x" in self.check_type:
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_x};")
            else:  # yz
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_y};")
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_z};")

        elif basis == "Y":
            if "y" in self.check_type:
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_y};")
            else:  # xz
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_x};")
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_z};")

        elif basis == "Z":
            if "z" in self.check_type:
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_z};")
            else:  # xy
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_x};")
                qasm_list.append(f"{syn_meas} = {syn_meas} ^ {last_raw_syn_y};")

        else:
            msg = f"Measurement basis must be X, Y, or Z, not {basis}!"
            raise Exception(msg)

        if self.ft_meas:
            cor_qasm = f"""
                // Correct logical output based on measured out syndromes
                {log} = {log_raw};
                if({syn_meas} == 2) {log} = {log} ^ 1;
                if({syn_meas} == 4) {log} = {log} ^ 1;
                if({syn_meas} == 6) {log} = {log} ^ 1;
                """
        else:
            cor_qasm = f"""
                // non-FT measure out
                {log} = {log_raw};
                """

        qasm_list.extend(util.str2list(cor_qasm))

        qasm_list.append("// Apply Pauli frame update (flip the logical output)")

        if basis == "X":
            qasm_list.append("// Update for logical X out")
            qasm_list.append(f"{log} = {log} ^ {self.pf_z};")
        elif basis == "Y":
            qasm_list.append("// Update for logical Y out")
            qasm_list.append(f"{log} = {log} ^ {self.pf_x};")
            qasm_list.append(f"{log} = {log} ^ {self.pf_z};")
        elif basis == "Z":
            qasm_list.append("// Update for logical Z out")
            qasm_list.append(f"{log} = {log} ^ {self.pf_x};")
        else:
            msg = f"Basis `{basis}` not supported!"
            raise Exception(msg)

        return util.rm_white_space(util.list2str(qasm_list))


def MeasDecode(
    q: QReg,
    meas_basis: str,
    meas: CReg,
    log_raw: CReg,
    log: CReg,
    syn_meas: CReg,
    pf_x: CReg,
    pf_z: CReg,
    last_raw_syn_x: CReg,
    last_raw_syn_z: CReg,
) -> Block:
    """Measure out in the appropriate logical basis, determine correction,
    apply to logical output."""
    return Block(
        Measure(q, meas, log_raw, meas_basis),
        ProcessMeas(
            meas_basis,
            meas,
            log_raw,
            log,
            syn_meas,
            pf_x,
            pf_z,
            last_raw_syn_z=last_raw_syn_z,
            last_raw_syn_x=last_raw_syn_x,
        ),
    )
