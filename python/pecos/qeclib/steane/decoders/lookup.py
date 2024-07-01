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

from pecos.slr import Bit, util


class FlagLookupQASM:
    def __init__(self, basis, syn, syndromes, raw_syn, pf: Bit, flag, flags, scratch):
        # qasm_syn_decoder('X', syn_x, flag_x, 'last_raw_syn_x', 'pf_z1')
        # qasm_syn_decoder(basis_check, syn, flag, raw_syn, pf, pf_index=0)
        self.basis = basis
        self.syn = syn
        self.syndromes = syndromes
        self.raw_syn = raw_syn
        self.pf = pf
        self.flag = flag
        self.flags = flags
        self.scratch = scratch

    def qasm(self):
        syn = self.syn  # syn_x
        syndromes = self.syndromes  # syndromes
        raw_syn = self.raw_syn  # last_raw_syn_x
        pf = self.pf  # pf_z1[0]
        flag = self.flag  # flag_x
        flags = self.flags  # flags
        scratch = self.scratch  # scratch

        qasm = f"""
        // =========================
        // BEGIN Run {self.basis} decoder
        // =========================

        if({flags}!=0) {syndromes} = {syn} ^ {raw_syn};
        if({flags}==0) {syndromes} = 0;

        // apply corrections
        if({syndromes} == 2) {pf} = {pf} ^ 1;
        if({syndromes} == 4) {pf} = {pf} ^ 1;
        if({syndromes} == 6) {pf} = {pf} ^ 1;

        // alter correction based on flags
        // ===============================

        // 1&2 (1 -> 2)
        // ------------
        {scratch} = 0;
        if({flag} == 1) {scratch[0]} = 1;
        if({syndromes} == 2) {scratch[1]} = 1;

        {scratch[2]} = {scratch[0]} & {scratch[1]};
        if({scratch[2]} == 1) {pf} = {pf} ^ 1;

        // 1&4 (1 -> 3)
        // ------------
        {scratch} = 0;
        if({flag} == 1) {scratch[0]} = 1;
        if({syndromes} == 4) {scratch[1]} = 1;

        {scratch[2]} = {scratch[0]} & {scratch[1]};
        if({scratch[2]} == 1) {pf} = {pf} ^ 1;


        // 6&4 (2,3 -> 3)
        // ------------
        {scratch} = 0;
        if({flag} == 6) {scratch[0]} = 1;
        if({syndromes} == 4) {scratch[1]} = 1;

        {scratch[2]} = {scratch[0]} & {scratch[1]};
        if({scratch[2]} == 1) {pf} = {pf} ^ 1;

        if({flags}!=0) {raw_syn} = {syn};

        // =========================
        // END Run {self.basis} decoder
        // =========================
        """

        return util.rm_white_space(qasm)


class FlagLookupQASMActiveCorrectionX:
    def __init__(self, qubits, syn, syndromes, raw_syn, pf, flag, flags, scratch, pf_bit_copy: Bit = None):
        # qasm_syn_decoder('X', syn_x, flag_x, 'last_raw_syn_x', 'pf_z1')
        # qasm_syn_decoder(basis_check, syn, flag, raw_syn, pf, pf_index=0)
        self.qubits = qubits
        self.syn = syn
        self.syndromes = syndromes
        self.raw_syn = raw_syn
        self.pf = pf
        self.flag = flag
        self.flags = flags
        self.scratch = scratch
        self.pf_bit_copy = pf_bit_copy  # copy of pf_z bit

    def qasm(self):
        q = self.qubits

        qasm = FlagLookupQASM(
            basis="X",
            syn=self.syn,
            syndromes=self.syndromes,
            raw_syn=self.raw_syn,
            pf=self.pf,
            flag=self.flag,
            flags=self.flags,
            scratch=self.scratch,
        ).qasm()

        qasm_active = ""

        if self.pf_bit_copy:
            qasm_active += f"""
            // copy Pauli frame
            {self.pf_bit_copy} = {self.pf};
            """

        qasm_active += f"""

        // ACTIVE ERROR CORRECTION FOR X SYNDROMES

        {self.scratch}  = 0;

        if({self.syndromes[0]} == 1) {self.scratch} = {self.scratch}  ^ 1;  // only part that differs for X vs Z syns
        if({self.syndromes[1]} == 1) {self.scratch}  = {self.scratch}  ^ 12;
        if({self.syndromes[2]} == 1) {self.scratch}  = {self.scratch}  ^ 48;

        if({self.pf}==1) {self.scratch}  = {self.scratch}  ^ 112;  // logical operator

        if({self.scratch[0]} == 1) z {q[0]};
        // if({self.scratch[1]} == 1) z {q[1]};  // not possible for X stabilizers
        if({self.scratch[2]} == 1) z {q[2]};
        if({self.scratch[3]} == 1) z {q[3]};
        if({self.scratch[4]} == 1) z {q[4]};
        if({self.scratch[5]} == 1) z {q[5]};
        if({self.scratch[6]} == 1) z {q[6]};

        {self.pf} = 0;
        // {self.syndromes} = 0;
        {self.raw_syn} = 0;
        // {self.syn} = 0;
        // {self.flag} = 0;
        // {self.flags} = 0;

        """

        qasm_list = util.str2list(qasm)
        qasm_list.extend(util.str2list(qasm_active))

        return util.list2str(qasm_list)


class FlagLookupQASMActiveCorrectionZ:
    def __init__(self, qubits, syn, syndromes, raw_syn, pf, flag, flags, scratch, pf_bit_copy: Bit = None):
        # qasm_syn_decoder('X', syn_x, flag_x, 'last_raw_syn_x', 'pf_z1')
        # qasm_syn_decoder(basis_check, syn, flag, raw_syn, pf, pf_index=0)
        self.qubits = qubits
        self.syn = syn
        self.syndromes = syndromes
        self.raw_syn = raw_syn
        self.pf = pf
        self.flag = flag
        self.flags = flags
        self.scratch = scratch
        self.pf_bit_copy = pf_bit_copy  # copy of pf_x bit

    def qasm(self):
        q = self.qubits

        qasm = FlagLookupQASM(
            basis="Z",
            syn=self.syn,
            syndromes=self.syndromes,
            raw_syn=self.raw_syn,
            pf=self.pf,
            flag=self.flag,
            flags=self.flags,
            scratch=self.scratch,
        ).qasm()

        qasm_active = ""

        if self.pf_bit_copy:
            qasm_active += f"""
            // copy Pauli frame
            {self.pf_bit_copy} = {self.pf};
            """

        qasm_active += f"""

        // ACTIVE ERROR CORRECTION FOR Z SYNDROMES

        {self.scratch}  = 0;

        if({self.syndromes[0]} == 1) {self.scratch} = {self.scratch}  ^ 14;  // only part that differs for X vs Z syns
        if({self.syndromes[1]} == 1) {self.scratch}  = {self.scratch}  ^ 12;
        if({self.syndromes[2]} == 1) {self.scratch}  = {self.scratch}  ^ 48;

        if({self.pf}==1) {self.scratch}  = {self.scratch}  ^ 112;  // logical operator

        // if({self.scratch[0]} == 1) z {q[0]}; // not possible for X stabilizers
        if({self.scratch[1]} == 1) x {q[1]};
        if({self.scratch[2]} == 1) x {q[2]};
        if({self.scratch[3]} == 1) x {q[3]};
        if({self.scratch[4]} == 1) x {q[4]};
        if({self.scratch[5]} == 1) x {q[5]};
        if({self.scratch[6]} == 1) x {q[6]};

        {self.pf} = 0;
        // {self.syndromes} = 0;
        {self.raw_syn} = 0;
        // {self.syn} = 0;
        // {self.flag} = 0;
        // {self.flags} = 0;
        """

        qasm_list = util.str2list(qasm)
        qasm_list.extend(util.str2list(qasm_active))

        return util.list2str(qasm_list)
