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

from pecos.slr import CReg, QReg, util


class ThreeParallelFlaggingXZZ:
    """
    Args:
        data (QReg[7]):
        ancillas (QReg[3]):
        flag_x (CReg[3]):
        flag_z (CReg[3]):
        flags (CReg[3]):
        last_raw_syn_x (CReg[3]):
        last_raw_syn_z (CReg[3]):
    """

    def __init__(
        self,
        data: QReg,
        ancillas: QReg,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
    ):
        self.data = data
        self.ancillas = ancillas
        self.flag_x = flag_x
        self.flag_z = flag_z
        self.flags = flags
        self.last_raw_syn_x = last_raw_syn_x
        self.last_raw_syn_z = last_raw_syn_z

    def qasm(self):
        d = self.data
        a = self.ancillas
        q = [a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]]
        flag_x = self.flag_x
        flag_z = self.flag_z
        flags = self.flags
        last_raw_syn_x = self.last_raw_syn_x
        last_raw_syn_z = self.last_raw_syn_z

        qasm = f"""
        {flag_x} = 0;
        {flag_z} = 0;

        // X check 1, Z check 2, Z check 3
        // ===============================

        reset {q[0]};
        reset {q[8]};
        reset {q[9]};

        h {q[0]};
        h {q[8]};
        h {q[9]};

        cx {q[0]},{q[4]};  // 5 -> 4
        cz {q[8]},{q[6]};  // 6 -> 6
        cz {q[9]},{q[3]};  // 7 -> 3

        barrier {q[0]},{q[8]};
        cz {q[0]},{q[8]};
        barrier {q[0]},{q[8]};

        cx {q[0]},{q[1]};  // 1 -> 1
        cz {q[8]},{q[5]};  // 2 -> 5
        cz {q[9]},{q[4]};  // 5 -> 4

        cx {q[0]},{q[2]};  // 3 -> 2
        cz {q[8]},{q[3]};  // 7 -> 3
        cz {q[9]},{q[7]};  // 4 -> 7

        barrier {q[0]},{q[9]};
        cz {q[0]},{q[9]};
        barrier {q[0]},{q[9]};

        cx {q[0]},{q[3]};  // 7 -> 3
        cz {q[8]},{q[2]};  // 3 -> 2
        cz {q[9]},{q[6]};  // 6 -> 6

        h {q[0]};
        h {q[8]};
        h {q[9]};

        measure {q[0]} -> {flag_x[0]};
        measure {q[8]} -> {flag_z[1]};
        measure {q[9]} -> {flag_z[2]};

        {flag_x[0]} = {flag_x[0]} ^ {last_raw_syn_x[0]};
        {flag_z[1]} = {flag_z[1]} ^ {last_raw_syn_z[1]};
        {flag_z[2]} = {flag_z[2]} ^ {last_raw_syn_z[2]};

        {flags} = {flag_x} | {flag_z};
        """

        return util.rm_white_space(qasm)


class ThreeParallelFlaggingZXX:
    def __init__(self, data, ancillas, flag_x, flag_z, flags, last_raw_syn_x, last_raw_syn_z):
        self.data = data
        self.ancillas = ancillas
        self.flag_x = flag_x
        self.flag_z = flag_z
        self.flags = flags
        self.last_raw_syn_x = last_raw_syn_x
        self.last_raw_syn_z = last_raw_syn_z

    def qasm(self):
        d = self.data
        a = self.ancillas
        q = [a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]]
        flag_x = self.flag_x
        flag_z = self.flag_z
        flags = self.flags
        last_raw_syn_x = self.last_raw_syn_x
        last_raw_syn_z = self.last_raw_syn_z

        qasm = f"""
        // Z check 1, X check 2, X check 3
        // ===============================

        reset {q[0]};
        reset {q[8]};
        reset {q[9]};

        h {q[0]};
        h {q[8]};
        h {q[9]};


        barrier {q[0]},{q[4]};
        cz {q[0]},{q[4]};  // 5 -> 4
        barrier {q[0]},{q[4]};

        barrier {q[8]},{q[6]};
        cx {q[8]},{q[6]};  // 6 -> 6
        barrier {q[8]},{q[6]};

        barrier {q[9]},{q[3]};
        cx {q[9]},{q[3]};  // 7 -> 3
        barrier {q[9]},{q[3]};



        barrier {a[0]}, {d[0]}, {d[1]}, {d[2]}, {d[3]}, {d[4]}, {d[5]}, {d[6]}, {a[1]}, {a[2]};
        cz {q[8]},{q[0]};
        barrier {a[0]}, {d[0]}, {d[1]}, {d[2]}, {d[3]}, {d[4]}, {d[5]}, {d[6]}, {a[1]}, {a[2]};


        barrier {q[0]},{q[1]};
        cz {q[0]},{q[1]};  // 1 -> 1
        barrier {q[0]},{q[1]};

        barrier {q[8]},{q[5]};
        cx {q[8]},{q[5]};  // 2 -> 5
        barrier {q[8]},{q[5]};

        barrier {q[9]},{q[4]};
        cx {q[9]},{q[4]};  // 5 -> 4
        barrier {q[9]},{q[4]};



        barrier {q[0]},{q[2]};
        cz {q[0]},{q[2]};  // 3 -> 2
        barrier {q[0]},{q[2]};

        barrier {q[8]},{q[3]};
        cx {q[8]},{q[3]};  // 7 -> 3
        barrier {q[8]},{q[3]};

        barrier {q[9]},{q[7]};
        cx {q[9]},{q[7]};  // 4 -> 7
        barrier {q[9]},{q[7]};


        barrier {a[0]}, {d[0]}, {d[1]}, {d[2]}, {d[3]}, {d[4]}, {d[5]}, {d[6]}, {a[1]}, {a[2]};
        cz {q[9]},{q[0]};
        barrier {a[0]}, {d[0]}, {d[1]}, {d[2]}, {d[3]}, {d[4]}, {d[5]}, {d[6]}, {a[1]}, {a[2]};



        barrier {q[0]},{q[3]};
        cz {q[0]},{q[3]};  // 7 -> 3
        barrier {q[0]},{q[3]};

        barrier {q[8]},{q[2]};
        cx {q[8]},{q[2]};  // 3 -> 2
        barrier {q[8]},{q[2]};

        barrier {q[9]},{q[6]};
        cx {q[9]},{q[6]};  // 6 -> 6
        barrier {q[9]},{q[6]};


        h {q[0]};
        h {q[8]};
        h {q[9]};



        measure {q[0]} -> {flag_z[0]};
        measure {q[8]} -> {flag_x[1]};
        measure {q[9]} -> {flag_x[2]};

        // XOR flags/syndromes
        {flag_z[0]} = {flag_z[0]} ^ {last_raw_syn_z[0]};
        {flag_x[1]} = {flag_x[1]} ^ {last_raw_syn_x[1]};
        {flag_x[2]} = {flag_x[2]} ^ {last_raw_syn_x[2]};

        {flags} = {flag_x} | {flag_z};
        """

        return util.rm_white_space(qasm)
