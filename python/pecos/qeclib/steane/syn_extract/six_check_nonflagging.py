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

from pecos.slr import util


class SixUnflaggedSyn:
    def __init__(self, data, ancillas, syn_x, syn_z):
        self.data = data
        self.ancillas = ancillas
        self.syn_x = syn_x
        self.syn_z = syn_z

    def qasm(self):
        d = self.data
        a = self.ancillas
        q = [a[0], d[0], d[1], d[2], d[3], d[4], d[5], d[6], a[1], a[2]]
        syn_x = self.syn_x
        syn_z = self.syn_z

        qasm = f"""
        // Run the 6 non-flagged checks (if non-trivial flags)
        // ===================================================
        // // X check 1, Z check 2, Z check 3

        {syn_x} = 0;
        {syn_z} = 0;

        reset {q[0]};
        reset {q[8]};
        reset {q[9]};

        h {q[0]};
        h {q[8]};
        h {q[9]};

        cx {q[0]},{q[3]};  // X1
        cz {q[8]},{q[6]};  // Z2
        cz {q[9]},{q[7]};  // Z3

        cx {q[0]},{q[2]};  // X1
        cz {q[8]},{q[3]};  // Z2
        cz {q[9]},{q[6]};  // Z3

        cx {q[0]},{q[4]};  // X1
        cz {q[8]},{q[2]};  // Z2
        cz {q[9]},{q[3]};  // Z3

        cx {q[0]},{q[1]};  // X1
        cz {q[8]},{q[5]};  // Z2
        cz {q[9]},{q[4]};  // Z3

        h {q[0]};
        h {q[8]};
        h {q[9]};

        measure {q[0]} -> {syn_x[0]};
        measure {q[8]} -> {syn_z[1]};
        measure {q[9]} -> {syn_z[2]};

        // // Z check 1, X check 2, X check 3

        reset {q[0]};
        reset {q[8]};
        reset {q[9]};

        h {q[0]};
        h {q[8]};
        h {q[9]};

        cz {q[0]},{q[3]};  // Z1 0,3
        cx {q[8]},{q[6]};  // X2 6,8
        cx {q[9]},{q[7]};  // X3 7,9

        cz {q[0]},{q[2]};  // Z1 0,2
        cx {q[8]},{q[3]};  // X2 3,8
        cx {q[9]},{q[6]};  // X3 6,9

        cz {q[0]},{q[4]};  // Z1 0,4
        cx {q[8]},{q[2]};  // X2 2,8
        cx {q[9]},{q[3]};  // X3 3,9

        cz {q[0]},{q[1]};  // Z1 0,1
        cx {q[8]},{q[5]};  // X2 5,8
        cx {q[9]},{q[4]};  // X3 4,9

        h {q[0]};
        h {q[8]};
        h {q[9]};

        measure {q[0]} -> {syn_z[0]};
        measure {q[8]} -> {syn_x[1]};
        measure {q[9]} -> {syn_x[2]};
        """

        return util.rm_white_space(qasm)
