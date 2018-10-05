#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from projectq.ops import H, C, Z


def II(state, qubits, **kwargs):
    pass


def G2(state, qubits):
    """
    Applies a CZ.H(1).H(2).CZ

    Returns:

    """
    q1 = state.qids[qubits[0]]
    q2 = state.qids[qubits[0]]

    C(Z) | (q1, q2)
    H | q1
    H | q2
    C(Z) | (q1, q2)
