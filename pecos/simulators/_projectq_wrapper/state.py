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

"""
A simple wrapper for the ProjectQ simulator.

Compatibility checked for: ProjectQ version 0.3.6
"""

from .. import BaseSim

try:
    from projectq import MainEngine
    from . import bindings
    from .bindings import MakeFunc
    hasprojq = True
except ModuleNotFoundError:
    hasprojq = False


class State(BaseSim):
    """
    Initializes the stabilizer state.

    Args:
        num_qubits (int): Number of qubits being represented.

    Returns:

    """

    def __init__(self, num_qubits):

        if not hasprojq:
            raise Exception("ProjectQ must be installed to use this class.")

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        super().__init__()

        self.gate_dict = bindings.gate_dict
        self.num_qubits = num_qubits
        self.eng = MainEngine()

        self.qs = list(self.eng.allocate_qureg(num_qubits))
        self.qids = {i: q for i, q in enumerate(self.qs)}

    def add_gate(self, symbol, gate_obj, make_func=True):

        if symbol in self.gate_dict:
            print('WARNING: Can not add gate as the symbol has already been taken.')
        else:
            if make_func:
                self.gate_dict[symbol] = MakeFunc(gate_obj).func
            else:
                self.gate_dict[symbol] = gate_obj
