# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.qeccs.gate_parent_class import LogicalGate
from pecos.qeccs.helper_functions import expected_params


class GateIdentity(LogicalGate):
    """Logical Identity.

    This is equivalent to ``distance`` number of syndrome of extraction rounds.
    """

    def __init__(self, qecc, symbol, **gate_params) -> None:
        """Args:
        ----
            qecc(QECC):
            symbol(str):
            **gate_params(dict): kwargs including keys: 'num_syn_extract' (default: qecc.distance).
        """
        super().__init__(qecc, symbol, **gate_params)

        expected_params(
            gate_params,
            {"num_syn_extract", "error_free", "forced_outcome"},
        )

        self.num_syn_extract = gate_params.get("num_syn_extract", qecc.distance)

        # This specifies the logical instructions used for the gate.
        self.instr_symbols = ["instr_syn_extract"] * self.num_syn_extract


class GateInitZero(LogicalGate):
    """Initialize logical state zero."""

    def __init__(self, qecc, symbol, **gate_params) -> None:
        """Args:
        ----
            qecc(QECC):
            symbol(str):
            **gate_params(dict): kwargs including keys: 'num_syn_extract' (default: 0).
        """
        super().__init__(qecc, symbol, **gate_params)

        expected_params(
            gate_params,
            {"num_syn_extract", "error_free", "forced_outcome"},
        )

        self.num_syn_extract = gate_params.get("num_syn_extract", 0)

        # This specifies the logical instructions used for the gate.
        self.instr_symbols = ["instr_init_zero"]
        syn_extract = ["instr_syn_extract"] * self.num_syn_extract
        self.instr_symbols.extend(syn_extract)


class GateInitPlus(LogicalGate):
    """Initialize logical state plus."""

    def __init__(self, qecc, symbol, **gate_params) -> None:
        """Args:
        ----
            qecc(QECC):
            symbol(str):
            **gate_params(dict): kwargs including keys: 'num_syn_extract' (default: 0).
        """
        super().__init__(qecc, symbol, **gate_params)

        expected_params(
            gate_params,
            {"num_syn_extract", "error_free", "forced_outcome"},
        )

        self.num_syn_extract = gate_params.get("num_syn_extract", 0)

        # This specifies the logical instructions used for the gate.
        self.instr_symbols = ["instr_init_plus"]
        syn_extract = ["instr_syn_extract"] * self.num_syn_extract
        self.instr_symbols.extend(syn_extract)
