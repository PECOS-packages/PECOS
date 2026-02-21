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

"""Contains the default implementation of the logical gate protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pecos.protocols import LogicalGateProtocol
from pecos.qeccs.helper_functions import expected_params, make_hashable_params

if TYPE_CHECKING:
    from pecos.protocols import LogicalInstructionProtocol, QECCProtocol
    from pecos.typing import QECCGateParams


class DefaultLogicalGate:
    """Default logical gate class providing standard implementations.

    The main role of logical gates is to identify the sequence of logical instructions the gate is made out of.

    This class provides default implementations of the LogicalGateProtocol interface.
    Gate implementations can inherit from this class to get the standard behavior,
    or implement the LogicalGateProtocol directly for custom behavior.
    """

    def __init__(
        self,
        qecc: QECCProtocol,
        symbol: str,
        **gate_params: QECCGateParams,
    ) -> None:
        """Initialize a default logical gate.

        Args:
            qecc: The parent QECC instance that this gate belongs to.
            symbol: The gate symbol/identifier.
            **gate_params: Additional gate parameters including:
                - error_free: Whether errors should occur for this gate (default: False).
                - forced_outcome: Whether measurements are forced to specific outcomes (default: True).
        """
        self.symbol = symbol
        self.qecc = qecc  # The qecc the gate is a member of.
        self.gate_params = gate_params  # Gate parameters
        self.params = self.gate_params
        self.instr_symbols = None
        self.instr_instances = []
        self.circuits = []  # The circuits of the logical instructions. (Either instr instances or a QuantumCircuit or
        # something with the same methods as a QuantumCircuit.)
        self.error_free = gate_params.get(
            "error_free",
            False,
        )  # Whether errors should occur for this gate.
        self.forced_outcome = gate_params.get(
            "forced_outcome",
            True,
        )  # Whether the measurements are random
        # (if True-> force -1)
        # Can choose 0 or 1.

        self.qecc_params_tuple = make_hashable_params(
            qecc.qecc_params,
        )  # Used for hashing.
        self.gate_params_tuple = make_hashable_params(gate_params)  # Used for hashing.

    def final_instr(self) -> LogicalInstructionProtocol:
        """Gives the final Logical Instruction instance."""
        return self.instr_instances[-1]

    def final_logical_stabs(self) -> dict:
        """Gives the final_logical_ops dict."""
        return self.instr_instances[-1].final_logical_ops

    def expected_params(self, params: dict[str, Any], expected_set: set[str]) -> None:
        """Validate that gate parameters match expected parameter set.

        Args:
            params: Dictionary of gate parameters to validate.
            expected_set: Set of expected parameter names.
        """
        expected_params(params, expected_set)

    def __hash__(self) -> int:
        """Return hash value for use as dictionary key in QuantumCircuit."""
        # Added so the logical gate can be a key (gate symbol) in a ``QuantumCircuit``.

        # These uniquely identify the logical and do not change.
        return hash(
            ("gate", self.symbol, self.qecc_params_tuple, self.gate_params_tuple),
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another logical gate."""
        # Check if other implements the LogicalGateProtocol
        if not isinstance(other, LogicalGateProtocol):
            return NotImplemented
        return (self.symbol, self.qecc_params_tuple, self.gate_params_tuple, True) == (
            other.symbol,
            getattr(other, "qecc_params_tuple", None),
            getattr(other, "gate_params_tuple", None),
            hasattr(other, "instr_symbols"),
        )

    def __ne__(self, other: object) -> bool:
        """Check inequality with another logical gate."""
        return not (self == other)

    def __str__(self) -> str:
        """Return string representation of the logical gate."""
        return (
            f"Logical gate: '{self.symbol}' params={self.gate_params} - QECC: {self.qecc.name} "
            f"params={self.qecc.qecc_params} - Instructions: {self.instr_symbols}"
        )
