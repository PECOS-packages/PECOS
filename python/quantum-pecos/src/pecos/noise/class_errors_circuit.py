"""Error circuit generation and management.

This module provides the ErrorCircuits class for generating and managing
error circuits in quantum error correction simulations. It handles the
creation of error patterns and their application to quantum circuits.
"""

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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.circuits.quantum_circuit import QuantumCircuit


class ErrorCircuits(dict):
    """Used to store error circuits."""

    def __init__(self) -> None:
        """Initialize an ErrorCircuits instance.

        This creates an empty dictionary to store error circuits.
        """
        super().__init__()

    def add_circuits(
        self,
        time: int,
        before_faults: QuantumCircuit | None = None,
        after_faults: QuantumCircuit | None = None,
        replaced_locations: set[int] | None = None,
    ) -> None:
        """Add error circuits and gate locations to ignore (replaced_locations).

        Args:
        ----
            time: The time tick at which to add the error circuits.
            before_faults: Circuit containing faults to apply before the main circuit.
            after_faults: Circuit containing faults to apply after the main circuit.
            replaced_locations: Set of gate locations that should be replaced/ignored.

        """
        error_dict = {}

        if before_faults and len(before_faults) > 0:
            before_faults.metadata["circuit_type"] = "faults"
            error_dict["before"] = before_faults

        if after_faults and len(after_faults) > 0:
            after_faults.metadata["circuit_type"] = "faults"
            error_dict["after"] = after_faults

        if replaced_locations and len(replaced_locations) > 0:
            error_dict["replaced"] = replaced_locations

        if error_dict:
            self[time] = error_dict
