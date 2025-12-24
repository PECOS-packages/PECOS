# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Data type definitions for PyPHIR intermediate representation.

This module defines data types and structures used in PyPHIR (Python PECOS Medium-level Intermediate Representation) for
representing quantum and classical data in quantum circuit execution.
"""

from __future__ import annotations

from pecos.reps.pyphir.instr_type import Instr


class Data(Instr):
    """General data type."""


class DefineVar(Data):
    """Base class for variable definitions in PHIR.

    This class provides the foundation for defining variables of various types
    in the Pecos Machine Intermediate Representation.
    """

    def __init__(
        self,
        data_type: str | type,
        variable: str,
        metadata: dict | None = None,
    ) -> None:
        """Initialize a variable definition.

        Args:
            data_type: The data type of the variable (string or type object).
            variable: The variable name.
            metadata: Optional metadata dictionary.
        """
        super().__init__(metadata=metadata)
        self.data_type = data_type
        self.variable = variable


class CVarDefine(DefineVar):
    """Classical variable definition in PHIR.

    This class represents the definition of a classical variable with its associated
    data type, size, and unique identifier.
    """

    def __init__(
        self,
        data_type: str | type,
        variable: str,
        cvar_id: int,
        size: int,
        metadata: dict | None = None,
    ) -> None:
        """Initialize a classical variable definition.

        Args:
            data_type: The data type of the classical variable.
            variable: The variable name.
            cvar_id: The unique identifier for this classical variable.
            size: The size of the classical variable.
            metadata: Optional metadata dictionary.
        """
        super().__init__(data_type=data_type, variable=variable, metadata=metadata)
        self.size = size
        self.cvar_id = cvar_id


class QVarDefine(DefineVar):
    """Quantum variable definition in PHIR.

    This class represents the definition of a quantum variable with its associated
    data type, size (number of qubits), and qubit identifiers.
    """

    def __init__(
        self,
        data_type: str | type,
        variable: str,
        size: int,
        qubit_ids: list[int],
        metadata: dict | None = None,
    ) -> None:
        """Initialize a quantum variable definition.

        Args:
            data_type: The data type of the quantum variable.
            variable: The variable name.
            size: The size of the quantum variable (number of qubits).
            qubit_ids: List of qubit identifiers associated with this variable.
            metadata: Optional metadata dictionary.
        """
        super().__init__(data_type=data_type, variable=variable, metadata=metadata)
        self.size = size
        self.qubit_ids = qubit_ids


class ExportVar(Data):
    """Variable export instruction in PHIR.

    This class represents an instruction to export variables from the current
    scope, optionally renaming them in the export destination.
    """

    def __init__(
        self,
        variables: list[str],
        to: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize a variable export instruction.

        Args:
            variables: List of variable names to export.
            to: Optional list of destination names for the exported variables.
            metadata: Optional metadata dictionary.
        """
        super().__init__(metadata=metadata)
        self.variables = variables
        self.to = to
