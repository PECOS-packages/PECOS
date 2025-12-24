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

"""Operation type definitions for PyPHIR intermediate representation.

This module defines operation classes for PyPHIR (Python PECOS Medium-level Intermediate Representation) including
quantum operations, classical operations, and machine operations for quantum circuit execution.
"""

from __future__ import annotations

from pecos.reps.pyphir.instr_type import Instr


class Op(Instr):
    """Parent class of operations."""

    def __init__(
        self,
        name: str,
        args: list | None = None,
        returns: list | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize an operation.

        Args:
            name: The operation name.
            args: Optional list of operation arguments.
            returns: Optional list of return values (cvars or cbits).
            metadata: Optional metadata dictionary.
        """
        super().__init__(metadata=metadata)
        self.name = name
        self.args = args
        self.returns = returns

        if returns is not None:
            for r in returns:
                if isinstance(r, str):
                    pass
                elif isinstance(r, list):
                    sym, id_ = r
                    if not isinstance(sym, str) or not isinstance(id_, int):
                        msg = f"Returns not of correct form of cvar (str) or cbit ([str, int]): {returns}"
                        raise TypeError(msg)
                else:
                    msg = f"Returns not of correct form of cvar (str) or cbit ([str, int]): {returns}"
                    raise TypeError(msg)

    def __str__(self) -> str:
        """Return string representation of the operation."""
        return f"<{self.name}, {self.args}, {self.returns}, {self.metadata}>"


class QOp(Op):
    """Quantum operation."""

    def __init__(
        self,
        name: str,
        args: list,
        returns: list | None = None,
        metadata: dict | None = None,
        angles: tuple[float, ...] | None = None,
        sim_name: str | None = None,
    ) -> None:
        """Initialize a quantum operation.

        Args:
            name: The operation name.
            args: List of operation arguments (typically qubits).
            returns: Optional list of return values.
            metadata: Optional metadata dictionary.
            angles: Optional tuple of rotation angles.
            sim_name: Optional simulator-specific name. If not provided, uses name.
        """
        super().__init__(
            name=name,
            args=args,
            returns=returns,
            metadata=metadata,
        )
        self.angles = angles
        self.sim_name = sim_name
        if self.sim_name is None:
            self.sim_name = name

    def __repr__(self) -> str:
        """Return detailed string representation of the quantum operation."""
        repr_str = f"<QOP: {self.name}"

        if self.angles:
            repr_str += f", angles={self.angles}"

        if self.args:
            repr_str += f", args={self.args}"

        if self.returns:
            repr_str += f", returns={self.returns}"

        if self.metadata:
            repr_str += f", metadata={self.metadata}"

        repr_str += ">"

        return repr_str

    def __str__(self) -> str:
        """Return string representation of the quantum operation."""
        return self.__repr__()


class COp(Op):
    """Classical operation."""

    def __init__(
        self,
        name: str,
        args: list,
        returns: list | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize a classical operation.

        Args:
            name: The operation name.
            args: List of operation arguments.
            returns: Optional list of return values.
            metadata: Optional metadata dictionary.
        """
        super().__init__(
            name=name,
            args=args,
            returns=returns,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Return detailed string representation of the classical operation."""
        repr_str = f"<COP: {self.name}"

        if self.args:
            repr_str += f", args={self.args}"

        if self.returns:
            repr_str += f", returns={self.returns}"

        if self.metadata:
            repr_str += f", metadata={self.metadata}"

        repr_str += ">"

        return repr_str

    def __str__(self) -> str:
        """Return string representation of the classical operation."""
        return self.__repr__()


class FFCall(COp):
    """Represents a call to a foreign function."""


class MOp(Op):
    """Machine operation."""


class EMOp(Op):
    """Error model operation."""


class SOp(Op):
    """Simulation model."""
