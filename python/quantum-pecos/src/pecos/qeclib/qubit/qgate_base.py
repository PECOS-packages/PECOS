"""Base classes for quantum gate implementations.

This module provides the foundational base classes for quantum gate
implementations in the PECOS quantum error correction library,
defining interfaces and common functionality for quantum operations.
"""

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

from __future__ import annotations

import copy
import sys
from abc import ABCMeta
from typing import TYPE_CHECKING

from pecos.slr.gen_codes.gen_qasm import QASMGenerator

# Handle Python 3.10 compatibility for Self type
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar

    Self = TypeVar("Self", bound="QGate")

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.slr import Qubit


# TODO: Try to move more into using the class instead of instance. E.g., class methods, don't override call or
#   use the whole H = HGate() type thing. H should be a class not an instance.
class QGate:
    """Quantum gates including unitaries, measurements, and preparations."""

    is_qgate = True
    qsize = 1
    csize = 0
    has_parameters = False

    def __init__(self, *qargs: Qubit) -> None:
        """Initialize a quantum gate.

        Args:
            *qargs: Qubit(s) that the gate acts on.
        """
        self.sym = type(self).__name__
        if self.sym.endswith("Gate"):
            self.sym = self.sym[:-4]

        self.qargs = None
        self.params = None

        self.add_qargs(qargs)

    def add_qargs(self, qargs: Sequence[Qubit] | Qubit) -> None:
        """Add quantum arguments to the gate.

        Args:
            qargs: Qubit or sequence of qubits to add as arguments.
        """
        if isinstance(qargs, tuple):
            self.qargs = qargs
        else:
            self.qargs = (qargs,)

    def copy(self) -> Self:
        """Create a shallow copy of the gate.

        Returns:
            Copy of the gate instance.
        """
        return copy.copy(self)

    def __getitem__(self, *params: complex) -> Self:
        """Set gate parameters using square bracket notation."""
        g = self.copy()

        if params and not self.has_parameters:
            msg = "This gate does not accept parameters. You might of meant to put qubits in square brackets."
            raise Exception(msg)
        g.params = params

        return g

    def qubits(self, *qargs: Qubit) -> None:
        """Add qubits to the gate.

        Args:
            *qargs: Variable number of qubits to add.
        """
        self(*qargs)

    def __call__(self, *qargs: Qubit) -> Self:
        """Create a new gate instance with specified qubits.

        Args:
            *qargs: Variable number of qubits to apply the gate to.

        Returns:
            New gate instance with the specified qubits.
        """
        g = self.copy()

        g.add_qargs(qargs)

        return g

    def gen(self, target: object | str, *, add_versions: bool = False) -> str:
        """Generate code for the gate using the specified target generator.

        Args:
            target: Either a generator object or string specifying the target ("qasm").
            add_versions: Whether to add version information to generated code.

        Returns:
            Generated code as a string.
        """
        # TODO: Get rid of this as much as possible...
        if isinstance(target, str):
            if target == "qasm":
                target = QASMGenerator(add_versions=add_versions)
            else:
                msg = f"Code gen target '{target}' is not supported."
                raise NotImplementedError(msg)

        return target.process_qgate(self)


class TQGate(QGate, metaclass=ABCMeta):
    """Two qubit gates."""

    qsize = 2
