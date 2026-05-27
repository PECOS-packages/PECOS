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
    # Number of leading angle parameters for a parameterized gate. The
    # SLR call convention is angle(s)-FIRST: `RX(theta, q)`,
    # `CRZ(theta, control, target)`, `RZZ(theta, q0, q1)`. A
    # parameterized gate must set `num_params` (and `has_parameters`).
    num_params = 0

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
        """Reject the legacy bracket-parameter form.

        The SLR API now takes rotation angles as leading positional
        arguments -- ``RX(theta, q)`` -- not via brackets. The old
        ``RX[theta](q)`` form is removed (angles-first is the single
        supported convention); raise a clear migration error.
        """
        msg = (
            f"The bracket-parameter form `{self.sym}[angle](qubit)` is no longer "
            f"supported. Pass the angle as a leading positional argument instead: "
            f"`{self.sym}(angle, qubit)` (angles come before qubit ids)."
        )
        raise TypeError(msg)

    def qubits(self, *qargs: Qubit) -> None:
        """Add qubits to the gate.

        Args:
            *qargs: Variable number of qubits to add.
        """
        self(*qargs)

    def __call__(self, *args: Qubit | complex) -> Self:
        """Create a new gate instance from angle(s) and qubit(s).

        For a parameterized gate the first `num_params` arguments are
        the rotation angle(s); the remaining arguments are the qubits:
        `RX(theta, q)`, `CRZ(theta, control, target)`. For a
        non-parameterized gate every argument is a qubit.

        Args:
            *args: `num_params` leading angle parameter(s) (if any)
                followed by the qubit(s) the gate acts on.

        Returns:
            New gate instance with the specified params + qubits.
        """
        g = self.copy()

        if self.has_parameters:
            n = self.num_params
            if len(args) < n:
                msg = (
                    f"{self.sym} is a parameterized gate; call it as "
                    f"`{self.sym}(angle, qubit...)` with {n} leading angle "
                    f"parameter(s) before the qubit(s). Got {len(args)} argument(s)."
                )
                raise TypeError(msg)
            params = tuple(args[:n])
            qargs = tuple(args[n:])
            # Typed-angle guard: each angle slot must be a typed `Angle`
            # (built with `rad(...)` / `turns(...)`), and each qubit slot
            # must be a quantum qubit shape. This rejects the classic
            # mis-ordered call (`RX(q, 0.5)` instead of `RX(rad(0.5), q)`)
            # AND the now-removed bare-float form (`RX(0.5, q)`) loudly at
            # the call, so a typo can never reach codegen as a no-op or as
            # a rotation on a classical register. Qubit slots accept ONLY
            # `Qubit`/`QReg`/`SymbolicQubit` -- NOT the broad `Var` (which
            # also covers classical `CReg`/`Bit`/`SymbolicBit`).
            from pecos.slr.angle import Angle  # noqa: PLC0415  (avoid import cycle)
            from pecos.slr.vars import QReg, Qubit, SymbolicQubit, Var  # noqa: PLC0415  (avoid import cycle)

            qubit_types = (Qubit, QReg, SymbolicQubit)
            for p in params:
                if isinstance(p, Angle):
                    continue
                if isinstance(p, Var):
                    msg = (
                        f"{self.sym}: a register/qubit reference {p!r} was passed in an angle "
                        f"position. Call as `{self.sym}(angle, qubit...)` -- angles come before qubit "
                        "ids, and the angle must be a typed `Angle` (use `rad(...)` / `turns(...)`)."
                    )
                    raise TypeError(msg)
                if isinstance(p, (bool, int, float, complex)):
                    msg = (
                        f"{self.sym}: bare numeric angle {p!r} is no longer accepted. Wrap it in a "
                        f"typed `Angle`: `{self.sym}(rad({p}), qubit...)` (radians) or "
                        f"`{self.sym}(turns(...), qubit...)`."
                    )
                    raise TypeError(msg)
                msg = (
                    f"{self.sym}: angle parameter {p!r} must be a typed `Angle` "
                    "built with `rad(...)` / `turns(...)`."
                )
                raise TypeError(msg)
            for qa in qargs:
                if not isinstance(qa, qubit_types):
                    kind = "classical register/bit" if isinstance(qa, Var) else "non-qubit"
                    msg = (
                        f"{self.sym}: a {kind} {qa!r} was passed in a qubit position. "
                        f"Call as `{self.sym}(angle, qubit...)` with {n} leading angle parameter(s); "
                        "qubit positions accept only qubits/QRegs."
                    )
                    raise TypeError(msg)
            # Construction-time arity guard: a parameterized call must
            # supply enough qubits, so a malformed `gate(angle)` (no
            # qubit) or `RZZ(angle, q[0])` (one short) fails loud here
            # rather than surviving to QIR/QASM. A whole `QReg` broadcasts
            # (its size is only known on expansion), so the explicit-qubit
            # count is only checked when no register is passed.
            if not qargs:
                msg = (
                    f"{self.sym}: a parameterized gate needs at least one qubit; got only angle(s). "
                    f"Call as `{self.sym}(angle, qubit...)`."
                )
                raise TypeError(msg)
            if not any(isinstance(qa, QReg) for qa in qargs) and len(qargs) < self.qsize:
                msg = (
                    f"{self.sym}: needs at least {self.qsize} qubit(s) for a {self.qsize}-qubit gate, "
                    f"got {len(qargs)}. (Pass a whole QReg to broadcast.)"
                )
                raise TypeError(msg)
            g.params = params
            g.add_qargs(qargs)
        else:
            g.add_qargs(args)

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
                target = QASMGenerator(add_versions=add_versions, _internal=True)
            else:
                msg = f"Code gen target '{target}' is not supported."
                raise NotImplementedError(msg)

        return target.process_qgate(self)


class TQGate(QGate, metaclass=ABCMeta):
    """Two qubit gates."""

    qsize = 2
