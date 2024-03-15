# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import copy

import numpy as np

from pecos.circuits.hyqc.fund import Expression


class QOp(Expression):
    def __init__(self) -> None:
        super().__init__()


class QGate(QOp):
    # qargs: Qubits, Qubits, ...; (Qubit, Qubit)

    def __init__(self) -> None:
        super().__init__()
        self.symbol = self.__class__.__name__

        self.qsize = None
        self.csize = None

        self.qargs = None
        self.params = None

    def copy(self):
        return copy.copy(self)

    def __call__(self, *params):
        g = self.copy()
        g.params = params
        return g

    def __getitem__(self, *qargs):
        g = self.copy()
        g.qargs = qargs
        return g

    def __repr__(self) -> str:
        repr_str = self.symbol

        if self.params:
            str_cargs = ", ".join([str(p) for p in self.params])
            repr_str = f"{repr_str}({str_cargs})"

        str_qargs = ", ".join([str(q) for q in self.qargs])
        return f"{repr_str}[{str_qargs}]"


class BarrierGate(QGate):
    """A gate that prevents compilers from moving quantum gates past the barrier."""


Barrier = BarrierGate()


class ResetGate(QGate):
    """Resetting a qubit to the zero state."""


Reset = ResetGate()


class MeasureGate(QGate):
    """A measurement of a qubit in the Z basis."""

    def __init__(self) -> None:
        super().__init__()
        self.cout = None

    def __gt__(self, *cout):
        g = self.copy()
        g.cout = cout

        return g

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        couts_str = ", ".join([str(c) for c in self.cout])
        return f"{repr_str} > {couts_str}"


Measure = MeasureGate()


class UnitaryGate(QGate):
    """A Unitary gate."""

    def __init__(self) -> None:
        super().__init__()
        self.matrix = None


class CliffordGate(UnitaryGate):
    """A Clifford gate."""

    def __init__(self) -> None:
        super().__init__()
        self.csize = 0
        self.pauli_rules = None


class PauliGate(CliffordGate):
    """A Pauli gate."""


class NonCliffordGate(UnitaryGate):
    """A non-Clifford gate."""


class XGate(PauliGate):
    """The Pauli X unitary."""

    def __init__(self) -> None:
        super().__init__()
        self.qsize = 1

        self.matrix = np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "+X",
            "Y": "-Y",
            "Z": "-Z",
        }


X = XGate()


class YGate(PauliGate):
    """The Pauli Y unitary."""

    def __init__(self) -> None:
        super().__init__()
        self.qsize = 1

        self.matrix = np.array(
            [
                [0, -1j],
                [1j, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "-X",
            "Y": "+Y",
            "Z": "-Z",
        }


Y = YGate()


class ZGate(PauliGate):
    """The Pauli Z unitary."""

    def __init__(self) -> None:
        super().__init__()
        self.qsize = 1

        self.matrix = np.array(
            [
                [0, -1j],
                [1j, 0],
            ],
            dtype=complex,
        )

        self.pauli_rules = {
            "X": "-X",
            "Y": "-Y",
            "Z": "+Z",
        }


Z = ZGate()


class SqrtXGate(CliffordGate): ...


SqrtX = SqrtXGate()


class SqrtYGate(CliffordGate): ...


SqrtY = SqrtYGate()


class SqrtZGate(CliffordGate): ...


S = SqrtZ = SqrtZGate()


class SqrtXdgGate(CliffordGate): ...


SqrtXdg = SqrtXdgGate()


class SqrtYdgGate(CliffordGate): ...


SqrtYdg = SqrtYdgGate()


class SqrtZdgGate(CliffordGate): ...


Sdg = SqrtZdg = SqrtZGate()


class RXGate(NonCliffordGate): ...


RX = RXGate()


class RYGate(NonCliffordGate): ...


RY = RYGate()


class RZGate(NonCliffordGate): ...


RZ = RZGate()


class TGate(NonCliffordGate): ...


T = TGate()


class TdgGate(NonCliffordGate): ...


Tdg = TdgGate()


class HGate(CliffordGate): ...


H = HGate()


class FGate(CliffordGate): ...


F = FGate()


class FdgGate(CliffordGate): ...


Fdg = FdgGate()


class CXGate(CliffordGate): ...


CNOT = CX = CXGate()


class CYGate(CliffordGate): ...


CY = CYGate()


class CZGate(CliffordGate): ...


CZ = CZGate()


class SqrtXXGate(CliffordGate): ...


SqrtXX = SqrtXXGate()


class SqrtYYGate(CliffordGate): ...


SqrtYY = SqrtYYGate()


class SqrtZZGate(CliffordGate): ...


SqrtZZ = SqrtZZGate()


class SqrtXXdgGate(CliffordGate): ...


SqrtXXdg = SqrtXXdgGate()


class SqrtYYdgGate(CliffordGate): ...


SqrtYYdg = SqrtYYdgGate()


class SqrtZZdgGate(CliffordGate): ...


SqrtZZdg = SqrtZZdgGate()


# TODO: SWAP, iSWAP, Toffoli (CCNOT, CCX, TOFF)
