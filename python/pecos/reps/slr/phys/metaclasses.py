from __future__ import annotations

import copy
from abc import ABCMeta

# ruff: noqa: B024


# TODO: Try to move more into using the class instead of instance. E.g., class methods, don't override call or
#   use the whole H = HGate() type thing. H should be a class not an instance.
class QGate(metaclass=ABCMeta):
    def __init__(self, sym: str | None = None, qasm_sym: str | None = None, qsize: int = 1, csize: int = 0):
        if sym is None:
            self.sym = self.__class__.__name__
            if self.sym.endswith("Gate"):
                self.sym = self.sym[:-4]
        else:
            self.sym = sym

        self.qasm_sym = qasm_sym if qasm_sym else self.sym

        self.qsize = qsize
        self.csize = csize

        self.qargs = None
        self.params = None

    def copy(self):
        return copy.copy(self)

    def __getitem__(self, *params):
        g = self.copy()
        g.params = params
        return g

    def qubits(self, *qargs):
        self.__call__(qargs)

    def __call__(self, *qargs, params=None):
        g = self.copy()

        if isinstance(qargs, tuple):
            g.qargs = qargs
        else:
            g.qargs = (qargs,)

        return g

    def qasm(self):
        repr_str = self.qasm_sym

        if self.params:
            str_cargs = ", ".join([str(p) for p in self.params])
            repr_str = f"{repr_str}({str_cargs})"

        str_list = []

        # TODO: Make sure this all works for different sized multi-qubit gates!!!!!!!!!!!!!!! <<<<<<<<<<

        if self.qsize > 1:
            if not isinstance(self.qargs[0], tuple) and len(self.qargs) > 1:
                self.qargs = (self.qargs,)

        for q in self.qargs:
            if isinstance(q, tuple):
                if len(q) != self.qsize:
                    msg = f"Expected size {self.qsize} got size {len(q)}"
                    raise Exception(msg)
                qs = ",".join([str(qi) for qi in q])
                str_list.append(f"{repr_str} {qs};")

            str_list.append(f"{repr_str} {str(q)};")

        return "\n".join(str_list)


class NoParamsQGate(QGate, metaclass=ABCMeta):
    def __call__(self, *params):
        g = super().__call__(*params)

        if g.params:
            msg = "This gate does not accept parameters. You might of meant to put qubits in square brackets."
            raise Exception(msg)

        return g


class UnitaryGate(QGate, metaclass=ABCMeta):
    """A Unitary gate"""

    def __init__(self, sym=None, qasm_sym=None):
        super().__init__(sym, qasm_sym)
        self.matrix = None


class SingleQubitUnitary(UnitaryGate, metaclass=ABCMeta):
    """Unitaries that act on a single qubit."""

    def __init__(self, sym=None, qasm_sym=None):
        super().__init__(sym, qasm_sym)
        self.qsize = 1


class TwoQubitUnitary(UnitaryGate, metaclass=ABCMeta):
    """Unitaries that act on two qubits."""

    def __init__(self, sym=None, qasm_sym=None):
        super().__init__(sym, qasm_sym)
        self.qsize = 2

    def qasm(self):
        repr_str = self.qasm_sym

        if self.params:
            str_cargs = ",".join([str(p) for p in self.params])
            repr_str = f"{repr_str}({str_cargs})"

        str_list = []

        if not isinstance(self.qargs[0], tuple) and len(self.qargs) == 2:
            self.qargs = (self.qargs,)

        for q in self.qargs:
            if isinstance(q, tuple):
                q1, q2 = q
                str_list.append(f"{repr_str} {str(q1)}, {str(q2)};")
            else:
                msg = f"For TQ gate, expected args to be a collection of size two tuples! Got: {self.qargs}"
                raise TypeError(msg)

        return "\n".join(str_list)


class CliffordGate(NoParamsQGate, UnitaryGate, metaclass=ABCMeta):
    """A Clifford gate"""

    def __init__(self, sym=None, qasm_sym=None):
        super().__init__(sym, qasm_sym)
        self.csize = 0
        self.pauli_rules = None


class SQCliffordGate(CliffordGate, SingleQubitUnitary, metaclass=ABCMeta):
    """A Clifford gate that acts on a single qubit."""


class TQCliffordGate(CliffordGate, TwoQubitUnitary, metaclass=ABCMeta):
    """A Clifford gate that acts on a single qubit."""


class PauliGate(CliffordGate, metaclass=ABCMeta):
    """A Pauli gate"""


class SQPauliGate(SQCliffordGate, metaclass=ABCMeta):
    """A single-qubit Pauli gate"""
