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
from abc import ABCMeta

from pecos.slr.gen_codes.gen_qasm import QASMGenerator

# ruff: noqa: B024


# TODO: Try to move more into using the class instead of instance. E.g., class methods, don't override call or
#   use the whole H = HGate() type thing. H should be a class not an instance.
class QGate(metaclass=ABCMeta):
    """Quantum gates including unitaries, measurements, and preparations."""

    is_qgate = True
    qsize = 1
    csize = 0
    has_parameters = False

    def __init__(self, *qargs):
        self.sym = type(self).__name__
        if self.sym.endswith("Gate"):
            self.sym = self.sym[:-4]

        self.qargs = None
        self.params = None

        self.add_qargs(qargs)

    def add_qargs(self, qargs):
        if isinstance(qargs, tuple):
            self.qargs = qargs
        else:
            self.qargs = (qargs,)

    def copy(self):
        return copy.copy(self)

    def __getitem__(self, *params):
        g = self.copy()

        if params and not self.has_parameters:
            msg = "This gate does not accept parameters. You might of meant to put qubits in square brackets."
            raise Exception(msg)
        else:
            g.params = params

        return g

    def qubits(self, *qargs):
        self.__call__(qargs)

    def __call__(self, *qargs):
        g = self.copy()

        g.add_qargs(qargs)

        return g

    def gen(self, target: object | str):
        # TODO: Get rid of this as much as possible...
        if isinstance(target, str):
            if target == "qasm":
                target = QASMGenerator()
            else:
                msg = f"Code gen target '{target}' is not supported."
                raise NotImplementedError(msg)

        return target.process_qgate(self)


class TQGate(QGate, metaclass=ABCMeta):
    """Two qubit gates"""

    qsize = 2
