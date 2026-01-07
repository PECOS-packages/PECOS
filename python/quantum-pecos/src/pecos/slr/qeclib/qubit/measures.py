"""Quantum measurement gate implementations.

This module provides measurement gate implementations for qubits,
including various measurement bases and projective measurements
used in quantum error correction protocols.
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

from typing import TYPE_CHECKING

from pecos.slr.qeclib.qubit.qgate_base import QGate

if TYPE_CHECKING:
    from pecos.slr import Bit, Qubit


class Measure(QGate):
    """A measurement of a qubit in the Z basis."""

    csize = 1

    def __init__(self, *qargs: Qubit) -> None:
        """Initialize a measurement gate.

        Args:
            *qargs: Qubit(s) to measure in the Z basis.
        """
        super().__init__(*qargs)
        self.cout = None

    def __gt__(self, cout: Bit | tuple[Bit, ...]) -> Measure:
        """Set the classical output bit(s) for measurement using > operator."""
        g = self.copy()

        if isinstance(cout, tuple):
            g.cout = cout
        else:
            g.cout = (cout,)

        return g
