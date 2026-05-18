"""Quantum state preparation gate implementations.

This module provides gate implementations for preparing and resetting
qubits to specific states, including computational basis states and
other states used in quantum error correction.
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

from pecos.slr.qeclib.qubit.qgate_base import QGate


class PZ(QGate):
    """Prepare/reset a qubit to |0> (+Z eigenstate)."""


class PNZ(QGate):
    """Prepare/reset a qubit to |1> (-Z eigenstate)."""


class PX(QGate):
    """Prepare/reset a qubit to |+> (+X eigenstate)."""


class PNX(QGate):
    """Prepare/reset a qubit to |-> (-X eigenstate)."""


class PY(QGate):
    """Prepare/reset a qubit to |+i> (+Y eigenstate)."""


class PNY(QGate):
    """Prepare/reset a qubit to |-i> (-Y eigenstate)."""
