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

"""Pauli projection preparation blocks for surface code operations."""

from pecos.slr import Block, QReg, Qubit
from pecos.slr.qeclib.qubit.qubit import PhysicalQubit as Q


class PrepZ(Block):
    """Prepare the +Z operator."""

    def __init__(self, q: QReg, data_indices: list[int]) -> None:
        """Initialize the +Z state preparation block.

        Args:
            q: Quantum register containing the qubits.
            data_indices: List of indices for data qubits to prepare in +Z state.
        """
        super().__init__()

        for i in data_indices:
            self.extend(
                Q.pz(q[i]),
            )


class PrepProjectZ(Block):
    """Prepare the +Z operator."""

    def __init__(self, qs: list[Qubit]) -> None:
        """Initialize the +Z projection preparation block.

        Args:
            qs: List of qubits to prepare and project into +Z eigenstate.
        """
        super().__init__()

        self.extend(
            PrepZ(*qs),
        )
        # TODO: Measure the X checks
