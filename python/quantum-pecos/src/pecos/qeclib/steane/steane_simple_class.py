"""Steane 7-qubit quantum error correction code implementation.

This module provides the main Steane class that implements the Steane 7-qubit quantum error correction code, including
all necessary operations for fault-tolerant quantum computation.
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

from pecos.qeclib.steane.gates_sq import paulis, sqrt_paulis
from pecos.qeclib.steane.gates_sq.hadamards import H
from pecos.qeclib.steane.gates_tq import transversal_tq
from pecos.qeclib.steane.meas.destructive_meas import Measure
from pecos.qeclib.steane.preps.pauli_states import PrepRUS
from pecos.qeclib.steane.syn_extract.bare import SynExtractBare
from pecos.qeclib.steane.syn_extract.flagged import SynExtractFlagged
from pecos.slr import CReg, QReg, Vars

if TYPE_CHECKING:
    from pecos.slr import Bit, Block


class Steane(Vars):
    """A generic implementation of a Steane code and operations.

    This represents one particular choice of Steane protocols. For finer control construct your own class
    or utilize the library of Steane code protocols directly.
    """

    def __init__(
        self,
        name: str,
        default_rus_limit: int = 3,
        num_ancilla_qubits: int = 2,
    ) -> None:
        """Initialize a Steane code instance with associated quantum and classical registers.

        Args:
            name: Name prefix for all registers associated with this Steane code instance.
            default_rus_limit: Default limit for repeat-until-success procedures. Defaults to 3.
            num_ancilla_qubits: Number of ancilla qubits to allocate for syndrome extraction.

        Raises:
            ValueError: If provided ancilla register has fewer than 3 qubits.
        """
        super().__init__()
        self.check_indices = [[2, 1, 3, 0], [5, 2, 1, 4], [6, 5, 2, 3]]

        self.num_ancilla_qubits = num_ancilla_qubits

        self.d = QReg(f"{name}_d", 7)
        self.a = QReg(f"{name}_a", num_ancilla_qubits)

        self.verify_prep = CReg(f"{name}_verify_prep", 32)

        self.vars = [
            self.d,
            self.a,
        ]

        self.vars.extend(
            [
                self.verify_prep,
            ],
        )

        self.default_rus_limit = default_rus_limit

    def p(
        self,
        state: str,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Prepare a logical qubit in a logical Pauli basis state."""
        block = PrepRUS(
            q=self.d,
            a=self.a[0],
            init=self.verify_prep[0],
            limit=rus_limit or self.default_rus_limit,
            state=state,
            first_round_reset=True,
        )
        if reject is not None:
            block.extend(reject.set(self.verify_prep[0]))
        return block

    def x(self) -> Block:
        """Logical Pauli X gate."""
        return paulis.X(self.d)

    def y(self) -> Block:
        """Logical Pauli Y gate."""
        return paulis.Y(self.d)

    def z(self) -> Block:
        """Logical Pauli Z gate."""
        return paulis.Z(self.d)

    def h(self) -> Block:
        """Logical Hadamard gate."""
        return H(self.d)

    def sx(self) -> Block:
        """Sqrt of X."""
        return sqrt_paulis.SX(self.d)

    def sxdg(self) -> Block:
        """Adjoint of sqrt of X."""
        return sqrt_paulis.SXdg(self.d)

    def sy(self) -> Block:
        """Sqrt of Y."""
        return sqrt_paulis.SY(self.d)

    def sydg(self) -> Block:
        """Adjoint of sqrt of Y."""
        return sqrt_paulis.SYdg(self.d)

    def sz(self) -> Block:
        """Sqrt of Z. Also known as the S gate."""
        return sqrt_paulis.SZ(self.d)

    def szdg(self) -> Block:
        """Adjoint of Sqrt of Z. Also known as the Sdg gate."""
        return sqrt_paulis.SZdg(self.d)

    def cx(self, target: Steane) -> Block:
        """Logical CX."""
        return transversal_tq.CX(self.d, target.d)

    def cy(self, target: Steane) -> Block:
        """Logical CY."""
        return transversal_tq.CY(self.d, target.d)

    def cz(self, target: Steane) -> Block:
        """Logical CZ."""
        return transversal_tq.CZ(self.d, target.d)

    def m(
        self,
        meas_basis: str,
        meas: CReg,
        log: Bit,
        syn: CReg | None = None,
    ) -> Block:
        """Destructively measure the logical qubit in some Pauli basis."""
        block = Measure(
            q=self.d,
            meas_basis=meas_basis,
            log_raw=log,
            meas_creg=meas,
            barrier=False,
        )

        if syn is not None:
            block.extend(
                syn[0].set(meas[0] ^ meas[1] ^ meas[2] ^ meas[3]),
                syn[1].set(meas[1] ^ meas[2] ^ meas[4] ^ meas[5]),
                syn[2].set(meas[2] ^ meas[3] ^ meas[5] ^ meas[6]),
            )

        return block

    def syn_bare(self, syn: CReg) -> Block:
        """One single syndrome bit per check using bare syndrome extraction."""
        return SynExtractBare(self.d, self.a, self.check_indices, syn)

    def syn_flagged(
        self,
        syn_x: CReg,
        syn_z: CReg,
        flags_x: CReg,
        flags_z: CReg,
    ) -> Block:
        """One single syndrome bit and one single flag bit per check."""
        return SynExtractFlagged(
            self.d,
            self.a,
            self.check_indices,
            syn_x,
            syn_z,
            flags_x,
            flags_z,
        )
