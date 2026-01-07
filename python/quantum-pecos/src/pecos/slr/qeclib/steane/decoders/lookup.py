"""Lookup table decoder for the Steane 7-qubit code.

This module provides lookup table decoding implementations for the Steane 7-qubit quantum error correction code,
enabling syndrome-based error correction through pre-computed correction tables.
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

from pecos.slr import Block, Comment, If
from pecos.slr.qeclib import qubit

if TYPE_CHECKING:
    from pecos.slr import Bit, CReg, QReg


class FlagLookupQASM(Block):
    """Flag-based lookup decoder for Steane code syndrome decoding.

    This class implements a lookup table-based decoder that uses flag qubits
    to determine the appropriate error correction based on measured syndromes.
    """

    def __init__(
        self,
        basis: str,
        syn: CReg,
        syndromes: CReg,
        raw_syn: CReg,
        pf: Bit,
        flag: CReg,
        flags: CReg,
        scratch: CReg,
    ) -> None:
        """Initialize flag-based lookup decoder for Steane code.

        Args:
            basis: The basis ('X' or 'Z') for syndrome decoding.
            syn: Classical register containing current syndrome values.
            syndromes: Classical register for processed syndrome values.
            raw_syn: Classical register storing raw syndrome values.
            pf: Pauli frame bit for tracking logical operations.
            flag: Classical register containing flag qubit measurements.
            flags: Classical register for flag status.
            scratch: Classical register used for intermediate calculations.
        """
        super().__init__()

        # qasm_syn_decoder('X', syn_x, flag_x, 'last_raw_syn_x', 'pf_z1')
        # qasm_syn_decoder(basis_check, syn, flag, raw_syn, pf, pf_index=0)

        self.extend(
            Comment(
                f"""
=========================
BEGIN Run {basis} decoder
=========================\n""",
            ),
            If(flags != 0).Then(syndromes.set(syn ^ raw_syn)),
            If(flags == 0).Then(syndromes.set(0)),
            Comment("\napply corrections"),
            If(syndromes == 2).Then(pf.set(pf ^ 1)),
            If(syndromes == 4).Then(pf.set(pf ^ 1)),
            If(syndromes == 6).Then(pf.set(pf ^ 1)),
            Comment(),
            Comment("alter correction based on flags"),
            Comment("==============================="),
            Comment(),
            Comment("1&2 (1 -> 2)"),
            Comment("------------"),
            scratch.set(0),
            If(flag == 1).Then(scratch[0].set(1)),
            If(syndromes == 2).Then(scratch[1].set(1)),
            Comment(),
            scratch[2].set(scratch[0] & scratch[1]),
            If(scratch[2] == 1).Then(pf.set(pf ^ 1)),
            Comment(),
            Comment("1&4 (1 -> 3)"),
            Comment("------------"),
            scratch.set(0),
            If(flag == 1).Then(scratch[0].set(1)),
            If(syndromes == 4).Then(scratch[1].set(1)),
            Comment(),
            scratch[2].set(scratch[0] & scratch[1]),
            If(scratch[2] == 1).Then(pf.set(pf ^ 1)),
            Comment(),
            Comment("6&4 (2,3 -> 3)"),
            Comment("------------"),
            scratch.set(0),
            If(flag == 6).Then(scratch[0].set(1)),
            If(syndromes == 4).Then(scratch[1].set(1)),
            Comment(),
            scratch[2].set(scratch[0] & scratch[1]),
            If(scratch[2] == 1).Then(pf.set(pf ^ 1)),
            Comment(),
            If(flags != 0).Then(raw_syn.set(syn)),
            Comment(),
            Comment("========================="),
            Comment(f"END Run {basis} decoder"),
            Comment("=========================\n"),
        )


class FlagLookupQASMActiveCorrectionX(Block):
    """Active X-basis correction using flag-based lookup for Steane code.

    This class applies X-basis error corrections based on syndrome measurements
    and flag qubit information using a lookup table approach.
    """

    def __init__(
        self,
        qubits: QReg,
        syn: CReg,
        syndromes: CReg,
        raw_syn: CReg,
        pf: Bit,
        flag: CReg,
        flags: CReg,
        scratch: CReg,
        pf_bit_copy: Bit | None = None,
    ) -> None:
        """Initialize X-basis active error correction with flag-based lookup.

        Args:
            qubits: Quantum register containing the logical qubit data.
            syn: Classical register containing current syndrome values.
            syndromes: Classical register for processed syndrome values.
            raw_syn: Classical register storing raw syndrome values.
            pf: Pauli frame bit for tracking logical operations.
            flag: Classical register containing flag qubit measurements.
            flags: Classical register for flag status.
            scratch: Classical register used for intermediate calculations.
            pf_bit_copy: Optional bit to copy the Pauli frame value to.
        """
        super().__init__()
        # qasm_syn_decoder('X', syn_x, flag_x, 'last_raw_syn_x', 'pf_z1')
        # qasm_syn_decoder(basis_check, syn, flag, raw_syn, pf, pf_index=0)
        q = qubits

        self.extend(
            FlagLookupQASM(
                basis="X",
                syn=syn,
                syndromes=syndromes,
                raw_syn=raw_syn,
                pf=pf,
                flag=flag,
                flags=flags,
                scratch=scratch,
            ),
        )

        if pf_bit_copy is not None:
            self.extend(
                Comment(),
                Comment("copy Pauli frame"),
                pf_bit_copy.set(pf),
            )

        self.extend(
            Comment(),
            Comment(),
            Comment("ACTIVE ERROR CORRECTION FOR X SYNDROMES"),
            Comment(),
            scratch.set(0),
            Comment(),
            Comment("only part that differs for X vs Z syns V"),
            If(syndromes[0] == 1).Then(scratch.set(scratch ^ 1)),
            If(syndromes[1] == 1).Then(scratch.set(scratch ^ 12)),
            If(syndromes[2] == 1).Then(scratch.set(scratch ^ 48)),
            Comment(),
            Comment("logical operator"),
            If(pf == 1).Then(scratch.set(scratch ^ 112)),
            Comment(),
            If(scratch[0] == 1).Then(qubit.Z(q[0])),
            Comment("not possible for X stabilizers V"),
            Comment(
                f"if({scratch[1]} == 1) z {q[1]};",
            ),  # If(scratch[2] == 1).Then(qubit.Z(q[1])),
            If(scratch[2] == 1).Then(qubit.Z(q[2])),
            If(scratch[3] == 1).Then(qubit.Z(q[3])),
            If(scratch[4] == 1).Then(qubit.Z(q[4])),
            If(scratch[5] == 1).Then(qubit.Z(q[5])),
            If(scratch[6] == 1).Then(qubit.Z(q[6])),
            Comment(),
            pf.set(0),
            Comment(f"{syndromes} = 0;"),
            raw_syn.set(0),
            Comment(f"{syn} = 0;"),
            Comment(f"{flag} = 0;"),
            Comment(f"{flags} = 0;"),
            Comment(),
            Comment(),
        )


class FlagLookupQASMActiveCorrectionZ(Block):
    """Active Z-basis correction using flag-based lookup for Steane code.

    This class applies Z-basis error corrections based on syndrome measurements
    and flag qubit information using a lookup table approach.
    """

    def __init__(
        self,
        qubits: QReg,
        syn: CReg,
        syndromes: CReg,
        raw_syn: CReg,
        pf: Bit,
        flag: CReg,
        flags: CReg,
        scratch: CReg,
        pf_bit_copy: Bit = None,
    ) -> None:
        """Initialize Z-basis active error correction with flag-based lookup.

        Args:
            qubits: Quantum register containing the logical qubit data.
            syn: Classical register containing current syndrome values.
            syndromes: Classical register for processed syndrome values.
            raw_syn: Classical register storing raw syndrome values.
            pf: Pauli frame bit for tracking logical operations.
            flag: Classical register containing flag qubit measurements.
            flags: Classical register for flag status.
            scratch: Classical register used for intermediate calculations.
            pf_bit_copy: Optional bit to copy the Pauli frame value to.
        """
        super().__init__()
        q = qubits

        # qasm_syn_decoder('X', syn_x, flag_x, 'last_raw_syn_x', 'pf_z1')
        # qasm_syn_decoder(basis_check, syn, flag, raw_syn, pf, pf_index=0)
        self.extend(
            FlagLookupQASM(
                basis="Z",
                syn=syn,
                syndromes=syndromes,
                raw_syn=raw_syn,
                pf=pf,
                flag=flag,
                flags=flags,
                scratch=scratch,
            ),
        )

        if pf_bit_copy is not None:
            self.extend(
                Comment(),
                Comment("copy Pauli frame"),
                pf_bit_copy.set(pf),
            )

        self.extend(
            Comment(),
            Comment(),
            Comment("ACTIVE ERROR CORRECTION FOR Z SYNDROMES"),
            Comment(),
            scratch.set(0),
            Comment(),
            Comment("only part that differs for X vs Z syns V"),
            If(syndromes[0] == 1).Then(scratch.set(scratch ^ 14)),
            If(syndromes[1] == 1).Then(scratch.set(scratch ^ 12)),
            If(syndromes[2] == 1).Then(scratch.set(scratch ^ 48)),
            Comment(),
            Comment("logical operator"),
            If(pf == 1).Then(scratch.set(scratch ^ 112)),
            Comment(),
            Comment("not possible for X stabilizers V"),
            Comment(f"if({scratch[0]} == 1) z {q[0]};"),
            If(scratch[1] == 1).Then(qubit.X(q[1])),
            If(scratch[2] == 1).Then(qubit.X(q[2])),
            If(scratch[3] == 1).Then(qubit.X(q[3])),
            If(scratch[4] == 1).Then(qubit.X(q[4])),
            If(scratch[5] == 1).Then(qubit.X(q[5])),
            If(scratch[6] == 1).Then(qubit.X(q[6])),
            Comment(),
            pf.set(0),
            Comment(f"{syndromes} = 0;"),
            raw_syn.set(0),
            Comment(f"{syn} = 0;"),
            Comment(f"{flag} = 0;"),
            Comment(f"{flags} = 0;"),
            Comment(),
        )
