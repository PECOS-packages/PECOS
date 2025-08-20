"""Three-parallel quantum error correction for the Steane 7-qubit code.

This module provides three-parallel quantum error correction implementations for the Steane 7-qubit code, enabling
fault-tolerant error detection and correction using parallel syndrome extraction circuits.
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

from pecos.qeclib.steane.decoders.lookup import (
    FlagLookupQASM,
    FlagLookupQASMActiveCorrectionX,
    FlagLookupQASMActiveCorrectionZ,
)
from pecos.qeclib.steane.syn_extract.six_check_nonflagging import SixUnflaggedSyn
from pecos.qeclib.steane.syn_extract.three_parallel_flagging import (
    ThreeParallelFlaggingXZZ,
    ThreeParallelFlaggingZXX,
)
from pecos.slr import Bit, Block, CReg, If, QReg


class ParallelFlagQECActiveCorrection(Block):
    """Defining QEC Block that does adaptive syndrome extraction, decodes, and updates the Paul frame."""

    def __init__(
        self,
        q: QReg,
        a: QReg,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        syn_x: CReg,
        syn_z: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        syndromes: CReg,
        pf_x: Bit,
        pf_z: Bit,
        scratch: CReg,
    ) -> None:
        """Initialize ParallelFlagQECActiveCorrection block for error correction.

        Args:
            q: Data register containing the 7 qubits of the Steane code.
            a: Ancilla register for syndrome extraction.
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            syn_x: Classical register for X syndromes.
            syn_z: Classical register for Z syndromes.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            syndromes: Classical register for syndrome storage.
            pf_x: Pauli frame bit for X errors.
            pf_z: Pauli frame bit for Z errors.
            scratch: Scratch classical register for intermediate calculations.
        """
        super().__init__(
            # flagging XZZ checks
            ThreeParallelFlaggingXZZ(
                q,
                a,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
            # flagging ZXX checks
            If(flags == 0).Then(
                ThreeParallelFlaggingZXX(
                    q,
                    a,
                    flag_x,
                    flag_z,
                    flags,
                    last_raw_syn_x,
                    last_raw_syn_z,
                ),
            ),
            # Remeasure all the checks unflagged
            If(flags != 0).Then(
                SixUnflaggedSyn(q, a, syn_x, syn_z),
            ),
            FlagLookupQASMActiveCorrectionX(
                q,
                syn_x,
                syndromes,
                last_raw_syn_x,
                pf_z,
                flag_x,
                flags,
                scratch,
            ),
            FlagLookupQASMActiveCorrectionZ(
                q,
                syn_z,
                syndromes,
                last_raw_syn_z,
                pf_x,
                flag_z,
                flags,
                scratch,
            ),
        )


class ParallelFlagQEC(Block):
    """Defining QEC Block that does adaptive syndrome extraction, decodes, and updates the Paul frame."""

    def __init__(
        self,
        q: QReg,
        a: QReg,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        syn_x: CReg,
        syn_z: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        syndromes: CReg,
        pf_x: Bit,
        pf_z: Bit,
        scratch: CReg,
    ) -> None:
        """Initialize ParallelFlagQECActiveCorrection block for error correction.

        Args:
            q: Data register containing the 7 qubits of the Steane code.
            a: Ancilla register for syndrome extraction.
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            syn_x: Classical register for X syndromes.
            syn_z: Classical register for Z syndromes.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            syndromes: Classical register for syndrome storage.
            pf_x: Pauli frame bit for X errors.
            pf_z: Pauli frame bit for Z errors.
            scratch: Scratch classical register for intermediate calculations.
        """
        super().__init__(
            # flagging XZZ checks
            ThreeParallelFlaggingXZZ(
                q,
                a,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
            # flagging ZXX checks
            If(flags == 0).Then(
                ThreeParallelFlaggingZXX(
                    q,
                    a,
                    flag_x,
                    flag_z,
                    flags,
                    last_raw_syn_x,
                    last_raw_syn_z,
                ),
            ),
            # Remeasure all the checks unflagged
            If(flags != 0).Then(
                SixUnflaggedSyn(q, a, syn_x, syn_z),
            ),
            FlagLookupQASM(
                basis="X",
                syn=syn_x,
                syndromes=syndromes,
                raw_syn=last_raw_syn_x,
                pf=pf_z,
                flag=flag_x,
                flags=flags,
                scratch=scratch,
            ),
            FlagLookupQASM(
                basis="Z",
                syn=syn_z,
                syndromes=syndromes,
                raw_syn=last_raw_syn_z,
                pf=pf_x,
                flag=flag_z,
                flags=flags,
                scratch=scratch,
            ),
        )
