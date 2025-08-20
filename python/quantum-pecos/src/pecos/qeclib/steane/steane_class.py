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

from typing import TYPE_CHECKING, NoReturn
from warnings import warn

from pecos.qeclib.steane.decoders.lookup import (
    FlagLookupQASMActiveCorrectionX,
    FlagLookupQASMActiveCorrectionZ,
)
from pecos.qeclib.steane.gates_sq import paulis, sqrt_paulis
from pecos.qeclib.steane.gates_sq.hadamards import H
from pecos.qeclib.steane.gates_tq import transversal_tq
from pecos.qeclib.steane.meas.destructive_meas import MeasDecode
from pecos.qeclib.steane.preps.pauli_states import PrepRUS
from pecos.qeclib.steane.preps.t_plus_state import (
    PrepEncodeTPlusFTRUS,
    PrepEncodeTPlusNonFT,
)
from pecos.qeclib.steane.qec.qec_3parallel import (
    ParallelFlagQEC,
    ParallelFlagQECActiveCorrection,
)
from pecos.qeclib.steane.syn_extract.bare import SynExtractBare
from pecos.qeclib.steane.syn_extract.flagged import SynExtractFlagged
from pecos.slr import Block, CReg, If, Permute, QReg, Vars

if TYPE_CHECKING:
    from pecos.slr import Bit


class Steane(Vars):
    """A generic implementation of a Steane code and operations.

    This represents one particular choice of Steane protocols. For finer control construct your own class
    or utilize the library of Steane code protocols directly.
    """

    def __init__(
        self,
        name: str,
        default_rus_limit: int = 3,
        ancillas: QReg | None = None,
        flag_qubits: QReg | None = None,
    ) -> None:
        """Initialize a Steane code instance with associated quantum and classical registers.

        Args:
            name: Name prefix for all registers associated with this Steane code instance.
            default_rus_limit: Default limit for repeat-until-success procedures. Defaults to 3.
            ancillas: Optional pre-existing ancilla register. If None, creates a new 3-qubit
                ancilla register. Must have at least 3 qubits if provided.
            flag_qubits: Optional pre-existing flag qubit register. If None, creates a new 3-qubit
                flag register when needed.

        Raises:
            ValueError: If provided ancilla register has fewer than 3 qubits.
        """
        super().__init__()
        # Set the source class for code generation
        self.source_class = self.__class__.__name__
        self.check_indices = [[2, 1, 3, 0], [5, 2, 1, 4], [6, 5, 2, 3]]

        self.d = QReg(f"{name}_d", 7)
        self.a = ancillas or QReg(f"{name}_a", 3)
        if flag_qubits is not None:
            self.f = flag_qubits or QReg(f"{name}_f", 3)
        else:
            self.f = None
        self.c = CReg(f"{name}_c", 32)

        # if self.a.size < 3:
        #     msg = f"Steane ancilla registers must have >= 3 qubits (provided: {self.a.size})"
        #     raise ValueError(msg)

        # TODO: Make it so I can put these in self.c... need to convert things like if(c) and c = a ^ b, a = 0;
        #  to allow lists of bits
        self.syn_meas = CReg(f"{name}_syn_meas", 32)
        self.last_raw_syn_x = CReg(f"{name}_last_raw_syn_x", 32)
        self.last_raw_syn_z = CReg(f"{name}_last_raw_syn_z", 32)
        self.scratch = CReg(f"{name}_scratch", 32)
        self.flag_x = CReg(f"{name}_flag_x", 3)
        self.flag_z = CReg(f"{name}_flags_z", 3)

        self.flags = CReg(f"{name}_flags", 3)  # weird error when using [c, c, c]

        self.raw_meas = CReg(f"{name}_raw_meas", 7)

        self.syn_x = CReg(f"{name}_syn_x", 3)
        self.syn_z = CReg(f"{name}_syn_z", 3)
        self.syndromes = CReg(f"{name}_syndromes", 3)
        self.verify_prep = CReg(f"{name}_verify_prep", 32)

        self.vars = [
            self.d,
        ]

        if ancillas is None:
            self.vars.append(self.a)

        if flag_qubits is None:
            self.vars.append(self.f)

        self.vars.extend(
            [
                self.c,
                self.syn_meas,
                self.last_raw_syn_x,
                self.last_raw_syn_z,
                self.scratch,
                self.flag_x,
                self.flag_z,
                self.flags,
                self.raw_meas,
                self.syn_x,
                self.syn_z,
                self.syndromes,
                self.verify_prep,
            ],
        )

        # derived classical registers
        c = self.c
        self.log_raw = c[1]
        self.log = c[2]
        self.pf_x = c[3]
        self.pf_z = c[4]
        self.t_meas = c[5]
        self.tdg_meas = c[6]

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

    def px(self, reject: Bit | None = None, rus_limit: int | None = None) -> Block:
        """Prepare logical |+X>, a.k.a. |+>."""
        return self.p("+X", reject=reject, rus_limit=rus_limit)

    def pnx(self, reject: Bit | None = None, rus_limit: int | None = None) -> Block:
        """Prepare logical |-X>, a.k.a. |->."""
        return self.p("-X", reject=reject, rus_limit=rus_limit)

    def py(self, reject: Bit | None = None, rus_limit: int | None = None) -> Block:
        """Prepare logical |+Y>, a.k.a. |+i>."""
        return self.p("+Y", reject=reject, rus_limit=rus_limit)

    def pny(self, reject: Bit | None = None, rus_limit: int | None = None) -> Block:
        """Prepare logical |-Y>, a.k.a. |-i>."""
        return self.p("-Y", reject=reject, rus_limit=rus_limit)

    def pz(self, reject: Bit | None = None, rus_limit: int | None = None) -> Block:
        """Prepare logical |+Z>, a.k.a. |0>."""
        return self.p("+Z", reject=reject, rus_limit=rus_limit)

    def pnz(self, reject: Bit | None = None, rus_limit: int | None = None) -> Block:
        """Prepare logical |-Z>, a.k.a. |1>."""
        return self.p("-Z", reject=reject, rus_limit=rus_limit)

    def nonft_prep_t_plus_state(self) -> Block:
        """Prepare logical T|+X> in a non-fault tolerant manner."""
        return PrepEncodeTPlusNonFT(
            q=self.d,
        )

    def prep_t_plus_state(
        self,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Prepare logical T|+X> in a fault tolerant manner."""
        block = Block(
            self.scratch.set(0),
            PrepEncodeTPlusFTRUS(
                d=self.d,
                a=self.a,
                out=self.scratch,
                reject=self.scratch[2],  # the first two bits are used by "out"
                flag_x=self.flag_x,
                flag_z=self.flag_z,
                flags=self.flags,
                last_raw_syn_x=self.last_raw_syn_x,
                last_raw_syn_z=self.last_raw_syn_z,
                limit=rus_limit or self.default_rus_limit,
            ),
        )
        if reject is not None:
            block.extend(reject.set(self.scratch[2]))
        return block

    def nonft_prep_tdg_plus_state(self) -> Block:
        """Prepare logical Tdg|+X> in a non-fault tolerant manner."""
        return Block(
            self.nonft_prep_t_plus_state(),
            self.z(),
        )

    def prep_tdg_plus_state(
        self,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Prepare logical Tdg|+X> in a fault tolerant manner."""
        return Block(
            self.prep_t_plus_state(reject=reject, rus_limit=rus_limit),
            self.szdg(),
        )

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

    def nonft_t(self, aux: Steane) -> Block:
        """T gate via teleportation using non-fault-tolerant initialization of the T|+> state."""
        return Block(
            aux.nonft_prep_t_plus_state(),
            self.cx(aux),
            aux.mz(self.t_meas),
            If(self.t_meas == 1).Then(self.sz()),
        )

    def t(
        self,
        aux: Steane,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """T gate via teleportation using fault-tolerant initialization of the T|+> state."""
        return Block(
            aux.prep_t_plus_state(reject=reject, rus_limit=rus_limit),
            self.cx(aux),
            aux.mz(self.t_meas),
            If(self.t_meas == 1).Then(self.sz()),  # SZ/S correction.
        )

    def nonft_tdg(self, aux: Steane) -> Block:
        """Tdg gate via teleportation using non-fault-tolerant initialization of the Tdg|+> state."""
        return Block(
            aux.nonft_prep_tdg_plus_state(),
            self.cx(aux),
            aux.mz(self.tdg_meas),
            If(self.tdg_meas == 1).Then(self.szdg()),
        )

    def tdg(
        self,
        aux: Steane,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Tdg gate via teleportation using fault-tolerant initialization of the Tdg|+> state."""
        return Block(
            aux.prep_tdg_plus_state(reject=reject, rus_limit=rus_limit),
            self.cx(aux),
            aux.mz(self.tdg_meas),
            If(self.tdg_meas == 1).Then(self.szdg()),  # SZdg/Sdg correction.
        )

    #  Begin Experimental: ------------------------------------

    def nonft_t_tel(self, aux: Steane) -> Block:
        """T gate via teleportation (non-fault-tolerant, experimental).

        Warning:
            This is experimental.

        T gate via teleportation using non-fault-tolerant initialization of the T|+> state.

        This version teleports the logical qubit from the original qubit to the auxiliary logical qubit. For
        convenience, the qubits are relabeled, so you can continue to use the original Steane code logical qubit.
        """
        warn("Using experimental feature: nonft_t_tel", stacklevel=2)
        return Block(
            aux.nonft_prep_t_plus_state(),
            aux.cx(self),
            self.mz(self.t_meas),
            If(self.t_meas == 1).Then(aux.x(), aux.sz()),
            self.permute(aux),
        )

    def t_tel(
        self,
        aux: Steane,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """T gate via teleportation (fault-tolerant, experimental).

        Warning:
            This is experimental.

        T gate via teleportation using fault-tolerant initialization of the T|+> state.

        This version teleports the logical qubit from the original qubit to the auxiliary logical qubit. For
        convenience, the qubits are relabeled, so you can continue to use the original Steane code logical qubit.
        """
        warn("Using experimental feature: t_tel", stacklevel=2)
        return Block(
            aux.prep_t_plus_state(reject=reject, rus_limit=rus_limit),
            aux.cx(self),
            self.mz(self.t_meas),
            If(self.t_meas == 1).Then(aux.x(), aux.sz()),  # SZ/S correction.
            self.permute(aux),
        )

    def nonft_tdg_tel(self, aux: Steane) -> Block:
        """T† gate via teleportation (non-fault-tolerant, experimental).

        Warning:
            This is experimental.

        Tdg gate via teleportation using non-fault-tolerant initialization of the Tdg|+> state.

        This version teleports the logical qubit from the original qubit to the auxiliary logical qubit. For
        convenience, the qubits are relabeled, so you can continue to use the original Steane code logical qubit.
        """
        warn("Using experimental feature: nonft_tdg_tel", stacklevel=2)
        return Block(
            aux.nonft_prep_tdg_plus_state(),
            aux.cx(self),
            self.mz(self.tdg_meas),
            If(self.tdg_meas == 1).Then(aux.x(), aux.szdg()),
            self.permute(aux),
        )

    def tdg_tel(
        self,
        aux: Steane,
        reject: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """T† gate via teleportation (fault-tolerant, experimental).

        Warning:
            This is experimental.

        Tdg gate via teleportation using fault-tolerant initialization of the Tdg|+> state.

        This version teleports the logical qubit from the original qubit to the auxiliary logical qubit. For
        convenience, the qubits are relabeled, so you can continue to use the original Steane code logical qubit.
        """
        warn("Using experimental feature: tdg_tel", stacklevel=2)
        return Block(
            aux.prep_tdg_plus_state(reject=reject, rus_limit=rus_limit),
            aux.cx(self),
            self.mz(self.tdg_meas),
            If(self.t_meas == 1).Then(aux.x(), aux.szdg()),  # SZdg/Sdg correction.
            self.permute(aux),
        )

    def t_cor(
        self,
        aux: Steane,
        reject: Bit | None = None,
        flag: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """T gate via teleportation using fault-tolerant initialization of the T|+> state.

        Applies active corrections of errors diagnozed by the measurement for gate teleportation.
        """
        warn("Using experimental feature: t_cor", stacklevel=2)
        block = Block(
            # gate teleportation without logical correction
            aux.prep_t_plus_state(reject=reject, rus_limit=rus_limit),
            self.cx(aux),
            aux.mz(self.t_meas),
            # active error correction
            self.syn_z.set(aux.syn_meas),
            self.last_raw_syn_z.set(0),
            self.pf_x.set(0),
            FlagLookupQASMActiveCorrectionZ(
                self.d,
                self.syn_z,
                self.syn_z,
                self.last_raw_syn_z,
                self.pf_x,
                self.syn_z,
                self.syn_z,
                self.scratch,
            ),
            # logical correction
            If(self.t_meas == 1).Then(self.sz()),
        )
        if flag is not None:
            block.extend(If(self.syn_z != 0).Then(flag.set(1)))
        return block

    def tdg_cor(
        self,
        aux: Steane,
        reject: Bit | None = None,
        flag: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Tdg gate via teleportation using fault-tolerant initialization of the Tdg|+> state.

        Applies active corrections of errors diagnozed by the measurement for gate teleportation.
        """
        warn("Using experimental feature: t_cor", stacklevel=2)
        block = Block(
            # gate teleportation without logical correction
            aux.prep_tdg_plus_state(reject=reject, rus_limit=rus_limit),
            self.cx(aux),
            aux.mz(self.tdg_meas),
            # active error correction
            self.syn_z.set(aux.syn_meas),
            self.last_raw_syn_z.set(0),
            self.pf_x.set(0),
            FlagLookupQASMActiveCorrectionZ(
                self.d,
                self.syn_z,
                self.syn_z,
                self.last_raw_syn_z,
                self.pf_x,
                self.syn_z,
                self.syn_z,
                self.scratch,
            ),
            # logical correction
            If(self.tdg_meas == 1).Then(self.szdg()),
        )
        if flag is not None:
            block.extend(If(self.syn_z != 0).Then(flag.set(1)))
        return block

    # End Experimental: ------------------------------------

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

    def m(self, meas_basis: str, log: Bit | None = None) -> Block:
        """Destructively measure the logical qubit in some Pauli basis."""
        block = MeasDecode(
            q=self.d,
            meas_basis=meas_basis,
            meas=self.raw_meas,
            log_raw=self.log_raw,
            log=self.log,
            syn_meas=self.syn_meas,
            pf_x=self.pf_x,
            pf_z=self.pf_z,
            last_raw_syn_x=self.last_raw_syn_x,
            last_raw_syn_z=self.last_raw_syn_z,
        )
        if log is not None:
            block.extend(log.set(self.log))
        return block

    def mx(self, log: Bit | None = None) -> Block:
        """Logical destructive measurement of the logical X operator."""
        return self.m("X", log=log)

    def my(self, log: Bit | None = None) -> Block:
        """Logical destructive measurement of the logical Y operator."""
        return self.m("Y", log=log)

    def mz(self, log: Bit | None = None) -> Block:
        """Logical destructive measurement of the logical Z operator."""
        return self.m("Z", log=log)

    def qec(self, flag: Bit | None = None) -> Block:
        """Perform quantum error correction using parallel flag-based active correction.

        Args:
            flag: Optional flag bit for conditional execution.

        Returns:
            Block containing the quantum error correction operations.
        """
        block = ParallelFlagQECActiveCorrection(
            q=self.d,
            a=self.a,
            flag_x=self.flag_x,
            flag_z=self.flag_z,
            flags=self.flags,
            syn_x=self.syn_x,
            syn_z=self.syn_z,
            last_raw_syn_x=self.last_raw_syn_x,
            last_raw_syn_z=self.last_raw_syn_z,
            syndromes=self.syndromes,
            pf_x=self.pf_x,
            pf_z=self.pf_z,
            scratch=self.scratch,
        )
        if flag is not None:
            block.extend(If(self.flags != 0).Then(flag.set(1)))
        return block

    def qec_not_active(
        self,
        flag: Bit | None = None,
        pf_x: Bit | None = None,
        pf_z: Bit | None = None,
        flag_x: CReg | None = None,
        flag_z: CReg | None = None,
        syn_x: CReg | None = None,
        syn_z: CReg | None = None,
    ) -> Block:
        """Perform quantum error correction using parallel flag-based without active correction.

        There are potentially three syndrome extraction paths take:
            0: XZZ flag_x = 000, flag_z = 000 -> ZXX flag_x = 000, flag_z = 000 -> Done
            1: XZZ flag_x = 00*, flag_z = **0 -> measure XXXZZZ (syn_x, syn_z)
            2: XZZ flag_x = 000, flag_z = 000 -> ZXX flag_x = **0, flag_z = 00* -> measure XXXZZZ (syn_x, syn_z)
        (where at least one of the *s is 1)

        Therefore:
            if flag_x & flag_z == 0, we went down path 0
            if flag_x[0] | flag_z[1] | flag_z[2] == 1, we went down path 1
            if flag_x[1] | flag_z[2] | flag_z[0] == 1, we went down path 2

        Args:
            flag: Optional flag bit for conditional execution.
            flag_x: Optional CReg of the syndrome measured by the X checks for the first two of flagged syndrome
                   extractions. It is a raw syndrome made during the first two rounds of syndrome extraction.
            flag_z: Optional CReg of the syndrome measured by the X checks for the first two of flagged syndrome
                   extractions. It is a raw syndrome made during the first two rounds of syndrome extraction.
            syn_x: Optional CReg of the syndrome measured by the X checks for the last round of non-flagged syndrome
                   extraction. It is a raw syndrome made during the final round of syndrome extraction.
            syn_z: Optional CReg of the syndrome measured by the Z checks for the last round of non-flagged syndrome
                   extraction. It is a raw syndrome made during the final round of syndrome extraction.
            pf_x: Optional Pauli frame bit for logical X corrections determined by lookup table decoder
            pf_z: Optional Pauli frame bit for logical Z corrections determined by lookup table decoder

        Returns:
            Block containing the quantum error correction operations.
        """
        block = Block()

        block.extend(
            ParallelFlagQEC(
                q=self.d,
                a=self.a,
                flag_x=self.flag_x,
                flag_z=self.flag_z,
                flags=self.flags,
                syn_x=self.syn_x,
                syn_z=self.syn_z,
                last_raw_syn_x=self.last_raw_syn_x,
                last_raw_syn_z=self.last_raw_syn_z,
                syndromes=self.syndromes,
                pf_x=self.pf_x,
                pf_z=self.pf_z,
                scratch=self.scratch,
            ),
        )
        if flag is not None:
            block.extend(
                If(self.flags != 0).Then(flag.set(1)),
            )

        if flag_x is not None:
            if len(flag_x) != 3:
                msg = f"flag_x must have length 3, got {len(flag_x)}"
                raise ValueError(msg)
            block.extend(flag_x.set(self.flag_x))

        if flag_z is not None:
            if len(flag_z) != 3:
                msg = f"flag_z must have length 3, got {len(flag_z)}"
                raise ValueError(msg)
            block.extend(flag_z.set(self.flag_z))

        if syn_x is not None:
            if len(syn_x) != 3:
                msg = f"syn_x must have length 3, got {len(syn_x)}"
                raise ValueError(msg)
            block.extend(syn_x.set(self.syn_x))

        if syn_z is not None:
            if len(syn_z) != 3:
                msg = f"syn_z must have length 3, got {len(syn_z)}"
                raise ValueError(msg)
            block.extend(syn_z.set(self.syn_z))

        if pf_x is not None:
            block.extend(pf_x.set(self.pf_x))

        if pf_z is not None:
            block.extend(pf_z.set(self.pf_z))

        return block

    def qec_steane(
        self,
        aux: Steane,
        reject_x: Bit | None = None,
        reject_z: Bit | None = None,
        flag_x: Bit | None = None,
        flag_z: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Run a Steane-type error-correction cycle of this code."""
        return Block(
            self.qec_steane_x(
                aux,
                reject=reject_x,
                flag=flag_x,
                rus_limit=rus_limit,
            ),
            self.qec_steane_z(
                aux,
                reject=reject_z,
                flag=flag_z,
                rus_limit=rus_limit,
            ),
        )

    def qec_steane_x(
        self,
        aux: Steane,
        reject: Bit | None = None,
        flag: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Run a Steane-type error-correction cycle for X errors."""
        warn("Using experimental feature: qec_steane_x", stacklevel=2)
        block = Block(
            aux.px(reject=reject, rus_limit=rus_limit),
            self.cx(aux),
            aux.mz(),
            self.syn_z.set(aux.syn_meas),
            self.last_raw_syn_z.set(0),
            self.pf_x.set(0),
            FlagLookupQASMActiveCorrectionZ(
                self.d,
                self.syn_z,
                self.syn_z,
                self.last_raw_syn_z,
                self.pf_x,
                self.syn_z,
                self.syn_z,
                self.scratch,
            ),
        )
        if flag is not None:
            block.extend(If(self.syn_z != 0).Then(flag.set(1)))
        return block

    def qec_steane_z(
        self,
        aux: Steane,
        reject: Bit | None = None,
        flag: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Run a Steane-type error-correction cycle for Z errors."""
        warn("Using experimental feature: qec_steane_z", stacklevel=2)
        block = Block(
            aux.pz(reject=reject, rus_limit=rus_limit),
            aux.cx(self),
            aux.mx(),
            self.syn_x.set(aux.syn_meas),
            self.last_raw_syn_x.set(0),
            self.pf_z.set(0),
            FlagLookupQASMActiveCorrectionX(
                self.d,
                self.syn_x,
                self.syn_x,
                self.last_raw_syn_x,
                self.pf_z,
                self.syn_x,
                self.syn_x,
                self.scratch,
            ),
        )
        if flag is not None:
            block.extend(If(self.syn_x != 0).Then(flag.set(1)))
        return block

    def qec_tel(
        self,
        aux: Steane,
        reject_x: Bit | None = None,
        reject_z: Bit | None = None,
        flag_x: Bit | None = None,
        flag_z: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Run a teleportation-based error correction cycle."""
        return Block(
            self.qec_tel_x(aux, reject_x, flag_x, rus_limit),
            self.qec_tel_z(aux, reject_z, flag_z, rus_limit),
        )

    def qec_tel_x(
        self,
        aux: Steane,
        reject: Bit | None = None,
        flag: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Run a teleportation-based error correction cycle for X errors."""
        warn("Using experimental feature: qec_tel_x", stacklevel=2)
        block = Block(
            # teleport
            aux.px(reject=reject, rus_limit=rus_limit),
            aux.cx(self),
            self.mz(),
            If(self.log == 1).Then(aux.x()),
            Permute(self.d, aux.d),
            # update syndromes and pauli frame
            self.last_raw_syn_x.set(0),
            self.last_raw_syn_z.set(0),
            self.syn_z.set(self.syn_meas),
            self.pf_x.set(0),
        )
        if flag is not None:
            block.extend(If(self.syn_meas != 0).Then(flag.set(1)))
        return block

    def qec_tel_z(
        self,
        aux: Steane,
        reject: Bit | None = None,
        flag: Bit | None = None,
        rus_limit: int | None = None,
    ) -> Block:
        """Run a teleportation-based error correction cycle for Z errors."""
        warn("Using experimental feature: qec_tel_z", stacklevel=2)
        block = Block(
            # teleport
            aux.pz(reject=reject, rus_limit=rus_limit),
            self.cx(aux),
            self.mx(),
            If(self.log == 1).Then(aux.z()),
            Permute(self.d, aux.d),
            # update syndromes and pauli frame
            self.last_raw_syn_x.set(0),
            self.last_raw_syn_z.set(0),
            self.syn_x.set(self.syn_meas),
            self.pf_z.set(0),
        )
        if flag is not None:
            block.extend(If(self.syn_meas != 0).Then(flag.set(1)))
        return block

    def qec_knill(self) -> NoReturn:
        """Prepare a Bell state and then teleport."""
        # TODO: ...
        msg = "qec_knill not implemented."
        raise NotImplementedError(msg)

    def syn_bare(self, syn: CReg) -> Block:
        """One single syndrome bit per check using bare syndrome extraction."""
        return SynExtractBare(self.d, self.a, self.check_indices, syn)

    def syn_flagged(self, syn: CReg, flags: CReg) -> Block:
        """One single syndrome bit and one single flag bit per check."""
        return SynExtractFlagged(self.d, self.a, self.f, self.check_indices, syn, flags)

    def syn_2para_v1_flagged(self) -> NoReturn:
        """Two-parallel syndrome extraction version 1 with flagging (not implemented).

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        # TODO: ...
        msg = "syn_2para_v1_flagged not implemented."
        raise NotImplementedError(msg)

    def syn_2para_v2_flagged(self) -> NoReturn:
        """Two-parallel syndrome extraction version 2 with flagging (not implemented).

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        # TODO: ...
        msg = "syn_2para_v2_flagged not implemented."
        raise NotImplementedError(msg)

    def permute(self, other: Steane) -> Block:
        """Permute this code block (including both quantum and classical registers) with another."""
        block = Block(
            Permute(self.d, other.d),
            Permute(self.a, other.a),
        )
        # TODO: Use Permute on classical variables rather that a custom solution
        for var_a, var_b in zip(self.vars, other.vars, strict=False):
            if isinstance(var_a, CReg):
                block.extend(
                    var_a.set(var_a ^ var_b),
                    var_b.set(var_b ^ var_a),
                    var_a.set(var_a ^ var_b),
                )
        return block
