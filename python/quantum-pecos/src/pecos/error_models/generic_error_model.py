"""Generic error model with comprehensive noise support.

This module provides a generic error model implementation that supports
various noise types including bitflip, depolarizing, and leakage errors
for comprehensive quantum error correction simulations.
"""

# Copyright 2023 The PECOS Developers
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

import pecos as pc
from pecos.error_models.noise_impl.noise_initz_bitflip_leakage import (
    noise_initz_bitflip_leakage,
)
from pecos.error_models.noise_impl.noise_meas_bitflip_leakage import (
    noise_meas_bitflip_leakage,
)
from pecos.error_models.noise_impl.noise_sq_depolarizing_leakage import (
    noise_sq_depolarizing_leakage,
)
from pecos.error_models.noise_impl.noise_tq_depolarizing_leakage import (
    noise_tq_depolarizing_leakage,
)
from pecos.error_models.noise_impl_old.gate_groups import one_qubits, two_qubits

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos.protocols import MachineProtocol
    from pecos.reps.pyphir.block_types import SeqBlock
    from pecos.reps.pyphir.op_types import QOp

two_qubit_paulis = {
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
}
SYMMETRIC_P2_PAULI_MODEL = dict.fromkeys(two_qubit_paulis, 1 / 15)

one_qubit_paulis = {
    "X",
    "Y",
    "Z",
}
SYMMETRIC_P1_PAULI_MODEL = dict.fromkeys(one_qubit_paulis, 1 / 3)


class GenericErrorModel:
    """Parameterized error mode."""

    def __init__(self, error_params: dict) -> None:
        """Initialize a generic error model with support for leakage.

        Args:
            error_params: Dictionary containing error parameters including:
                - p1: Single-qubit gate error probability
                - p2: Two-qubit gate error probability
                - p_meas: Measurement error probability
                - p_init: Initialization error probability
                - scale: Optional scaling factor for all error rates
                - p1_error_model: Optional custom single-qubit Pauli error distribution
                - p2_error_model: Optional custom two-qubit Pauli error distribution
                - p2_mem: Optional memory error probability for two-qubit gates
        """
        self.error_params = dict(error_params)
        self.machine = None
        self.num_qubits = None
        self._eparams = None

    def reset(self) -> GenericErrorModel:
        """Reset error generator for another round of syndrome extraction."""
        return GenericErrorModel(error_params=self.error_params)

    def init(self, num_qubits: int, machine: MachineProtocol | None = None) -> None:
        """Initialize the generic error model.

        Args:
            num_qubits: Number of qubits in the system.
            machine: Optional machine protocol for hardware-specific behavior.
        """
        self.machine = machine
        self.num_qubits = num_qubits

        if not self.error_params:
            msg = "Error params not set!"
            raise Exception(msg)

        self._eparams = dict(self.error_params)
        self._scale()

        if "p1_error_model" not in self._eparams:
            self._eparams["p1_error_model"] = SYMMETRIC_P1_PAULI_MODEL

        if "p2_error_model" not in self._eparams:
            self._eparams["p2_error_model"] = SYMMETRIC_P2_PAULI_MODEL

        if "p2_mem" in self._eparams and "p2_mem_error_model" not in self._eparams:
            self._eparams["p2_mem_error_model"] = SYMMETRIC_P2_PAULI_MODEL

    def _scale(self) -> None:
        # conversion from average error to total error
        self._eparams["p1"] *= 3 / 2
        self._eparams["p2"] *= 5 / 4

        scale = self._eparams.get("scale", 1.0)
        self._eparams["p1"] *= scale
        self._eparams["p2"] *= scale

        if isinstance(self._eparams["p_meas"], tuple):
            self._eparams["p_meas"] = pc.mean(self._eparams["p_meas"])

        self._eparams["p_meas"] *= scale
        self._eparams["p_init"] *= scale

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""

    def process(
        self,
        qops: list[QOp],
        _call_back: Callable | None = None,
    ) -> list[QOp | SeqBlock]:
        """Process quantum operations and apply generic errors.

        Args:
            qops: List of quantum operations to process.
            call_back: Optional callback function for additional processing.

        Returns:
            List of quantum operations with applied errors.
        """
        noisy_ops = []

        for op in qops:
            qops_after = None
            qops_before = None
            erroneous_ops = None

            # ########################################
            # INITS WITH X NOISE
            if op.name in {"init |0>", "Init", "Init +Z"}:
                qops_after = noise_initz_bitflip_leakage(
                    op,
                    p=self._eparams["p_init"],
                    machine=self.machine,
                )

            # ########################################
            # ONE QUBIT GATES
            elif op.name in one_qubits:
                erroneous_ops = noise_sq_depolarizing_leakage(
                    op,
                    p=self._eparams["p1"],
                    noise_dict=self._eparams["p1_error_model"],
                    machine=self.machine,
                )

            # ########################################
            # TWO QUBIT GATES
            elif op.name in two_qubits:
                qops_after = noise_tq_depolarizing_leakage(
                    op,
                    p=self._eparams["p2"],
                    noise_dict=self._eparams["p2_error_model"],
                    machine=self.machine,
                )

                if self._eparams.get("p2_mem"):
                    qops_mem = noise_tq_depolarizing_leakage(
                        op,
                        p=self._eparams["p2_mem"],
                        noise_dict=self._eparams["p2_mem_error_model"],
                        machine=self.machine,
                    )

                    if qops_after:
                        qops_after = qops_after.extend(qops_mem)

            # ########################################
            # MEASURE X NOISE
            elif op.name in {"measure Z", "Measure", "Measure +Z"}:
                erroneous_ops = noise_meas_bitflip_leakage(
                    op,
                    p=self._eparams["p_meas"],
                    machine=self.machine,
                )

            else:
                msg = f"This error model doesn't handle gate: {op.name}!"
                raise Exception(msg)

            if qops_before:
                noisy_ops.extend(qops_before)

            if erroneous_ops is None:
                noisy_ops.append(op)
            else:
                noisy_ops.extend(erroneous_ops)

            if qops_after:
                noisy_ops.extend(qops_after)

        return noisy_ops
