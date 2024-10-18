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

import numpy as np

from pecos.error_models.error_model_abc import ErrorModel
from pecos.error_models.noise_impl.noise_initz_bitflip import noise_initz_bitflip
from pecos.error_models.noise_impl.noise_meas_bitflip import noise_meas_bitflip
from pecos.error_models.noise_impl.noise_sq_depolarizing import noise_sq_depolarizing
from pecos.error_models.noise_impl.noise_tq_depolarizing import noise_tq_depolarizing
from pecos.error_models.noise_impl_old.gate_groups import one_qubits, two_qubits

if TYPE_CHECKING:
    from pecos.reps.pypmir.block_types import SeqBlock
    from pecos.reps.pypmir.op_types import QOp

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
SYMMETRIC_P2_PAULI_MODEL = {p: 1 / 15 for p in two_qubit_paulis}

one_qubit_paulis = {
    "X",
    "Y",
    "Z",
}
SYMMETRIC_P1_PAULI_MODEL = {p: 1 / 3 for p in one_qubit_paulis}


class DepolarizingErrorModel(ErrorModel):
    """Parameterized error mode."""

    def __init__(self, error_params: dict) -> None:
        super().__init__(error_params=error_params)
        self._eparams = None

    def reset(self):
        """Reset error generator for another round of syndrome extraction."""
        return DepolarizingErrorModel(error_params=self.error_params)

    def init(self, num_qubits, machine=None):
        self.machine = machine

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

    def _scale(self):
        # conversion from average error to total error
        self._eparams["p1"] *= 3 / 2
        self._eparams["p2"] *= 5 / 4

        scale = self._eparams.get("scale", 1.0)
        self._eparams["p1"] *= scale
        self._eparams["p2"] *= scale

        if isinstance(self._eparams["p_meas"], tuple):
            self._eparams["p_meas"] = np.mean(self._eparams["p_meas"])

        self._eparams["p_meas"] *= scale
        self._eparams["p_init"] *= scale

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""

    def process(self, qops: list[QOp], call_back=None) -> list[QOp | SeqBlock]:
        noisy_ops = []

        for op in qops:
            qops_after = None
            qops_before = None
            erroneous_ops = None

            # ########################################
            # INITS WITH X NOISE
            if op.name in ["init |0>", "Init", "Init +Z"]:
                qops_after = noise_initz_bitflip(
                    op,
                    p=self._eparams["p_init"],
                )

            # ########################################
            # ONE QUBIT GATES
            elif op.name in one_qubits:
                erroneous_ops = noise_sq_depolarizing(
                    op,
                    p=self._eparams["p1"],
                    noise_dict=self._eparams["p1_error_model"],
                )

            # ########################################
            # TWO QUBIT GATES
            elif op.name in two_qubits:
                qops_after = noise_tq_depolarizing(
                    op,
                    p=self._eparams["p2"],
                    noise_dict=self._eparams["p2_error_model"],
                )

                if self._eparams.get("p2_mem"):
                    qops_mem = noise_tq_depolarizing(
                        op,
                        p=self._eparams["p2_mem"],
                        noise_dict=self._eparams["p2_mem_error_model"],
                    )

                    if qops_after:
                        qops_after = qops_after.extend(qops_mem)

            # ########################################
            # MEASURE X NOISE
            elif op.name in ["measure Z", "Measure", "Measure +Z"]:
                erroneous_ops = noise_meas_bitflip(
                    op,
                    p=self._eparams["p_meas"],
                )

            else:
                raise Exception("This error model doesn't handle gate: %s!" % op.name)

            if qops_before:
                noisy_ops.extend(qops_before)

            if erroneous_ops is None:
                noisy_ops.append(op)
            else:
                noisy_ops.extend(erroneous_ops)

            if qops_after:
                noisy_ops.extend(qops_after)

        return noisy_ops
