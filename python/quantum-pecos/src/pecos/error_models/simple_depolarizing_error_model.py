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
from pecos.error_models.noise_impl_old.gate_groups import one_qubits, two_qubits
from pecos.reps.pypmir.op_types import QOp

if TYPE_CHECKING:
    from pecos.reps.pypmir.block_types import SeqBlock

one_qubit_paulis = ["X", "Y", "Z"]

two_qubit_paulis = [
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
]


class SimpleDepolarizingErrorModel(ErrorModel):
    """Parameterized error mode."""

    def __init__(self, error_params: dict) -> None:
        super().__init__(error_params=error_params)
        self._eparams = None

    def reset(self):
        """Reset error generator for another round of syndrome extraction."""
        return SimpleDepolarizingErrorModel(error_params=self.error_params)

    def init(self, num_qubits, machine=None):
        self.machine = machine

        if not self.error_params:
            msg = "Error params not set!"
            raise Exception(msg)

        self._eparams = dict(self.error_params)
        self._scale()

    def _scale(self):
        # conversion from average error to total error
        self._eparams["p1"] *= 3 / 2
        self._eparams["p2"] *= 5 / 4

        if isinstance(self._eparams["p_meas"], tuple):
            self._eparams["p_meas"] = np.mean(self._eparams["p_meas"])

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""

    def process(self, qops: list[QOp], call_back=None) -> list[QOp | SeqBlock]:
        noisy_ops = []

        for op in qops:
            erroneous_ops = None

            # ########################################
            # INITS WITH X NOISE
            if op.name in ["init |0>", "Init", "Init +Z"]:
                erroneous_ops = [op]
                rand_nums = np.random.random(len(op.args)) <= self._eparams["p_init"]

                if np.any(rand_nums):
                    for r, loc in zip(rand_nums, op.args):
                        if r:
                            erroneous_ops.append(QOp(name="X", args=[loc], metadata={}))

            # ########################################
            # ONE QUBIT GATES
            if op.name in one_qubits:
                erroneous_ops = [op]
                rand_nums = np.random.random(len(op.args)) <= self._eparams["p1"]

                if np.any(rand_nums):
                    for r, loc in zip(rand_nums, op.args):
                        if r:
                            err = np.random.choice(one_qubit_paulis)
                            erroneous_ops.append(
                                QOp(name=err[0], args=[loc], metadata={}),
                            )

            # ########################################
            # TWO QUBIT GATES
            elif op.name in two_qubits:
                erroneous_ops = [op]
                rand_nums = np.random.random(len(op.args)) <= self._eparams["p2"]

                if np.any(rand_nums):
                    for r, loc in zip(rand_nums, op.args):
                        if r:
                            err = np.random.choice(two_qubit_paulis)
                            loc1, loc2 = loc
                            if err[0] != "I":
                                erroneous_ops.append(
                                    QOp(name=err[0], args=[loc1], metadata={}),
                                )
                            if err[1] != "I":
                                erroneous_ops.append(
                                    QOp(name=err[1], args=[loc2], metadata={}),
                                )

            # ########################################
            # MEASURE X NOISE
            elif op.name in ["measure Z", "Measure", "Measure +Z"]:
                erroneous_ops = []
                rand_nums = np.random.random(len(op.args)) <= self._eparams["p_meas"]

                if np.any(rand_nums):
                    for r, loc in zip(rand_nums, op.args):
                        if r:
                            erroneous_ops.append(QOp(name="X", args=[loc], metadata={}))

                erroneous_ops.append(op)

            else:
                raise Exception("This error model doesn't handle gate: %s!" % op.name)

            if erroneous_ops is None:
                noisy_ops.append(op)
            else:
                noisy_ops.extend(erroneous_ops)

        return noisy_ops
