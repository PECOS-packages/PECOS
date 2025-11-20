"""Simple depolarizing error model implementation.

This module provides a simplified depolarizing error model for basic
quantum error correction simulations with uniform depolarizing noise
applied to quantum gates.
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
from pecos.error_models.noise_impl_old.gate_groups import one_qubits, two_qubits
from pecos.reps.pyphir.op_types import QOp

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos.protocols import MachineProtocol
    from pecos.reps.pyphir.block_types import SeqBlock

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


class SimpleDepolarizingErrorModel:
    """Parameterized error mode."""

    def __init__(self, error_params: dict) -> None:
        """Initialize a simple depolarizing error model.

        Args:
            error_params: Dictionary containing error parameters including:
                - p1: Single-qubit gate error probability
                - p2: Two-qubit gate error probability
                - p_meas: Measurement error probability
                - p_init: Initialization error probability
        """
        self.error_params = dict(error_params)
        self.machine = None
        self.num_qubits = None
        self._eparams = None

    def reset(self) -> SimpleDepolarizingErrorModel:
        """Reset error generator for another round of syndrome extraction."""
        return SimpleDepolarizingErrorModel(error_params=self.error_params)

    def init(self, num_qubits: int, machine: MachineProtocol | None = None) -> None:
        """Initialize the simple depolarizing error model.

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

    def _scale(self) -> None:
        # conversion from average error to total error
        self._eparams["p1"] *= 3 / 2
        self._eparams["p2"] *= 5 / 4

        if isinstance(self._eparams["p_meas"], tuple):
            self._eparams["p_meas"] = pc.mean(self._eparams["p_meas"])

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""

    def process(
        self,
        qops: list[QOp],
        _call_back: Callable[..., None] | None = None,
    ) -> list[QOp | SeqBlock]:
        """Process quantum operations and apply simple depolarizing errors.

        Args:
            qops: List of quantum operations to process.
            call_back: Optional callback function for additional processing.

        Returns:
            List of quantum operations with applied errors.
        """
        noisy_ops = []

        for op in qops:
            erroneous_ops = None

            # ########################################
            # INITS WITH X NOISE
            if op.name in {"init |0>", "Init", "Init +Z"}:
                erroneous_ops = [op]
                # Use fused operation to check and get error indices in one pass
                error_indices = pc.random.compare_indices(
                    len(op.args),
                    self._eparams["p_init"],
                )

                for idx in error_indices:
                    erroneous_ops.append(
                        QOp(name="X", args=[op.args[idx]], metadata={}),
                    )

            # ########################################
            # ONE QUBIT GATES
            if op.name in one_qubits:
                erroneous_ops = [op]
                # Use fused operation to check and get error indices in one pass
                error_indices = pc.random.compare_indices(
                    len(op.args),
                    self._eparams["p1"],
                )

                for idx in error_indices:
                    err = pc.random.choice(one_qubit_paulis, 1)[0]
                    erroneous_ops.append(
                        QOp(name=err[0], args=[op.args[idx]], metadata={}),
                    )

            # ########################################
            # TWO QUBIT GATES
            elif op.name in two_qubits:
                erroneous_ops = [op]
                # Use fused operation to check and get error indices in one pass
                error_indices = pc.random.compare_indices(
                    len(op.args),
                    self._eparams["p2"],
                )

                for idx in error_indices:
                    err = pc.random.choice(two_qubit_paulis, 1)[0]
                    loc1, loc2 = op.args[idx]
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
            elif op.name in {"measure Z", "Measure", "Measure +Z"}:
                erroneous_ops = []
                # Use fused operation to check and get error indices in one pass
                error_indices = pc.random.compare_indices(
                    len(op.args),
                    self._eparams["p_meas"],
                )

                for idx in error_indices:
                    erroneous_ops.append(
                        QOp(name="X", args=[op.args[idx]], metadata={}),
                    )

                erroneous_ops.append(op)

            else:
                msg = f"This error model doesn't handle gate: {op.name}!"
                raise Exception(msg)

            if erroneous_ops is None:
                noisy_ops.append(op)
            else:
                noisy_ops.extend(erroneous_ops)

        return noisy_ops
