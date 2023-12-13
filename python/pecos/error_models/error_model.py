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

from typing import Callable

from pecos.error_models.error_model_abc import ErrorModel
from pecos.reps.pypmir.op_types import EMOp, MOp, QOp


class NoErrorModel(ErrorModel):
    """Represents having no error model."""

    def __init__(self):
        super().__init__(error_params={})

    def reset(self) -> None:
        """Reset state to initialization state."""

    def init(self, num_qubits, machine=None):
        super().init(num_qubits=num_qubits, machine=machine)
        if self.error_params:
            msg = "No error model is being utilized but error parameters are being provided!"
            raise Exception(msg)

    def shot_reinit(self):
        pass

    def process(self, ops: list, call_back: Callable | None = None) -> list | None:
        noisy_ops = []
        for op in ops:
            if isinstance(op, QOp):
                noisy_ops.append(op)
            elif isinstance(op, (MOp, EMOp)):
                pass
            else:
                msg = f"Operation type '{type(op)}' is not supported!"
                raise NotImplementedError(msg)
        return noisy_ops
