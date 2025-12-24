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

"""Quantum simulator interface and implementation for PECOS.

This module provides a unified quantum simulator interface that can dispatch
quantum operations to different backend simulators including state vector
and sparse stabilizer implementations.
"""

from __future__ import annotations

from typing import Any

from pecos.reps.pyphir.op_types import QOp
from pecos.simulators import Qulacs, StateVec
from pecos.simulators.sparsesim.state import SparseSim

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None


try:
    from pecos.simulators import MPS
except ImportError:
    MPS = None

try:
    from pecos.simulators import Qulacs
except ImportError:
    Qulacs = None


try:
    from pecos.simulators import CuStateVec
except ImportError:
    CuStateVec = None

try:
    from pecos.simulators import QuestDensityMatrix, QuestStateVec
except ImportError:
    QuestStateVec = None
    QuestDensityMatrix = None


class QuantumSimulator:
    """General-purpose quantum simulator with multiple backend support.

    This class provides a unified interface for various quantum simulation backends
    including stabilizer simulators, state vector simulators, and specialized
    simulators like MPS, Qulacs, and cuQuantum.
    """

    def __init__(self, backend: str | object | None = None, **params: JSONType) -> None:
        """Initialize the QuantumSimulator.

        Args:
        ----
            backend: The simulation backend to use. Can be a string identifier
                (e.g., 'stabilizer', 'state-vector', 'MPS', 'Qulacs', 'CuStateVec')
                or a custom backend object. Defaults to None, which uses SparseSim.
            **params: Additional parameters passed to the underlying simulator backend.

        """
        self.num_qubits = None
        self.state = None
        self.backend = backend
        self.qsim_params = params

    def reset(self) -> None:
        """Reset the quantum simulator to its initial state."""
        self.num_qubits = None
        self.state = None

    def init(self, num_qubits: int) -> None:
        """Initialize the quantum simulator with specified number of qubits.

        Args:
            num_qubits: Number of qubits to initialize.
        """
        self.num_qubits = num_qubits

        if isinstance(self.backend, str):
            if self.backend == "stabilizer":
                self.state = SparseSim
            elif self.backend in "state-vector":
                if Qulacs is not None:
                    self.state = Qulacs
                else:
                    self.state = StateVec
            elif self.backend == "StateVec":
                self.state = StateVec
            elif self.backend in {"MPS", "mps"}:
                self.state = MPS
            elif self.backend == "Qulacs":
                self.state = Qulacs
            elif self.backend == "CuStateVec":
                self.state = CuStateVec
            elif self.backend == "QuestStateVec":
                self.state = QuestStateVec
            elif self.backend == "QuestDensityMatrix":
                self.state = QuestDensityMatrix
            else:
                msg = f"simulator `{self.backend}` not currently implemented!"
                raise NotImplementedError(msg)

        if self.backend is None:
            self.state = SparseSim

        # Try to initialize with all params including seed
        # If the simulator doesn't support seed, retry without it
        try:
            self.state = self.state(num_qubits=num_qubits, **self.qsim_params)
        except TypeError as e:
            if "seed" in str(e) and "unexpected keyword argument" in str(e):
                # Simulator doesn't support seed parameter, retry without it
                params_without_seed = {
                    k: v for k, v in self.qsim_params.items() if k != "seed"
                }
                self.state = self.state(num_qubits=num_qubits, **params_without_seed)
            else:
                # Different TypeError, re-raise it
                raise

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        self.state.reset()

    def run(self, qops: list[QOp]) -> list:
        """Run a list of quantum operations and return measurement results.

        Given a list of quantum operations, run them, update the state, and return any measurement results that
        are generated in the form {qid: result, ...}.
        """
        meas = []
        for op in qops:
            if op.metadata is None:
                op.metadata = {}
            if isinstance(op, QOp):
                output = self.state.run_gate(op.sim_name, op.args, **op.metadata)
                if op.returns:
                    temp = {}
                    bitflips = op.metadata.get("bitflips")
                    for q, r in zip(op.args, op.returns, strict=False):
                        out = output.get(q, 0)
                        if bitflips and q in bitflips:
                            out ^= 1

                        temp[tuple(r)] = out

                    meas.append(temp)
            else:
                msg = f"Quantum simulators process type QOp but got type {type(op)} from op: {op}"
                raise TypeError(msg)

        return meas
