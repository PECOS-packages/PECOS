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

from pecos.simulators.sparsesim.state import SparseSim

try:
    from pecos.simulators.projectq.state import ProjectQSim
except ImportError:
    ProjectQSim = None

try:
    from pecos.simulators import MPS
except ImportError:
    MPS = None

try:
    from pecos.simulators import Qulacs
except ImportError:
    Qulacs = None

try:
    from pecos.simulators import QuEST
except ImportError:
    QuEST = None

try:
    from pecos.simulators import CuStateVec
except ImportError:
    CuStateVec = None

from pecos.reps.pypmir.op_types import QOp


class QuantumSimulator:
    def __init__(self, backend: str | object | None = None, **params) -> None:
        self.num_qubits = None
        self.state = None
        self.backend = backend
        self.qsim_params = params

    def reset(self):
        self.num_qubits = None
        self.state = None

    def init(self, num_qubits: int):
        self.num_qubits = num_qubits

        if isinstance(self.backend, str):
            if self.backend == "stabilizer":
                self.state = SparseSim
            elif self.backend in ("state-vector", "ProjectQSim"):
                self.state = ProjectQSim
            elif self.backend in {"MPS", "mps"}:
                self.state = MPS
            elif self.backend == "Qulacs":
                self.state = Qulacs
            elif self.backend == "QuEST":
                self.state = QuEST
            elif self.backend == "CuStateVec":
                self.state = CuStateVec
            else:
                msg = f"simulator `{self.state}` not currently implemented!"
                raise NotImplementedError(msg)

        if self.backend is None:
            self.state = SparseSim

        self.state = self.state(num_qubits=num_qubits, **self.qsim_params)

    def shot_reinit(self):
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        self.state.reset()

    def run(self, qops: list[QOp]) -> list:
        """Given a list of quantum operations, run them, update the state, and return any measurement results that
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
                    for q, r in zip(op.args, op.returns):
                        out = output.get(q, 0)
                        if bitflips and q in bitflips:
                            out ^= 1

                        temp[tuple(r)] = out

                    meas.append(temp)
            else:
                msg = f"Quantum simulators process type QOp but got type {type(op)}"
                raise TypeError(msg)

        return meas
