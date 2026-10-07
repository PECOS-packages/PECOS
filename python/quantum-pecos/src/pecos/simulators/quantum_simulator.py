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

import importlib
from typing import Any

from pecos.reps.pyphir.op_types import QOp
from pecos.simulators import StateVec
from pecos.simulators.sparsestab.state import SparseStabPy

JSONType = dict[str, Any] | list[Any] | str | int | float | bool | None


try:
    from pecos.simulators import MPS
except ImportError:
    MPS = None

try:
    from pecos.simulators import CuStateVec
except ImportError:
    CuStateVec = None


# Loaders resolve module attributes at use time and keep Rust CUDA imports lazy.
_BACKENDS = {
    # SparseStabPy draws measurement randomness from the engine-seeded pc.random.
    "stabilizer": (lambda: SparseStabPy, False, None),
    "state-vector": (lambda: StateVec, True, None),
    "StateVec": (lambda: StateVec, True, None),
    "MPS": (lambda: MPS, True, "pytket-cutensornet and its CUDA dependencies"),
    "mps": (lambda: MPS, True, "pytket-cutensornet and its CUDA dependencies"),
    # CuStateVec draws measurement randomness from the engine-seeded pc.random.
    "CuStateVec": (lambda: CuStateVec, False, "CuPy and cuQuantum >= 25.03"),
    "CudaStateVec": (
        lambda: importlib.import_module("pecos.simulators").CudaStateVec,
        True,
        "pecos-rslib-cuda",
    ),
    None: (lambda: SparseStabPy, False, None),
}


class QuantumSimulator:
    """General-purpose quantum simulator with multiple backend support.

    Accepted backends are "stabilizer", "state-vector", "StateVec", "MPS", "mps",
    "CuStateVec", "CudaStateVec", and None (SparseStabPy).
    """

    def __init__(self, backend: str | None = None, **params: JSONType) -> None:
        """Initialize the QuantumSimulator.

        Args:
        ----
            backend: One of "stabilizer", "state-vector", "StateVec", "MPS", "mps",
                "CuStateVec", "CudaStateVec", or None (the default, SparseStabPy).
            **params: Additional parameters passed to the underlying simulator backend.
                ``seed`` reaches only backends that own their RNG (StateVec, MPS,
                CudaStateVec). "stabilizer" and "CuStateVec" draw measurement
                randomness from PECOS's global RNG, which ``HybridEngine`` seeds; seed
                it with ``pecos.random.seed`` when using those backends directly.

        Raises:
            ValueError: If ``backend`` is not one of the accepted names.

        """
        if (backend is not None and not isinstance(backend, str)) or backend not in _BACKENDS:
            accepted = ", ".join(repr(name) for name in _BACKENDS)
            msg = f"Unknown simulator {backend!r}; accepted backends: {accepted}"
            raise ValueError(msg)

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

        loader, takes_seed, dependency = _BACKENDS[self.backend]
        simulator = loader()
        if simulator is None:
            msg = f"Backend {self.backend!r} is unavailable; requires {dependency}"
            raise ImportError(msg)

        params = self.qsim_params.copy()
        if not takes_seed:
            params.pop("seed", None)
        self.state = simulator(num_qubits=num_qubits, **params)

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
