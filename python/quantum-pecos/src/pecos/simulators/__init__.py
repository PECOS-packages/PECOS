"""Quantum simulators for PECOS.

This package provides various quantum simulators including state vector, sparse stabilizer,
and fault propagation simulators.
"""

# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from importlib import import_module

# Rust simulators (direct exports without Python wrappers)
# Simulator engine builder factory functions
from pecos_rslib import (
    coin_toss,
    density_matrix,
    sparse_stab,
    stab_vec,
    stabilizer,
    state_vector,
)
from pecos_rslib.simulators import (
    DensityMatrixDiagnostics,
    InstrumentSample,
    KrausSample,
    MeasurementSample,
    QuditDensityMatrix,
    QuditStateVec,
    QutritDensityMatrix,
    QutritStateVec,
    SparseStab,
    Stabilizer,
    StabVec,
    basis_swap,
    embedded_qubit_unitary,
    qutrit_leakage_channel,
    qutrit_seepage_channel,
)

from pecos.simulators import sim_class_types
from pecos.simulators.cointoss import CoinToss
from pecos.simulators.default_simulator import DefaultSimulator
from pecos.simulators.pauliprop import (
    PauliFaultProp,  # Backward compatibility
    PauliProp,
)
from pecos.simulators.sparsestab import (
    SparseStabPy as SparseStabPy,
)
from pecos.simulators.statevec import StateVec

# Python cuQuantum CuStateVec backend. Import always succeeds if the package is
# present; CuPy / cuQuantum availability is checked at construction time (like the
# Rust CudaStateVec below), so users get a clear error only when they use it.
try:
    from pecos.simulators.custatevec.state import CuStateVec
except ImportError:
    CuStateVec = None

# Attempt to import optional pytket-cutensornet for MPS simulator
try:
    from pecos.simulators.mps_pytket import (
        MPS,
    )
except ImportError:
    MPS = None


# Rust cuQuantum bindings (pecos-rslib-cuda). Resolve these public classes only
# when requested; GPU availability is still checked at construction time.
def _load_cuda_simulators() -> None:
    """Populate the public Rust CUDA simulator classes on first access."""
    try:
        cuda_stabilizer = import_module("pecos.simulators.cuda_stabilizer")
        cuda_statevec = import_module("pecos.simulators.cuda_statevec")
        cuda_stabilizer_class = cuda_stabilizer.CudaStabilizer
        cuda_statevec_class = cuda_statevec.CudaStateVec
    except ImportError:
        cuda_statevec_class = None
        cuda_stabilizer_class = None

    globals().update(CudaStabilizer=cuda_stabilizer_class, CudaStateVec=cuda_statevec_class)


def __getattr__(name: str) -> object:
    if name in {"CudaStabilizer", "CudaStateVec"}:
        _load_cuda_simulators()
        return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted({*globals(), "CudaStabilizer", "CudaStateVec"})


__all__ = [
    "MPS",
    "CoinToss",
    "CuStateVec",
    "CudaStabilizer",
    "CudaStateVec",
    "DefaultSimulator",
    "DensityMatrixDiagnostics",
    "InstrumentSample",
    "KrausSample",
    "MeasurementSample",
    "PauliFaultProp",
    "PauliProp",
    "QuditDensityMatrix",
    "QuditStateVec",
    "QutritDensityMatrix",
    "QutritStateVec",
    "SparseStab",
    "SparseStabPy",
    "StabVec",
    "Stabilizer",
    "StateVec",
    "basis_swap",
    "coin_toss",
    "density_matrix",
    "embedded_qubit_unitary",
    "qutrit_leakage_channel",
    "qutrit_seepage_channel",
    "sim_class_types",
    "sparse_stab",
    "stab_vec",
    "stabilizer",
    "state_vector",
]
