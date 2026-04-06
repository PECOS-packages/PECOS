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

# Rust simulators (direct exports without Python wrappers)
# Simulator engine builder factory functions
from pecos_rslib import clifford_rz, coin_toss, density_matrix, sparse_stab, stabilizer, state_vector
from pecos_rslib.simulators import CliffordRz, SparseStab, Stabilizer

from pecos.simulators import sim_class_types

# Coin toss simulator (uses Rust backend)
from pecos.simulators.cointoss import CoinToss

# Ignores quantum gates, coin toss for measurements
from pecos.simulators.default_simulator import DefaultSimulator
from pecos.simulators.pauliprop import (
    PauliFaultProp,  # Backward compatibility
    PauliProp,
)
from pecos.simulators.quest_densitymatrix import QuestDensityMatrix

# QuEST simulators
from pecos.simulators.quest_statevec import QuestStateVec

# Use Qulacs (Rust version) as the primary Qulacs implementation
from pecos.simulators.qulacs import Qulacs

# Pauli fault propagation sim
from pecos.simulators.sparsestab import (
    SparseStabPy as SparseStabPy,
)
from pecos.simulators.statevec import StateVec

# Attempt to import optional cuquantum and cupy packages (Python cuQuantum bindings)
try:
    import cupy
    import cuquantum

    from pecos.simulators.custatevec.state import (
        CuStateVec,
    )
except ImportError:
    CuStateVec = None

# Attempt to import optional pytket-cutensornet for MPS simulator
try:
    from pecos.simulators.mps_pytket import (
        MPS,
    )
except ImportError:
    MPS = None

# Rust cuQuantum bindings (pecos-rslib-cuda)
# Import always succeeds if the package is installed -- GPU availability is
# checked at construction time, not import time. This lets users reference
# the classes and get clear error messages when they try to use them.
try:
    from pecos.simulators.cuda_stabilizer import CudaStabilizer
    from pecos.simulators.cuda_statevec import CudaStateVec
except ImportError:
    CudaStateVec = None
    CudaStabilizer = None

__all__ = [
    "MPS",
    # Rust simulators
    "CliffordRz",
    # Python simulators
    "CoinToss",
    "CuStateVec",
    "CudaStabilizer",
    # CUDA simulators (Rust cuQuantum bindings)
    "CudaStateVec",
    "DefaultSimulator",
    "PauliFaultProp",
    "PauliProp",
    "QuestDensityMatrix",
    "QuestStateVec",
    "Qulacs",
    "SparseStab",
    "SparseStabPy",
    "Stabilizer",
    "StateVec",
    # Factory functions
    "clifford_rz",
    "coin_toss",
    "density_matrix",
    # Submodules
    "sim_class_types",
    "sparse_stab",
    "stabilizer",
    "state_vector",
]
