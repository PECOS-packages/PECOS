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
from pecos_rslib.simulators import SparseSim, SparseSimCpp

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
from pecos.simulators.sparsesim import (
    SparseSim as SparseSimPy,
)
from pecos.simulators.statevec import StateVec

# Attempt to import optional cuquantum and cupy packages
try:
    import cupy
    import cuquantum

    from pecos.simulators.custatevec.state import (
        CuStateVec,
    )

    # wrapper for cuQuantum's cuStateVec
    from pecos.simulators.mps_pytket import (
        MPS,
    )
except ImportError:
    CuStateVec = None
    MPS = None

__all__ = [
    "MPS",
    # Python simulators
    "CoinToss",
    "CuStateVec",
    "DefaultSimulator",
    "PauliFaultProp",
    "PauliProp",
    "QuestDensityMatrix",
    "QuestStateVec",
    "Qulacs",
    # Rust simulators
    "SparseSim",
    "SparseSimCpp",
    "SparseSimPy",
    "StateVec",
    # Submodules
    "sim_class_types",
]
