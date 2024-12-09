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

# Rust version of stabilizer sim
from pecos_rslib import SparseSimRs
from pecos_rslib import SparseSimRs as SparseSim

from pecos.simulators import sim_class_types
from pecos.simulators.basic_sv.state import BasicSV  # Basic numpy statevector simulator
from pecos.simulators.cointoss import (
    CoinToss,
)

# Ignores quantum gates, coin toss for measurements
from pecos.simulators.parent_sim_classes import Simulator
from pecos.simulators.paulifaultprop import (
    PauliFaultProp,
)

# Pauli fault propagation sim
from pecos.simulators.sparsesim import (
    SparseSim as SparseSimPy,
)

# C++ version of SparseStabSim wrapper
try:
    from pecos.simulators.cysparsesim import (
        SparseSim as SparseSimCy,
    )
except ImportError:
    SparseSimCy = None

# Attempt to import optional ProjectQ package
try:
    import projectq

    from pecos.simulators.projectq.state import ProjectQSim  # wrapper for ProjectQ sim
except ImportError:
    ProjectQSim = None

# Attempt to import optional Qulacs package
try:
    from pecos.simulators.qulacs.state import Qulacs  # wrapper for Qulacs sim
except ImportError:
    Qulacs = None

# Attempt to import optional QuEST package
try:
    from pecos.simulators.quest.state import QuEST  # wrapper for QuEST sim
except ImportError:
    QuEST = None

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
