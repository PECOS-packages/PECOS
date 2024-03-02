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

from pypecos.simulators import sim_class_types
from pypecos.simulators.parent_sim_classes import Simulator
from pypecos.simulators.paulifaultprop import (
    PauliFaultProp,
)

# Pauli fault propagation sim
from pypecos.simulators.sparsesim import (
    SparseSim as pySparseSim,
)

# C++ version of SparseStabSim wrapper
try:
    from pypecos.simulators.cysparsesim import SparseSim
    from pypecos.simulators.cysparsesim import (
        SparseSim as cySparseSim,
    )
except ImportError:
    from pypecos.simulators.sparsesim import SparseSim

# Attempt to import optional ProjectQ package
try:
    import projectq

    from pypecos.simulators.projectq.state import ProjectQSim  # wrapper for ProjectQ sim
except ImportError:
    pass
