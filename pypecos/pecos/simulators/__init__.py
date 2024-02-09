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

from pecos.simulators import sim_class_types
from pecos.simulators.parent_sim_classes import Simulator
from pecos.simulators.paulifaultprop import PauliFaultProp  # Pauli fault propagation sim
from pecos.simulators.sparsesim import SparseSim as pySparseSim  # Python sparse stabilizer sim

# C++ version of SparseStabSim wrapper
try:
    from pecos.simulators.cysparsesim import SparseSim
    from pecos.simulators.cysparsesim import SparseSim as cySparseSim  # Cython wrapped C++ sparse stabilizer sim
except ImportError:
    from pecos.simulators.sparsesim import SparseSim

# Attempt to import optional ProjectQ package
try:
    import projectq

    from pecos.simulators.projectq.state import ProjectQSim  # wrapper for ProjectQ sim
except ImportError:
    pass

# Attempt to import optional qcgpu package
try:
    import qcgpu

    from pecos.simulators._qcgpu_wrapper.state import QCQPUSim  # wrapper for qcgpu
except ImportError:
    pass
