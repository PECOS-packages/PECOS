#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from ._sparsesim import State as pySparseSim

# C++ version of SparseStabSim wrapper
try:
    from .cysparsesim import State as SparseSim
    from .cysparsesim import State as cySparseSim
except ImportError:
    from ._sparsesim import State as SparseSim

StabSim = SparseSim  # Default stabilizer simulator

# Attempt to import optional ProjectQ interface
try:
    import projectq
    from ._projectq_wrapper.state import State as ProjectQSim
except ImportError:
    pass

# Attempt to import optional qcgpu interface
try:
    import qcgpu
    from ._qcgpu_wrapper.state import State as QCGPU
except ImportError:
    pass
