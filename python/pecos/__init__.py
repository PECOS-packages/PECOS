# Copyright 2018 The PECOS Developers
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

"""Performance Estimator of Codes On Surfaces (PECOS)
==================================================.

A framework for developing, studying, and evaluating quantum error-correcting codes.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quantum-pecos")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Allow for other namespace packages
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# PECOS namespaces
from pecos import circuit_converters, circuits, decoders, engines, error_models, misc, qeccs, simulators, tools
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.engines import circuit_runners
from pecos.engines.cvm.binarray import BinArray
from pecos.engines.hybrid_engine_old import HybridEngine

__copyright__ = (
    "Copyright 2018-2013 The PECOS Developers. Copyright 2018 National Technology & Engineering Solutions of Sandia, "
    "LLC (NTESS)."
)
__license__ = "Apache-2.0"
__author__ = "The PECOS Developers"
__maintainer__ = "Ciaran Ryan-Anderson"
__email__ = "ciaran@pecos.io"

__all__ = [
    "circuits",
    "qeccs",
    "simulators",
    "error_models",
    "circuit_runners",
    "engines",
    "decoders",
    "circuit_converters",
    "misc",
    "tools",
    "QuantumCircuit",
    "BinArray",
    "HybridEngine",
]
