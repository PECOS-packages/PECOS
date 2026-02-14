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

"""Circuit converters for transforming between different circuit representations.

This name space contains classes and functions for converting circuit data-structures
into other circuit data-structures.

Modules:
    checks2circuit: Convert check matrices to circuits.
    hugr_to_dag: Convert HUGR (from Guppy) to DAG (requires hugr package).
    hugr_to_ast: Convert HUGR (from Guppy) to SLR-AST (requires hugr package).
"""

from pecos.circuit_converters.checks2circuit import Check2Circuits

# hugr_to_dag and hugr_to_ast require the optional hugr package
try:
    from pecos.circuit_converters.hugr_to_dag import (
        UnsupportedHugrStructureError,
        dag_to_gate_sequence,
        guppy_to_dag,
        hugr_to_dag,
    )
    from pecos.circuit_converters.hugr_to_ast import (
        guppy_to_ast,
        hugr_to_ast,
    )

    _HAS_HUGR = True
except ImportError:
    _HAS_HUGR = False

__all__ = [
    "Check2Circuits",
]

if _HAS_HUGR:
    __all__ += [
        "UnsupportedHugrStructureError",
        "dag_to_gate_sequence",
        "guppy_to_dag",
        "hugr_to_dag",
        # HUGR to AST conversion
        "guppy_to_ast",
        "hugr_to_ast",
    ]
