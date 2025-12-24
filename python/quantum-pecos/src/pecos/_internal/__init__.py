# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Internal utilities for PECOS.

This package contains internal implementation details that are not part of
the public API. These modules should not be imported directly by users.

**WARNING**: The contents of this package are subject to change without notice.
Do not rely on any APIs defined here.

Internal modules (re-exports from misc for backwards compatibility):
    - commute: Commutation relation utilities
    - stabilizer_funcs: Stabilizer manipulation functions
    - threshold_curve: Threshold curve fitting
    - symbol_library: Gate symbol library definitions
    - std_output: Standard output formatting
    - gate_groups: Gate classification utilities
"""

# Re-export internal modules from misc for backwards compatibility
# These are internal APIs and subject to change
from pecos.misc import commute
from pecos.misc.gate_groups import (
    error_one_paulis_collection,
    error_two_paulis_collection,
    one_qubits,
    two_qubits,
)
from pecos.misc.stabilizer_funcs import (
    circ2set,
    find_stab,
    op_commutes,
    remove_stab,
)
from pecos.misc.std_output import StdOutput
from pecos.misc.symbol_library import SymbolLibrary
from pecos.misc.threshold_curve import func as threshold_func
from pecos.misc.threshold_curve import threshold_fit

__all__ = [
    # Output
    "StdOutput",
    # Symbol library
    "SymbolLibrary",
    # Stabilizer functions
    "circ2set",
    # Commute module
    "commute",
    # Gate groups
    "error_one_paulis_collection",
    "error_two_paulis_collection",
    "find_stab",
    "one_qubits",
    "op_commutes",
    "remove_stab",
    # Threshold fitting
    "threshold_fit",
    "threshold_func",
    "two_qubits",
]
