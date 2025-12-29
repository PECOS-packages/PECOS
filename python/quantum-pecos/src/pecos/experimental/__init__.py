# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Experimental features for PECOS.

This module contains experimental features that are under active development.
APIs in this module may change without notice.

Currently available:
- Symbolic HUGR execution for efficient sampling of Clifford circuits
- Noisy symbolic execution with depolarizing noise model
"""

from pecos_rslib.experimental import (
    NoisySymbolicExecutionResult,
    SymbolicExecutionResult,
    execute_dag_circuit_symbolic,
    execute_dag_circuit_symbolic_noisy,
    execute_hugr_symbolic,
    execute_hugr_symbolic_noisy,
)

__all__ = [
    "NoisySymbolicExecutionResult",
    "SymbolicExecutionResult",
    "execute_dag_circuit_symbolic",
    "execute_dag_circuit_symbolic_noisy",
    "execute_hugr_symbolic",
    "execute_hugr_symbolic_noisy",
]
