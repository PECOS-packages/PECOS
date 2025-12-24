"""Execution engines for PECOS.

This package provides various execution engines for quantum simulations.

Engine classes (from pecos_rslib.engines):
    - StateVecEngine: State vector execution engine
    - SparseStabEngine: Sparse stabilizer execution engine
    - PhirJsonEngine: PHIR JSON execution engine

Builder classes (from pecos_rslib.engines):
    - StateVectorEngineBuilder: Builder for state vector engines
    - SparseStabilizerEngineBuilder: Builder for sparse stabilizer engines
    - QasmEngineBuilder: Builder for QASM engines (Rust version)
    - QisEngineBuilder: Builder for QIS engines (Rust version)
    - PhirJsonEngineBuilder: Builder for PHIR JSON engines (Rust version)

Factory functions (from pecos_rslib.engines):
    - qasm_engine(): Create a QASM engine builder
    - qis_engine(): Create a QIS engine builder
    - phir_json_engine(): Create a PHIR JSON engine builder

Note: For Python wrappers that accept pecos.programs types, use:
    - pecos.qasm_engine()
    - pecos.qis_engine()
    - pecos.phir_json_engine()

Note: Selene Bridge Plugin is now located in pecos.simulators.selene_bridge
"""

# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Re-export Rust engines from pecos_rslib.engines submodule
from pecos_rslib.engines import (
    # Engine classes
    PhirJsonEngine,
    # Builder classes
    PhirJsonEngineBuilder,
    QasmEngineBuilder,
    QisEngineBuilder,
    SparseStabEngine,
    SparseStabilizerEngineBuilder,
    StateVecEngine,
    StateVectorEngineBuilder,
    # Factory functions
    phir_json_engine,
    qasm_engine,
    qis_engine,
)

__all__ = [
    "PhirJsonEngine",
    "PhirJsonEngineBuilder",
    "QasmEngineBuilder",
    "QisEngineBuilder",
    "SparseStabEngine",
    "SparseStabilizerEngineBuilder",
    # Engine classes
    "StateVecEngine",
    # Builder classes
    "StateVectorEngineBuilder",
    "phir_json_engine",
    # Factory functions
    "qasm_engine",
    "qis_engine",
]
