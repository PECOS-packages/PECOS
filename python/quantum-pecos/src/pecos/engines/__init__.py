"""Execution engines for PECOS.

This package provides various execution engines for quantum simulations.

Engine classes (from pecos_rslib.engines):
    - StateVecEngine: State vector execution engine
    - SparseStabEngine: Sparse stabilizer execution engine
    - PhirJsonEngine: PHIR JSON execution engine

Builder classes (from pecos_rslib.engines):
    - StateVectorEngineBuilder: Builder for state vector engines
    - SparseStabEngineBuilder: Builder for sparse stabilizer engines
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

# Re-export Rust engines from pecos_rslib
from pecos_rslib import (
    ByteMessage,
    ByteMessageBuilder,
    HugrEngineBuilder,
    HugrSimulation,
    PhirJsonSimulation,
    PhirSimulation,
    QasmSimulation,
    QisControlSimulation,
    QisInterfaceBuilder,
    SimBuilder,
    compile_hugr_to_qis,
    find_llvm_tool,
    get_compilation_backends,
    qis_helios_interface,
    qis_selene_helios_interface,
    sim_builder,
)
from pecos_rslib.engines import (
    PhirEngineBuilder,
    PhirJsonEngine,
    PhirJsonEngineBuilder,
    QasmEngineBuilder,
    QisEngineBuilder,
    SparseStabEngine,
    SparseStabEngineBuilder,
    StateVecEngine,
    StateVectorEngineBuilder,
    phir_engine,
    phir_json_engine,
    qasm_engine,
    qis_engine,
)

from pecos.engines.hybrid_engine import HybridEngine

__all__ = [
    "ByteMessage",
    "ByteMessageBuilder",
    "HugrEngineBuilder",
    "HugrSimulation",
    "HybridEngine",
    "PhirEngineBuilder",
    "PhirJsonEngine",
    "PhirJsonEngineBuilder",
    "PhirJsonSimulation",
    "PhirSimulation",
    "QasmEngineBuilder",
    "QasmSimulation",
    "QisControlSimulation",
    "QisEngineBuilder",
    "QisInterfaceBuilder",
    "SimBuilder",
    "SparseStabEngine",
    "SparseStabEngineBuilder",
    "StateVecEngine",
    "StateVectorEngineBuilder",
    "compile_hugr_to_qis",
    "find_llvm_tool",
    "get_compilation_backends",
    "phir_engine",
    "phir_json_engine",
    "qasm_engine",
    "qis_engine",
    "qis_helios_interface",
    "qis_selene_helios_interface",
    "sim_builder",
]
