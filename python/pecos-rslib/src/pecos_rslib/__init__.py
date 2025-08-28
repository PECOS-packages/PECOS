# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""PECOS Rust library Python bindings.

This package provides Python bindings for high-performance Rust implementations of quantum simulators and computational
components within the PECOS framework, enabling efficient quantum circuit simulation and error correction computations.
"""

# ruff: noqa: TID252
from importlib.metadata import PackageNotFoundError, version

from pecos_rslib.rssparse_sim import SparseSimRs
from pecos_rslib.cppsparse_sim import CppSparseSimRs
from pecos_rslib.rsstate_vec import StateVecRs
from pecos_rslib._pecos_rslib import ByteMessage
from pecos_rslib._pecos_rslib import ByteMessageBuilder
from pecos_rslib._pecos_rslib import StateVecEngineRs
from pecos_rslib._pecos_rslib import SparseStabEngineRs

# QASM simulation exports
from pecos_rslib._pecos_rslib import NoiseModel
from pecos_rslib._pecos_rslib import QuantumEngine
from pecos_rslib._pecos_rslib import run_qasm
from pecos_rslib._pecos_rslib import get_noise_models
from pecos_rslib._pecos_rslib import get_quantum_engines
from pecos_rslib._pecos_rslib import GeneralNoiseModelBuilder

# Import the qasm_sim function for easy access
from pecos_rslib.qasm_sim import qasm_sim

# Also import the noise model dataclasses for convenience
from pecos_rslib.qasm_sim import (
    PassThroughNoise,
    DepolarizingNoise,
    DepolarizingCustomNoise,
    BiasedDepolarizingNoise,
    GeneralNoise,
)

# Import GeneralNoiseFactory and convenience functions
from pecos_rslib.general_noise_factory import (
    GeneralNoiseFactory,
    create_noise_from_dict,
    create_noise_from_json,
    IonTrapNoiseFactory,
)

try:
    __version__ = version("pecos-rslib")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "SparseSimRs",
    "CppSparseSimRs",
    "StateVecRs",
    "ByteMessage",
    "ByteMessageBuilder",
    "StateVecEngineRs",
    "SparseStabEngineRs",
    # QASM simulation
    "NoiseModel",
    "QuantumEngine",
    "run_qasm",
    "get_noise_models",
    "get_quantum_engines",
    "qasm_sim",
    "GeneralNoiseModelBuilder",
    # Noise model dataclasses
    "PassThroughNoise",
    "DepolarizingNoise",
    "DepolarizingCustomNoise",
    "BiasedDepolarizingNoise",
    "GeneralNoise",
    # Noise factory
    "GeneralNoiseFactory",
    "create_noise_from_dict",
    "create_noise_from_json",
    "IonTrapNoiseFactory",
]
