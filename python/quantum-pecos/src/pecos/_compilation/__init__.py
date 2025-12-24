# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Internal compilation utilities for PECOS.

This module contains internal utilities for compiling quantum programs from
various formats (Guppy, HUGR) to executable formats (LLVM/QIS).

These are implementation details and should not be imported directly by users.
"""

from pecos._compilation.guppy import GuppyFrontend, compile_guppy_to_qir, guppy_to_hugr
from pecos._compilation.hugr_llvm import HugrLlvmCompiler, compile_hugr_bytes_to_llvm

__all__ = [
    "GuppyFrontend",
    "HugrLlvmCompiler",
    "compile_guppy_to_qir",
    "compile_hugr_bytes_to_llvm",
    "guppy_to_hugr",
]
