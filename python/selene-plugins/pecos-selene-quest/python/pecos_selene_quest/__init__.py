# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""PECOS Quest Selene plugin.

This plugin provides a Selene-compatible interface to the QuEST (Quantum Exact
Simulation Toolkit) simulator through the PECOS wrapper.

QuEST is developed by the QuEST-Kit team and is available at:
https://github.com/quest-kit/QuEST

QuEST is licensed under the MIT License.
"""

import os
import platform
from pathlib import Path


# Set the QuEST CUDA backend path environment variable if the backend library exists.
# This allows the Rust library to find and load the CUDA-accelerated QuEST backend
# at runtime via dlopen when CUDA acceleration is requested.
def _setup_cuda_library_path() -> None:
    """Configure the QuEST CUDA backend library path for runtime loading."""
    # Only set if not already configured by the user
    if "PECOS_QUEST_CUDA_LIB" in os.environ:
        return

    # Determine the QuEST CUDA backend filename based on platform
    system = platform.system()
    if system == "Linux":
        cuda_backend_name = "libpecos_quest_cuda.so"
    elif system == "Darwin":
        cuda_backend_name = "libpecos_quest_cuda.dylib"
    elif system == "Windows":
        cuda_backend_name = "pecos_quest_cuda.dll"
    else:
        return  # Unknown platform

    # Look for the QuEST CUDA backend in the package's _dist/lib directory
    package_dir = Path(__file__).parent
    cuda_backend_path = package_dir / "_dist" / "lib" / cuda_backend_name

    if cuda_backend_path.exists():
        os.environ["PECOS_QUEST_CUDA_LIB"] = str(cuda_backend_path)


_setup_cuda_library_path()

# Import after setting up CUDA path - the Rust library reads the env var at load time
from pecos_selene_quest.plugin import QuestPlugin, SimulatorMode  # noqa: E402

__all__ = ["QuestPlugin", "SimulatorMode"]
