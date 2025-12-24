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

"""PECOS StateVec plugin for Selene."""

import platform
from dataclasses import dataclass
from pathlib import Path

from selene_core import Simulator


@dataclass
class StateVecPlugin(Simulator):
    """PECOS StateVec simulator plugin for Selene.

    This plugin provides a state vector simulator backend for Selene. Unlike
    stabilizer simulators, it can simulate arbitrary rotation angles, making
    it suitable for any quantum circuit.

    The memory requirement scales exponentially with the number of qubits
    (16 bytes * 2^n_qubits).

    Parameters
    ----------
    random_seed : int, optional
        Seed for the random number generator. If not provided, the seed
        will be determined by Selene's shot management.
    """

    random_seed: int | None = None

    def get_init_args(self) -> list[str]:
        """Return the initialization arguments for the Rust plugin.

        Returns:
        -------
        list[str]
            Empty list as StateVec plugin doesn't require additional arguments.
        """
        return []

    @property
    def library_file(self) -> Path:
        """Return the path to the compiled Rust library.

        Returns:
        -------
        Path
            Path to the shared library file.

        Raises:
        ------
        FileNotFoundError
            If no matching library file is found.
        """
        libdir = Path(__file__).parent / "_dist" / "lib"

        # Platform-specific library naming
        system = platform.system().lower()
        if system == "darwin":
            patterns = ["libpecos_selene_statevec*.dylib"]
        elif system == "windows":
            patterns = ["pecos_selene_statevec*.dll", "pecos_selene_statevec*.pyd"]
        else:  # Linux and others
            patterns = ["libpecos_selene_statevec*.so"]

        for pattern in patterns:
            matches = list(libdir.glob(pattern))
            if matches:
                return matches[0]

        msg = f"Could not find PECOS StateVec library in {libdir}"
        raise FileNotFoundError(msg)
