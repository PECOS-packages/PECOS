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

"""PECOS Quest plugin for Selene."""

import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from selene_core import Simulator


class SimulatorMode(Enum):
    """Simulator mode for Quest plugin.

    Attributes:
    ----------
    STATE_VECTOR
        State vector simulation. Memory scales as 16 bytes * 2^n_qubits.
    DENSITY_MATRIX
        Density matrix simulation. Memory scales as 16 bytes * 4^n_qubits.
        Required for simulating mixed states and certain noise models.
    """

    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"


@dataclass
class QuestPlugin(Simulator):
    """PECOS Quest simulator plugin for Selene.

    This plugin provides a Quest simulator backend for Selene, using the PECOS
    Quest wrapper. Quest is a high-performance quantum simulator that supports
    arbitrary rotation angles and can utilize GPU acceleration.

    Parameters
    ----------
    mode : SimulatorMode, default SimulatorMode.STATE_VECTOR
        The simulation mode to use. STATE_VECTOR for pure state simulation,
        DENSITY_MATRIX for mixed state simulation.
    use_gpu : bool, default False
        Whether to use GPU acceleration. Requires the library to be compiled
        with GPU support and a compatible CUDA GPU to be available.
    random_seed : int, optional
        Seed for the random number generator. If not provided, the seed
        will be determined by Selene's shot management.

    Examples:
    --------
    Basic state vector simulation (default):

    >>> plugin = QuestPlugin()

    Density matrix simulation:

    >>> plugin = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX)

    GPU-accelerated state vector simulation:

    >>> plugin = QuestPlugin(use_gpu=True)

    GPU-accelerated density matrix simulation:

    >>> plugin = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX, use_gpu=True)
    """

    mode: SimulatorMode = SimulatorMode.STATE_VECTOR
    use_gpu: bool = False
    random_seed: int | None = None

    def get_init_args(self) -> list[str]:
        """Return the initialization arguments for the Rust plugin.

        Returns:
        -------
        list[str]
            List of command-line style arguments for the Rust plugin.
        """
        args = [f"--mode={self.mode.value}"]
        if self.use_gpu:
            args.append("--use-gpu")
        return args

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
            patterns = ["libpecos_selene_quest*.dylib"]
        elif system == "windows":
            patterns = ["pecos_selene_quest*.dll", "pecos_selene_quest*.pyd"]
        else:  # Linux and others
            patterns = ["libpecos_selene_quest*.so"]

        for pattern in patterns:
            matches = list(libdir.glob(pattern))
            if matches:
                return matches[0]

        msg = f"Could not find PECOS Quest library in {libdir}"
        raise FileNotFoundError(msg)
