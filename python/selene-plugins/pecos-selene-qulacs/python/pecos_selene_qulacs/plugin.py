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

"""PECOS Qulacs plugin for Selene."""

import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from selene_core import Simulator


class SimulatorMode(Enum):
    """Simulator mode for Qulacs plugin.

    Currently only state vector simulation is supported.

    Attributes:
    ----------
    STATE_VECTOR
        State vector simulation. Memory scales as 16 bytes * 2^n_qubits.
    """

    STATE_VECTOR = "state_vector"


@dataclass
class QulacsPlugin(Simulator):
    """PECOS Qulacs simulator plugin for Selene.

    This plugin provides a Qulacs state vector simulator backend for Selene,
    using the PECOS Qulacs wrapper. Qulacs is a high-performance quantum simulator
    that supports arbitrary rotation angles.

    Parameters
    ----------
    mode : SimulatorMode, default SimulatorMode.STATE_VECTOR
        The simulation mode to use. Currently only STATE_VECTOR is supported.
    random_seed : int, optional
        Seed for the random number generator. If not provided, the seed
        will be determined by Selene's shot management.

    Examples:
    --------
    Basic state vector simulation (default):

    >>> plugin = QulacsPlugin()

    With explicit mode:

    >>> plugin = QulacsPlugin(mode=SimulatorMode.STATE_VECTOR)
    """

    mode: SimulatorMode = SimulatorMode.STATE_VECTOR
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate plugin configuration."""
        if self.mode != SimulatorMode.STATE_VECTOR:
            msg = f"Qulacs plugin only supports state_vector mode, got {self.mode.value}"
            raise ValueError(msg)

    def get_init_args(self) -> list[str]:
        """Return the initialization arguments for the Rust plugin.

        Returns:
        -------
        list[str]
            List of command-line style arguments for the Rust plugin.
        """
        return [f"--mode={self.mode.value}"]

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
            patterns = ["libpecos_selene_qulacs*.dylib"]
        elif system == "windows":
            patterns = ["pecos_selene_qulacs*.dll", "pecos_selene_qulacs*.pyd"]
        else:  # Linux and others
            patterns = ["libpecos_selene_qulacs*.so"]

        for pattern in patterns:
            matches = list(libdir.glob(pattern))
            if matches:
                return matches[0]

        msg = f"Could not find PECOS Qulacs library in {libdir}"
        raise FileNotFoundError(msg)
