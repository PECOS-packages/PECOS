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

"""PECOS SparseStab simulator plugin for Selene."""

import platform
from dataclasses import dataclass
from pathlib import Path

from selene_core import Simulator


@dataclass
class SparseStabPlugin(Simulator):
    """A plugin for using the PECOS SparseStab stabilizer simulator as a backend for Selene.

    PECOS SparseStab is a sparse stabilizer simulator that can efficiently simulate
    Clifford circuits. As a stabilizer simulator, it can only simulate Clifford operations
    (rotations that are multiples of pi/2).

    Attributes:
        angle_threshold: The threshold for angle approximation. Angles within this threshold
            of a multiple of pi/2 will be rounded to that multiple. Must be greater than zero
            to avoid numerical instability. Default is 1e-4.
    """

    angle_threshold: float = 1e-4

    def __post_init__(self) -> None:
        """Validate plugin configuration."""
        if self.angle_threshold <= 0:
            msg = "angle_threshold must be greater than zero to avoid numerical instability"
            raise ValueError(msg)

    def get_init_args(self) -> list[str]:
        """Return initialization arguments to pass to the Rust plugin."""
        return [
            f"--angle-threshold={self.angle_threshold}",
        ]

    @property
    def library_file(self) -> Path:
        """Return the path to the compiled shared library for the current platform."""
        libdir = Path(__file__).parent / "_dist" / "lib"
        system = platform.system()
        if system == "Linux":
            return libdir / "libpecos_selene_sparsestab.so"
        if system == "Darwin":
            return libdir / "libpecos_selene_sparsestab.dylib"
        if system == "Windows":
            return libdir / "pecos_selene_sparsestab.dll"
        msg = f"Unsupported platform: {system}"
        raise RuntimeError(msg)
