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

"""Base instruction types for PyPHIR intermediate representation.

This module defines the fundamental instruction base classes for PyPHIR (Python PECOS Medium-level Intermediate
Representation) used in quantum circuit compilation and execution.
"""

from __future__ import annotations


class Instr:
    """Base type for all PyMIR instructions including QOps, Blocks, MOps, etc.

    Attributes:
        metadata: Always-present metadata dictionary.
    """

    def __init__(self, metadata: dict | None = None) -> None:
        """Initialize an instruction.

        Args:
            metadata: Metadata dictionary stored by identity. None creates a fresh empty dict.

        Raises:
            TypeError: If metadata is neither a dict nor None.
        """
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            msg = f"Metadata must be a dict or None, got {type(metadata).__name__}"
            raise TypeError(msg)
        self.metadata: dict = metadata
