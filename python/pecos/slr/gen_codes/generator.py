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


from __future__ import annotations

from abc import ABC, abstractmethod


class Generator(ABC):
    """An abstract class representing a code generator for an slr block."""

    @abstractmethod
    def __init__(self, includes: list[str] | None = None):
        pass

    @abstractmethod
    def generate_block(self, block):
        """Takes an slr block and generates a series of operations in the target format."""

    @abstractmethod
    def get_output(self) -> str:
        """Resolves an slr block into a string. For QIR this is only expected to be called
        on a block containing the entire program."""
