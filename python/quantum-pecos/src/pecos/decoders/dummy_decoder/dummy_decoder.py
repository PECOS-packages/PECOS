# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""A dummy decoder that gives no recovery (outputs do nothing) given any input."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.misc.std_output import StdOutput


class DummyDecoder:
    """This decoder is just a simple look-up decoder."""

    def __init__(self) -> None:
        """Initialize the DummyDecoder.

        This decoder provides no recovery operations for any syndrome input.
        """

    @staticmethod
    def decode(
        _measurements: StdOutput,
        **_kwargs: object,
    ) -> list[QuantumCircuit]:
        """Decode measurements and return recovery operations.

        Args:
        ----
            measurements: The stabilizer measurements to decode
            **kwargs: Additional keyword arguments (ignored)

        Returns:
        -------
            Empty list - no recovery operations

        """
        return []
