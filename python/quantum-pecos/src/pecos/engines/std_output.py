# Copyright 2018 The PECOS Developers
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

"""Standard output handling and formatting for PECOS measurements.

This module provides utilities for processing and formatting measurement
results and simulation outputs in standard formats for analysis and debugging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class StdOutput(dict):
    """Class used to record results of gates (typically, measurements).

    (logical space, logical time) -> time(tick) -> {location: result}
    """

    def record(self, result_dict: dict[Any, Any], time: int | tuple[int, ...]) -> None:
        """Record result dictionary at specified time.

        Args:
        ----
            result_dict: Dictionary of results to record.
            time: Time value to associate with the results.

        """
        if result_dict:
            logical_dict = self.setdefault(time, {})
            logical_dict.update(result_dict)

    def simplified(self, *, last: bool = False) -> dict | set:
        """Gives output in a simplified version. {logical coord=>{set of locations}, ...}.

        Outputs the syndromes of the final logical instruction.

        """
        simple = {}
        for time, results in self.items():
            fired = set(results.keys())

            simple[time] = fired

        if last and simple:
            # Get the last coordinate
            keys = simple.keys()

            last_id = sorted(keys)[-1]
            simple = simple[last_id]  # just a set of qids

        return simple
