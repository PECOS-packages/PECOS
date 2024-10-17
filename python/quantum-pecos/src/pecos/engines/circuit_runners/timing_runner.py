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

from time import perf_counter as default_timer

from pecos.engines.circuit_runners.standard import Standard


class TimingRunner(Standard):
    """This class represents a standard model for running quantum circuits and adding in errors."""

    def __init__(self, seed=None, timer=None) -> None:
        """Args:
        ----
            seed:
        """
        super().__init__(seed)

        self.total_time = 0.0
        self.num_gates = 0

        if timer is None:
            self.timer = default_timer
        else:
            self.timer = timer

    def reset_time(self):
        """Used to clear the time data in `total_time`.

        Returns:
        -------

        """
        self.total_time = 0.0
        self.num_gates = 0

    def run_gates(self, state, gates, removed_locations=None):
        """Directly apply a collection of quantum gates to a state.

        Args:
        ----
            state:
            gates:
            removed_locations:

        Returns:
        -------

        """
        timer = self.timer

        if removed_locations is None:
            removed_locations = set()

        gate_results = {}
        for symbol, physical_gate_locations, gate_kwargs in gates.items():
            ti = timer()
            gate_results = state.run_gate(
                symbol,
                physical_gate_locations - removed_locations,
                **gate_kwargs,
            )
            tf = timer()
            self.total_time += tf - ti
            self.num_gates += len(physical_gate_locations - removed_locations)

        return gate_results
