#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from time import perf_counter as default_timer
from .standard import Standard


class TimingRunner(Standard):
    """
    This class represents a standard model for running quantum circuits and adding in errors.
    """

    def __init__(self, simulator=None, seed=None, timer=None):
        """

        Args:
            seed:
        """

        super().__init__(simulator, seed)

        self.total_time = 0.0
        self.num_gates = 0

        if timer is None:
            self.timer = default_timer
        else:
            self.timer = timer

    def reset_time(self):
        """
        Used to clear the time data in `total_time`.

        Returns:

        """
        self.total_time = 0.0
        self.num_gates = 0


    def run_circuit(self, state, circuit, give_output=False, removed_locations=None, gate_dict=None):
        """
        Apply a ``QuantumCircuit`` directly to a state without output.

        Args:
            state:
            circuit:
            give_output:
            removed_locations:
            gate_dict:

        Returns:

        """

        # TIMER: It is assumed that this method is the only part of the code that calls gates.

        timer = self.timer

        if removed_locations is None:
            removed_locations = set([])

        result = {}

        try:
            num_ticks = len(circuit)
        except TypeError:
            num_ticks = 1

        if gate_dict:

            for t in range(num_ticks):
                gate_result = {}
                for symbol, physical_gate_locations, params in circuit.items(params=True, tick=t):
                    gate_kwargs = params.get('gate_kwargs', {})
                    locations = physical_gate_locations - removed_locations

                    if give_output:
                        gate_result = {}
                        for location in locations:
                            # TIME GATES
                            # ----------
                            ti = timer()
                            results = gate_dict[symbol](location, **gate_kwargs)
                            tf = timer()
                            self.total_time += tf - ti
                            self.num_gates += 1
                            # ---------

                            if results:
                                gate_result[location] = results

                    else:
                        for location in locations:
                            gate_dict[symbol](location, **gate_kwargs)

                # Record the measurement results of the tick
                if gate_result:
                    result[t] = gate_result

        else:
            for t in range(num_ticks):
                for symbol, physical_gate_locations, params in circuit.items(params=True, tick=t):

                    gate_kwargs = params.get('gate_kwargs', {})

                    locations = physical_gate_locations - removed_locations

                    # TIME GATES
                    # ----------
                    ti = timer()
                    gate_result = state.run_gate(symbol, locations, give_output, **gate_kwargs)
                    tf = timer()
                    self.total_time += tf - ti
                    self.num_gates += len(locations)
                    # ---------

                    if gate_result:
                        result[t] = gate_result

        return result
