# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2018 Ciarán Ryan-Anderson
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


class Simulator(object):
    """
    A parent class to provide standard methods for simulators.
    """

    def __init__(self):
        self.bindings = {}

    def run_gate(self, symbol, locations, **params):
        """

        Args:
            symbol:
            locations:
            **params:

        Returns:

        """

        output = {}
        for location in locations:
            results = self.bindings[symbol](self, location, **params)

            if results:
                output[location] = results

        return output

    def run_circuit(self, circuit, removed_locations=None):
        """

        Args:
            circuit (QuantumCircuit): A circuit instance or object with an appropriate items() generator.
            removed_locations:

        Returns (list): If output is True then the circuit output is returned. Note that this output format may differ
        from what a ``circuit_runner`` will return for the same method named ``run_circuit``.

        """

        # TODO: removed_locations doesn't make sense except if circuit is tick_circuit
        # because can't say not to do gates for particular ticks....

        if removed_locations is None:
            removed_locations = set([])

        results = {}
        for symbol, locations, params in circuit.items():
            gate_results = self.run_gate(symbol, locations - removed_locations, **params)
            results.update(gate_results)

        return results
