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

import struct
import os
import random
import numpy as np
from ..outputs import StdOutput
from ..circuits import QuantumCircuit
from .. import simulators as sims

empty_circuit = QuantumCircuit()


class Standard(object):
    """
    This class represents a standard model for running quantum circuits and adding in errors.
    """

    def __init__(self, simulator=None, seed=None, gate_dict=None):
        """

        Args:
            seed:
        """

        if seed is None:
            # Create a good random number seed
            self.seed = struct.unpack("<L", os.urandom(4))[0]
        else:
            self.seed = seed

        if simulator is None:
            self.simulator = sims.SparseSim

        else:
            if simulator == 'sparsesim':
                self.simulator = sims.SparseSim
            elif simulator == 'cysparsesim':
                self.simulator = sims.cySparseSim
            elif simulator == 'projectq':
                self.simulator = sims.ProjectQSim
            else:
                raise Exception('Simulator not recognized!')

        if gate_dict is None:
            self.gate_dict = self.simulator.gate_dict
        else:
            self.gate_dict = gate_dict

        # Set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def init(self, num_qudits, **kwargs):
        """

        Args:
            num_qudits:

        Returns:

        """
        return self.simulator(num_qudits, **kwargs)

    def run(self, state, circuit, error_gen=None, error_params=None, error_circuits=None, output=None):
        """
        Run logical circuit and these circuits to update a state.

        Args:
            state:
            circuit:
            output:
            error_gen:
            error_params:
            error_circuits:

        Returns:

        """

        if output is None:
            output = StdOutput()

        # ---------------------------------------------------
        # Determine if there are no, previous, or new errors.
        #

        # No errors
        if error_gen is None:
            generate_errors = False
            if error_circuits is None:
                error_circuits = {}

        # new errors
        else:
            generate_errors = True
            error_circuits = error_gen.start(circuit, error_params)
        # --------------------------------------------------

        for tick_circuit, time, params in circuit.iter_ticks():

            # Get errors
            if params.get('circuit', {}).get('error_free', False):
                errors = {}

            else:

                if generate_errors:
                    error_circuits = error_gen.generate_tick_errors(tick_circuit, time, **params)

                errors = error_circuits.get(time, {})

            # --------------------
            # RUN QUANTUM CIRCUITS
            # --------------------

            # Before errors
            # -------------
            self.run_circuit(state, errors.get('before', empty_circuit))

            # Ideal tick
            # ----------
            result = self.run_gates(state, tick_circuit, removed_locations=errors.get('replaced'))
            output.record(result, time)

            # After errors
            # ------------
            self.run_circuit(state, errors.get('after', empty_circuit))

        return output, error_circuits

    def run_circuit(self, state, circuit, removed_locations=None, output=None):
        """
        Apply a ``QuantumCircuit`` directly to a state.

        Args:
            state:
            circuit:
            removed_locations:
            output:

        Returns:

        """

        if output is None:
            output = StdOutput()

        for tick_circuit, tick_index, params in circuit.iter_ticks():
            results = self.run_gates(state, tick_circuit, removed_locations)

            output.record(results, tick_index)

        return output

    def run_gates(self, state, gates, removed_locations=None):
        """
        Directly apply a collection of quantum gates to a state.

        Args:
            state:
            gates:
            removed_locations:

        Returns:

        """

        if removed_locations is None:
            removed_locations = set([])

        gate_results = {}
        for symbol, physical_gate_locations, params in gates.items():

            if self.gate_dict:
                gate_results = {}
                for location in physical_gate_locations - removed_locations:
                    this_result = self.gate_dict[symbol](state, location, **params.get('gate_kwargs', {}))

                    if this_result:
                        gate_results[location] = this_result
            else:
                gate_results = self.simulator.run_gate(symbol, physical_gate_locations - removed_locations,
                                                       **params.get('gate_kwargs', {}))

        return gate_results
