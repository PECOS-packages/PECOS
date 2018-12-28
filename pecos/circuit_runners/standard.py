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
from collections import namedtuple
import numpy as np
from ..outputs import StdOutput
from ..circuits import QuantumCircuit, LogicalCircuit
from .. import simulators as sims

# TODO: This seem a bit overly complicated:
LogicalTime = namedtuple('LogicalTime', 'logical_tick_index, instr_index')
LogicalSpace = namedtuple('LogicalSpace', 'logical_gate_location')

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

    def run(self, state, circuit, error_gen=None, error_params=None, error_circuits=None, output=None,
            removed_locations=None):
        """
        Runs a generic circuit whether logical, physical, or otherwise.
        Args:
            state:
            circuit:
            error_gen:
            error_params:
            error_circuits:
            output:
            removed_locations:

        Returns:

        """
        # TODO: Figure out this output buisness...

        pass

    def run_logic(self, state, logical_circuit, error_gen=None, error_params=None, error_circuits=None, output=None):
        """
        Run logical circuit and these circuits to update a state.

        Args:
            state:
            logical_circuit:
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

        generate_errors = False
        # No errors
        if error_gen is None:
            if error_circuits is None:
                error_circuits = {}

        # new errors
        else:
            generate_errors = True
            error_circuits = error_gen.start(logical_circuit, error_params)
        # --------------------------------------------------

        # Logical gates
        for logical_index in range(len(logical_circuit)):

            for logical_gate, _, _ in logical_circuit.items(tick=logical_index):

                # logical locations
                # for logical_location in logical_gate_locations:

                # TODO: need an iter for both QC and logical circuits that loops through ticks and places all the time
                # in one variable... so error generators can work on both seamlessly...
                # TODO: combine run_logic and run_circuit... make a new... run_tick
                # TODO: run through ticks or circuits?
                # Ticks
                for instr_index, tick_index, tick_circuit in logical_gate.iter_physical_ticks():

                    logical_time = (logical_index, instr_index)

                    # Get errors
                    if logical_gate.error_free:
                        errors = {}

                    else:

                        if generate_errors:
                            error_gen.generate(logical_gate, logical_time, tick_index)
                            # TODO: how does this affect the code? what is the output??? error_circuits???

                        errors = error_circuits.get((logical_index, instr_index), {}).get(tick_index, {})

                    # --------------------
                    # RUN QUANTUM CIRCUITS
                    # --------------------

                    # Before errors
                    # -------------
                    result = self.run_circuit(state, errors.get('before', empty_circuit))
                    output.record(result, logical_time, tick_index)

                    # Ideal tick
                    # ----------
                    result = self.run_circuit(state, tick_circuit, removed_locations=errors.get('replaced'))
                    output.record(result, logical_time, tick_index)

                    # After errors
                    # ------------
                    result = self.run_circuit(state, errors.get('after', empty_circuit))
                    output.record(result, logical_time, tick_index)

        return output, error_circuits

    def run_circuit(self, state, circuit, removed_locations=None):
        """
        Apply a ``QuantumCircuit`` directly to a state without output.

        Args:
            state:
            circuit:
            removed_locations:

        Returns:

        """

        if removed_locations is None:
            removed_locations = set([])

        results = {}

        try:
            num_ticks = len(circuit)
        except TypeError:
            num_ticks = 1

        for t in range(num_ticks):

            gate_results = {}
            for symbol, physical_gate_locations, params in circuit.items(tick=t):
                gate_kwargs = params.get('gate_kwargs', {})

                gate_results = {}

                if self.gate_dict:
                    for location in physical_gate_locations - removed_locations:
                        this_result = self.gate_dict[symbol](state, location, **gate_kwargs)

                        if this_result:
                            gate_results[location] = this_result
                else:
                    gate_results = self.simulator.run_gate(symbol, physical_gate_locations - removed_locations,
                                                           **gate_kwargs)

            if gate_results:
                results[t] = gate_results

        return results

    def run_tick(self, state, tick, removed_locations=None):

        if removed_locations is None:
            removed_locations = set([])

        results = {}

        # for each gate in tick: symbol, locations, params

        return results
