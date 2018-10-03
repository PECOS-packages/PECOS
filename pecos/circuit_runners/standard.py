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

try:
    from ..simulators import cySparseSim as Sim
except ImportError:
    from ..simulators import SparseSim as Sim

# TODO: This seem a bit overly complicated:
LogicalTime = namedtuple('LogicalTime', 'logical_tick_index, instr_index')
LogicalSpace = namedtuple('LogicalSpace', 'logical_gate_location')

empty_circuit = QuantumCircuit()


class Standard(object):
    """
    This class represents a standard model for running quantum circuits and adding in errors.
    """

    def __init__(self, simulator=None, seed=None):
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
            self.default_simulator = Sim

        elif isinstance(simulator, str):
            if simulator == 'sparsesim':
                self.default_simulator = sims.SparseSim
            elif simulator == 'cysparsesim':
                self.default_simulator = sims.cySparseSim
            elif simulator == 'projectq':
                self.default_simulator = sims.ProjectQSim
            else:
                raise Exception('String not recognized!')

        else:
            self.default_simulator = simulator

        # Set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def init(self, num_qudits, simulator=None, **kwargs):
        """

        Args:
            num_qudits:
            simulator:

        Returns:

        """
        if simulator is None:
            return self.default_simulator(num_qudits, **kwargs)
        else:
            return simulator(num_qudits, **kwargs)

    def run(self, state, circuit, error_gen=None, error_params=None, error_circuits=None, gate_dict=None,
            output=None, removed_locations=None, give_output=None):
        """
        Runs a generic circuit whether logical, physical, or otherwise.
        Args:
            state:
            circuit:
            error_gen:
            error_params:
            error_circuits:
            gate_dict:
            output:
            removed_locations:
            give_output:

        Returns:

        """
        # TODO: Figure out this output buisness...
        # TODO: Raise exceptions if an arguement is given that is not expected.

        if isinstance(circuit, LogicalCircuit):
            return self.run_logic(state, circuit, error_gen, error_params, error_circuits, gate_dict, output)
        else:
            return self.run_circuit(state, circuit, give_output, removed_locations, gate_dict)

    def run_logic(self, state, logical_circuit, error_gen=None, error_params=None, error_circuits=None, gate_dict=None,
                  output=None):
        """
        Run logical circuit and these circuits to update a state.

        Args:
            state:
            logical_circuit:
            output:
            error_gen:
            error_params:
            error_circuits:
            gate_dict:

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

            for logical_gate, _ in logical_circuit.items(tick=logical_index, params=False):

                # logical locations
                # for logical_location in logical_gate_locations:

                # Ticks
                for instr_index, tick_index, tick_circuit in logical_gate.iter_physical_ticks():

                    # logical_coord = (LogicalSpace(logical_location), LogicalTime(logical_index, instr_index))
                    # logical_coord = (LogicalSpace(logical_gate_locations), LogicalTime(logical_index, instr_index))
                    logical_time = (logical_index, instr_index)

                    # Get errors
                    if logical_gate.error_free:
                        errors = {}

                    else:

                        if generate_errors:
                            error_gen.generate(logical_gate, logical_time, tick_index)

                        # errors = error_circuits.get((logical_time, tick_index), {})
                        # gate_tick_id, instr_id => tick_id
                        errors = error_circuits.get((logical_index, instr_index), {}).get(tick_index, {})

                    # --------------------
                    # RUN QUANTUM CIRCUITS
                    # --------------------

                    # Before errors
                    # -------------
                    result = self.run_circuit(state, errors.get('before', empty_circuit), gate_dict=gate_dict,
                                              give_output=True)
                    output.record(result, logical_time, tick_index)
                    # self.run_circuit(state, errors.get('before', empty_circuit), gate_dict=gate_dict)

                    # Ideal tick
                    # ----------
                    result = self.run_circuit(state, tick_circuit, removed_locations=errors.get('replaced'),
                                              gate_dict=gate_dict,  give_output=True)
                    output.record(result, logical_time, tick_index)

                    # After errors
                    # ------------
                    result = self.run_circuit(state, errors.get('after', empty_circuit), gate_dict=gate_dict,
                                              give_output=True)
                    output.record(result, logical_time, tick_index)

                    # self.run_circuit(state, errors.get('after', empty_circuit), gate_dict=gate_dict)

        return output, error_circuits

    def run_circuit(self, state, circuit, give_output=True, removed_locations=None, gate_dict=None):
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
        # TODO: Allow the application of errors...
        # TODO: Simplify gate dict... maybe move that responsibility to simulator...

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
                            results = gate_dict[symbol](location, **gate_kwargs)

                            if results:
                                gate_result[location] = results

                    else:
                        for location in locations:
                            gate_dict[symbol](location, **gate_kwargs)

                if gate_result:
                    result[t] = gate_result

        else:

            for t in range(num_ticks):
                for symbol, physical_gate_locations, params in circuit.items(params=True, tick=t):

                    gate_kwargs = params.get('gate_kwargs', {})

                    gate_result = state.run_gate(symbol, physical_gate_locations-removed_locations, give_output,
                                                 **gate_kwargs)
                    if gate_result:
                        result[t] = gate_result

        return result
