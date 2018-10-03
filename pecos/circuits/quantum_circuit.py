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
"""
Contains the class ``QuantumCircuit``, which is used to represent quantum circuits.
"""
from collections.abc import MutableSequence
from collections import defaultdict, namedtuple


class QuantumCircuit(MutableSequence):
    """A representation of a quantum circuit.

    Similar to [{gate_symbol: set of qudits, ...}, ...] where each element is a time step in which gates act in
    parallel.

    """

    def __init__(self, circuit_setup=None, **params):
        """

        Args:
            circuit_setup (None, int, list of dict):
            **params: Quantum circuit parameters.

        Attributes:
            self._ticks(list of dict): A list of parallel gates. Each element is a dictionary of gate symbol => gate set.
            This gate dictionary is assumed to be a collection of gates that act in parallel on the qudits.
            self.active_qudits(list): If `check_overlap` == True then ``active_qudits`` will be tracked; otherwise,
            ``active_qudits`` will not be tracked.
        """
        self._gates_class = ParamGateCollection
        self._ticks_class = list
        self._ticks = self._ticks_class()
        self.params = params

        if 'tracked_qudits' in params:
            raise Exception('error')

        if circuit_setup is not None:
            self._circuit_setup(circuit_setup)

    def append(self, symbol, locations=None, **params):
        """Adds a new gate=>gate_locations (set) pair to the end of ``self.gates``.

        Args:
            symbol(str or dict): A gate dictionary of gate symbol => set of qudit ids or tuples of qudit ids
            locations:

        Example:
            >>> quantum_circuit = QuantumCircuit()
            >>> quantum_circuit.append({'X': {0, 1, 5, 6}, 'Z':{7, 8, 9}})
            >>> quantum_circuit.append('X', {0, 1, 3, 7})

        This then creates a new time step at the end of ``self._ticks`` and adds the gate to it.

        """

        gates = self._gates_class(self, symbol, locations, **params)

        self._ticks.append(gates)

    def update(self, symbol, locations=None, tick=-1, emptyappend=False, **params):
        """Updates that last group of parallel gates to include the gate acting on the set of qudits.

        Args:
            symbol(str or dict): A gate dictionary of gate symbol => set of qudit ids or tuples of qudit ids
            locations(set or None):
            tick(int): The time (tick) when the update should occur.
            emptyappend(bool): Whether it is allowed to add an empty tick if the QuantumCircuit is empty.
            **params:

        """

        if emptyappend:  # Allowed to add a tick if QuantumCircuit is empty
            try:
                gates = self._ticks[tick]

            except IndexError:
                if len(self) == 0:
                    self.add_ticks(1)
                    gates = self._ticks[tick]
                else:
                    raise

            gates.add(symbol, locations, **params)

        else:
            gates = self._ticks[tick]
            gates.add(symbol, locations, **params)

    def discard(self, locations, tick=-1):
        """Discards ``locations`` for tick ``tick``.

        Args:
            locations:
            tick:

        Returns:

        """

        self._ticks[tick].discard(locations)

    def add_ticks(self, num_ticks, **params):
        """Adds `num_ticks` number of ticks (empty dictionaries) to the end of `self._ticks`.

        Args:
            num_ticks:

        Returns:

        """

        for _ in range(num_ticks):
            gates = self._gates_class(self, {}, **params)
            self._ticks.append(gates)

    def items(self, tick=None, params=True):
        """An iterator through all gates/qudits in the quantum circuit.

        If ``tick`` is not None then it will iterate over only the qudits/qudits in the corresponding tick.

        Args:
            tick:
            params:

        Returns:

        """

        if tick is None:

            for gates in self._ticks:
                for args in gates.items(params):
                    yield args

        else:

            for args in self._ticks[tick].items(params):
                yield args

    def insert(self, tick, item):
        """Inserts ``gate_dict`` into ``ticks`` at index ``tick``.

        Args:
            tick:
            item:

        Returns:

        """
        gate_dict, params = item
        gates = self._gates_class(self, gate_dict, **params)
        self._ticks.insert(tick, gates)

    @property
    def active_qudits(self):
        """
        Returns the active_qudits of all the ticks.

        Returns:

        """

        active = []
        for gates in self._ticks:
            active.append(gates.active_qudits)
        return active

    def _circuit_setup(self, circuit_setup):

        if isinstance(circuit_setup, int):
            # Reserve ticks
            self.add_ticks(circuit_setup)

        else:
            # Build circuit from other description (a shallow copy).
            for other_tick in circuit_setup:
                self.append(other_tick)

    def __getitem__(self, tick):
        """Returns tick when instance[index] is used.

        Args:
            tick(int): Tick index of ``self._ticks``.

        Returns:

        """

        return self._ticks[tick]

    def __setitem__(self, tick, item):

        gate_dict, params = item
        self._ticks[tick] = self._gates_class(self, gate_dict, **params)

    def __len__(self):
        """Used to return number of ticks when len() is used on an instance of this class.

        Returns:

        """

        return len(self._ticks)

    def __delitem__(self, tick):
        """Used to delete a tick. For example: del instance[key]

        Args:
            tick:

        Returns:

        """
        self._ticks[tick] = self._gates_class(self)

    def __str__(self):
        """String returned when a string representation is requested. This occurs during printing.

        Returns:

        """
        str_list = []
        for gates in self._ticks:
            # str_list.append(dict(tick.symbols))
            tick_list = []
            for symbol, locations, params in gates.items(params=True):
                if len(params) == 0:
                    tick_list.append("'%s': %s" % (symbol, locations))
                else:
                    tick_list.append("'%s': loc: %s - params=%s" % (symbol, locations, params))
            tick_list = ', '.join(tick_list)
            str_list.append('{%s}' % tick_list)

        if self.params:
            return "QuantumCircuit(params=%s, ticks=[%s])" % (str(self.params), ', '.join(str_list))
        else:
            return "QuantumCircuit([%s])" % ', '.join(str_list)

    def __repr__(self):
        """

        Returns:

        """

        return self.__str__()

    def __copy__(self):
        """Create a shallow copy."""

        newone = QuantumCircuit()
        newone.params = dict(self.params)
        newone._ticks = self._ticks_class(self._ticks)

        return newone

    def copy(self):
        """
        Create a shallow copy of the circuit.

        Returns:

        """
        return self.__copy__()

    def __iter__(self):
        return self.items()


class ParamGateCollection:
    """
    Data structure for a tick.
    """

    Gate = namedtuple('Gate', 'symbol, params, locations')

    def __init__(self, circuit, symbol=None, locations=None, **params):
        self.circuit = circuit
        self.active_qudits = set([])
        self.symbols = defaultdict(list)

        self.add(symbol, locations, **params)

    def add(self, symbol, locations=None, **params):
        """

        Args:
            symbol:
            locations:

        Returns:

        """

        # If locations is None then assume symbol is a gate_dict.
        if locations is None:
            gate_dict = symbol
        else:
            gate_dict = {symbol: locations}

        self._verify_qudits(gate_dict)

        for gate_symbol, gate_locations in gate_dict.items():

            if gate_locations:

                for gate in self.symbols[gate_symbol]:
                    if params == gate.params:
                        gate.locations.update(gate_locations)
                        break
                else:
                    self.symbols[gate_symbol].append(self.Gate(gate_symbol, params, gate_locations))

        return self

    def discard(self, locations):
        """
        Remove gate locations.

        Args:
            locations:

        Returns:

        """
        for _, gate_list in self.symbols.items():

            for gate in gate_list:

                for location in locations:

                    if location in gate.locations:
                        gate.locations.discard(location)
                        break

        # symbols: dict-> symbol: list[gate]

        # Remove keys with empty locations
        # --------------------------------
        # remove symbol from dictionary
        for symbol in list(self.symbols.keys()):
            for gate in self.symbols[symbol]:
                if not gate.locations:
                    self.symbols[symbol].remove(gate)

        # Update active_qudits
        # --------------------
        # Remove locations from active_qudits
        for location in locations:
            if isinstance(location, tuple):
                for loc in location:
                    self.active_qudits.discard(loc)
            else:
                self.active_qudits.discard(location)

        return self

    def _verify_qudits(self, gate_dict):
        """Verifies that all qudits are being acted on in parallel during a time step (tick).

        The qudit ids are added to ``self.active_qudits``.

        Args:
            gate_dict:
        Raises:
            Exception: If qudit ids are not int or if non-parallel gates are found (i.e., a qudit ha already been acted
            on by a gate.

        """

        for _, qudit_locations in gate_dict.items():

            for location in qudit_locations:

                # Make sure we can iterate over q.
                if not isinstance(location, tuple):
                    q_iter = (location,)
                else:
                    q_iter = location

                for qi in q_iter:

                    if qi in self.active_qudits:
                        raise Exception('Qudit %s has already been acted on by a gate!' % str(qi))
                    else:
                        self.active_qudits.add(qi)

    def items(self, params=True, tick=None):
        """
        Generator to return a dictionary-like iter.
        Returns:

        """

        if params:
            for gate_symbol, gate_list in self.symbols.items():
                for gate in gate_list:
                    yield gate_symbol, gate.locations, gate.params

        else:
            for gate_symbol, gate_list in self.symbols.items():
                for gate in gate_list:
                    yield gate_symbol, gate.locations

    def __str__(self):

        tick_list = []
        for symbol, locations, params in self.items(params=True):
            if len(params) == 0:
                tick_list.append("'%s': %s" % (symbol, locations))
            else:
                tick_list.append("'%s': loc: %s - params=%s" % (symbol, locations, params))
        tick_list = ', '.join(tick_list)

        return "Tick({%s})" % tick_list

    def __repr__(self):
        return self.__str__()
