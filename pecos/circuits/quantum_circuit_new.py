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
# from collections.abc import MutableSequence
# locations (set of ints, set of tuples of ints): A set of ints or a set of tuples of ints that are the qudit
#                 ids that the gates act on.
from warnings import warn
from typing import Union, Optional, Any, Dict, Set, Tuple, Generator, List
import copy
from sortedcontainers import SortedDict
from ..misc.errors import PECOSTypeError, GateError, GateOverlapError
from ..types import GateLocations, Time, Symbol, Params

# Defining types for typing:
GateDict = Dict[Symbol, GateLocations]

# TODO: In the tick, store gates based on duration, then symbol. (See CircuitSequence)


# class QuantumCircuit(MutableSequence):
class QuantumCircuit(object):
    """A representation of a quantum circuit.

    Attributes:
        params (dict): A dictionary containing extra information about the quantum circuit.
        verify: Whether to  check for gate overlap.
        active_qudits: The set of qudit ids that have been acted on by a gate in this quantum circuit.

    """

    def __init__(self,
                 circuit_setup: Optional[int] = None,
                 verify: bool = True,
                 **params: Any) -> None:
        """

        Args:
            circuit_setup (None, int, list of dict):
            verify (bool):
            **params (dict): A dictionary containing extra information about the quantum circuit.

        Notes:
            If verify is set to True, `active_qudits` will be empty as it will not be updated.
        """

        # TODO: work out what circuit_setup data type is

        self.verify = verify
        self.params = params

        if not verify:
            self.params['verify'] = verify

        self._ticks = SortedDict()
        self._tick_class = self.params.get('tick_class', Tick)

        self._active_qudits_end = {}
        # qid -> SortedSet{time+duration, ...} To track when do gates stop acting on the qudits with duration

        if circuit_setup is not None:
            self._circuit_setup(circuit_setup)

    @property
    def active_qudits(self) -> Set[int]:
        """The set of qudit ids that are acted upon by the gates in this `QuantumCircuit`."""

        qudits = set()

        for tick in self._ticks.values():
            qudits.update(tick.active_qudits)

        return qudits

    def duration(self):
        # TODO: !!!
        pass

    def gate(self,
             symbol: Symbol,
             locations: Locations,
             tick_time: Time,
             interval: Time = 1,
             **params: Any) -> None:
        """
        Adds gates of a single gate type (indicated by `symbol`) at time `tick_time`.

        Args:
            symbol (str): The string indicating the type of gate, e.g., 'H' for a Hadamard gate.
            locations (set of ints, set of tuples of ints): A set of ints or a set of tuples of ints that are the qudit
                ids that the gates act on.
            tick_time (int, float): The time at which the gates occur. If `tick_time` == -1, then the gates will be
                added to the last tick. If `tick_time` == -2, then the gates will be appended to the end of the circuit.
            interval (int, float): If appending, the time interval between this tick and the previous one.
            **params:

        Returns:

        """

        if tick_time == -2:  # Append to the end
            self.append({symbol: locations}, interval=interval, **params)
        else:  # Add gates whether a corresponding tick exists or not
            # Note: if tick_time == -1 then the last Tick will be updated
            self.add({symbol: locations}, tick_time=tick_time, override=False, **params)

    def append(self,
               gate_dict: Union[Symbol, GateDict],
               locations: Optional[Locations] = None,
               interval: Time = 1,
               **params: Any) -> None:
        """
        Appends a new tick to the end of the circuit.

        Args:
            gate_dict (Union[str, Dict[str, Union[Set[int], Set[Tuple[int, ...]]]]): A gate dictionary of gate
                symbol => set of qudit ids or tuples of qudit ids
            locations:
            interval (int, float): The time interval between this tick and the previous one.

            **params:

        Example:
            >>> quantum_circuit = QuantumCircuit()
            >>> quantum_circuit.append({'X': {0, 1, 5, 6}, 'Z':{7, 8, 9}})

        This then creates a new time step at the end of ``self._ticks`` and adds the gate to it.

        """

        if len(self._ticks) == 0:
            tick_time = 0
        else:
            tick_time = self.itime(-1) + interval

        # Create new Tick instance
        tick_instance = self._tick_class(time=tick_time, circuit=self)

        # Add gates to the Tick and append the tick to the end of `_ticks`
        tick_instance.add_gates(gate_dict, locations, **params)
        self._ticks[tick_time] = tick_instance

    def update(self,
               gate_dict: Union[Symbol, GateDict],
               locations: Optional[Locations] = None,
               tick_time: Time = None,
               **params: Any) -> None:
        """
        Updates an existing tick with more gates. If the requested tick does not exist, then an error is raised.

        Args:
            gate_dict:
            locations:
            tick_time (int, float):
            **params:

        Returns:

        """

        if tick_time is None:
            raise GateError('Must specify a tick time!')

        # Add the gates to the Tick
        self._ticks[tick_time].add_gates(gate_dict, locations, **params)

    def add(self,
            gate_dict: Union[Symbol, GateDict],
            locations: Optional[Locations] = None,
            tick_time: Optional[Time] = None,
            override: bool = False,
            **params: Any) -> None:
        """
        Adds gates to the quantum circuit at a given tick time (whether the tick already exist or not).

        By default, if the tick already exist then this method acts like `update`. That is, this method only adds gates
        and doesn't overrides them.

        Args:
            gate_dict:
            locations:
            tick_time:
            override (bool):
            **params:

        Returns:

        """

        if tick_time in self._ticks:

            if override:
                del self._ticks[tick_time]

            # Add the gates to the Tick
            self._ticks[tick_time].add_gates(gate_dict, locations, **params)

        else:

            # Insert tick at some new time
            tick_instance = self._tick_class(time=tick_time, circuit=self)
            tick_instance.add_gates(gate_dict, locations, **params)

            self._ticks[tick_time] = tick_instance

    def discard(self,
                locations: Locations,
                tick_time: Time = -1) -> None:
        """Discards ``locations`` from the `Tick` of index `tick`.

        Args:
            locations (set of ints, set of tuples of ints): A set of ints or a set of tuples of ints that are the qudit
            ids that the gates act on.

            tick_time:

        Returns:

        """

        self._ticks[tick_time].discard(locations)  # TODO: FINISH!!!

    def append_empty_ticks(self, num_ticks: int) -> None:
        """Appends empty ticks."""

        for _ in range(num_ticks):
            self.append({})

    def add_ticks(self, num_ticks: int) -> None:
        """Appends empty ticks."""

        self.append_empty_ticks(num_ticks)
        warn('`add_ticks` is being deprecated and being replaced by `append_empty_ticks`.')

    def extend(self, quantum_circuit: 'QuantumCircuit'):
        """Appends ticks from another `QuantumCircuit` to this one."""
        pass
        # TODO: Finish!!!

    def iter_times(self,
                   minimum: Optional[Time] = None,
                   maximum: Optional[Time] = None,
                   inclusive: Tuple[bool, bool] = (True, True),
                   reverse: bool = False) -> Generator['Tick', None, None]:
        """
        A generator allowing one to loop through the tick times.

        This method gives access to the `irange` method of `self._tick` provided by the class `SortedDict`.

        Args:
            minimum (int, float, None): Minimum tick time to start iterating.
            maximum (int, float, None): Maximum tick time to stop iterating.
            inclusive (tuple of two bools): A tuple of two booleans that indicate whether the minimum tick time and/or
                the maximum tick time should be included.
            reverse (bool): Whether to reverse the order of iterated values (go from largest tick time to smallest).

        Returns: iterator

        """
        for t in self._ticks.irange(minimum, maximum, inclusive, reverse):
            yield t

    def iter_ticks(self,
                   minimum: Optional[Time] = None,
                   maximum: Optional[Time] = None,
                   inclusive: Tuple[bool, bool] = (True, True),
                   reverse: bool = False) -> Generator['Tick', None, None]:
        """
        A generator allowing one to loop through the tick times.

        This method gives access to the `irange` method of `self._tick` provided by the class `SortedDict`.

        Args:
            minimum (int, float, None): Minimum tick time to start iterating.
            maximum (int, float, None): Maximum tick time to stop iterating.
            inclusive (tuple of two bools): A tuple of two booleans that indicate whether the minimum tick time and/or
                the maximum tick time should be included.
            reverse (bool): Whether to reverse the order of iterated values (go from largest tick time to smallest).

        Returns: iterator

        """
        for t in self._ticks.irange(minimum, maximum, inclusive, reverse):
            yield self._ticks[t]

    def iter_gates(self,
                   minimum: Optional[Time] = None,
                   maximum: Optional[Time] = None,
                   inclusive: Tuple[bool, bool] = (True, True),
                   reverse: bool = False) -> Generator['Gate', None, None]:
        """
        A generator allowing one to loop through the tick times.

        This method gives access to the `irange` method of `self._tick` provided by the class `SortedDict`.

        Args:
            minimum (int, float, None): Minimum tick time to start iterating.
            maximum (int, float, None): Maximum tick time to stop iterating.
            inclusive (tuple of two bools): A tuple of two booleans that indicate whether the minimum tick time and/or
                the maximum tick time should be included.
            reverse (bool): Whether to reverse the order of iterated values (go from largest tick time to smallest).

        Returns: iterator

        """
        for t in self._ticks.irange(minimum, maximum, inclusive, reverse):
            for gate in self._ticks[t].iter_gates():
                yield gate

    def copy(self) -> 'QuantumCircuit':
        """Provides a shallow copy of the data structure."""
        return self.__copy__()

    def deepcopy(self) -> 'QuantumCircuit':
        """Provides a deep copy of the data structure."""
        return self.__deepcopy__()

    def get(self,
            tick_time: Time,
            default: Optional[Any] = None) -> Union['Tick', Any]:
        """Returns the `Tick` that occurs at time `tick_time`. If `tick_time` not found, returns `default`."""

        return self._ticks.get(tick_time, default)[1]

    def iget(self,
             index: int,
             default: Optional[Any] = None) -> Union['Tick', Any]:
        """Returns the `Tick` indexed by `index`. If `index` not found, returns `default`."""

        try:
            return self._ticks.peekitem(index)[1]
        except IndexError:
            return default

    def itime(self,
              index: int,
              default: Optional[Any] = None) -> Union[Time, Any]:
        """Returns the tick time of the `Tick` indexed by `index`. If `index` not found, returns `default`."""

        try:
            return self._ticks.peekitem(index)[0]
        except IndexError:
            return default

    def _qc2py(self, make_deepcopy: bool = False) -> Dict[Time, List[Tuple[Symbol, Locations, Params]]]:

        # TODO: document...

        circ_dict = {}
        for tick in self.iter_ticks():
            tick_time = tick.time

            tick_list = []  # can't use a set as the elements aren't hashable
            for gate in tick.iter_gates():

                if make_deepcopy:
                    locations = set(gate.locations)
                    params = copy.deepcopy(gate.params)
                else:
                    locations = gate.locations
                    params = gate.params

                tick_list.append((gate.symbol, locations, params))
            circ_dict[tick_time] = tick_list

        return circ_dict

    @staticmethod
    def _py2qc(pydata,
               qc: Optional['QuantumCircuit'] = None) -> 'QuantumCircuit':
        """

        Args:
            pydata:
            qc:

        Returns:

        """

        # TODO: document...

        if qc is None:  # TODO: Explain why this is being done.
            qc = QuantumCircuit()

        if isinstance(pydata, dict):

            for tick_time, tick_list in pydata.items():
                for symbol, locations, params in tick_list:
                    qc.add({str(symbol): set(locations)}, tick_time=tick_time, **dict(params))

        else:  # An ordered sequence... hopefully...

            for tick_time, tick_list in enumerate(pydata):
                for symbol, locations, params in tick_list:
                    qc.add({str(symbol): set(locations)}, tick_time=tick_time, **dict(params))

        return qc

    def _circuit_setup(self, circuit_setup):
        """Create the data structure using the Quantum([...]) format."""

        if isinstance(circuit_setup, int):
            # Reserve ticks
            self.append_empty_ticks(circuit_setup)

        else:
            # Build circuit from other description (a shallow copy).
            self._py2qc(circuit_setup, self)

    def __getitem__(self, tick_time: Time):
        """Returns tick when using the bracket notation.

        Examples:
            >>> qc = QuantumCircuit()
            >>> qc.add('X', {1, 2, 3}, 0.123)
            >>> print(qc[0.123])
            <Tick 0.123>

        Notes:
            To retrieve a Tick corresponding to a particular index, use the method `iget`.

        Args:
            tick_time (int, float): A time when a tick (represented by a Tick object) occurs.

        Returns: Tick

        """

        return self._ticks.peekitem(tick_time)[1]

    def __setitem__(self,
                    tick_time: Time,
                    item: Set[Tuple[Symbol, Locations, Params]]):
        """
        Either sets or replaces the `Tick` at time `tick_time` using the bracket notation and =.

        Examples:

        >>> qc = QuantumCircuit()
        >>> qc[1] = {('X', {1, 2, 3}, {}), ('Z', {4, 5, 6}, {'duration': 1.0})}

        Args:
            tick_time:
            item:

        Returns:

        """

        del self._ticks[tick_time]

        for symbol, locations, params in item:
            self.add(symbol, locations, tick_time=tick_time, **params)

    def __len__(self):
        """Number of ticks in this quantum circuit."""

        return len(self._ticks)

    def __delitem__(self, tick_time: Time) -> None:
        """Used to delete a tick. For example: del instance[key]

        Args:
            tick_time:

        Returns:

        """
        del self._ticks[tick_time]

    def __copy__(self):
        """Create a shallow copy."""

        pydata = self._qc2py()
        return self._py2qc(pydata)

    def __deepcopy__(self, memodict=None):

        pydata = self._qc2py(make_deepcopy=True)
        return self._py2qc(pydata)

    def __str__(self) -> str:
        """String returned when a string representation is requested. This occurs during printing.

        Returns:

        """

        # ticks = {0: [('U', {}, {1, 2, 3}), ('V', {}, {4, 5, 6}), ], }

        circ_dict = self._qc2py()
        ticks = sorted(circ_dict.keys())

        if ticks == list(range(len(ticks))):
            str_list = [str(circ_dict[t]) for t in ticks]
            return 'QuantumCircuit([%s, ])' % ', '.join(str_list)

        else:
            str_list = ['%s: %s' % (t, circ_dict[t])for t in ticks]
            return 'QuantumCircuit({%s, })' % ', '.join(str_list)

    def __repr__(self) -> str:
        """

        Returns:

        """

        return self.__str__()

    def __iter__(self,
                 minimum: Optional[Time] = None,
                 maximum: Optional[Time] = None,
                 inclusive: Tuple[bool, bool] = (True, True),
                 reverse: bool = False) -> Generator['Gate', None, None]:

        yield self.iter_gates(minimum, maximum, inclusive, reverse)


class Tick(object):
    """
    Represents a point in time when gates are applied. This data structure collects and manages these gates.

    Attributes:
        time:
        circuit (QuantumCircuit):
        active_qudits:

    """

    def __init__(self,
                 time: Time,
                 circuit: QuantumCircuit) -> None:

        self.time = time
        self.circuit = circuit  # parent class
        self.active_qudits = set()

        self._gates = {}

    def add_gates(self,
                  gate_dict: Union[Symbol, GateDict],
                  locations: Locations = None,
                  **params: Any) -> None:
        """Adds a one or more gate types."""

        # symbol -> either a string or a dictionary, e.g., {'X': {1, 2}, 'Y': {3, 4}}
        # locations -> if symbol not None: set of locations

        if locations is not None:
            gate_dict = {gate_dict: locations}

        for symbol, loc in gate_dict.items():
            gate = self.request_gate(symbol, params)
            gate.update(loc)

    def discard(self, locations: Locations) -> None:
        """
        Removes gates indicated by `locations`.

        Args:
            locations (set of ints, set of tuples of ints): A set of ints or a set of tuples of ints that are the qudit
                ids that the gates act on.

        Returns:

        """

        # find gates that have locations... if the locations match... delete them and modify active qudits.

        locations = set(locations)

        for gate in self.iter_gates():
            overlap = gate.locations & locations

            if overlap:
                locations -= overlap

                gate.discard(overlap)  # TODO: FINISH!!!

        if locations:
            raise Exception('The following qudits were not found: %s' % locations)

    def iter_gates(self) -> Generator['Gate', None, None]:
        """A generator that yields the gates held in this `Tick`."""
        for gate_list in self._gates.values():
            for gate in gate_list:
                yield gate

    def request_gate(self,
                     symbol: Symbol,
                     params: Params) -> Optional['Gate']:
        """
        Finds a gate with matching `sym` and `params`; otherwise, it creates a new `Gate` instance.

        Args:
            symbol:
            params:

        Returns: A `Gate` that was found to match `sym` and `params` or a new `Gate` that has been appended to
        self._gates[sym].

        """
        gate_list = self._gates.setdefault(symbol, [])

        for gate in gate_list:
            if params == gate.params:
                break

        else:  # No match found
            gate = Gate(symbol, self, params)
            gate_list.append(gate)

        return gate

    def __str__(self):
        return "<Tick %s>" % self.time


class Gate(object):
    """
    A collection of quantum gates that are of the same type (indicated by `sym`) that have the same gate parameters
    `params` and that each act on a gate locations in `locations`.

    Attributes:
        symbol (str): The string indicating the gate type of the gates being represented by this data structure.
        params (dict): Parameters further specifying the behavior of these gate or how these gate should be treated
            (such as by error models or circuit runners).
        tick (Tick): The tick instance that contains this data structure.
        duration (int, float, None): How many time units it took to apply the gate type.
        locations (set of ints, set of tuples of ints): A set of ints or a set of tuples of ints that are the qudit ids
            that the gates act on.
        gate_size: The number of qudits this gate acts on.
        active_qudits (set of ints): The set of qudit ids that are acted upon by the gates in this data structure.

    """

    # TODO: What about gates that work on qudits of different dimensions as inputs... of datatypes...?

    def __init__(self,
                 symbol: Symbol,
                 tick: Tick,
                 params: Params) -> None:
        self.symbol = symbol
        self.params = params
        self.tick = tick  # parent instance
        self.duration = params.get('duration')
        self.locations = set()
        self.gate_size = None  # TODO: Make sure this is properly accounted for!
        # TODO: Think about how to deal with a gate that acts on mixed q/cbit types/dimensions...
        self.active_qudits = set()

        if self.duration is None:
            self.duration = self.tick.circuit.params.get('default_duration', None)

    def update(self, locations: Locations) -> None:
        """
        Verifies and adds gate locations (qudit ids) to the gate. Also, adds these qudit ids to `active_qudits`.

        Args:
            locations (set of ints, set of tuples of ints): A set of ints or a set of tuples of ints that are the qudit
                ids that the gates act on.
        Returns:

        """

        if self.tick.circuit.verify:
            qudits = self._verify_locations(locations)
            self.active_qudits.update(qudits)
            self.tick.active_qudits.update(qudits)

        self.locations.update(locations)

    def discard(self, locations: Locations) -> None:
        # TODO: FINISH!!!
        # TODO: flatten overlap
        flat_loc = set()
        self.active_qudits -= flat_loc
        self.tick.active_qudits -= flat_loc
        # TODO: FINISH!!!

    def _verify_locations(self, locations: Locations) -> Set[int]:

        qudits = set()

        if self.gate_size is None:
            q = locations.pop()

            if isinstance(q, int):
                self.gate_size = 1
            else:
                self.gate_size = len(q)

            locations.add(q)

        if self.gate_size == 1:

            for q in locations:

                if not isinstance(q, int):
                    raise PECOSTypeError('Qudit not identified by an integer (qudit id)! Instead type %s.' % type(q))

                if q in qudits:
                    raise GateOverlapError('Qudit %s has already been acted on.' % q)
                else:
                    qudits.add(q)

        elif self.gate_size > 1:

            for q in locations:

                if self.gate_size != len(q):
                    raise GateError('Gate size not consistent!')

                for i in q:

                    if not isinstance(i, int):
                        raise PECOSTypeError('Qudit not identified by an integer (qudit id)! Instead type %s.' % type(i))

                    if i in qudits:
                        raise GateOverlapError('Qudit %s has already been acted on.' % i)
                    else:
                        qudits.add(i)

        else:
            raise GateError('Something weird happen.')

        if self.tick.active_qudits & qudits:
            overlap = self.tick.active_qudits & qudits
            raise GateOverlapError('Qudits %s have already been acted on.' % overlap)

        if self.duration is not None:
            self._verify_duration(qudits)

        return qudits

    def _verify_duration(self, qudits: Set[int]) -> None:
        # Get all ticks within time (not including very last... exclusive)... if match..
        time_begin = self.tick.time
        time_end = time_begin + self.duration

        for tick in self.tick.circuit.iter_ticks(minimum=time_begin, maximum=time_end, inclusive=(False, False)):
            if qudits & tick.active_qudits:
                raise GateOverlapError('Duration error! Gates overlap in time with those in tick %s!' % tick.time)

        active_qudits_end = self.tick.circuit._active_qudits_end

        # Search for end times that either lie withing (time, end) or run over it...

        for q in qudits:

            # Search for end points within the duration
            try:
                # Find the first gate that ends within the duration of the current gate
                for _ in active_qudits_end[q].irange(minimum=time_begin, maximum=time_end, inclusive=(False, True)):
                    raise GateOverlapError('Another gate that comes before this one has a duration that overlaps this '
                                           'one!')

                # Find the first gate on this qudit that ends after this gate.
                for key in active_qudits_end[q].irange(minimum=time_end, inclusive=(False, True)):
                    if active_qudits_end[q][key] < time_end:
                        raise GateOverlapError('Another gate that comes before this one has a duration that overlaps '
                                               'this one!')
                    break

            except KeyError:
                pass

            ends = active_qudits_end.setdefault(q, SortedDict())
            ends[time_end] = time_begin

    def __str__(self) -> str:
        return "<Gate %s params=%s qudits: %s>" % (self.symbol, self.params, self.locations)
