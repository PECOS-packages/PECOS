# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2019 Ciarán Ryan-Anderson
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
Contains the class:
    Resource, Gate, Circuit, and Tick
"""

from typing import Any, Union, Collection, Sequence, Set, Optional, Tuple, List, Generator
from sortedcontainers import SortedDict
from ..types import GateLocations, Time

from .. import SymbolLibrary as CircuitLibrary

# TODO: !!!!!!! HOW TO EFFICIENTLY MAP INTERNAL RESOURCE IDS TO EXTERNAL WHILE SIMULATING OR GENERATING ERRORS.

# TODO: !!!!! MAKE SURE `ADD` WORKS AND CHECK SPACETIME OVERLAP
# TODO: verify TIME OVERLAP duration = 0 works/makes sense


class Resource(object):
    """
    A class that represents a resource such as a qubit, qutrit, ...

    Attributes:
        id (int): A unique id identifying the resource.
        type (str): A string indicating the type of resource.
        activity_begins:
        activity_ends:
        metadata (Dict[str, Any]): Any extra information about the resource.

    """

    def __init__(self,
                 resource_id: int,
                 resource_type: str = 'Qubit',
                 **metadata: Any) -> None:

        self.id = resource_id
        self.type = resource_type
        self.metadata = metadata

    def __int__(self):
        return self.id


class Gate(object):
    """
    A class to represent the basic structure of a gate (something that acts on resources).

    Attributes:
        resources (List[Resource]): The resources acted on by the gate.
        metadata (Dict[str, Any]): Any additional information supplied about this gate, e.g. angle of rotation.
        duration (Time): How long it takes to apply this gate. Default is 0.
            TODO: should the default be 1?
        size (int): The number of resources the gate acts on. Note, this is actually a method.

    TODO:
        Allow for smart resizing of resources? In circuits, we might want to add or subtract elements... dyanmically
        or should the resource size be fixed at the beginning? You might also want to reduce size with discard...
        If dynamically sizing up... then the size of resources has to equal the max id... Will have to request more
        resources...

    """

    def __init__(self, **metadata: Any) -> None:

        self.resources = []
        self.metadata = metadata
        self.duration = metadata.get('duration', 0)

    @property
    def size(self) -> int:
        """Returns the number of resources this circuit acts on."""
        return len(self.resources)

    def set_size(self, size: Union[int, Sequence['Resource']]) -> None:
        """
        Method to set size of circuit and identify the type of resources this gate acts on.

        Args:
            size (Union[int, List['Resource']]): The number of resources requested.

        Returns:
            None

        """

        n = len(self.resources)

        if isinstance(size, int):
            self.resources.extend([Resource(i+n) for i in range(size)])

        elif isinstance(size, Sequence):

            for i, res in enumerate(size):
                res.id = i + n
                self.resources.append(res)

        else:
            raise Exception('`size` input not accepted!')
        

class Circuit(Gate):
    """
    An gate that is a collection of other, smaller gates.

    Attributes:
        ticks(SortedDict[Time, Tick]): The a sorted dictionary of Ticks. The keys are the times when the ticks occur.
            The values are `Ticks`.
        resource_activity (Dict[int, SortedDict[Time, Time]]): A dictionary where the keys are resource ids and the
            values are `SortedDicts`. For these `SortedDicts`, each key is the time when the resource is acted on by a
            `Gate` and the value is the time when the gate stops acting on the resource.
        library (CircuitLibrary): A library of circuits storing those circuits that are associated with symbols. This
            allow one to add a gate via symbols rather than inputting circuit instances.

    Notes:
        This class inherits the attributes `resources`, `metadata`, `duration`, and `size` from the parent class `Gate`.
        The method `set_size` is also inherited from the parent class `Gate`.
    """

    def __init__(self,
                 library: Optional[CircuitLibrary] = None,
                 **metadata: Any) -> None:

        super(Circuit, self).__init__(**metadata)

        self.ticks = SortedDict()
        self.resource_activity = {}  # resource_id -> SortedDict[time_start, time_end]
        self.library = library

        if self.library is None:
            self.library = CircuitLibrary()

    @property
    def duration(self):
        """
        Returns the duration of the sequence.

        Returns:
            Time

        """
        # Get the duration of the last tick.
        time, tick = self.ticks.peekitem(-1)
        return time + tick.durations.peekitem(-1)[0]

    def add(self,
            gate: Union[Gate, str],
            locations: GateLocations,
            time: Time,
            **metadata: Any) -> None:
        """
        Adds a `Circuit` to the sequence at time `time`. Applies map.

        Args:
            gate (Union[Gate, str]): Either the circuit/gate instance (Gate) or the symbol (string) that
                identifies the circuit/gate in the `library`.
            locations (Locations): The set of resource ids (or set of tuple of ids) that the circuit/gate is applied to.
            time (Time): When the circuit/gate is applied.

        Returns:
            None

        """

        if isinstance(gate, str):
            symbol = gate
            gate = self.library.get(symbol, metadata)

            if gate is None:
                raise Exception('The circuit associated with the symbol "%s" could not be found or constructed!' %
                                symbol)
        else:
            if metadata:
                raise Exception('`metadata` is only allowed if the circuit is specified by a symbol (str)!')
                # otherwise the metadata should be stored in the circuit instance.

        # TODO: !!! should gate without str be registered to the library?  (symbol could be the hash)

        tick = self.request_tick(time)

        tick.add(gate, locations)

    def request_tick(self, time: Time) -> 'Tick':
        """
        Gets an existing tick or creates a new one.

        Args:
            time (Time): Time when the tick starts.

        Returns:
            Tick

        """

        if time in self.ticks:
            tick = self.ticks[time]
        else:
            tick = Tick(time, self)
            self.ticks[time] = tick

        return tick

    def discard(self,
                locations: GateLocations,
                time: Time) -> None:
        pass

    def iter_ticks(self,
                   time_start: Optional[Time] = None,
                   time_end: Optional[Time] = None):
        pass

    def iter_gates(self,
                   time_start: Optional[Time] = None,
                   time_end: Optional[Time] = None):
        pass

    def iter__physical_gates(self,
                             time_start: Optional[Time] = None,
                             time_end: Optional[Time] = None,
                             external_locations=None):
        pass

    def is_active(self,
                  resource_id: Union[int, Collection[int]],
                  time: Time,
                  duration: Time = 0) -> bool:
        """
        Whether the `resource_ids` are active during the time period given.

        Args:
            resource_id (Union[int, Collection[int]]): A resource id or collection of resource ids.
            time (Time): Initial time to check if resources are active.
            duration (Time): How long after `time` to check whether a resource is active.

        Returns:
            bool

        """
        """ Returns whether the resource is active at the time and duration given."""

        if isinstance(resource_id, int):
            resource_id = {resource_id}

        for rid in resource_id:

            activity = self.resource_activity[rid]

            after_index = activity.bisect_left(time)
            before_index = after_index - 1

            if before_index >= 0 and activity.peekitem(before_index)[1] > time:
                return True

            elif after_index < len(activity) and activity.peekitem(after_index)[0] < time + duration:
                return True

        return False

    def active_resources(self,
                         time: Time,
                         duration: Time = 0) -> Set[int]:
        """Returns the set of resource ids of the resources that are active at the time and duration given."""

        active = set()

        for rid in range(len(self.resources)):
            if self.is_active(rid, time, duration):
                active.add(rid)

        return active


class Tick(object):
    """
    A class representing a collection of all blocks that start at the same time.

    Attributes:
        time (Time):
        circuit (Gate):
        durations (SortedDict[Time, Tick]):
        resource_ids (Set[int]):

    TODO:
        get when the resources are inactive
    """

    def __init__(self,
                 time: Time,
                 circuit: 'Circuit') -> None:

        self.time = time
        self.circuit = circuit
        self.durations = SortedDict()  # duration -> Gate -> Set[Locations]
        self.resource_ids = set()

    def add(self,
            gate: Gate,
            locations: GateLocations) -> None:

        new_resource_ids = self.validate(gate, locations)
        # validate size of gate location, set gate size, id type, spacial overlap < duration overlap
        # TODO: we can capture the "spacial overlap" with the "durational" overlap...

        gates = self.durations.setdefault(gate.duration, {})
        locations = gates.setdefault(gate, set())
        locations.add(locations)

    def discard(self, locations: GateLocations) -> None:
        # match set(location) to set(location)...
        # see what QC does...
        pass

    def iter_gates(self) -> Generator[Gate, None, None]:
        """
        Generator yielding the `Gates` stored in this `Tick`.

        Returns:
            Circuit

        """

        for circuit in self.durations.values():
            yield circuit

    def iter_physical_gates(self,
                            time_start: Optional[Time] = None,
                            time_end: Optional[Time] = None,
                            inclusive: Tuple[bool, bool] = (True, True),
                            reverse: bool = False) -> Generator['Gate', None, None]:

        # external_locations -> external_locations -> internal_locations
        pass

    def add_resource(self, resource_ids):
        for rid in resource_ids:
            if rid in self.circuit.resources:
                pass

    def validate_duration(self, gate, resource_ids):

        time = self.time
        duration = gate.duration

        for rid in resource_ids:
            if rid in self.circuit.resources:
                pass
            else:
                pass

    def validate(self, gate: Gate, locations: GateLocations):
        """
        Validates the format of the Locations, checks that a resource hasn't already been acted on during this tick.

        Args:
            gate:
            locations:

        Returns:

        """

        if not isinstance(locations, set):
            raise Exception('`locations` must be a set.')

        resource_ids = set()

        self._verify_gate_size(gate, locations)

        if gate.size == 1:

            self._validate_int_locations(locations, resource_ids)

        else:

            for loc in locations:
                if len(loc) != gate.size:
                    raise Exception('Number of resource ids does not match gate size!')

                if not isinstance(loc, tuple):
                    raise Exception('Location must be a tuple!')

                self._validate_int_locations(loc, resource_ids)

        if self.resource_ids & resource_ids:
            overlap = self.resource_ids & resource_ids
            raise Exception('Resources %s has already be acted on during this tick!' % overlap)

        # check duration overlap
        time = self.time
        duration = gate.duration
        has_active = self.circuit.is_active(resource_ids, time, duration)
        if has_active:
            raise Exception('Gate(s) acting on these resources have a time interval that overlaps with the time '
                            'interval of this gate.')

        # record the resource activities.
        for rid in resource_ids:
            self.circuit.resource_activity[rid][time] = time + duration

        self.resource_ids.update(resource_ids)

        return resource_ids

    @staticmethod
    def _verify_gate_size(gate, locations):

        # check if size has been set.
        if gate.size == 0:

            # We will assume that the gate input size is equal to the size of the first location tuple encountered.
            loc = locations.pop()

            if isinstance(loc, int):
                gate.set_size(1)
            elif isinstance(loc, tuple):
                gate.set_size(len(loc))
            else:
                raise Exception('Location format not valid.')

            locations.add(loc)

    @staticmethod
    def _validate_int_locations(locations, resource_ids):

        for rid in locations:

            if not isinstance(rid, int):
                raise Exception('Resource id not an integer! Instead type %s.' % type(rid))

            if rid in resource_ids:
                raise Exception('Resource %s has already been acted on.' % rid)
            else:
                resource_ids.add(rid)
