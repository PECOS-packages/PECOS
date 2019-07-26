from typing import Any, Optional, Union, Tuple, Set, Sequence, Generator
from collections import defaultdict
from sortedcontainers import SortedDict
from .. import SymbolLibrary

GateLocation = Union[int, Tuple[int, ...]]


class ResourceType(object):
    """Class representing a informantion resource type."""
    def __init__(self):
        pass


class QuantumResource(ResourceType):
    """A quantum resource type."""
    def __init__(self):
        super().__init__()


class Qubit(QuantumResource):
    """A qubit resource type."""
    def __init__(self):
        super().__init__()


class Resource(object):
    """
    Represents a resource of a gate such as a qubit, qutrit, ...

    Attributes:
        rtype:
        **params:
    """

    def __init__(self,
                 rtype: Optional[Any] = None,
                 **params: Any) -> None:

        if rtype is None:
            rtype = Qubit

        self.rtype = rtype
        self.params = params


class Gate(object):
    """
    Represents a gate that acts on resource, for example, a quantum operation or, more abstractly, a quantum circuit.
    """

    def __init__(self,
                 **params: Any) -> None:

        self.resources = []  # TODO: Meant for specifying Type
        self.duration = params.get('duration', 0)
        self.params = params

    def add_resources(self,
                      resource_sequence: Sequence[Resource]) -> None:
        """Adds resources being acted on."""

        self.resources.extend(resource_sequence)

    def set_size(self):
        pass
        # TODO: DO THIS!

    def __len__(self) -> int:
        """The number of resources the gate acts on."""
        return len(self.resources)


class GateSet(object):
    """
    A set of gates acting in parallel.

    Attributes:
        gate_locations:
        resource_ids:
        durations:
    """

    # TODO: Should be able to this of this as a gate... try applying Gate...

    def __init__(self, symbol_library=None):
        self.gate_locations = defaultdict(set)
        self.resource_ids = set()
        self.durations = SortedDict()  # the time it takes to apply all of these gates...

        if symbol_library is None:
            symbol_library = SymbolLibrary()

        self.symbol_library = symbol_library

    @property
    def duration(self):
        """
        The maximum duration of all gates in this `GateSet`.

        Note that other GateSets can still occur before the end of this duration provided that no gates overlap in time.
        """
        return self.durations.peekitem(-1)[0]

    def add(self,
            symbol: str,
            locations: Set[GateLocation], **params):
        """
        Adds a gate to the `GateSet`.

        Args:
            symbol:
            locations:
            **params:

        Returns:

        """

        gate: Gate = self.symbol_library.get(symbol, params)
        self.gate_locations[symbol].update((gate, locations))
        self._register_locations(gate, locations)

        gates = self.durations.setdefault(gate.duration, set())
        gates.add((symbol, gate))

    def _register_locations(self,
                            gate: Gate,
                            locations: Set[GateLocation]):
        """Adds new resource ideas to `resource_ids` and checks that no resource has been acted on more than once."""

        if len(gate) == 0:
            self._set_gate_size(gate, locations)

        if len(gate) == 1:

            self._validate_loc_ids(locations)

        else:

            for loc in locations:

                if len(loc) != len(gate):
                    raise Exception('Number of resource ids does not match gate size!')

                if not isinstance(loc, Sequence):
                    raise Exception('Location must be a Sequence!')

                self._validate_loc_ids(loc)

    @staticmethod
    def _set_gate_size(gate, locations):

        # We will assume that the gate input size is equal to the size of the first location tuple encountered.
        loc = locations.pop()

        if isinstance(loc, int):
            gate.set_size(1)
        elif isinstance(loc, tuple):
            gate.set_size(len(loc))
        else:
            raise Exception('Location format not valid.')

        locations.add(loc)

    def _validate_loc_ids(self, locations):

        for rid in locations:

            if not isinstance(rid, int):
                raise Exception('Resource id not an integer! Instead type %s.' % type(rid))

            if rid in self.resource_ids:
                raise Exception('Resource %s has already been acted on.' % rid)
            else:
                self.resource_ids.add(rid)

    def discard(self, locations):
        pass
        # TODO: Find matching locations... remove

    def iter_gates(self, recursive=False) -> Generator[Gate, None, None]:
        """
        Generator yielding the `Gates` stored in this `Tick`.

        Returns:
            str, Gate, Set[GateLocation]

        """

        if not recursive:
            for symbol, (gate, locations) in self.gate_locations.items():
                yield symbol, gate, locations
        else:
            # TODO: give the gates all the way to the physical level.
            pass

    def __len__(self) -> int:
        """The number of resources the gate acts on."""
        return len(self.resource_ids)
