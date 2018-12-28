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

from collections.abc import MutableSequence
from sortedcontainers import SortedDict

# TODO: Make sure, when searching for duration the search is: [t, t+d)
# TODO: Consider making this backwards compatible with QuantumCircuit and giving warnings...


class QuantumCircuitDuration(MutableSequence):
    """
    An alternative to the QuantumCircuit class where time/ticks can be a floating point numbers and duration taken into
    consideration when determining if gates overlap.

    Attributes:
        params (dict): A dictionary to store extra information about the circuit (i.e., circuit parameters).
        ignore_duration (bool): Whether to consider duration such as when determining if gate overlap. This defaults to
        the simple case where duration is not consider. In this case, duration will be == 1.
        times (SortedDict): time -> {gates: Gates(), active: set()}
        active (dict): # qudit id -> (time, duration) Tracks when a qudit is active.
            # TODO: Need this since tracking this in `Gates`?
    """

    def __init__(self, circuit_rep=None, **params):
        """

        Args:
            circuit_rep:
            **params:
        """

        self.params = params
        self.ignore_duration = params.get('ignore_duration', True)
        self.times = SortedDict()
        self.active = {}

    def add(self, symbol, locations, time=-1, duration=1.0, **params):
        """

        Args:
            symbol (str):
            locations (set of ints OR set of tuple of ints): The ints or tuple of ints correspond to the qudit ids that
                the gates act on.
            time (real number [int, float, etc.]):
            duration (real number [int, float, etc.]):
            **params (dict):

        Returns:
            Nothing.

        """

        # see if the time already exists...
        # see if

        if time == -1:  # Add new time right after the old time
            time_dict = self.times.peekitem()  # Get last item
        else:
            time_dict = self.times[time]

    def update(self, gates, time=-1):
        """
        Similar to add, but allows multiple gate types to be added.

        Each gate type is indicates by the tuple:
            symbol, locations, duration (optional), param dict (optional))

        Args:
            gates:
            time (real number [int, float, etc.]):

        Returns:
            Nothing.

        """
        pass

    def discard(self, symbol, locations, time):
        """

        Args:
            symbol (str):
            locations (set of ints OR set of tuple of ints): The ints or tuple of ints correspond to the qudit ids that
                the gates act on.
            time (real number [int, float, etc.]):

        Returns:
            Nothing.

        """

        # TODO: Destroy Gates class if empty.
        pass

    def tick(self, index):
        """
        Gives the `Gates` instance corresponding to `index`.

        Args:
            index (int): Index in the `self.

        Returns:
            The `Gate` instance indexed by `index` within the `self.times` data structure.

        """
        return self.times.peekitem(index)[1]

    def items(self):
        pass

    def insert(self, time, item):
        pass

    def _circuit_setup(self):
        pass

    def __getitem__(self, time):

        pass

    def __setitem__(self, time, item):
        pass

    def __delitem__(self, time):
        pass

    def __len__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        """

        Returns:

        """

        return self.__str__()

    def __iter__(self):
        return self.items()


class Gates:
    """
    A collection of gates of the same type (Hadamard, X, Y, Z, etc.), parameters, time, and duration.

    Attributes:
        symbol (str):
        locations (set of ints OR set of tuple of ints): The ints or tuple of ints correspond to the qudit ids that
            the gates act on.
        time (real number [int, float, etc.]):
        duration (real number [int, float, etc.]):
        params (dict):
    """

    def __init__(self, symbol, locations, time, duration, **params):
        """

        Args:
            symbol (str): Symbol representing the gate.
            locations (set of ints OR set of tuple of ints): The ints or tuple of ints correspond to the qudit ids that
                the gates act on.
            time (real number [int, float, etc.]): Initial time when the gates are applied.
            duration (real number [int, float, etc.]): How long the gates are applied for.
            **params (dict):
        """

        self.symbol = symbol
        self.locations = set()
        self.locations.add(locations)
        self.time = time
        self.duration = duration
        self.params = params

    def add(self, locations):
        """

        Args:
            locations:

        Returns:

        """

        self.locations.add(locations)
        # TODO: Check that these locations can be added... otherwise... raise exception.

    def discard(self, locations):
        """

        Args:
            locations (set of ints OR set of tuple of ints): The ints or tuple of ints correspond to the qudit ids that
                the gates act on.

        Returns:

        """
        # TODO: Destroy Gates class if empty.
        pass
