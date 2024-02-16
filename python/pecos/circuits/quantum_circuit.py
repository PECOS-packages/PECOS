# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Contains the class ``QuantumCircuit``, which is used to represent quantum circuits."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import MutableSequence
from typing import Any, NamedTuple

from pecos import __version__
from pecos.circuits import qc2phir


class QuantumCircuit(MutableSequence):
    """A representation of a quantum circuit.

    Similar to [{gate_symbol: set of qudits, ...}, ...] where each element is a time step in which gates act in
    parallel.

    """

    def __init__(self, circuit_setup=None, **metadata) -> None:
        """Args:
            circuit_setup (None, int, list of dict):
            **params: Quantum circuit parameters.

        Attributes:
            self._ticks(list of dict): A list of parallel gates. Each element is a dictionary of gate symbol => gate
            set.
            This gate dictionary is assumed to be a collection of gates that act in parallel on the qudits.
            self.active_qudits(list): If `check_overlap` == True then ``active_qudits`` will be tracked; otherwise,
            ``active_qudits`` will not be tracked.
        """
        self._gates_class = ParamGateCollection
        self._ticks_class = list
        self._ticks = self._ticks_class()
        self.metadata = metadata
        self.qudits = set()
        # TODO: If all the gates on a qudit are discarded... then the qudit will not be removed from this set... fix

        if "tracked_qudits" in metadata:
            msg = "error"
            raise Exception(msg)

        if circuit_setup is not None:
            self._circuit_setup(circuit_setup)

    @property
    def active_qudits(self):
        """Returns the active_qudits of all the ticks.

        Returns:

        """
        """
        if flat:
            active = {}
            for gates in self._ticks:
                active.update(gates.active_qudits)

        else:
        """

        active = []
        for gates in self._ticks:
            active.append(gates.active_qudits)
        return active

    def append(self, symbol, locations=None, **params):
        """Adds a new gate=>gate_locations (set) pair to the end of ``self.gates``.

        Args:
            symbol(str or dict): A gate dictionary of gate symbol => set of qudit ids or tuples of qudit ids
            locations:

        Example:
            >>> quantum_circuit = QuantumCircuit()
            >>> quantum_circuit.append({"X": {0, 1, 5, 6}, "Z": {7, 8, 9}})
            >>> quantum_circuit.append("X", {0, 1, 3, 7})

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
        if emptyappend and len(self) == 0:
            self.add_ticks(1)

        self._ticks[tick].add(symbol, locations, **params)

    def discard(self, locations, tick=-1):
        """Discards ``locations`` for tick ``tick``.

        Args:
            locations:
            tick:

        Returns:

        """
        self._ticks[tick].discard(locations)

    def add_ticks(self, num_ticks) -> None:
        """Makes sure that QuantumCircuit has at least `num_tick` number of ticks. If the number of ticks in the data
        structure is less than `num_tick` then empty ticks are appended until the total number of ticks == `num_ticks`.

        Args:
            num_ticks:

        Returns: Nothing
        """
        for _ in range(num_ticks):
            self.append({})

    def items(self, tick=None):
        """An iterator through all gates/qudits in the quantum circuit.

        If ``tick`` is not None then it will iterate over only the qudits/qudits in the corresponding tick.

        Args:
            tick:

        Returns:

        """
        if tick is None:
            for gates in self._ticks:
                for symbol, locations, params in gates.items():
                    yield symbol, locations, params

        else:
            for symbol, locations, params in self._ticks[tick].items():
                yield symbol, locations, params

    def iter_ticks(self):
        for tick in range(len(self)):
            gates = self[tick]
            yield gates, tick, self.metadata
            # TODO: note this is the circuit params
            # TODO: need something like: params {logical_circuit: ..., gate: ..., qecc: ...}

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

    def _circuit_setup(self, circuit_setup):
        if isinstance(circuit_setup, int):
            # Reserve ticks
            self.add_ticks(circuit_setup)

        else:
            # Build circuit from other description (a shallow copy).
            for other_tick in circuit_setup:
                self.append(other_tick)

    def to_json_str(self) -> str:
        """Creates a json str representation of the QuantumCircuit listing all the gates. It does not preserve ticks or
        parallel gating of different gate types.
        """
        metadata = self.metadata

        gates = []
        for sym, qubits, meta in self.items():
            gate = {
                "sym": sym,
                "qubits": list(qubits),
                "metadata": meta,
            }
            gates.append(gate)

        prog = {
            "prog_type": "PECOS.QuantumCircuit",
            "PECOS_version": str(__version__),
            "prog_metadata": metadata,
            "gates": gates,
        }

        return json.dumps(prog)

    @staticmethod
    def _fix_json_meta(meta):
        """Fix some of the type issues for converting json rep back to a QuantumCircuit."""
        if "var_output" in meta:
            meta["var_output"] = {
                int(k): tuple(v) for k, v in meta["var_output"].items()
            }
        return meta

    @classmethod
    def from_json_str(cls, qc_json) -> QuantumCircuit:
        """Converts a json str that represents a QuantumCircuit back into a QuantumCircuit object."""
        qc_dict = json.loads(qc_json)

        qc_meta = qc_dict["prog_metadata"]
        qc = QuantumCircuit(**qc_meta)

        for gate_dict in qc_dict["gates"]:
            sym = gate_dict["sym"]

            qubits = gate_dict["qubits"]
            qubits = (
                set(qubits)
                if qubits and isinstance(qubits[0], int)
                else {tuple(q) for q in qubits}
            )

            meta = gate_dict["metadata"]
            meta = cls._fix_json_meta(meta)

            qc.append(sym, qubits, **meta)

        return qc

    def to_phir_dict(self) -> dict:
        """Converts this QuantumCircuit into the PHIR format as a dict."""
        return qc2phir.to_phir_dict(self)

    def to_phir_json(self) -> str:
        """Converts this QuantumCircuit into the PHIR/JSON format."""
        return qc2phir.to_phir_json(self)

    def __getitem__(self, tick):
        """Returns tick when instance[index] is used.

        Args:
            tick(int): Tick index of ``self._ticks``.

        Returns:

        """
        return self._ticks[tick]

    def __setitem__(self, tick, item) -> None:
        gate_dict, params = item
        self._ticks[tick] = self._gates_class(self, gate_dict, **params)

    def __len__(self) -> int:
        """Used to return number of ticks when len() is used on an instance of this class.

        Returns:

        """
        return len(self._ticks)

    def __delitem__(self, tick) -> None:
        """Used to delete a tick. For example: del instance[key].

        Args:
            tick:

        Returns:

        """
        self._ticks[tick] = self._gates_class(self)

    def __str__(self) -> str:
        """String returned when a string representation is requested. This occurs during printing.

        Returns:

        """
        str_list = []
        for gates in self._ticks:
            tick_list = []
            for symbol, locations, params in gates.items():
                if len(params) == 0:
                    tick_list.append(f"'{symbol}': {locations}")
                else:
                    tick_list.append(f"'{symbol}': loc: {locations} - params={params}")
            tick_list = ", ".join(tick_list)
            str_list.append("{%s}" % tick_list)

        if self.metadata:
            return "QuantumCircuit(params={}, ticks=[{}])".format(
                str(self.metadata),
                ", ".join(str_list),
            )
        else:
            return "QuantumCircuit([%s])" % ", ".join(str_list)

    def __repr__(self) -> str:
        """Returns:"""
        return self.__str__()

    def __copy__(self):
        """Create a shallow copy."""
        newone = QuantumCircuit()
        newone.metadata = dict(self.metadata)
        newone._ticks = self._ticks_class(self._ticks)  # noqa: SLF001

        return newone

    def copy(self):
        """Create a shallow copy of the circuit.

        Returns:

        """
        return self.__copy__()

    def __iter__(self):
        return self.items()


class ParamGateCollection:
    """Data structure for a tick."""

    class Gate(NamedTuple):
        symbol: str
        params: Any
        locations: set[int | tuple[int]]

    def __init__(self, circuit, symbol=None, locations=None, **params) -> None:
        self.circuit = circuit
        self.metadata = circuit.metadata
        self.active_qudits = set()
        self.symbols = defaultdict(list)

        self.add(symbol, locations, **params)

    def add(self, symbol, locations=None, **params):
        """Args:
            symbol:
            locations:

        Returns:

        """
        # If locations is None then assume symbol is a gate_dict.
        gate_dict = symbol if locations is None else {symbol: locations}

        self._verify_qudits(gate_dict)

        for gate_symbol, gate_locations in gate_dict.items():
            # if gate_locations:  #TODO: Why was this here?

            for gate in self.symbols[gate_symbol]:
                if params == gate.params:
                    gate.locations.update(gate_locations)
                    break
            else:
                self.symbols[gate_symbol].append(
                    self.Gate(gate_symbol, params, gate_locations),
                )

        return self

    def discard(self, locations):
        """Remove gate locations.

        Args:
            locations:

        Returns:

        """
        for gate_list in self.symbols.values():
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
        for qudit_locations in gate_dict.values():
            for location in qudit_locations:
                # Make sure we can iterate over q.
                q_iter = (location,) if not isinstance(location, tuple) else location

                for qi in q_iter:
                    self.circuit.qudits.add(qi)

                    if qi in self.active_qudits:
                        raise Exception(
                            "Qudit %s has already been acted on by a gate!" % str(qi),
                        )
                    else:
                        self.active_qudits.add(qi)

    def items(self, tick=None):
        """Generator to return a dictionary-like iter.

        Returns:

        """
        for gate_symbol, gate_list in self.symbols.items():
            for gate in gate_list:
                yield gate_symbol, gate.locations, gate.params

    def __str__(self) -> str:
        tick_list = []
        for symbol, locations, params in self.items():
            if len(params) == 0:
                tick_list.append(f"'{symbol}': {locations}")
            else:
                tick_list.append(f"'{symbol}': loc: {locations} - params={params}")
        tick_list = ", ".join(tick_list)

        return "Tick({%s})" % tick_list

    def __repr__(self) -> str:
        return self.__str__()
