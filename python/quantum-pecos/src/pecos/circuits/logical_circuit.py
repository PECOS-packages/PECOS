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

"""Provides class to represent a logical circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn, Protocol

from pecos.circuits.quantum_circuit import (
    QuantumCircuit,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pecos.circuits.quantum_circuit import (
        GateDict,
        JSONValue,
        LocationSet,
        ParamGateCollection,
    )


class LogicalGateProtocol(Protocol):
    """Protocol for logical gate objects."""

    qecc: Any
    circuits: list[Any]


class LogicalCircuit(QuantumCircuit):
    """Data structure used to represent a logical circuit."""

    def __init__(
        self,
        layout: dict[Any, Any] | None = None,
        *,
        suppress_warning: bool = True,
        **params: JSONValue,
    ) -> None:
        """Initialize a LogicalCircuit.

        Args:
        ----
            layout: Optional layout dictionary defining the qudit arrangement.
            suppress_warning: If True, suppresses warnings about missing layout.
            **params: Additional parameters passed to the parent QuantumCircuit.
        """
        self.layout = layout
        if self.layout is not None:
            self.qudit_set = set(self.layout.keys())
            self.num_qudits = len(self.qudit_set)
        else:
            self.num_qudits = None
            self.qudit_set = None

        self.suppress_warning = suppress_warning

        # Store logical gates separately (they have .circuits attribute that iter_ticks needs)
        self._logical_gates: list[list[tuple[LogicalGateProtocol, LocationSet, dict[str, Any]]]] = []

        super().__init__(**params)

    def append(
        self,
        logical_gate: LogicalGateProtocol | GateDict,
        gate_locations: LocationSet | None = None,
        **params: JSONValue,
    ) -> None:
        """Append a logical gate to the circuit.

        Args:
        ----
            logical_gate: The logical gate to append, either a LogicalGateProtocol or GateDict.
            gate_locations: Set of locations where the gate should be applied.
            **params: Additional parameters for the gate.
        """
        # Store the logical gate in our separate storage (for iter_ticks)
        if gate_locations is None and not isinstance(logical_gate, dict):
            locations: LocationSet = frozenset([None])
        else:
            locations = gate_locations or set()

        # Add a new tick to logical gates storage
        self._logical_gates.append([(logical_gate, locations, params)])  # type: ignore[arg-type]

        # Also call parent's append to maintain tick count
        # Use a dummy symbol since logical gates aren't actual physical gates
        self.add_ticks(1)

        if self.layout is None:
            qecc = logical_gate.qecc

            self.layout = qecc.layout
            self.num_qudits = qecc.num_qudits
            self.qudit_set = qecc.qudit_set

            if not self.suppress_warning:
                print("No layout given. The layout of the first QECC is assumed.")

        elif isinstance(logical_gate, dict):
            for gate in logical_gate:
                # TODO: Probably not the best... need total number of qudits
                qecc = gate.qecc
                if self.num_qudits < qecc.num_qudits:
                    msg = (
                        f"Number of QECC qudits greater than those in assumed layout "
                        f"({self.num_qudits} < {qecc.num_qudits})"
                    )
                    raise Exception(msg)

                if self.qudit_set is None:
                    msg = "Qudit set should be set!"
                    raise Exception(msg)
        else:
            qecc = logical_gate.qecc
            if self.num_qudits < qecc.num_qudits:
                msg = (
                    f"Number of QECC qudits greater than those in assumed layout "
                    f"({self.num_qudits} < {qecc.num_qudits})"
                )
                raise Exception(msg)

            if self.qudit_set is None:
                msg = "Qudit set should be set!"
                raise Exception(msg)

    @staticmethod
    def update(
        _symbol: str | GateDict,
        _locations: LocationSet | None = None,
        _tick: int = -1,
        *,
        _emptyappend: bool = False,
        **_params: JSONValue,
    ) -> NoReturn:
        """Update operation (not implemented for logical circuits).

        Args:
            _symbol: Gate symbol or dictionary.
            _locations: Target locations.
            _tick: Time tick.
            _emptyappend: Whether to append empty operations.
            **_params: Additional parameters.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        msg = "!!!"
        raise NotImplementedError(msg)

    def discard(self, _locations: LocationSet, _tick: int = -1) -> NoReturn:
        """Discard operations at specified locations (not implemented).

        Args:
            _locations: Locations to discard operations from.
            _tick: Time tick to discard from.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        msg = "!!!"
        raise NotImplementedError(msg)

    def items(
        self,
        tick: int | None = None,
    ) -> Iterator[tuple[Any, LocationSet, dict[str, Any]]]:
        """Return an iterator over logical gates.

        Args:
            tick: If specified, return only gates from that tick.
                  If None, return gates from all ticks.

        Yields:
            Tuples of (logical_gate, locations, params).
        """
        if tick is not None:
            # Return gates from specific tick
            if 0 <= tick < len(self._logical_gates):
                yield from self._logical_gates[tick]
        else:
            # Return gates from all ticks
            for tick_gates in self._logical_gates:
                yield from tick_gates

    def iter_ticks(self) -> Iterator[tuple[Any, tuple[int, int, int], dict[str, Any]]]:
        """An iterator for looping over the various quantum circuits comprising this data structure."""
        for logical_tick in range(len(self)):
            for logical_gate, _, _ in self.items(tick=logical_tick):
                for instr_index, instr_circuit in enumerate(logical_gate.circuits):
                    params = {
                        "logical_circuit_params": self.metadata,
                        "gate": logical_gate,
                    }
                    params.update(instr_circuit.metadata)

                    for tick in range(len(instr_circuit)):
                        tick_gates = instr_circuit[tick]
                        time = (logical_tick, instr_index, tick)
                        yield tick_gates, time, params

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all logical gates in the circuit."""
        for gate, _, _ in self.items():
            yield gate

    def __str__(self) -> str:
        """Return string representation of the logical circuit."""
        ticks_repr = []
        for _tick_idx, tick_gates in enumerate(self._logical_gates):
            gates_str = ", ".join(f"{getattr(gate, 'symbol', str(gate))}: {locs}" for gate, locs, _ in tick_gates)
            ticks_repr.append(f"Tick({{{gates_str}}})")
        return f"LogicalCircuit([{', '.join(ticks_repr)}])"

    def __repr__(self) -> str:
        """Return detailed string representation of the logical circuit."""
        return self.__str__()

    def __len__(self) -> int:
        """Return the number of logical ticks in the circuit."""
        return len(self._logical_gates)

    def __getitem__(self, tick: int | tuple[int, int, int]) -> ParamGateCollection:
        """Returns tick when instance[index] is used.

        Args:
        ----
            tick(int): Tick index of the circuit.
        """
        if isinstance(tick, int):
            return super().__getitem__(tick)
        logical_tick, _, _ = tick
        return self[logical_tick]
