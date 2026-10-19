# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Static operation packing metrics with explicit selection and barrier policy.

These metrics describe selected operation batches, not executable schedules or
bounds on conditional execution. Compilation, durations, and device policy are
outside this module.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection

    from pecos.circuits import QuantumCircuit


@dataclass(frozen=True)
class ParallelLayer:
    """Disjoint operation supports sharing one condition, copied from the input."""

    locations: frozenset[tuple[int, ...]]
    condition: Any = None

    @property
    def qubits(self) -> frozenset[int]:
        """Qubits occupied by the selected operations."""
        return frozenset(q for location in self.locations for q in location)


@dataclass(frozen=True)
class ParallelismSummary:
    """Averages over nonempty selected layers in one static scenario."""

    layer_count: int
    operation_count: int
    mean_operations: float
    mean_active_qubits: float
    required_qubits: int

    def density(self, num_qubits: int) -> float:
        """Return mean active-qubit fraction; an empty zero-qubit circuit is zero."""
        if isinstance(num_qubits, bool):
            msg = "num_qubits must be a nonnegative integer, not bool"
            raise TypeError(msg)
        width = index(num_qubits)
        if width < self.required_qubits or width < 0:
            msg = f"num_qubits must be nonnegative and at least {self.required_qubits}, got {width}"
            raise ValueError(msg)
        return self.mean_active_qubits / width if width else 0.0


def _summarize_layers(layers: tuple[ParallelLayer, ...]) -> ParallelismSummary:
    """Summarize already packed layers without changing their placement."""
    nonempty = [layer for layer in layers if layer.locations]
    count = sum(len(layer.locations) for layer in nonempty)
    active = sum(len(layer.qubits) for layer in nonempty)
    required = max((q + 1 for layer in nonempty for q in layer.qubits), default=0)
    return ParallelismSummary(
        layer_count=len(nonempty),
        operation_count=count,
        mean_operations=count / len(nonempty) if nonempty else 0.0,
        mean_active_qubits=active / len(nonempty) if nonempty else 0.0,
        required_qubits=required,
    )


@dataclass(frozen=True)
class CircuitParallelism:
    """Packed supports and all-selected/unconditional-only scenario summaries.

    The unconditional scenario removes conditional layers without repacking.
    Neither scenario is a minimum/maximum bound or an execution probability.
    """

    layers: tuple[ParallelLayer, ...]

    @property
    def all_operations(self) -> ParallelismSummary:
        """Include every selected layer regardless of its condition."""
        return _summarize_layers(self.layers)

    @property
    def unconditional(self) -> ParallelismSummary:
        """Exclude layers whose condition is truthy, without repacking."""
        return _summarize_layers(tuple(layer for layer in self.layers if not layer.condition))


def circuit_parallelism(
    circuit: QuantumCircuit,
    *,
    gate_symbols: Collection[str] | None = None,
    compact: bool = True,
    ignored_operations_are_barriers: bool = True,
) -> CircuitParallelism:
    """Pack selected operations while preserving per-qubit and condition order.

    Iterate the circuit's operation batches in order. With ``compact=False``,
    each selected batch becomes one layer; this is not the original tick depth.
    Otherwise, move each operation back across disjoint layers with equal
    conditions. Unequal conditions and qubit conflicts stop movement.

    ``gate_symbols=None`` selects every symbol. Unselected and zero-qubit
    batches are barriers by default. Setting ``ignored_operations_are_barriers``
    to false explicitly projects them out: the resulting metric ignores their
    dependencies. Empty location batches never contribute to the averages.
    Empty results and scenarios have zero averages. This function never mutates
    circuit locations or metadata and does not evaluate classical conditions.
    """
    if isinstance(gate_symbols, str):
        msg = "gate_symbols must be a collection of symbols, not a single string"
        raise TypeError(msg)
    symbols = None if gate_symbols is None else frozenset(gate_symbols)
    layers: list[ParallelLayer] = []
    barrier = 0
    for symbol, locations, metadata in circuit.items():
        if not locations or (symbols is not None and symbol not in symbols):
            if ignored_operations_are_barriers:
                barrier = len(layers)
            continue
        supports = frozenset((location,) if isinstance(location, int) else tuple(location) for location in locations)
        condition = deepcopy(metadata.get("cond"))
        if not compact:
            layers.append(ParallelLayer(supports, condition))
            continue
        for support in sorted(supports):
            target = len(layers)
            for position in range(len(layers) - 1, barrier - 1, -1):
                layer = layers[position]
                if layer.condition != condition or layer.qubits.intersection(support):
                    break
                target = position
            if target == len(layers):
                layers.append(ParallelLayer(frozenset({support}), condition))
            else:
                layer = layers[target]
                layers[target] = ParallelLayer(layer.locations | {support}, layer.condition)
    return CircuitParallelism(tuple(layers))
