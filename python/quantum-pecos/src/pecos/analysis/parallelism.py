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


"""Static metrics for how densely selected circuit operations pack into layers.

The packing rules, and what these metrics do not measure, are described in
``docs/user-guide/circuit-parallelism.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from operator import index
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection

    from pecos.circuits import QuantumCircuit


@dataclass(frozen=True)
class ParallelLayer:
    """Disjoint operation supports sharing one condition copied from the circuit.

    Unhashable because ``condition`` is a mutable copy of the ``cond`` metadata.
    """

    locations: frozenset[tuple[int, ...]]
    condition: Any = None

    __hash__ = None

    @property
    def qubits(self) -> frozenset[int]:
        """Qubits occupied by the selected operations."""
        return frozenset(q for location in self.locations for q in location)


@dataclass(frozen=True)
class ParallelismSummary:
    """Averages over the selected layers in one static scenario."""

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
    count = sum(len(layer.locations) for layer in layers)
    active = sum(len(layer.qubits) for layer in layers)
    required = max((q + 1 for layer in layers for q in layer.qubits), default=0)
    return ParallelismSummary(
        layer_count=len(layers),
        operation_count=count,
        mean_operations=count / len(layers) if layers else 0.0,
        mean_active_qubits=active / len(layers) if layers else 0.0,
        required_qubits=required,
    )


@dataclass(frozen=True)
class CircuitParallelism:
    """Packed layers with summaries of two static scenarios.

    Neither scenario is a bound or a probability. Unhashable because its layers are.
    """

    layers: tuple[ParallelLayer, ...]

    @cached_property
    def all_operations(self) -> ParallelismSummary:
        """Summary over every selected layer."""
        return _summarize_layers(self.layers)

    @cached_property
    def unconditional(self) -> ParallelismSummary:
        """Summary over layers without a truthy condition, without repacking."""
        return _summarize_layers(tuple(layer for layer in self.layers if not layer.condition))


def circuit_parallelism(
    circuit: QuantumCircuit,
    *,
    gate_symbols: Collection[str] | None = None,
    ignored_operations_are_barriers: bool = True,
) -> CircuitParallelism:
    """Pack selected operations as early as per-qubit and condition order allow.

    Each operation moves back across layers that share its condition and none of
    its qubits. ``gate_symbols=None`` selects every symbol. Unselected and
    zero-qubit operations fence later ticks unless
    ``ignored_operations_are_barriers`` is false. A tick's batches are packed in
    a canonical condition order, so insertion order within a tick does not
    matter. The circuit is not modified. Time is linear in the selected supports.
    """
    if isinstance(gate_symbols, str):
        msg = "gate_symbols must be a collection of symbols, not a single string"
        raise TypeError(msg)
    symbols = None if gate_symbols is None else frozenset(gate_symbols)
    layers: list[tuple[set[tuple[int, ...]], Any]] = []  # (supports, condition) per layer
    last_layer_on_qubit: dict[int, int] = {}
    trailing_run_start = 0  # First layer of the trailing run of layers that share one condition.
    trailing_condition: Any = None
    barrier = 0
    for tick in range(len(circuit)):
        tick_has_ignored_operation = False
        batches = sorted(circuit.items(tick), key=lambda batch: json.dumps(batch[2].get("cond"), sort_keys=True))
        for symbol, locations, metadata in batches:
            if not locations or (symbols is not None and symbol not in symbols):
                tick_has_ignored_operation = True
                continue
            condition = metadata.get("cond")
            for support in sorted((loc,) if isinstance(loc, int) else tuple(loc) for loc in locations):
                # Movement stops at the latest layer with a different condition or a shared qubit.
                if not layers or condition != trailing_condition:
                    trailing_run_start, trailing_condition = len(layers), condition
                target = max(barrier, trailing_run_start, *(last_layer_on_qubit.get(q, -1) + 1 for q in support))
                if target == len(layers):
                    layers.append((set(), condition))
                layer_supports, _ = layers[target]
                layer_supports.add(support)
                last_layer_on_qubit.update(dict.fromkeys(support, target))
        if tick_has_ignored_operation and ignored_operations_are_barriers:
            barrier = len(layers)
    return CircuitParallelism(tuple(ParallelLayer(frozenset(supports), condition) for supports, condition in layers))
