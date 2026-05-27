# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Guppy-only slot ownership tracking for AST code generation.

This module is deliberately target-scoped. It tracks the Guppy local that
currently owns each logical SLR qubit slot while `ast/codegen/guppy.py`
emits source. It does not annotate AST nodes and does not model non-Guppy
codegens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Slot:
    """Logical qubit slot from an SLR allocator, such as `q[0]`."""

    allocator: str
    index: int

    def __str__(self) -> str:
        """Return a compact user-facing slot name."""
        return f"{self.allocator}[{self.index}]"


class SlotState(Enum):
    """Guppy ownership state for a logical qubit slot."""

    LIVE = auto()
    CONSUMED = auto()


@dataclass(frozen=True, slots=True)
class Binding:
    """Current Guppy local name and ownership state for one slot."""

    local: str
    state: SlotState


LinearitySnapshot = dict[Slot, Binding]


class LinearityError(Exception):
    """Raised when AST emission would produce unsound Guppy ownership."""


class GuppyLinearityState:
    """Track logical qubit slots while the Guppy emitter writes locals."""

    def __init__(self, bindings: Mapping[Slot, Binding]) -> None:
        """Create state from an explicit binding table in stable order."""
        self._order = tuple(bindings)
        self._bindings = dict(bindings)

    @classmethod
    def from_allocators(
        cls,
        allocators: Mapping[str, int],
        *,
        slot_locals: Mapping[Slot, str] | None = None,
    ) -> GuppyLinearityState:
        """Create live slot bindings for root QReg/QAlloc declarations.

        `slot_locals`, when provided, is the single namespace-wide
        slot-to-Guppy-local name table from `GuppyContext.slot_locals`
        (disambiguates the default `f"{allocator}_{index}"` against
        register names so the entry-unpack LHS does not shadow another
        declared register). When omitted, the default name is used --
        kept for callers that build a linearity table outside the main
        emitter (no register collision risk for those isolated paths).
        """
        bindings: dict[Slot, Binding] = {}
        for allocator, size in allocators.items():
            if size < 0:
                msg = f"Allocator {allocator!r} has negative size {size}"
                raise LinearityError(msg)
            for index in range(size):
                slot = Slot(allocator, index)
                local = slot_locals[slot] if slot_locals is not None and slot in slot_locals else f"{allocator}_{index}"
                bindings[slot] = Binding(local=local, state=SlotState.LIVE)
        return cls(bindings)

    def bindings(self) -> Iterable[tuple[Slot, Binding]]:
        """Iterate bindings in stable allocator/index order."""
        return ((slot, self._bindings[slot]) for slot in self._order)

    def binding(self, slot: Slot) -> Binding:
        """Return the current binding for a slot, including consumed slots."""
        self._require_known(slot)
        return self._bindings[slot]

    def status(self, slot: Slot) -> SlotState:
        """Return whether a slot is live or consumed."""
        return self.binding(slot).state

    def live(self, slot: Slot) -> str:
        """Return the live Guppy local for a slot, or raise if consumed."""
        binding = self.binding(slot)
        if binding.state is not SlotState.LIVE:
            msg = f"Slot {slot} is consumed and has no live Guppy local"
            raise LinearityError(msg)
        return binding.local

    def set_live(self, slot: Slot, local: str) -> None:
        """Record the current live owner for a slot; the name may be unchanged."""
        self._require_known(slot)
        self._bindings[slot] = Binding(local=local, state=SlotState.LIVE)

    def consume(self, slot: Slot) -> str:
        """Return the live local and mark the slot consumed; raise if unavailable."""
        local = self.live(slot)
        self._bindings[slot] = Binding(local=local, state=SlotState.CONSUMED)
        return local

    def discard_live(self) -> list[tuple[Slot, str]]:
        """Consume all remaining live slots for end-of-function cleanup."""
        discarded: list[tuple[Slot, str]] = []
        for slot in self._order:
            binding = self._bindings[slot]
            if binding.state is SlotState.LIVE:
                discarded.append((slot, binding.local))
                self._bindings[slot] = Binding(local=binding.local, state=SlotState.CONSUMED)
        return discarded

    def snapshot(self) -> LinearitySnapshot:
        """Return an opaque copy for speculative branch or loop emission."""
        return dict(self._bindings)

    def restore(self, snapshot: LinearitySnapshot) -> None:
        """Restore a previous snapshot before emitting another region."""
        self._require_valid_snapshot(snapshot, label="restore")
        self._bindings = dict(snapshot)

    def merge_if(
        self,
        before: LinearitySnapshot,
        then_state: LinearitySnapshot,
        else_state: LinearitySnapshot | None = None,
        *,
        label: str,
    ) -> None:
        """Accept an if only when both exits leave identical slot bindings."""
        self._require_valid_snapshot(before, label=f"{label} before")
        self._require_valid_snapshot(then_state, label=f"{label} then")
        merged_else = before if else_state is None else else_state
        self._require_valid_snapshot(merged_else, label=f"{label} else")

        if then_state != merged_else:
            msg = (
                f"{label} leaves divergent Guppy slot states; "
                f"then={self._snapshot_summary(then_state)}, else={self._snapshot_summary(merged_else)}"
            )
            raise LinearityError(msg)
        self._bindings = dict(then_state)

    def assert_same(
        self,
        before: LinearitySnapshot,
        after: LinearitySnapshot,
        *,
        label: str,
    ) -> None:
        """Require a loop/region body to preserve exact slot bindings."""
        self._require_valid_snapshot(before, label=f"{label} before")
        self._require_valid_snapshot(after, label=f"{label} after")
        if before != after:
            msg = (
                f"{label} changes Guppy slot state across a required invariant; "
                f"before={self._snapshot_summary(before)}, after={self._snapshot_summary(after)}"
            )
            raise LinearityError(msg)
        self._bindings = dict(after)

    def permute(self, mapping: Mapping[Slot, Slot], *, label: str) -> None:
        """Apply a static logical-slot permutation to the binding table.

        `mapping` is interpreted as `logical_source -> old_logical_target`:
        after `permute({a[0]: a[1], a[1]: a[0]})`, references to `a[0]`
        use the binding that previously belonged to `a[1]`.
        """
        keys = set(mapping)
        values = set(mapping.values())
        if keys != values:
            msg = f"{label} must be bijective over the same slot set"
            raise LinearityError(msg)

        for slot in keys:
            self._require_known(slot)
        for slot in values:
            self._require_known(slot)

        old_bindings = dict(self._bindings)
        for source, target in mapping.items():
            self._bindings[source] = old_bindings[target]

    def _require_known(self, slot: Slot) -> None:
        if slot not in self._bindings:
            msg = f"Unknown Guppy slot {slot}"
            raise LinearityError(msg)

    def _require_valid_snapshot(self, snapshot: LinearitySnapshot, *, label: str) -> None:
        if set(snapshot) != set(self._bindings):
            msg = f"{label} snapshot has different slot set"
            raise LinearityError(msg)

    def _snapshot_summary(self, snapshot: LinearitySnapshot) -> str:
        parts = []
        for slot in self._order:
            binding = snapshot[slot]
            parts.append(f"{slot}:{binding.local}/{binding.state.name}")
        return "{" + ", ".join(parts) + "}"


__all__ = [
    "Binding",
    "GuppyLinearityState",
    "LinearityError",
    "LinearitySnapshot",
    "Slot",
    "SlotState",
]
