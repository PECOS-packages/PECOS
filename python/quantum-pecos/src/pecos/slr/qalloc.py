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

"""
Qubit Allocator for SLR.

Inspired by Zig's allocator pattern and NASA's Power of 10 rules.
Provides hierarchical qubit slot management with explicit lifecycle states.

See docs/proposals/slr-qubit-allocators.md for full design documentation.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class SlotState(Enum):
    """State of a qubit slot in an allocator.

    Two states only:
    - UNPREPARED: Not ready for gates (initial state or after measurement)
    - PREPARED: Ready for gate operations
    """

    UNPREPARED = "unprepared"
    PREPARED = "prepared"


class QubitRef:
    """A reference to a qubit slot in an allocator.

    Used as arguments to gate operations. Not a standalone qubit -
    always tied to its parent allocator.

    QubitRef is ephemeral - created for immediate gate application,
    not stored. Like Zig slices, valid only while allocator is alive.

    Provides compatibility with legacy Qubit interface:
    - `ref.reg` returns the allocator (like Qubit.reg returns QReg)
    - `ref.index` returns the slot index
    """

    __slots__ = ("_alloc", "_index")

    def __init__(self, alloc: QAlloc, index: int) -> None:
        self._alloc = alloc
        self._index = index

    @property
    def alloc(self) -> QAlloc:
        """The allocator this ref belongs to."""
        return self._alloc

    @property
    def reg(self) -> QAlloc:
        """Alias for alloc - compatibility with Qubit.reg interface."""
        return self._alloc

    @property
    def index(self) -> int:
        """The slot index within the allocator."""
        return self._index

    @property
    def state(self) -> SlotState:
        """Current state of this slot."""
        return self._alloc.state(self._index)

    @property
    def is_prepared(self) -> bool:
        """Whether this slot is prepared and ready for gates."""
        return self.state == SlotState.PREPARED

    def __repr__(self) -> str:
        return f"QubitRef({self._alloc.name}[{self._index}])"

    def __str__(self) -> str:
        return f"{self._alloc.name}[{self._index}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QubitRef):
            return NotImplemented
        return self._alloc is other._alloc and self._index == other._index

    def __hash__(self) -> int:
        return hash((id(self._alloc), self._index))


class QAlloc:
    """Qubit allocator managing N qubit slots.

    Inspired by Zig's allocator pattern. Provides:
    - Hierarchical ownership (parent-child relationships)
    - Slot-based access (logical indices, not physical qubits)
    - Lifecycle tracking (unprepared/prepared states)
    - Natural scoping (unreturned allocators release to parent)

    Usage:
        # Base allocator in main
        base = QAlloc(capacity=100)

        # Child allocators partition the resource
        data = base.child(7, name="data")
        ancilla = base.child(6, name="ancilla")

        # Prepare slots before use
        data.prepare_all()

        # Access via indexing
        H(data[0])
        CNOT(data[0], data[1])

        # Measure transitions to unprepared
        result = Measure(data[0])  # data[0] now unprepared

        # Re-prepare for reuse
        data.prepare(0)
    """

    def __init__(
        self,
        capacity: int,
        *,
        name: str | None = None,
        parent: QAlloc | None = None,
        _parent_indices: list[int] | None = None,
    ) -> None:
        """Create a qubit allocator.

        Args:
            capacity: Number of qubit slots in this allocator.
            name: Optional name for this allocator (for debugging/codegen).
            parent: Parent allocator (None for base allocator).
            _parent_indices: Internal - indices reserved from parent.
        """
        if capacity < 0:
            msg = f"Capacity must be non-negative, got {capacity}"
            raise ValueError(msg)

        self._capacity = capacity
        self._name = name
        self._parent = parent
        self._parent_indices = _parent_indices or []

        # Slot states - all start unprepared
        self._slot_states: list[SlotState] = [SlotState.UNPREPARED] * capacity

        # Track which slots are reserved by children
        self._reserved: set[int] = set()

        # Track child allocators
        self._children: list[QAlloc] = []

        # Track if this allocator has been released
        self._released = False

    # --- Properties ---

    @property
    def capacity(self) -> int:
        """Total number of slots in this allocator."""
        return self._capacity

    @property
    def size(self) -> int:
        """Alias for capacity - compatibility with QReg.size interface."""
        return self._capacity

    @property
    def name(self) -> str:
        """Name of this allocator."""
        return self._name or f"alloc_{id(self)}"

    @property
    def sym(self) -> str:
        """Alias for name - compatibility with QReg.sym interface."""
        return self.name

    @property
    def parent(self) -> QAlloc | None:
        """Parent allocator, or None if this is a base allocator."""
        return self._parent

    @property
    def is_base(self) -> bool:
        """Whether this is a base (root) allocator."""
        return self._parent is None

    @property
    def available(self) -> int:
        """Number of slots not reserved by children."""
        return self._capacity - len(self._reserved)

    @property
    def is_released(self) -> bool:
        """Whether this allocator has been released."""
        return self._released

    # --- Child Allocator Creation ---

    def child(self, size: int, *, name: str | None = None) -> QAlloc:
        """Create a child allocator with `size` slots.

        Reserves `size` slots from this allocator's available pool.
        Child allocator slots start unprepared.

        Args:
            size: Number of slots for the child allocator.
            name: Optional name for the child allocator.

        Returns:
            A new child QAlloc with `size` slots.

        Raises:
            ValueError: If insufficient capacity available.
            RuntimeError: If this allocator has been released.
        """
        self._check_not_released()

        if size < 0:
            msg = f"Size must be non-negative, got {size}"
            raise ValueError(msg)

        if size > self.available:
            msg = f"Insufficient capacity: requested {size}, available {self.available}"
            raise ValueError(msg)

        # Find available indices to reserve
        available_indices = [i for i in range(self._capacity) if i not in self._reserved]
        indices_to_reserve = available_indices[:size]

        # Mark as reserved
        self._reserved.update(indices_to_reserve)

        # Create child
        child_alloc = QAlloc(
            capacity=size,
            name=name,
            parent=self,
            _parent_indices=indices_to_reserve,
        )
        self._children.append(child_alloc)

        return child_alloc

    # --- Lifecycle Operations ---

    def prepare(self, *indices: int) -> None:
        """Prepare specific slots (unprepared -> prepared).

        Requesting a qubit to be associated with a slot and initialized to |0>.

        Args:
            *indices: Slot indices to prepare.

        Raises:
            IndexError: If any index is out of range.
            ValueError: If any index is reserved by a child.
            RuntimeError: If this allocator has been released.
        """
        self._check_not_released()

        for idx in indices:
            self._check_index(idx)
            self._check_not_reserved(idx)
            self._slot_states[idx] = SlotState.PREPARED

    def prepare_all(self) -> None:
        """Prepare all slots in this allocator.

        Only prepares slots not reserved by children.

        Raises:
            RuntimeError: If this allocator has been released.
        """
        self._check_not_released()

        for i in range(self._capacity):
            if i not in self._reserved:
                self._slot_states[i] = SlotState.PREPARED

    def mark_unprepared(self, *indices: int) -> None:
        """Mark specific slots as unprepared (after measurement).

        This is called by Measure operations to transition slots.

        Args:
            *indices: Slot indices to mark unprepared.

        Raises:
            IndexError: If any index is out of range.
            RuntimeError: If this allocator has been released.
        """
        self._check_not_released()

        for idx in indices:
            self._check_index(idx)
            self._slot_states[idx] = SlotState.UNPREPARED

    def mark_all_unprepared(self) -> None:
        """Mark all slots as unprepared.

        Raises:
            RuntimeError: If this allocator has been released.
        """
        self._check_not_released()

        for i in range(self._capacity):
            if i not in self._reserved:
                self._slot_states[i] = SlotState.UNPREPARED

    # --- Slot State Queries ---

    def state(self, index: int) -> SlotState:
        """Get the state of a specific slot.

        Args:
            index: Slot index.

        Returns:
            SlotState.UNPREPARED or SlotState.PREPARED.

        Raises:
            IndexError: If index is out of range.
        """
        self._check_index(index)
        return self._slot_states[index]

    def is_prepared(self, index: int) -> bool:
        """Check if a specific slot is prepared.

        Args:
            index: Slot index.

        Returns:
            True if the slot is prepared, False otherwise.
        """
        return self.state(index) == SlotState.PREPARED

    def all_prepared(self) -> bool:
        """Check if all non-reserved slots are prepared."""
        return all(self._slot_states[i] == SlotState.PREPARED for i in range(self._capacity) if i not in self._reserved)

    def prepared_count(self) -> int:
        """Count of prepared slots."""
        return sum(
            1 for i in range(self._capacity) if i not in self._reserved and self._slot_states[i] == SlotState.PREPARED
        )

    def unprepared_count(self) -> int:
        """Count of unprepared slots."""
        return sum(
            1 for i in range(self._capacity) if i not in self._reserved and self._slot_states[i] == SlotState.UNPREPARED
        )

    # --- Slot Access ---

    def __getitem__(self, index: int) -> QubitRef:
        """Access a slot for use in gates.

        Returns a QubitRef that can be passed to gate operations.

        Args:
            index: Slot index.

        Returns:
            QubitRef for the slot.

        Raises:
            IndexError: If index is out of range.
            ValueError: If index is reserved by a child.
            RuntimeError: If this allocator has been released.
        """
        self._check_not_released()
        self._check_index(index)
        self._check_not_reserved(index)

        return QubitRef(self, index)

    def __len__(self) -> int:
        """Total capacity of this allocator."""
        return self._capacity

    def __iter__(self) -> Iterator[QubitRef]:
        """Iterate over all non-reserved slots as QubitRefs."""
        self._check_not_released()
        for i in range(self._capacity):
            if i not in self._reserved:
                yield QubitRef(self, i)

    # --- Release ---

    def release(self) -> None:
        """Explicitly release this allocator back to parent.

        Normally handled automatically by scoping, but can be called explicitly.
        All child allocators are also released.

        Raises:
            RuntimeError: If already released.
        """
        if self._released:
            msg = f"Allocator {self.name} already released"
            raise RuntimeError(msg)

        # Release all children first
        for child in self._children:
            if not child.is_released:
                child.release()

        # Clear our slots back to parent's reserved set
        if self._parent is not None:
            self._parent.unreserve_slots(self._parent_indices)
            self._parent.unregister_child(self)

        self._released = True

    # --- Parent/Child Coordination ---

    def unreserve_slots(self, indices: list[int]) -> None:
        """Release the given slot indices back to the available pool.

        Called by child allocators when they are released.
        Not intended for external use.
        """
        self._reserved -= set(indices)

    def unregister_child(self, child: QAlloc) -> None:
        """Remove a child allocator from this allocator's children list.

        Called by child allocators when they are released.
        Not intended for external use.
        """
        self._children.remove(child)

    # --- Validation Helpers ---

    def _check_not_released(self) -> None:
        """Raise if this allocator has been released."""
        if self._released:
            msg = f"Allocator {self.name} has been released"
            raise RuntimeError(msg)

    def _check_index(self, index: int) -> None:
        """Raise if index is out of range."""
        if not 0 <= index < self._capacity:
            msg = f"Slot index {index} out of range [0, {self._capacity})"
            raise IndexError(msg)

    def _check_not_reserved(self, index: int) -> None:
        """Raise if index is reserved by a child."""
        if index in self._reserved:
            msg = f"Slot {index} is reserved by a child allocator"
            raise ValueError(msg)

    # --- Representation ---

    def __repr__(self) -> str:
        parent_info = f", parent={self._parent.name}" if self._parent else ", base"
        return f"QAlloc({self.name}, capacity={self._capacity}, available={self.available}{parent_info})"

    def __str__(self) -> str:
        return self.name


# Backward compatibility: QReg as alias
# In the future, QReg could be implemented as a thin wrapper around QAlloc
# that auto-prepares all slots on creation (matching old behavior)
