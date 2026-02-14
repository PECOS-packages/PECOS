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

"""Tests for QAlloc - Qubit Allocator for SLR."""

import pytest

from pecos.slr.qalloc import QAlloc, QubitRef, SlotState


class TestSlotState:
    """Tests for SlotState enum."""

    def test_two_states(self):
        """Only two states: unprepared and prepared."""
        assert len(SlotState) == 2
        assert SlotState.UNPREPARED.value == "unprepared"
        assert SlotState.PREPARED.value == "prepared"


class TestQubitRef:
    """Tests for QubitRef."""

    def test_creation(self):
        """QubitRef is created via allocator indexing."""
        alloc = QAlloc(5, name="test")
        ref = alloc[0]

        assert isinstance(ref, QubitRef)
        assert ref.alloc is alloc
        assert ref.index == 0

    def test_string_representation(self):
        """QubitRef has readable string representation."""
        alloc = QAlloc(5, name="data")
        ref = alloc[2]

        assert str(ref) == "data[2]"
        assert "data[2]" in repr(ref)

    def test_state_property(self):
        """QubitRef reflects slot state."""
        alloc = QAlloc(5, name="test")
        ref = alloc[0]

        assert ref.state == SlotState.UNPREPARED
        assert not ref.is_prepared

        alloc.prepare(0)

        assert ref.state == SlotState.PREPARED
        assert ref.is_prepared

    def test_equality(self):
        """QubitRefs are equal if same allocator and index."""
        alloc = QAlloc(5, name="test")

        ref1 = alloc[0]
        ref2 = alloc[0]
        ref3 = alloc[1]

        assert ref1 == ref2
        assert ref1 != ref3

    def test_hashable(self):
        """QubitRefs can be used in sets/dicts."""
        alloc = QAlloc(5, name="test")

        ref_set = {alloc[0], alloc[1], alloc[0]}
        assert len(ref_set) == 2


class TestQAllocBase:
    """Tests for base QAlloc creation and properties."""

    def test_create_base_allocator(self):
        """Base allocator has no parent."""
        base = QAlloc(100, name="base")

        assert base.capacity == 100
        assert base.name == "base"
        assert base.parent is None
        assert base.is_base
        assert base.available == 100

    def test_default_name(self):
        """Allocator gets default name if none provided."""
        alloc = QAlloc(10)
        assert alloc.name.startswith("alloc_")

    def test_invalid_capacity(self):
        """Negative capacity raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            QAlloc(-1)

    def test_zero_capacity(self):
        """Zero capacity is allowed."""
        alloc = QAlloc(0)
        assert alloc.capacity == 0
        assert alloc.available == 0


class TestQAllocSlotStates:
    """Tests for slot state management."""

    def test_initial_state_unprepared(self):
        """All slots start unprepared."""
        alloc = QAlloc(5, name="test")

        for i in range(5):
            assert alloc.state(i) == SlotState.UNPREPARED
            assert not alloc.is_prepared(i)

    def test_prepare_single(self):
        """Prepare individual slots."""
        alloc = QAlloc(5, name="test")

        alloc.prepare(0)
        assert alloc.is_prepared(0)
        assert not alloc.is_prepared(1)

    def test_prepare_multiple(self):
        """Prepare multiple slots at once."""
        alloc = QAlloc(5, name="test")

        alloc.prepare(0, 2, 4)

        assert alloc.is_prepared(0)
        assert not alloc.is_prepared(1)
        assert alloc.is_prepared(2)
        assert not alloc.is_prepared(3)
        assert alloc.is_prepared(4)

    def test_prepare_all(self):
        """Prepare all slots."""
        alloc = QAlloc(5, name="test")

        alloc.prepare_all()

        assert alloc.all_prepared()
        assert alloc.prepared_count() == 5
        assert alloc.unprepared_count() == 0

    def test_mark_unprepared(self):
        """Mark slots as unprepared (after measurement)."""
        alloc = QAlloc(5, name="test")
        alloc.prepare_all()

        alloc.mark_unprepared(0, 2)

        assert not alloc.is_prepared(0)
        assert alloc.is_prepared(1)
        assert not alloc.is_prepared(2)
        assert alloc.is_prepared(3)

    def test_mark_all_unprepared(self):
        """Mark all slots as unprepared."""
        alloc = QAlloc(5, name="test")
        alloc.prepare_all()

        alloc.mark_all_unprepared()

        assert alloc.prepared_count() == 0
        assert alloc.unprepared_count() == 5

    def test_prepare_unprepared_cycle(self):
        """Slots can be prepared, measured (unprepared), and re-prepared."""
        alloc = QAlloc(3, name="ancilla")

        # Initial: unprepared
        assert alloc.unprepared_count() == 3

        # Prepare
        alloc.prepare_all()
        assert alloc.prepared_count() == 3

        # Measure (marks unprepared)
        alloc.mark_all_unprepared()
        assert alloc.unprepared_count() == 3

        # Re-prepare
        alloc.prepare_all()
        assert alloc.prepared_count() == 3


class TestQAllocChild:
    """Tests for child allocator creation."""

    def test_create_child(self):
        """Child allocator reserves slots from parent."""
        base = QAlloc(100, name="base")

        child = base.child(10, name="data")

        assert child.capacity == 10
        assert child.name == "data"
        assert child.parent is base
        assert not child.is_base
        assert base.available == 90

    def test_multiple_children(self):
        """Multiple children can be created."""
        base = QAlloc(100, name="base")

        data = base.child(7, name="data")
        ancilla = base.child(6, name="ancilla")

        assert base.available == 87
        assert data.capacity == 7
        assert ancilla.capacity == 6

    def test_nested_children(self):
        """Children can have their own children."""
        base = QAlloc(100, name="base")
        level1 = base.child(50, name="level1")
        level2 = level1.child(20, name="level2")

        assert level2.parent is level1
        assert level1.parent is base
        assert level1.available == 30

    def test_insufficient_capacity(self):
        """Cannot create child larger than available."""
        base = QAlloc(10, name="base")
        base.child(8, name="child1")

        with pytest.raises(ValueError, match="Insufficient capacity"):
            base.child(5, name="child2")  # only 2 available

    def test_child_slots_independent(self):
        """Child slot states are independent of parent."""
        base = QAlloc(10, name="base")
        child = base.child(5, name="child")

        child.prepare_all()

        # Parent's unreserved slots are still unprepared
        assert base.state(5) == SlotState.UNPREPARED

    def test_cannot_access_reserved_slots(self):
        """Parent cannot access slots reserved by child."""
        base = QAlloc(10, name="base")
        base.child(5, name="child")

        # Slots 0-4 are reserved
        with pytest.raises(ValueError, match="reserved"):
            base[0]  # noqa: B018 - intentional access for test

    def test_cannot_prepare_reserved_slots(self):
        """Parent cannot prepare slots reserved by child."""
        base = QAlloc(10, name="base")
        base.child(5, name="child")

        with pytest.raises(ValueError, match="reserved"):
            base.prepare(0)


class TestQAllocRelease:
    """Tests for allocator release."""

    def test_explicit_release(self):
        """Explicit release returns slots to parent."""
        base = QAlloc(10, name="base")
        child = base.child(5, name="child")

        assert base.available == 5

        child.release()

        assert base.available == 10
        assert child.is_released

    def test_released_allocator_unusable(self):
        """Released allocator cannot be used."""
        base = QAlloc(10, name="base")
        child = base.child(5, name="child")
        child.release()

        with pytest.raises(RuntimeError, match="released"):
            child[0]  # noqa: B018

        with pytest.raises(RuntimeError, match="released"):
            child.prepare(0)

    def test_double_release_error(self):
        """Cannot release twice."""
        base = QAlloc(10, name="base")
        child = base.child(5, name="child")
        child.release()

        with pytest.raises(RuntimeError, match="already released"):
            child.release()

    def test_release_cascades_to_children(self):
        """Releasing parent releases all children."""
        base = QAlloc(100, name="base")
        level1 = base.child(50, name="level1")
        level2 = level1.child(20, name="level2")

        level1.release()

        assert level1.is_released
        assert level2.is_released


class TestQAllocIteration:
    """Tests for iterating over allocator slots."""

    def test_len(self):
        """len() returns capacity."""
        alloc = QAlloc(7, name="test")
        assert len(alloc) == 7

    def test_iterate_slots(self):
        """Can iterate over slots as QubitRefs."""
        alloc = QAlloc(3, name="test")

        refs = list(alloc)

        assert len(refs) == 3
        assert all(isinstance(r, QubitRef) for r in refs)
        assert [r.index for r in refs] == [0, 1, 2]

    def test_iterate_skips_reserved(self):
        """Iteration skips slots reserved by children."""
        base = QAlloc(10, name="base")
        base.child(5, name="child")  # reserves 0-4

        refs = list(base)

        assert len(refs) == 5
        assert [r.index for r in refs] == [5, 6, 7, 8, 9]


class TestQAllocIndexOutOfRange:
    """Tests for index validation."""

    def test_negative_index_error(self):
        """Negative index raises error."""
        alloc = QAlloc(5, name="test")

        with pytest.raises(IndexError):
            alloc[-1]  # noqa: B018

    def test_too_large_index_error(self):
        """Index >= capacity raises error."""
        alloc = QAlloc(5, name="test")

        with pytest.raises(IndexError):
            alloc[5]  # noqa: B018

    def test_prepare_invalid_index(self):
        """Prepare with invalid index raises error."""
        alloc = QAlloc(5, name="test")

        with pytest.raises(IndexError):
            alloc.prepare(10)

    def test_state_invalid_index(self):
        """State query with invalid index raises error."""
        alloc = QAlloc(5, name="test")

        with pytest.raises(IndexError):
            alloc.state(10)


class TestQRegCompatibility:
    """Tests for QReg/Qubit interface compatibility."""

    def test_qalloc_sym_property(self):
        """QAlloc.sym is alias for name (like QReg.sym)."""
        alloc = QAlloc(5, name="data")
        assert alloc.sym == "data"
        assert alloc.sym == alloc.name

    def test_qalloc_size_property(self):
        """QAlloc.size is alias for capacity (like QReg.size)."""
        alloc = QAlloc(7, name="test")
        assert alloc.size == 7
        assert alloc.size == alloc.capacity

    def test_qubitref_reg_property(self):
        """QubitRef.reg is alias for alloc (like Qubit.reg)."""
        alloc = QAlloc(5, name="data")
        ref = alloc[0]

        assert ref.reg is alloc
        assert ref.reg is ref.alloc

    def test_qubitref_str_matches_qubit_pattern(self):
        """QubitRef string format matches Qubit: 'regname[index]'."""
        alloc = QAlloc(5, name="q")
        ref = alloc[2]

        assert str(ref) == "q[2]"

    def test_compatibility_pattern(self):
        """Common pattern: ref.reg.sym works like qubit.reg.sym."""
        alloc = QAlloc(5, name="ancilla")
        ref = alloc[0]

        # This pattern is used in code generation
        assert ref.reg.sym == "ancilla"
        assert ref.reg.size == 5


class TestQAllocQECPattern:
    """Tests demonstrating QEC usage patterns."""

    def test_syndrome_extraction_pattern(self):
        """Test typical syndrome extraction pattern."""
        # Base allocator
        base = QAlloc(17, name="base")

        # Partition into data and ancilla
        data = base.child(9, name="data")
        ancilla = base.child(8, name="ancilla")

        # Initialize data
        data.prepare_all()
        assert data.all_prepared()

        # Syndrome extraction rounds
        for _round in range(3):
            # Prepare ancilla
            ancilla.prepare_all()
            assert ancilla.all_prepared()

            # After syndrome extraction circuit, measure ancilla
            ancilla.mark_all_unprepared()
            assert ancilla.unprepared_count() == 8

            # Data qubits remain prepared
            assert data.all_prepared()

    def test_nested_workspace_pattern(self):
        """Test nested workspace allocation."""
        base = QAlloc(20, name="base")
        main_qubits = base.child(10, name="main")

        main_qubits.prepare_all()

        # Function that needs temporary workspace
        def operation_needing_workspace(alloc: QAlloc):
            workspace = alloc.child(3, name="workspace")
            workspace.prepare_all()

            # Do some work...

            # Workspace released when function returns
            workspace.release()

        operation_needing_workspace(base)

        # Base has full capacity back
        assert base.available == 10  # main_qubits still reserved
