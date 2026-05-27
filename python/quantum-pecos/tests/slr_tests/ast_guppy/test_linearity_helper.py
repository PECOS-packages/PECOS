"""Unit tests for the Guppy-only linearity helper."""

from __future__ import annotations

import pytest
from pecos.slr.ast.codegen.guppy_linearity import (
    Binding,
    GuppyLinearityState,
    LinearityError,
    Slot,
    SlotState,
)


def test_slot_and_binding_are_stable_values() -> None:
    slot = Slot("q", 0)
    binding = Binding("q_0", SlotState.LIVE)

    assert str(slot) == "q[0]"
    assert slot == Slot("q", 0)
    assert hash(slot) == hash(Slot("q", 0))
    assert binding == Binding("q_0", SlotState.LIVE)


def test_from_allocators_creates_live_bindings_in_stable_order() -> None:
    state = GuppyLinearityState.from_allocators({"q": 2, "anc": 1})

    assert list(state.bindings()) == [
        (Slot("q", 0), Binding("q_0", SlotState.LIVE)),
        (Slot("q", 1), Binding("q_1", SlotState.LIVE)),
        (Slot("anc", 0), Binding("anc_0", SlotState.LIVE)),
    ]


def test_from_allocators_rejects_negative_sizes() -> None:
    with pytest.raises(LinearityError, match="negative size"):
        GuppyLinearityState.from_allocators({"q": -1})


def test_set_live_consume_and_error_paths() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    slot = Slot("q", 0)

    assert state.status(slot) is SlotState.LIVE
    assert state.live(slot) == "q_0"

    assert state.consume(slot) == "q_0"
    assert state.status(slot) is SlotState.CONSUMED

    with pytest.raises(LinearityError, match="consumed"):
        state.live(slot)
    with pytest.raises(LinearityError, match="consumed"):
        state.consume(slot)

    state.set_live(slot, "q_0")
    assert state.binding(slot) == Binding("q_0", SlotState.LIVE)

    unknown = Slot("q", 1)
    with pytest.raises(LinearityError, match="Unknown"):
        state.binding(unknown)
    with pytest.raises(LinearityError, match="Unknown"):
        state.set_live(unknown, "q_1")
    with pytest.raises(LinearityError, match="Unknown"):
        state.consume(unknown)


def test_discard_live_consumes_only_live_slots() -> None:
    state = GuppyLinearityState.from_allocators({"q": 3})

    state.consume(Slot("q", 1))

    assert state.discard_live() == [
        (Slot("q", 0), "q_0"),
        (Slot("q", 2), "q_2"),
    ]
    assert state.status(Slot("q", 0)) is SlotState.CONSUMED
    assert state.status(Slot("q", 1)) is SlotState.CONSUMED
    assert state.status(Slot("q", 2)) is SlotState.CONSUMED
    assert state.discard_live() == []


def test_snapshot_and_restore_round_trip() -> None:
    state = GuppyLinearityState.from_allocators({"q": 2})
    before = state.snapshot()

    state.consume(Slot("q", 0))
    state.set_live(Slot("q", 1), "custom_q_1")

    assert state.status(Slot("q", 0)) is SlotState.CONSUMED
    assert state.live(Slot("q", 1)) == "custom_q_1"

    state.restore(before)
    assert list(state.bindings()) == [
        (Slot("q", 0), Binding("q_0", SlotState.LIVE)),
        (Slot("q", 1), Binding("q_1", SlotState.LIVE)),
    ]


def test_restore_rejects_snapshot_for_different_slot_set() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})

    with pytest.raises(LinearityError, match="different slot set"):
        state.restore({})


def test_merge_if_accepts_matching_explicit_branches() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = state.snapshot()

    state.set_live(Slot("q", 0), "q_0")
    then_state = state.snapshot()
    state.restore(before)
    state.set_live(Slot("q", 0), "q_0")
    else_state = state.snapshot()

    state.merge_if(before, then_state, else_state, label="If(c[0])")
    assert state.live(Slot("q", 0)) == "q_0"


def test_merge_if_accepts_identity_else_when_none() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = state.snapshot()

    state.set_live(Slot("q", 0), "q_0")
    then_state = state.snapshot()

    state.merge_if(before, then_state, else_state=None, label="If(c[0])")
    assert state.live(Slot("q", 0)) == "q_0"


def test_merge_if_rejects_divergent_branch_states() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = state.snapshot()

    state.consume(Slot("q", 0))
    then_state = state.snapshot()

    with pytest.raises(LinearityError, match="divergent"):
        state.merge_if(before, then_state, else_state=None, label="If(c[0])")


def test_assert_same_accepts_preserved_loop_body() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = state.snapshot()

    state.set_live(Slot("q", 0), "q_0")
    after = state.snapshot()

    state.assert_same(before, after, label="Repeat(3)")
    assert state.live(Slot("q", 0)) == "q_0"


def test_assert_same_rejects_loop_body_that_changes_state() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = state.snapshot()

    state.consume(Slot("q", 0))
    after = state.snapshot()

    with pytest.raises(LinearityError, match="changes Guppy slot state"):
        state.assert_same(before, after, label="Repeat(3)")


def test_permute_clean_swap() -> None:
    state = GuppyLinearityState.from_allocators({"q": 2})

    state.permute(
        {
            Slot("q", 0): Slot("q", 1),
            Slot("q", 1): Slot("q", 0),
        },
        label="Permute(q[0], q[1])",
    )

    assert state.live(Slot("q", 0)) == "q_1"
    assert state.live(Slot("q", 1)) == "q_0"


def test_permute_empty_mapping_is_noop() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = list(state.bindings())

    state.permute({}, label="empty Permute")

    assert list(state.bindings()) == before


def test_permute_identity_mapping_is_noop() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})
    before = list(state.bindings())

    state.permute({Slot("q", 0): Slot("q", 0)}, label="identity Permute")

    assert list(state.bindings()) == before


def test_permute_cross_allocator_cycle() -> None:
    state = GuppyLinearityState.from_allocators({"a": 1, "b": 1, "c": 1})

    state.permute(
        {
            Slot("a", 0): Slot("b", 0),
            Slot("b", 0): Slot("c", 0),
            Slot("c", 0): Slot("a", 0),
        },
        label="Permute(a[0], b[0], c[0])",
    )

    assert state.live(Slot("a", 0)) == "b_0"
    assert state.live(Slot("b", 0)) == "c_0"
    assert state.live(Slot("c", 0)) == "a_0"


def test_permute_rejects_non_bijective_mapping() -> None:
    state = GuppyLinearityState.from_allocators({"q": 2})

    with pytest.raises(LinearityError, match="bijective"):
        state.permute(
            {
                Slot("q", 0): Slot("q", 1),
                Slot("q", 1): Slot("q", 1),
            },
            label="bad Permute",
        )


def test_permute_rejects_unknown_slots() -> None:
    state = GuppyLinearityState.from_allocators({"q": 1})

    with pytest.raises(LinearityError, match="Unknown"):
        state.permute(
            {
                Slot("q", 0): Slot("missing", 0),
                Slot("missing", 0): Slot("q", 0),
            },
            label="bad Permute",
        )
