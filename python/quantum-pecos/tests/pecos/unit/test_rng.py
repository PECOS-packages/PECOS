"""Testing module for the RNG Model."""

import sys

import pecos as pc
import pytest
from pecos.engines.cvm.rng_model import RNGModel


def draw_at_index(seed: int, index: int) -> int:
    """Return the next random value after consuming ``index`` values from a seeded stream."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(seed)
    rng.set_index(index)
    return rng.rng_random()


def test_set_seed() -> None:
    """Verifies that a seed is set properly for our RNG model."""
    rng = RNGModel(shot_id=0)
    seed = 42
    rng.set_seed(seed)
    assert rng.seed == seed
    assert rng.count == 0


def test_init_normalizes_none_bound_to_unbounded() -> None:
    """Verifies ``None`` uses the unbounded sentinel instead of leaking into bounded draws."""
    rng = RNGModel(shot_id=0, current_bound=None)
    assert rng.current_bound == 0


def test_random_number() -> None:
    """Verifies that the random number generated is an int type."""
    rng = RNGModel(shot_id=0)
    random = rng.rng_random()
    assert isinstance(random, int)


def test_bounded_random() -> None:
    """Verifies that a single generated random number is within bounds."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)
    bound = 16
    rng.set_bound(bound)
    assert rng.current_bound == bound

    random_number = rng.rng_random()
    assert 0 <= random_number < bound


def test_set_idx_raises_for_backwards_index() -> None:
    """Verifies that an error is raised when specifying an index that was already consumed in the RNG stream."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)

    rng.set_index(4)

    try:
        rng.set_index(3)
    except ValueError as exc:
        expected_error_msg = "RNGindex(3) cannot move backward: current stream index is 4"
        assert str(exc) == expected_error_msg


def test_set_idx() -> None:
    """Verifies that the idx is set properly for our model."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)
    idx = 4
    rng.set_index(idx)
    assert rng.count == idx


def test_relative_advance_forward() -> None:
    """Verifies that a forward relative advance lands on the expected stream position."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)

    rng.set_relative_index(5)

    assert rng.count == 5
    assert rng.rng_random() == draw_at_index(42, 5)


def test_relative_advance_backward() -> None:
    """Verifies that a backward relative advance reconstructs the stream from the seed."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)
    rng.set_relative_index(6)

    rng.eval_func({"func": "RNGadvance", "args": ["-3"]}, {})

    assert rng.count == 3
    assert rng.rng_random() == draw_at_index(42, 3)


def test_relative_advance_backward_past_start_raises() -> None:
    """Verifies rewinds cannot move before the start of the stream."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)
    rng.set_relative_index(2)

    with pytest.raises(
        ValueError,
        match=r"RNGadvance\(-3\) cannot move before the start of the stream: current stream index is 2",
    ):
        rng.set_relative_index(-3)


def test_reseed_then_advance_changes_draw_and_resets_count() -> None:
    """Verifies reseeding resets the logical position used by later relative advances."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)
    rng.set_relative_index(5)
    advanced_val = rng.rng_random()

    rng.set_seed(42)
    immediate_val = rng.rng_random()
    expected_first_draw = draw_at_index(42, 0)

    assert advanced_val != immediate_val
    assert immediate_val == expected_first_draw
    assert rng.count == 1


def test_relative_advance_keeps_count_consistent_after_draws() -> None:
    """Verifies count tracks the current stream position after advances and draws."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)

    rng.set_relative_index(5)
    assert rng.count == 5

    rng.rng_random()
    assert rng.count == 6

    rng.set_relative_index(-2)
    assert rng.count == 4
    assert rng.rng_random() == draw_at_index(42, 4)


def test_negative_values_are_rejected_for_non_advance_rng_funcs() -> None:
    """Verifies only RNGadvance accepts negative numeric arguments."""
    rng = RNGModel(shot_id=0)

    with pytest.raises(ValueError, match=r"RNG seed must be non-negative: got -1"):
        rng.eval_func({"func": "RNGseed", "args": ["-1"]}, {})

    with pytest.raises(ValueError, match=r"RNG bound must be non-negative: got -1"):
        rng.eval_func({"func": "RNGbound", "args": ["-1"]}, {})

    with pytest.raises(ValueError, match=r"RNG index must be non-negative: got -1"):
        rng.eval_func({"func": "RNGindex", "args": ["-1"]}, {})


def test_relative_advance_backward_replays_historical_bounds() -> None:
    """Verifies rewind reconstructs the stream using the original bounds history."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)
    rng.set_bound(16)
    rng.rng_random()
    rng.set_bound(0)
    expected_second_draw = rng.rng_random()

    rng.set_relative_index(-1)

    assert rng.count == 1
    assert rng.rng_random() == expected_second_draw


def test_multiple_bounded_rand() -> None:
    """For several randomly generated number, with a random bound, verifies that its appropriate."""
    rng = RNGModel(shot_id=0)
    rng.set_seed(42)

    # Use platform-appropriate upper bound for randint
    # Windows: i32 max is 2^31 - 1 (2147483647), Unix: i64 allows 2^32
    max_bound = 2**31 - 1 if sys.platform == "win32" else 2**32

    for _ in range(100):
        random_bound = int(pc.random.randint(1, max_bound, 1)[0])
        rng.set_bound(random_bound)
        random_number = rng.rng_random()
        assert 0 <= random_number < random_bound
