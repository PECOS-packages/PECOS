"""Regression contracts for signed fixed-width values and canonical hashes."""

import pytest
from pecos_rslib import BitInt, BitUInt


@pytest.mark.parametrize("value", [0, 1, -1, 2**70, -(2**70)])
def test_cross_width_hash(value: int) -> None:
    """Equal mathematical values work as dictionary keys and set members."""
    narrow, wide = BitInt(100, value), BitInt(130, value)
    assert narrow == wide
    assert hash(narrow) == hash(wide) == hash(value)
    assert {narrow: "found"}[wide] == "found"
    assert {wide: "found"}[narrow] == "found"
    assert len({narrow, wide}) == 1


@pytest.mark.parametrize("value", [-(2**63), -1, 0, 1, 2**63 - 1])
def test_hash_newly_convertible_wide_value(value: int) -> None:
    """The new i64 fast path and the arbitrary-precision value hash agree."""
    assert int(BitInt(100, value)) == value
    assert hash(BitInt(100, value)) == hash(BitInt(63, value)) == hash(value)


@pytest.mark.parametrize(
    ("size", "a", "b", "quotient", "remainder"),
    [
        (64, -8, 2, -4, 0),
        (64, -7, 2, -3, -1),
        (64, 7, -2, -3, 1),
        (100, -(2**70), 2, -590295810358705651712, 0),
        (100, 2**70, -2, -590295810358705651712, 0),
        (100, -(2**70 + 1), 3, -393530540239137101141, -2),
        (63, -(2**63), -1, -(2**63), 0),
    ],
)
def test_wide_division(
    size: int, a: int, b: int, quotient: int, remainder: int
) -> None:
    """Python // keeps BitInt's truncation toward zero and dividend-sign %."""
    lhs, rhs = BitInt(size, a), BitInt(size, b)
    assert int(lhs // rhs) == quotient
    assert int(lhs % rhs) == remainder


def test_wide_zero_divisor() -> None:
    """The binding's existing ZeroDivisionError checks remain in effect."""
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        BitInt(100, 2**70) // BitInt(130, 0)
    with pytest.raises(ZeroDivisionError, match="modulo by zero"):
        BitInt(100, 2**70) % BitInt(130, 0)


def test_conversion_downstream_observations() -> None:
    """Pin effects of to_i64 without repairing the other binding defects (#958)."""
    value = BitInt(100, -1)
    value.clamp(4)
    assert int(value) == 15
    assert value.size == 100
    assert int(BitUInt(8, 5) + BitInt(100, -1)) == 4
    # Unrepresentable values still hit the existing no-op / zero fallback.
    value = BitInt(100, 2**70)
    value.clamp(4)
    assert int(value) == 2**70
    assert int(BitUInt(8, 5) + value) == 5


def test_cross_type_comparison_defect_remains() -> None:
    """Cross-type equality still truncates operands; hash cannot repair that."""
    value = BitInt(1, 1)
    assert value == 1
    assert value == 5
