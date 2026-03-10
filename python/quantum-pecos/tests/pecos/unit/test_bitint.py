# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""Tests for BitInt signed fixed-width integer.

BitInt(N) is always signed. Internally wraps BitUInt(N+1) where the extra bit
is the sign bit. BitInt(1, 1) returns 1 (not -1).
"""

from typing import Final

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pecos import BitInt

# BitInt(N) range: with N+1 internal bits, two's complement gives -2^N to 2^N - 1
# For N=63: range is -2^63 to 2^63 - 1 (same as i64)
DEFAULT_SIZE: Final = 63
MIN: Final = -(2**DEFAULT_SIZE)
MAX: Final = 2**DEFAULT_SIZE - 1
int_range = st.integers(min_value=MIN, max_value=MAX)


@given(st.text(alphabet=["0", "1"], min_size=1))
def test_init_binary_string(x: str) -> None:
    """Test BitInt initialization from binary string. Sign bit is implicitly 0."""
    ba = BitInt(x)
    # Binary string creates unsigned value with sign bit 0
    expected_val = int(x, 2)
    assert int(ba) == expected_val


def test_set_bit() -> None:
    """Verify setting a single bit updates the value."""
    ba = BitInt("0000")
    ba[2] = 1
    assert int(ba) == 0b0100


def test_get_bit() -> None:
    """Verify individual bit access returns correct values."""
    ba = BitInt("1010")
    assert ba[2] == 0
    assert ba[3] == 1


def test_to_int() -> None:
    """Verify int conversion of a binary-string-constructed BitInt."""
    ba = BitInt("1010")
    assert int(ba) == 10


def test_1bit_positive() -> None:
    """BitInt(1, 1) returns 1, not -1. The extra sign bit makes this possible."""
    b = BitInt(1, 1)
    assert int(b) == 1


def test_1bit_negative() -> None:
    """BitInt(1, -1) returns -1."""
    b = BitInt(1, -1)
    assert int(b) == -1


def test_1bit_zero() -> None:
    """Verify BitInt(1, 0) is zero."""
    b = BitInt(1, 0)
    assert int(b) == 0


@given(int_range, int_range)
def test_addition(x: int, y: int) -> None:
    """Verify addition of two BitInt values matches Python int addition."""
    assume(MIN <= x + y <= MAX)
    ba1 = BitInt(DEFAULT_SIZE, x)
    ba2 = BitInt(DEFAULT_SIZE, y)
    result = ba1 + ba2
    assert int(result) == x + y


def test_subtraction() -> None:
    """Verify subtraction of two BitInt values."""
    ba1 = BitInt("1101")  # 13
    ba2 = BitInt("1010")  # 10
    result = ba1 - ba2
    assert int(result) == 3


@given(int_range, int_range)
def test_multiplication(x: int, y: int) -> None:
    """Verify multiplication of two BitInt values matches Python int multiplication."""
    assume(MIN <= x * y <= MAX)
    ba1 = BitInt(DEFAULT_SIZE, x)
    ba2 = BitInt(DEFAULT_SIZE, y)
    result = ba1 * ba2
    assert int(result) == x * y


def test_comparison() -> None:
    """Verify equality, inequality, less-than, and greater-than comparisons."""
    ba1 = BitInt("1010")  # 10
    ba2 = BitInt("1010")  # 10
    ba3 = BitInt("1101")  # 13
    assert ba1 == ba2
    assert ba1 != ba3
    assert ba1 < ba3
    assert ba3 > ba1


def test_bitwise_and() -> None:
    """Verify bitwise AND of two BitInt values."""
    ba1 = BitInt("1010")
    ba2 = BitInt("1101")
    result = ba1 & ba2
    assert int(result) == 0b1000


def test_bitwise_or() -> None:
    """Verify bitwise OR of two BitInt values."""
    ba1 = BitInt("1010")
    ba2 = BitInt("1101")
    result = ba1 | ba2
    assert int(result) == 0b1111


def test_bitwise_xor() -> None:
    """Verify bitwise XOR of two BitInt values."""
    ba1 = BitInt("1010")
    ba2 = BitInt("1101")
    result = ba1 ^ ba2
    assert int(result) == 0b0111


@given(int_range)
def test_signed_bitwise_not(x: int) -> None:
    """Verify bitwise NOT produces -(x+1) for signed values."""
    ba = BitInt(DEFAULT_SIZE, x)
    result = ~ba
    assert int(result) == -x - 1


def test_signed_comparison_semantics() -> None:
    """Verify signed comparison: negative values compare less than zero."""
    b = BitInt(8, -1)
    assert b < 0
    assert b <= 0
    assert not (b > 0)
    assert not (b >= 1)
    assert b == -1


def test_signed_int_still_works() -> None:
    """Verify int() returns correct signed values including min and zero."""
    b = BitInt(63, -1)
    assert int(b) == -1

    b2 = BitInt(63, -(2**63))
    assert int(b2) == -(2**63)

    b3 = BitInt(63, 0)
    assert int(b3) == 0


def test_index_protocol() -> None:
    """Verify __index__ returns the same value as __int__."""
    import operator

    b_signed = BitInt(8, -1)
    assert operator.index(b_signed) == int(b_signed) == -1


def test_signed_always_true() -> None:
    """Verify the signed property is always True for BitInt."""
    b = BitInt(8, 42)
    assert b.signed is True


def test_negative_values() -> None:
    """Verify storage and retrieval of negative values including min."""
    b = BitInt(8, -128)
    assert int(b) == -128

    b2 = BitInt(8, -1)
    assert int(b2) == -1


def test_arithmetic_shift_right() -> None:
    """Shift right is arithmetic for BitInt: fills with sign bit."""
    b = BitInt(8, -8)
    result = b >> 2
    assert int(result) == -2

    b2 = BitInt(8, 8)
    result2 = b2 >> 2
    assert int(result2) == 2


def test_lshift() -> None:
    """Verify left shift by 4 positions."""
    a = BitInt(8, 0b0000_1111)
    b = a << 4
    assert int(b) == 0b1111_0000


def test_floordiv() -> None:
    """Verify floor division of two positive values."""
    a = BitInt(8, 100)
    b = BitInt(8, 10)
    c = a // b
    assert int(c) == 10


def test_floordiv_signed() -> None:
    """Verify floor division with a negative dividend."""
    a = BitInt(8, -100)
    b = BitInt(8, 10)
    c = a // b
    assert int(c) == -10


def test_floordiv_by_zero() -> None:
    """Verify floor division by zero raises ZeroDivisionError."""
    a = BitInt(8, 42)
    b = BitInt(8, 0)
    with pytest.raises(ZeroDivisionError):
        a // b


def test_mod() -> None:
    """Verify modulo of two positive values."""
    a = BitInt(8, 100)
    b = BitInt(8, 30)
    c = a % b
    assert int(c) == 10


def test_mod_signed() -> None:
    """Verify modulo with a negative dividend preserves sign."""
    a = BitInt(8, -7)
    b = BitInt(8, 3)
    c = a % b
    assert int(c) == -1


def test_mod_by_zero() -> None:
    """Verify modulo by zero raises ZeroDivisionError."""
    a = BitInt(8, 42)
    b = BitInt(8, 0)
    with pytest.raises(ZeroDivisionError):
        a % b


def test_reject_size_0() -> None:
    """Verify BitInt(0) raises ValueError."""
    with pytest.raises(ValueError, match="at least 1"):
        BitInt(0)


def test_zeros() -> None:
    """Verify zeros() creates a zero-valued BitInt of given size."""
    z = BitInt.zeros(8)
    assert int(z) == 0
    assert z.is_zero()
    assert z.size == 8


def test_ones() -> None:
    """Verify ones() sets all data bits to 1 with sign bit 0."""
    o = BitInt.ones(8)
    assert int(o) == 255  # All 8 data bits set, sign bit 0
    assert o.size == 8


def test_from_binary() -> None:
    """Verify from_binary() constructs from a binary string."""
    b = BitInt.from_binary("1100")
    assert b.size == 4
    assert int(b) == 0b1100


def test_str() -> None:
    """Verify str() returns the binary representation of data bits."""
    a = BitInt(8, 0b0010_1010)
    assert str(a) == "00101010"


def test_repr() -> None:
    """Verify repr() includes type name and size."""
    a = BitInt(8, 42)
    r = repr(a)
    assert "BitInt" in r
    assert "8" in r


def test_bool_true() -> None:
    """Verify non-zero values are truthy."""
    assert bool(BitInt(8, 1)) is True
    assert bool(BitInt(8, -1)) is True


def test_bool_false() -> None:
    """Verify zero is falsy."""
    assert bool(BitInt(8, 0)) is False


def test_hash() -> None:
    """Verify equal BitInt values produce equal hashes."""
    a = BitInt(8, 42)
    b = BitInt(8, 42)
    assert hash(a) == hash(b)


def test_hash_different_values() -> None:
    """Verify different BitInt values produce different hashes."""
    a = BitInt(8, 42)
    b = BitInt(8, 43)
    assert hash(a) != hash(b)


def test_len() -> None:
    """Verify len() returns the user-visible size."""
    assert len(BitInt(8, 0)) == 8
    assert len(BitInt(16, 0)) == 16


def test_set() -> None:
    """Verify set() updates the value in place."""
    a = BitInt(8, 0)
    a.set(42)
    assert int(a) == 42


def test_set_negative() -> None:
    """Verify set() works with negative values."""
    a = BitInt(8, 0)
    a.set(-5)
    assert int(a) == -5


def test_set_clip() -> None:
    """Verify set_clip() wraps values via two's complement truncation.

    BitInt(N) has internal_size = N+1 bits. set_clip truncates to
    internal_size bits using mask_to_width, giving standard two's
    complement wrapping for out-of-range values.
    """
    a = BitInt(4, 0)
    # 0xFF = 255. BitInt(4) internal_size = 5 bits.
    # 255 in 5 bits: 255 & 31 = 31 = 0b11111. Sign bit = 1. Value = 31 - 32 = -1.
    a.set_clip(0xFF)
    assert int(a) == -1

    # Value in range is stored as-is
    a.set_clip(10)
    assert int(a) == 10

    # Negative value in range is preserved
    a.set_clip(-3)
    assert int(a) == -3


def test_set_clip_large_positive_into_bitint64() -> None:
    """Verify set_clip handles values exceeding i64::MAX in BitInt(64).

    BitInt(64) has range [-2^64, 2^64-1]. Values in [2^63, 2^64-1]
    exceed i64::MAX and go through the slow path in set_clip.
    """
    a = BitInt(64, 0)
    # 2^63 exceeds i64::MAX (2^63 - 1) but fits in BitInt(64)
    a.set_clip(2**63)
    assert int(a) == 2**63

    # 2^64 - 1 is BitInt(64) max value
    a.set_clip(2**64 - 1)
    assert int(a) == 2**64 - 1


def test_set_clip_large_negative_into_bitint64() -> None:
    """Verify set_clip preserves sign for negative values below i64::MIN.

    BitInt(64) min value is -2^64. Values like -2^64 don't fit in i64
    (i64::MIN = -2^63) and go through the slow path.
    Without the fix, the slow path masked to 64 user bits, which cleared
    the sign bit and corrupted negative values.
    """
    a = BitInt(64, 0)
    # -2^64 is the minimum value for BitInt(64)
    a.set_clip(-(2**64))
    assert int(a) == -(2**64)

    # -(2^63 + 1) is just below i64::MIN, triggers slow path
    a.set_clip(-(2**63 + 1))
    assert int(a) == -(2**63 + 1)


def test_set_clip_wrapping_outside_bitint64_range() -> None:
    """Verify set_clip wraps values outside BitInt(64) range.

    Values outside [-2^64, 2^64-1] should wrap via two's complement
    truncation to 65 internal bits.
    """
    a = BitInt(64, 0)
    # 2^64 is one past the max; should wrap to -2^64
    a.set_clip(2**64)
    assert int(a) == -(2**64)


def test_count_ones() -> None:
    """Verify count_ones() returns the number of set bits."""
    a = BitInt(8, 0b1010_1010)
    assert a.count_ones() == 4


def test_count_zeros() -> None:
    """Verify count_zeros() returns the number of unset bits."""
    a = BitInt(8, 0b1010_1010)
    assert a.count_zeros() == 4


def test_is_zero() -> None:
    """Verify is_zero() for zero, positive, and negative values."""
    assert BitInt(8, 0).is_zero()
    assert not BitInt(8, 1).is_zero()
    assert not BitInt(8, -1).is_zero()


def test_bit_access_out_of_range() -> None:
    """Verify accessing a bit beyond the register size raises IndexError."""
    a = BitInt(4, 0)
    with pytest.raises(IndexError):
        _ = a[4]


def test_bit_access_negative_index() -> None:
    """Verify negative indexing accesses from the most significant bit."""
    a = BitInt(8, 0b1000_0000)
    assert a[-1] == 1  # bit 7


def test_add_with_int() -> None:
    """Verify addition with a plain Python int."""
    a = BitInt(8, 100)
    c = a + 50
    assert int(c) == 150


def test_radd() -> None:
    """Verify reverse addition (int + BitInt)."""
    a = BitInt(8, 100)
    c = 50 + a
    assert int(c) == 150


def test_sub_with_int() -> None:
    """Verify subtraction with a plain Python int."""
    a = BitInt(8, 100)
    c = a - 30
    assert int(c) == 70


def test_comparison_le() -> None:
    """Verify less-than-or-equal comparison."""
    a = BitInt(8, 10)
    b = BitInt(8, 10)
    c = BitInt(8, 20)
    assert a <= b
    assert a <= c


def test_comparison_ge() -> None:
    """Verify greater-than-or-equal comparison."""
    a = BitInt(8, 20)
    b = BitInt(8, 20)
    c = BitInt(8, 10)
    assert a >= b
    assert a >= c


def test_interop_add_with_bituint() -> None:
    """Verify addition between BitInt and BitUInt."""
    from pecos import BitUInt

    a = BitInt(8, 100)
    b = BitUInt(8, 50)
    c = a + b
    assert int(c) == 150


def test_size_property() -> None:
    """Verify the size property returns the user-visible size."""
    assert BitInt(16, 0).size == 16
