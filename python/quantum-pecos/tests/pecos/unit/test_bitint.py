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
    ba = BitInt("0000")
    ba[2] = 1
    assert int(ba) == 0b0100


def test_get_bit() -> None:
    ba = BitInt("1010")
    assert ba[2] == 0
    assert ba[3] == 1


def test_to_int() -> None:
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
    b = BitInt(1, 0)
    assert int(b) == 0


@given(int_range, int_range)
def test_addition(x: int, y: int) -> None:
    assume(MIN <= x + y <= MAX)
    ba1 = BitInt(DEFAULT_SIZE, x)
    ba2 = BitInt(DEFAULT_SIZE, y)
    result = ba1 + ba2
    assert int(result) == x + y


def test_subtraction() -> None:
    ba1 = BitInt("1101")  # 13
    ba2 = BitInt("1010")  # 10
    result = ba1 - ba2
    assert int(result) == 3


@given(int_range, int_range)
def test_multiplication(x: int, y: int) -> None:
    assume(MIN <= x * y <= MAX)
    ba1 = BitInt(DEFAULT_SIZE, x)
    ba2 = BitInt(DEFAULT_SIZE, y)
    result = ba1 * ba2
    assert int(result) == x * y


def test_comparison() -> None:
    ba1 = BitInt("1010")  # 10
    ba2 = BitInt("1010")  # 10
    ba3 = BitInt("1101")  # 13
    assert ba1 == ba2
    assert ba1 != ba3
    assert ba1 < ba3
    assert ba3 > ba1


def test_bitwise_and() -> None:
    ba1 = BitInt("1010")
    ba2 = BitInt("1101")
    result = ba1 & ba2
    assert int(result) == 0b1000


def test_bitwise_or() -> None:
    ba1 = BitInt("1010")
    ba2 = BitInt("1101")
    result = ba1 | ba2
    assert int(result) == 0b1111


def test_bitwise_xor() -> None:
    ba1 = BitInt("1010")
    ba2 = BitInt("1101")
    result = ba1 ^ ba2
    assert int(result) == 0b0111


@given(int_range)
def test_signed_bitwise_not(x: int) -> None:
    ba = BitInt(DEFAULT_SIZE, x)
    result = ~ba
    assert int(result) == -x - 1


def test_signed_comparison_semantics() -> None:
    b = BitInt(8, -1)
    assert b < 0
    assert b <= 0
    assert not (b > 0)
    assert not (b >= 1)
    assert b == -1


def test_signed_int_still_works() -> None:
    b = BitInt(63, -1)
    assert int(b) == -1

    b2 = BitInt(63, -(2**63))
    assert int(b2) == -(2**63)

    b3 = BitInt(63, 0)
    assert int(b3) == 0


def test_index_protocol() -> None:
    import operator

    b_signed = BitInt(8, -1)
    assert operator.index(b_signed) == int(b_signed) == -1


def test_signed_always_true() -> None:
    b = BitInt(8, 42)
    assert b.signed is True


def test_negative_values() -> None:
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
    a = BitInt(8, 0b0000_1111)
    b = a << 4
    assert int(b) == 0b1111_0000


def test_floordiv() -> None:
    a = BitInt(8, 100)
    b = BitInt(8, 10)
    c = a // b
    assert int(c) == 10


def test_floordiv_signed() -> None:
    a = BitInt(8, -100)
    b = BitInt(8, 10)
    c = a // b
    assert int(c) == -10


def test_floordiv_by_zero() -> None:
    a = BitInt(8, 42)
    b = BitInt(8, 0)
    with pytest.raises(ZeroDivisionError):
        a // b


def test_mod() -> None:
    a = BitInt(8, 100)
    b = BitInt(8, 30)
    c = a % b
    assert int(c) == 10


def test_mod_signed() -> None:
    a = BitInt(8, -7)
    b = BitInt(8, 3)
    c = a % b
    assert int(c) == -1


def test_mod_by_zero() -> None:
    a = BitInt(8, 42)
    b = BitInt(8, 0)
    with pytest.raises(ZeroDivisionError):
        a % b


def test_reject_size_0() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BitInt(0)


def test_zeros() -> None:
    z = BitInt.zeros(8)
    assert int(z) == 0
    assert z.is_zero()
    assert z.size == 8


def test_ones() -> None:
    o = BitInt.ones(8)
    assert int(o) == 255  # All 8 data bits set, sign bit 0
    assert o.size == 8


def test_from_binary() -> None:
    b = BitInt.from_binary("1100")
    assert b.size == 4
    assert int(b) == 0b1100


def test_str() -> None:
    a = BitInt(8, 0b0010_1010)
    assert str(a) == "00101010"


def test_repr() -> None:
    a = BitInt(8, 42)
    r = repr(a)
    assert "BitInt" in r
    assert "8" in r


def test_bool_true() -> None:
    assert bool(BitInt(8, 1)) is True
    assert bool(BitInt(8, -1)) is True


def test_bool_false() -> None:
    assert bool(BitInt(8, 0)) is False


def test_hash() -> None:
    a = BitInt(8, 42)
    b = BitInt(8, 42)
    assert hash(a) == hash(b)


def test_hash_different_values() -> None:
    a = BitInt(8, 42)
    b = BitInt(8, 43)
    assert hash(a) != hash(b)


def test_len() -> None:
    assert len(BitInt(8, 0)) == 8
    assert len(BitInt(16, 0)) == 16


def test_set() -> None:
    a = BitInt(8, 0)
    a.set(42)
    assert int(a) == 42


def test_set_negative() -> None:
    a = BitInt(8, 0)
    a.set(-5)
    assert int(a) == -5


def test_set_clip() -> None:
    a = BitInt(4, 0)
    a.set_clip(0xFF)
    assert int(a) == 0x0F


def test_count_ones() -> None:
    a = BitInt(8, 0b1010_1010)
    assert a.count_ones() == 4


def test_count_zeros() -> None:
    a = BitInt(8, 0b1010_1010)
    assert a.count_zeros() == 4


def test_is_zero() -> None:
    assert BitInt(8, 0).is_zero()
    assert not BitInt(8, 1).is_zero()
    assert not BitInt(8, -1).is_zero()


def test_bit_access_out_of_range() -> None:
    a = BitInt(4, 0)
    with pytest.raises(IndexError):
        _ = a[4]


def test_bit_access_negative_index() -> None:
    a = BitInt(8, 0b1000_0000)
    assert a[-1] == 1  # bit 7


def test_add_with_int() -> None:
    a = BitInt(8, 100)
    c = a + 50
    assert int(c) == 150


def test_radd() -> None:
    a = BitInt(8, 100)
    c = 50 + a
    assert int(c) == 150


def test_sub_with_int() -> None:
    a = BitInt(8, 100)
    c = a - 30
    assert int(c) == 70


def test_comparison_le() -> None:
    a = BitInt(8, 10)
    b = BitInt(8, 10)
    c = BitInt(8, 20)
    assert a <= b
    assert a <= c


def test_comparison_ge() -> None:
    a = BitInt(8, 20)
    b = BitInt(8, 20)
    c = BitInt(8, 10)
    assert a >= b
    assert a >= c


def test_interop_add_with_bituint() -> None:
    from pecos import BitUInt

    a = BitInt(8, 100)
    b = BitUInt(8, 50)
    c = a + b
    assert int(c) == 150


def test_size_property() -> None:
    assert BitInt(16, 0).size == 16
