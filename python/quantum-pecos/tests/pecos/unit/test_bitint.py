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


"""Tests for BitInt binary integer operations."""

from typing import Final

import pecos as pc
from hypothesis import assume, given
from hypothesis import strategies as st
from pecos import BitInt

# BitInt uses actual fixed-width arithmetic, unlike the original BitInt which used
# Python's arbitrary-precision int internally with i64 dtype. Use 64 bits to match i64 range.
DEFAULT_SIZE: Final = 64
MIN: Final = -(2 ** (DEFAULT_SIZE - 1))  # -2^63 for signed 64-bit
MAX: Final = 2 ** (DEFAULT_SIZE - 1) - 1  # 2^63 - 1 for signed 64-bit
int_range = st.integers(min_value=MIN, max_value=MAX)


@given(st.text(alphabet=["0", "1"], min_size=1))
def test_init(x: str) -> None:
    """Test BitInt initialization from binary string."""
    ba = BitInt(x)
    assert ba == f"0b{x}"


def test_set_bit() -> None:
    """Test setting individual bits in BitInt."""
    ba = BitInt("0000")
    ba[2] = 1
    assert ba == 0b0100


def test_get_bit() -> None:
    """Test getting individual bits from BitInt."""
    ba = BitInt("1010")
    assert ba[2] == 0
    assert ba[3] == 1


def test_to_int() -> None:
    """Test converting BitInt to integer."""
    ba = BitInt("1010")
    assert int(ba) == 10


@given(int_range, int_range)
def test_addition(x: int, y: int) -> None:
    """Test BitInt addition operation."""
    assume(MIN <= x + y <= MAX)
    ba1 = BitInt(DEFAULT_SIZE, x)
    ba2 = BitInt(DEFAULT_SIZE, y)
    result = ba1 + ba2
    assert int(result) == x + y


def test_subtraction() -> None:
    """Test BitInt subtraction operation."""
    ba1 = BitInt("1101")  # 13
    ba2 = BitInt("1010")  # 10
    result = ba1 - ba2
    assert int(result) == 3


@given(int_range, int_range)
def test_multiplication(x: int, y: int) -> None:
    """Test BitInt multiplication operation."""
    assume(MIN <= x * y <= MAX)
    ba1 = BitInt(DEFAULT_SIZE, x)
    ba2 = BitInt(DEFAULT_SIZE, y)
    result = ba1 * ba2
    assert int(result) == x * y


def test_comparison() -> None:
    """Test BitInt comparison operations."""
    ba1 = BitInt("1010")  # 10
    ba2 = BitInt("1010")  # 10
    ba3 = BitInt("1101")  # 13
    assert ba1 == ba2
    assert ba1 != ba3
    assert ba1 != ba3
    assert ba1 < ba3
    assert ba3 > ba1


def test_bitwise_and() -> None:
    """Test BitInt bitwise AND operation."""
    ba1 = BitInt("1010")  # 10
    ba2 = BitInt("1101")  # 13
    result = ba1 & ba2
    assert result == 0b1000


def test_bitwise_or() -> None:
    """Test BitInt bitwise OR operation."""
    ba1 = BitInt("1010")  # 10
    ba2 = BitInt("1101")  # 13
    result = ba1 | ba2
    assert result == 0b1111


def test_bitwise_xor() -> None:
    """Test BitInt bitwise XOR operation."""
    ba1 = BitInt("1010")  # 10
    ba2 = BitInt("1101")  # 13
    result = ba1 ^ ba2
    assert result == 0b0111


def test_unsigned_bitwise_not() -> None:
    """Test BitInt bitwise NOT operation for unsigned data."""
    ba = BitInt("1010", dtype=pc.u64)  # 10
    result = ~ba
    assert result == 0b0101


@given(int_range)
def test_signed_bitwise_not(x: int) -> None:
    """Test BitInt bitwise NOT operation for signed data."""
    ba = BitInt(DEFAULT_SIZE, x)
    result = ~ba
    assert int(result) == -x - 1  # (two's complement)


# ============================================================================
# Unsigned / signed boundary tests
# ============================================================================

UMAX: Final = 2**64 - 1  # 0xFFFFFFFF_FFFFFFFF


def test_unsigned_int_conversion() -> None:
    """Test that int() on unsigned BitInt returns a positive value, not negative."""
    b = BitInt(64, UMAX, signed=False)
    val = int(b)
    assert val == UMAX
    assert val > 0


def test_unsigned_construction_large_value() -> None:
    """Test constructing unsigned BitInt with values >= 2^63."""
    b = BitInt(64, 2**63, signed=False)
    assert int(b) == 2**63

    b2 = BitInt(64, 2**64 - 1, signed=False)
    assert int(b2) == 2**64 - 1


def test_unsigned_set_upper_bits() -> None:
    """Test setting upper bits on unsigned 64-bit BitInt and converting to int."""
    b = BitInt(64, signed=False)
    for i in range(32, 64):
        b[i] = 1
    val = int(b)
    expected = 2**64 - 2**32  # 0xFFFFFFFF_00000000
    assert val == expected
    assert val > 0


def test_index_protocol() -> None:
    """Test that __index__ works and matches __int__."""
    import operator

    b_unsigned = BitInt(64, UMAX, signed=False)
    assert operator.index(b_unsigned) == int(b_unsigned) == UMAX

    b_signed = BitInt(64, -1)
    assert operator.index(b_signed) == int(b_signed) == -1


def test_signed_comparison_semantics() -> None:
    """Test that signed BitInt with all bits set (= -1) compares less than 0."""
    b = BitInt(64, -1)
    assert b < 0
    assert b <= 0
    assert not (b > 0)
    assert not (b >= 1)
    assert b == -1


def test_unsigned_comparison_semantics() -> None:
    """Test that unsigned BitInt with all bits set compares greater than 0."""
    b = BitInt(64, UMAX, signed=False)
    assert b > 0
    assert b >= 0
    assert not (b < 0)
    assert b == UMAX


def test_unsigned_rng_seed() -> None:
    """Test that unsigned BitInt value works as an RNG seed."""
    from pecos_rslib import RngPcg

    b = BitInt(64, signed=False)
    for i in range(32, 64):
        b[i] = 1

    pcg = RngPcg()
    pcg.srandom(int(b))  # should not raise


def test_signed_int_still_works() -> None:
    """Test that signed BitInt int() conversion still returns negative values correctly."""
    b = BitInt(64, -1)
    assert int(b) == -1

    b2 = BitInt(64, -(2**63))
    assert int(b2) == -(2**63)

    b3 = BitInt(64, 0)
    assert int(b3) == 0
