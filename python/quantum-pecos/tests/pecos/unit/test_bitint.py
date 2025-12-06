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
