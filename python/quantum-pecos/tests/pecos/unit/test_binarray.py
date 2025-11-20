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


"""Tests for BinArray binary array operations."""

from typing import Final

import pecos as pc
from hypothesis import assume, given
from hypothesis import strategies as st
from pecos.engines.cvm.binarray import BinArray

DEFAULT_SIZE: Final = 63
MIN: Final = -(2**DEFAULT_SIZE)
MAX: Final = 2**DEFAULT_SIZE - 1
int_range = st.integers(min_value=MIN, max_value=MAX)


@given(st.text(alphabet=["0", "1"], min_size=1))
def test_init(x: str) -> None:
    """Test BinArray initialization from binary string."""
    ba = BinArray(x)
    assert ba == f"0b{x}"


def test_set_bit() -> None:
    """Test setting individual bits in BinArray."""
    ba = BinArray("0000")
    ba[2] = 1
    assert ba == 0b0100


def test_get_bit() -> None:
    """Test getting individual bits from BinArray."""
    ba = BinArray("1010")
    assert ba[2] == 0
    assert ba[3] == 1


def test_to_int() -> None:
    """Test converting BinArray to integer."""
    ba = BinArray("1010")
    assert int(ba) == 10


@given(int_range, int_range)
def test_addition(x: int, y: int) -> None:
    """Test BinArray addition operation."""
    assume(MIN <= x + y <= MAX)
    ba1 = BinArray(DEFAULT_SIZE, x)
    ba2 = BinArray(DEFAULT_SIZE, y)
    result = ba1 + ba2
    assert int(result) == x + y


def test_subtraction() -> None:
    """Test BinArray subtraction operation."""
    ba1 = BinArray("1101")  # 13
    ba2 = BinArray("1010")  # 10
    result = ba1 - ba2
    assert int(result) == 3


@given(int_range, int_range)
def test_multiplication(x: int, y: int) -> None:
    """Test BinArray multiplication operation."""
    assume(MIN <= x * y <= MAX)
    ba1 = BinArray(DEFAULT_SIZE, x)
    ba2 = BinArray(DEFAULT_SIZE, y)
    result = ba1 * ba2
    assert int(result) == x * y


def test_comparison() -> None:
    """Test BinArray comparison operations."""
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1010")  # 10
    ba3 = BinArray("1101")  # 13
    assert ba1 == ba2
    assert ba1 != ba3
    assert ba1 != ba3
    assert ba1 < ba3
    assert ba3 > ba1


def test_bitwise_and() -> None:
    """Test BinArray bitwise AND operation."""
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1101")  # 13
    result = ba1 & ba2
    assert result == 0b1000


def test_bitwise_or() -> None:
    """Test BinArray bitwise OR operation."""
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1101")  # 13
    result = ba1 | ba2
    assert result == 0b1111


def test_bitwise_xor() -> None:
    """Test BinArray bitwise XOR operation."""
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1101")  # 13
    result = ba1 ^ ba2
    assert result == 0b0111


def test_unsigned_bitwise_not() -> None:
    """Test BinArray bitwise NOT operation for unsigned data."""
    ba = BinArray("1010", dtype=pc.u64)  # 10
    result = ~ba
    assert result == 0b0101


@given(int_range)
def test_signed_bitwise_not(x: int) -> None:
    """Test BinArray bitwise NOT operation for signed data."""
    ba = BinArray(DEFAULT_SIZE, x)
    result = ~ba
    assert int(result) == -x - 1  # (two's complement)
