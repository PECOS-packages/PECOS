from typing import Final

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from pecos.engines.cvm.binarray2 import BinArray2 as BinArray

# from pecos.engines.cvm.binarray import BinArray

DEFAULT_SIZE: Final = 63
MIN: Final = -(2**DEFAULT_SIZE)
MAX: Final = 2**DEFAULT_SIZE - 1
int_range = st.integers(min_value=MIN, max_value=MAX)


@given(st.text(alphabet=["0", "1"], min_size=1))
def test_init(x):
    ba = BinArray(x)
    assert ba == f"0b{x}"


def test_set_bit():
    ba = BinArray("0000")
    ba[2] = 1
    assert ba == 0b0100


def test_get_bit():
    ba = BinArray("1010")
    assert ba[2] == 0
    assert ba[3] == 1


def test_to_int():
    ba = BinArray("1010")
    assert int(ba) == 10


@given(int_range, int_range)
def test_addition(x, y):
    assume(MIN <= x + y <= MAX)
    ba1 = BinArray(DEFAULT_SIZE, x)
    ba2 = BinArray(DEFAULT_SIZE, y)
    result = ba1 + ba2
    assert int(result) == x + y


def test_subtraction():
    ba1 = BinArray("1101")  # 13
    ba2 = BinArray("1010")  # 10
    result = ba1 - ba2
    assert int(result) == 3


@given(int_range, int_range)
def test_multiplication(x, y):
    assume(MIN <= x * y <= MAX)
    ba1 = BinArray(DEFAULT_SIZE, x)
    ba2 = BinArray(DEFAULT_SIZE, y)
    result = ba1 * ba2
    assert int(result) == x * y


def test_comparison():
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1010")  # 10
    ba3 = BinArray("1101")  # 13
    assert ba1 == ba2
    assert ba1 != ba3
    assert ba1 != ba3
    assert ba1 < ba3
    assert ba3 > ba1


def test_bitwise_and():
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1101")  # 13
    result = ba1 & ba2
    assert result == 0b1000


def test_bitwise_or():
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1101")  # 13
    result = ba1 | ba2
    assert result == 0b1111


def test_bitwise_xor():
    ba1 = BinArray("1010")  # 10
    ba2 = BinArray("1101")  # 13
    result = ba1 ^ ba2
    assert result == 0b0111


def test_unsigned_bitwise_not():
    ba = BinArray("1010", dtype=np.uint64)  # 10
    result = ~ba
    assert result == 0b0101


@given(int_range)
def test_signed_bitwise_not(x):
    ba = BinArray(DEFAULT_SIZE, x)
    result = ~ba
    assert int(result) == -x - 1  # (two's complement)
