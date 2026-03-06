# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for BitUInt unsigned fixed-width integer type."""

import pytest

from pecos import BitInt, BitUInt


class TestConstruction:
    """Test BitUInt construction."""

    def test_basic(self):
        u = BitUInt(8)
        assert u.size == 8
        assert not u.signed
        assert int(u) == 0

    def test_with_value(self):
        u = BitUInt(8, 42)
        assert int(u) == 42

    def test_binary_string(self):
        u = BitUInt("1010")
        assert u.size == 4
        assert int(u) == 0b1010

    def test_64bit(self):
        u = BitUInt(64, 1)
        assert u.size == 64
        assert int(u) == 1

    def test_64bit_max(self):
        u = BitUInt(64, 2**64 - 1)
        assert int(u) == 2**64 - 1

    def test_reject_size_0(self):
        with pytest.raises(ValueError, match="at least 1"):
            BitUInt(0)

    def test_masking(self):
        u = BitUInt(4, 0xFF)
        assert int(u) == 0x0F

    def test_from_binary(self):
        u = BitUInt.from_binary("1100")
        assert u.size == 4
        assert int(u) == 0b1100

    def test_zeros(self):
        u = BitUInt.zeros(8)
        assert int(u) == 0
        assert u.is_zero()

    def test_ones(self):
        u = BitUInt.ones(8)
        assert int(u) == 0xFF


class TestIntAlwaysNonNegative:
    """The core motivation for BitUInt: int() always returns >= 0."""

    def test_1bit_value_1(self):
        u = BitUInt(1, 1)
        assert int(u) == 1  # Not -1 like BitInt(1, 1, signed=True)

    def test_1bit_all_ones(self):
        u = BitUInt.ones(1)
        assert int(u) == 1

    def test_8bit_max(self):
        u = BitUInt(8, 255)
        assert int(u) == 255

    def test_contrast_with_bitint(self):
        """Show the difference vs signed BitInt."""
        # Both BitInt(1,1) and BitUInt(1,1) return 1
        # The difference: BitInt can represent -1, BitUInt cannot
        bi_pos = BitInt(1, 1)
        bu = BitUInt(1, 1)
        assert int(bi_pos) == 1  # signed 1-bit with N+1 sign bit: 1 is positive
        assert int(bu) == 1  # unsigned: 1 means 1

        bi_neg = BitInt(1, -1)
        assert int(bi_neg) == -1  # signed: -1 is representable


class TestArithmetic:
    """Test arithmetic operations."""

    def test_add(self):
        a = BitUInt(8, 100)
        b = BitUInt(8, 50)
        c = a + b
        assert int(c) == 150

    def test_add_overflow(self):
        a = BitUInt(8, 200)
        b = BitUInt(8, 100)
        c = a + b
        assert int(c) == 44  # (200+100) % 256

    def test_sub_underflow(self):
        a = BitUInt(8, 5)
        b = BitUInt(8, 10)
        c = a - b
        assert int(c) == 251  # wraps: (5 - 10) % 256

    def test_sub(self):
        a = BitUInt(8, 100)
        b = BitUInt(8, 30)
        c = a - b
        assert int(c) == 70

    def test_mul(self):
        a = BitUInt(8, 10)
        b = BitUInt(8, 5)
        c = a * b
        assert int(c) == 50

    def test_floordiv(self):
        a = BitUInt(8, 100)
        b = BitUInt(8, 10)
        c = a // b
        assert int(c) == 10

    def test_mod(self):
        a = BitUInt(8, 100)
        b = BitUInt(8, 30)
        c = a % b
        assert int(c) == 10

    def test_add_with_int(self):
        a = BitUInt(8, 100)
        c = a + 50
        assert int(c) == 150

    def test_radd(self):
        a = BitUInt(8, 100)
        c = 50 + a
        assert int(c) == 150

    def test_rsub(self):
        a = BitUInt(8, 30)
        c = 100 - a
        assert int(c) == 70

    def test_rmul(self):
        a = BitUInt(8, 5)
        c = 10 * a
        assert int(c) == 50

    def test_floordiv_by_zero(self):
        a = BitUInt(8, 42)
        b = BitUInt(8, 0)
        with pytest.raises(ZeroDivisionError):
            a // b

    def test_mod_by_zero(self):
        a = BitUInt(8, 42)
        b = BitUInt(8, 0)
        with pytest.raises(ZeroDivisionError):
            a % b


class TestBitwise:
    """Test bitwise operations."""

    def test_xor(self):
        a = BitUInt(8, 0b1010_1010)
        b = BitUInt(8, 0b0101_0101)
        c = a ^ b
        assert int(c) == 0xFF

    def test_and(self):
        a = BitUInt(8, 0b1010_1010)
        b = BitUInt(8, 0b1111_0000)
        c = a & b
        assert int(c) == 0b1010_0000

    def test_or(self):
        a = BitUInt(8, 0b1010_0000)
        b = BitUInt(8, 0b0000_0101)
        c = a | b
        assert int(c) == 0b1010_0101

    def test_not(self):
        a = BitUInt(8, 0b1010_1010)
        b = ~a
        assert int(b) == 0b0101_0101

    def test_lshift(self):
        a = BitUInt(8, 0b0000_1111)
        b = a << 4
        assert int(b) == 0b1111_0000

    def test_rshift(self):
        a = BitUInt(8, 0b1111_0000)
        b = a >> 4
        assert int(b) == 0b0000_1111


class TestComparison:
    """Test comparison operations."""

    def test_eq(self):
        a = BitUInt(8, 42)
        b = BitUInt(8, 42)
        assert a == b

    def test_ne(self):
        a = BitUInt(8, 42)
        b = BitUInt(8, 43)
        assert a != b

    def test_lt(self):
        a = BitUInt(8, 10)
        b = BitUInt(8, 20)
        assert a < b

    def test_gt(self):
        a = BitUInt(8, 20)
        b = BitUInt(8, 10)
        assert a > b

    def test_le(self):
        a = BitUInt(8, 10)
        b = BitUInt(8, 10)
        c = BitUInt(8, 20)
        assert a <= b
        assert a <= c

    def test_ge(self):
        a = BitUInt(8, 20)
        b = BitUInt(8, 20)
        c = BitUInt(8, 10)
        assert a >= b
        assert a >= c

    def test_compare_with_int(self):
        a = BitUInt(8, 42)
        assert a == 42
        assert a < 43
        assert a > 41

    def test_unsigned_semantics(self):
        """All comparisons are unsigned."""
        a = BitUInt(8, 200)
        b = BitUInt(8, 100)
        assert a > b  # unsigned: 200 > 100


class TestBitAccess:
    """Test bit get/set."""

    def test_getitem(self):
        a = BitUInt(8, 0b1010_0101)
        assert a[0] == 1
        assert a[1] == 0
        assert a[2] == 1

    def test_setitem(self):
        a = BitUInt(8, 0)
        a[0] = 1
        assert int(a) == 1
        a[7] = 1
        assert int(a) == 0b1000_0001

    def test_out_of_range(self):
        a = BitUInt(4, 0)
        with pytest.raises(IndexError):
            _ = a[4]

    def test_negative_index(self):
        a = BitUInt(8, 0b1000_0000)
        assert a[-1] == 1  # bit 7

    def test_len(self):
        a = BitUInt(8, 0)
        assert len(a) == 8


class TestInteropWithBitInt:
    """Test interop between BitUInt and BitInt."""

    def test_set_from_bitint(self):
        u = BitUInt(8, 0)
        b = BitInt(8, 42)
        u.set(b)
        assert int(u) == 42

    def test_set_clip(self):
        u = BitUInt(4, 0)
        u.set_clip(0xFF)
        assert int(u) == 0x0F

    def test_add_with_bitint(self):
        u = BitUInt(8, 100)
        b = BitInt(8, 50)
        c = u + b
        assert int(c) == 150


class TestProperties:
    """Test properties and conversions."""

    def test_size(self):
        u = BitUInt(16, 0)
        assert u.size == 16

    def test_signed_always_false(self):
        u = BitUInt(8, 0)
        assert u.signed is False

    def test_str(self):
        u = BitUInt(8, 0b1010_0101)
        assert str(u) == "10100101"

    def test_repr(self):
        u = BitUInt(8, 0b1010_0101)
        r = repr(u)
        assert "BitUInt" in r
        assert "8" in r

    def test_bool_true(self):
        u = BitUInt(8, 1)
        assert bool(u) is True

    def test_bool_false(self):
        u = BitUInt(8, 0)
        assert bool(u) is False

    def test_hash(self):
        a = BitUInt(8, 42)
        b = BitUInt(8, 42)
        assert hash(a) == hash(b)

    def test_count_ones(self):
        u = BitUInt(8, 0b1010_1010)
        assert u.count_ones() == 4

    def test_count_zeros(self):
        u = BitUInt(8, 0b1010_1010)
        assert u.count_zeros() == 4

    def test_is_zero(self):
        assert BitUInt(8, 0).is_zero()
        assert not BitUInt(8, 1).is_zero()

    def test_index(self):
        """Test that BitUInt can be used as an index (via __index__)."""
        u = BitUInt(8, 3)
        lst = [10, 20, 30, 40]
        assert lst[u] == 40  # noqa: RUF015
