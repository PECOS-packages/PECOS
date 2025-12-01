"""Binary array implementation for the PECOS classical virtual machine.

This module provides the BinArray class for efficient binary array operations
within the classical virtual machine (CVM) framework. It supports various
binary representations and operations needed for classical computations
in quantum error correction simulations.
"""

# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc
from pecos.reps.pyphir import unsigned_data_types

if TYPE_CHECKING:
    from pecos.typing import Integer


class BinArray:
    """As opposed to the original unsigned 32-bit BinArray, this class defaults to signed 64-bit type."""

    __hash__ = None  # BinArray instances are not hashable since __eq__ returns BinArray

    def __init__(
        self,
        size: int | str,
        value: int | str | BinArray | None = 0,
        dtype: type[Integer] = pc.i64,
    ) -> None:
        """Initialize a binary array with given size and value.

        Args:
            size: The number of bits in the array. Can be an integer or a binary
                string (e.g., '1101'). If a binary string is provided, its length
                becomes the size and its value is used.
            value: The initial value for the array. Can be an integer, binary string,
                or another BinArray. Defaults to 0.
            dtype: The PECOS integer data type to use for internal storage.
                Defaults to pc.i64 for signed 64-bit integers.
        """
        self.size = size
        self.value = None
        self.dtype = dtype

        if isinstance(size, int):
            self.size = size

            if value is not None:
                self.set(value)
        elif isinstance(size, str):
            self.size = len(size)
            value = int(size, 2)
            self.set(value)

    def set(self, value: int | str | BinArray) -> None:
        """Set the binary array value.

        Args:
            value: New value as integer, binary string, or BinArray.
        """
        if isinstance(value, self.dtype):
            self.value = value
        elif isinstance(value, BinArray):
            self.value = value.value
        else:
            if isinstance(value, str):
                value = int(value, 2)

            self.value = self.dtype(value)

    def new_val(self, value: int | str | BinArray) -> BinArray:
        """Create a new BinArray with the given value.

        Args:
            value: Value for the new BinArray.

        Returns:
            New BinArray instance with the specified value.
        """
        b = BinArray(self.size, value, self.dtype)
        if self.dtype in unsigned_data_types.values():
            b.clamp(self.size)
        return b

    def num_bits(self) -> int:
        """Get the number of bits required to represent the current value.

        Returns:
            Number of bits in the binary representation.
        """
        return len(f"{self.value:b}")

    def check_size(self) -> None:
        """Check if the current value fits within the allocated size.

        Raises:
            Exception: If the value requires more bits than allocated.
        """
        if self.num_bits() > self.size:
            num = self.num_bits()
            val = f"{self.value:b}"
            msg = f'Number of bits ({num}) exceeds size ({self.size}) for bits "{val}"!'
            raise Exception(msg)

    def clamp(self, size: int) -> None:
        """Clamp the value to fit within the specified bit size.

        Args:
            size: Maximum number of bits allowed.
        """
        if self.num_bits() > size:
            bits = format(self.value, f"0{size}b")
            bits = int(bits[-size:], 2)
            self.value = self.dtype(bits)

    def set_clip(self, value: int | BinArray) -> None:
        """Set value with clipping to fit within the allocated size.

        Args:
            value: Value to set, clipped if necessary.
        """
        value = int(value)

        if len(f"{value:b}") > self.size:
            bits = format(value, f"0{self.size}b")
            bits = int(bits[-self.size :], 2)
            self.value = self.dtype(bits)
        else:
            self.value = self.dtype(value)

    def _set_clip(self, ba: int | BinArray) -> None:
        """Take values up to the size of this BinArray. If this BinArray array is larger, fill with zeros."""
        if isinstance(ba, int):
            ba = self.new_val(ba)

        if isinstance(ba, BinArray):
            self._set_clip(ba)
        else:
            msg = "Expected int or BinArray!"
            raise TypeError(msg)

    def __getitem__(self, item: int) -> int:
        """Get bit value at specified index.

        Args:
            item: Index of the bit to retrieve.

        Returns:
            Bit value at the specified index.
        """
        return int(str(self)[self.size - item - 1])

    def __setitem__(self, key: int, value: int | str) -> None:
        """Set bit value at specified index.

        Args:
            key: Index of the bit to set.
            value: New bit value.
        """
        b = list(str(self))
        b[self.size - key - 1] = str(value)
        b = "".join(b)

        self.set(b)

    def __str__(self) -> str:
        """Return string representation of the binary array.

        Returns:
            Binary string representation.
        """
        self.check_size()
        return format(self.value, f"0{self.size}b")

    def __repr__(self) -> str:
        """Return detailed string representation of the binary array.

        Returns:
            Detailed string representation for debugging.
        """
        return self.__str__()

    def __int__(self) -> int:
        """Return integer representation of the binary array.

        Returns:
            Integer value of the binary array.
        """
        return int(self.value)

    def __len__(self) -> int:
        """Return the size of the binary array.

        Returns:
            Number of bits in the array.
        """
        return self.size

    def do_binop(self, op: str, other: BinArray | str | int) -> BinArray:
        """Perform binary operation with another value.

        Args:
            op: Name of the operation method to call.
            other: Other operand for the binary operation.

        Returns:
            New BinArray with the result of the operation.
        """
        if hasattr(other, "value") and isinstance(other.value, self.dtype):
            value = other.value
        elif isinstance(other, str):
            value = self.dtype(int(other, 2))
        else:
            value = self.dtype(other)

        op = getattr(self.value, op)
        value = op(value)

        return self.new_val(value)

    def __bool__(self) -> bool:
        """Return boolean representation of the binary array.

        Returns:
            True if the value is non-zero, False otherwise.
        """
        return bool(self.value)

    def __xor__(self, other: BinArray | str | int) -> BinArray:
        """Perform bitwise XOR operation.

        Args:
            other: Other operand for XOR operation.

        Returns:
            New BinArray with XOR result.
        """
        return self.do_binop("__xor__", other)

    def __and__(self, other: BinArray | str | int) -> BinArray:
        """Perform bitwise AND operation.

        Args:
            other: Other operand for AND operation.

        Returns:
            New BinArray with AND result.
        """
        return self.do_binop("__and__", other)

    def __or__(self, other: BinArray | str | int) -> BinArray:
        """Perform bitwise OR operation.

        Args:
            other: Other operand for OR operation.

        Returns:
            New BinArray with OR result.
        """
        return self.do_binop("__or__", other)

    def __eq__(self, other: BinArray | str | int) -> BinArray:
        """Check equality with another value.

        Args:
            other: Other value for comparison.

        Returns:
            New BinArray with equality result.
        """
        return self.do_binop("__eq__", other)

    def __ne__(self, other: BinArray | str | int) -> BinArray:
        """Check inequality with another value.

        Args:
            other: Other value for comparison.

        Returns:
            New BinArray with inequality result.
        """
        return self.do_binop("__ne__", other)

    def __lt__(self, other: BinArray | str | int) -> BinArray:
        """Check if less than another value.

        Args:
            other: Other value for comparison.

        Returns:
            New BinArray with less-than result.
        """
        return self.do_binop("__lt__", other)

    def __gt__(self, other: BinArray | str | int) -> BinArray:
        """Check if greater than another value.

        Args:
            other: Other value for comparison.

        Returns:
            New BinArray with greater-than result.
        """
        return self.do_binop("__gt__", other)

    def __le__(self, other: BinArray | str | int) -> BinArray:
        """Check if less than or equal to another value.

        Args:
            other: Other value for comparison.

        Returns:
            New BinArray with less-than-or-equal result.
        """
        return self.do_binop("__le__", other)

    def __ge__(self, other: BinArray | str | int) -> BinArray:
        """Check if greater than or equal to another value.

        Args:
            other: Other value for comparison.

        Returns:
            New BinArray with greater-than-or-equal result.
        """
        return self.do_binop("__ge__", other)

    def __add__(self, other: BinArray | str | int) -> BinArray:
        """Perform addition with another value.

        Args:
            other: Other operand for addition.

        Returns:
            New BinArray with addition result.
        """
        return self.do_binop("__add__", other)

    def __sub__(self, other: BinArray | str | int) -> BinArray:
        """Perform subtraction with another value.

        Args:
            other: Other operand for subtraction.

        Returns:
            New BinArray with subtraction result.
        """
        return self.do_binop("__sub__", other)

    def __rshift__(self, other: BinArray | str | int) -> BinArray:
        """Perform right bit shift operation.

        Args:
            other: Number of positions to shift right.

        Returns:
            New BinArray with right shift result.
        """
        return self.do_binop("__rshift__", other)

    def __lshift__(self, other: BinArray | str | int) -> BinArray:
        """Perform left bit shift operation.

        Args:
            other: Number of positions to shift left.

        Returns:
            New BinArray with left shift result.
        """
        return self.do_binop("__lshift__", other)

    def __invert__(self) -> BinArray:
        """Perform bitwise NOT operation.

        Returns:
            New BinArray with inverted bits.
        """
        return self.new_val(~self.value)

    def __mul__(self, other: BinArray | str | int) -> BinArray:
        """Perform multiplication with another value.

        Args:
            other: Other operand for multiplication.

        Returns:
            New BinArray with multiplication result.
        """
        return self.do_binop("__mul__", other)

    def __floordiv__(self, other: BinArray | str | int) -> BinArray:
        """Perform floor division with another value.

        Args:
            other: Other operand for floor division.

        Returns:
            New BinArray with floor division result.
        """
        return self.do_binop("__floordiv__", other)

    def __mod__(self, other: BinArray | str | int) -> BinArray:
        """Perform modulo operation with another value.

        Args:
            other: Other operand for modulo operation.

        Returns:
            New BinArray with modulo result.
        """
        return self.do_binop("__mod__", other)
