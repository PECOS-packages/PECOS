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

import numpy as np

from pecos.reps.pypmir import unsigned_data_types


class BinArray2:
    """As opposed to the original unsigned 32-bit BinArray, this class defaults to signed 64-bit type."""

    def __init__(self, size, value=0, dtype=np.int64) -> None:
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

    def set(self, value):
        if isinstance(value, self.dtype):
            self.value = value
        elif isinstance(value, BinArray2):
            self.value = value.value
        else:
            if isinstance(value, str):
                value = int(value, 2)

            self.value = self.dtype(value)

    def new_val(self, value):
        b = BinArray2(self.size, value, self.dtype)
        if self.dtype in unsigned_data_types.values():
            b.clamp(self.size)
        return b

    def num_bits(self):
        return len(f"{self.value:b}")

    def check_size(self):
        if self.num_bits() > self.size:
            num = self.num_bits()
            val = f"{self.value:b}"
            msg = f'Number of bits ({num}) exceeds size ({self.size}) for bits "{val}"!'
            raise Exception(msg)

    def clamp(self, size):
        if self.num_bits() > size:
            bits = format(self.value, f"0{size}b")
            bits = int(bits[-size:], 2)
            self.value = self.dtype(bits)

    def set_clip(self, value):
        value = int(value)

        if len(f"{value:b}") > self.size:
            bits = format(value, f"0{self.size}b")
            bits = int(bits[-self.size :], 2)
            self.value = self.dtype(bits)
        else:
            self.value = self.dtype(value)

    def _set_clip(self, ba):
        """Take values up to the size of this BinArray. If this BinArray array is larger, fill with zeros."""
        if isinstance(ba, int):
            ba = self.new_val(ba)

        if isinstance(ba, BinArray2):
            self._set_clip(ba)
        else:
            msg = "Expected int or BinArray!"
            raise TypeError(msg)

    def __getitem__(self, item):
        return int(str(self)[self.size - item - 1])

    def __setitem__(self, key, value) -> None:
        b = list(str(self))
        b[self.size - key - 1] = str(value)
        b = "".join(b)

        self.set(b)

    def __str__(self) -> str:
        self.check_size()
        return format(self.value, f"0{self.size}b")

    def __repr__(self) -> str:
        return self.__str__()

    def __int__(self) -> int:
        return int(self.value)

    def __len__(self) -> int:
        return self.size

    def do_binop(self, op, other):
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
        return bool(self.value)

    def __xor__(self, other):
        return self.do_binop("__xor__", other)

    def __and__(self, other):
        return self.do_binop("__and__", other)

    def __or__(self, other):
        return self.do_binop("__or__", other)

    def __eq__(self, other):
        return self.do_binop("__eq__", other)

    def __ne__(self, other):
        return self.do_binop("__ne__", other)

    def __lt__(self, other):
        return self.do_binop("__lt__", other)

    def __gt__(self, other):
        return self.do_binop("__gt__", other)

    def __le__(self, other):
        return self.do_binop("__le__", other)

    def __ge__(self, other):
        return self.do_binop("__ge__", other)

    def __add__(self, other):
        return self.do_binop("__add__", other)

    def __sub__(self, other):
        return self.do_binop("__sub__", other)

    def __rshift__(self, other):
        return self.do_binop("__rshift__", other)

    def __lshift__(self, other):
        return self.do_binop("__lshift__", other)

    def __invert__(self):
        return self.new_val(~self.value)

    def __mul__(self, other):
        return self.do_binop("__mul__", other)

    def __floordiv__(self, other):
        return self.do_binop("__floordiv__", other)

    def __mod__(self, other):
        return self.do_binop("__floordiv__", other)
