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


class BinArray:
    def __init__(self, size: int | str, value: int | None = None) -> None:
        if isinstance(size, int):
            self.size = size
            self.array = [0] * size

            if value is not None:
                self.set(value)
        elif isinstance(size, str):
            str_rep = size

            self.array = []
            for i in reversed(str_rep):
                if i == "0":
                    self.array.append(0)
                elif i == "1":
                    self.array.append(1)
                else:
                    msg = f"Can only accept a string made of 0s and 1s! Got {str_rep}."
                    raise Exception(msg)

            self.size = len(self.array)
        else:
            msg = f"First argument must be int or str! Got {size} of type {type(size)}."
            raise TypeError(msg)

    def __str__(self) -> str:
        bin_str = [
            "1" if self.array[i] else "0" for i in range(len(self.array) - 1, -1, -1)
        ]
        return "".join(bin_str)

    def __repr__(self) -> str:
        return self.__str__()

    def __int__(self) -> int:
        return int(str(self), 2)

    def set_clip(self, ba):
        """Take values up to the size of this BinArray. If this BinArray array is larger, fill with zeros."""
        if isinstance(ba, int):
            ba = BinArray(format(ba, "b"))

        if isinstance(ba, BinArray):
            for i in range(self.size):
                if i >= ba.size:
                    self.array[i] = 0
                else:
                    self.array[i] = ba.array[i]
        else:
            msg = "Expected int or BinArray!"
            raise TypeError(msg)

    def set(self, value: BinArray | int):
        value = int(value)

        value = (2**self.size - 1) & value

        for i, b in enumerate(reversed(format(value, f"0{self.size}b"))):
            # Don't add more elements than size
            if i >= self.size:
                break

            self.array[i] = int(b)

        """
        if isinstance(value, int):
            for i, b in enumerate(reversed(format(value, f'0{self.size}b'))):

                # Don't add more elements than size
                if i >= self.size:
                    break

                self.array[i] = int(b)

        elif isinstance(value, BinArray):
            if self.size != value.size:
                raise Exception('Binary array must be the same size as the array being set.')

            # Copy the other's array into this one
            self.array = list(value.array)

        else:
            raise Exception(f"Can't set value of type {type(value)}")
        """

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value) -> None:
        value_temp = int(value)

        if value_temp not in {0, 1}:
            msg = "Can only set an element to a binary value!"
            raise Exception(msg)

        self.array[key] = value_temp

    def __len__(self) -> int:
        return self.size

    def __xor__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) ^ int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __and__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) & int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __or__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) | int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __eq__(self, other):
        val = int(self) == int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __ne__(self, other):
        val = int(self) != int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __lt__(self, other):
        val = int(self) < int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __gt__(self, other):
        val = int(self) > int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __le__(self, other):
        val = int(self) <= int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __ge__(self, other):
        val = int(self) >= int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __add__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) + int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __sub__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) - int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __rshift__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) >> int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __lshift__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) << int(other)
        result = BinArray(self.size)
        result.set_clip(val)

        return result

    def __invert__(self):
        val = int(self)
        result = BinArray(self.size)
        result.set(val)

        for i, b in enumerate(result.array):
            if b == 0:
                result.array[i] = 1
            else:
                result.array[i] = 0

        return result

    def __mul__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) * int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __floordiv__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) // int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __mod__(self, other):
        if isinstance(other, BinArray) and other.size != self.size:
            msg = "Can only do bitwise operations between BinArrays of the same size."
            raise Exception(msg)

        val = int(self) % int(other)
        result = BinArray(self.size)
        result.set(val)

        return result

    def __bool__(self) -> bool:
        return bool(int(self))
