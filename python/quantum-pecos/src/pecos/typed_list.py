# Copyright 2024 The PECOS Developers
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

from collections.abc import Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class TypedList(list[T], Generic[T]):
    """A list that only accepts items of a specific type, where are checked at runtime.

    Attributes:
        type (Type[T]): The type of the elements that the list should contain.

    Methods:
        append(item): Appends an item to the end of the list, checking its type.
        extend(iterable): Extends the list by appending elements from the iterable, checking their types.
        insert(index, item): Inserts an item at a given position, checking its type.
        __setitem__(index, item): Sets the item at the specified position or slice, checking its type.
    """

    def __init__(self, _type: type, data: Iterable[T] | None = None) -> None:
        """Initializes a TypedList with as specified element type and optional initial data.

        Args:
            _type (Type[T]): The type of the elements that the list should contain.
            data (Iterable[T], optional): Initial data to populate the typed list. Defaults to None.

        Raises:
             TypeError: If any item in the initial data is not of the correct type.
        """
        self.type = _type
        if data is not None:
            for i in data:
                self._check_type(i)
            super().__init__(data)
        else:
            super().__init__()

    def _check_type(self, /, __object: T) -> None:
        """Checks if an item is of the specified type.

        Args:
            __object (T): The item to check.

        Raises:
            TypeError: If the item is not of the correct type.
        """
        if not isinstance(__object, self.type):
            msg = f"Item must be of type {self.type.__name__}, got type {type(__object).__name__} instead"
            raise TypeError(msg)

    def append(self, /, __object: T):
        """Appends an item to the end of the list, checking its type.

        Args:
            __object (T): The item to append.

        Raises:
            TypeError: If the item is not of the correct type.
        """
        self._check_type(__object)
        super().append(__object)

    def extend(self, /, __iterable: Iterable[T]) -> None:
        """Extends the list by appending elements from the iterable, checking their type.

        Args:
            __iterable (Iterable[T]): An iterable of items to extend the list with.

        Raises:
            TypeError: If any item in the iterable is not of the correct type.
        """

        for i in __iterable:
            self._check_type(i)
        super().extend(__iterable)

    def insert(self, /, __index: int, __object: T) -> None:
        """Inserts an item en position, checking its type.

        Args:
            __index (int): The pto insert the .
            __object (T): The item to insert.
        """

        self._check_type(__object)
        super().insert(__index, __object)

    def __setitem__(self, key: int | slice, value: T | Iterable[T]) -> None:
        """Sets the item at the specified position or slice, checking its type.

        Args:
            key (int | slice): The position or slice to set the item(s) at.
            value (T | Iterable[T]): The item(s) to set.

        Raises:
            TypeError: If the item or any item in the slice is not of the correct type.
        """

        if isinstance(value, slice):
            if isinstance(value, Iterable):
                for i in value:
                    self._check_type(i)
            else:
                self._check_type(value)
        else:
            self._check_type(value)
        super().__setitem__(key, value)
