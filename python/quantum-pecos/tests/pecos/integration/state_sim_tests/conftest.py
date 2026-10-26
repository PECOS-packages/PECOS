# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Dependency-free location protocol examples for wrapper tests."""

import pytest


class SizedSequence:
    """An ordered batch/pair whose truth value, like an array's, is ambiguous."""

    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, position):
        return self.values[position]

    def __bool__(self):
        msg = "ambiguous truth value"
        raise ValueError(msg)


class IndexScalar:
    """An integer scalar that advertises unusable zero-dimensional sequence methods."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

    def __len__(self):
        msg = "len() of unsized object"
        raise TypeError(msg)

    def __getitem__(self, _position):
        msg = "a scalar is not iterable"
        raise TypeError(msg)

    def __bool__(self):
        return bool(self.value)


@pytest.fixture
def sized_sequence():
    return SizedSequence


@pytest.fixture
def index_scalar():
    return IndexScalar


class UnhashableIndexScalar(IndexScalar):
    """An integer scalar that cannot be hashed, like a zero-dimensional array."""

    __hash__ = None


@pytest.fixture
def unhashable_index_scalar():
    return UnhashableIndexScalar
