# Copyright 2019 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Integration tests for stabilizer simulator gate initialization."""

from pecos.simulators import SparseStab, SparseStabPy, Stabilizer

states = [
    SparseStabPy,
    SparseStab,
    Stabilizer,
]


def test_init_zero() -> None:
    """Test initializing |0>."""
    for state_class in states:
        state = state_class(1)
        state.run_gate("init |0>", {0})

        assert state.stabs.print_tableau(verbose=False) == ["  Z"]
        assert state.destabs.print_tableau(verbose=False) == ["  X"]


def test_init_one() -> None:
    """Test initializing |1>."""
    for state_class in states:
        state = state_class(1)
        state.run_gate("init |1>", {0})

        assert state.stabs.print_tableau(verbose=False) == [" -Z"]
        assert state.destabs.print_tableau(verbose=False) == ["  X"]


def test_init_plus() -> None:
    """Test initializing |+>."""
    for state_class in states:
        state = state_class(1)
        state.run_gate("init |+>", {0})

        assert state.stabs.print_tableau(verbose=False) == ["  X"]
        assert state.destabs.print_tableau(verbose=False) == ["  Z"]


def test_init_minus() -> None:
    """Test initializing |->."""
    for state_class in states:
        state = state_class(1)
        state.run_gate("init |->", {0})

        assert state.stabs.print_tableau(verbose=False) == [" -X"]
        assert state.destabs.print_tableau(verbose=False) == ["  Z"]


def test_init_plus_i() -> None:
    """Test initializing |+i>."""
    for state_class in states:
        state = state_class(1)
        state.run_gate("init |+i>", {0})

        assert state.stabs.print_tableau(verbose=False) == [" iW"]
        assert state.destabs.print_tableau(verbose=False) in [["  X"], ["  Z"]]


def test_init_minus_i() -> None:
    """Test initializing |-i>."""
    for state_class in states:
        state = state_class(1)
        state.run_gate("init |-i>", {0})

        assert state.stabs.print_tableau(verbose=False) == ["-iW"]
        assert state.destabs.print_tableau(verbose=False) in [["  X"], ["  Z"]]
