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

from pecos.simulators import SparseSimPy, SparseSimRs

states = [
    SparseSimPy,
    SparseSimRs,
]


def test_init_zero():
    """
    Test initializing |0>.

    :return:
    """

    for state in states:
        state = state(1)
        state.run_gate("init |0>", {0})

        # Test stabilizers
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == ["  Z"]

        # Test destabilizers
        destab_rep = state.destabs.print_tableau(verbose=False)
        assert destab_rep == ["  X"]


def test_init_one():
    """
    Test initializing |1>.

    stab: +Z
    destab: X


    :return:
    """

    for state in states:
        state = state(1)
        state.run_gate("init |1>", {0})

        # Test stabilizers
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == [" -Z"]

        # Test destabilizers
        destab_rep = state.destabs.print_tableau(verbose=False)
        assert destab_rep == ["  X"]


def test_init_plus():
    """
    Test initializing |+>.

    stab: +X
    destab: Z


    :return:
    """

    for state in states:
        state = state(1)
        state.run_gate("init |+>", {0})

        # Test stabilizers
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == ["  X"]

        # Test destabilizers
        destab_rep = state.destabs.print_tableau(verbose=False)
        assert destab_rep == ["  Z"]


def test_init_minus():
    """
    Test initializing |->.

    stab: -X
    destab: Z

    :return:
    """

    for state in states:
        state = state(1)
        state.run_gate("init |->", {0})

        # Test stabilizers
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == [" -X"]

        # Test destabilizers
        destab_rep = state.destabs.print_tableau(verbose=False)
        assert destab_rep == ["  Z"]


def test_init_plus_i():
    """
    Test initializing |+i>.

    stab: +Y
    destab: X | Z

    :return:
    """

    for state in states:
        state = state(1)
        state.run_gate("init |+i>", {0})

        # Test stabilizers
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == [" iW"]

        # Test destabilizers
        destab_rep = state.destabs.print_tableau(verbose=False)
        assert destab_rep in [["  X"], ["  Z"]]


def test_init_minus_i():
    """
    Test initializing |+i>.

    stab: -Y
    destab: X | Z

    :return:
    """

    for state in states:
        state = state(1)
        state.run_gate("init |-i>", {0})

        # Test stabilizers
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == ["-iW"]

        # Test destabilizers
        destab_rep = state.destabs.print_tableau(verbose=False)
        assert destab_rep in [["  X"], ["  Z"]]
