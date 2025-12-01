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

"""Test all one-qubit gates."""

from pecos.simulators import SparseSim, SparseSimCpp, SparseSimPy

states = [
    SparseSimPy,
    SparseSim,
    SparseSimCpp,
]


def gate_test(gate_symbol: str, stab_dict: dict[str, list[str]]) -> None:
    """Function that is called to test one-qubit gates.

    :param gate_symbol:
    :param stab_dict:
    :return:
    """
    for state_class in states:
        state = state_class(1)

        # X stabilizer
        state.run_gate("init |+>", {0})
        init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {0})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == [stab_dict["X"]]
        # destab_test(state, init_destab, stab_dict)

        # Z stabilizer
        state.run_gate("init |0>", {0})
        init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {0})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == [stab_dict["Z"]]
        destab_test(state, init_destab, stab_dict)

        # Y (iW) stabilizer
        state.run_gate("init |+i>", {0})
        init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {0})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep == [stab_dict["iW"]]
        # destab_test(state, init_destab, stab_dict)


def destab_test(
    state: SparseSimPy | SparseSim,
    init_destab: str,
    stab_dict: dict[str, list[str]],
) -> None:
    """Test destabilizer operations for stabilizer simulators."""
    destab = state.destabs.print_tableau(verbose=False)[0]

    init_destab = init_destab.strip()
    if init_destab == "W":
        init_destab = "iW"

    assert destab[2:] == stab_dict[init_destab][2:]


def test_I() -> None:
    """Test Pauli I."""
    stab_transform = {
        "X": "  X",
        "Z": "  Z",
        "iW": " iW",
    }

    gate_test("I", stab_transform)


def test_X() -> None:
    """Test Pauli X."""
    stab_transform = {
        "X": "  X",
        "Z": " -Z",
        "iW": "-iW",
    }

    gate_test("X", stab_transform)


def test_Y() -> None:
    """Test Pauli Y."""
    stab_transform = {
        "X": " -X",
        "Z": " -Z",
        "iW": " iW",
    }

    gate_test("Y", stab_transform)


def test_Z() -> None:
    """Test Pauli Y."""
    stab_transform = {
        "X": " -X",
        "Z": "  Z",
        "iW": "-iW",
    }

    gate_test("Z", stab_transform)


def test_Q() -> None:
    """Test Q (sqrt{X})."""
    stab_transform = {
        "X": "  X",
        "Z": "-iW",
        "iW": "  Z",
    }

    gate_test("Q", stab_transform)


def test_Qd() -> None:
    """Test Q^{dagger}."""
    stab_transform = {
        "X": "  X",
        "Z": " iW",
        "iW": " -Z",
    }

    gate_test("Qd", stab_transform)


def test_R() -> None:
    """Test R (sqrt{Y})."""
    stab_transform = {
        "X": " -Z",
        "Z": "  X",
        "iW": " iW",
    }

    gate_test("R", stab_transform)


def test_Rd() -> None:
    """Test R^{dagger}."""
    stab_transform = {
        "X": "  Z",
        "Z": " -X",
        "iW": " iW",
    }

    gate_test("Rd", stab_transform)


def test_S() -> None:
    """Test S (sqrt{Z})."""
    stab_transform = {
        "X": " iW",
        "Z": "  Z",
        "iW": " -X",
    }

    gate_test("S", stab_transform)


def test_Sd() -> None:
    """Test S^{dagger}."""
    stab_transform = {
        "X": "-iW",
        "Z": "  Z",
        "iW": "  X",
    }

    gate_test("Sd", stab_transform)


def test_H() -> None:
    """Test the Hadamard."""
    stab_transform = {
        "X": "  Z",
        "Z": "  X",
        "iW": "-iW",
    }

    gate_test("H", stab_transform)


def test_H2() -> None:
    """Test H2.

    :return:
    """
    stab_transform = {
        "X": " -Z",
        "Z": " -X",
        "iW": "-iW",
    }

    gate_test("H2", stab_transform)


def test_H3() -> None:
    """Test H3.

    :return:
    """
    stab_transform = {
        "X": " iW",
        "Z": " -Z",
        "iW": "  X",
    }

    gate_test("H3", stab_transform)


def test_H4() -> None:
    """Test H4.

    :return:
    """
    stab_transform = {
        "X": "-iW",
        "Z": " -Z",
        "iW": " -X",
    }

    gate_test("H4", stab_transform)


def test_H5() -> None:
    """Test H5.

    :return:
    """
    stab_transform = {
        "X": " -X",
        "Z": " iW",
        "iW": "  Z",
    }

    gate_test("H5", stab_transform)


def test_H6() -> None:
    """Test H6.

    :return:
    """
    stab_transform = {
        "X": " -X",
        "Z": "-iW",
        "iW": " -Z",
    }

    gate_test("H6", stab_transform)


def test_F1() -> None:
    """Test F1.

    :return:
    """
    stab_transform = {
        "X": " iW",
        "Z": "  X",
        "iW": "  Z",
    }

    gate_test("F1", stab_transform)


def test_F1d() -> None:
    """Test F1d.

    :return:
    """
    stab_transform = {
        "X": "  Z",
        "Z": " iW",
        "iW": "  X",
    }

    gate_test("F1d", stab_transform)


def test_F2() -> None:
    """Test F2.

    :return:
    """
    stab_transform = {
        "X": " -Z",
        "Z": " iW",
        "iW": " -X",
    }

    gate_test("F2", stab_transform)


def test_F2d() -> None:
    """Test F2d.

    :return:
    """
    stab_transform = {
        "X": "-iW",
        "Z": " -X",
        "iW": "  Z",
    }

    gate_test("F2d", stab_transform)


def test_F3() -> None:
    """Test F3.

    :return:
    """
    stab_transform = {
        "X": " iW",
        "Z": " -X",
        "iW": " -Z",
    }

    gate_test("F3", stab_transform)


def test_F3d() -> None:
    """Test F3d.

    :return:
    """
    stab_transform = {
        "X": " -Z",
        "Z": "-iW",
        "iW": "  X",
    }

    gate_test("F3d", stab_transform)


def test_F4() -> None:
    """Test F4.

    :return:
    """
    stab_transform = {
        "X": "  Z",
        "Z": "-iW",
        "iW": " -X",
    }

    gate_test("F4", stab_transform)


def test_F4d() -> None:
    """Test F4d.

    :return:
    """
    stab_transform = {
        "X": "-iW",
        "Z": "  X",
        "iW": " -Z",
    }

    gate_test("F4d", stab_transform)
