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
    for s in states:
        # XI, IX
        state = s(2)
        # control -> target
        state.run_gate("init |+>", {0})
        state.run_gate("init |+>", {1})
        assert state.stabs.print_tableau(verbose=False) == ["  XI", "  IX"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["XI"]
        assert stab_rep[1] == stab_dict["IX"]
        # destab_test(state, init_destab, stab_dict)

        # ZI, IZ
        state = s(2)
        # control -> target
        state.run_gate("init |0>", {0})
        state.run_gate("init |0>", {1})
        assert state.stabs.print_tableau(verbose=False) == ["  ZI", "  IZ"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["ZI"]
        assert stab_rep[1] == stab_dict["IZ"]
        # destab_test(state, init_destab, stab_dict)

        # iWI, iIW
        state = s(2)
        # control -> target
        state.run_gate("init |+i>", {0})
        state.run_gate("init |+i>", {1})
        assert state.stabs.print_tableau(verbose=False) == [" iWI", " iIW"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["iWI"]
        assert stab_rep[1] == stab_dict["iIW"]
        # destab_test(state, init_destab, stab_dict)

        # by now we have shown the single Cliffords and CNOT: XI -> XX, IZ -> ZZ

        # XX, ZZ
        state = s(2)
        # control -> target
        state.run_gate("init |+>", {0})
        state.run_gate("init |0>", {1})
        state.run_gate("CNOT", {(0, 1)})
        assert state.stabs.print_tableau(verbose=False) == ["  XX", "  ZZ"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["XX"]
        assert stab_rep[1] == stab_dict["ZZ"]
        # destab_test(state, init_destab, stab_dict)

        # ZX, XZ
        state = s(2)
        # control -> target
        state.run_gate("init |+>", {0})
        state.run_gate("init |0>", {1})
        state.run_gate("CNOT", {(0, 1)})
        state.run_gate("H", {0})
        assert state.stabs.print_tableau(verbose=False) == ["  ZX", "  XZ"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["ZX"]
        assert stab_rep[1] == stab_dict["XZ"]
        # destab_test(state, init_destab, stab_dict)

        # iXW, iWZ
        state = s(2)
        # control -> target
        state.run_gate("init |+>", {0})
        state.run_gate("init |0>", {1})
        state.run_gate("CNOT", {(0, 1)})  # -> XX, ZZ
        state.run_gate("H5", {0})  # -> -XX, iWZ
        state.run_gate("H3", {1})  # -> -iXW, -iWZ
        state.run_gate("Y", {0})  # -> iXW, -iWZ
        state.run_gate("Y", {1})  # -> iXW, iWZ
        assert state.stabs.print_tableau(verbose=False) == [" iXW", " iWZ"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["iXW"]
        assert stab_rep[1] == stab_dict["iWZ"]
        # destab_test(state, init_destab, stab_dict)

        # iWX, iZW
        state = s(2)
        # control -> target
        state.run_gate("init |+>", {0})
        state.run_gate("init |0>", {1})
        state.run_gate("CNOT", {(0, 1)})  # -> XX, ZZ
        state.run_gate("H5", {1})  # -> -XX, iZW
        state.run_gate("H3", {0})  # -> -iWX, -iZW
        state.run_gate("Y", {0})  # -> iWX, -iZW
        state.run_gate("Y", {1})  # -> iWX, iZW
        assert state.stabs.print_tableau(verbose=False) == [" iWX", " iZW"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["iWX"]
        assert stab_rep[1] == stab_dict["iZW"]
        # destab_test(state, init_destab, stab_dict)

        # -WW
        state = s(2)
        # control -> target
        state.run_gate("init |+>", {0})
        state.run_gate("CNOT", {(0, 1)})  # -> XX, ZZ
        state.run_gate("H3", {0})  # -> iXW, -ZZ
        state.run_gate("H3", {1})  # -> -WW, ZZ
        assert state.stabs.print_tableau(verbose=False) == [" -WW", "  ZZ"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(gate_symbol, {(0, 1)})
        stab_rep = state.stabs.print_tableau(verbose=False)
        assert stab_rep[0] == stab_dict["-WW"]
        # destab_test(state, init_destab, stab_dict)


def test_CNOT() -> None:
    """Test CNOT.

    II -> II
    .XI -> XX
    .ZI -> ZI
    .iWI -> iWX
    .IX -> IX
    .IZ -> ZZ
    .iIW -> iZW
    .XX -> XI
    .XZ -> WW
    .iXW -> iWZ
    .ZX -> ZX
    .ZZ -> IZ
    .iZW -> iIW
    .iWX -> iWI
    .iWZ -> iXW
    -WW -> -XZ
    """
    stab_transform = {
        "XI": "  XX",
        "IX": "  IX",
        "ZI": "  ZI",
        "IZ": "  ZZ",
        "iWI": " iWX",
        "iIW": " iZW",
        "XX": "  XI",
        "ZZ": "  IZ",
        "ZX": "  ZX",
        "XZ": "  WW",
        "iXW": " iWZ",
        "iWZ": " iXW",
        "iWX": " iWI",
        "iZW": " iIW",
        "-WW": " -XZ",
    }

    gate_test("CNOT", stab_transform)


def test_CZ() -> None:
    """Test CZ.

    II -> II
    XI -> XZ
    ZI -> ZI
    WI -> WZ
    IX -> ZX
    IZ -> IZ
    IW -> ZW
    XX -> -WW
    XZ -> XI
    XW -> -WX
    ZX -> IX
    ZZ -> ZZ
    ZW -> IW
    WX -> -XW
    WZ -> WI
    WW -> -XX
    """
    stab_transform = {
        "XI": "  XZ",
        "ZI": "  ZI",
        "iWI": " iWZ",
        "IX": "  ZX",
        "IZ": "  IZ",
        "iIW": " iZW",
        "XX": " -WW",
        "XZ": "  XI",
        "iXW": "-iWX",
        "ZX": "  IX",
        "ZZ": "  ZZ",
        "iZW": " iIW",
        "iWX": "-iXW",
        "iWZ": " iWI",
        "-WW": "  XX",
    }

    gate_test("CZ", stab_transform)


def test_SWAP() -> None:
    """Test SWAP."""
    stab_transform = {
        "XI": "  IX",
        "ZI": "  IZ",
        "iWI": " iIW",
        "IX": "  XI",
        "IZ": "  ZI",
        "iIW": " iWI",
        "XX": "  XX",
        "XZ": "  ZX",
        "iXW": " iWX",
        "ZX": "  XZ",
        "ZZ": "  ZZ",
        "iZW": " iWZ",
        "iWX": " iXW",
        "iWZ": " iZW",
        "-WW": " -WW",
    }

    gate_test("SWAP", stab_transform)


def test_G2() -> None:
    """Test G2.

    II -> II
    XI -> IX
    ZI -> XZ
    WI -> XW
    IX -> XI
    IZ -> ZX
    IW -> WX
    XX -> XX
    XZ -> ZI
    XW -> WI
    ZX -> IZ
    ZZ -> -WW
    ZW -> -ZW
    WX -> IW
    WZ -> -WZ
    WW -> ZZ
    """
    stab_transform = {
        "XI": "  IX",
        "ZI": "  XZ",
        "iWI": " iXW",
        "IX": "  XI",
        "IZ": "  ZX",
        "iIW": " iWX",
        "XX": "  XX",
        "XZ": "  ZI",
        "iXW": " iWI",
        "ZX": "  IZ",
        "ZZ": " -WW",
        "iZW": "-iZW",
        "iWX": " iIW",
        "iWZ": "-iWZ",
        "-WW": "  ZZ",
    }

    gate_test("G2", stab_transform)


def test_SqrtXX() -> None:
    """Test 'Sqrt XX test'.

    II -> II
    XI -> XI
    ZI -> -iWX
    WI -> -iZX
    IX -> IX
    IZ -> -iXW
    IW -> -iXZ
    XX -> XX
    XZ -> -iIW
    XW -> -iIZ
    ZX -> -iWI
    ZZ -> ZZ
    ZW -> ZW
    WX -> -iZI
    WZ -> WZ
    WW -> WW
    """
    stab_transform = {
        "XI": "  XI",
        "ZI": "-iWX",
        "iWI": "  ZX",
        "IX": "  IX",
        "IZ": "-iXW",
        "iIW": "  XZ",
        "XX": "  XX",
        "XZ": "-iIW",
        "iXW": "  IZ",
        "ZX": "-iWI",
        "ZZ": "  ZZ",
        "iZW": " iZW",
        "iWX": "  ZI",
        "iWZ": " iWZ",
        "-WW": " -WW",
    }

    gate_test("SqrtXX", stab_transform)
