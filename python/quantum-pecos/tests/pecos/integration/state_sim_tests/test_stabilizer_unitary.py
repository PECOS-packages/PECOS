"""Testing the unitaries of the stabilizer sim against reference matrix definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pecos as pc
from pecos.simulators import SparseSim

# Load gate_matrix_def from the same directory (importlib mode doesn't auto-add it to sys.path)
_gate_matrix_def_path = Path(__file__).parent / "gate_matrix_def.py"
_spec = importlib.util.spec_from_file_location("gate_matrix_def", _gate_matrix_def_path)
g = importlib.util.module_from_spec(_spec)
sys.modules["gate_matrix_def"] = g
_spec.loader.exec_module(g)

states = [
    SparseSim,
]


def check_transformation(transform: dict[str, str], matrix: pc.Array) -> None:
    """Check transformation of Paulis with the unitary matrix defined by hqslib.inc."""
    gm = {
        "X": g.X,
        "Z": g.Z,
        "Y": g.Y,
    }

    u = matrix.copy()
    ud = u.conj().T

    for ga, gb in transform.items():
        has_minus = False
        pauli_key = gb
        if gb[0] == "-":
            pauli_key = gb[1]
            has_minus = True

        # USU^dagger
        uf = gm[pauli_key]
        if has_minus:
            uf = -uf
        assert pc.isclose(u.dot(gm[ga]).dot(ud), uf).all()


def refsym(sym: str) -> str:
    """Refactor Pauli string."""
    sym = sym.strip()
    return sym.replace("iW", "Y")


def gate_test(gate_symbol: str, stab_dict: dict[str, str]) -> None:
    """Test one-qubit gate stabilizer transformations.

    Args:
        gate_symbol: The gate name to test.
        stab_dict: Expected stabilizer transformations.
    """
    for state_cls in states:
        state = state_cls(1)

        # X stabilizer
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(
            gate_symbol,
            {
                0,
            },
        )
        stab_rep = refsym(state.stabs.print_tableau(verbose=False)[0])
        assert stab_rep == stab_dict["X"], f"{stab_rep} != {stab_dict['X']}"
        # destab_test(state, init_destab, stab_dict)

        # Z stabilizer
        state.run_gate(
            "init |0>",
            {
                0,
            },
        )
        init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(
            gate_symbol,
            {
                0,
            },
        )
        stab_rep = refsym(state.stabs.print_tableau(verbose=False)[0])
        assert stab_rep == stab_dict["Z"], f"{stab_rep} != {stab_dict['Z']}"
        destab_test(state, init_destab, stab_dict)

        # Y (iW) stabilizer
        state.run_gate(
            "init |+i>",
            {
                0,
            },
        )
        init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(
            gate_symbol,
            {
                0,
            },
        )
        stab_rep = refsym(state.stabs.print_tableau(verbose=False)[0])
        assert stab_rep == stab_dict["Y"], f"{stab_rep} != {stab_dict['Y']}"
        # destab_test(state, init_destab, stab_dict)


def destab_test(state: SparseSim, init_destab: str, stab_dict: dict[str, str]) -> None:
    """Test destabilizer transformations match expected results."""
    destab = refsym(state.destabs.print_tableau(verbose=False)[0])

    init_destab = init_destab.strip()
    if init_destab == "W":
        init_destab = "Y"

    assert destab == stab_dict[init_destab].replace(
        "-",
        "",
    ), f"{refsym(destab)} != {stab_dict[init_destab].replace('-', '')}"


def test_I() -> None:
    """Test Pauli I."""
    stab_transform = {"X": "X", "Z": "Z", "Y": "Y"}

    check_transformation(stab_transform, g.I)

    gate_test("I", stab_transform)


def test_X() -> None:
    """Test Pauli X."""
    stab_transform = {"X": "X", "Z": "-Z", "Y": "-Y"}

    check_transformation(stab_transform, g.X)

    gate_test("X", stab_transform)


def test_Y() -> None:
    """Test Pauli Y."""
    stab_transform = {"X": "-X", "Z": "-Z", "Y": "Y"}

    check_transformation(stab_transform, g.Y)

    gate_test("Y", stab_transform)


def test_Z() -> None:
    """Test Pauli Y."""
    stab_transform = {"X": "-X", "Z": "Z", "Y": "-Y"}

    check_transformation(stab_transform, g.Z)

    gate_test("Z", stab_transform)


def test_Q() -> None:
    """Test Q (sqrt{X})."""
    stab_transform = {"X": "X", "Z": "-Y", "Y": "Z"}

    check_transformation(stab_transform, g.RX(pc.f64.pi / 2))

    gate_test("Q", stab_transform)
    gate_test("SqrtX", stab_transform)


def test_Qd() -> None:
    """Test Q^{dagger}."""
    stab_transform = {"X": "X", "Z": "Y", "Y": "-Z"}

    check_transformation(stab_transform, g.RX(-pc.f64.pi / 2))

    gate_test("Qd", stab_transform)
    gate_test("SqrtXd", stab_transform)


def test_R() -> None:
    """Test R (sqrt{Y})."""
    stab_transform = {"X": "-Z", "Z": "X", "Y": "Y"}

    check_transformation(stab_transform, g.RY(pc.f64.pi / 2))

    gate_test("R", stab_transform)
    gate_test("SqrtY", stab_transform)


def test_Rd() -> None:
    """Test R^{dagger}."""
    stab_transform = {"X": "Z", "Z": "-X", "Y": "Y"}

    check_transformation(stab_transform, g.RY(-pc.f64.pi / 2))

    gate_test("Rd", stab_transform)
    gate_test("SqrtYd", stab_transform)


def test_S() -> None:
    """Test S (sqrt{Z})."""
    stab_transform = {"X": "Y", "Z": "Z", "Y": "-X"}

    check_transformation(stab_transform, g.RZ(pc.f64.pi / 2))

    gate_test("S", stab_transform)
    gate_test("SqrtZ", stab_transform)


def test_Sd() -> None:
    """Test S^{dagger}."""
    stab_transform = {"X": "-Y", "Z": "Z", "Y": "X"}

    check_transformation(stab_transform, g.RZ(-pc.f64.pi / 2))

    gate_test("Sd", stab_transform)
    gate_test("SqrtZd", stab_transform)


def test_H() -> None:
    """Test the Hadamard."""
    stab_transform = {"X": "Z", "Z": "X", "Y": "-Y"}

    check_transformation(stab_transform, g.h_def)
    h_def2 = g.X.dot(g.RY(pc.f64.pi / 2))
    check_transformation(stab_transform, h_def2)

    gate_test("H", stab_transform)


def test_H2() -> None:
    """Test H2.

    :return:
    """
    stab_transform = {"X": "-Z", "Z": "-X", "Y": "-Y"}

    h2_def = g.Z.dot(g.RY(pc.f64.pi / 2))
    check_transformation(stab_transform, h2_def)

    gate_test("H2", stab_transform)


def test_H3() -> None:
    """Test H3.

    :return:
    """
    stab_transform = {"X": "Y", "Z": "-Z", "Y": "X"}

    h3_def = g.Y.dot(g.RZ(pc.f64.pi / 2))
    check_transformation(stab_transform, h3_def)

    gate_test("H3", stab_transform)


def test_H4() -> None:
    """Test H4.

    :return:
    """
    stab_transform = {"X": "-Y", "Z": "-Z", "Y": "-X"}

    h4_def = g.X.dot(g.RZ(pc.f64.pi / 2))
    check_transformation(stab_transform, h4_def)

    gate_test("H4", stab_transform)


def test_H5() -> None:
    """Test H5.

    :return:
    """
    stab_transform = {"X": "-X", "Z": "Y", "Y": "Z"}

    h5_def = g.Z.dot(g.RX(pc.f64.pi / 2))
    check_transformation(stab_transform, h5_def)

    gate_test("H5", stab_transform)


def test_H6() -> None:
    """Test H6.

    :return:
    """
    stab_transform = {"X": "-X", "Z": "-Y", "Y": "-Z"}

    h6_def = g.Y.dot(g.RX(pc.f64.pi / 2))
    check_transformation(stab_transform, h6_def)

    gate_test("H6", stab_transform)


def test_F1() -> None:
    """Test F1.

    :return:
    """
    stab_transform = {"X": "Y", "Z": "X", "Y": "Z"}

    f1_def = g.RZ(pc.f64.pi / 2).dot(g.RX(pc.f64.pi / 2))
    check_transformation(stab_transform, f1_def)

    gate_test("F1", stab_transform)


def test_F1d() -> None:
    """Test F1d.

    :return:
    """
    stab_transform = {"X": "Z", "Z": "Y", "Y": "X"}

    f1_def = g.RZ(pc.f64.pi / 2).dot(g.RX(pc.f64.pi / 2))
    check_transformation(stab_transform, f1_def.conj().T)

    gate_test("F1d", stab_transform)


def test_F2() -> None:
    """Test F2.

    :return:
    """
    stab_transform = {"X": "-Z", "Z": "Y", "Y": "-X"}

    f2_def = g.RX(-pc.f64.pi / 2).dot(g.RZ(pc.f64.pi / 2))
    check_transformation(stab_transform, f2_def)

    gate_test("F2", stab_transform)


def test_F2d() -> None:
    """Test F2d.

    :return:
    """
    stab_transform = {"X": "-Y", "Z": "-X", "Y": "Z"}

    f2_def = g.RX(-pc.f64.pi / 2).dot(g.RZ(pc.f64.pi / 2))
    check_transformation(stab_transform, f2_def.conj().T)

    gate_test("F2d", stab_transform)


def test_F3() -> None:
    """Test F3.

    :return:
    """
    stab_transform = {"X": "Y", "Z": "-X", "Y": "-Z"}

    f3_def = g.RZ(pc.f64.pi / 2).dot(g.RX(-pc.f64.pi / 2))
    check_transformation(stab_transform, f3_def)

    gate_test("F3", stab_transform)


def test_F3d() -> None:
    """Test F3d.

    :return:
    """
    stab_transform = {"X": "-Z", "Z": "-Y", "Y": "X"}

    f3_def = g.RZ(pc.f64.pi / 2).dot(g.RX(-pc.f64.pi / 2))
    check_transformation(stab_transform, f3_def.conj().T)

    gate_test("F3d", stab_transform)


def test_F4() -> None:
    """Test F4.

    :return:
    """
    stab_transform = {"X": "Z", "Z": "-Y", "Y": "-X"}

    f4_def = g.RX(pc.f64.pi / 2).dot(g.RZ(pc.f64.pi / 2))
    check_transformation(stab_transform, f4_def)

    gate_test("F4", stab_transform)


def test_F4d() -> None:
    """Test F4d.

    :return:
    """
    stab_transform = {"X": "-Y", "Z": "X", "Y": "-Z"}

    f4_def = g.RX(pc.f64.pi / 2).dot(g.RZ(pc.f64.pi / 2))
    check_transformation(stab_transform, f4_def.conj().T)

    gate_test("F4d", stab_transform)


def tqrefsym(sym: str | list[str]) -> str | list[str]:
    """Refactor symbols."""
    if isinstance(sym, str):
        result = sym.strip()
        result = result.replace("-WW", "YY")
        if result.count("W") == 1 and result.count("i") == 1:
            result = result.replace("i", "")
            result = result.replace("W", "Y")
        return result
    return [tqrefsym(s) for s in sym]


def check_transformation_tq(transform: dict[str, str], matrix: pc.Array) -> None:
    """Check transformation of Paulis for two-qubit gates with reference unitary matrices."""
    gm = {
        "I": g.I,
        "X": g.X,
        "Z": g.Z,
        "Y": g.Y,
    }

    u = matrix.copy()
    ud = u.conj().T

    for ga, gb in transform.items():
        has_minus = False
        pauli_keys = gb
        if gb[0] == "-":
            pauli_keys = gb[1:3]
            has_minus = True

        # USU^dagger
        uf = gm[pauli_keys[0]] & gm[pauli_keys[1]]
        if has_minus:
            uf = -uf

        ui = gm[ga[0]] & gm[ga[1]]
        # using matrices, transform the Paulis and show the Paulis transform as the rule suggests
        ui_transform = u.dot(ui).dot(ud)
        assert pc.isclose(
            ui_transform,
            uf,
        ).all(), f"rule: {ga} -> {gb}\n ui_transform = \n {ui_transform}\n uf = \n {uf}"


def gate_test_tq(gate_symbol: str, stab_dict: dict[str, str]) -> None:
    """Test two-qubit gate stabilizer transformations.

    Args:
        gate_symbol: The gate name to test.
        stab_dict: Expected stabilizer transformations.
    """
    for state_cls in states:
        # XI, IX
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        state.run_gate(
            "init |+>",
            {
                1,
            },
        )
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["XI", "IX"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["XI"]
        assert stab_rep[1] == stab_dict["IX"]

        # ZI, IZ
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |0>",
            {
                0,
            },
        )
        state.run_gate(
            "init |0>",
            {
                1,
            },
        )
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["ZI", "IZ"]
        # init_destab = state.destabs.print_tableau(verbose=False)[0]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["ZI"]
        assert stab_rep[1] == stab_dict["IZ"]

        # iWI, iIW
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+i>",
            {
                0,
            },
        )
        state.run_gate(
            "init |+i>",
            {
                1,
            },
        )
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["YI", "IY"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["YI"]
        assert stab_rep[1] == stab_dict["IY"]

        # by now we have shown the single Cliffords and CNOT: XI -> XX, IZ -> ZZ

        # XX, ZZ
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        state.run_gate(
            "init |0>",
            {
                1,
            },
        )
        state.run_gate(
            "CNOT",
            {
                (0, 1),
            },
        )
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["XX", "ZZ"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["XX"], f"{stab_rep[0]} != {stab_dict['XX']}"
        assert stab_rep[1] == stab_dict["ZZ"]

        # ZX, XZ
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        state.run_gate(
            "init |0>",
            {
                1,
            },
        )
        state.run_gate(
            "CNOT",
            {
                (0, 1),
            },
        )
        state.run_gate(
            "H",
            {
                0,
            },
        )
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["ZX", "XZ"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["ZX"]
        assert stab_rep[1] == stab_dict["XZ"]

        # iXW, iWZ
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        state.run_gate(
            "init |0>",
            {
                1,
            },
        )
        state.run_gate(
            "CNOT",
            {
                (0, 1),
            },
        )  # -> XX, ZZ
        state.run_gate(
            "H5",
            {
                0,
            },
        )  # -> -XX, iWZ
        state.run_gate(
            "H3",
            {
                1,
            },
        )  # -> -iXW, -iWZ
        state.run_gate(
            "Y",
            {
                0,
            },
        )  # -> iXW, -iWZ
        state.run_gate(
            "Y",
            {
                1,
            },
        )  # -> iXW, iWZ
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["XY", "YZ"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["XY"], f"{stab_rep[0]} != {stab_dict['XY']}"
        assert stab_rep[1] == stab_dict["YZ"]

        # iWX, iZW
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        state.run_gate(
            "init |0>",
            {
                1,
            },
        )
        state.run_gate(
            "CNOT",
            {
                (0, 1),
            },
        )  # -> XX, ZZ
        state.run_gate(
            "H5",
            {
                1,
            },
        )  # -> -XX, iZW
        state.run_gate(
            "H3",
            {
                0,
            },
        )  # -> -iWX, -iZW
        state.run_gate(
            "Y",
            {
                0,
            },
        )  # -> iWX, -iZW
        state.run_gate(
            "Y",
            {
                1,
            },
        )  # -> iWX, iZW
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["YX", "ZY"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["YX"]
        assert stab_rep[1] == stab_dict["ZY"]

        # -WW
        state = state_cls(2)
        # control -> target
        state.run_gate(
            "init |+>",
            {
                0,
            },
        )
        state.run_gate(
            "CNOT",
            {
                (0, 1),
            },
        )  # -> XX, ZZ
        state.run_gate(
            "H3",
            {
                0,
            },
        )  # -> iXW, -ZZ
        state.run_gate(
            "H3",
            {
                1,
            },
        )  # -> YY, ZZ
        assert tqrefsym(state.stabs.print_tableau(verbose=False)) == ["YY", "ZZ"]
        state.run_gate(
            gate_symbol,
            {
                (0, 1),
            },
        )
        stab_rep = tqrefsym(state.stabs.print_tableau(verbose=False))
        assert stab_rep[0] == stab_dict["YY"]


def test_SqrtXX() -> None:
    """Test 'Sqrt XX test'."""
    stab_transform = {
        "XI": "XI",
        "ZI": "-YX",
        "YI": "ZX",
        "IX": "IX",
        "IZ": "-XY",
        "IY": "XZ",
        "XX": "XX",
        "XZ": "-IY",
        "XY": "IZ",
        "ZX": "-YI",
        "ZZ": "ZZ",
        "ZY": "ZY",
        "YX": "ZI",
        "YZ": "YZ",
        "YY": "YY",
    }

    gate_test_tq("SqrtXX", stab_transform)


def test_SqrtZZ() -> None:
    """Test 'SqrtZZ test'."""
    stab_transform = {
        "XI": "YZ",
        "IX": "ZY",
        "ZI": "ZI",
        "IZ": "IZ",
        "YI": "-XZ",
        "IY": "-ZX",
        "ZZ": "ZZ",
        "ZX": "IY",
        "ZY": "-IX",
        "XZ": "YI",
        "XX": "XX",
        "XY": "XY",
        "YX": "YX",
        "YZ": "-XI",
        "YY": "YY",
    }

    check_transformation_tq(stab_transform, g.sqrtzz_def)

    gate_test_tq("SqrtZZ", stab_transform)
