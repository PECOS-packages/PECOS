"""Test SLR to physical quantum circuit compilation for various cases."""

import re

import pytest
from pecos.slr import Barrier, Block, Comment, CReg, If, Main, Parallel, QReg, Repeat, Return, SlrConverter
from pecos.slr.qeclib import qubit as p
from pecos.slr.qeclib.steane.steane_class import Steane

# TODO: Remove reference to hqslib1.inc... better yet, don't have tests on qasm


def telep(prep_basis: str, meas_basis: str) -> str:
    """A simple example of creating a logical teleportation circuit.

    Args:
        prep_basis (str):  A string indicating what Pauli basis to prepare the state in. Acceptable inputs include:
            "+X"/"X", "-X", "+Y"/"Y", "-Y", "+Z"/"Z", and "-Z".
        meas_basis (str): A string indicating what Pauli basis the measure out the logical qubit in. Acceptable inputs
            include: "X", "Y", and "Z".

    Returns:
        A logical program written in extended OpenQASM 2.0
    """
    return Main(
        m_bell := CReg("m_bell", size=2),
        m_out := CReg("m_out", size=1),
        # Input state:
        sin := Steane("sin", default_rus_limit=2),
        smid := Steane("smid"),
        sout := Steane("sout"),
        # Create Bell state
        smid.pz(),  # prep logical qubit in |0>/|+Z> state with repeat-until-success initialization
        sout.pz(),
        Barrier(smid.d, sout.d),
        smid.h(),
        smid.cx(sout),  # CX with control on smid and target on sout
        smid.qec(),
        sout.qec(),
        # prepare input state in some Pauli basis state
        sin.p(prep_basis, rus_limit=3),
        sin.qec(),
        # entangle input with one of the logical qubits of the Bell pair
        sin.cx(smid),
        sin.h(),
        # Bell measurement
        sin.mz(m_bell[0]),
        smid.mz(m_bell[1]),
        # Corrections
        If(m_bell[1] == 0).Then(sout.x()),
        If(m_bell[0] == 0).Then(sout.z()),
        # Final output stored in `m_out[0]`
        sout.m(meas_basis, m_out[0]),
        Return(m_bell, m_out),
    )


@pytest.mark.optional_dependency
def test_bell_qir() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog: Main = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.H(q[0]),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir


@pytest.mark.optional_dependency
def test_bell_qreg_qir() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog: Main = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.H(q),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir


@pytest.mark.optional_dependency
def test_qir_creg_size_too_large() -> None:
    """Test that a simple Bell prep and measure circuit can be created."""
    prog: Main = Main(
        q := QReg("q", 2),
        m := CReg("m", 75),
        p.H(q[0]),
        p.CX(q[0], q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    # #76/#80: the M-B2-static classical model packs each CReg into a
    # single i64 (`__quantum__rt__int_record_output`), so a >64-bit
    # CReg fails LOUD with NotImplementedError (was the older
    # ValueError message; updated to the current #76/#80 guard).
    with pytest.raises(NotImplementedError, match=re.escape("has 75 bits")):
        SlrConverter(prog).qir()


@pytest.mark.optional_dependency
def test_control_flow_qir() -> None:
    """Test a program with control flow into QIR."""
    prog = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        m_hidden := CReg("m_hidden", 2),
        Repeat(3).block(
            p.H(q[0]),
        ),
        Comment("Comments go here"),
        If(m == 0)
        .Then(
            p.H(q[0]),
            Block(
                p.H(q[1]),
            ),
        )
        .Else(
            p.RX[0.3](q[0]),
        ),
        If(m < m_hidden).Then(
            p.H(q[0]),
        ),
        Barrier(q[0], q[1]),
        p.F4dg(q[1]),
        p.SZdg(q[0]),
        p.CX(q[0], q[1]),
        Barrier(q[1], q[0]),
        p.RX[0.3](q[0]),
        p.Measure(q) > m,
        Return(m),
    )
    # #74/#80: whole-CReg scalar conditions (`If(m == 0)` /
    # `If(m < m_hidden)`) are classical-variable lowering, which the
    # QIR backend deliberately fails LOUD on (unimplemented; must not
    # be silently evaluated as 0). Correct post-#74/#80 behavior.
    with pytest.raises(NotImplementedError, match=r"classical variable"):
        SlrConverter(prog).qir()


@pytest.mark.optional_dependency
def test_plus_qir() -> None:
    """Test a program with addition compiling into QIR."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),
        m.set(2),
        n.set(2),
        o.set(m + n),
        Return(m, n, o),
    )
    # #74/#80: whole-CReg scalar arithmetic (`o.set(m + n)`) is
    # classical-variable lowering, deliberately FAIL-LOUD in the QIR
    # backend (unimplemented; must not be silently evaluated as 0).
    with pytest.raises(NotImplementedError, match=r"classical variable"):
        SlrConverter(prog).qir()


@pytest.mark.optional_dependency
def test_nested_xor_qir() -> None:
    """Test a program with addition compiling into QIR."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),
        p := CReg("p", 2),
        m.set(2),
        n.set(2),
        o.set(2),
        p[0].set((m[0] ^ n[0]) ^ o[0]),
        Return(m, n, o, p),
    )
    qir = SlrConverter(prog).qir()
    assert "xor" in qir


@pytest.mark.optional_dependency
def test_minus_qir() -> None:
    """Test a program with addition compiling into QIR."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),
        m.set(2),
        n.set(2),
        o.set(m - n),
        Return(m, n, o),
    )
    # #74/#80: whole-CReg scalar arithmetic (`o.set(m - n)`) is
    # classical-variable lowering, deliberately FAIL-LOUD in the QIR
    # backend (unimplemented; must not be silently evaluated as 0).
    with pytest.raises(NotImplementedError, match=r"classical variable"):
        SlrConverter(prog).qir()


@pytest.mark.optional_dependency
def test_steane_qir() -> None:
    """Test the teleportation program using the Steane code."""
    # #74/#80: the Steane teleportation uses a classical scalar var
    # (`smid_flag_x`), classical-variable lowering -> deliberately
    # FAIL-LOUD in the QIR backend (unimplemented; not silent-0).
    with pytest.raises(NotImplementedError, match=r"classical variable"):
        SlrConverter(telep("X", "X")).qir()


@pytest.mark.optional_dependency
def test_steane_qir_bc() -> None:
    """Test the teleportation program using the Steane code."""
    # #74/#80: same `smid_flag_x` classical scalar var -> the QIR
    # bitcode path also fails LOUD (unimplemented classical-variable
    # lowering; must not be silently evaluated as 0).
    with pytest.raises(NotImplementedError, match=r"classical variable"):
        SlrConverter(telep("X", "X")).qir_bc()


@pytest.mark.optional_dependency
def test_sx_sxdg() -> None:
    """SX/SXdg lower to a verified executable-Clifford sequence.

    #93: SX/SXdg have no direct QIR primitive but ARE Clifford
    sqrt-X gates. They lower to `H;S;H` / `H;Sdg;H` (executable
    Clifford only -- NOT rx, which is a pinned build/exec failure
    and would silently no-op on the Stim backend). The sequence was
    verified equal up to a global phase to the PECOS `StateVec`
    simulator's unitary AND end-to-end via the #79 executable path
    (SX;SX == X, SXdg;SX == I). (#88A's earlier fail-loud was the
    correct interim until this verified lowering landed.)
    """
    prog: Main = Main(
        q := QReg("q", 2),
        m := CReg("m", 2),
        p.CX(q[0], q[1]),
        p.SX(q[0]),
        p.SXdg(q[1]),
        p.Measure(q) > m,
        Return(m),
    )

    qir = SlrConverter(prog).qir()
    # SX(q0) -> h;s;h ; SXdg(q1) -> h;s__adj;h. No rotation, no
    # NotImplementedError, deterministic.
    assert "__quantum__qis__h__body" in qir
    assert "__quantum__qis__s__body" in qir
    assert "__quantum__qis__s__adj" in qir
    assert "__quantum__qis__rx__body" not in qir, "SX must NOT lower to rx (not executable)"
    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_parallel_qir() -> None:
    """Test that a parallel block can be compiled to QIR."""
    prog: Main = Main(
        q := QReg("q", 4),
        m := CReg("m", 4),
        Parallel(
            p.H(q[0]),
            p.X(q[1]),
            p.Y(q[2]),
            p.Z(q[3]),
        ),
        p.Measure(q) > m,
        Return(m),
    )
    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir
    assert "__quantum__qis__x__body" in qir
    assert "__quantum__qis__y__body" in qir
    assert "__quantum__qis__z__body" in qir


@pytest.mark.optional_dependency
def test_nested_parallel_qir() -> None:
    """Test that nested parallel blocks can be compiled to QIR."""
    prog: Main = Main(
        q := QReg("q", 4),
        m := CReg("m", 4),
        Parallel(
            p.H(q[0]),
            Block(
                p.X(q[1]),
                p.Y(q[2]),
            ),
            p.Z(q[3]),
        ),
        Barrier(q),
        p.Measure(q) > m,
        Return(m),
    )
    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir
    assert "__quantum__qis__x__body" in qir
    assert "__quantum__qis__y__body" in qir
    assert "__quantum__qis__z__body" in qir


@pytest.mark.optional_dependency
def test_parallel_in_control_flow_qir() -> None:
    """Test parallel blocks within control flow structures in QIR."""
    prog: Main = Main(
        q := QReg("q", 4),
        m := CReg("m", 4),
        p.H(q[0]),
        p.Measure(q[0]) > m[0],
        If(m[0] == 1).Then(
            Parallel(
                p.X(q[1]),
                p.Y(q[2]),
                p.Z(q[3]),
            ),
        ),
        Repeat(2).block(
            Parallel(
                p.RX[0.5](q[0]),
                p.RY[0.5](q[1]),
                p.RZ[0.5](q[2]),
            ),
        ),
        p.Measure(q) > m,
        Return(m),
    )
    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir
    assert "__quantum__qis__x__body" in qir
    assert "__quantum__qis__rx__body" in qir
