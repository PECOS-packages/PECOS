"""Test SLR to physical quantum circuit compilation for various cases."""

import re

import pytest
from pecos.slr import Barrier, Block, Comment, CReg, If, Main, Parallel, QReg, Repeat, Return, SlrConverter, rad
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

    # The static classical model packs each CReg into a
    # single i64 (`__quantum__rt__int_record_output`), so a >64-bit
    # CReg fails LOUD with NotImplementedError (was the older
    # ValueError message; updated to the current guard).
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
            p.RX(rad(0.3), q[0]),
        ),
        If(m < m_hidden).Then(
            p.H(q[0]),
        ),
        Barrier(q[0], q[1]),
        p.F4dg(q[1]),
        p.SZdg(q[0]),
        p.CX(q[0], q[1]),
        Barrier(q[1], q[0]),
        p.RX(rad(0.3), q[0]),
        p.Measure(q) > m,
        Return(m),
    )
    # Whole-CReg scalar conditions (`If(m == 0)` / `If(m <
    # m_hidden)`) are now lowered via `_pack_creg` + `_op_map`; the
    # QIR builds. CAVEAT: the `RX(rad(0.3), q)` rotations DO build + lower
    # (angle-first API; `rx` is qir-qis-allowlisted), but rx(0.3)
    # is NON-CLIFFORD so the program cannot execute on the Stim
    # backend -- it is build-/lower-only here, NOT end-to-end. (The
    # angle-first-executable rotation path is covered by the Guppy
    # Quest-backed behavioral suite; the pure-classical arithmetic is
    # covered end-to-end by `test_plus_qir` / `test_minus_qir`.)
    # Assert the build succeeds and the QIR has the expected
    # classical-comparison structure.
    qir = SlrConverter(prog).qir()
    # `If(m == 0)` and `If(m < m_hidden)` lower to icmp + i64 packs.
    assert "icmp" in qir, "If(m == 0) must lower to an icmp"
    assert "or i64" in qir, "whole-CReg pack must emit OR_i (zext c[i] << i)"


@pytest.mark.optional_dependency
def test_plus_qir() -> None:
    """Whole-CReg scalar arithmetic (`o.set(m + n)`) lowers via
    `_pack_creg` + i64 `add`, then unpacks back to `o`'s bits."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 3),  # 3 bits hold up to 7; 2-bit `o` would silently truncate m+n=2+2=4
        m.set(2),
        n.set(2),
        o.set(m + n),
        Return(m, n, o),
    )
    qir = SlrConverter(prog).qir()
    assert "add i64" in qir, "m + n must lower to an i64 add"

    # End-to-end: m=2, n=2, o=m+n=4. Pure-classical + Clifford only,
    # so the program executes through qir_to_qis + Stim.
    import qir_qis
    import selene_sim

    qis = qir_qis.qir_to_qis(SlrConverter(prog).qir_bc())
    inst = selene_sim.build(selene_sim.BitcodeString(qis))
    for shot in inst.run_shots(selene_sim.Stim(random_seed=1), n_qubits=2, n_shots=2):
        assert [v for (_t, v) in shot] == [2, 2, 4], shot


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
    """Whole-CReg scalar subtraction (`o.set(m - n)`) lowers via
    `_pack_creg` + i64 `sub`. Pure-classical + Clifford only -- runs
    end-to-end through qir_to_qis + Stim."""
    prog = Main(
        _q := QReg("q", 2),
        m := CReg("m", 2),
        n := CReg("n", 2),
        o := CReg("o", 2),  # 2 bits: m=3, n=1, o=m-n=2 fits.
        m.set(3),
        n.set(1),
        o.set(m - n),
        Return(m, n, o),
    )
    qir = SlrConverter(prog).qir()
    assert "sub i64" in qir, "m - n must lower to an i64 sub"

    import qir_qis
    import selene_sim

    qis = qir_qis.qir_to_qis(SlrConverter(prog).qir_bc())
    inst = selene_sim.build(selene_sim.BitcodeString(qis))
    for shot in inst.run_shots(selene_sim.Stim(random_seed=1), n_qubits=2, n_shots=2):
        assert [v for (_t, v) in shot] == [3, 1, 2], shot


@pytest.mark.optional_dependency
def test_steane_qir() -> None:
    """The Steane teleportation uses a classical scalar var
    (`smid_flag_x` -- a `CReg(..., 3)`); `_pack_creg` lowers
    this scalar reference, so the program builds. The full
    teleportation contains non-Clifford rotations and the executable
    path is out of scope for this test -- assert the build succeeds
    + the emitted QIR has the expected classical-pack structure."""
    qir = SlrConverter(telep("X", "X")).qir()
    assert "icmp" in qir, "Steane telep `If(flag == 0)` must lower to icmp"
    assert "or i64" in qir, "Steane telep flag pack must emit OR_i (zext c[i] << i)"


@pytest.mark.optional_dependency
def test_steane_qir_bc() -> None:
    """Same Steane telep program through the QIR bitcode path.
    The bitcode builds (no longer fails loud on classical-variable
    lowering)."""
    bc = SlrConverter(telep("X", "X")).qir_bc()
    assert bc, "qir_bc must return non-empty bitcode for Steane telep"


@pytest.mark.optional_dependency
def test_sx_sxdg() -> None:
    """SX/SXdg lower to a verified executable-Clifford sequence.

    SX/SXdg have no direct QIR primitive but ARE Clifford
    sqrt-X gates. They lower to `H;S;H` / `H;Sdg;H` (executable
    Clifford only -- NOT rx, which is a pinned build/exec failure
    and would silently no-op on the Stim backend). The sequence was
    verified equal up to a global phase to the PECOS `StateVec`
    simulator's unitary AND end-to-end via the executable path
    (SX;SX == X, SXdg;SX == I). (The earlier fail-loud was the
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
                p.RX(rad(0.5), q[0]),
                p.RY(rad(0.5), q[1]),
                p.RZ(rad(0.5), q[2]),
            ),
        ),
        p.Measure(q) > m,
        Return(m),
    )
    qir = SlrConverter(prog).qir()
    assert "__quantum__qis__h__body" in qir
    assert "__quantum__qis__x__body" in qir
    assert "__quantum__qis__rx__body" in qir
