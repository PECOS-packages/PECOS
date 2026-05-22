"""Tests for quantum permutation functionality in both QASM and QIR generation."""

import re

import pytest
from pecos.slr import CReg, Main, Permute, QReg, SlrConverter, rad
from pecos.slr.qeclib import qubit

# QASM Tests


def test_permutation_consistency_with_multiple_calls() -> None:
    """Test that multiple calls to qasm() produce the same result."""
    prog = Main(
        a := QReg("a", 2),
        b := QReg("b", 2),
        Permute(
            [a[0], a[1], b[0], b[1]],
            [b[0], b[1], a[0], a[1]],
        ),
        qubit.H(a[0]),  # Should become H b[0];
        qubit.X(a[1]),  # Should become X b[1];
        qubit.Z(b[0]),  # Should become Z a[0];
        qubit.Y(b[1]),  # Should become Y a[1];
    )

    qasm1 = SlrConverter(prog).qasm()
    qasm2 = SlrConverter(prog).qasm()
    qasm3 = SlrConverter(prog).qasm()

    assert qasm1 == qasm2
    assert qasm2 == qasm3

    # Check that the permutation was applied correctly
    assert "h b[0];" in qasm1.lower()
    assert "x b[1];" in qasm1.lower()
    assert "z a[0];" in qasm1.lower()
    assert "y a[1];" in qasm1.lower()


def test_quantum_permutation_qasm(quantum_permutation_program: tuple) -> None:
    """Test permutation with quantum gates in QASM generation."""
    prog, _, _ = quantum_permutation_program

    # Generate QASM
    qasm = SlrConverter(prog).qasm()

    # Verify that the QASM contains the correct permuted quantum operations
    assert "h b[0];" in qasm
    assert "cx b[0], a[1];" in qasm

    # Verify that running QASM generation twice produces consistent results
    qasm2 = SlrConverter(prog).qasm()
    assert qasm == qasm2, "QASM generation is not deterministic"


# QIR Tests
#
# A Permute is realized as a static logical relabel (a
# permutation_map consulted at every qubit/classical-bit lowering,
# mirroring the Guppy linearity tracker's `.permute()` -- QIR/Selene
# have no runtime permute intrinsic). These tests pin the realized
# post-permute targeting from the actual emitted QIR (qubit indices
# are deterministic in register-declaration order). The legacy
# `; Permutation:` comment is preserved. The bespoke
# @set_creg_bit/@mz_to_creg_bit helpers the old tests pinned were
# removed by the static CReg model (measurement is the standard 2-arg
# `@__quantum__qis__mz__body(%Qubit*, %Result*)`).


def _q(name: str, qir: str) -> list[int]:
    """All qubit indices a single-qubit `name` gate is applied to."""
    return [
        int(m) for m in re.findall(rf"call void @__quantum__qis__{name}__body\(%Qubit\* inttoptr \(i64 (\d+) ", qir)
    ]


@pytest.mark.optional_dependency
def test_quantum_permutation_qir(quantum_permutation_program: tuple) -> None:
    """Element-wise QReg Permute is realized (a[0] <-> b[0])."""
    prog, _, _ = quantum_permutation_program

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir
    # Qubits: a[0]=0, a[1]=1, b[0]=2, b[1]=3. After the swap, H(a[0])
    # targets b[0]'s qubit (2) and CX(a[0], a[1]) -> cnot(2, 1).
    assert _q("h", qir) == [2], qir
    cnot = re.findall(
        r"call void @__quantum__qis__cnot__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    assert cnot == [("2", "1")], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_permutation_with_bell_circuit_qir() -> None:
    """Element-wise QReg + CReg Permute is realized in a Bell circuit."""
    a = QReg("a", 2)
    b = QReg("b", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        m,
        n,
        Permute([a[0], b[1]], [b[1], a[0]]),
        Permute([m[0], n[0]], [n[0], m[0]]),
        qubit.H(a[0]),
        qubit.CX(a[0], a[1]),
        qubit.Measure(a[0]) > m[0],
        qubit.Measure(a[1]) > m[1],
    )

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[1], b[1] -> a[0]" in qir
    assert "; Permutation: m[0] -> n[0], n[0] -> m[0]" in qir
    # Qubits: a[0]=0, a[1]=1, b[0]=2, b[1]=3. a[0] now -> b[1] (q3):
    # H(a[0]) -> q3, CX(a[0], a[1]) -> cnot(3, 1).
    assert _q("h", qir) == [3], qir
    # Standard 2-arg measurement (the removed @mz_to_creg_bit is
    # gone): Measure(a[0]) reads q3, Measure(a[1]) reads q1.
    mz = re.findall(
        r"call void @__quantum__qis__mz__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), %Result\*",
        qir,
    )
    assert mz == ["3", "1"], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_comprehensive_qir_verification() -> None:
    """Two element-wise QReg Permutes are realized across many gates.

    Previously this silently passed on a miscompile (the
    `; Permutation` comment was emitted but the qubit remap was a
    no-op). The relabel is now real; pin the corrected targeting.
    """
    a = QReg("a", 2)
    b = QReg("b", 2)
    c = QReg("c", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        c,
        m,
        n,
        qubit.H(a[0]),  # q0
        qubit.X(a[1]),  # q1
        qubit.Y(b[0]),  # q2
        qubit.Z(b[1]),  # q3
        Permute([a[0], b[0]], [b[0], a[0]]),
        qubit.H(a[0]),  # -> b[0] = q2
        qubit.X(b[0]),  # -> a[0] = q0
        Permute([a[1], b[1]], [b[1], a[1]]),
        qubit.Y(a[1]),  # -> b[1] = q3
        qubit.Z(b[1]),  # -> a[1] = q1
        qubit.CX(a[0], b[1]),  # a[0]->b[0]=q2, b[1]->a[1]=q1
        qubit.Measure(a[0]) > m[0],  # a[0]->b[0]=q2
        qubit.Measure(b[1]) > n[0],  # b[1]->a[1]=q1
    )

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir
    assert "; Permutation: a[1] -> b[1], b[1] -> a[1]" in qir
    assert _q("h", qir) == [0, 2], qir  # initial a[0]=0, then ->b[0]=2
    assert _q("x", qir) == [1, 0], qir  # initial a[1]=1, then ->a[0]=0
    assert _q("y", qir) == [2, 3], qir  # initial b[0]=2, then ->b[1]=3
    assert _q("z", qir) == [3, 1], qir  # initial b[1]=3, then ->a[1]=1
    cnot = re.findall(
        r"call void @__quantum__qis__cnot__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), "
        r"%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\)\)",
        qir,
    )
    assert cnot == [("2", "1")], qir
    mz = re.findall(
        r"call void @__quantum__qis__mz__body\(%Qubit\* inttoptr \(i64 (\d+) to %Qubit\*\), %Result\*",
        qir,
    )
    assert mz == ["2", "1"], qir

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_rotation_gates_with_permutation() -> None:
    """Element-wise QReg Permute is realized across rotation gates."""
    a = QReg("a", 2)
    b = QReg("b", 2)

    prog = Main(
        a,
        b,
        qubit.RX(rad(0.1), a[0]),  # q0
        qubit.RY(rad(0.2), a[1]),  # q1
        qubit.RZ(rad(0.3), b[0]),  # q2
        qubit.SZ(b[1]),  # q3
        Permute([a[0], b[0]], [b[0], a[0]]),
        qubit.RX(rad(0.4), a[0]),  # -> b[0] = q2
        qubit.RY(rad(0.5), b[0]),  # -> a[0] = q0
        qubit.T(a[1]),  # unpermuted = q1
        qubit.Tdg(b[1]),  # unpermuted = q3
    )

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a[0] -> b[0], b[0] -> a[0]" in qir
    rx = re.findall(r"call void @__quantum__qis__rx__body\(double [^,]+, %Qubit\* inttoptr \(i64 (\d+) ", qir)
    ry = re.findall(r"call void @__quantum__qis__ry__body\(double [^,]+, %Qubit\* inttoptr \(i64 (\d+) ", qir)
    assert rx == ["0", "2"], qir  # initial a[0]=0, then ->b[0]=2
    assert ry == ["1", "0"], qir  # initial a[1]=1, then ->a[0]=0
    assert _q("t", qir) == [1], qir  # T(a[1]) unpermuted
    tdg = re.findall(r"call void @__quantum__qis__t__adj\(%Qubit\* inttoptr \(i64 (\d+) ", qir)
    assert tdg == ["3"], qir  # Tdg(b[1]) unpermuted

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_whole_register_qreg_permutation_realized_qir() -> None:
    """Whole-register *qubit*-register Permute IS realized in QIR.

    `Permute(a, b)` on two same-size QRegs relabels every (a, i) <->
    (b, i) so subsequent `a[i]`/`b[i]` references resolve to the
    swapped qubits, and emits the legacy-format `; Permutation: a <->
    b` comment. Protects the realizable path against regression.
    """
    a = QReg("a", 2)
    b = QReg("b", 2)

    prog = Main(
        a,
        b,
        qubit.H(a[0]),  # original a[0] = q0
        qubit.X(b[0]),  # original b[0] = q2
        Permute(a, b),  # whole-register qubit swap
        qubit.Y(a[0]),  # after swap -> original b[0] = q2
        qubit.Z(b[0]),  # after swap -> original a[0] = q0
    )

    qir = SlrConverter(prog).qir()

    assert "; Permutation: a <-> b" in qir, f"Expected whole-register permutation comment in QIR:\n{qir}"
    assert _q("h", qir) == [0], qir
    assert _q("x", qir) == [2], qir
    assert _q("y", qir) == [2], qir  # Y(a[0]) after swap -> q2
    assert _q("z", qir) == [0], qir  # Z(b[0]) after swap -> q0

    assert qir == SlrConverter(prog).qir(), "QIR generation is not deterministic"


@pytest.mark.optional_dependency
def test_non_bijective_permute_fails_loud() -> None:
    """A non-bijective Permute must fail loud, not silently miscompile.

    The bijectivity guard must validate the EXPANDED
    source/target lists BEFORE building the map -- a dict would
    collapse a duplicate expanded source, so a genuinely
    non-bijective Permute would compile (silent miscompile). All
    malformed shapes must raise.
    """
    # Distinct refs, src set != tgt set.
    a = QReg("a", 2)
    b = QReg("b", 1)
    prog = Main(a, b, Permute([a[0], a[1]], [a[1], b[0]]), qubit.H(a[0]))
    with pytest.raises(NotImplementedError, match=r"bijective over the same ref set"):
        SlrConverter(prog).qir()

    # Duplicate source ref: would collapse in a dict and bypass the
    # set-equality check -- must be rejected on the duplicate.
    a = QReg("a", 2)
    b = QReg("b", 1)
    prog = Main(a, b, Permute([a[0], a[0]], [b[0], a[0]]), qubit.H(a[0]))
    with pytest.raises(NotImplementedError, match=r"duplicate source ref"):
        SlrConverter(prog).qir()

    # Duplicate target ref (symmetry: the duplicate-target guard must
    # also be live, not just duplicate-source).
    a = QReg("a", 2)
    b = QReg("b", 1)
    prog = Main(a, b, Permute([a[0], b[0]], [a[0], a[0]]), qubit.H(a[0]))
    with pytest.raises(NotImplementedError, match=r"duplicate target ref"):
        SlrConverter(prog).qir()


def test_permute_realized_quantum_circuit() -> None:
    """Element-wise + whole-register Permute is realized in the
    QuantumCircuit codegen (was a silent no-op -- same class as the
    QIR Permute bug; the allocator_offsets swap never reached gate
    qubit-index resolution / self-cancelled for a whole-reg pair).
    """
    a = QReg("a", 2)
    b = QReg("b", 2)
    elem = Main(a, b, qubit.H(a[0]), qubit.X(b[0]), Permute([a[0], b[0]], [b[0], a[0]]), qubit.Y(a[0]), qubit.Z(b[0]))
    # a=0,1 b=2,3. After a[0]<->b[0]: Y(a[0])->q2, Z(b[0])->q0.
    expected = "QuantumCircuit([{'H': {0}}, {'X': {2}}, {'Y': {2}}, {'Z': {0}}])"
    assert str(SlrConverter(elem).quantum_circuit()) == expected

    a = QReg("a", 2)
    b = QReg("b", 2)
    whole = Main(a, b, qubit.H(a[0]), qubit.X(b[0]), Permute(a, b), qubit.Y(a[0]), qubit.Z(b[0]))
    assert str(SlrConverter(whole).quantum_circuit()) == expected

    # Non-bijective must fail loud, not silently mis-resolve.
    a = QReg("a", 2)
    b = QReg("b", 1)
    with pytest.raises(NotImplementedError, match=r"duplicate source ref"):
        SlrConverter(Main(a, b, Permute([a[0], a[0]], [b[0], a[0]]), qubit.H(a[0]))).quantum_circuit()


@pytest.mark.optional_dependency
def test_permute_realized_stim() -> None:
    """Element-wise + whole-register Permute is realized in the
    Stim codegen (was a silent no-op -- same class as the QIR Permute bug).
    """
    a = QReg("a", 2)
    b = QReg("b", 2)
    elem = Main(a, b, qubit.H(a[0]), qubit.X(b[0]), Permute([a[0], b[0]], [b[0], a[0]]), qubit.Y(a[0]), qubit.Z(b[0]))
    # a=0,1 b=2,3. After a[0]<->b[0]: Y(a[0])->q2, Z(b[0])->q0.
    assert str(SlrConverter(elem).stim()).split() == ["H", "0", "X", "2", "Y", "2", "Z", "0"]

    a = QReg("a", 2)
    b = QReg("b", 2)
    whole = Main(a, b, qubit.H(a[0]), qubit.X(b[0]), Permute(a, b), qubit.Y(a[0]), qubit.Z(b[0]))
    assert str(SlrConverter(whole).stim()).split() == ["H", "0", "X", "2", "Y", "2", "Z", "0"]

    a = QReg("a", 2)
    b = QReg("b", 1)
    with pytest.raises(NotImplementedError, match=r"bijective over the same ref set"):
        SlrConverter(Main(a, b, Permute([a[0], a[1]], [a[1], b[0]]), qubit.H(a[0]))).stim()
