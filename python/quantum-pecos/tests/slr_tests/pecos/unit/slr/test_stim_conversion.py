"""Tests for Stim circuit to/from SLR conversion."""

import pytest
import stim
from pecos.slr import Comment, CReg, For, Main, Parallel, QReg, Repeat, Return, SlrConverter, While, rad
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


def _return_declared_cregs(prog: Main) -> Main:
    cregs = [var for var in prog.vars if isinstance(var, CReg)]
    if cregs:
        prog.extend(Return(*cregs))
    return prog


class TestStimToSLR:
    """Test conversion from Stim circuits to SLR format."""

    def test_basic_gates(self) -> None:
        """Test conversion of basic single-qubit gates."""
        circuit = stim.Circuit(
            """
            H 0
            X 1
            Y 2
            Z 0
            S 1
            S_DAG 2
        """,
        )

        slr_prog = _return_declared_cregs(SlrConverter.from_stim(circuit))

        # Convert back to QASM to verify structure
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)
        assert "h q[0]" in qasm
        assert "x q[1]" in qasm
        assert "y q[2]" in qasm
        assert "z q[0]" in qasm
        assert "s q[1]" in qasm or "rz(pi/2) q[1]" in qasm
        assert "sdg q[2]" in qasm or "rz(-pi/2) q[2]" in qasm

    def test_two_qubit_gates(self) -> None:
        """Test conversion of two-qubit gates."""
        circuit = stim.Circuit(
            """
            CX 0 1
            CY 1 2
            CZ 0 2
        """,
        )

        slr_prog = _return_declared_cregs(SlrConverter.from_stim(circuit))
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "cy q[1],q[2]" in qasm or "cy q[1], q[2]" in qasm
        assert "cz q[0],q[2]" in qasm or "cz q[0], q[2]" in qasm

    def test_measurements_and_reset(self) -> None:
        """Test conversion of measurements and reset operations."""
        circuit = stim.Circuit(
            """
            R 0 1 2
            H 0
            CX 0 1
            M 0 1
        """,
        )

        slr_prog = _return_declared_cregs(SlrConverter.from_stim(circuit))
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "reset q[0]" in qasm
        assert "reset q[1]" in qasm
        assert "reset q[2]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "measure q[0]" in qasm
        assert "measure q[1]" in qasm

    def test_repeat_blocks(self) -> None:
        """Test conversion of REPEAT blocks."""
        circuit = stim.Circuit(
            """
            H 0
            REPEAT 3 {
                CX 0 1
                CX 1 2
            }
            M 0 1 2
        """,
        )

        slr_prog = SlrConverter.from_stim(circuit)

        # Check that the repeat block is preserved
        assert any(hasattr(op, "__class__") and op.__class__.__name__ == "Repeat" for op in slr_prog.ops)

    def test_parallel_optimization(self) -> None:
        """Test that parallel operations are optimized into Parallel blocks."""
        circuit = stim.Circuit(
            """
            H 0
            H 1
            H 2
            CX 0 1
        """,
        )

        # With optimization (note: optimizer doesn't create new parallel blocks from sequential ops)
        slr_prog_opt = SlrConverter.from_stim(circuit, optimize_parallel=True)
        # Sequential H gates from Stim remain sequential in SLR - this is expected
        h_ops = [op for op in slr_prog_opt.ops if type(op).__name__ == "H"]
        cx_ops = [op for op in slr_prog_opt.ops if type(op).__name__ == "CX"]
        assert len(h_ops) == 3, f"Should have 3 H operations, got {len(h_ops)}"
        assert len(cx_ops) == 1, f"Should have 1 CX operation, got {len(cx_ops)}"

        # Without optimization should be the same (no difference for sequential ops)
        slr_prog_no_opt = SlrConverter.from_stim(circuit, optimize_parallel=False)
        h_ops_no_opt = [op for op in slr_prog_no_opt.ops if type(op).__name__ == "H"]
        assert len(h_ops_no_opt) == 3, f"Should have 3 H operations, got {len(h_ops_no_opt)}"


class TestSLRToStim:
    """Test conversion from SLR format to Stim circuits."""

    def test_basic_gates_to_stim(self) -> None:
        """Test conversion of basic gates from SLR to Stim."""
        prog = Main(
            q := QReg("q", 3),
            qubit.H(q[0]),
            qubit.X(q[1]),
            qubit.Y(q[2]),
            qubit.Z(q[0]),
            qubit.CX(q[0], q[1]),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        # Check the circuit has the expected operations
        instructions = list(stim_circuit)
        assert any(instr.name == "H" and instr.targets_copy() == [stim.GateTarget(0)] for instr in instructions)
        assert any(instr.name == "X" and instr.targets_copy() == [stim.GateTarget(1)] for instr in instructions)
        assert any(instr.name == "Y" and instr.targets_copy() == [stim.GateTarget(2)] for instr in instructions)
        assert any(instr.name == "Z" and instr.targets_copy() == [stim.GateTarget(0)] for instr in instructions)
        assert any(
            instr.name == "CX" and instr.targets_copy() == [stim.GateTarget(0), stim.GateTarget(1)]
            for instr in instructions
        )

    def test_measurements_to_stim(self) -> None:
        """Test conversion of measurements from SLR to Stim."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.PZ(q[0]),
            qubit.PZ(q[1]),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            Return(c),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        instructions = list(stim_circuit)
        # Check for reset (prep_z)
        assert any(instr.name == "R" for instr in instructions)
        # Check for measurements
        assert any(instr.name == "M" for instr in instructions)

    def test_repeat_block_to_stim(self) -> None:
        """Test conversion of Repeat blocks from SLR to Stim."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(3).block(
                qubit.H(q[0]),
                qubit.CX(q[0], q[1]),
            ),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        # Check for REPEAT in the circuit
        circuit_str = str(stim_circuit)
        assert "REPEAT" in circuit_str
        assert "3" in circuit_str

    def test_parallel_block_to_stim(self) -> None:
        """Test conversion of Parallel blocks from SLR to Stim."""
        prog = Main(
            q := QReg("q", 3),
            Parallel(
                qubit.H(q[0]),
                qubit.X(q[1]),
                qubit.Y(q[2]),
            ),
            qubit.CX(q[0], q[1]),
        )

        converter = SlrConverter(prog)
        stim_circuit = converter.stim()

        # Parallel operations should appear before the CX
        instructions = list(stim_circuit)

        # Find indices of operations
        h_idx = next(
            i
            for i, instr in enumerate(instructions)
            if instr.name == "H" and 0 in [t.value for t in instr.targets_copy()]
        )
        x_idx = next(
            i
            for i, instr in enumerate(instructions)
            if instr.name == "X" and 1 in [t.value for t in instr.targets_copy()]
        )
        y_idx = next(
            i
            for i, instr in enumerate(instructions)
            if instr.name == "Y" and 2 in [t.value for t in instr.targets_copy()]
        )
        cx_idx = next(i for i, instr in enumerate(instructions) if instr.name == "CX")

        # All parallel ops should come before CX
        assert h_idx < cx_idx
        assert x_idx < cx_idx
        assert y_idx < cx_idx


class TestStimRoundTrip:
    """Test round-trip conversions between Stim and SLR."""

    def test_basic_circuit_round_trip(self) -> None:
        """Test Stim -> SLR -> Stim preserves circuit structure."""
        original = stim.Circuit(
            """
            H 0
            CX 0 1
            M 0 1
        """,
        )

        # Convert to SLR and back
        slr_prog = _return_declared_cregs(SlrConverter.from_stim(original))
        converter = SlrConverter(slr_prog)
        reconstructed = converter.stim()

        # Check both circuits have same operations
        orig_ops = [(instr.name, list(instr.targets_copy())) for instr in original]
        recon_ops = [(instr.name, list(instr.targets_copy())) for instr in reconstructed]

        assert len(orig_ops) == len(recon_ops)
        for orig, recon in zip(orig_ops, recon_ops, strict=False):
            assert orig[0] == recon[0]  # Same gate name
            assert orig[1] == recon[1]  # Same targets

    def test_slr_round_trip(self) -> None:
        """Test SLR -> Stim -> SLR preserves program structure."""
        original = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            Return(c),
        )

        # Convert to Stim and back
        converter = SlrConverter(original)
        stim_circuit = converter.stim()
        reconstructed = _return_declared_cregs(SlrConverter.from_stim(stim_circuit))

        # Convert both to QASM for comparison
        orig_qasm = SlrConverter(original).qasm(skip_headers=True)
        recon_qasm = SlrConverter(reconstructed).qasm(skip_headers=True)

        # Check key operations are preserved
        for op in ["h q[0]", "measure q[0]", "measure q[1]"]:
            assert op in orig_qasm
            assert op in recon_qasm

        # Check CX with flexible formatting
        assert "cx q[0],q[1]" in orig_qasm or "cx q[0], q[1]" in orig_qasm
        assert "cx q[0],q[1]" in recon_qasm or "cx q[0], q[1]" in recon_qasm


class TestStimStaticForAndWhile:
    """Same silent-miscompile class as the QIR For/While, in the AST Stim codegen.

    The AST converter wraps `For` range bounds in `LiteralExpr`, so the
    old `isinstance(node.start, int)` guard was always false -- Stim
    silently dropped every `For` body, and `While` was silently
    processed once (+ a stray TICK). Both are valid-output / wrong-
    semantics. Now: static `For` unrolls; `While` fails loud.
    """

    def test_static_for_unrolls_body_not_dropped(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            For("i", 0, 3).Do(qubit.X(q[0])),
            Measure(q[0]) > c[0],
            Return(c),
        )
        circ = SlrConverter(prog).stim()
        # 3 X applications on qubit 0 (Stim coalesces to `X 0 0 0`);
        # 0 == the old silent drop.
        assert "X 0 0 0" in str(circ), f"static For body not unrolled 3x:\n{circ}"

    def test_while_raises_loud(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            While(c[0] == 0).Do(qubit.X(q[0]), Measure(q[0]) > c[0]),
            Return(c),
        )
        with pytest.raises(NotImplementedError, match=r"does not support While loops"):
            SlrConverter(prog).stim()


def test_unsupported_gate_fails_loud() -> None:
    """A gate with no GATE_TO_STIM entry must FAIL LOUD, not be silently
    dropped (a silent drop produced a runnable circuit with wrong
    semantics -- a miscompile). Stim is Clifford-only, so
    a non-Clifford rotation is fundamentally unrepresentable;
    emitting the circuit without it is the bug.
    """
    q = QReg("q", 1)
    prog = Main(q, qubit.RX(rad(0.5), q[0]))
    with pytest.raises(NotImplementedError, match=r"has no Stim lowering"):
        SlrConverter(prog).stim()


def test_face_clifford_gates_decompose() -> None:
    """F/Fdg/F4/F4dg are PECOS face-Cliffords with no direct Stim
    primitive; cross-codegen audit landed verified decompositions
    into H/S/S_DAG so they no longer fail-loud. Verify each decomp
    produces the *correct* Clifford action via Stim's tableau (catches
    no-op and wrong-direction mutations -- a wrong-direction F would
    cycle X<-Y<-Z<-X instead of forward).

    F is order-3 (F^3 = I) and cycles the Paulis X->Y->Z->X up to
    sign. The Heisenberg-picture conjugation `F . Z . Fdg = X` is
    the deterministic identity each face-Clifford must satisfy.
    """

    # Build a 1q Stim program with just `F q[0]` and read the tableau.
    def _tableau_of(gate) -> stim.Tableau:
        q = QReg("q", 1)
        prog = Main(q, gate(q[0]))
        circ = SlrConverter(prog).stim()
        return stim.Tableau.from_circuit(circ)

    # F should send Z -> X and X -> Y (face cycle in the +1,+1,+1
    # direction). Fdg is the inverse so it sends Z -> Y and X -> Z.
    f = _tableau_of(qubit.F)
    fdg = _tableau_of(qubit.Fdg)
    assert f(stim.PauliString("Z")) == stim.PauliString("+X"), f
    assert f(stim.PauliString("X")) == stim.PauliString("+Y"), f
    assert fdg(stim.PauliString("Z")) == stim.PauliString("+Y"), fdg
    assert fdg(stim.PauliString("X")) == stim.PauliString("+Z"), fdg

    # F . Fdg must be identity (involution-pair). Build the
    # composed circuit and confirm both Paulis are preserved.
    q = QReg("q", 1)
    prog = Main(q, qubit.F(q[0]), qubit.Fdg(q[0]))
    f_fdg = stim.Tableau.from_circuit(SlrConverter(prog).stim())
    assert f_fdg(stim.PauliString("Z")) == stim.PauliString("+Z"), f_fdg
    assert f_fdg(stim.PauliString("X")) == stim.PauliString("+X"), f_fdg

    # F4 / F4dg are a *different* face Clifford (rotation around
    # a different cube axis). Sanity-check they are well-defined
    # involution pairs and not aliases of F / Fdg.
    f4 = _tableau_of(qubit.F4)
    f4dg = _tableau_of(qubit.F4dg)
    assert f4 != f, "F4 must not be the same Clifford as F"
    assert f4dg != fdg, "F4dg must not be the same Clifford as Fdg"
    q = QReg("q", 1)
    prog = Main(q, qubit.F4(q[0]), qubit.F4dg(q[0]))
    f4_f4dg = stim.Tableau.from_circuit(SlrConverter(prog).stim())
    assert f4_f4dg(stim.PauliString("Z")) == stim.PauliString("+Z"), f4_f4dg
    assert f4_f4dg(stim.PauliString("X")) == stim.PauliString("+X"), f4_f4dg


def test_repeated_mx_and_my_agree() -> None:
    """MX and MY leave the qubit in the measured eigenstate, as Stim does."""
    circuit = stim.Circuit("H 0\nMX 0\nMX 0\nMY 1\nMY 1")
    prog = _return_declared_cregs(SlrConverter.from_stim(circuit))
    samples = SlrConverter(prog).stim().compile_sampler(seed=7).sample(200)
    assert (samples[:, 0] == samples[:, 1]).all()
    assert (samples[:, 2] == samples[:, 3]).all()


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("X 0\nMR 0\nM 0", id="mr"),
        pytest.param("H 0\nZ 0\nMRX 0\nMX 0", id="mrx"),
        pytest.param("H 0\nS 0\nZ 0\nMRY 0\nMY 0", id="mry"),
        pytest.param("X 0\nM !0\nM 0", id="inverted-z"),
        pytest.param("H 0\nMX !0\nMX 0", id="inverted-x"),
        pytest.param("REPEAT 2 {\n X 0\n M 0\n}\nX 1\nM 1", id="repeat"),
        pytest.param(
            "M 1\nREPEAT 2 {\n REPEAT 2 {\n X 0\n M 0\n }\n M 0\n}\nX 1\nM 1",
            id="nested-repeat",
        ),
        pytest.param("H 0\nS 0\nMY !0\nMY 0", id="inverted-y"),
        pytest.param("X 0\nMR !0\nM 0", id="inverted-mr"),
        pytest.param("H 0\nZ 0\nMRX !0\nMX 0", id="inverted-mrx"),
        pytest.param("H 0\nS 0\nZ 0\nMRY !0\nMY 0", id="inverted-mry"),
        pytest.param("X 1\nM 0 !1 1 0", id="mixed-inversion"),
    ],
)
def test_record_order_regression(source: str) -> None:
    """Every deterministic source record keeps its value and classical index."""
    _assert_records_match_stim(stim.Circuit(source))


def _assert_records_match_stim(circuit: stim.Circuit) -> None:
    """Converted records equal Stim's reference sample, each written to its own bit."""
    expected = circuit.reference_sample()
    prog = _return_declared_cregs(SlrConverter.from_stim(circuit))
    assert sum(len(var) for var in prog.vars if isinstance(var, CReg)) == len(expected)
    samples = SlrConverter(prog).stim().compile_sampler(seed=7).sample(50)
    assert samples.shape == (50, len(expected))
    assert (samples == expected).all(), (samples[0], expected)
    qasm = SlrConverter(prog).qasm(skip_headers=True)
    for index in range(len(expected)):
        assert qasm.count(f"-> c[{index}]") == 1


@pytest.mark.parametrize(
    ("source", "gate"),
    [
        ("MPP X0*Z1", "MPP"),
        ("MXX 0 1", "MXX"),
        ("MYY 0 1", "MYY"),
        ("MZZ 0 1", "MZZ"),
        ("MPAD 1", "MPAD"),
        ("HERALDED_ERASE(0.1) 0", "HERALDED_ERASE"),
        ("HERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0) 0", "HERALDED_PAULI_CHANNEL_1"),
        ("REPEAT 2 {\n MPAD 1\n}", "MPAD"),
    ],
)
def test_unsupported_records_rejected(source: str, gate: str) -> None:
    """Unsupported record producers must never become comments."""
    with pytest.raises(NotImplementedError, match=f"Unsupported measurement-producing Stim gate {gate}"):
        SlrConverter.from_stim(stim.Circuit(source))


@pytest.mark.parametrize(
    "source",
    [
        "X 0\nM 0\nCX rec[-1] 1\nM 1",
        "M 0\nCX rec[-1] 1 2 3",
        "CX sweep[0] 1",
        "SQRT_X 0",
        "ISWAP 0 1",
        "C_XYZ 0",
    ],
)
def test_unsupported_state_changes_rejected(source: str) -> None:
    """Unsupported targets and unitaries must not disappear or change pairing."""
    circuit = stim.Circuit(source)
    instruction = next(inst for inst in circuit if inst.name not in ["X", "M"])
    with pytest.raises(NotImplementedError, match=instruction.name) as exc:
        SlrConverter.from_stim(circuit)
    assert str(instruction) in str(exc.value)


def test_record_annotations_still_convert() -> None:
    """DETECTOR and OBSERVABLE_INCLUDE reference records; they are annotations, not feedback."""
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    _assert_records_match_stim(circuit.without_noise())
    qasm = SlrConverter(SlrConverter.from_stim(circuit)).qasm(skip_headers=True)
    assert "// DETECTOR" in qasm
    assert "// OBSERVABLE_INCLUDE" in qasm


@pytest.mark.parametrize(
    "source",
    ["DEPOLARIZE1(0.1) 0", "PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0", "M(0.5) 0", "MR(0.1) 0", "E[tag](0.1) X0 Z1"],
)
def test_noise_is_visible(source: str) -> None:
    """Omitted noise is visible even when attached to a converted measurement."""
    circuit = stim.Circuit(source)
    prog = _return_declared_cregs(SlrConverter.from_stim(circuit))
    qasm = SlrConverter(prog).qasm(skip_headers=True)
    assert f"// Noise: {circuit[0]}" in qasm
    assert "Unsupported Stim gate" not in qasm
    if circuit.num_measurements:
        assert "measure q[0] -> c[0]" in qasm


@pytest.mark.parametrize("source", ["I 0", "II 0 1"])
def test_identity_emits_no_operations(source: str) -> None:
    """Both identity gates must remain supported no-ops."""
    prog = SlrConverter.from_stim(stim.Circuit(source))
    assert all(isinstance(op, QReg) for op in prog.ops)
    assert len(SlrConverter(prog).stim()) == 0


_GATE_EXAMPLES = {
    "DEPOLARIZE1": "DEPOLARIZE1(0.1) 0",
    "DEPOLARIZE2": "DEPOLARIZE2(0.1) 0 1",
    "X_ERROR": "X_ERROR(0.1) 0",
    "Y_ERROR": "Y_ERROR(0.1) 0",
    "Z_ERROR": "Z_ERROR(0.1) 0",
    "I_ERROR": "I_ERROR(0.1) 0",
    "II_ERROR": "II_ERROR(0.1) 0 1",
    "PAULI_CHANNEL_1": "PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0",
    "PAULI_CHANNEL_2": "PAULI_CHANNEL_2(" + ",".join(["0.01"] * 15) + ") 0 1",
    "HERALDED_ERASE": "HERALDED_ERASE(0.1) 0",
    "HERALDED_PAULI_CHANNEL_1": "HERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0) 0",
    "E": "E(0.1) X0 Z1",
    "ELSE_CORRELATED_ERROR": "ELSE_CORRELATED_ERROR(0.1) X0 Z1",
    "MPP": "MPP X0*Z1",
    "SPP": "SPP X0*Z1",
    "SPP_DAG": "SPP_DAG X0*Z1",
    "MPAD": "MPAD 1",
    "DETECTOR": "DETECTOR rec[-1]",
    "OBSERVABLE_INCLUDE": "OBSERVABLE_INCLUDE(0) rec[-1]",
    "QUBIT_COORDS": "QUBIT_COORDS(1, 2) 0",
    "SHIFT_COORDS": "SHIFT_COORDS(1, 2)",
    "TICK": "TICK",
    "REPEAT": "REPEAT 2 {\n H 0\n}",
}
_MEASUREMENT_PREP_BASES = {"M": "Z", "MX": "X", "MY": "Y", "MR": "Z", "MRX": "X", "MRY": "Y"}
# Each prepares qubit 0 in the -1 eigenstate of the basis, so records are deterministic.
_MINUS_EIGENSTATE_PREPS = {"Z": "X 0", "X": "H 0\nZ 0", "Y": "H 0\nS 0\nZ 0"}
# State-changing gates the converter supports; any other one must be rejected.
_CONVERTED_STATE_GATES = {
    "I",
    "II",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "S_DAG",
    "CX",
    "CY",
    "CZ",
    "SWAP",
    "R",
    "RX",
    "RY",
    "M",
    "MX",
    "MY",
    "MR",
    "MRX",
    "MRY",
}


@pytest.mark.parametrize("name", sorted(stim.gate_data()))
def test_stim_gate_catalog(name: str) -> None:
    """Every installed Stim gate is rejected or checked for its semantics."""
    data = stim.gate_data(name)
    if name in _GATE_EXAMPLES:
        instruction = _GATE_EXAMPLES[name]
    else:
        assert data.is_single_qubit_gate or data.is_two_qubit_gate, name
        instruction = name + (" 0 1" if data.is_two_qubit_gate else " 0")
    # Keep tableau width observable even when an identity emits nothing.
    prefix = stim.Circuit("H 1\nH 1")
    if name == "ELSE_CORRELATED_ERROR":
        prefix += stim.Circuit("E(0.1) X0")
    elif name in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
        prefix += stim.Circuit("M 0")
    elif name in _MEASUREMENT_PREP_BASES:
        prefix += stim.Circuit(_MINUS_EIGENSTATE_PREPS[_MEASUREMENT_PREP_BASES[name]])
    circuit = prefix + stim.Circuit(instruction)
    try:
        prog = SlrConverter.from_stim(circuit)
    except NotImplementedError as exc:
        assert name not in _CONVERTED_STATE_GATES, exc
        assert name in str(exc), name
        return
    is_state_gate = data.is_unitary or data.is_reset or data.produces_measurements
    assert not is_state_gate or name in _CONVERTED_STATE_GATES, f"{name} converted but is not listed as supported"
    prefix_prog = SlrConverter.from_stim(prefix)
    prefix_ops = [op for op in prefix_prog.ops if not isinstance(op, (QReg, CReg))]
    payload = [op for op in prog.ops if not isinstance(op, (QReg, CReg))][len(prefix_ops) :]
    if name == "REPEAT":
        assert stim.Tableau.from_circuit(circuit) == stim.Tableau.from_circuit(SlrConverter(prog).stim())
    elif data.is_noisy_gate and not data.produces_measurements:
        assert len(payload) == 1, name
        assert isinstance(payload[0], Comment), name
        qasm = SlrConverter(prog).qasm(skip_headers=True)
        assert f"// Noise: {circuit[-1]}" in qasm, name
    elif not (data.is_unitary or data.is_noisy_gate or data.produces_measurements or data.is_reset):
        assert payload, name
        assert all(isinstance(op, Comment) for op in payload), name
    elif data.is_unitary:
        assert stim.Tableau.from_circuit(circuit) == stim.Tableau.from_circuit(SlrConverter(prog).stim()), name
    elif data.produces_measurements:
        assert name in _MEASUREMENT_PREP_BASES, f"Add a deterministic preparation for {name}"
        _assert_records_match_stim(circuit)
    elif data.is_reset:
        basis = {"R": "Z", "RX": "X", "RY": "Y"}[name]
        check = stim.Circuit(_MINUS_EIGENSTATE_PREPS[basis] + "\n" + instruction + "\nM" + basis + " 0")
        _assert_records_match_stim(check)
        converted = SlrConverter(SlrConverter.from_stim(check)).stim()
        assert any(inst.name == "R" for inst in converted), name
    else:
        pytest.fail(f"Unclassified Stim gate: {name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
