"""Tests for QuantumCircuit to/from SLR conversion."""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parent / "../../../../quantum-pecos/src"),
)

import pytest
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.slr import CReg, For, If, Main, Parallel, QReg, Repeat, Return, SlrConverter, While
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator
from pecos.slr.qeclib import qubit
from pecos.slr.qeclib.qubit.measures import Measure


def _return_declared_cregs(prog: Main) -> Main:
    cregs = [var for var in prog.vars if isinstance(var, CReg)]
    if cregs:
        prog.extend(Return(*cregs))
    return prog


class TestQuantumCircuitToSLR:
    """Test conversion from QuantumCircuit to SLR format."""

    def test_basic_gates(self) -> None:
        """Test conversion of basic single-qubit gates."""
        qc = QuantumCircuit()
        qc.append({"H": {0, 1, 2}})  # Hadamards on qubits 0, 1, 2
        qc.append({"X": {0}, "Y": {1}, "Z": {2}})  # Different gates
        qc.append({"S": {0}, "SDG": {1}, "T": {2}})  # Phase gates

        slr_prog = _return_declared_cregs(SlrConverter.from_quantum_circuit(qc))

        # Convert to QASM to verify structure
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        # First tick - all H gates
        assert "h q[0]" in qasm
        assert "h q[1]" in qasm
        assert "h q[2]" in qasm

        # Second tick
        assert "x q[0]" in qasm
        assert "y q[1]" in qasm
        assert "z q[2]" in qasm

        # Third tick
        assert "s q[0]" in qasm or "rz(pi/2) q[0]" in qasm
        assert "sdg q[1]" in qasm or "rz(-pi/2) q[1]" in qasm
        assert "t q[2]" in qasm or "rz(pi/4) q[2]" in qasm

    def test_two_qubit_gates(self) -> None:
        """Test conversion of two-qubit gates."""
        qc = QuantumCircuit()
        qc.append({"CX": {(0, 1), (2, 3)}})  # Two CNOT gates in parallel
        qc.append({"CY": {(1, 2)}})
        qc.append({"CZ": {(0, 3)}})

        slr_prog = _return_declared_cregs(SlrConverter.from_quantum_circuit(qc))
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "cx q[2],q[3]" in qasm or "cx q[2], q[3]" in qasm
        assert "cy q[1],q[2]" in qasm or "cy q[1], q[2]" in qasm
        assert "cz q[0],q[3]" in qasm or "cz q[0], q[3]" in qasm

    def test_measurements(self) -> None:
        """Test conversion of measurement operations."""
        qc = QuantumCircuit()
        qc.append({"RESET": {0, 1}})  # Reset/prep
        qc.append({"H": {0}})
        qc.append({"CX": {(0, 1)}})
        qc.append({"Measure": {0, 1}})

        slr_prog = _return_declared_cregs(SlrConverter.from_quantum_circuit(qc))
        qasm = SlrConverter(slr_prog).qasm(skip_headers=True)

        assert "reset q[0]" in qasm
        assert "reset q[1]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm or "cx q[0], q[1]" in qasm
        assert "measure q[0]" in qasm
        assert "measure q[1]" in qasm

    def test_parallel_detection(self) -> None:
        """Test that parallel operations in same tick are detected."""
        qc = QuantumCircuit()
        # All gates in one tick - should become a Parallel block
        qc.append({"H": {0}, "X": {1}, "Y": {2}})
        qc.append({"CX": {(0, 1)}})

        slr_prog = SlrConverter.from_quantum_circuit(qc, optimize_parallel=True)

        # Check for Parallel block (either direct Parallel or Block containing multiple ops)
        def has_parallel_structure(op: object) -> bool:
            if op.__class__.__name__ == "Parallel":
                return True
            # If it's a Block with multiple operations, it came from a Parallel optimization
            return bool(
                op.__class__.__name__ == "Block" and hasattr(op, "ops") and len(op.ops) > 1,
            )

        has_parallel = any(has_parallel_structure(op) for op in slr_prog.ops)
        assert has_parallel, "Should have detected parallel operations"

    def test_empty_circuit(self) -> None:
        """Test conversion of empty circuit."""
        qc = QuantumCircuit()

        slr_prog = SlrConverter.from_quantum_circuit(qc)

        # Should have minimal structure
        assert hasattr(slr_prog, "vars")
        assert hasattr(slr_prog, "ops")


class TestSLRToQuantumCircuit:
    """Test conversion from SLR format to QuantumCircuit."""

    def test_basic_gates_to_qc(self) -> None:
        """Test conversion of basic gates from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 3),
            qubit.H(q[0]),
            qubit.X(q[1]),
            qubit.Y(q[2]),
            qubit.Z(q[0]),
            qubit.CX(q[0], q[1]),
        )

        # Use the already imported generator

        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Check the circuit structure
        assert len(qc) == 5  # 5 separate ticks (no parallel optimization)

        # Check specific gates
        tick0_gates = {symbol: locations for symbol, locations, _params in qc[0].items()}
        assert "H" in tick0_gates
        assert 0 in tick0_gates["H"]

        tick1_gates = {symbol: locations for symbol, locations, _params in qc[1].items()}
        assert "X" in tick1_gates
        assert 1 in tick1_gates["X"]

        tick2_gates = {symbol: locations for symbol, locations, _params in qc[2].items()}
        assert "Y" in tick2_gates
        assert 2 in tick2_gates["Y"]

        tick3_gates = {symbol: locations for symbol, locations, _params in qc[3].items()}
        assert "Z" in tick3_gates
        assert 0 in tick3_gates["Z"]

        tick4_gates = {symbol: locations for symbol, locations, _params in qc[4].items()}
        assert "CX" in tick4_gates
        assert (0, 1) in tick4_gates["CX"]

    def test_measurements_to_qc(self) -> None:
        """Test conversion of measurements from SLR to QuantumCircuit."""
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

        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Check for reset and measure operations
        circuit_str = str(qc)
        assert "RESET" in circuit_str or "PZ" in circuit_str
        assert "Measure" in circuit_str

    def test_parallel_block_to_qc(self) -> None:
        """Test conversion of Parallel blocks from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 3),
            Parallel(
                qubit.H(q[0]),
                qubit.X(q[1]),
                qubit.Y(q[2]),
            ),
            qubit.CX(q[0], q[1]),
        )

        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should have exactly 2 ticks
        assert len(qc) == 2, f"Expected 2 ticks but got {len(qc)}"

        # First tick should have all three gates
        tick0_gates = {symbol: locations for symbol, locations, _params in qc[0].items()}

        assert "H" in tick0_gates
        assert 0 in tick0_gates["H"]
        assert "X" in tick0_gates
        assert 1 in tick0_gates["X"]
        assert "Y" in tick0_gates
        assert 2 in tick0_gates["Y"]

        # Second tick should have CX
        tick1_gates = {symbol: locations for symbol, locations, _params in qc[1].items()}

        assert "CX" in tick1_gates
        assert (0, 1) in tick1_gates["CX"]

    def test_repeat_block_to_qc(self) -> None:
        """Test conversion of Repeat blocks from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 2),
            Repeat(3).block(
                qubit.H(q[0]),
                qubit.CX(q[0], q[1]),
            ),
        )

        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should have 6 ticks (3 repetitions x 2 gates)
        assert len(qc) == 6, f"Expected 6 ticks but got {len(qc)}"

        # Check pattern repeats
        def get_tick_gates(tick: object) -> dict:
            return {symbol: locations for symbol, locations, _params in tick.items()}

        for i in range(3):
            tick_h = get_tick_gates(qc[i * 2])
            tick_cx = get_tick_gates(qc[i * 2 + 1])
            assert "H" in tick_h
            assert 0 in tick_h["H"]
            assert "CX" in tick_cx
            assert (0, 1) in tick_cx["CX"]

    def test_for_loop_to_qc(self) -> None:
        """Test conversion of For loops from SLR to QuantumCircuit."""
        prog = Main(
            q := QReg("q", 2),
            For("i", range(2)).Do(
                qubit.H(q[0]),
                qubit.X(q[1]),
            ),
        )

        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(prog)
        qc = generator.get_circuit()

        # Should unroll the loop
        assert len(qc) == 4, f"Expected 4 ticks but got {len(qc)}"


class TestQuantumCircuitRoundTrip:
    """Test round-trip conversions between QuantumCircuit and SLR."""

    def test_qc_round_trip(self) -> None:
        """Test QuantumCircuit -> SLR -> QuantumCircuit preserves structure."""
        original = QuantumCircuit()
        original.append({"H": {0, 1}})
        original.append({"CX": {(0, 1)}})
        original.append({"Measure": {0, 1}})

        # Convert to SLR
        slr_prog = SlrConverter.from_quantum_circuit(original)

        # Convert back to QuantumCircuit
        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(slr_prog)
        reconstructed = generator.get_circuit()

        # Both should have same number of ticks
        assert len(original) == len(reconstructed)

        # Check each tick matches
        def get_tick_gates(tick: object) -> dict:
            return {symbol: locations for symbol, locations, _params in tick.items()}

        for i in range(len(original)):
            orig_tick = get_tick_gates(original[i])
            recon_tick = get_tick_gates(reconstructed[i])

            # Same gates in each tick
            assert set(orig_tick.keys()) == set(recon_tick.keys())

            # Same targets for each gate
            for gate in orig_tick:
                assert orig_tick[gate] == recon_tick[gate]

    def test_slr_to_qc_round_trip(self) -> None:
        """Test SLR -> QuantumCircuit -> SLR preserves program structure."""
        original = Main(
            q := QReg("q", 3),
            Parallel(
                qubit.H(q[0]),
                qubit.H(q[1]),
                qubit.H(q[2]),
            ),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
        )

        # Convert to QuantumCircuit
        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(original)
        qc = generator.get_circuit()

        # Convert back to SLR
        reconstructed = SlrConverter.from_quantum_circuit(qc, optimize_parallel=True)

        # Convert both to QASM for comparison
        orig_qasm = SlrConverter(original).qasm(skip_headers=True)
        recon_qasm = SlrConverter(reconstructed).qasm(skip_headers=True)

        # Check key operations are preserved
        single_qubit_ops = ["h q[0]", "h q[1]", "h q[2]"]
        for op in single_qubit_ops:
            assert op in orig_qasm, f"'{op}' not in original QASM"
            assert op in recon_qasm, f"'{op}' not in reconstructed QASM"

        # Check CX gates with flexible formatting
        cx_ops = [("cx q[0],q[1]", "cx q[0], q[1]"), ("cx q[1],q[2]", "cx q[1], q[2]")]
        for op_nospace, op_space in cx_ops:
            assert (
                op_nospace in orig_qasm or op_space in orig_qasm
            ), f"Neither '{op_nospace}' nor '{op_space}' in original QASM"
            assert (
                op_nospace in recon_qasm or op_space in recon_qasm
            ), f"Neither '{op_nospace}' nor '{op_space}' in reconstructed QASM"

    def test_complex_circuit_preservation(self) -> None:
        """Test that complex circuit features are preserved."""
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 4),
            # Initialize
            qubit.PZ(q[0]),
            qubit.PZ(q[1]),
            qubit.PZ(q[2]),
            qubit.PZ(q[3]),
            # Create entanglement
            qubit.H(q[0]),
            qubit.CX(q[0], q[1]),
            qubit.CX(q[1], q[2]),
            qubit.CX(q[2], q[3]),
            # Measure
            qubit.Measure(q[0]) > c[0],
            qubit.Measure(q[1]) > c[1],
            qubit.Measure(q[2]) > c[2],
            qubit.Measure(q[3]) > c[3],
            Return(c),
        )

        # Convert to QuantumCircuit and back
        generator = QuantumCircuitGenerator(_internal=True)
        generator.generate_block(prog)
        qc = generator.get_circuit()

        reconstructed = _return_declared_cregs(SlrConverter.from_quantum_circuit(qc))

        # Both should produce similar QASM
        orig_qasm = SlrConverter(prog).qasm(skip_headers=True)
        recon_qasm = SlrConverter(reconstructed).qasm(skip_headers=True)

        # Check all major operations are present
        for op in ["reset", "h q[0]", "measure"]:
            assert op in orig_qasm.lower()
            assert op in recon_qasm.lower()

        # Check CX gates with flexible formatting
        cx_gates = [
            ("cx q[0],q[1]", "cx q[0], q[1]"),
            ("cx q[1],q[2]", "cx q[1], q[2]"),
            ("cx q[2],q[3]", "cx q[2], q[3]"),
        ]
        for op_nospace, op_space in cx_gates:
            assert op_nospace in orig_qasm.lower() or op_space in orig_qasm.lower()
            assert op_nospace in recon_qasm.lower() or op_space in recon_qasm.lower()


class TestQuantumCircuitStaticForAndWhile:
    """Same silent-miscompile class as the QIR For/While, in the AST
    QuantumCircuit codegen.

    The AST converter wraps `For` range bounds in `LiteralExpr`, so the
    old `isinstance(node.start, int)` guard was always false -- the
    `else` *rejected every* static `For` (even valid `For(i, 0, 3)`)
    with a `TypeError`. Now: static `For` unrolls; symbolic bound fails
    loud with a clear message. (`While` already failed loud here.)
    """

    def test_static_for_unrolls_body_not_dropped(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            For("i", 0, 3).Do(qubit.X(q[0])),
            Measure(q[0]) > c[0],
            Return(c),
        )
        qc = SlrConverter(prog).quantum_circuit()
        # 3 unrolled X on qubit 0 in the QuantumCircuit op list.
        assert str(qc).count("{'X': {0}}") == 3, f"static For not unrolled 3x: {qc}"

    def test_while_raises_loud(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            While(c[0] == 0).Do(qubit.X(q[0]), Measure(q[0]) > c[0]),
            Return(c),
        )
        with pytest.raises(NotImplementedError, match=r"While loops cannot be converted"):
            SlrConverter(prog).quantum_circuit()


def _legacy_quantum_circuit(prog: Main) -> QuantumCircuit:
    generator = QuantumCircuitGenerator(_internal=True)
    generator.generate_block(prog)
    return generator.get_output()


def _ast_quantum_circuit(prog: Main) -> QuantumCircuit:
    return SlrConverter(prog).quantum_circuit()


class TestQuantumCircuitIf:
    """The circuit records no classical results, so an If cannot be conditioned.

    Both generators used to emit If branches without their condition: the AST
    one ran both branches, and the deprecated one dropped the then-branch and
    ran the else-branch (#1237).
    """

    @pytest.mark.parametrize("generate", [_ast_quantum_circuit, _legacy_quantum_circuit], ids=["ast", "legacy"])
    @pytest.mark.parametrize("shape", ["then", "then-else", "inside-repeat"])
    def test_if_is_rejected(self, generate, shape: str) -> None:
        q, c = QReg("q", 2), CReg("c", 1)
        conditional = If(c[0] == 1).Then(qubit.X(q[1]))
        if shape == "then-else":
            conditional.Else(qubit.Z(q[1]))
        body = Repeat(2).block(conditional) if shape == "inside-repeat" else conditional
        prog = Main(q, c, Measure(q[0]) > c[0], body, Return(c))
        with pytest.raises(NotImplementedError, match=r"If cannot be converted to QuantumCircuit"):
            generate(prog)

    @pytest.mark.parametrize("generate", [_ast_quantum_circuit, _legacy_quantum_circuit], ids=["ast", "legacy"])
    def test_classical_assignments_without_if_still_convert(self, generate) -> None:
        """Classical values cannot reach a quantum operation without an If, so the circuit stays exact."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            Measure(q[0]) > c[0],
            c[1].set(c[0] ^ 1),
            qubit.H(q[0]),
            Return(c),
        )
        assert "{'H': {0}}" in str(generate(prog))


def test_repeated_mx_and_my_agree() -> None:
    """MX and MY leave the qubit in the measured eigenstate."""
    qc = QuantumCircuit()
    qc.append({"H": {0}})
    qc.append({"MX": {0}})
    qc.append({"MX": {0}})
    qc.append({"MY": {1}})
    qc.append({"MY": {1}})

    slr_prog = _return_declared_cregs(SlrConverter.from_quantum_circuit(qc))
    samples = SlrConverter(slr_prog).stim().compile_sampler(seed=7).sample(200)
    assert (samples[:, 0] == samples[:, 1]).all()
    assert (samples[:, 2] == samples[:, 3]).all()


@pytest.mark.parametrize(
    ("symbol", "basis"),
    [
        ("M", "Z"),
        ("mz", "Z"),
        ("mX", "X"),
        ("my", "Y"),
        ("MEASURE", "Z"),
        ("measure Z", "Z"),
        ("Measure", "Z"),
        ("Measure +Z", "Z"),
        ("Measure Z", "Z"),
        ("measure", "Z"),
        ("measure X", "X"),
        ("Measure +X", "X"),
        ("measure Y", "Y"),
        ("Measure +Y", "Y"),
    ],
)
@pytest.mark.parametrize("tuple_ticks", [False, True])
def test_measurement_aliases(symbol: str, basis: str, *, tuple_ticks: bool) -> None:
    """Sizing, emission and advancement share the same aliases in both tick formats.

    Qubit 0 is prepared in the -1 eigenstate of the measured basis, so the
    records are [1, 0] only when the alias is measured in its basis and the
    offset advances past it.
    """
    preparation = {"Z": ["X"], "X": ["H", "Z"], "Y": ["H", "S", "Z"]}[basis]
    ticks = [(gate, {0}, {}) for gate in preparation] + [(symbol, {(0,)}, {}), ("M", {1}, {})]
    qc = ticks
    if not tuple_ticks:
        qc = QuantumCircuit()
        for gate, locations, _params in ticks:
            qc.append({gate: locations})
    prog = _return_declared_cregs(SlrConverter.from_quantum_circuit(qc))
    assert sum(len(var) for var in prog.vars if isinstance(var, CReg)) == 2
    samples = SlrConverter(prog).stim().compile_sampler(seed=7).sample(50)
    assert samples.shape == (50, 2)
    assert (samples == [1, 0]).all()
    qasm = SlrConverter(prog).qasm(skip_headers=True)
    assert qasm.count("-> c[0]") == 1
    assert qasm.count("-> c[1]") == 1


@pytest.mark.parametrize("symbol", ["M", "MX", "MY", "measure X", "Measure +Y"])
@pytest.mark.parametrize("location", [(), (0, 1), ("bad",), "bad"])
def test_invalid_measurement_location(symbol: str, location: object) -> None:
    """Malformed locations fail before qubit sizing or measurement emission."""
    with pytest.raises(ValueError, match="Unsupported one-qubit location") as exc:
        SlrConverter.from_quantum_circuit([(symbol, {location}, {})])
    assert symbol in str(exc.value)
    assert repr(location) in str(exc.value)


@pytest.mark.parametrize("symbol", ["Measure -Z", "MeasureFree", "MPZ", "force output"])
def test_unknown_symbol_rejected(symbol: str) -> None:
    """Unknown symbols cannot be distinguished from unknown record producers."""
    qc = QuantumCircuit()
    qc.append({"X": {0}})
    qc.append({symbol: {0}})
    qc.append({"M": {1}})
    with pytest.raises(NotImplementedError, match="Unsupported QuantumCircuit gate") as exc:
        SlrConverter.from_quantum_circuit(qc)
    assert symbol in str(exc.value)
    assert "{0}" in str(exc.value)


@pytest.mark.parametrize("key", ["Pauli", "var_output", "var"])
@pytest.mark.parametrize("tuple_ticks", [False, True])
def test_measurement_params_rejected(key: str, *, tuple_ticks: bool) -> None:
    """Measurement parameters cannot silently change basis or destination."""
    params = {key: {0: ("c", 0)} if key == "var_output" else "X"}
    qc = [("measure Z", {0}, params)]
    if not tuple_ticks:
        qc = QuantumCircuit()
        qc.append({"measure Z": {0}}, **params)
    with pytest.raises(ValueError, match="Unsupported parameters") as exc:
        SlrConverter.from_quantum_circuit(qc)
    assert "measure Z" in str(exc.value)
    assert key in str(exc.value)


@pytest.mark.parametrize("symbol", ["RZ", "RX", "RY"])
@pytest.mark.parametrize("tuple_ticks", [False, True])
def test_rotation_parameters_rejected(symbol: str, *, tuple_ticks: bool) -> None:
    """QuantumCircuit rotations must not become resets."""
    qc = [(symbol, {0}, {"angles": [0.5]})]
    if not tuple_ticks:
        qc = QuantumCircuit()
        qc.append({symbol: {0}}, angles=[0.5])
    with pytest.raises(ValueError, match=symbol):
        SlrConverter.from_quantum_circuit(qc)


@pytest.mark.parametrize(("symbol", "params"), [("X", {"angles": [0.3]}), ("RESET", {"foo": 1})])
def test_nonmeasurement_params_rejected(symbol: str, params: dict) -> None:
    """No gate parameters may be silently ignored."""
    with pytest.raises(ValueError, match=symbol) as exc:
        SlrConverter.from_quantum_circuit([(symbol, {0}, params)])
    assert next(iter(params)) in str(exc.value)


@pytest.mark.parametrize(("symbol", "locations"), [("H", {(0, 1)}), ("CX", {0, 1})])
def test_malformed_gate_locations_rejected(symbol: str, locations: set) -> None:
    """Invalid arities must not silently remove gates."""
    with pytest.raises(ValueError, match=symbol) as exc:
        SlrConverter.from_quantum_circuit([(symbol, locations, {})])
    assert any(repr(loc) in str(exc.value) for loc in locations)


@pytest.mark.parametrize("symbol", ["RZ", "RX", "RY", "SX", "CH"])
def test_unsupported_quantum_circuit_gates(symbol: str) -> None:
    """Valid but unsupported gates report their symbol and locations."""
    locations = {(0, 1)} if symbol == "CH" else {0}
    with pytest.raises(NotImplementedError, match=f"Unsupported QuantumCircuit gate {symbol}"):
        SlrConverter.from_quantum_circuit([(symbol, locations, {})])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
