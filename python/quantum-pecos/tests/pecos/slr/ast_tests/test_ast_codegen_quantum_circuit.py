# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for AST to QuantumCircuit code generator."""

import json
import re
from collections import Counter

import pytest
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.slr import Barrier, CReg, For, If, Main, Parallel, Permute, QReg, Repeat, Return, SlrConverter, While
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import (
    AstToQuantumCircuit,
    ast_to_quantum_circuit,
    ast_to_quantum_circuit_str,
)
from pecos.slr.qeclib import qubit as qb


def tick_to_dict(tick: object) -> dict[str, set]:
    """Convert TickView to a dict {gate_symbol: set(locations)}."""
    return {symbol: locations for symbol, locations, _ in tick.items()}


class TestAstToQuantumCircuitBasic:
    """Basic code generation tests."""

    def test_empty_program(self) -> None:
        """Empty program generates empty QuantumCircuit."""
        prog = Main()
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        assert isinstance(circuit, QuantumCircuit)
        assert len(circuit) == 0

    def test_program_with_qreg(self) -> None:
        """Program with QReg and gate generates non-empty circuit."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        assert isinstance(circuit, QuantumCircuit)
        assert len(circuit) > 0

    def test_string_output(self) -> None:
        """String output function returns string representation."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_quantum_circuit_str(ast)

        assert isinstance(code, str)


class TestAstToQuantumCircuitGates:
    """Gate code generation tests."""

    def test_hadamard_gate(self) -> None:
        """Hadamard gate creates tick with H gate on correct qubit."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # QuantumCircuit has ticks with gate dictionaries
        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "H" in tick
        assert 0 in tick["H"]

    def test_pauli_gates(self) -> None:
        """Pauli gates each create separate tick."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.Y(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # Each gate is a separate tick
        assert len(circuit) == 3

    def test_two_qubit_cx_gate(self) -> None:
        """CX gate creates tick with control-target pair."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "CX" in tick
        # CX stored as tuple (control, target)
        assert (0, 1) in tick["CX"]

    def test_two_qubit_cz_gate(self) -> None:
        """CZ gate creates tick with qubit pair."""
        prog = Main(
            q := QReg("q", 2),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "CZ" in tick
        assert (0, 1) in tick["CZ"]

    def test_multiple_gates_different_qubits(self) -> None:
        """Gates on different qubits each get their own tick."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # Each gate gets its own tick unless in parallel block
        assert len(circuit) == 2


class TestAstToQuantumCircuitPrepMeasure:
    """PZ and measure code generation tests."""

    def test_measurement(self) -> None:
        """Measurement creates tick with Measure operation."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "Measure" in tick
        assert 0 in tick["Measure"]

    def test_prep_reset(self) -> None:
        """PZ creates tick with RESET operation."""
        prog = Main(
            q := QReg("q", 1),
            qb.PZ(q[0]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "RESET" in tick
        assert 0 in tick["RESET"]


class TestAstToQuantumCircuitControlFlow:
    """Control flow code generation tests."""

    def test_barrier_flushes_tick(self) -> None:
        """Barrier forces operations into separate ticks."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            Barrier(q),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # H, then barrier flushes, then CX
        assert len(circuit) == 2

    def test_repeat_unrolled(self) -> None:
        """Repeat unrolls into multiple gate ticks."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # Repeat is unrolled: 3 H gates
        assert len(circuit) == 3
        for i in range(len(circuit)):
            tick = tick_to_dict(circuit[i])
            assert "H" in tick

    def test_while_raises_error(self) -> None:
        """While loop raises NotImplementedError for static circuits."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            While(c[0] == 1).Do(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        with pytest.raises(NotImplementedError):
            ast_to_quantum_circuit(ast)


class TestAstToQuantumCircuitParallel:
    """Parallel block tests."""

    def test_parallel_gates_same_tick(self) -> None:
        """Parallel gates are placed in the same tick."""
        from pecos.slr import Parallel

        prog = Main(
            q := QReg("q", 2),
            Parallel(
                qb.H(q[0]),
                qb.X(q[1]),
            ),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # Parallel operations should be in same tick
        assert len(circuit) == 1
        tick = tick_to_dict(circuit[0])
        assert "H" in tick
        assert "X" in tick


class TestAstToQuantumCircuitQEC:
    """QEC pattern code generation tests."""

    def test_syndrome_extraction(self) -> None:
        """Syndrome extraction generates correct qubit indices."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            qb.CX(data[0], ancilla[0]),
            qb.CX(data[1], ancilla[0]),
            qb.Measure(ancilla[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_quantum_circuit(ast)

        # Two CX gates and one measurement
        assert len(circuit) == 3

        # Check CX gates use correct qubit indices
        # data[0] -> 0, data[1] -> 1, ancilla[0] -> 2
        tick0 = tick_to_dict(circuit[0])
        tick1 = tick_to_dict(circuit[1])
        tick2 = tick_to_dict(circuit[2])
        assert "CX" in tick0
        assert (0, 2) in tick0["CX"]
        assert "CX" in tick1
        assert (1, 2) in tick1["CX"]
        assert "Measure" in tick2
        assert 2 in tick2["Measure"]


class TestAstToQuantumCircuitGenerator:
    """Tests for AstToQuantumCircuit generator class."""

    def test_generator_reusable(self) -> None:
        """Generator can be reused for multiple programs."""
        generator = AstToQuantumCircuit()

        prog1 = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )

        prog2 = Main(
            r := QReg("r", 2),
            qb.X(r[0]),
        )

        ast1 = slr_to_ast(prog1)
        ast2 = slr_to_ast(prog2)

        circuit1 = generator.generate(ast1)
        circuit2 = generator.generate(ast2)

        assert "H" in tick_to_dict(circuit1[0])
        assert "X" in tick_to_dict(circuit2[0])

    def test_qubit_mapping(self) -> None:
        """Generator creates correct qubit index mapping."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            qb.H(a[0]),
            qb.X(b[1]),
        )
        ast = slr_to_ast(prog)

        generator = AstToQuantumCircuit()
        generator.generate(ast)

        # a[0] -> 0, a[1] -> 1, b[0] -> 2, b[1] -> 3
        assert generator.context.qubit_map[("a", 0)] == 0
        assert generator.context.qubit_map[("a", 1)] == 1
        assert generator.context.qubit_map[("b", 0)] == 2
        assert generator.context.qubit_map[("b", 1)] == 3


class TestAstToQuantumCircuitFullPipeline:
    """End-to-end tests: SLR -> AST -> QuantumCircuit."""

    def test_bell_state_circuit(self) -> None:
        """Bell state generates H tick followed by CX tick."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_quantum_circuit(ast)

        # Two ticks: H then CX
        assert len(circuit) == 2
        tick0 = tick_to_dict(circuit[0])
        tick1 = tick_to_dict(circuit[1])
        assert "H" in tick0
        assert "CX" in tick1

    def test_ghz_state_circuit(self) -> None:
        """GHZ state generates H tick followed by two CX ticks."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_quantum_circuit(ast)

        assert len(circuit) == 3
        tick0 = tick_to_dict(circuit[0])
        tick1 = tick_to_dict(circuit[1])
        tick2 = tick_to_dict(circuit[2])
        assert "H" in tick0
        assert "CX" in tick1
        assert (0, 1) in tick1["CX"]
        assert "CX" in tick2
        assert (1, 2) in tick2["CX"]

    def test_circuit_with_repeated_syndrome(self) -> None:
        """Repeated syndrome extraction unrolls to correct number of ticks."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            Repeat(cond=2).block(
                qb.CX(data[0], ancilla[0]),
                qb.CX(data[1], ancilla[0]),
                qb.Measure(ancilla[0]) > c[0],
                qb.PZ(ancilla[0]),
            ),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_quantum_circuit(ast)

        # 2 iterations * 4 operations = 8 ticks
        assert len(circuit) == 8


def _qasm_measurements(qasm):
    """Read ordered qubit/destination pairs directly from the QASM oracle."""
    return [
        (int(qubit), (register, int(bit)) if register else None)
        for qubit, register, bit in re.findall(r"measure q\[(\d+)\](?: -> (\w+)\[(\d+)\])?;", qasm)
    ]


def _qc_measurements(circuit):
    """Read ordered measurement groups and check that metadata is gate-local."""
    measurements = []
    for name, locations, params in circuit.items():
        if name != "Measure":
            assert "var_output" not in params
            continue
        destinations = params.get("var_output", {})
        if destinations:
            assert set(destinations) == locations
        measurements.extend((qubit, destinations.get(qubit)) for qubit in locations)
    return measurements


def _assert_measurements_match(actual, expected) -> None:
    """Stored results retain order; bare results retain every measured target."""
    assert [entry for entry in actual if entry[1] is not None] == [entry for entry in expected if entry[1] is not None]
    assert Counter(qubit for qubit, bit in actual if bit is None) == Counter(
        qubit for qubit, bit in expected if bit is None
    )


@pytest.fixture(
    params=[
        "out_of_order",
        "two_registers",
        "rewritten_bit",
        "bare",
        "short_results",
        "whole_register",
        "reversed_targets",
        "repeat",
        "static_for",
        "parallel",
        "permute",
        "duplicate_targets",
        "parallel_duplicate",
        "parallel_bare_duplicate",
        "parallel_bare_order",
        "parallel_gate_between",
    ],
)
def measurement_program(request):
    """Exercise SLR constructs with a QASM measurement oracle; optimizer on and off."""
    q, c, d = QReg("q", 4), CReg("c", 2), CReg("d", 2)
    if request.param == "whole_register":
        c = CReg("c", 4)
    pair = (qb.Measure(q[1]) > c[1], qb.Measure(q[0]) > c[0])
    bodies = {
        "out_of_order": pair,
        "two_registers": (qb.Measure(q[1]) > d[0], qb.Measure(q[0]) > c[1]),
        "rewritten_bit": (qb.Measure(q[0]) > c[0], qb.Measure(q[1]) > c[0]),
        "bare": (qb.Measure(q[1]), qb.Measure(q[0]) > c[0]),
        "short_results": (qb.Measure(q[1], q[0], q[2]) > c[1],),
        "whole_register": (qb.Measure(q) > c,),
        "reversed_targets": (qb.Measure(q[1], q[0]) > (c[1], c[0]),),
        "repeat": (Repeat(3).block(*pair),),
        "static_for": (For("i", 0, 3).Do(*pair),),
        "parallel": (Parallel(qb.X(q[2]), *pair, qb.Measure(q[3])),),
        "permute": (Permute([q[0], q[1]], [q[1], q[0]]), *pair),
        "duplicate_targets": (qb.Measure(q[0], q[0]) > (c[0], c[1]),),
        "parallel_duplicate": (Parallel(qb.Measure(q[0]) > c[0], qb.Measure(q[0]) > c[1]),),
        "parallel_bare_duplicate": (Parallel(qb.Measure(q[0]), qb.Measure(q[0]), qb.Measure(q[1]) > c[0]),),
        "parallel_bare_order": (Parallel(qb.Measure(q[1]), qb.Measure(q[0]), qb.Measure(q[2]) > c[0]),),
        "parallel_gate_between": (Parallel(qb.Measure(q[0]) > c[0], qb.X(q[0]), qb.Measure(q[0]) > c[1]),),
    }
    program = Main(q, c, d, *bodies[request.param])
    # QASM does not lower static For: use its explicit unrolling as the oracle.
    oracle = Main(q, c, d, *(pair * 3)) if request.param == "static_for" else program
    return program, oracle


@pytest.fixture(params=[False, True], ids=["unoptimized", "optimized"])
def optimize_parallel(request):
    """Both SlrConverter paths must keep every measurement destination."""
    return request.param


def test_measurement_results_match_qasm(measurement_program, optimize_parallel) -> None:
    """No result destination or repeated measurement disappears during codegen."""
    program, oracle = measurement_program
    qasm = SlrConverter(oracle, optimize_parallel=optimize_parallel).qasm()
    circuit = SlrConverter(program, optimize_parallel=optimize_parallel).quantum_circuit()
    _assert_measurements_match(_qc_measurements(circuit), _qasm_measurements(qasm))


def test_measurement_results_reach_phir(measurement_program, optimize_parallel) -> None:
    """The downstream reader accepts every emitted result group, including bare ones."""
    program, oracle = measurement_program
    qasm = SlrConverter(oracle, optimize_parallel=optimize_parallel).qasm()
    circuit = SlrConverter(program, optimize_parallel=optimize_parallel).quantum_circuit()
    # PHIR otherwise assigns new qubit indices in first-use order.
    circuit.metadata["qvars"] = {qubit: ("q", qubit) for qubit in circuit.qudits}
    operations = json.loads(circuit.to_phir_json())["ops"]
    # PHIR requires returns, so a bare measurement writes to an internal scratch variable (#1497).
    user_registers = set(circuit.metadata.get("cvar_spec", {}))
    measurements = []
    for op in operations:
        if op.get("qop") == "Measure":
            assert "var_output" not in op.get("metadata", {})
            measurements.extend(
                (qubit[1], tuple(bit) if bit[0] in user_registers else None)
                for qubit, bit in zip(op["args"], op["returns"], strict=True)
            )
    _assert_measurements_match(measurements, _qasm_measurements(qasm))


def test_parallel_measurement_params_are_local() -> None:
    """Disjoint gates share a tick while keeping each measurement's own metadata."""
    q, c, d = QReg("q", 4), CReg("c", 1), CReg("d", 1)
    program = Main(q, c, d, Parallel(qb.X(q[2]), qb.Measure(q[1]) > c[0], qb.Measure(q[0]) > d[0], qb.Measure(q[3])))
    converter = SlrConverter(program, optimize_parallel=False)
    circuit = converter.quantum_circuit()
    assert len(circuit) == 1
    assert _qc_measurements(circuit) == _qasm_measurements(converter.qasm())


def test_measurements_flush_conflicting_gates_in_order() -> None:
    """A gate between two measurements cannot move before the first measurement."""
    q, c = QReg("q", 1), CReg("c", 2)
    program = Main(q, c, Parallel(qb.Measure(q[0]) > c[0], qb.X(q[0]), qb.Measure(q[0]) > c[1]))
    converter = SlrConverter(program, optimize_parallel=False)
    circuit = converter.quantum_circuit()
    assert [name for name, _locations, _params in circuit.items()] == ["Measure", "X", "Measure"]
    assert len(circuit) == 3
    assert _qc_measurements(circuit) == _qasm_measurements(converter.qasm())


@pytest.mark.parametrize("operation", ["assignment", "bit_permute", "mixed_permute"])
def test_classical_effects_are_rejected(operation) -> None:
    """Unsupported classical effects must not silently change the final results."""
    q, c = QReg("q", 2), CReg("c", 2)
    operations = {
        "assignment": (c[1].set(c[0]), "Assign.*QASM"),
        "bit_permute": (Permute([c[0], c[1]], [c[1], c[0]]), "Permute.*c\\[0\\].*QASM"),
        "mixed_permute": (Permute([q[0], c[0]], [c[0], q[0]]), "Permute.*c\\[0\\].*QASM"),
    }
    op, message = operations[operation]
    program = Main(q, c, qb.X(q[1]), qb.Measure(q[0]) > c[0], qb.Measure(q[1]) > c[1], op)
    with pytest.raises(NotImplementedError, match=message):
        SlrConverter(program).quantum_circuit()


def test_unresolved_block_call_is_rejected() -> None:
    """A direct AST block call cannot disappear when no declaration handles it."""
    from pecos.slr.ast.nodes import AllocatorDecl, BlockCall, Program

    op = BlockCall(callee="missing", arg_bindings=())
    program = Program(name="unlowered", declarations=(AllocatorDecl(name="q", capacity=2),), body=(op,))
    with pytest.raises(ValueError, match=r"undefined block 'missing'; provide its block declaration"):
        ast_to_quantum_circuit(program)


def test_bare_register_measurement_preserves_depth(optimize_parallel) -> None:
    """A whole-register bare measurement occupies one tick."""
    q = QReg("q", 100)
    circuit = SlrConverter(Main(q, qb.H(q), qb.Measure(q)), optimize_parallel=optimize_parallel).quantum_circuit()
    measurement_ticks = [tick_to_dict(circuit[i]) for i in range(len(circuit)) if "Measure" in tick_to_dict(circuit[i])]
    assert measurement_ticks == [{"Measure": set(range(100))}]


def test_parallel_bare_measurements_preserve_depth() -> None:
    """Explicit Parallel keeps disjoint bare measurements in a single tick."""
    q = QReg("q", 2)
    program = Main(q, Parallel(qb.Measure(q[0]), qb.Measure(q[1])))
    circuit = SlrConverter(program, optimize_parallel=False).quantum_circuit()
    assert len(circuit) == 1
    assert tick_to_dict(circuit[0]) == {"Measure": {0, 1}}


def test_nested_parallel_preserves_depth() -> None:
    """A nested Parallel must not end the surrounding Parallel block."""
    q, r = QReg("q", 2), QReg("r", 2)
    program = Main(q, r, Parallel(Parallel(qb.H(q[0]), qb.H(q[1])), qb.X(r[0]), qb.Z(r[1])))
    circuit = SlrConverter(program, optimize_parallel=False).quantum_circuit()
    assert len(circuit) == 1
    assert tick_to_dict(circuit[0]) == {"H": {0, 1}, "X": {2}, "Z": {3}}


def test_repeated_parallel_targets_stay_sequential() -> None:
    """Repeated gates in Parallel must not disappear into a set."""
    q = QReg("q", 1)
    circuit = SlrConverter(Main(q, Parallel(qb.X(q[0]), qb.X(q[0]))), optimize_parallel=False).quantum_circuit()
    assert len(circuit) == 2
    assert [tick_to_dict(circuit[i]) for i in range(len(circuit))] == [{"X": {0}}, {"X": {0}}]


def test_aliased_parallel_targets_stay_sequential() -> None:
    """Logical targets sharing a physical qubit must not merge in a tick."""
    from pecos.slr.ast.nodes import AllocatorDecl, GateKind, GateOp, ParallelBlock, Program, SlotRef

    program = Program(
        name="aliased_parallel",
        declarations=(AllocatorDecl(name="q", capacity=2), AllocatorDecl(name="a", capacity=1, parent="q")),
        body=(
            ParallelBlock(
                body=tuple(GateOp(gate=GateKind.H, targets=(SlotRef(allocator=name, index=0),)) for name in ("q", "a")),
            ),
        ),
    )
    circuit = ast_to_quantum_circuit(program)
    assert len(circuit) == 2
    assert [tick_to_dict(circuit[i]) for i in range(len(circuit))] == [{"H": {0}}, {"H": {0}}]


def test_measurement_registers_execute_in_phir() -> None:
    """Generated PHIR declares every CReg and executes the QASM destinations."""
    from pecos.engines.hybrid_engine import HybridEngine

    q, c, d, unused = QReg("q", 4), CReg("c", 3), CReg("d", 2), CReg("unused", 4)
    program = Main(
        q,
        c,
        d,
        unused,
        qb.X(q[1]),
        qb.X(q[3]),
        qb.Measure(q[1]) > d[1],
        qb.Measure(q[0]) > c[1],
        qb.Measure(q[3]) > c[0],
        qb.Measure(q[2]) > d[0],
    )
    converter = SlrConverter(program)
    expected = {"c": 0, "d": 0, "unused": 0}
    for qubit, (register, bit) in _qasm_measurements(converter.qasm()):
        expected[register] = (expected[register] & ~(1 << bit)) | (int(qubit in {1, 3}) << bit)
    assert expected == {"c": 1, "d": 2, "unused": 0}

    circuit = converter.quantum_circuit()
    phir = json.loads(circuit.to_phir_json())
    results = HybridEngine().run(phir, shots=3, return_int=True)
    for register, value in expected.items():
        assert results[register] == [value] * 3
    assert circuit.metadata["cvar_spec"] == {"c": 3, "d": 2, "unused": 4}
    declarations = {op["variable"]: op["size"] for op in phir["ops"] if op.get("data") == "cvar_define"}
    assert declarations == circuit.metadata["cvar_spec"]


@pytest.mark.parametrize("empty", [False, True])
def test_codegen_declares_empty_classical_spec(*, empty: bool) -> None:
    program = Main() if empty else Main(q := QReg("q", 1), qb.X(q[0]))
    assert SlrConverter(program).quantum_circuit().metadata["cvar_spec"] == {}


@pytest.mark.parametrize("semantics", ["default", "unsigned"])
def test_empty_classical_spec_reader_compatibility(semantics: str) -> None:
    """Explicitly empty metadata and omitted metadata initialize identical storage."""
    from types import SimpleNamespace

    from pecos.engines.cvm.semantics import DefaultClassicalSemantics, UnsignedClassicalSemantics

    reader = DefaultClassicalSemantics() if semantics == "default" else UnsignedClassicalSemantics()
    state = SimpleNamespace(num_qubits=2)
    omitted = reader.set_output(state, QuantumCircuit(), None, None)
    explicit = reader.set_output(state, QuantumCircuit(cvar_spec={}), None, None)
    assert explicit.keys() == omitted.keys() == {"__pecos_scratch"}
    assert int(explicit["__pecos_scratch"]) == int(omitted["__pecos_scratch"]) == 0


def test_return_preserves_measurement_results() -> None:
    """Return declares outputs without adding circuit operations."""
    q, c = QReg("q", 1), CReg("c", 1)
    program = Main(q, c, qb.X(q[0]), qb.Measure(q[0]) > c[0])
    expected = SlrConverter(program).quantum_circuit()
    program.extend(Return(c))
    circuit = SlrConverter(program).quantum_circuit()
    assert list(circuit.items()) == list(expected.items())
    assert circuit.metadata == expected.metadata
    assert _qc_measurements(circuit) == [(0, ("c", 0))]


@pytest.mark.parametrize("basis", ["PZ", "PNZ", "PX", "PNX", "PY", "PNY"])
def test_allocator_wide_prepare_matches_explicit_slots(basis) -> None:
    """Allocator-wide preparation resets all slots and applies the basis tail."""
    from dataclasses import replace

    from pecos.slr.ast.nodes import PrepareOp

    q, c = QReg("q", 3), CReg("c", 3)
    program = Main(q, c, qb.X(q), getattr(qb, basis)(q), qb.Measure(q) > c)
    qasm = SlrConverter(program).qasm()
    # SLR expands PZ(q) into explicit slots; exercise the allocator-wide AST form.
    ast = slr_to_ast(program)
    explicit = ast_to_quantum_circuit(ast)
    ast = replace(ast, body=tuple(replace(op, slots=None) if isinstance(op, PrepareOp) else op for op in ast.body))
    circuit = ast_to_quantum_circuit(ast)
    assert list(circuit.items()) == list(explicit.items())
    resets = [qubit for name, locations, _params in circuit.items() if name == "RESET" for qubit in sorted(locations)]
    assert resets == [int(qubit) for qubit in re.findall(r"reset q\[(\d+)\];", qasm)] == [0, 1, 2]
    assert _qc_measurements(circuit) == _qasm_measurements(qasm)


def test_allocator_wide_pz_executes_in_phir() -> None:
    """Preparing the whole allocator after X resets all three qubits to zero."""
    from dataclasses import replace

    from pecos.engines.hybrid_engine import HybridEngine
    from pecos.slr.ast.nodes import PrepareOp

    q, c = QReg("q", 3), CReg("c", 3)
    program = Main(q, c, qb.X(q), qb.PZ(q), qb.Measure(q) > c)
    qasm = SlrConverter(program).qasm()
    ast = slr_to_ast(program)
    ast = replace(ast, body=tuple(replace(op, slots=None) if isinstance(op, PrepareOp) else op for op in ast.body))
    circuit = ast_to_quantum_circuit(ast)
    resets = [qubit for name, locations, _params in circuit.items() if name == "RESET" for qubit in sorted(locations)]
    assert resets == [int(qubit) for qubit in re.findall(r"reset q\[(\d+)\];", qasm)] == [0, 1, 2]
    assert _qc_measurements(circuit) == _qasm_measurements(qasm)
    results = HybridEngine().run(json.loads(circuit.to_phir_json()), shots=3, return_int=True)
    assert results["c"] == [0, 0, 0]


@pytest.mark.parametrize("shape", ["bells", "nested_then_gate", "nested_siblings"])
def test_parallel_conflicts_preserve_qasm_order(shape, optimize_parallel) -> None:
    """Nested Parallel preserves each qubit's operation and result order."""
    from collections import defaultdict

    from pecos.slr import Block

    q, c = QReg("q", 4), CReg("c", 4)

    def bell(a, b):
        return Block(Parallel(qb.H(a), qb.Z(b)), qb.CX(a, b))

    bodies = {
        "bells": Parallel(bell(q[0], q[1]), bell(q[2], q[3])),
        "nested_then_gate": Parallel(Parallel(qb.H(q[0])), qb.X(q[0])),
        "nested_siblings": Parallel(Parallel(qb.X(q[0])), Parallel(qb.Z(q[0]))),
    }
    program = Main(q, c, bodies[shape], qb.Measure(q[3], q[1], q[2], q[0]) > (c[0], c[2], c[1], c[3]))
    converter = SlrConverter(program, optimize_parallel=optimize_parallel)
    expected = defaultdict(list)
    for line in converter.qasm().splitlines():
        match = re.fullmatch(r"(h|x|z|cx|measure) (q\[.*);", line.strip())
        if match:
            name, arguments = match.groups()
            qubits = tuple(int(index) for index in re.findall(r"q\[(\d+)\]", arguments))
            result = re.search(r"-> (\w+)\[(\d+)\]", arguments)
            bit = (result[1], int(result[2])) if result else None
            for qubit in qubits:
                expected[qubit].append((name.lower(), qubits, bit))
    actual = defaultdict(list)
    circuit = converter.quantum_circuit()
    for name, locations, params in circuit.items():
        for location in locations:
            qubits = (location,) if isinstance(location, int) else location
            bit = params.get("var_output", {}).get(location)
            for qubit in qubits:
                actual[qubit].append((name.lower(), qubits, bit))
    assert dict(actual) == dict(expected)
    _assert_measurements_match(_qc_measurements(circuit), _qasm_measurements(converter.qasm()))


def test_named_qubits_execute_in_phir() -> None:
    """A classical q register must not collide with auto-named quantum storage."""
    from pecos.engines.hybrid_engine import HybridEngine

    data, other, bits = QReg("data", 2), QReg("other", 1), CReg("q", 2)
    program = Main(data, other, bits, qb.X(data[1]), qb.Measure(data[0]) > bits[1], qb.Measure(data[1]) > bits[0])
    converter = SlrConverter(program)
    qasm = converter.qasm()
    circuit = converter.quantum_circuit()
    phir = json.loads(circuit.to_phir_json())
    results = HybridEngine().run(phir, shots=3, return_int=True)
    assert results["q"] == [1, 1, 1]
    expected_registers = {name: int(size) for name, size in re.findall(r"qreg (\w+)\[(\d+)\];", qasm)}
    assert {op["variable"]: op["size"] for op in phir["ops"] if op.get("data") == "qvar_define"} == expected_registers
    expected_measurements = [
        ((name, int(index)), (register, int(bit)))
        for name, index, register, bit in re.findall(r"measure (\w+)\[(\d+)\] -> (\w+)\[(\d+)\];", qasm)
    ]
    actual = [
        (tuple(qubit), tuple(bit))
        for op in phir["ops"]
        if op.get("qop") == "Measure"
        for qubit, bit in zip(op["args"], op["returns"], strict=True)
    ]
    assert actual == expected_measurements
    assert circuit.metadata["qvars"] == {0: ("data", 0), 1: ("data", 1), 2: ("other", 0)}


def test_qubit_metadata_resolves_child_allocators() -> None:
    """Child references retain their root name and absolute index in PHIR."""
    from pecos.slr.ast.codegen import ast_to_qasm
    from pecos.slr.ast.nodes import AllocatorDecl, GateKind, GateOp, Program, SlotRef

    program = Program(
        name="children",
        declarations=(
            AllocatorDecl(name="data", capacity=4),
            AllocatorDecl(name="padding", capacity=1, parent="data"),
            AllocatorDecl(name="child", capacity=2, parent="data"),
        ),
        body=(GateOp(gate=GateKind.X, targets=(SlotRef(allocator="child", index=1),)),),
    )
    circuit = ast_to_quantum_circuit(program)
    phir = json.loads(circuit.to_phir_json())
    expected = [(name, int(index)) for name, index in re.findall(r"x (\w+)\[(\d+)\];", ast_to_qasm(program))]
    assert [tuple(qubit) for op in phir["ops"] if op.get("qop") == "X" for qubit in op["args"]] == expected
    assert circuit.metadata["qvars"] == {i: ("data", i) for i in range(4)}


@pytest.mark.parametrize("whole_register", [False, True])
def test_block_local_classical_permute_is_rejected(whole_register) -> None:
    """A local CReg without a declaration must never become a qubit relabel."""
    from pecos.slr import Block

    q = QReg("q", 2)

    class LocalBits(Block):
        def __init__(self) -> None:
            c = CReg("local", 2)
            permutation = Permute(c, c) if whole_register else Permute([c[0], c[1]], [c[1], c[0]])
            super().__init__(qb.Measure(q[0]) > c[0], permutation)

    ref = "local" if whole_register else r"local\[0\]"
    with pytest.raises(NotImplementedError, match=rf"Permute.*{ref}.*declare.*qubit"):
        SlrConverter(Main(q, LocalBits())).quantum_circuit()


def test_prepare_all_requires_declared_allocator() -> None:
    """An unknown allocator needs a declaration, not a bare dictionary error."""
    from pecos.slr.ast.nodes import PrepareOp, Program

    program = Program(name="missing_allocator", body=(PrepareOp(allocator="missing"),))
    with pytest.raises(NotImplementedError, match=r"prepare_all.*missing.*declare.*allocator"):
        ast_to_quantum_circuit(program)
