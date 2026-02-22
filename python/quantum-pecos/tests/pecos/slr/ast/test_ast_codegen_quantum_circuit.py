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

import pytest
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.slr import Barrier, CReg, If, Main, QReg, Repeat, While
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
    """Prep and measure code generation tests."""

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
        """Prep creates tick with RESET operation."""
        prog = Main(
            q := QReg("q", 1),
            qb.Prep(q[0]),
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
                qb.Prep(ancilla[0]),
            ),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_quantum_circuit(ast)

        # 2 iterations * 4 operations = 8 ticks
        assert len(circuit) == 8
