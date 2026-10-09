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

"""Tests for AST to Stim code generator."""

import pytest
import stim
from pecos.slr import Barrier, CReg, If, Main, Permute, QReg, Repeat
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import AstToStim, ast_to_stim, ast_to_stim_str
from pecos.slr.gen_codes import StimGenerator
from pecos.slr.qeclib import qubit as qb


class TestAstToStimBasic:
    """Basic code generation tests."""

    def test_empty_program(self) -> None:
        """Empty program generates empty Stim circuit."""
        prog = Main()
        ast = slr_to_ast(prog)

        circuit = ast_to_stim(ast)

        assert isinstance(circuit, stim.Circuit)
        assert len(circuit) == 0

    def test_program_with_qreg(self) -> None:
        """Program with QReg generates non-empty circuit."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        circuit = ast_to_stim(ast)

        assert isinstance(circuit, stim.Circuit)
        assert len(circuit) > 0

    def test_string_output(self) -> None:
        """String output function returns circuit as string."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert isinstance(code, str)
        assert "H" in code


class TestAstToStimGates:
    """Gate code generation tests."""

    def test_hadamard_gate(self) -> None:
        """Hadamard gate generates H instruction."""
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "H 0" in code

    def test_pauli_gates(self) -> None:
        """Pauli gates generate X, Y, Z instructions."""
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            qb.Y(q[0]),
            qb.Z(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "X 0" in code
        assert "Y 0" in code
        assert "Z 0" in code

    def test_phase_gates(self) -> None:
        """Phase gates generate S and S_DAG instructions."""
        prog = Main(
            q := QReg("q", 1),
            qb.SZ(q[0]),
            qb.SZdg(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "S 0" in code
        assert "S_DAG 0" in code

    def test_t_gates(self) -> None:
        """Stim rejects the non-Clifford T gate."""
        prog = Main(
            q := QReg("q", 1),
            qb.T(q[0]),
        )
        ast = slr_to_ast(prog)

        with pytest.raises(IndexError, match="Gate not found: 'T'"):
            ast_to_stim_str(ast)

    def test_two_qubit_cx_gate(self) -> None:
        """CX gate generates CX instruction with correct qubits."""
        prog = Main(
            q := QReg("q", 2),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "CX 0 1" in code

    def test_two_qubit_cz_gate(self) -> None:
        """CZ gate generates CZ instruction with correct qubits."""
        prog = Main(
            q := QReg("q", 2),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "CZ 0 1" in code

    def test_multiple_gates(self) -> None:
        """Multiple gates generate in sequence."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.X(q[1]),
            qb.CZ(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "H 0" in code
        assert "X 1" in code
        assert "CZ 0 1" in code


class TestAstToStimPrepMeasure:
    """PZ and measure code generation tests."""

    def test_measurement(self) -> None:
        """Measurement generates M instruction."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "M 0" in code

    def test_multiple_measurements(self) -> None:
        """Multiple measurements generate M instructions for all qubits."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        # Stim may combine measurements: "M 0 1" or "M 0\nM 1"
        assert "M" in code
        # Check both qubits are measured (format may vary)
        assert "0" in code
        assert "1" in code

    def test_prep_reset(self) -> None:
        """PZ generates R (reset) instruction."""
        prog = Main(
            q := QReg("q", 1),
            qb.PZ(q[0]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "R 0" in code


class TestAstToStimControlFlow:
    """Control flow code generation tests."""

    def test_barrier_becomes_tick(self) -> None:
        """Barrier generates TICK instruction."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            Barrier(q),
            qb.CX(q[0], q[1]),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "TICK" in code

    def test_repeat_uses_repeat_block(self) -> None:
        """Repeat generates REPEAT block."""
        prog = Main(
            q := QReg("q", 1),
            Repeat(cond=3).block(
                qb.H(q[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        # Stim uses REPEAT blocks
        assert "REPEAT 3" in code
        assert "H 0" in code


def _sample(prog: Main, shots: int = 256) -> list[list[int]]:
    """Sample the measurement record of the Stim circuit compiled from `prog`."""
    circuit = ast_to_stim(slr_to_ast(prog))
    return [[int(bit) for bit in row] for row in circuit.compile_sampler(seed=1234).sample(shots)]


class TestAstToStimConditionals:
    """Measurement-conditioned If lowering (issue #978).

    Each program measures a random bit and conditionally corrects a second
    qubit; Stim's own sampler checks that the correction fires exactly on
    the shots the condition selects.
    """

    @pytest.mark.parametrize(
        ("make_condition", "fires_when_one"),
        [
            (lambda c: c[0], True),
            (lambda c: c[0] == 1, True),
            (lambda c: c[0] != 0, True),
            (lambda c: c[0] == 0, False),
            (lambda c: c[0] != 1, False),
        ],
    )
    @pytest.mark.parametrize("pauli", [qb.X, qb.Y])
    def test_bit_flip_correction_fires_only_when_selected(self, make_condition, fires_when_one, pauli) -> None:
        """An X or Y correction flips the target exactly when the condition holds."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            If(make_condition(c)).Then(pauli(q[1])),
            qb.Measure(q[1]) > c[1],
        )

        rows = _sample(prog)

        assert {row[0] for row in rows} == {0, 1}, "the conditioning bit must vary"
        for m_condition, m_target in rows:
            assert m_target == (m_condition if fires_when_one else 1 - m_condition)

    def test_phase_flip_correction_fires_only_when_selected(self) -> None:
        """A Z correction on |+> is observed as a flip in the X basis."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            qb.H(q[1]),
            If(c[0] == 1).Then(qb.Z(q[1])),
            qb.H(q[1]),
            qb.Measure(q[1]) > c[1],
        )

        rows = _sample(prog)

        assert all(m_target == m_condition for m_condition, m_target in rows)

    def test_condition_reads_the_most_recent_measurement_of_the_bit(self) -> None:
        """Re-measuring into a bit retargets the condition to the newer record."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.Measure(q[0]) > c[0],
            qb.H(q[1]),
            qb.Measure(q[1]) > c[0],
            If(c[0] == 1).Then(qb.X(q[2])),
            qb.Measure(q[2]) > c[1],
        )

        rows = _sample(prog)

        assert all(row[2] == row[1] for row in rows)

    def test_condition_inside_repeat_reads_the_current_iteration(self) -> None:
        """A condition on a bit measured earlier in the same REPEAT body."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Repeat(cond=3).block(
                qb.PZ(q[0]),
                qb.PZ(q[1]),
                qb.H(q[0]),
                qb.Measure(q[0]) > c[0],
                If(c[0] == 1).Then(qb.X(q[1])),
                qb.Measure(q[1]) > c[1],
            ),
        )

        rows = _sample(prog)

        assert all(len(row) == 6 and row[1::2] == row[0::2] for row in rows)

    def test_condition_after_repeat_reads_the_last_iteration(self) -> None:
        """A bit written inside a REPEAT holds its last iteration's measurement."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            Repeat(cond=3).block(
                qb.PZ(q[0]),
                qb.H(q[0]),
                qb.Measure(q[0]) > c[0],
                qb.Measure(q[1]) > c[1],
            ),
            If(c[0] == 1).Then(qb.X(q[2])),
            qb.Measure(q[2]) > c[2],
        )

        rows = _sample(prog)

        assert all(len(row) == 7 and row[6] == row[4] for row in rows)

    def test_condition_reads_an_older_record(self) -> None:
        """Later measurements push the conditioning record past rec[-1]."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.H(q[0]),
            qb.H(q[1]),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            If(c[0] == 1).Then(qb.X(q[2])),
            qb.Measure(q[2]) > c[2],
        )

        rows = _sample(prog)

        assert any(row[0] != row[1] for row in rows), "the two records must differ on some shots"
        assert all(row[2] == row[0] for row in rows)

    def test_condition_inside_repeat_reads_an_older_record(self) -> None:
        """A lookback past rec[-1] inside a REPEAT body."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            Repeat(cond=2).block(
                qb.PZ(q[0]),
                qb.PZ(q[1]),
                qb.PZ(q[2]),
                qb.H(q[0]),
                qb.H(q[1]),
                qb.Measure(q[0], q[1]) > (c[0], c[1]),
                If(c[0] == 1).Then(qb.X(q[2])),
                qb.Measure(q[2]) > c[2],
            ),
        )

        rows = _sample(prog)

        assert any(row[0] != row[1] for row in rows), "the two records must differ on some shots"
        assert all(len(row) == 6 and row[2] == row[0] and row[5] == row[3] for row in rows)

    def test_permuted_bit_is_rejected(self) -> None:
        """A permuted classical bit no longer has a tracked measurement record."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 2),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            Permute([c[0], c[1]], [c[1], c[0]]),
            If(c[0] == 1).Then(qb.X(q[2])),
        )

        with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
            ast_to_stim(slr_to_ast(prog))

    def test_bit_permuted_inside_repeat_is_rejected_after_it(self) -> None:
        """A Permute inside a REPEAT invalidates the record taken before it."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 2),
            qb.Measure(q[0]) > c[0],
            Repeat(cond=2).block(
                Permute([c[0], c[1]], [c[1], c[0]]),
            ),
            If(c[0] == 1).Then(qb.X(q[2])),
        )

        with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
            ast_to_stim(slr_to_ast(prog))

    def test_qubit_permute_still_applies_when_a_creg_shares_the_name(self) -> None:
        """SLR allows a QReg and a CReg with one name; the qubit swap must still happen."""
        q = QReg("same", 2)
        c = CReg("same", 2)
        prog = Main(
            q,
            c,
            out := CReg("out", 2),
            qb.X(q[0]),
            Permute([q[0], q[1]], [q[1], q[0]]),
            qb.Measure(q) > out,
        )

        rows = _sample(prog, shots=4)

        assert all(row == [0, 1] for row in rows)

    def test_else_body_is_rejected(self) -> None:
        """Stim has no else branch for a record-controlled Pauli."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(qb.X(q[1])).Else(qb.Z(q[1])),
        )

        with pytest.raises(NotImplementedError, match="else-body"):
            ast_to_stim(slr_to_ast(prog))

    def test_non_pauli_body_is_rejected(self) -> None:
        """A conditional Hadamard cannot be expressed as a record-controlled Pauli."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(qb.H(q[1])),
        )

        with pytest.raises(NotImplementedError, match="If body contains H"):
            ast_to_stim(slr_to_ast(prog))

    def test_register_condition_is_rejected(self) -> None:
        """A whole-register comparison is not a single record target."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.Measure(q[0]) > c[0],
            If(c == 1).Then(qb.X(q[1])),
        )

        with pytest.raises(NotImplementedError, match="unsupported If condition"):
            ast_to_stim(slr_to_ast(prog))

    def test_unmeasured_bit_is_rejected(self) -> None:
        """A bit that never received a measurement has no record target."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            If(c[0] == 1).Then(qb.X(q[0])),
        )

        with pytest.raises(NotImplementedError, match="holds no measurement result"):
            ast_to_stim(slr_to_ast(prog))

    def test_bit_measured_before_repeat_is_rejected_inside_it(self) -> None:
        """The record offset of an outer measurement changes every iteration."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.Measure(q[0]) > c[0],
            Repeat(cond=2).block(
                If(c[0] == 1).Then(qb.X(q[1])),
                qb.Measure(q[1]) > c[1],
            ),
        )

        with pytest.raises(NotImplementedError, match="holds no measurement result"):
            ast_to_stim(slr_to_ast(prog))

    def test_assigned_undeclared_register_is_rejected(self) -> None:
        """Whole-register assignment invalidates bits even without a register declaration."""
        q = QReg("q", 2)
        c = CReg("inline", 1)
        prog = Main(
            q,
            qb.Measure(q[0]) > c[0],
            c.set(1),
            If(c[0] == 1).Then(qb.X(q[1])),
        )

        with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
            ast_to_stim(slr_to_ast(prog))

    def test_assigned_bit_is_rejected(self) -> None:
        """A classical assignment replaces the measured value Stim could condition on."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
            c[0].set(1),
            If(c[0] == 1).Then(qb.X(q[1])),
        )

        with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
            ast_to_stim(slr_to_ast(prog))

    def test_register_assigned_inside_repeat_is_rejected_after_it(self) -> None:
        """An assignment inside a REPEAT invalidates the outer measurement record."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
            Repeat(cond=2).block(
                c.set(0),
            ),
            If(c[0] == 1).Then(qb.X(q[1])),
        )

        with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
            ast_to_stim(slr_to_ast(prog))

    def test_legacy_generator_rejects_if(self) -> None:
        """The deprecated StimGenerator fails loud instead of dropping the condition."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.Measure(q[0]) > c[0],
            If(c[0] == 1).Then(qb.X(q[1])),
        )

        with pytest.raises(NotImplementedError, match="StimGenerator does not support If"):
            StimGenerator(_internal=True).generate_block(prog)


class TestAstToStimQEC:
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

        code = ast_to_stim_str(ast)

        # data[0] -> qubit 0, data[1] -> qubit 1, ancilla[0] -> qubit 2
        # Stim may combine CX gates into one line: "CX 0 2 1 2"
        assert "CX" in code
        assert "0 2" in code or "0, 2" in code  # First CX pair
        assert "1 2" in code or "1, 2" in code  # Second CX pair
        assert "M 2" in code

    def test_repeated_syndrome_extraction(self) -> None:
        """Repeated syndrome extraction generates REPEAT block."""
        prog = Main(
            data := QReg("data", 2),
            ancilla := QReg("ancilla", 1),
            c := CReg("c", 1),
            Repeat(cond=3).block(
                qb.CX(data[0], ancilla[0]),
                qb.CX(data[1], ancilla[0]),
                qb.Measure(ancilla[0]) > c[0],
                qb.PZ(ancilla[0]),
            ),
        )
        ast = slr_to_ast(prog)

        code = ast_to_stim_str(ast)

        assert "REPEAT 3" in code


class TestAstToStimGenerator:
    """Tests for AstToStim generator class."""

    def test_generator_reusable(self) -> None:
        """Generator can be reused for multiple programs."""
        generator = AstToStim()

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

        code1 = str(circuit1)
        code2 = str(circuit2)

        assert "H 0" in code1
        assert "X 0" in code2

    def test_measurement_count_tracked(self) -> None:
        """Generator tracks measurement count."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
            qb.Measure(q[2]) > c[2],
        )
        ast = slr_to_ast(prog)

        generator = AstToStim()
        generator.generate(ast)

        assert generator.context.measurement_count == 3


class TestAstToStimFullPipeline:
    """End-to-end tests: SLR -> AST -> Stim."""

    def test_bell_state_circuit(self) -> None:
        """Bell state generates H and CX instructions."""
        prog = Main(
            q := QReg("q", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_stim(ast)

        # Verify circuit structure
        code = str(circuit)
        assert "H 0" in code
        assert "CX 0 1" in code

    def test_circuit_is_valid_stim(self) -> None:
        """Test that generated circuit can be used by Stim."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.Measure(q[0]) > c[0],
            qb.Measure(q[1]) > c[1],
        )

        ast = slr_to_ast(prog)
        circuit = ast_to_stim(ast)

        # Verify Stim can sample from the circuit
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=10)

        assert samples.shape == (10, 2)

    def test_ghz_state_circuit(self) -> None:
        """Test a GHZ state circuit."""
        prog = Main(
            q := QReg("q", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
        )

        ast = slr_to_ast(prog)
        code = ast_to_stim_str(ast)

        assert "H 0" in code
        # Stim may combine CX gates: "CX 0 1 1 2" or separate lines
        assert "CX" in code
        assert "0 1" in code or "0, 1" in code
        assert "1 2" in code or "1, 2" in code
