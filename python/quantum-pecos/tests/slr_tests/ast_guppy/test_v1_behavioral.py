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

"""v1 behavioral tests for the AST -> Guppy emitter via Selene.

Compile-only tests in `test_v1_acceptance.py` prove linearity and
HUGR construction. Behavioral tests prove that observable outcomes
match SLR intent. Wrong CReg ordering, wrong Permute mapping,
swapped reset/discard semantics all type-check; only Selene
execution catches them.

Test classes per stage 4 plan (`step4-cutover-plan.md`):

- Deterministic: 1-shot exact-match assertions
- Bell/GHZ correlation: ~100 shots, exact correlation every shot
- Marginal frequency: ~1000 shots, fixed seed, broad bounds
"""

from __future__ import annotations

from pecos.slr import CReg, If, Main, Permute, QReg, Return
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from ._selene_harness import run_ast_guppy_via_selene  # noqa: TID252

# ── Deterministic tests ──────────────────────────────────────────────────


class TestDeterministic:
    """Programs with deterministic measurement outcomes."""

    def test_x_then_measure_is_one(self) -> None:
        """`X(q[0]); Measure(q[0]) > c[0]` always measures 1."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)

    def test_no_op_then_measure_is_zero(self) -> None:
        """Fresh qubit measured without gates is always 0."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 0 for r in records)

    def test_x_then_x_then_measure_is_zero(self) -> None:
        """X is its own inverse."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 0 for r in records)

    def test_measure_prep_remeasure_is_zero(self) -> None:
        """Prep after measurement resets the slot to |0>."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            qb.Prep(q[0]),
            Measure(q[0]) > c[1],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)
        assert all(r["measurement_1"] == 0 for r in records)

    def test_measure_prep_x_remeasure_is_one(self) -> None:
        """Prep after measurement produces a fresh |0> that can be inverted."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            qb.Prep(q[0]),
            qb.X(q[0]),
            Measure(q[0]) > c[1],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)
        assert all(r["measurement_1"] == 1 for r in records)

    def test_h_z_h_then_measure_is_one(self) -> None:
        """HZH is equivalent to X, covering deterministic Z-basis behavior."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.Z(q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)

    def test_quantum_permute_is_observed_by_later_measurements(self) -> None:
        """Qubit slot permutation must remap the owned local, not just typecheck."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            Permute([q[0], q[1]], [q[1], q[0]]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 0 for r in records)
        assert all(r["measurement_1"] == 1 for r in records)

    def test_quantum_permute_three_cycle(self) -> None:
        """3-cycle Permute (q0, q1, q2) -> (q2, q0, q1).

        State before permute: |1>|0>|0> (X on q[0]). Permute moves the X
        excitation from slot 0 to slot 1 in the post-permute view, so a
        per-slot measurement should read out (0, 1, 0).
        """
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.X(q[0]),
            Permute([q[0], q[1], q[2]], [q[2], q[0], q[1]]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Measure(q[2]) > c[2],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 0 for r in records)
        assert all(r["measurement_1"] == 1 for r in records)
        assert all(r["measurement_2"] == 0 for r in records)

    def test_quantum_permute_cross_register(self) -> None:
        """Permute spanning two QRegs must remap slots across owned-local groups."""
        prog = Main(
            a := QReg("a", 2),
            b := QReg("b", 2),
            c := CReg("c", 4),
            qb.X(a[0]),
            qb.X(b[1]),
            Permute([a[0], a[1], b[0], b[1]], [b[1], b[0], a[1], a[0]]),
            Measure(a[0]) > c[0],
            Measure(a[1]) > c[1],
            Measure(b[0]) > c[2],
            Measure(b[1]) > c[3],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)
        assert all(r["measurement_1"] == 0 for r in records)
        assert all(r["measurement_2"] == 0 for r in records)
        assert all(r["measurement_3"] == 1 for r in records)


# ── Bell / GHZ correlation tests ──────────────────────────────────────────


class TestBellGHZ:
    """Entangled-state correlation tests; correlation is the strong signal."""

    def test_bell_correlation_every_shot(self) -> None:
        """Bell state: m_0 == m_1 in every shot."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            Measure(q) > c,
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=100)
        assert all(r["measurement_0"] == r["measurement_1"] for r in records)

    def test_ghz_three_correlation_every_shot(self) -> None:
        """GHZ state: m_0 == m_1 == m_2 in every shot."""
        prog = Main(
            q := QReg("q", 3),
            c := CReg("c", 3),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            qb.CX(q[1], q[2]),
            Measure(q) > c,
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=100)
        for r in records:
            assert r["measurement_0"] == r["measurement_1"] == r["measurement_2"]


# ── Marginal frequency tests ──────────────────────────────────────────────


class TestMarginalFrequency:
    """Statistical tests with fixed seed and broad tolerances."""

    def test_bell_marginal_frequency_in_range(self) -> None:
        """Each Bell qubit measures 0/1 roughly 50/50 over 1000 shots."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            Measure(q) > c,
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=1000, seed=42)
        ones_0 = sum(r["measurement_0"] for r in records)
        # Broad bound: 350-650 out of 1000. Catches gross emitter errors
        # that would skew the marginal (e.g., wrong gate emission) without
        # flaking on legitimate stochastic variation.
        assert 350 <= ones_0 <= 650, f"Bell m_0 ones={ones_0}/1000 outside 350-650 band"


# ── Conditional correctness ───────────────────────────────────────────────


class TestConditionalCorrectness:
    """Verify If/Then routes the conditional gate through correct slot."""

    def test_conditional_x_flips_remapped_branch(self) -> None:
        """Measure(q[0]) > c[0]; If(c[0]).Then(X(q[1])); Measure(q[1]) > c[1].

        - When q[0] starts |0> -> c[0]=0, branch skipped, c[1]=0.
        - When q[0] starts |1> -> c[0]=1, branch fires, c[1]=1.
        Verify the c[1] outcome matches c[0].
        """
        # Case 1: q[0] starts |0>
        prog_zero = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Measure(q[0]) > c[0],
            If(c[0]).Then(qb.X(q[1])),
            Measure(q[1]) > c[1],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog_zero, shots=10)
        assert all(r["measurement_0"] == 0 for r in records)
        assert all(r["measurement_1"] == 0 for r in records)

        # Case 2: q[0] flipped to |1> first
        prog_one = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(qb.X(q[1])),
            Measure(q[1]) > c[1],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog_one, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)
        assert all(r["measurement_1"] == 1 for r in records)

    def test_creg_permute_remaps_condition_bit(self) -> None:
        """CReg Permute must affect a later If condition."""
        prog_without_permute = Main(
            q := QReg("q", 1),
            flag := CReg("flag", 2, result=False),
            out := CReg("out", 1),
            flag[0].set(1),
            flag[1].set(0),
            If(flag[1]).Then(qb.X(q[0])),
            Measure(q[0]) > out[0],
            Return(out),
        )
        records = run_ast_guppy_via_selene(prog_without_permute, shots=10)
        assert all(r["measurement_0"] == 0 for r in records)

        prog_with_permute = Main(
            q := QReg("q", 1),
            flag := CReg("flag", 2, result=False),
            out := CReg("out", 1),
            flag[0].set(1),
            flag[1].set(0),
            Permute([flag[0], flag[1]], [flag[1], flag[0]]),
            If(flag[1]).Then(qb.X(q[0])),
            Measure(q[0]) > out[0],
            Return(out),
        )
        records = run_ast_guppy_via_selene(prog_with_permute, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)
