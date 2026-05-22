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

import math

from pecos.slr import CReg, If, Main, Permute, QReg, Return, rad
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure
from pecos.slr.qeclib.steane.steane_class import Steane

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
        """PZ after measurement resets the slot to |0>."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            qb.PZ(q[0]),
            Measure(q[0]) > c[1],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 1 for r in records)
        assert all(r["measurement_1"] == 0 for r in records)

    def test_measure_prep_x_remeasure_is_one(self) -> None:
        """PZ after measurement produces a fresh |0> that can be inverted."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            qb.PZ(q[0]),
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
            flag := CReg("flag", 2),
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
            flag := CReg("flag", 2),
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


class TestS5SteanePzBehavioral:
    """Steane pz() prepares a valid logical |0>.

    `Main(c := Steane("c"), c.pz())` has no user Return -> main() -> None;
    the flattened PrepRUS block-boundary Returns are elided at convert time.
    The companion adds an explicit Measure(c.d)+Return so Selene
    yields the 7-bit data record; every shot must satisfy the Steane Z-check
    syndrome == (0, 0, 0) (codespace membership = a real logical-|0>
    lock-in, not "deterministic bits").

    Previously a strict xfail under a wrong diagnosis ("v1 FT-RUS pz()
    non-codeword"). Re-diagnosed: pz() is correct (stim-verified); the
    failure was `_selene_harness` mis-mapping `measurement_N` because the
    Steane RUS performs an internal (non-returned) verify measurement. With
    named return-tag mapping the harness reads the returned data CReg
    by name, so this now passes.
    """

    # Steane Z-stabilizer check supports (steane_class.py: check_indices).
    _CHECKS = ((2, 1, 3, 0), (5, 2, 1, 4), (6, 5, 2, 3))

    def test_steane_pz_prepares_logical_zero(self) -> None:
        prog = Main(
            c := Steane("c"),
            c.pz(),
            m := CReg("m", 7),
            Measure(c.d) > m,
            Return(m),
        )
        records = run_ast_guppy_via_selene(prog, shots=4, seed=42)
        for rec in records:
            bits = [rec[f"measurement_{i}"] for i in range(7)]
            syndrome = tuple(bits[a] ^ bits[b] ^ bits[c] ^ bits[d] for (a, b, c, d) in self._CHECKS)
            assert syndrome == (0, 0, 0), (rec, syndrome)


class TestHarnessInternalMeasurementMapping:
    """Regression: an internal (non-returned) measurement before the
    returned data must not shift the public `measurement_N` mapping.

    `internal` is measured (q[0], forced to 1) but NOT returned; `out`
    (q[1], stays 0) IS returned. Previously the harness counted positionally
    and read `measurement_0` = the internal measurement (=1) -> wrong.
    With named return tags it reads `out` by name -> 0 every shot.
    """

    def test_internal_measurement_does_not_shift_mapping(self) -> None:
        prog = Main(
            q := QReg("q", 2),
            internal := CReg("internal", 1),
            out := CReg("out", 1),
            qb.X(q[0]),
            Measure(q[0]) > internal[0],
            Measure(q[1]) > out[0],
            Return(out),
        )
        records = run_ast_guppy_via_selene(prog, shots=10)
        assert all(r["measurement_0"] == 0 for r in records), records


class TestNativeGuppyGates:
    """Cross-codegen Phase A: PECOS gates now mapped to native Guppy
    stdlib gates -- SX/SXdg -> v/vdg (sqrt-X), and the parameterized
    rotations RX/RY/RZ -> rx/ry/rz, CRZ -> crz (all `guppylang.std`).

    Each discriminator is deterministic and mutation-resistant:
    catches a no-op (mapping dropped), and -- where handedness
    matters -- a v<->vdg swap. Expected outcomes pre-computed against
    a numpy reference for the PECOS gate conventions.
    """

    def test_sx_squares_to_x(self) -> None:
        """SX; SX = X on |0> -> 1 (catches SX mapped to a no-op)."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.SX(q[0]),
            qb.SX(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 1 for r in records), records

    def test_sx_handedness(self) -> None:
        """`|0>; SX; SZ; H; M` -> 0 deterministically (numpy-verified).

        A v<->vdg swap gives 1; a no-op gives 50/50. This pins SX = v
        (the correct sqrt-X handedness), not vdg.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.SX(q[0]),
            qb.SZ(q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 0 for r in records), records

    def test_sxdg_handedness_and_inverse(self) -> None:
        """`|0>; SXdg; SZ; H; M` -> 1 (mirror of SX); and SX;SXdg = I."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.SXdg(q[0]),
            qb.SZ(q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 1 for r in records), records

        inverse = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.SX(q[0]),
            qb.SXdg(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        recs2 = run_ast_guppy_via_selene(inverse, shots=20)
        assert all(r["measurement_0"] == 0 for r in recs2), recs2

    def test_rx_pi_is_x(self) -> None:
        """RX(pi)|0> = X|0> = |1> (up to phase) -> 1. RX(pi/2)^2 = RX(pi)."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.RX(rad(math.pi), q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 1 for r in records), records

        # Angle propagation: RX(pi/2) twice == RX(pi).
        halves = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.RX(rad(math.pi / 2), q[0]),
            qb.RX(rad(math.pi / 2), q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        recs2 = run_ast_guppy_via_selene(halves, shots=20)
        assert all(r["measurement_0"] == 1 for r in recs2), recs2

    def test_ry_pi_is_y(self) -> None:
        """RY(pi)|0> flips to |1> (up to phase) -> 1."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.RY(rad(math.pi), q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 1 for r in records), records

    def test_rz_phase_via_hadamard_sandwich(self) -> None:
        """`|0>; H; RZ(pi); H; M` -> 1 (numpy-verified; HZH=X via RZ(pi)~Z).

        A no-op RZ gives H;H=I -> 0, so this catches a dropped RZ.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.RZ(rad(math.pi), q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 1 for r in records), records

    def test_crz_control_phase(self) -> None:
        """CRZ(pi) with control=|1>: target gets RZ(pi)~Z.

        `X q0; H q1; CRZ(pi); H q1; M q1` -> 1 (Z|+>=|->, H|->=|1>).
        A no-op CRZ gives H;H=I -> 0; this is the control-active pin.
        """
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.X(q[0]),
            qb.H(q[1]),
            qb.CRZ(rad(math.pi), q[0], q[1]),
            qb.H(q[1]),
            Measure(q[1]) > c[0],
            Return(c),
        )
        records = run_ast_guppy_via_selene(prog, shots=20)
        assert all(r["measurement_0"] == 1 for r in records), records


class TestDecomposedGuppyGates:
    """Cross-codegen Phase B: PECOS gates with no native single Guppy
    gate, lowered via `GUPPY_GATE_DECOMP` into Guppy-native gates
    (1q Cliffords; the qsystem `zz_phase` = RZZ for 2q sqrt-Paulis;
    native `crz` for CRX/CRY).

    Discriminators are deterministic, numpy-verified against the PECOS
    gate conventions, and mutation-resistant (catch no-op / wrong-gate /
    dagger-swap).
    """

    @staticmethod
    def _bits(prog: Main, shots: int = 20) -> list[dict[str, int]]:
        return run_ast_guppy_via_selene(prog, shots=shots)

    # ---- single-qubit Cliffords (SY/SYdg, F-family) ----

    def test_sy_squares_to_y_and_handedness(self) -> None:
        sq = Main(q := QReg("q", 1), c := CReg("c", 1), qb.SY(q[0]), qb.SY(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 1 for r in self._bits(sq)), "SY;SY must be Y -> 1"
        # Handedness: SY;H -> 0 (SYdg;H -> 1; no-op -> 50/50).
        hd = Main(q := QReg("q", 1), c := CReg("c", 1), qb.SY(q[0]), qb.H(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 0 for r in self._bits(hd)), "SY;H must be 0 (handedness)"

    def test_sydg_handedness_and_inverse(self) -> None:
        hd = Main(q := QReg("q", 1), c := CReg("c", 1), qb.SYdg(q[0]), qb.H(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 1 for r in self._bits(hd)), "SYdg;H must be 1"
        inv = Main(q := QReg("q", 1), c := CReg("c", 1), qb.SY(q[0]), qb.SYdg(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 0 for r in self._bits(inv)), "SY;SYdg must be I -> 0"

    def test_f_family(self) -> None:
        # F;H -> 0 (distinguishes F from F4 (50/50) and no-op).
        f = Main(q := QReg("q", 1), c := CReg("c", 1), qb.F(q[0]), qb.H(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 0 for r in self._bits(f)), "F;H must be 0"
        # Fdg;Sdg;H -> 0; and F;Fdg = I.
        fdg = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.Fdg(q[0]),
            qb.SZdg(q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        assert all(r["measurement_0"] == 0 for r in self._bits(fdg)), "Fdg;Sdg;H must be 0"
        finv = Main(q := QReg("q", 1), c := CReg("c", 1), qb.F(q[0]), qb.Fdg(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 0 for r in self._bits(finv)), "F;Fdg must be I -> 0"

    def test_f4_family(self) -> None:
        # F4;S;H -> 0 (deterministic; F4;Sdg;H -> 1).
        f4 = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.F4(q[0]),
            qb.SZ(q[0]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        assert all(r["measurement_0"] == 0 for r in self._bits(f4)), "F4;S;H must be 0"
        # F4dg;H -> 0; and F4;F4dg = I.
        f4dg = Main(q := QReg("q", 1), c := CReg("c", 1), qb.F4dg(q[0]), qb.H(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 0 for r in self._bits(f4dg)), "F4dg;H must be 0"
        f4inv = Main(q := QReg("q", 1), c := CReg("c", 1), qb.F4(q[0]), qb.F4dg(q[0]), Measure(q[0]) > c[0], Return(c))
        assert all(r["measurement_0"] == 0 for r in self._bits(f4inv)), "F4;F4dg must be I -> 0"

    # ---- two-qubit sqrt-Paulis via native zz_phase ----

    def test_rzz_native_zz_phase(self) -> None:
        # H q0; H q1; RZZ(pi); H q0; H q1 -> both measure 1 (numpy-verified).
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.H(q[1]),
            qb.RZZ(rad(math.pi), q[0], q[1]),
            qb.H(q[0]),
            qb.H(q[1]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Return(c),
        )
        recs = self._bits(prog)
        assert all(r["measurement_0"] == 1 and r["measurement_1"] == 1 for r in recs), recs

    def test_szz_squared_and_inverse(self) -> None:
        # SZZ^2 with q1=|0> acts as Z on q0: H q0; SZZ; SZZ; H q0 -> 1.
        sq = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.SZZ(q[0], q[1]),
            qb.SZZ(q[0], q[1]),
            qb.H(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        assert all(r["measurement_0"] == 1 for r in self._bits(sq)), "H;SZZ;SZZ;H must be X on q0 -> 1"
        # SZZ;SZZdg = I on |00>.
        inv = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.SZZ(q[0], q[1]),
            qb.SZZdg(q[0], q[1]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Return(c),
        )
        assert all(r["measurement_0"] == 0 and r["measurement_1"] == 0 for r in self._bits(inv)), "SZZ;SZZdg = I"

    def test_sxx_syy_inverse_pairs(self) -> None:
        for g, gdg, name in [(qb.SXX, qb.SXXdg, "SXX"), (qb.SYY, qb.SYYdg, "SYY")]:
            prog = Main(
                q := QReg("q", 2),
                c := CReg("c", 2),
                g(q[0], q[1]),
                gdg(q[0], q[1]),
                Measure(q[0]) > c[0],
                Measure(q[1]) > c[1],
                Return(c),
            )
            recs = self._bits(prog)
            assert all(r["measurement_0"] == 0 and r["measurement_1"] == 0 for r in recs), f"{name};{name}dg = I"

    # ---- controlled rotations via native crz ----

    def test_crx_cry_control_active(self) -> None:
        # X q0; H q1; CR*(pi); H q1; M q1.  CRX -> 0, CRY -> 1 (numpy-verified;
        # the differing outcomes also confirm CRX != CRY).
        crx = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.X(q[0]),
            qb.H(q[1]),
            qb.CRX(rad(math.pi), q[0], q[1]),
            qb.H(q[1]),
            Measure(q[1]) > c[0],
            Return(c),
        )
        assert all(r["measurement_0"] == 0 for r in self._bits(crx)), "X q0; H q1; CRX(pi); H q1 -> 0"
        cry = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.X(q[0]),
            qb.H(q[1]),
            qb.CRY(rad(math.pi), q[0], q[1]),
            qb.H(q[1]),
            Measure(q[1]) > c[0],
            Return(c),
        )
        assert all(r["measurement_0"] == 1 for r in self._bits(cry)), "X q0; H q1; CRY(pi); H q1 -> 1"
