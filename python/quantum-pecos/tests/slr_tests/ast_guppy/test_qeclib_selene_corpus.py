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

"""qeclib Selene roundtrip corpus (byte-identity safety net).

This
corpus pins per-shot measurement records under fixed seed for each tracked
qeclib Block class. Each entry was first pinned **pre-conversion** (when
the Block was still flattened) and remained byte-identical **post-conversion**
(once `block_inputs` was declared and the SLR converter began emitting
BlockDecl + BlockCall).

When a new Block is converted, the workflow is:
1. Pin its Selene record here against today's flattened compile.
2. Add `block_inputs` to the Block class.
3. Re-run this corpus; identical records prove behavior preservation.

Empirical records were probed on phase3a-blockcall branch on 2026-05-15.
Steane CX/CY/CZ + Steane X/Y/Z + Steane H are converted; their entries
now exercise the BlockCall path (parametrized lock-in in
`test_block_call_smoke.py::TestConvertedQeclibBlocksUseBlockCallPath`).

## Scope (baseline + early Block conversions)

Pattern A Blocks only -- side-effect-only Blocks that compile under v1
without errors. The BlockDecl/BlockCall lowering and the first Block
conversions have landed on
`phase3a-blockcall`; the Steane Block rows below now exercise the
real BlockCall lowering path.

Pattern B Blocks (Steane preps, surface_std_pz, color488 syn
extraction) currently fail v1 compile and are deferred-XFAIL in the
audit manifest; they'll join this corpus in later
iterations once the shape-gap expansions (single-qubit input,
single-bit input with write-back, list[Qubit] bundles,
PRODUCED/DROPPED effects) land.

## Tracked Blocks

| Block | qeclib path | Effect classification (tentative) |
|---|---|---|
| `Check` | qeclib/generic/check.py | data: live_preserved; ancilla: scratch; out: live_preserved |
| `Check1Flag` | qeclib/generic/check_1flag.py | live_preserved data; scratch ancilla/flag; live_preserved out/flag |
| `transversal_tq` (CX) | qeclib/generic/transversal.py | both registers: live_preserved (no internal measurements) |
| Steane CX/CY/CZ | qeclib/steane/gates_tq/transversal_tq.py | both registers: live_preserved (converted iters 1+2) |
| Steane X/Y/Z | qeclib/steane/gates_sq/paulis.py | single register: live_preserved |
| Steane H | qeclib/steane/gates_sq/hadamards.py | single register: live_preserved |

"""

from __future__ import annotations

from pecos import Hugr, selene_engine, sim
from pecos.slr import CReg, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.generic.check import Check
from pecos.slr.qeclib.generic.check_1flag import Check1Flag
from pecos.slr.qeclib.generic.transversal import transversal_tq
from pecos.slr.qeclib.qubit.measures import Measure
from pecos.slr.qeclib.steane.gates_sq import hadamards as steane_h
from pecos.slr.qeclib.steane.gates_sq import paulis as steane_paulis
from pecos.slr.qeclib.steane.gates_tq import transversal_tq as steane_tq


def _run_via_selene(prog: Main, *, shots: int = 4, seed: int = 42, qubits: int) -> dict:
    """Compile prog through SlrConverter.hugr() and run through Selene.

    The corpus programs use explicit `Return(...)` (the implicit-return
    path was removed).
    """
    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")
    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(qubits).seed(seed).run(shots)
    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert isinstance(raw, dict)
    return raw


class TestQeclibCheck:
    """Pin Selene records for `qeclib.generic.check.Check` (Pattern A)."""

    def test_check_xyz_on_zero_state_pinned_records(self) -> None:
        """Check([q[0], q[1], q[2]], "XYZ", q[3], c[0]) on |000> data.

        Ancilla is the only measurement target; data qubits are
        live_preserved through the Check. Probe-pinned 4-shot records.

        `Check` is now converted (`a: scratch`): Guppy allocates the
        ancilla INTERNALLY, so the run needs one physical qubit beyond
        the declared 4-qubit QReg (`qubits=5`). The Selene records are
        byte-identical to the pre-conversion flattened form -- design
        R2: parity is on behavioral records, not resource counts.
        """
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 1),
            Check([q[0], q[1], q[2]], "XYZ", q[3], c[0], with_barriers=True),
            Return(c),
        )
        raw = _run_via_selene(prog, shots=4, seed=42, qubits=5)
        # Empirical probe 2026-05-15: {'measurement_0': [1, 0, 0, 1]}
        # Unchanged post-conversion (scratch internal alloc).
        assert raw == {"measurement_0": [1, 0, 0, 1]}, raw

    def test_check_xyz_on_plus_state_pinned_records(self) -> None:
        """Same Check, data q[0]/q[1] put into |+> by H before the Check.

        Probe-pinned 4-shot records show that the Hadamard prep doesn't
        change the deterministic-under-fixed-seed ancilla measurement
        record (Check internal randomness is what dominates).
        """
        prog = Main(
            q := QReg("q", 4),
            c := CReg("c", 1),
            qb.H(q[0]),
            qb.H(q[1]),
            Check([q[0], q[1], q[2]], "XYZ", q[3], c[0], with_barriers=True),
            Return(c),
        )
        # qubits=5: converted Check allocates its scratch ancilla
        # internally (design R2 -- records unchanged, +1 physical qubit).
        raw = _run_via_selene(prog, shots=4, seed=42, qubits=5)
        # Empirical probe 2026-05-15: {'measurement_0': [1, 0, 0, 1]}
        assert raw == {"measurement_0": [1, 0, 0, 1]}, raw


class TestQeclibCheck1Flag:
    """Pin Selene records for `qeclib.generic.check_1flag.Check1Flag` (Pattern A)."""

    def test_check_1flag_xyz_on_zero_state_pinned_records(self) -> None:
        """Check1Flag with a flag qubit + two classical output bits.

        `a` (q[3]) and `flag` (q[4]) are now `scratch` (converted 5e.5):
        Guppy allocates BOTH internally, so the run needs 2 physical
        qubits beyond the declared 5-qubit QReg (`qubits=7`). Data
        qubits and the two classical bits are live_preserved. Selene
        records are byte-identical to the pre-conversion flattened form
        -- design R2: parity on behavioral records, not resource counts.
        """
        prog = Main(
            q := QReg("q", 5),
            c := CReg("c", 2),
            Check1Flag([q[0], q[1], q[2]], "XYZ", q[3], q[4], c[0], c[1], with_barriers=True),
            Return(c),
        )
        raw = _run_via_selene(prog, shots=4, seed=42, qubits=7)
        # Empirical probe 2026-05-15: unchanged post-conversion
        # {'measurement_0': [1, 0, 0, 1], 'measurement_1': [0, 0, 0, 0]}
        assert raw == {
            "measurement_0": [1, 0, 0, 1],
            "measurement_1": [0, 0, 0, 0],
        }, raw


class TestQeclibTransversalCX:
    """Pin Selene records for `qeclib.generic.transversal.transversal_tq` (Pattern A).

    transversal_tq applies a 2-qubit gate (CX here) elementwise across two
    QRegs. No internal measurements -- it's pure live_preserved on both
    input registers, with classical output coming from the trailing
    explicit Measure(a) / Measure(b) at Main scope.
    """

    def test_transversal_cx_propagates_x_pinned_records(self) -> None:
        """X on a[0] and a[2], then transversal CX(a, b) -> b[0] and b[2] flip."""
        prog = Main(
            a := QReg("a", 3),
            b := QReg("b", 3),
            c := CReg("c", 6),
            qb.X(a[0]),
            qb.X(a[2]),
            transversal_tq(qb.CX, a, b),
            Measure(a) > c[0:3],
            Measure(b) > c[3:6],
            Return(c),
        )
        raw = _run_via_selene(prog, shots=4, seed=42, qubits=6)
        # Empirical probe 2026-05-15: each measurement_N has 4 identical shots
        # because gates are deterministic (no H anywhere).
        assert raw == {
            "measurement_0": [1, 1, 1, 1],  # a[0]: X -> measured 1
            "measurement_1": [0, 0, 0, 0],  # a[1]: untouched -> 0
            "measurement_2": [1, 1, 1, 1],  # a[2]: X -> measured 1
            "measurement_3": [1, 1, 1, 1],  # b[0]: flipped by CX from a[0]=1
            "measurement_4": [0, 0, 0, 0],  # b[1]: CX from a[1]=0
            "measurement_5": [1, 1, 1, 1],  # b[2]: flipped by CX from a[2]=1
        }, raw

    def test_transversal_cx_pure_zero_pinned_records(self) -> None:
        """transversal CX on |000>|000> -> all zeros (CX from 0 is identity)."""
        prog = Main(
            a := QReg("a", 3),
            b := QReg("b", 3),
            c := CReg("c", 6),
            transversal_tq(qb.CX, a, b),
            Measure(a) > c[0:3],
            Measure(b) > c[3:6],
            Return(c),
        )
        raw = _run_via_selene(prog, shots=4, seed=42, qubits=6)
        # Empirical probe 2026-05-15: |000>|000> through CX stays |000>|000>;
        # all 6 measurements are [0, 0, 0, 0] across 4 shots.
        assert raw == {f"measurement_{i}": [0, 0, 0, 0] for i in range(6)}, raw


class TestQeclibSteaneTransversalCX:
    """Pin Selene records for `qeclib.steane.gates_tq.transversal_tq.CX` (Pattern A).

    Steane logical CX is a transversal pairwise CX between two 7-qubit registers
    (the two logical Steane patches). Both registers are live_preserved (no internal
    measurements). Block conversion adds `block_inputs={"q1": "live_preserved",
    "q2": "live_preserved"}` and these pinned records must remain byte-identical.
    """

    def _x_all_then_gate_then_measure(self, gate_class: type) -> Main:
        """Helper: a=|1>^7 (X on each), apply two-register gate, measure both registers."""
        return Main(
            a := QReg("a", 7),
            b := QReg("b", 7),
            c := CReg("c", 14),
            qb.X(a[0]),
            qb.X(a[1]),
            qb.X(a[2]),
            qb.X(a[3]),
            qb.X(a[4]),
            qb.X(a[5]),
            qb.X(a[6]),
            gate_class(a, b),
            Measure(a) > c[0:7],
            Measure(b) > c[7:14],
            Return(c),
        )

    def test_steane_cx_all_ones_to_zero_propagates_pinned_records(self) -> None:
        """a in |1>^7 (each X applied), b in |0>^7; Steane CX should flip b to |1>^7."""
        raw = _run_via_selene(self._x_all_then_gate_then_measure(steane_tq.CX), shots=4, seed=42, qubits=14)
        # Empirical probe 2026-05-15: all 14 measurements are [1, 1, 1, 1] across 4 shots.
        # a stays |1>^7 (X is preserved through Steane CX); b becomes |1>^7 from CX propagation.
        assert raw == {f"measurement_{i}": [1, 1, 1, 1] for i in range(14)}, raw

    def test_steane_cy_all_ones_to_zero_propagates_pinned_records(self) -> None:
        """Steane CY also propagates X (since Y = iXZ; X-stab passes through both factors)."""
        raw = _run_via_selene(self._x_all_then_gate_then_measure(steane_tq.CY), shots=4, seed=42, qubits=14)
        # Empirical probe 2026-05-15: same |1>^14 pattern as Steane CX.
        assert raw == {f"measurement_{i}": [1, 1, 1, 1] for i in range(14)}, raw

    def test_steane_cz_x_on_control_only_phase_propagates_pinned_records(self) -> None:
        """Steane CZ propagates phase, not bit-flips: a stays |1>^7, b stays |0>^7."""
        raw = _run_via_selene(self._x_all_then_gate_then_measure(steane_tq.CZ), shots=4, seed=42, qubits=14)
        # Empirical probe 2026-05-15: a's 7 qubits measure 1 (X preserved), b's 7 measure 0
        # (CZ doesn't propagate Z stabilizer to the standard basis measurement).
        expected: dict[str, list[int]] = {f"measurement_{i}": [1, 1, 1, 1] for i in range(7)}
        expected.update({f"measurement_{i}": [0, 0, 0, 0] for i in range(7, 14)})
        assert raw == expected, raw


class TestQeclibSteaneLogicalPaulis:
    """Pin Selene records for `qeclib.steane.gates_sq.paulis.{X,Y,Z}` (Pattern A).

    Steane logical X/Y/Z apply the gate transversally on a subset of physical qubits
    (q[4], q[5], q[6]) for the 7-qubit code. Single-input live_preserved blocks.
    """

    def _gate_then_measure(self, gate_class: type) -> Main:
        return Main(
            q := QReg("q", 7),
            c := CReg("c", 7),
            gate_class(q),
            Measure(q) > c,
            Return(c),
        )

    def test_steane_logical_x_on_zero_pinned_records(self) -> None:
        """Steane X applies X to q[4..6]; q[0..3] stay |0>, q[4..6] become |1>."""
        raw = _run_via_selene(self._gate_then_measure(steane_paulis.X), shots=4, seed=42, qubits=7)
        # Empirical probe 2026-05-15:
        expected: dict[str, list[int]] = {f"measurement_{i}": [0, 0, 0, 0] for i in range(4)}
        expected.update({f"measurement_{i}": [1, 1, 1, 1] for i in range(4, 7)})
        assert raw == expected, raw

    def test_steane_logical_y_on_zero_pinned_records(self) -> None:
        """Steane Y applies Y to q[4..6]; in Z-basis measurement outcomes match Steane X."""
        raw = _run_via_selene(self._gate_then_measure(steane_paulis.Y), shots=4, seed=42, qubits=7)
        # Empirical probe 2026-05-15: q[0..3] stay 0, q[4..6] become 1 (same as X
        # because Y|0> = i|1>; Z-basis ignores the global phase).
        expected: dict[str, list[int]] = {f"measurement_{i}": [0, 0, 0, 0] for i in range(4)}
        expected.update({f"measurement_{i}": [1, 1, 1, 1] for i in range(4, 7)})
        assert raw == expected, raw

    def test_steane_logical_z_on_zero_pinned_records(self) -> None:
        """Steane Z applies Z to q[4..6]; phase only, all measure 0 in Z-basis."""
        raw = _run_via_selene(self._gate_then_measure(steane_paulis.Z), shots=4, seed=42, qubits=7)
        # Empirical probe 2026-05-15: all 7 measurements are [0, 0, 0, 0] across 4
        # shots (Z is phase-only on |0> eigenstates).
        assert raw == {f"measurement_{i}": [0, 0, 0, 0] for i in range(7)}, raw

    def test_steane_logical_h_on_zero_pinned_records(self) -> None:
        """Steane H on |0>^7 -> |+>^7; Z-basis measurement is seeded-random per-qubit per-shot."""
        raw = _run_via_selene(self._gate_then_measure(steane_h.H), shots=4, seed=42, qubits=7)
        # Empirical probe 2026-05-15: pinned per-qubit, per-shot bit pattern.
        assert raw == {
            "measurement_0": [1, 1, 1, 0],
            "measurement_1": [0, 1, 1, 0],
            "measurement_2": [0, 1, 1, 1],
            "measurement_3": [1, 0, 1, 0],
            "measurement_4": [0, 0, 0, 0],
            "measurement_5": [0, 0, 1, 1],
            "measurement_6": [0, 1, 1, 1],
        }, raw
