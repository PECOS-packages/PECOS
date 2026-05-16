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

"""Audit runner for Step 4 Workstream B (cutover gap discovery).

Iterates a curated list of `(source_label, slr_program_factory)`
pairs from PECOS examples, qeclib, and existing test fixtures.
Runs each through `SlrConverter.hugr()` (now AST-routed by default
post-cutover) and captures any failures.

This is NOT a pytest test file. It's an audit tool run
during Workstream B and at cutover. Output is the seed for new
rows in `~/Repos/pecos-docs/design/slr/v1-audit-manifest.md`.

Invocation:
    cd /home/ciaranra/Repos/PECOS
    uv run python python/quantum-pecos/tests/slr_tests/ast_guppy/audit_runner.py

For each program: emits one of
- OK   <label>
- FAIL <label> <ExceptionType>: <truncated message>

The curated list is intentionally small at first and grows as we
identify canonical examples worth exercising. The list is the audit
surface; growing it is part of Workstream B.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.schedule import compute_cnot_schedule
from pecos.slr import (
    Barrier,
    Block,
    CReg,
    For,
    If,
    LoopVar,
    Main,
    Parallel,
    QReg,
    Repeat,
    SlrConverter,
    While,
)
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.color488 import Color488Patch
from pecos.slr.qeclib.generic.check import Check
from pecos.slr.qeclib.generic.check_1flag import Check1Flag
from pecos.slr.qeclib.generic.transversal import transversal_tq
from pecos.slr.qeclib.qubit.measures import Measure
from pecos.slr.qeclib.steane.steane_class import Steane
from pecos.slr.qeclib.surface import (
    LatticeType,
    SurfacePatchBuilder,
    SurfacePatchOrientation,
    SurfaceStdGates,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class AuditCase:
    """One audit entry: a label and a factory that builds the SLR program."""

    label: str
    factory: Callable[[], Block]
    expected_failure: ExpectedFailure | None = None


@dataclass(frozen=True)
class ExpectedFailure:
    """One accepted red-light audit outcome."""

    exception_type: str
    message_contains: str
    classification: str
    reason: str


@dataclass(frozen=True)
class AuditResult:
    """Audit outcome for one program."""

    label: str
    passed: bool
    expected: bool = False
    classification: str | None = None
    exception_type: str | None = None
    exception_message: str | None = None


def _bell() -> Block:
    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        Measure(q) > c,
    )


def _ghz_three() -> Block:
    return Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.CX(q[1], q[2]),
        Measure(q) > c,
    )


def _conditional_correction() -> Block:
    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        Measure(q[0]) > c[0],
        If(c[0]).Then(qb.X(q[1])),
        Measure(q[1]) > c[1],
    )


def _repeat_idle() -> Block:
    return Main(
        q := QReg("q", 1),
        Repeat(3).block(qb.H(q[0]), qb.H(q[0])),
        Measure(q[0]),
    )


def _legacy_individual_measurements() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_individual_measurements_compile."""
    return Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        Measure(q[0]) > c[0],
        Measure(q[1]) > c[1],
        Measure(q[2]) > c[2],
        Measure(q[3]) > c[3],
    )


def _legacy_multiple_qregs() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_multiple_qregs_compile."""
    return Main(
        q1 := QReg("q1", 2),
        q2 := QReg("q2", 2),
        c1 := CReg("c1", 2),
        c2 := CReg("c2", 2),
        qb.H(q1[0]),
        qb.H(q2[0]),
        qb.CX(q1[0], q2[0]),
        Measure(q1) > c1,
        Measure(q2) > c2,
    )


def _legacy_empty_main() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_empty_main_compiles."""
    return Main()


def _legacy_gates_only_no_measurement() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_gates_only_with_cleanup_compiles."""
    return Main(
        q := QReg("q", 3),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.CX(q[1], q[2]),
    )


def _legacy_partial_consumption_with_block() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_partial_consumption_compiles.

    Uses a `Block` subclass `MeasureAncillas` that takes data + ancilla and
    measures the ancilla. v1 flattens nested blocks (BlockCall is v2), so
    this is the cleanest test of "did flattening preserve linearity correctly?"
    """

    class MeasureAncillas(Block):
        def __init__(self, data: QReg, ancilla: QReg, syndrome: CReg) -> None:
            super().__init__()
            self.data = data
            self.ancilla = ancilla
            self.syndrome = syndrome
            self.ops = [
                qb.CX(data[0], ancilla[0]),
                Measure(ancilla) > syndrome,
            ]

    return Main(
        data := QReg("data", 2),
        ancilla := QReg("ancilla", 1),
        syndrome := CReg("syndrome", 1),
        result := CReg("result", 2),
        MeasureAncillas(data, ancilla, syndrome),
        qb.H(data[0]),
        Measure(data) > result,
    )


def _legacy_function_with_returns() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_function_with_returns_compiles.

    ProcessQubits Block uses qubits without measuring them; the outer scope
    measures. Tests that nested-block flattening preserves "live qubits flow
    out" semantics.
    """

    class ProcessQubits(Block):
        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.ops = [
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
            ]

    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        ProcessQubits(q),
        Measure(q) > c,
    )


def _legacy_nested_blocks() -> Block:
    """tests/slr_tests/guppy/test_hugr_compilation.py::test_nested_blocks_compile.

    Two-level nesting: OuterBlock contains InnerBlock. v1 flattening must
    preserve the InnerBlock's measurement consuming q[0] for the outer
    sequence to remain linearity-valid.
    """

    class InnerBlock(Block):
        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                Measure(q[0]) > c[0],
            ]

    class OuterBlock(Block):
        def __init__(self, q: QReg, c: CReg) -> None:
            super().__init__()
            self.q = q
            self.c = c
            self.ops = [
                qb.H(q[0]),
                InnerBlock(q, c),
                qb.H(q[1]),
                Measure(q[1]) > c[1],
            ]

    return Main(
        q := QReg("q", 2),
        c := CReg("c", 2),
        OuterBlock(q, c),
    )


def _examples_surface_d3_z_1round() -> Block:
    """Surface-code memory experiment from
    examples/surface_code_slr_exploration.ipynb::build_surface_code_slr.

    distance=3, num_rounds=1, basis="Z". Pure-v1 gate surface
    (Prep, H, CX, Measure, Barrier) over multi-QReg. Real-shaped
    program: 9 data + 4 X-anc + 4 Z-anc, 4 CNOT rounds.
    """
    patch = SurfacePatch.create(distance=3)
    geom = patch.geometry
    cnot_rounds = compute_cnot_schedule(patch)
    num_x = len(geom.x_stabilizers)
    num_z = len(geom.z_stabilizers)

    data = QReg("data", geom.num_data)
    x_anc = QReg("ax", num_x)
    z_anc = QReg("az", num_z)

    ops: list = []
    ops.extend(qb.Prep(data[i]) for i in range(geom.num_data))
    ops.append(Barrier())

    ops.extend(qb.Prep(x_anc[i]) for i in range(num_x))
    ops.extend(qb.Prep(z_anc[i]) for i in range(num_z))
    ops.append(Barrier())

    ops.extend(qb.H(x_anc[i]) for i in range(num_x))
    ops.append(Barrier())

    for cx_round in cnot_rounds:
        for stab_type, stab_idx, data_q in cx_round:
            if stab_type == "X":
                ops.append(qb.CX(x_anc[stab_idx], data[data_q]))
            else:
                ops.append(qb.CX(data[data_q], z_anc[stab_idx]))
        ops.append(Barrier())

    ops.extend(qb.H(x_anc[i]) for i in range(num_x))
    ops.append(Barrier())

    ops.extend(Measure(x_anc[i]) for i in range(num_x))
    ops.extend(Measure(z_anc[i]) for i in range(num_z))
    ops.append(Barrier())

    ops.extend(Measure(data[i]) for i in range(geom.num_data))

    return Main(data, x_anc, z_anc, *ops)


def _examples_surface_d3_x_1round() -> Block:
    """X-basis variant of the surface-code memory experiment.

    Same as Z variant plus H wrapping on data qubits at the start and
    before final measurement. Tiny diff but covers the X-basis path
    distinctly in case wrap-around H scheduling surfaces anything.
    """
    patch = SurfacePatch.create(distance=3)
    geom = patch.geometry
    cnot_rounds = compute_cnot_schedule(patch)
    num_x = len(geom.x_stabilizers)
    num_z = len(geom.z_stabilizers)

    data = QReg("data", geom.num_data)
    x_anc = QReg("ax", num_x)
    z_anc = QReg("az", num_z)

    ops: list = []
    ops.extend(qb.Prep(data[i]) for i in range(geom.num_data))
    ops.extend(qb.H(data[i]) for i in range(geom.num_data))
    ops.append(Barrier())

    ops.extend(qb.Prep(x_anc[i]) for i in range(num_x))
    ops.extend(qb.Prep(z_anc[i]) for i in range(num_z))
    ops.append(Barrier())

    ops.extend(qb.H(x_anc[i]) for i in range(num_x))
    ops.append(Barrier())

    for cx_round in cnot_rounds:
        for stab_type, stab_idx, data_q in cx_round:
            if stab_type == "X":
                ops.append(qb.CX(x_anc[stab_idx], data[data_q]))
            else:
                ops.append(qb.CX(data[data_q], z_anc[stab_idx]))
        ops.append(Barrier())

    ops.extend(qb.H(x_anc[i]) for i in range(num_x))
    ops.append(Barrier())

    ops.extend(Measure(x_anc[i]) for i in range(num_x))
    ops.extend(Measure(z_anc[i]) for i in range(num_z))
    ops.append(Barrier())

    ops.extend(qb.H(data[i]) for i in range(geom.num_data))
    ops.append(Barrier())
    ops.extend(Measure(data[i]) for i in range(geom.num_data))

    return Main(data, x_anc, z_anc, *ops)


def _examples_parallel_bell_pairs() -> Block:
    """Parallel/Block bell pairs lifted from
    examples/Dusting off color code code.ipynb.

    Tests v1 emitter handling of Parallel + nested Block constructs
    over a 6-qubit register with register-broadcast measurement.
    """
    return Main(
        q := QReg("q", 6),
        c := CReg("m", 6),
        Parallel(
            Block(
                qb.H(q[0]),
                qb.CX(q[0], q[1]),
            ),
            Block(
                qb.H(q[2]),
                qb.CX(q[2], q[3]),
            ),
            Block(
                qb.H(q[4]),
                qb.CX(q[4], q[5]),
            ),
        ),
        Measure(q) > c,
    )


def _examples_measure_register_to_creg() -> Block:
    """Register-broadcast measure-to-creg at 4q scale.

    Distinct from v1.bell (2q->2c) in size; same shape. Confirms v1
    emitter handles per-element fanout from QReg to CReg at >2-bit
    width without relying on Parallel/Block scaffolding.
    """
    return Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        qb.H(q[0]),
        qb.CX(q[0], q[1]),
        qb.H(q[2]),
        qb.CX(q[2], q[3]),
        Measure(q) > c,
    )


def _qeclib_generic_check_xyz() -> Block:
    """Generic Check over X/Y/Z Paulis with barriers."""
    return Main(
        q := QReg("q", 4),
        c := CReg("c", 1),
        Check([q[0], q[1], q[2]], "XYZ", q[3], c[0], with_barriers=True),
    )


def _qeclib_generic_check_1flag_ch() -> Block:
    """Flagged generic Check using the CH branch in Check1Flag."""
    return Main(
        q := QReg("q", 5),
        c := CReg("c", 2),
        Check1Flag([q[0], q[1], q[2]], "XYZ", q[3], q[4], c[0], c[1], with_barriers=True),
    )


def _qeclib_generic_transversal_cx() -> Block:
    """Generic transversal CX helper across two registers."""
    return Main(
        a := QReg("a", 3),
        b := QReg("b", 3),
        c := CReg("c", 6),
        transversal_tq(qb.CX, a, b),
        Measure(a) > c[0:3],
        Measure(b) > c[3:6],
    )


def _qeclib_surface_patch_builder_empty() -> Block:
    """SLR qeclib SurfacePatchBuilder construct from examples/Surface SLR.ipynb.

    The current surface gate methods are mostly TODOs, so this green
    case only verifies that a built RotatedSurfacePatch contributes its
    underlying QReg cleanly to AST -> Guppy -> HUGR.
    """
    return Main(
        SurfacePatchBuilder()
        .set_name("s")
        .with_distances(3, 5)
        .with_lattice(LatticeType.SQUARE)
        .with_orientation(SurfacePatchOrientation.Z_TOP_BOTTOM)
        .build(),
    )


def _qeclib_steane_pz_v2_defer() -> Block:
    """Deliberate red-light: Steane pz() is v2 per stage3-synthesis Q B."""
    return Main(
        c := Steane("c"),
        c.pz(),
    )


def _qeclib_surface_std_pz_preexisting() -> Block:
    """Deliberate red-light: surface std pz() currently fails during construction."""
    patch = (
        SurfacePatchBuilder()
        .set_name("s")
        .with_distances(3, 5)
        .with_lattice(LatticeType.SQUARE)
        .with_orientation(SurfacePatchOrientation.Z_TOP_BOTTOM)
        .build()
    )
    return Main(
        patch,
        *SurfaceStdGates.pz(patch),
    )


def _qeclib_color488_syn_extract_bare_preexisting() -> Block:
    """Deliberate red-light: color488 bare extraction currently fails during construction."""
    patch = Color488Patch("c", 5, num_ancillas=4)
    return Main(
        patch,
        syn := CReg("syn", patch.num_data - 1),
        patch.syn_extract_bare(syn),
    )


def _docs_for_static_indexing() -> Block:
    """v1-shaped For loop: string iteration variable, integer bounds, fixed-slot body.

    Shape per v1-feature-matrix: v1 supports For("i", 0, 3) over fixed
    slots, no symbolic indexing.
    """
    return Main(
        q := QReg("q", 3),
        c := CReg("c", 3),
        For("i", 0, 3).Do(
            qb.H(q[0]),
        ),
        Measure(q) > c,
    )


def _docs_flat_parallel_h_gates() -> Block:
    """Parallel of flat gates (no Block scaffolding), from
    docs/development/slr-qeclib.md::test_slr_qeclib_block_4."""
    return Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        Parallel(
            qb.H(q[0]),
            qb.H(q[1]),
            qb.H(q[2]),
            qb.H(q[3]),
        ),
        Measure(q) > c,
    )


def _docs_repeat_state_preserving() -> Block:
    """Repeat block lifted from docs/development/slr-qeclib.md::test_slr_qeclib_block_6.

    Single-H body iterates 3 times. Slot stays live across iterations.
    """
    return Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        Repeat(3).block(
            qb.H(q[0]),
        ),
        Measure(q) > c,
    )


def _docs_while_loop_v2_defer() -> Block:
    """Deliberate red-light: While loop is v2-defer per v1-feature-matrix.

    Body consumes-and-replaces the qubit slot inside an unbounded loop;
    "Fixed-point linear state through unknown iteration count is too large
    for first sound emitter."
    """
    return Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        While(c[0] == 0).Do(
            qb.H(q[0]),
            Measure(q[0]) > c[0],
        ),
    )


def _docs_for_loopvar_symbolic_v2_defer() -> Block:
    """Deliberate red-light: For with LoopVar + symbolic q[i] is v2-defer.

    Per v1-feature-matrix: "Current AST converter cannot represent symbolic
    SlotRef.index; supporting it would touch shared conversion semantics."
    """
    i = LoopVar("i")
    return Main(
        q := QReg("q", 4),
        c := CReg("c", 4),
        For(i, range(4)).Do(
            qb.H(q[i]),
        ),
        Measure(q) > c,
    )


def _docs_prep_basis_x_v2_defer() -> Block:
    """Deliberate red-light: Prep with explicit "X" basis is v2-defer.

    Per v1-feature-matrix: "Current Prep gate is reset-to-zero in the AST.
    V1 supports Z reset only. Use Prep(q); H(q) for X-basis prep."
    """
    return Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qb.Prep(q[0], "X"),
        Measure(q) > c,
    )


def _docs_rotation_rx_probe() -> Block:
    """Probe: RX rotation gate from docs/development/slr-qeclib.md.

    Per v1-feature-matrix: rotations need design; doc-test form
    `RX(q[0], 0.5)` "currently looks like extra qargs." Probe to record.
    """
    return Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        qb.RX(q[0], 0.5),
        Measure(q) > c,
    )


def _docs_inline_measure_creg() -> Block:
    """Inline measurement result CReg without a root declaration."""
    return Main(
        q := QReg("q", 2),
        Measure(q) > CReg("final", 2),
    )


def _docs_surface_syndrome_block18_probe() -> Block:
    """Probe: surface_code_syndrome doc-test shape from test_slr_qeclib_block_18.

    Uses basis-arg Prep("Z") and Prep("X") plus Block-grouped operations.
    Likely XFAIL on the Prep basis arg (v2-defer); probe to record exact
    failure or surface a real gap.
    """
    d = 2
    num_data = d * d
    num_ancilla = 2
    return Main(
        data := QReg("data", num_data),
        ancilla := QReg("anc", num_ancilla),
        syn := CReg("syn", num_ancilla),
        Block(*[qb.Prep(data[i], "Z") for i in range(num_data)]),
        Block(
            qb.Prep(ancilla[0], "X"),
            qb.H(ancilla[0]),
            qb.CX(ancilla[0], data[0]),
            qb.CX(ancilla[0], data[1]),
            qb.H(ancilla[0]),
            Measure(ancilla[0]) > syn[0],
        ),
        Block(
            qb.Prep(ancilla[1], "Z"),
            qb.CX(data[0], ancilla[1]),
            qb.CX(data[3], ancilla[1]),
            Measure(ancilla[1]) > syn[1],
        ),
        Measure(data) > CReg("final", num_data),
    )


def _curated_cases() -> list[AuditCase]:
    """v1 acceptance baseline + legacy HUGR-test corpus + examples/ + qeclib.

    The v1.* prefix is the curated acceptance baseline (mirrors
    test_v1_acceptance.py). The legacy.* prefix is the corpus from
    `tests/slr_tests/guppy/test_hugr_compilation.py` -- programs
    that currently pass via the legacy IR generator. The examples.*
    prefix is curated SLR programs lifted from
    `/home/ciaranra/Repos/PECOS/examples/`. The qeclib.* prefix
    audits programmatic qeclib constructs. Any unexpected failure on
    the AST path is a real gap candidate (manifest row). Accepted
    red-light cases are explicit XFAIL rows with classifications.

    As the audit progresses, additional cases come from docs.
    """
    return [
        # v1 acceptance baseline (all should pass; sanity checks)
        AuditCase("v1.bell", _bell),
        AuditCase("v1.ghz_three", _ghz_three),
        AuditCase("v1.conditional_correction", _conditional_correction),
        AuditCase("v1.repeat_idle", _repeat_idle),
        # Legacy HUGR corpus (passes on legacy IR; failures = real gaps)
        AuditCase("legacy.individual_measurements", _legacy_individual_measurements),
        AuditCase("legacy.multiple_qregs", _legacy_multiple_qregs),
        AuditCase("legacy.empty_main", _legacy_empty_main),
        AuditCase("legacy.gates_only_no_measurement", _legacy_gates_only_no_measurement),
        AuditCase("legacy.partial_consumption_with_block", _legacy_partial_consumption_with_block),
        AuditCase("legacy.function_with_returns", _legacy_function_with_returns),
        AuditCase("legacy.nested_blocks", _legacy_nested_blocks),
        # examples/ corpus (pass 2)
        AuditCase("examples.surface_d3_z_1round", _examples_surface_d3_z_1round),
        AuditCase("examples.surface_d3_x_1round", _examples_surface_d3_x_1round),
        AuditCase("examples.parallel_bell_pairs", _examples_parallel_bell_pairs),
        AuditCase("examples.measure_register_to_creg", _examples_measure_register_to_creg),
        # qeclib/ corpus (pass 3)
        AuditCase("qeclib.generic_check_xyz", _qeclib_generic_check_xyz),
        AuditCase("qeclib.generic_check_1flag_ch", _qeclib_generic_check_1flag_ch),
        AuditCase("qeclib.generic_transversal_cx", _qeclib_generic_transversal_cx),
        AuditCase("qeclib.surface_patch_builder_empty", _qeclib_surface_patch_builder_empty),
        AuditCase(
            "qeclib.steane_pz",
            _qeclib_steane_pz_v2_defer,
            expected_failure=ExpectedFailure(
                exception_type="GuppyCodegenError",
                message_contains="supports only one final root-level Return",
                classification="v2-defer",
                reason="Steane pz() requires nested Return / BlockCall semantics",
            ),
        ),
        AuditCase(
            "qeclib.surface_std_pz",
            _qeclib_surface_std_pz_preexisting,
            expected_failure=ExpectedFailure(
                exception_type="TypeError",
                message_contains="PrepZ.__init__()",
                classification="pre-existing",
                reason="SurfaceStdGates.pz() fails while building the SLR program",
            ),
        ),
        AuditCase(
            "qeclib.color488_syn_extract_bare",
            _qeclib_color488_syn_extract_bare_preexisting,
            expected_failure=ExpectedFailure(
                exception_type="TypeError",
                message_contains="'float' object cannot be interpreted as an integer",
                classification="pre-existing",
                reason="Color488 bare extraction fails while building the SLR program",
            ),
        ),
        # docs/ corpus (pass 4)
        AuditCase("docs.for_static_indexing", _docs_for_static_indexing),
        AuditCase("docs.flat_parallel_h_gates", _docs_flat_parallel_h_gates),
        AuditCase("docs.repeat_state_preserving", _docs_repeat_state_preserving),
        AuditCase("docs.inline_measure_creg", _docs_inline_measure_creg),
        AuditCase(
            "docs.while_loop",
            _docs_while_loop_v2_defer,
            expected_failure=ExpectedFailure(
                exception_type="GuppyCodegenError",
                message_contains="does not support While loops",
                classification="v2-defer",
                reason="While loop linearity fixed-points are outside v1 scope",
            ),
        ),
        AuditCase(
            "docs.for_loopvar_symbolic",
            _docs_for_loopvar_symbolic_v2_defer,
            expected_failure=ExpectedFailure(
                exception_type="GuppyCodegenError",
                message_contains="symbolic LoopVar indexing",
                classification="v2-defer",
                reason="Symbolic SlotRef indices require shared AST/converter design",
            ),
        ),
        AuditCase(
            "docs.prep_basis_x",
            _docs_prep_basis_x_v2_defer,
            expected_failure=ExpectedFailure(
                exception_type="GuppyCodegenError",
                message_contains="supports only Z-basis Prep",
                classification="v2-defer",
                reason="Non-Z Prep basis semantics are not represented in the v1 AST",
            ),
        ),
        AuditCase(
            "docs.rotation_rx",
            _docs_rotation_rx_probe,
            expected_failure=ExpectedFailure(
                exception_type="GuppyCodegenError",
                message_contains="does not support parameterized gate RX",
                classification="v2-defer",
                reason="Parameterized rotations need design beyond v1 fixed Clifford emission",
            ),
        ),
        AuditCase(
            "docs.surface_syndrome_block18",
            _docs_surface_syndrome_block18_probe,
            expected_failure=ExpectedFailure(
                exception_type="GuppyCodegenError",
                message_contains="supports only Z-basis Prep",
                classification="v2-defer",
                reason="Doc-test includes Prep('X'); inline result CReg shape is covered separately",
            ),
        ),
    ]


def _expected_result(case: AuditCase, exc: BaseException) -> AuditResult | None:
    expected = case.expected_failure
    if expected is None:
        return None

    exception_type = type(exc).__name__
    exception_message = str(exc).splitlines()[0][:200]
    if exception_type == expected.exception_type and expected.message_contains in str(exc):
        return AuditResult(
            label=case.label,
            passed=True,
            expected=True,
            classification=expected.classification,
            exception_type=exception_type,
            exception_message=exception_message,
        )

    return AuditResult(
        label=case.label,
        passed=False,
        exception_type="UnexpectedFailure",
        exception_message=(
            f"expected {expected.exception_type} containing {expected.message_contains!r}; "
            f"got {exception_type}: {exception_message}"
        ),
    )


def _run_case(case: AuditCase) -> AuditResult:
    """Run one program through the AST path and capture pass/fail."""
    try:
        prog = case.factory()
    except BaseException as exc:
        expected = _expected_result(case, exc)
        if expected is not None:
            return expected
        return AuditResult(
            label=case.label,
            passed=False,
            exception_type=type(exc).__name__,
            exception_message=f"factory raised: {exc}",
        )

    try:
        SlrConverter(prog).hugr()
    except BaseException as exc:
        expected = _expected_result(case, exc)
        if expected is not None:
            return expected
        return AuditResult(
            label=case.label,
            passed=False,
            exception_type=type(exc).__name__,
            exception_message=str(exc).splitlines()[0][:200],
        )

    if case.expected_failure is not None:
        return AuditResult(
            label=case.label,
            passed=False,
            exception_type="UnexpectedPass",
            exception_message=f"expected {case.expected_failure.classification} failure, but AST path compiled",
        )

    return AuditResult(label=case.label, passed=True)


def run_audit() -> list[AuditResult]:
    """Run the full curated list and return all results."""
    return [_run_case(case) for case in _curated_cases()]


def main() -> int:
    """CLI entrypoint. Returns 0 if all pass; non-zero if any fail."""
    results = run_audit()
    for r in results:
        if r.expected:
            print(f"XFAIL {r.label} {r.classification}: {r.exception_type}: {r.exception_message}")
        elif r.passed:
            print(f"OK   {r.label}")
        else:
            print(f"FAIL {r.label} {r.exception_type}: {r.exception_message}")

    failures = sum(1 for r in results if not r.passed)
    expected = sum(1 for r in results if r.expected)
    total = len(results)
    print()
    print(f"Audit summary: {total - failures}/{total} accepted; {expected} expected failures; {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
