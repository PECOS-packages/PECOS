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

"""Corpus-wide EXECUTABLE QIR validation.

For every audit-corpus program `qir_to_qis` accepts (the
`_qir_state()` QIS_OK set), actually EXECUTE the lowered QIS
(`SlrConverter(prog).qir_bc() -> qir_qis.qir_to_qis ->
selene_sim.build(BitcodeString) -> run_shots(Stim)`, via
`_qis_exec_records`) and assert the EXECUTED classical
records -- not merely "qir_to_qis accepts". This is the end-to-end
proof of the static CReg lowering + the fail-loud line
(While/Print/Permute/unsupported-gate), at corpus scale.

THE HARD PART IS THE ORACLE. Every QIS_OK program
is assigned a validation CLASS in `_MANIFEST`, derived from first
principles (reading the circuit) and CONFIRMED -- never
reverse-fitted -- by execution:

- **D** Deterministic exact-record: classical output statically
  determinable; assert the exact per-shot record over a fixed
  seed. Strongest, oracle-free, non-flaky.
- **P** Property/hard-invariant: quantum-random with a *hard*
  invariant (correlation / fixed bits / membership in a small
  exact value set). Asserted over a FIXED seed set; the invariant
  must HOLD and be EXERCISED (e.g. both Bell outcomes occur) --
  NEVER a statistical/tolerance compare (the non-negotiable
  no-flaky principle).
- **X** Excluded: no classical record (no `Return`/no Main-scope
  CReg), or a single unconstrained random bit / fully
  unconstrained register / no sound first-principles invariant.
  Documented per program; X with output still gets the
  record-shape contract.

Drift safety: the
candidate set is taken LIVE from `_qir_state()` QIS_OK, and every
QIS_OK label MUST be in `_MANIFEST` -- a new/changed corpus
program forces a classification decision (no silent gap), and the
executable set cannot drift from the structural gate.

n_qubits derivation: from the QIR
`required_num_qubits` entry attribute -- NOT a max-operand-
reference count (that is 0 for no-op QReg programs and panics
selene with "No more qubits available to allocate").

Two earlier failure modes -- inline/Return-only CReg output loss
and non-Z `Prep` silently Z-reset -- were RESOLVED upstream
(inline CReg fails loud -> BUILD_FAILED, out of QIS_OK; dedicated
`PX`/etc with correct reset+Clifford-tail lowering);
the prep cases are reclassified here from first principles under
the CORRECT semantics (`docs.prep_basis_x` is `PX`=|+>, a
single uniform Z-measure bit -> **X**, NOT an earlier
deterministic `[0]` which assumed the old broken Z-reset).

`@pytest.mark.slow` (one selene build per QIS_OK program; selene
caches artifacts). The `-m "not slow"` lane excludes it; it runs
in the full sweep.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest
from pecos.slr import CReg, Main, QReg, Return, SlrConverter, rad
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from .audit_runner import _curated_cases  # noqa: TID252
from .test_qir_spec_compliance import _qir_state  # noqa: TID252
from .test_tier2_semantic import _qis_exec_records  # noqa: TID252

# Fixed seed set for P invariants: chosen so a hard invariant must
# both HOLD and be EXERCISED (e.g. both Bell outcomes appear). NOT a
# distribution -- membership/equality only.
_SEEDS = (1, 2, 7, 42)
_SHOTS = 16


def _required_num_qubits(prog: Main) -> int:
    """n_qubits from the QIR `required_num_qubits` entry attr.

    A max-operand-reference count
    is 0 for no-op QReg programs (e.g. surface_patch_builder_empty)
    and panics selene. The entry attr is the AST root-allocator
    capacity the backend pins, which is correct for every program.
    """
    qir = SlrConverter(prog).qir()
    m = re.search(r'"required_num_qubits"="(\d+)"', qir)
    assert m, f"no required_num_qubits entry attr in QIR:\n{qir}"
    return int(m.group(1))


def _observed(prog: Main, n_qubits: int) -> set[tuple[int, ...]]:
    """Union of executed records across the fixed seed set."""
    obs: set[tuple[int, ...]] = set()
    for seed in _SEEDS:
        for rec in _qis_exec_records(prog, n_qubits, shots=_SHOTS, seed=seed):
            obs.add(tuple(rec))
    return obs


# ---------------------------------------------------------------------------
# THE MANIFEST. label -> (class, spec). Each entry derived from first
# principles (the circuit), confirmed-not-fitted by execution.
#
#  D: spec = the exact record tuple, asserted == observed (one value).
#  P: spec = a Callable[[set[tuple[int,...]]], None] hard-invariant
#     check over the union of records across the fixed seed set
#     (must HOLD and be EXERCISED; membership/equality only).
#  X: spec = a human rationale string. If the program still emits a
#     classical record, the record-shape contract still applies.
# ---------------------------------------------------------------------------


def _bell2(obs: set[tuple[int, ...]]) -> None:
    """One 2-bit CReg, H+CX Bell: every shot in {0,3}, both occur."""
    assert obs <= {(0,), (3,)}, obs
    assert obs == {(0,), (3,)}, f"Bell correlation not exercised over seeds: {obs}"


def _ghz3(obs: set[tuple[int, ...]]) -> None:
    """One 3-bit CReg, GHZ: every shot in {0,7}, both occur."""
    assert obs <= {(0,), (7,)}, obs
    assert obs == {(0,), (7,)}, f"GHZ correlation not exercised over seeds: {obs}"


def _two_indep_bell_4(obs: set[tuple[int, ...]]) -> None:
    """4-bit CReg = two independent Bell pairs (bits 0==1, 2==3)."""
    allowed = {(0,), (3,), (12,), (15,)}
    assert obs <= allowed, obs
    assert obs == allowed, f"both pairs' correlation not fully exercised: {obs}"


def _three_indep_bell_6(obs: set[tuple[int, ...]]) -> None:
    """6-bit CReg = three independent Bell pairs (bits {0,1},{2,3},{4,5})."""
    allowed = {(v,) for v in (0, 3, 12, 15, 48, 51, 60, 63)}
    assert obs <= allowed, f"a non-pair-correlated value occurred: {obs - allowed}"
    # Each pair must be exercised in both states (every bit position
    # toggles): the union must cover all 6 bits set and all clear.
    or_all = 0
    and_all = 63
    for (v,) in obs:
        or_all |= v
        and_all &= v
    assert or_all == 63, f"some pair never reached |11>: OR={or_all:06b} {obs}"
    assert and_all == 0, f"some pair never reached |00>: AND={and_all:06b} {obs}"


def _multiple_qregs(obs: set[tuple[int, ...]]) -> None:
    """Two declared 2-bit CRegs (c1, c2), two records per shot.

    First-principles HARD invariant: each CReg's high bit (bit 1)
    is provably 0 -- only qubit 0 of each QReg is measured into bit
    0, bit 1 is never written (zeroinitializer). So each record is
    in {0,1}. Execution CONFIRMED c1 and c2 are NOT correlated (all
    of (0,0)/(0,1)/(1,0)/(1,1) occur -- an earlier `c1==c2`
    first-principles guess was disproven by running it); we assert
    ONLY the sound bit-1==0 invariant, NOT a cross-register
    correlation, and require both bits exercised.
    """
    assert all(r0 <= 1 and r1 <= 1 for r0, r1 in obs), f"a CReg high bit was set: {obs}"
    assert {r0 for r0, _ in obs} == {0, 1}, f"c1 bit0 not exercised: {obs}"
    assert {r1 for _, r1 in obs} == {0, 1}, f"c2 bit0 not exercised: {obs}"


def _partial_consumption(obs: set[tuple[int, ...]]) -> None:
    """(syndrome:1, result:2), two records. CX from |0> leaves the
    ancilla |0> -> syndrome is ALWAYS 0 (hard, deterministic);
    result is unconstrained. Assert syndrome==0 every shot."""
    assert all(syn == 0 for syn, _ in obs), f"syndrome non-zero (ancilla not |0>): {obs}"


def _high_bits_zero_1(obs: set[tuple[int, ...]]) -> None:
    """docs.for_static_indexing: H unrolled 3x on q[0] (= H) ->
    q[0] random; q[1]=q[2]=|0>. 3-bit CReg -> record in {0,1}
    (bits 1,2 provably 0 == the static-For body actually
    unrolled and touched only q[0]); both 0 and 1 occur."""
    assert obs <= {(0,), (1,)}, f"a high bit set -> static-For body wrong: {obs}"
    assert obs == {(0,), (1,)}, f"q[0] randomness not exercised: {obs}"


_Spec = tuple[int, ...] | Callable[[set[tuple[int, ...]]], None] | str

_MANIFEST: dict[str, tuple[str, _Spec]] = {
    # ----- D: deterministic exact record -----
    "v1.conditional_correction": ("D", (0,)),  # |0>->c0=0; If-not-taken; |0>->c1=0
    "legacy.individual_measurements": ("D", (0,)),  # four |0> -> packed 0
    "qeclib.generic_transversal_cx": ("D", (0,)),  # CX|0..0>=|0..0> -> all 0
    # ----- P: hard invariant over fixed seeds -----
    "v1.bell": ("P", _bell2),
    "v1.ghz_three": ("P", _ghz3),
    "legacy.function_with_returns": ("P", _bell2),  # Bell, c:2
    "legacy.multiple_qregs": ("P", _multiple_qregs),
    "legacy.partial_consumption_with_block": ("P", _partial_consumption),
    "examples.measure_register_to_creg": ("P", _two_indep_bell_4),
    "examples.parallel_bell_pairs": ("P", _three_indep_bell_6),
    "docs.for_static_indexing": ("P", _high_bits_zero_1),
    # ----- X: excluded, with rationale -----
    "examples.surface_d3_x_1round": ("X", "no Main-scope CReg / no classical record"),
    "examples.surface_d3_z_1round": ("X", "no Main-scope CReg / no classical record"),
    "legacy.gates_only_no_measurement": ("X", "no CReg, no measurement record"),
    "qeclib.surface_patch_builder_empty": ("X", "build()-only: no gates/measures/records"),
    "qeclib.surface_std_pz": ("X", "pz() measures ancillas internally; no Main-scope CReg"),
    "v1.repeat_idle": ("X", "no declared CReg; bare internal measure not recorded"),
    "legacy.nested_blocks": (
        "X",
        "two independent H->meas; all 4 values uniformly possible -- "
        "no hard invariant (a distribution test is forbidden)",
    ),
    "docs.flat_parallel_h_gates": (
        "X",
        "4 independent H->meas; all 16 values uniformly possible -- no hard invariant",
    ),
    "docs.repeat_state_preserving": (
        "X",
        "Repeat(3) H on q[0] = H (odd) -> |+>; Z-measure is ONE uniform "
        "random bit -- no deterministic value or hard invariant "
        "(an earlier [0] guess assumed the result was discarded; it is recorded)",
    ),
    "docs.prep_basis_x": (
        "X",
        "PX = reset+H = |+>; Z-measure is ONE uniform random bit -- "
        "no hard invariant. PX correctness is validated behaviorally by "
        "test_prep_gates (Stim peek_bloch + Selene), not here. "
        "(An earlier deterministic [0] guess assumed the OLD broken "
        "Prep('X')=Z-reset, now fixed.)",
    ),
    "docs.rotation_rx": (
        "X",
        "Typed angle-first SLR API: `RX(rad(0.5), q)` builds + lowers via "
        "qir_to_qis (rx is qir-qis-allowlisted), but rx(0.5) is "
        "NON-CLIFFORD so it cannot execute on the Stim backend this "
        "suite uses -- excluded from the executable record (rx "
        "correctness is covered by the Guppy Quest-backed behavioral "
        "tests + the QIR rx emission). This suite is Stim-only.",
    ),
    "qeclib.color488_syn_extract_bare": (
        "X",
        "Color488 syndrome extraction: on |0..0> the X-stabiliser "
        "ancillas are not +1 eigenstates, so the syndrome is not a sound "
        "all-zero first-principles invariant at this scale without an "
        "independent decoder oracle; asserting one would ship false "
        "confidence and reverse-fitting is the forbidden anti-pattern",
    ),
    # Newly QIS_OK after the verified CY decomposition.
    # Both run the "XYZ" stabiliser check (X-Y-Z product) on |0..0>;
    # |0..0> is NOT an eigenstate of X_0 . Y_1 . Z_2 (only of
    # Z_0 . Z_1 . Z_2), so the recorded syndrome bit is uniformly
    # random -- no deterministic value or hard invariant.
    # (The pre-CY classification as D `[0]` was correct only
    # under the silent-CY-drop miscompile; with the real CY lowering
    # in place, the honest class is X.)
    "qeclib.generic_check_xyz": (
        "X",
        "XYZ stabiliser check on |0..0>: not an eigenstate -> single "
        "uniformly random syndrome bit; no hard invariant.",
    ),
    "qeclib.generic_check_1flag_ch": (
        "X",
        "XYZ flagged stabiliser check on |0..0>: same; single uniformly random syndrome bit; no hard invariant.",
    ),
}


def _qis_ok_labels() -> list[str]:
    _bf, _vf, qis_ok, _qf, _ea = _qir_state()
    return sorted(qis_ok)


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_qir_corpus_manifest_covers_qis_ok() -> None:
    """Drift guard: every live QIS_OK label has a manifest class.

    Ties the executable set to the `_qir_state()` QIS_OK so a
    new/changed corpus program (or a QIS_OK drift from a codegen
    change) forces a classification decision -- it cannot be
    silently skipped. Also pins the class histogram so a
    reclassification is deliberate.
    """
    qis_ok = set(_qis_ok_labels())
    missing = qis_ok - set(_MANIFEST)
    assert not missing, f"QIS_OK programs with no manifest class (classify them): {sorted(missing)}"
    stale = set(_MANIFEST) - qis_ok
    assert not stale, f"manifest entries no longer QIS_OK (remove/retriage): {sorted(stale)}"
    hist = {cls: sum(1 for c, _ in _MANIFEST.values() if c == cls) for cls in ("D", "P", "X")}
    assert hist == {"D": 3, "P": 8, "X": 14}, f"manifest class histogram changed (deliberate?): {hist}"


# QIS_OK programs that build + lower via qir_to_qis but contain
# NON-CLIFFORD operations the Stim backend (this suite's only
# simulator) cannot execute. They are classified X and the
# record-shape contract is waived: there is no Stim record to
# inspect. Their gate semantics are covered elsewhere (e.g. the
# Guppy Quest-backed behavioral suite). This is the Stim-only
# boundary of the executable corpus, made explicit.
_NON_CLIFFORD_UNEXECUTABLE_ON_STIM: frozenset[str] = frozenset({"docs.rotation_rx"})


@pytest.mark.slow
@pytest.mark.optional_dependency
@pytest.mark.parametrize("label", _qis_ok_labels())
def test_qir_corpus_executable(label: str) -> None:
    """Execute the lowered QIS and assert the manifest class.

    D: exact record == spec, every shot, every seed.
    P: the hard invariant holds AND is exercised over the fixed
       seed set.
    X: documented exclusion. Record-shape contract:
       a program with an explicit
       `Return(CReg)` and a Main-scope CReg `alloca` MUST emit >=1
       executed record -- an explicit-return program that produces
       no record is an output-loss miscompile, NOT a silent X.
    """
    assert label in _MANIFEST, f"unclassified QIS_OK program {label!r}"
    cls, spec = _MANIFEST[label]
    cases = {c.label: c for c in _curated_cases()}
    prog = cases[label].factory()
    n_qubits = _required_num_qubits(prog)

    qir = SlrConverter(prog).qir()
    has_record = "call void @__quantum__rt__int_record_output" in qir

    if cls == "X":
        if label in _NON_CLIFFORD_UNEXECUTABLE_ON_STIM:
            # Builds + lowers via qir_to_qis, but Stim cannot run the
            # non-Clifford op -- no executable record to assert here.
            return
        if has_record:
            # Record-shape contract: X-with-output must still
            # actually produce executable records (the exclusion is
            # "no sound invariant", NOT "no output"). An
            # explicit-return program emitting nothing would be an
            # output-loss miscompile and must fail here.
            obs = _observed(prog, n_qubits)
            assert obs, f"X-with-record {label!r} produced NO executed record (output-loss miscompile)"
            assert all(len(r) >= 1 for r in obs), f"{label!r} emitted an empty record tuple: {obs}"
        return

    obs = _observed(prog, n_qubits)
    assert obs, f"{label!r} ({cls}) produced no executed records"

    if cls == "D":
        assert obs == {tuple(spec)}, f"{label!r} D: expected exactly {{{tuple(spec)}}}, executed {sorted(obs)}"
    else:  # P
        assert callable(spec)
        spec(obs)


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_sqrt_clifford_gates_executable() -> None:
    """The SX/SXdg/SY/SYdg QIR lowering EXECUTES correctly.

    These have no direct QIR primitive; they are lowered to a
    verified executable-Clifford sequence (H;S;H / H;Sdg;H / H;X /
    H;Z -- NOT rx, which silently no-ops on the Stim backend). Pin
    the end-to-end behavioral proof through
    QIR -> qir_to_qis -> selene with deterministic, global-phase-
    and runtime-convention-immune identities: SX;SX == X,
    SXdg;SX == I, SY;SY == Y, SYdg;SY == I; and the single-gate
    cases must be genuinely Z-random (a dropped/no-op lowering
    would collapse to one value -- the silent-miscompile signature).
    """

    def _run(build):
        q = QReg("q", 1)
        c = CReg("c", 1)
        prog = Main(q, c, *build(q), Measure(q[0]) > c[0], Return(c))
        obs: set[tuple[int, ...]] = set()
        for seed in _SEEDS:
            for rec in _qis_exec_records(prog, 1, shots=_SHOTS, seed=seed):
                obs.add(tuple(rec))
        return obs

    assert _run(lambda q: [qb.SX(q[0]), qb.SX(q[0])]) == {(1,)}, "SX;SX must be X (|0>->|1>)"
    assert _run(lambda q: [qb.SXdg(q[0]), qb.SX(q[0])]) == {(0,)}, "SXdg;SX must be identity"
    assert _run(lambda q: [qb.SY(q[0]), qb.SY(q[0])]) == {(1,)}, "SY;SY must be Y (|0>->~|1>)"
    assert _run(lambda q: [qb.SYdg(q[0]), qb.SY(q[0])]) == {(0,)}, "SYdg;SY must be identity"
    assert _run(lambda q: [qb.SX(q[0])]) == {(0,), (1,)}, "SX|0> must be Z-random (not a no-op)"
    assert _run(lambda q: [qb.SY(q[0])]) == {(0,), (1,)}, "SY|0> must be Z-random (not a no-op)"


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_face_clifford_gates_executable() -> None:
    """The F/Fdg/F4/F4dg QIR lowering EXECUTES correctly.

    The single-qubit Clifford "face" rotations have no direct QIR
    primitive; they are lowered to executable-Clifford sequences
    (F=Sdg;H, Fdg=H;S, F4=H;Sdg, F4dg=S;H -- circuit order),
    verified equal up to a global phase to the PECOS StateVec
    unitary. Pin the end-to-end behavioral proof through
    QIR -> qir_to_qis -> selene with deterministic, global-phase-
    immune identities: the inverse pairs collapse to identity, the
    F|0>=|+> discriminator (a no-op lowering would be Z-random),
    and F is order-3 (F;F;F == I).
    """

    def _run(build):
        q = QReg("q", 1)
        c = CReg("c", 1)
        prog = Main(q, c, *build(q), Measure(q[0]) > c[0], Return(c))
        obs: set[tuple[int, ...]] = set()
        for seed in _SEEDS:
            for rec in _qis_exec_records(prog, 1, shots=_SHOTS, seed=seed):
                obs.add(tuple(rec))
        return obs

    assert _run(lambda q: [qb.F(q[0]), qb.Fdg(q[0])]) == {(0,)}, "F;Fdg must be identity"
    assert _run(lambda q: [qb.Fdg(q[0]), qb.F(q[0])]) == {(0,)}, "Fdg;F must be identity"
    assert _run(lambda q: [qb.F4(q[0]), qb.F4dg(q[0])]) == {(0,)}, "F4;F4dg must be identity"
    assert _run(lambda q: [qb.F4dg(q[0]), qb.F4(q[0])]) == {(0,)}, "F4dg;F4 must be identity"
    # F|0> = |+> ; H|+> = |0> -> deterministic 0. A no-op F would
    # leave H|0> = |+> -> Z-random (the silent-miscompile signature).
    assert _run(lambda q: [qb.F(q[0]), qb.H(q[0])]) == {(0,)}, "F|0>=|+>, H|+>=|0> -> 0 (F not a no-op)"
    assert _run(lambda q: [qb.F(q[0])]) == {(0,), (1,)}, "F|0> must be Z-random (not a no-op)"
    # PECOS F is the order-3 face Clifford: F;F;F == I.
    assert _run(lambda q: [qb.F(q[0]), qb.F(q[0]), qb.F(q[0])]) == {(0,)}, "F;F;F must be identity (order 3)"


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_sqrt_pauli_2q_gates_executable() -> None:
    """SZZ/SZZdg/SXX/SXXdg/SYY/SYYdg/CY EXECUTE correctly.

    These have no direct qir-qis primitive; they are lowered to
    decompositions over the qir-qis ALLOWED set (verified up-to-phase
    against the PECOS StateVec unitary). Pin the end-to-end
    behavioural proof through QIR -> qir_to_qis -> selene with
    deterministic, global-phase-immune identities:
      - inverse pairs collapse to I on |00> (G;Gdg -> 0);
      - CY|10> -> i|11>, so measuring q1 after `X q0; CY` is 1
        (a no-op CY would give 0 -- the silent-miscompile signature);
      - CY|00> -> |00> (no effect when control=0);
      - SZZ^2 with q1=|0> acts as Z on q0; H;SZZ;SZZ;H|0> -> 1
        (HZH=X; a no-op SZZ would give 0).
    """

    def _run(prep, ops, meas_idx=0):
        q = QReg("q", 2)
        c = CReg("c", 1)
        prog = Main(q, c, *[g(q) for g in prep], *[g(q) for g in ops], Measure(q[meas_idx]) > c[0], Return(c))
        obs: set[tuple[int, ...]] = set()
        for seed in _SEEDS:
            for rec in _qis_exec_records(prog, 2, shots=_SHOTS, seed=seed):
                obs.add(tuple(rec))
        return obs

    szz = lambda q: qb.SZZ(q[0], q[1])  # noqa: E731
    szzdg = lambda q: qb.SZZdg(q[0], q[1])  # noqa: E731
    sxx = lambda q: qb.SXX(q[0], q[1])  # noqa: E731
    sxxdg = lambda q: qb.SXXdg(q[0], q[1])  # noqa: E731
    syy = lambda q: qb.SYY(q[0], q[1])  # noqa: E731
    syydg = lambda q: qb.SYYdg(q[0], q[1])  # noqa: E731
    cy = lambda q: qb.CY(q[0], q[1])  # noqa: E731
    x_q0 = lambda q: qb.X(q[0])  # noqa: E731
    h_q0 = lambda q: qb.H(q[0])  # noqa: E731

    # Inverse pairs on |00> -> I -> 0
    assert _run([], [szz, szzdg]) == {(0,)}, "SZZ;SZZdg must be I on |00>"
    assert _run([], [szzdg, szz]) == {(0,)}, "SZZdg;SZZ must be I"
    assert _run([], [sxx, sxxdg]) == {(0,)}, "SXX;SXXdg must be I on |00>"
    assert _run([], [sxxdg, sxx]) == {(0,)}, "SXXdg;SXX must be I"
    assert _run([], [syy, syydg]) == {(0,)}, "SYY;SYYdg must be I on |00>"
    assert _run([], [syydg, syy]) == {(0,)}, "SYYdg;SYY must be I"

    # CY non-vacuity: X q0; CY; M q1 -> 1 (Y|0>=i|1>; no-op would give 0).
    assert _run([x_q0], [cy], meas_idx=1) == {(1,)}, "X q0; CY -> q1 must be 1 (CY not a no-op)"
    # CY|00>: control=0, no effect; measure q1 -> 0.
    assert _run([], [cy], meas_idx=1) == {(0,)}, "CY|00> -> q1 must be 0"

    # CY phase-sensitive interference discriminator:
    # H q1; CY(q0,q1); H q1; M q1 with q0=|0> -> 0 for the correct
    # Sdg;CX;S decomposition (CY is identity when control=0, so
    # H;I;H = I -> 0). A wrong-phase decomposition like S;CX;S
    # applies (Sdg.S);S=S^2=Z on the target even with control=0
    # (since CX is identity when control=0), giving H;Z;H=X -> 1.
    # The simpler computational-basis CY checks above don't catch
    # this -- this assertion is what makes the test discriminate
    # the target-phase decomposition.
    h_q1 = lambda q: qb.H(q[1])  # noqa: E731
    assert _run([h_q1], [cy, h_q1], meas_idx=1) == {
        (0,),
    }, "H q1; CY; H q1 with q0=|0> must give 0 -- a wrong-phase CY decomposition (e.g. S;CX;S) gives 1"

    # SZZ^2 discriminator: H;SZZ;SZZ;H q0 with q1=|0> -> 1 (Z on q0; HZH=X).
    # A no-op SZZ would give 0.
    assert _run([h_q0], [szz, szz, h_q0]) == {(1,)}, "H;SZZ;SZZ;H must be X on q0 (SZZ not a no-op)"


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_ch_executable() -> None:
    """CH (controlled-Hadamard) EXECUTES correctly.

    CH has no direct qir-qis primitive; it is lowered to the 2q-
    minimal 1-CX decomposition (I_c x Ry(-pi/4)_t) . CX . (I_c x
    Ry(pi/4)_t) -- verified up-to-phase against the PECOS oracle
    `gate_matrix_def.CH()` (max_err 3e-14), and exactly equal to
    the textbook block-diag(I, H). The 1-CX form beats the PECOS
    Clifford+T oracle's 2-CX form (per the "minimize 2q-gate count"
    decomposition-optimality principle; 2q ops are the hardware
    cost driver).

    CH is NOT a Clifford gate (CH . X_t . CH^{-1} =
    block-diag(X, Z) is not a Pauli string), so Stim cannot
    simulate it -- the executable test routes through the Quest
    statevector backend instead. The PECOS oracle in
    gate_matrix_def.CH() uses T gates for the same reason: any CH
    decomposition into the qir-qis allowlist requires non-Clifford
    rotations (T or arbitrary Ry/Rz angles).

    Pinned identities (deterministic, global-phase-immune):
      - CH|00> -> |00> (control=0 means no effect, measure q1 -> 0);
      - CH^2 = I on |10> (involution; X q0; CH; CH; M q1 -> 0);
      - phase/value discriminator (CH vs CX vs no-op vs Ry-sign-flip):
            X q0; H q1; CH; M q1 -> 0 deterministically
        (with c=1, target is |+>; H|+>=|0>; measurement is 0 always).
        no-op leaves |+>, giving 50/50 (both 0 and 1 observed across
        seeds/shots); CX-mutation leaves X|+>=|+>, also 50/50; an
        Ry-sign-flip mutation (Ry(pi/4) . X . Ry(-pi/4)) acts as
        (X-Z)/sqrt(2) = -H.Z, applied to |+> gives |1> -> measure 1.
    """
    import selene_sim

    quest = lambda s: selene_sim.Quest(random_seed=s)  # noqa: E731

    def _run(prep, ops, meas_idx=0):
        q = QReg("q", 2)
        c = CReg("c", 1)
        prog = Main(q, c, *[g(q) for g in prep], *[g(q) for g in ops], Measure(q[meas_idx]) > c[0], Return(c))
        obs: set[tuple[int, ...]] = set()
        for seed in _SEEDS:
            for rec in _qis_exec_records(prog, 2, shots=_SHOTS, seed=seed, simulator_factory=quest):
                obs.add(tuple(rec))
        return obs

    ch = lambda q: qb.CH(q[0], q[1])  # noqa: E731
    x_q0 = lambda q: qb.X(q[0])  # noqa: E731
    h_q1 = lambda q: qb.H(q[1])  # noqa: E731

    # control=0: CH is identity; measure q1 -> 0.
    assert _run([], [ch], meas_idx=1) == {(0,)}, "CH|00> -> q1 must be 0 (control=0; no effect)"

    # CH^2 = I (involution: H^2 = I on the c=1 sector):
    # X q0; CH; CH; M q1 -> 0.
    assert _run([x_q0], [ch, ch], meas_idx=1) == {(0,)}, "X q0; CH; CH; M q1 must be 0 (CH^2=I)"

    # Phase-sensitive discriminator: X q0; H q1; CH; M q1 -> 0.
    # With c=1, target starts at |+>; CH applies H, sending |+> -> |0>.
    # A no-op or CX-mutation leaves the target at |+> -> 50/50 (both
    # 0 and 1 appear); an Ry-sign-flip mutation (conjugation that
    # sends X -> -H.Z instead of H) sends |+> -> |1> -> measure 1.
    # All three mutations are caught by this single deterministic pin.
    assert _run([x_q0, h_q1], [ch], meas_idx=1) == {
        (0,),
    }, "X q0; H q1; CH; M q1 must be 0 deterministically (a CX-mutation, no-op, or Ry-sign-flip CH all fail)"


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_controlled_rotations_executable() -> None:
    """CRX(theta) / CRY(theta) / CRZ(theta) EXECUTE correctly.

    All three controlled rotations are non-Clifford for general theta;
    executable verification routes through the Quest statevector
    backend. Decompositions (each 1-RZZ, the 2q-minimal form) are
    verified up-to-global-phase against the canonical PECOS oracles
    in `gate_matrix_def.CRX/CRY/CRZ` (5 random angles each).

    Discriminator design: at theta=pi, PECOS R*(pi) equals the
    corresponding Pauli (RX(pi)=X, RY(pi)=Y, RZ(pi)=Z), so CR*(pi)
    acts as CX/CY/CZ in the c=1 sector. The (Test A, Test B) outcome
    pair uniquely identifies the gate (and catches no-op/wrong-gate
    mutations):
      Test A -- X q0; CR*(pi); M q1
        CRX: 1 (X|0>=|1>);   CRY: 1 (Y|0>=i|1>);   CRZ: 0 (Z|0>=|0>).
      Test B -- X q0; H q1; CR*(pi); H q1; M q1 (H-conjugate)
        CRX: 0 (HXH=Z, Z|0>=|0>);
        CRY: 1 (HYH=-Y, Y|+>=-i|->, H|->=|1>);
        CRZ: 1 (HZH=X, X|0>=|1>).
      Signatures (A,B): CRX=(1,0), CRY=(1,1), CRZ=(0,1) -- all
      three distinct, so any cross-swap or no-op is caught.

    Angle-propagation: CR*(pi/2)^2 must produce the same (Test A)
    outcome as CR*(pi) when q0=|1>. Catches a stuck-angle bug where
    the callable-params plumbing dropped theta on the floor.
    """
    import selene_sim

    quest = lambda s: selene_sim.Quest(random_seed=s)  # noqa: E731

    def _run(prep, ops, meas_idx=0):
        q = QReg("q", 2)
        c = CReg("c", 1)
        prog = Main(q, c, *[g(q) for g in prep], *[g(q) for g in ops], Measure(q[meas_idx]) > c[0], Return(c))
        obs: set[tuple[int, ...]] = set()
        for seed in _SEEDS:
            for rec in _qis_exec_records(prog, 2, shots=_SHOTS, seed=seed, simulator_factory=quest):
                obs.add(tuple(rec))
        return obs

    import math as _math

    crx_pi = lambda q: qb.CRX(rad(_math.pi), q[0], q[1])  # noqa: E731
    cry_pi = lambda q: qb.CRY(rad(_math.pi), q[0], q[1])  # noqa: E731
    crz_pi = lambda q: qb.CRZ(rad(_math.pi), q[0], q[1])  # noqa: E731
    crx_half = lambda q: qb.CRX(rad(_math.pi / 2), q[0], q[1])  # noqa: E731
    cry_half = lambda q: qb.CRY(rad(_math.pi / 2), q[0], q[1])  # noqa: E731
    crz_half = lambda q: qb.CRZ(rad(_math.pi / 2), q[0], q[1])  # noqa: E731
    x_q0 = lambda q: qb.X(q[0])  # noqa: E731
    h_q1 = lambda q: qb.H(q[1])  # noqa: E731

    # c=0 sanity: each CR*(pi)|00> -> q1 unchanged -> 0.
    assert _run([], [crx_pi], meas_idx=1) == {(0,)}, "CRX(pi)|00> -> q1 must be 0 (c=0)"
    assert _run([], [cry_pi], meas_idx=1) == {(0,)}, "CRY(pi)|00> -> q1 must be 0 (c=0)"
    assert _run([], [crz_pi], meas_idx=1) == {(0,)}, "CRZ(pi)|00> -> q1 must be 0 (c=0)"

    # Test A (computational basis): X q0; CR*(pi); M q1.
    assert _run([x_q0], [crx_pi], meas_idx=1) == {(1,)}, "Test A CRX(pi): X q0; CRX(pi); M q1 must be 1"
    assert _run([x_q0], [cry_pi], meas_idx=1) == {(1,)}, "Test A CRY(pi): X q0; CRY(pi); M q1 must be 1"
    assert _run([x_q0], [crz_pi], meas_idx=1) == {(0,)}, "Test A CRZ(pi): X q0; CRZ(pi); M q1 must be 0"

    # Test B (Hadamard sandwich -- distinguishes CR* from each other):
    # X q0; H q1; CR*(pi); H q1; M q1.
    assert _run([x_q0, h_q1], [crx_pi, h_q1], meas_idx=1) == {(0,)}, "Test B CRX(pi) must be 0"
    assert _run([x_q0, h_q1], [cry_pi, h_q1], meas_idx=1) == {(1,)}, "Test B CRY(pi) must be 1"
    assert _run([x_q0, h_q1], [crz_pi, h_q1], meas_idx=1) == {(1,)}, "Test B CRZ(pi) must be 1"

    # Angle-propagation: CR*(pi/2)^2 must match Test A CR*(pi). Verifies
    # the callable-params plumbing actually threads theta through the
    # decomp (a stuck-zero or stuck-pi bug would fail one of these).
    assert _run([x_q0], [crx_half, crx_half], meas_idx=1) == {(1,)}, "CRX(pi/2)^2 must agree with Test A CRX(pi)"
    assert _run([x_q0], [cry_half, cry_half], meas_idx=1) == {(1,)}, "CRY(pi/2)^2 must agree with Test A CRY(pi)"
    assert _run([x_q0], [crz_half, crz_half], meas_idx=1) == {(0,)}, "CRZ(pi/2)^2 must agree with Test A CRZ(pi)"

    # Control-superposition phase test (Test C). Tests A/B and the
    # angle-propagation tests above all put the control in a
    # computational basis state, so they cannot observe the c=1-only
    # relative phase that the control-side `RZ(theta/2)` in the
    # `_GATE_DECOMP[CR*]` is there to absorb. Mutations that drop
    # the control RZ (or change its angle to theta instead of
    # theta/2) survive Tests A/B silently.
    #
    # The discriminator: prep `|+>` on the
    # control and an R*(pi)-eigenstate on the target so the
    # CORRECT decomp leaves the control in `|+>`; then H(control)
    # takes `|+>` -> `|0>` deterministically.
    #   - CRZ(pi): target=|0>, RZ(pi)|0>=|0> in PECOS convention
    #     (no kickback when target=|0>); correct CR*: control
    #     stays in |+>, H sends to |0>, measure 0.
    #   - CRX(pi): target=|+>, RX(pi)~X, X|+>=|+>: same logic.
    #   - CRY(pi): target=|+i>=(|0>+i|1>)/sqrt(2), RY(pi)~Y,
    #     Y|+i>=|+i>: same logic.
    # A mutation (drop control RZ, or use theta on control instead
    # of theta/2) gives the control sector a c=1-only relative
    # phase, rotating it off |+>, so H(control); M(control) is no
    # longer deterministic 0.
    h_q0 = lambda q: qb.H(q[0])  # noqa: E731
    sz_q1 = lambda q: qb.SZ(q[1])  # noqa: E731

    # CRZ phase: H q0; CRZ(pi); H q0; M q0 -> 0 (target=|0>).
    assert _run([h_q0], [crz_pi, h_q0], meas_idx=0) == {
        (0,),
    }, "CRZ(pi) control-phase: H q0; CRZ; H q0; M q0 must be 0 (drop-control-RZ / wrong-angle mutations break this)"
    # CRX phase: H q0; H q1; CRX(pi); H q0; M q0 -> 0 (target=|+>).
    assert _run([h_q0, h_q1], [crx_pi, h_q0], meas_idx=0) == {
        (0,),
    }, "CRX(pi) control-phase: H q0; H q1; CRX; H q0; M q0 must be 0"
    # CRY phase: H q0; H q1; SZ q1; CRY(pi); H q0; M q0 -> 0 (target=|+i>).
    assert _run([h_q0, h_q1, sz_q1], [cry_pi, h_q0], meas_idx=0) == {
        (0,),
    }, "CRY(pi) control-phase: H q0; H q1; SZ q1; CRY; H q0; M q0 must be 0"
