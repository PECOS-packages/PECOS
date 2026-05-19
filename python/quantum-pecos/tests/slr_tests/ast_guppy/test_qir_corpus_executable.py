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

"""#79 corpus-wide EXECUTABLE QIR validation (generalises #77 Layer D).

For every audit-corpus program `qir_to_qis` accepts (the #71
`_qir_state()` QIS_OK set), actually EXECUTE the lowered QIS
(`SlrConverter(prog).qir_bc() -> qir_qis.qir_to_qis ->
selene_sim.build(BitcodeString) -> run_shots(Stim)`, via the
#77-proven `_qis_exec_records`) and assert the EXECUTED classical
records -- not merely "qir_to_qis accepts". This is the end-to-end
proof of the B2 M-B2-static lowering + #74/#78/#87(Permute)/#88A
fail-loud line, at corpus scale.

THE HARD PART IS THE ORACLE (per the scoped plan + dual
#79 design + dual review). Every QIS_OK program
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
  record-shape contract (#79 design blocker 1).

Drift safety (#79 design item 4): the
candidate set is taken LIVE from `_qir_state()` QIS_OK, and every
QIS_OK label MUST be in `_MANIFEST` -- a new/changed corpus
program forces a classification decision (no silent gap), and the
executable set cannot drift from the #71 structural gate.

n_qubits derivation (#79 design blocker 3): from the QIR
`required_num_qubits` entry attribute -- NOT a max-operand-
reference count (that is 0 for no-op QReg programs and panics
selene with "No more qubits available to allocate").

#79 design blockers 1 (inline/Return-only CReg output loss) and 2
(non-Z `Prep` silently Z-reset) were RESOLVED upstream by #80
(inline CReg fails loud -> BUILD_FAILED, out of QIS_OK) and #81
(dedicated `PX`/etc with correct reset+Clifford-tail lowering);
the prep cases are reclassified here from first principles under
the CORRECT #81 semantics (`docs.prep_basis_x` is `PX`=|+>, a
single uniform Z-measure bit -> **X**, NOT an earlier
deterministic `[0]` which assumed the old broken Z-reset).

`@pytest.mark.slow` (one selene build per QIS_OK program; selene
caches artifacts). The `-m "not slow"` lane excludes it; it runs
in the full sweep.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from pecos.slr import SlrConverter

from .audit_runner import _curated_cases  # noqa: TID252
from .test_qir_spec_compliance import _qir_state  # noqa: TID252

if TYPE_CHECKING:
    from pecos.slr import Main
from .tier2_semantic import _qis_exec_records  # noqa: TID252

# Fixed seed set for P invariants: chosen so a hard invariant must
# both HOLD and be EXERCISED (e.g. both Bell outcomes appear). NOT a
# distribution -- membership/equality only.
_SEEDS = (1, 2, 7, 42)
_SHOTS = 16


def _required_num_qubits(prog: Main) -> int:
    """n_qubits from the QIR `required_num_qubits` entry attr.

    #79 design blocker 3: a max-operand-reference count
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
    (bits 1,2 provably 0 == the #74 static-For body actually
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
        "#81 PX = reset+H = |+>; Z-measure is ONE uniform random bit -- "
        "no hard invariant. PX correctness is validated behaviorally by "
        "test_prep_gates (Stim peek_bloch + Selene), not here. "
        "(An earlier deterministic [0] guess assumed the OLD broken "
        "Prep('X')=Z-reset, fixed by #81.)",
    ),
    "qeclib.color488_syn_extract_bare": (
        "X",
        "Color488 syndrome extraction: on |0..0> the X-stabiliser "
        "ancillas are not +1 eigenstates, so the syndrome is not a sound "
        "all-zero first-principles invariant at this scale without an "
        "independent decoder oracle; asserting one would ship false "
        "confidence and reverse-fitting is the forbidden anti-pattern",
    ),
}


def _qis_ok_labels() -> list[str]:
    _bf, _vf, qis_ok, _qf, _ea = _qir_state()
    return sorted(qis_ok)


@pytest.mark.slow
@pytest.mark.optional_dependency
def test_qir_corpus_manifest_covers_qis_ok() -> None:
    """Drift guard: every live QIS_OK label has a manifest class.

    Ties the executable set to the #71 `_qir_state()` QIS_OK so a
    new/changed corpus program (or a QIS_OK drift from a codegen
    change) forces a classification decision -- it cannot be
    silently skipped. Also pins the class histogram so a
    reclassification is deliberate.
    """
    qis_ok = set(_qis_ok_labels())
    missing = qis_ok - set(_MANIFEST)
    assert not missing, f"QIS_OK programs with no #79 manifest class (classify them): {sorted(missing)}"
    stale = set(_MANIFEST) - qis_ok
    assert not stale, f"manifest entries no longer QIS_OK (remove/retriage): {sorted(stale)}"
    hist = {cls: sum(1 for c, _ in _MANIFEST.values() if c == cls) for cls in ("D", "P", "X")}
    assert hist == {"D": 3, "P": 8, "X": 11}, f"manifest class histogram changed (deliberate?): {hist}"


@pytest.mark.slow
@pytest.mark.optional_dependency
@pytest.mark.parametrize("label", _qis_ok_labels())
def test_qir_corpus_executable(label: str) -> None:
    """Execute the lowered QIS and assert the manifest class.

    D: exact record == spec, every shot, every seed.
    P: the hard invariant holds AND is exercised over the fixed
       seed set.
    X: documented exclusion. Record-shape contract (#79
       design blocker 1): a program with an explicit
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
        if has_record:
            # Record-shape contract: X-with-output must still
            # actually produce executable records (the exclusion is
            # "no sound invariant", NOT "no output"). An
            # explicit-return program emitting nothing would be the
            # #80-class output-loss miscompile and must fail here.
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
