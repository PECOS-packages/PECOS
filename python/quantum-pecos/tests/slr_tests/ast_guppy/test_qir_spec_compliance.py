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

"""#71 QIR spec-compliance gate over the audit corpus -- post-B2 baseline.

Runs Quantinuum's `qir-qis` over `SlrConverter(prog).qir_bc()` for
every audit-corpus program and pins the *current* two-tier state as
an explicit, honest baseline. NOT faked-green: it asserts exactly
what is and is not compliant, so any regression OR further progress
trips it deliberately.

**Tier 1 -- `validate_qir` (QIR spec metadata): Stage B1 DONE.**
B1 added the required QIR module metadata (`output_labeling_schema`,
`qir_profiles=adaptive_profile`, the `qir_*_version` / `dynamic_*` /
`arrays` module flags). Two `validate_qir` failures are pinned
(`_EXPECTED_VALIDATE_FAILED`), neither a metadata gap:
`legacy.empty_main` (qir-qis structurally requires the entry
function to have >=1 qubit; that program has none) and
`qeclib.steane_pz` (B2 made it *build* -- see below -- and it now
reaches the qir-qis call allowlist, which rejects PECOS's
non-standard `__quantum__qis__barrierN__body`; the barrier-naming
gap is pre-existing, orthogonal to the CReg model, and intentionally
out of B2 scope). Tier-1b additionally pins the exact entry attr
*values* (qir-qis only checks presence).

**Tier 2 -- `qir_to_qis` (ingestible): Stage B2 DONE.**
Stage B2 replaced the PECOS-bespoke CReg runtime helpers
(`create_creg`/`get_creg_bit`/`set_creg_bit`/`get_int_from_creg`/
`set_creg_to_int`/`mz_to_creg_bit`) with the standard M-B2-static
model (per-CReg entry-block `alloca [N x i1]` + zeroinitializer;
`mz__body` -> static `%Result*` -> `read_result` -> `store`;
point-of-use `gep`+`load`/`store`; `zext`/`shl`/`or` pack ->
`__quantum__rt__int_record_output`). Every validate-passing program
(`_EXPECTED_QIS_OK`, n=23) now lowers via `qir_to_qis`; `qis_failed`
is empty. A NEW qir_to_qis failure -- or a dropped program -- trips
this deliberately. (`docs.while_loop` was in this set pre-#74 on a
silently-wrong single-pass approximation; #74 makes the QIR backend
fail loud on `While`, moving it to the build-failure set below.) (`adaptive_profile` is now genuinely exercised:
B2 emits `__quantum__rt__read_result` for measurement feedback.)
The deeper *semantic* proof for the load-bearing CReg shapes is
`tier2_semantic.py` (real-compiler acceptance + emitted-QIR
structural invariants + a deterministic AST->Guppy->Selene
cross-anchor). The direct `qir_to_qis`->Selene EXECUTABLE
differential is delivered (#77 Layer D `_qis_exec_records` in
`tier2_semantic.py`): `selene_sim` natively runs the LLVM-21
opaque-pointer QIS bitcode `qir_to_qis` emits, via
`selene_helios_qis_plugin` -- there is no LLVM-version blocker.
#79 generalises it corpus-wide; this structural gate provides
that suite's authoritative QIS_OK set.

**Build failures** (6): `qir_bc()` raises for
`docs.for_loopvar_symbolic` (symbolic `LoopVar` indexing) /
`docs.rotation_rx` (`rx` gate) -- pre-existing AST-QIR feature
gaps; `docs.while_loop` (#74: the QIR backend now fails LOUD
on `While` instead of silently emitting a one-pass
approximation that qir-qis cannot catch; this aligns the QIR
path with the Guppy path, which already rejects `While` per
v1-feature-matrix "real While is out of scope for the sound
emitter"); and `docs.inline_measure_creg` /
`docs.prep_basis_x` / `docs.surface_syndrome_block18` (#80:
the two QIR silent-miscompile defects #79's dual pre-review
surfaced -- an inline/`Return`-only CReg that got no storage
so its value vanished from the records, and a non-Z `Prep`
basis the converter silently dropped so it lowered as a plain
Z reset -- are now FAIL-LOUD, mirroring #74/#78). Identity
pinned (`_EXPECTED_BUILD_FAILED`) so a NEW build regression
trips here. (`qeclib.steane_pz` was a pinned
build failure pre-B2 -- the bespoke model emitted invalid bitcode
for it; B2 produces valid bitcode so it now builds and moves to
the pinned validate set above. A deliberate, triaged improvement.)
#74 also fails loud on `VarExpr` (was silently 0) and `Print`
(was silently dropped); no corpus program exercises those, so
they add no build-failure pin -- but they are no longer silent
miscompiles.

`qir-qis` is a `[dependency-groups].test` dep (default-groups
includes `test`), so this runs in the default sweep.
"""

from __future__ import annotations

import qir_qis
from pecos.slr import SlrConverter

from .audit_runner import _curated_cases  # noqa: TID252

# Pinned pre-existing `qir_bc()` build failures (NOT #71 metadata;
# task #74 / audit v2-defer). (exc type, quote-free stable fragment) --
# LLVM text backslash-escapes embedded quotes, so a quote-bearing
# fragment would not match; the bare identifier/head is stable.
_EXPECTED_BUILD_FAILED: dict[str, tuple[str, str]] = {
    "docs.for_loopvar_symbolic": ("AttributeError", "SymbolicQubit"),
    "docs.rotation_rx": ("RuntimeError", "__quantum__qis__rx__body"),
    # #74: the QIR backend now fails LOUD on `While` (was a silent
    # single-pass approximation that qir-qis could not catch -- valid
    # QIR, wrong semantics). `docs.while_loop` moved QIS_OK -> here
    # deliberately; this aligns the QIR path with the Guppy path,
    # which already rejects `While` (v1-feature-matrix: real While is
    # out of scope for the sound emitter).
    "docs.while_loop": ("NotImplementedError", "does not support While loops"),
    # #80: the two QIR silent-miscompile defects #79's dual pre-review
    # surfaced are now FAIL-LOUD; the 3 affected corpus programs moved
    # QIS_OK -> here DELIBERATELY (the #74/#78 doctrine: a silent
    # miscompile qir-qis cannot catch must raise, not bury a wrong
    # answer). Was previously "QIS_OK" but the emitted QIS was a
    # miscompile, not a validation -- exactly what #79 must not call
    # validated.
    #  - inline_measure_creg: `final` is only `Return`ed, never
    #    declared at Main scope, so it got no `alloca [N x i1]`; the
    #    measure-store was silently skipped and the explicit returned
    #    value vanished from the QIS records (QIS recorded `[]`).
    #  - prep_basis_x / surface_syndrome_block18: contain
    #    `Prep(q, "X")`; the converter dropped the basis string and
    #    every AST codegen lowered it as a plain Z reset. Fixed at the
    #    shared converter root, matching the AST->Guppy path which
    #    already rejects non-Z Prep at preflight.
    "docs.inline_measure_creg": ("NotImplementedError", "was not declared at Main scope"),
    # #81 Stage A recast the guard: the prep basis is the gate
    # identity, so ANY stray string qarg (incl. these factories'
    # `Prep(q, "X")`) fails loud. Still BUILD_FAILED; only the
    # message fragment changed. Stage D rewrites the factories to
    # dedicated gates -> these move BUILD_FAILED -> QIS_OK (re-pin
    # then via `_qir_state()`).
    "docs.prep_basis_x": ("NotImplementedError", "stray string argument"),
    "docs.surface_syndrome_block18": ("NotImplementedError", "stray string argument"),
}

# Tier 1: the non-metadata `validate_qir` failures (label -> stable,
# quote-free message fragment) so a NEW validate failure (e.g. a B1
# metadata regression) trips deliberately. Neither is a metadata gap:
#  - legacy.empty_main: qir-qis requires the entry fn to have >=1
#    qubit; this program has none (structural limitation).
#  - qeclib.steane_pz: B2 made it *build* (the bespoke model emitted
#    invalid bitcode pre-B2); it now reaches the qir-qis call
#    allowlist, which rejects PECOS's non-standard
#    `__quantum__qis__barrierN__body`. The barrier-naming gap is
#    pre-existing and orthogonal to the CReg model -- intentionally
#    out of B2 scope (bounded; barrier lowering is a separate task).
_EXPECTED_VALIDATE_FAILED: dict[str, str] = {
    "legacy.empty_main": "at least one qubit",
    "qeclib.steane_pz": "Unsupported QIR QIS function",
}

# Tier 2: post-B2, EVERY validate-passing program lowers via
# `qir_to_qis` (M-B2-static replaced the bespoke CReg helpers).
# This is the full set (n=23); `qis_failed` must be empty. A new
# qir_to_qis failure -- or a dropped/added program -- trips the
# Tier-2 assertions and must be triaged deliberately.
_EXPECTED_QIS_OK: frozenset[str] = frozenset(
    {
        "docs.flat_parallel_h_gates",
        "docs.for_static_indexing",
        "docs.repeat_state_preserving",
        "examples.measure_register_to_creg",
        "examples.parallel_bell_pairs",
        "examples.surface_d3_x_1round",
        "examples.surface_d3_z_1round",
        "legacy.function_with_returns",
        "legacy.gates_only_no_measurement",
        "legacy.individual_measurements",
        "legacy.multiple_qregs",
        "legacy.nested_blocks",
        "legacy.partial_consumption_with_block",
        "qeclib.color488_syn_extract_bare",
        "qeclib.generic_check_1flag_ch",
        "qeclib.generic_check_xyz",
        "qeclib.generic_transversal_cx",
        "qeclib.surface_patch_builder_empty",
        "qeclib.surface_std_pz",
        "v1.bell",
        "v1.conditional_correction",
        "v1.ghz_three",
        "v1.repeat_idle",
    },
)

# Codex B1 post-review blocker: qir-qis validates the PRESENCE of these
# entry attributes but NOT their values, so a value regression (e.g.
# silently reverting `qir_profiles` to "custom"/"base_profile", or
# `output_labeling_schema` to something else) would pass both qir-qis
# AND this gate. Pin the exact values via `get_entry_attributes()`.
# (`qir_profiles="adaptive_profile"` is a deliberate forward-looking
# choice for B2 -- which introduces `__quantum__rt__read_result` for
# mid-circuit measurement feedback -- NOT a current-corpus requirement:
# `base_profile` also passes `validate_qir` today, since the corpus's
# `If(creg_bit)` lowers to a plain LLVM `br` on a loaded buffer value.)
_EXPECTED_ENTRY_ATTRS: dict[str, str] = {
    "qir_profiles": "adaptive_profile",
    "output_labeling_schema": "labeled",
}


def _qir_state() -> tuple[
    list[tuple[str, str, str]],
    list[tuple[str, str]],
    list[str],
    list[tuple[str, str]],
    list[tuple[str, dict[str, str | None]]],
]:
    """Categorize the corpus.

    Returns `(build_failed[(label,exc_type,exc_msg)],
    validate_failed[(label,msg)], qis_ok[label],
    qis_failed[(label,msg)],
    entry_attrs[(label, get_entry_attributes(bc))])` -- the last only
    for validate-passing cases (so the pinned attr *values* are
    checked, since qir-qis only enforces their presence).
    """
    build_failed: list[tuple[str, str, str]] = []
    validate_failed: list[tuple[str, str]] = []
    qis_ok: list[str] = []
    qis_failed: list[tuple[str, str]] = []
    entry_attrs: list[tuple[str, dict[str, str | None]]] = []
    for case in _curated_cases():
        try:
            bc = SlrConverter(case.factory()).qir_bc()
        except Exception as exc:  # pre-existing AST-QIR build gaps; identity pinned below
            build_failed.append((case.label, type(exc).__name__, str(exc)))
            continue
        try:
            qir_qis.validate_qir(bc)
        except qir_qis.ValidationError as exc:
            validate_failed.append((case.label, str(exc)))
            continue
        entry_attrs.append((case.label, qir_qis.get_entry_attributes(bc)))
        try:
            qir_qis.qir_to_qis(bc)
            qis_ok.append(case.label)
        except qir_qis.CompilerError as exc:
            qis_failed.append((case.label, str(exc)))
    return build_failed, validate_failed, qis_ok, qis_failed, entry_attrs


def test_audit_corpus_qir_compliance_baseline() -> None:
    """Pin the post-B1 two-tier QIR-compliance baseline (see module docstring)."""
    build_failed, validate_failed, qis_ok, qis_failed, entry_attrs = _qir_state()

    # Sanity: validate_qir actually ran on built QIR (not vacuous).
    assert (
        qis_ok or qis_failed or validate_failed
    ), f"validate_qir never ran on built QIR; build_failed={sorted(label for label, _, _ in build_failed)}"

    # Pre-existing build-failure set pinned (Codex Stage-A blocker): a
    # new build regression -- or a fixed one -- must be triaged
    # deliberately, not silently scoped out.
    got_bf = {label: (etype, emsg) for label, etype, emsg in build_failed}
    assert set(got_bf) == set(_EXPECTED_BUILD_FAILED), (
        f"qir_bc() build-failure set changed: got {sorted(got_bf)}, expected "
        f"{sorted(_EXPECTED_BUILD_FAILED)}. Triage (pre-existing AST-QIR gap "
        "vs real regression) before updating this pin."
    )
    for label, (exp_type, exp_frag) in _EXPECTED_BUILD_FAILED.items():
        got_type, got_msg = got_bf[label]
        assert got_type == exp_type, f"{label}: build-fail exc type {got_type!r}, expected {exp_type!r}"
        assert exp_frag in got_msg, f"{label}: build-fail message lacks {exp_frag!r}: {got_msg[:200]}"

    # Tier 1 -- validate_qir: B1 metadata done. ONLY the pinned
    # non-metadata structural failure(s) may fail validate_qir; a new
    # one (e.g. a B1 metadata regression) trips here.
    got_vf = dict(validate_failed)
    assert set(got_vf) == set(_EXPECTED_VALIDATE_FAILED), (
        f"validate_qir failure set changed: got {sorted(got_vf)}, expected "
        f"{sorted(_EXPECTED_VALIDATE_FAILED)}. A new validate failure likely "
        "means a B1 metadata regression -- triage before re-pinning."
    )
    for label, frag in _EXPECTED_VALIDATE_FAILED.items():
        assert frag in got_vf[label], f"{label}: validate msg lacks {frag!r}: {got_vf[label][:200]}"

    # Tier 1b -- pin the exact entry-attr VALUES (Codex B1 blocker).
    # qir-qis only enforces presence, so a value regression would pass
    # both qir-qis and the presence-only tier-1 check above.
    assert entry_attrs, "no validate-passing cases to check entry-attr values"
    for label, attrs in entry_attrs:
        for key, want in _EXPECTED_ENTRY_ATTRS.items():
            assert attrs.get(key) == want, (
                f"{label}: entry attr {key!r} = {attrs.get(key)!r}, expected {want!r}. "
                "B1 metadata value regression -- qir-qis does not catch this; "
                "this pin is the only guard."
            )

    # Tier 2 -- qir_to_qis: Stage B2 DONE. Every validate-passing
    # program now lowers via qir_to_qis (M-B2-static replaced the
    # bespoke CReg helpers). The OK set is pinned and `qis_failed`
    # must be empty -- a new qir_to_qis failure (B2 regression) or a
    # dropped/added program trips this and must be triaged.
    assert set(qis_ok) == set(_EXPECTED_QIS_OK), (
        f"qir_to_qis-OK set changed: got {sorted(qis_ok)}, expected "
        f"{sorted(_EXPECTED_QIS_OK)}. Triage (B2 regression vs further "
        "progress) before updating this baseline."
    )
    assert not qis_failed, (
        "unexpected qir_to_qis failure(s) post-B2: "
        f"{[(label, msg[:160]) for label, msg in qis_failed]}. After "
        "Stage B2 every validate-passing program must lower via "
        "qir_to_qis -- triage before re-pinning."
    )
