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

"""QIR spec-compliance gate over the audit corpus.

Runs Quantinuum's `qir-qis` over `SlrConverter(prog).qir_bc()` for
every audit-corpus program and pins the *current* two-tier state as
an explicit, honest baseline. NOT faked-green: it asserts exactly
what is and is not compliant, so any regression OR further progress
trips it deliberately.

**Tier 1 -- `validate_qir` (QIR spec metadata).**
The required QIR module metadata is emitted (`output_labeling_schema`,
`qir_profiles=adaptive_profile`, the `qir_*_version` / `dynamic_*` /
`arrays` module flags). Two `validate_qir` failures are pinned
(`_EXPECTED_VALIDATE_FAILED`), neither a metadata gap:
`legacy.empty_main` (qir-qis structurally requires the entry
function to have >=1 qubit; that program has none) and
`qeclib.steane_pz` (it *builds* -- see below -- and now
reaches the qir-qis call allowlist, which rejects PECOS's
non-standard `__quantum__qis__barrierN__body`; the barrier-naming
gap is pre-existing and orthogonal to the CReg model). Tier-1b
additionally pins the exact entry attr
*values* (qir-qis only checks presence).

**Tier 2 -- `qir_to_qis` (ingestible).**
The standard static CReg model (per-CReg entry-block
`alloca [N x i1]` + zeroinitializer;
`mz__body` -> static `%Result*` -> `read_result` -> `store`;
point-of-use `gep`+`load`/`store`; `zext`/`shl`/`or` pack ->
`__quantum__rt__int_record_output`) replaced the PECOS-bespoke CReg
runtime helpers (`create_creg`/`get_creg_bit`/`set_creg_bit`/
`get_int_from_creg`/`set_creg_to_int`/`mz_to_creg_bit`). Every
validate-passing program
(`_EXPECTED_QIS_OK`, n=25) now lowers via `qir_to_qis`; `qis_failed`
is empty. A NEW qir_to_qis failure -- or a dropped program -- trips
this deliberately. (`docs.while_loop` was once in this set on a
silently-wrong single-pass approximation; the QIR backend now
fails loud on `While`, moving it to the build-failure set below.)
(`adaptive_profile` is now genuinely exercised:
`__quantum__rt__read_result` is emitted for measurement feedback.)
The deeper *semantic* proof for the load-bearing CReg shapes is
`test_tier2_semantic.py` (real-compiler acceptance + emitted-QIR
structural invariants + a deterministic AST->Guppy->Selene
cross-anchor). The direct `qir_to_qis`->Selene EXECUTABLE
differential is delivered (`_qis_exec_records` in
`test_tier2_semantic.py`): `selene_sim` natively runs the LLVM-21
opaque-pointer QIS bitcode `qir_to_qis` emits, via
`selene_helios_qis_plugin` -- there is no LLVM-version blocker.
The corpus-wide generalisation lives in the executable suite; this
structural gate provides that suite's authoritative QIS_OK set.

**Build failures** (4): `qir_bc()` raises for
`docs.for_loopvar_symbolic` (symbolic `LoopVar` indexing) --
a pre-existing AST-QIR feature gap; `docs.while_loop` (the QIR
backend now fails LOUD on `While` instead of silently emitting a
one-pass approximation that qir-qis cannot catch; this aligns the
QIR path with the Guppy path, which already rejects `While` --
real While is out of scope for the sound emitter); and
`docs.inline_measure_creg` /
`docs.surface_syndrome_block18` (an inline/`Return`-only
CReg gets no storage, so the backend fails loud instead of silently
dropping its value -- the silent-output-loss defect this
surfaced; `surface_syndrome_block18`'s
`Measure(data) > CReg("final", ...)` is exactly that inline
CReg, so it still build-fails -- pinned honestly on
the inline-CReg reason, NOT masked by restructuring the
factory). `docs.prep_basis_x` MOVED here -> QIS_OK
(it is now a clean dedicated-gate `PX` program; the old
non-Z-`Prep` string form is gone -- the prep basis is the
gate identity). Identity pinned (`_EXPECTED_BUILD_FAILED`)
so a NEW build regression trips here. (`qeclib.steane_pz` was once a pinned
build failure -- the bespoke model emitted invalid bitcode
for it; the static model produces valid bitcode so it now builds and
moves to the pinned validate set above. A deliberate, triaged improvement.)
The backend also fails loud on `VarExpr` (was silently 0) and `Print`
(was silently dropped); no corpus program exercises those, so
they add no build-failure pin -- but they are no longer silent
miscompiles. `_process_gate` fails loud on any gate with no
QIR lowering (was a silent drop); subsequent verified lowerings
move gates *off* fail-loud. The verified
CY=Sdg;CX;S decomposition (and SZZ/SXX/SYY (+adjoints) via
RZZ) moved the two generic-check factories (use `CY`)
BUILD_FAILED -> QIS_OK (22 -> 24), re-pinned from the actual
`_qir_state()`.

`qir-qis` is a `[dependency-groups].test` dep (default-groups
includes `test`), so this runs in the default sweep.
"""

from __future__ import annotations

import qir_qis
from pecos.slr import SlrConverter

from .audit_runner import _curated_cases  # noqa: TID252

# Pinned pre-existing `qir_bc()` build failures (not a metadata gap).
# (exc type, quote-free stable fragment) --
# LLVM text backslash-escapes embedded quotes, so a quote-bearing
# fragment would not match; the bare identifier/head is stable.
_EXPECTED_BUILD_FAILED: dict[str, tuple[str, str]] = {
    "docs.for_loopvar_symbolic": ("AttributeError", "SymbolicQubit"),
    # Angle-first SLR API: `docs.rotation_rx` was BUILD_FAILED only
    # because the doc-test probe used the malformed `RX(q, 0.5)` form
    # (0.5 treated as a stray qarg). With `RX(theta, q)` the probe is
    # well-formed and `rx` is in the qir-qis allowlist, so it now builds
    # and lowers via qir_to_qis -> moved to _EXPECTED_QIS_OK (re-pinned
    # from the actual `_qir_state()`, never guessed).
    # The QIR backend now fails LOUD on `While` (was a silent
    # single-pass approximation that qir-qis could not catch -- valid
    # QIR, wrong semantics). `docs.while_loop` moved QIS_OK -> here
    # deliberately; this aligns the QIR path with the Guppy path,
    # which already rejects `While` (real While is
    # out of scope for the sound emitter).
    "docs.while_loop": ("NotImplementedError", "does not support While loops"),
    # Inline/Return-only CReg fail-loud (the silent-output-loss
    # defect this surfaced). A CReg used/measured/
    # returned but never declared at Main scope gets no
    # `alloca [N x i1]`; the backend raises instead of silently dropping it.
    #  - inline_measure_creg: `final` only `Return`ed, never declared.
    #  - surface_syndrome_block18: it is a valid
    #    dedicated-gate (`PX`/`PZ`) program (was non-Z-Prep-fail),
    #    but it still build-fails -- for a DIFFERENT, correct reason:
    #    its `Measure(data) > CReg("final", num_data)` is an inline
    #    CReg, caught by the SAME inline-CReg guard. Pinned honestly on the
    #    inline-CReg reason (NOT masked by restructuring the factory
    #    -- that would defeat the guard). `docs.prep_basis_x` moved
    #    BUILD_FAILED -> QIS_OK (now a clean `PX` program; re-pinned
    #    from the actual `_qir_state()`, never guessed).
    "docs.inline_measure_creg": ("NotImplementedError", "was not declared at Main scope"),
    "docs.surface_syndrome_block18": ("NotImplementedError", "was not declared at Main scope"),
    # The "XYZ" generic_check programs (use `CY`) were
    # BUILD_FAILED on `gate 'CY' has no QIR lowering`. A verified CY
    # decomposition (CY = Sdg(t); CX; S(t)) was added, so
    # they now build and run -- moved BUILD_FAILED -> QIS_OK
    # (re-pinned from the actual `_qir_state()`, never guessed):
    # BUILD_FAILED 7 -> 5; QIS_OK 22 -> 24.
}

# Tier 1: the non-metadata `validate_qir` failures (label -> stable,
# quote-free message fragment) so a NEW validate failure (e.g. a
# metadata regression) trips deliberately. Neither is a metadata gap:
#  - legacy.empty_main: qir-qis requires the entry fn to have >=1
#    qubit; this program has none (structural limitation).
#  - qeclib.steane_pz: the static CReg model made it *build* (the
#    bespoke model emitted invalid bitcode); it now reaches the
#    qir-qis call allowlist, which rejects PECOS's non-standard
#    `__quantum__qis__barrierN__body`. The barrier-naming gap is
#    pre-existing and orthogonal to the CReg model (barrier lowering
#    is a separate task).
_EXPECTED_VALIDATE_FAILED: dict[str, str] = {
    "legacy.empty_main": "at least one qubit",
    "qeclib.steane_pz": "Unsupported QIR QIS function",
}

# Tier 2: EVERY validate-passing program lowers via
# `qir_to_qis` (the static CReg model replaced the bespoke CReg helpers).
# This is the full set (n=25); `qis_failed` must be empty. A new
# qir_to_qis failure -- or a dropped/added program -- trips the
# Tier-2 assertions and must be triaged deliberately.
_EXPECTED_QIS_OK: frozenset[str] = frozenset(
    {
        "docs.flat_parallel_h_gates",
        "docs.for_static_indexing",
        # Now a clean dedicated-gate `PX` program (was the
        # non-Z-Prep BUILD_FAILED pin; re-pinned from actual _qir_state).
        "docs.prep_basis_x",
        "docs.repeat_state_preserving",
        # Angle-first SLR API: `RX(theta, q)` is well-formed and
        # `rx` is qir-qis-allowlisted, so the rotation_rx probe now
        # lowers via qir_to_qis (was BUILD_FAILED on the malformed
        # `RX(q, 0.5)` doc-form). Not end-to-end-executable on Stim
        # (rx is non-Clifford) -> the manifest classifies it X.
        "docs.rotation_rx",
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
        # The "XYZ" generic_check programs (use `CY`) now
        # build via the verified CY=Sdg;CX;S decomposition -- moved
        # _EXPECTED_BUILD_FAILED -> here.
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

# qir-qis validates the PRESENCE of these
# entry attributes but NOT their values, so a value regression (e.g.
# silently reverting `qir_profiles` to "custom"/"base_profile", or
# `output_labeling_schema` to something else) would pass both qir-qis
# AND this gate. Pin the exact values via `get_entry_attributes()`.
# (`qir_profiles="adaptive_profile"` is a deliberate forward-looking
# choice for the static CReg model -- which introduces `__quantum__rt__read_result` for
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
    """Pin the two-tier QIR-compliance baseline (see module docstring)."""
    build_failed, validate_failed, qis_ok, qis_failed, entry_attrs = _qir_state()

    # Sanity: validate_qir actually ran on built QIR (not vacuous).
    assert (
        qis_ok or qis_failed or validate_failed
    ), f"validate_qir never ran on built QIR; build_failed={sorted(label for label, _, _ in build_failed)}"

    # Pre-existing build-failure set pinned: a
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

    # Tier 1 -- validate_qir: metadata done. ONLY the pinned
    # non-metadata structural failure(s) may fail validate_qir; a new
    # one (e.g. a metadata regression) trips here.
    got_vf = dict(validate_failed)
    assert set(got_vf) == set(_EXPECTED_VALIDATE_FAILED), (
        f"validate_qir failure set changed: got {sorted(got_vf)}, expected "
        f"{sorted(_EXPECTED_VALIDATE_FAILED)}. A new validate failure likely "
        "means a metadata regression -- triage before re-pinning."
    )
    for label, frag in _EXPECTED_VALIDATE_FAILED.items():
        assert frag in got_vf[label], f"{label}: validate msg lacks {frag!r}: {got_vf[label][:200]}"

    # Tier 1b -- pin the exact entry-attr VALUES.
    # qir-qis only enforces presence, so a value regression would pass
    # both qir-qis and the presence-only tier-1 check above.
    assert entry_attrs, "no validate-passing cases to check entry-attr values"
    for label, attrs in entry_attrs:
        for key, want in _EXPECTED_ENTRY_ATTRS.items():
            assert attrs.get(key) == want, (
                f"{label}: entry attr {key!r} = {attrs.get(key)!r}, expected {want!r}. "
                "metadata value regression -- qir-qis does not catch this; "
                "this pin is the only guard."
            )

    # Tier 2 -- qir_to_qis: every validate-passing program now lowers
    # via qir_to_qis (the static `[N x i1]` CReg model replaced the
    # bespoke CReg helpers). The OK set is pinned and `qis_failed`
    # must be empty -- a new qir_to_qis failure (a lowering regression)
    # or a dropped/added program trips this and must be triaged.
    assert set(qis_ok) == set(_EXPECTED_QIS_OK), (
        f"qir_to_qis-OK set changed: got {sorted(qis_ok)}, expected "
        f"{sorted(_EXPECTED_QIS_OK)}. Triage (lowering regression vs further "
        "progress) before updating this baseline."
    )
    assert not qis_failed, (
        "unexpected qir_to_qis failure(s): "
        f"{[(label, msg[:160]) for label, msg in qis_failed]}. Every "
        "validate-passing program must lower via "
        "qir_to_qis -- triage before re-pinning."
    )
