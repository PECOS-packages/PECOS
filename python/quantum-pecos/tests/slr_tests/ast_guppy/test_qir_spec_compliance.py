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

"""#71 QIR spec-compliance gate over the audit corpus -- post-B1 baseline.

Runs Quantinuum's `qir-qis` over `SlrConverter(prog).qir_bc()` for
every audit-corpus program and pins the *current* two-tier state as
an explicit, honest baseline. NOT faked-green: it asserts exactly
what is and is not compliant, so any regression OR Stage-B progress
trips it deliberately.

**Tier 1 -- `validate_qir` (QIR spec metadata): Stage B1 DONE.**
27/28 buildable programs now PASS `validate_qir` (B1 added the
required QIR module metadata: `output_labeling_schema`,
`qir_profiles=adaptive_profile`, and the `qir_*_version` /
`dynamic_*` / `arrays` module flags). The single remaining
`validate_qir` failure is `legacy.empty_main` -- NOT a metadata
gap: qir-qis structurally requires the entry function to have >=1
qubit, and that program has none. Pinned (`_EXPECTED_VALIDATE_FAILED`)
as a known non-metadata structural limitation so a NEW validate
failure trips this gate. Tier-1b additionally pins the exact entry
attr *values* (qir-qis only checks presence). `adaptive_profile`
is a deliberate forward-looking choice for B2 (it introduces
`__quantum__rt__read_result`); it is NOT yet exercised by the
corpus (`base_profile` also validates today).

**Tier 2 -- `qir_to_qis` (ingestible): Stage B2 NOT done.**
Of the 27 validate-passing programs, 7 already lower via
`qir_to_qis` (`_EXPECTED_QIS_OK` -- the no-CReg programs); the other
20 fail uniformly with `Unsupported function: create_creg` -- the
PECOS-bespoke CReg helpers Stage B2 will replace with the standard
static `%Result` + mutable-local-buffer model. B1 is **metadata
compliance only, NOT qir-qis ingestibility** (proven: B1-alone is
validate_qir-green but qir_to_qis still rejects `create_creg`).
When B2 lands, the 20 flip to OK -> `_EXPECTED_QIS_OK` trips ->
update this baseline deliberately (the intended progress signal).

**Pre-existing build failures** (3): `qir_bc()` itself raises for
`docs.for_loopvar_symbolic` / `docs.rotation_rx` / `qeclib.steane_pz`
-- pre-existing AST-QIR feature gaps tracked separately (audit
v2-defer XFAILs / `optional_dependency` `*_qir`; the semantic-bug
split is task #74), NOT a #71 metadata concern. Identity pinned
(`_EXPECTED_BUILD_FAILED`) so a NEW build regression in a
currently-buildable program trips here instead of being silently
absorbed.

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
    "qeclib.steane_pz": ("RuntimeError", "Failed to compile QIR to bitcode"),
}

# Tier 1: the ONLY non-metadata `validate_qir` failure post-B1 --
# qir-qis requires the entry function to have >=1 qubit; this program
# has none. Pinned (label -> stable message fragment) so a NEW
# validate failure (e.g. a B1 metadata regression) trips deliberately.
_EXPECTED_VALIDATE_FAILED: dict[str, str] = {
    "legacy.empty_main": "at least one qubit",
}

# Tier 2: programs that already lower via `qir_to_qis` post-B1 (no
# bespoke CReg helpers). The remaining validate-passing programs fail
# uniformly with `_QIS_B2_GAP` until Stage B2. When B2 lands they flip
# into this set -> this pin trips -> update the baseline deliberately.
_EXPECTED_QIS_OK: frozenset[str] = frozenset(
    {
        "docs.inline_measure_creg",
        "examples.surface_d3_x_1round",
        "examples.surface_d3_z_1round",
        "legacy.gates_only_no_measurement",
        "qeclib.surface_patch_builder_empty",
        "qeclib.surface_std_pz",
        "v1.repeat_idle",
    },
)
_QIS_B2_GAP = "create_creg"

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

    # Tier 2 -- qir_to_qis: B2 NOT done. The already-ingestible set is
    # pinned; when B2 replaces the bespoke CReg helpers the currently
    # failing programs flip into it and THIS assertion trips -> update
    # the baseline deliberately (the intended progress signal).
    assert set(qis_ok) == set(_EXPECTED_QIS_OK), (
        f"qir_to_qis-OK set changed: got {sorted(qis_ok)}, expected "
        f"{sorted(_EXPECTED_QIS_OK)}. If this is Stage B2 progress, update "
        "this baseline deliberately."
    )
    # Every other validate-passing program must still fail qir_to_qis
    # for the single uniform B2 reason (the bespoke CReg helpers).
    assert qis_failed, "expected the non-OK validate-passing programs to fail qir_to_qis until B2"
    non_uniform = [(label, msg[:160]) for label, msg in qis_failed if _QIS_B2_GAP not in msg]
    assert not non_uniform, (
        f"qir_to_qis failures not uniformly the {_QIS_B2_GAP!r} (B2) gap: {non_uniform}. "
        "A new qir_to_qis failure kind must be triaged deliberately."
    )
