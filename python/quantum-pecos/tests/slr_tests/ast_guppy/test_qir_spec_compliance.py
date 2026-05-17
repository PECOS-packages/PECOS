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

"""#71 Stage A: QIR spec-compliance gate over the audit corpus.

Runs Quantinuum's `qir-qis` `validate_qir()` over `SlrConverter(prog)
.qir_bc()` for every audit-corpus program and pins the *current*
spec-compliance state as an explicit, honest baseline:

- **0** programs are spec-compliant today.
- Every program whose `qir_bc()` builds fails `validate_qir` with the
  SAME single root cause: a missing standard QIR module
  attribute/flags (`output_labeling_schema`, `qir_major_version`, ...).
- A few programs fail at `qir_bc()` build time -- those are
  pre-existing AST-QIR feature gaps (symbolic loopvar, parameterized
  RX, the Steane RUS path) tracked separately (the audit v2-defer
  XFAILs / the `optional_dependency` `*_qir` failures), NOT a #71
  concern. Their identity IS pinned (`_EXPECTED_BUILD_FAILED`) so a
  NEW build regression in a currently-buildable program trips this
  gate instead of being silently absorbed as "out of #71 scope".

This is the Stage A deliverable: a precise, regression-detecting
spec-noncompliance punch list. It is GREEN because it asserts the
*known* non-compliance -- it does NOT fake compliance. When Stage B
adds the missing QIR module metadata to `ast/codegen/qir.py`,
programs will start passing `validate_qir`, this baseline assertion
will trip, and it must be updated deliberately -- that is the
intended visible-progress signal.

`qir-qis` is a `[dependency-groups].test` dependency (installed in
the workspace dev/CI test env), so this runs in the default sweep.
"""

from __future__ import annotations

import qir_qis
from pecos.slr import SlrConverter

from .audit_runner import _curated_cases  # noqa: TID252

# Single uniform Stage-A non-compliance cause across all buildable QIR.
_KNOWN_GAP = "output_labeling_schema"

# Pinned pre-existing `qir_bc()` build failures: NOT #71 metadata
# concerns (tracked separately as audit v2-defer XFAILs / the
# `optional_dependency` `*_qir` failures). Pinned by (exc type, stable
# message fragment) -- NOT full messages, which embed `/tmp/...:line`
# paths. The point (Codex review blocker): a NEW qir_bc() build
# regression in a currently-buildable program must TRIP this gate, not
# be silently absorbed into "excluded from #71 scope".
# Fragments are deliberately quote-free + identifier/head-only: LLVM
# error text backslash-escapes embedded quotes (e.g. `function
# \'__quantum__qis__rx__body\'`), so a quote-bearing fragment would not
# match. The bare identifier / message head is the stable part.
_EXPECTED_BUILD_FAILED: dict[str, tuple[str, str]] = {
    "docs.for_loopvar_symbolic": ("AttributeError", "SymbolicQubit"),
    "docs.rotation_rx": ("RuntimeError", "__quantum__qis__rx__body"),
    "qeclib.steane_pz": ("RuntimeError", "Failed to compile QIR to bitcode"),
}


def _qir_compliance_state() -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Categorize the audit corpus.

    Returns `(accepted, valerr[(label,msg)], build_failed[(label,exc_type,exc_msg)])`.
    """
    accepted: list[str] = []
    valerr: list[tuple[str, str]] = []
    build_failed: list[tuple[str, str, str]] = []
    for case in _curated_cases():
        try:
            bc = SlrConverter(case.factory()).qir_bc()
        except Exception as exc:  # pre-existing AST-QIR build gaps; identity pinned in the test
            build_failed.append((case.label, type(exc).__name__, str(exc)))
            continue
        try:
            qir_qis.validate_qir(bc)
            accepted.append(case.label)
        except qir_qis.ValidationError as exc:
            valerr.append((case.label, str(exc)))
    return accepted, valerr, build_failed


def test_audit_corpus_qir_spec_noncompliance_baseline() -> None:
    """Pin the Stage-A spec-compliance baseline (see module docstring)."""
    accepted, valerr, build_failed = _qir_compliance_state()
    build_labels = sorted(label for label, _, _ in build_failed)

    # Sanity: validate_qir actually ran on built QIR. `accepted` also
    # counts as proof so that full Stage-B progress trips the intended
    # `accepted == []` baseline assertion below (with its clear message),
    # not this guard.
    assert valerr or accepted, f"validate_qir never ran on built QIR; build_failed={build_labels}"

    # Stage A reality: nothing is spec-compliant yet. When Stage B adds
    # the QIR module metadata this will (intentionally) fail -> update
    # this baseline as the visible-progress signal.
    assert accepted == [], (
        f"{len(accepted)} program(s) now pass qir-qis validate_qir "
        f"({sorted(accepted)}). If this is Stage B progress, update this "
        "baseline deliberately."
    )

    # The non-compliance is uniform: a single missing-module-metadata
    # root cause. A NEW kind of ValidationError must surface deliberately.
    non_uniform = [(label, msg[:160]) for label, msg in valerr if _KNOWN_GAP not in msg]
    assert not non_uniform, f"new/unexpected QIR ValidationError kinds (not the {_KNOWN_GAP!r} gap): {non_uniform}"

    # Pin the build-failure set + its (exc type, message fragment). A new
    # build regression -- or a fixed one -- must be triaged deliberately
    # rather than silently scoped out of #71 (Codex review blocker).
    got = {label: (etype, emsg) for label, etype, emsg in build_failed}
    assert set(got) == set(_EXPECTED_BUILD_FAILED), (
        f"qir_bc() build-failure set changed: got {sorted(got)}, "
        f"expected {sorted(_EXPECTED_BUILD_FAILED)}. Triage any new build "
        "regression (pre-existing AST-QIR gap vs real regression) before "
        "updating this pin."
    )
    for label, (exp_type, exp_frag) in _EXPECTED_BUILD_FAILED.items():
        got_type, got_msg = got[label]
        assert got_type == exp_type, f"{label}: build-fail exc type {got_type!r}, expected {exp_type!r}"
        assert exp_frag in got_msg, f"{label}: build-fail message lacks {exp_frag!r}: {got_msg[:200]}"
