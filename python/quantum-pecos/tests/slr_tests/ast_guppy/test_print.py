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

"""Phase 1 behavioral tests for `Print(value, *, tag=None, namespace="result")`.

Print lowers to Guppy's `result(name, value)` and surfaces in Selene's parsed
result dict under the key `f"{namespace}.{tag}"`. Tests verify:

- Tag derivation from CReg / Bit values (Print(c) -> tag "result.c";
  Print(c[0]) -> tag "result.c_0").
- Explicit tag/namespace overrides.
- Print inside Repeat(n) and fixed-bound For emits event-list under same key.
- Construction-time rejection of invalid values, tags, and namespaces.
"""

from __future__ import annotations

import pytest
from pecos import Hugr, selene_engine, sim
from pecos.slr import CReg, For, If, Main, Print, QReg, Repeat, SlrConverter
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure


def _run_and_get_result_dict(prog: Main, *, shots: int = 10, seed: int = 42, qubits: int = 1) -> dict:
    """Compile prog via SlrConverter, run through Selene, return raw result dict."""
    package = SlrConverter(prog).hugr()
    hugr_bytes = package.to_str().encode("utf-8")
    result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(qubits).seed(seed).run(shots)
    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert isinstance(raw, dict)
    return raw


# ── Tag derivation ───────────────────────────────────────────────────────


class TestTagDerivation:
    """Print's default tag is derived from the value's name."""

    def test_print_whole_creg_derives_register_name(self) -> None:
        """Print(c) emits under tag "result.c"."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q) > c,
            Print(c),
        )
        raw = _run_and_get_result_dict(prog, qubits=2)
        assert "result.c" in raw, f"expected 'result.c' tag in {list(raw.keys())}"

    def test_print_bit_ref_derives_register_index_name(self) -> None:
        """Print(c[0]) emits under tag "result.c_0"."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c[0]),
        )
        raw = _run_and_get_result_dict(prog)
        assert "result.c_0" in raw, f"expected 'result.c_0' tag in {list(raw.keys())}"


# ── Namespace + tag overrides ────────────────────────────────────────────


class TestNamespaceAndTag:
    """Namespace prefixes the tag; explicit tag overrides derived name."""

    def test_namespace_debug_emits_under_debug_prefix(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, namespace="debug"),
        )
        raw = _run_and_get_result_dict(prog)
        assert "debug.c" in raw

    def test_explicit_tag_overrides_derived(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="step_1"),
        )
        raw = _run_and_get_result_dict(prog)
        assert "result.step_1" in raw

    def test_namespace_and_tag_together(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="r1", namespace="debug"),
        )
        raw = _run_and_get_result_dict(prog)
        assert "debug.r1" in raw


# ── Loop bodies ──────────────────────────────────────────────────────────


class TestPrintInLoops:
    """Print inside Repeat(n) and fixed-bound For emits n times under the same tag."""

    def test_print_in_repeat_emits_event_per_iteration(self) -> None:
        """Each shot emits the tag once per Repeat iteration; assert count + value.

        Selene shape for `Print(c)` of a single-bit CReg: the dict value is
        a list of per-shot lists, each inner list holding one int per Print
        call. Example for 2 shots x 3 iterations: `[[1,1,1], [1,1,1]]`.
        """
        shots = 5
        n_iters = 3
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Repeat(n_iters).block(
                qb.X(q[0]),
                Measure(q[0]) > c[0],
                qb.Prep(q[0]),
                Print(c, tag="iter"),
            ),
        )
        raw = _run_and_get_result_dict(prog, shots=shots)
        assert "result.iter" in raw, f"expected 'result.iter' in {list(raw.keys())}"
        events = raw["result.iter"]
        assert len(events) == shots, f"expected {shots} per-shot event lists, got {len(events)}"
        for shot_events in events:
            assert len(shot_events) == n_iters, f"expected {n_iters} events per shot, got {len(shot_events)}"
            # Each iteration prepared then measured X|0>=|1>; expect every event = 1.
            assert all(int(bit) == 1 for bit in shot_events), shot_events

    def test_print_in_static_for_emits_event_per_iteration(self) -> None:
        shots = 5
        n_iters = 2
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            For("i", 0, n_iters).Do(
                qb.X(q[0]),
                Measure(q[0]) > c[0],
                qb.Prep(q[0]),
                Print(c, tag="loop"),
            ),
        )
        raw = _run_and_get_result_dict(prog, shots=shots)
        assert "result.loop" in raw
        events = raw["result.loop"]
        assert len(events) == shots
        for shot_events in events:
            assert len(shot_events) == n_iters
            assert all(int(bit) == 1 for bit in shot_events), shot_events


# ── Construction-time negative tests ─────────────────────────────────────


class TestPrintConstructionRejection:
    """SLR-construction-time validation rejects bad inputs immediately."""

    def test_print_of_non_creg_value_rejected(self) -> None:
        with pytest.raises(TypeError, match="requires a CReg or Bit value"):
            Print(42)

    def test_print_with_invalid_tag_chars_rejected(self) -> None:
        c = CReg("c", 1)
        with pytest.raises(ValueError, match="must match"):
            Print(c, tag="bad-tag")

    def test_print_with_invalid_namespace_chars_rejected(self) -> None:
        c = CReg("c", 1)
        with pytest.raises(ValueError, match="must match"):
            Print(c, namespace="1bad")

    def test_print_with_namespace_containing_dot_rejected(self) -> None:
        """The dot is reserved as the namespace-tag separator."""
        c = CReg("c", 1)
        with pytest.raises(ValueError, match="must match"):
            Print(c, namespace="bad.namespace")

    def test_print_tag_can_use_underscore_and_digits(self) -> None:
        """Identifier-rule chars are accepted."""
        c = CReg("c", 1)
        p = Print(c, tag="syn_round_0")
        assert p.tag == "syn_round_0"

    def test_print_namespace_with_underscore_accepted(self) -> None:
        c = CReg("c", 1)
        p = Print(c, namespace="my_ns")
        assert p.namespace == "my_ns"

    def test_print_derived_tag_validated_against_identifier_rules(self) -> None:
        """A CReg named with non-identifier chars yields a non-identifier derived tag.

        Construction must reject it; the user should pass `tag=...` explicitly.
        Without this check the value would silently produce a malformed
        `result()` tag.
        """
        bad_creg = CReg("bad-name", 1)
        with pytest.raises(ValueError, match="must match"):
            Print(bad_creg)


# ── Print and Selene's runtime output ───────────────────────────────────


class TestPrintAndSeleneOutput:
    """Print does not change the SLR/AST/Guppy return shape, but Selene's runtime
    flips representation modes once any `result()` call is present.

    Empirically (verified 2026-05-14):

    - No Print, implicit return: Selene exposes return-tuple positionally as
      `measurement_N` keys.
    - Any Print, implicit return still present in Guppy `return ...`: Selene
      switches to result-tag mode and the `measurement_N` keys are NOT
      exposed; only `result()` tags appear in `to_dict()`.

    So Print is **AST-scope-orthogonal** (no AST/Guppy return shape change)
    but **Selene-runtime-mode-flipping** (presence of any `result()` switches
    the output dict's representation). v2 breaking-migration anticipates this
    by making Print/Return the only output mechanisms.
    """

    def test_no_print_yields_implicit_measurement_records(self) -> None:
        """Sanity baseline: no Print, Selene exposes return-tuple as measurement_N."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )
        raw = _run_and_get_result_dict(prog, shots=2)
        assert "measurement_0" in raw
        assert all(int(bit) == 1 for bit in raw["measurement_0"])
        # Tag-mode keys are absent in this mode.
        assert "result.c" not in raw

    def test_print_switches_selene_to_tag_mode_and_hides_measurement_records(self) -> None:
        """Adding any Print suppresses the implicit `measurement_N` keys.

        This is a Selene runtime behavior, not an AST/Guppy semantics change:
        the generated Guppy still has `return c`, but Selene's parsed dict
        only shows result tags when any `result()` call exists.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="p"),
        )
        raw = _run_and_get_result_dict(prog, shots=2)
        assert "result.p" in raw
        assert "measurement_0" not in raw, f"expected measurement_0 hidden in tag mode, got {list(raw.keys())}"

    def test_multiple_prints_distinct_tags_each_appear(self) -> None:
        """Two Prints with distinct tags each yield a separate result-dict key."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="early"),
            qb.Prep(q[0]),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="late"),
        )
        raw = _run_and_get_result_dict(prog, shots=2)
        assert "result.early" in raw
        assert "result.late" in raw

    def test_two_prints_same_tag_become_event_list(self) -> None:
        """Two Print(c, tag="same") calls emit the tag twice per shot under one key.

        Selene returns an event-list shape `{tag: [[ev0_shot0, ev1_shot0], ...]}`.
        """
        shots = 3
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="same"),
            qb.Prep(q[0]),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="same"),  # same tag, second emission
        )
        raw = _run_and_get_result_dict(prog, shots=shots)
        assert "result.same" in raw
        events = raw["result.same"]
        assert len(events) == shots, f"expected {shots} per-shot event lists, got {len(events)}"
        for shot_events in events:
            assert len(shot_events) == 2, f"expected 2 events per shot, got {len(shot_events)}"
            assert all(int(bit) == 1 for bit in shot_events), shot_events


# ── Path-signature validator (If/Elif symmetry) ──────────────────────────


class TestPathSignatureValidator:
    """Reject asymmetric Print emission across If/Elif branches.

    Phase 1 requires that the ordered sequence of Print events along every
    conditional path is identical. Selene's parsed-result dict expects
    rectangular tag emission per shot; asymmetric emission triggers a
    register-count mismatch at runtime, so the AST validator fails fast.
    """

    def test_then_only_print_rejected(self) -> None:
        """Print in `Then` with no Else (or empty Else) → asymmetric."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(Print(c, tag="only_then")),
        )
        with pytest.raises(GuppyCodegenError, match="path-signature mismatch"):
            SlrConverter(prog).hugr()

    def test_symmetric_if_then_else_accepted(self) -> None:
        """Same Print on both branches compiles."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(Print(c, tag="branch_taken")).Else(Print(c, tag="branch_taken")),
        )
        SlrConverter(prog).hugr()

    def test_asymmetric_tag_rejected(self) -> None:
        """Same shape, different tags across branches → reject."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(Print(c, tag="branch_a")).Else(Print(c, tag="branch_b")),
        )
        with pytest.raises(GuppyCodegenError, match="path-signature mismatch"):
            SlrConverter(prog).hugr()

    def test_asymmetric_multiplicity_rejected(self) -> None:
        """Two Prints in Then, one in Else → reject."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0])
            .Then(
                Print(c, tag="event"),
                Print(c, tag="event"),
            )
            .Else(Print(c, tag="event")),
        )
        with pytest.raises(GuppyCodegenError, match="path-signature mismatch"):
            SlrConverter(prog).hugr()

    def test_asymmetric_namespace_rejected(self) -> None:
        """Same tag, different namespace across branches → reject."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(Print(c, tag="x")).Else(Print(c, namespace="debug", tag="x")),
        )
        with pytest.raises(GuppyCodegenError, match="path-signature mismatch"):
            SlrConverter(prog).hugr()

    def test_static_repeat_with_print_compiles(self) -> None:
        """Repeat(n) with Print inside is fine (static trip count)."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Repeat(3).block(
                qb.X(q[0]),
                Measure(q[0]) > c[0],
                qb.Prep(q[0]),
                Print(c, tag="iter"),
            ),
        )
        SlrConverter(prog).hugr()

    def test_static_for_with_print_compiles(self) -> None:
        """For with literal start/stop and Print inside compiles."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            For("i", 0, 3).Do(
                qb.X(q[0]),
                Measure(q[0]) > c[0],
                qb.Prep(q[0]),
                Print(c, tag="loop"),
            ),
        )
        SlrConverter(prog).hugr()

    def test_nested_if_with_symmetric_prints_accepted(self) -> None:
        """Outer If has symmetric Prints; inner If has symmetric Prints."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            If(c[0])
            .Then(
                If(c[1]).Then(Print(c, tag="inner")).Else(Print(c, tag="inner")),
                Print(c, tag="outer"),
            )
            .Else(
                If(c[1]).Then(Print(c, tag="inner")).Else(Print(c, tag="inner")),
                Print(c, tag="outer"),
            ),
        )
        SlrConverter(prog).hugr()

    def test_nested_if_with_asymmetric_inner_rejected(self) -> None:
        """Inner If has asymmetric Prints; rejected at the inner If."""
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            If(c[0]).Then(If(c[1]).Then(Print(c, tag="leak"))),  # Else missing on inner
        )
        with pytest.raises(GuppyCodegenError, match="path-signature mismatch"):
            SlrConverter(prog).hugr()


# ── Inline-CReg definite-assignment validator ────────────────────────────


class TestInlineCRegDefiniteAssignment:
    """Reject `Print(inline_creg)` when no prior Measure has populated it.

    Inline CRegs (those introduced only via `Measure(q) > CReg(...)` without
    being declared as a positional in `Main(...)`) auto-initialize to all-False
    at the start of `main()` in the generated Guppy. A `Print` running before
    any Measure has written to such a CReg silently emits zeros, which the
    user almost certainly did not intend.

    Declared CRegs are NOT validated: explicit declaration is the user's
    acknowledgement of the zero-init.
    """

    def test_print_bit_before_measure_on_inline_creg_rejected(self) -> None:
        """Original tracer-bullet bug case, lifted to bit-level form.

        Print(inline[0]) before Measure(...) > inline[0] is rejected because
        bit-level definite-assignment hasn't been established for index 0.
        """
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            Print(inline[0], tag="before_measure"),
            Measure(q[0]) > inline[0],
        )
        with pytest.raises(GuppyCodegenError, match=r"references inline CReg"):
            SlrConverter(prog).hugr()

    def test_print_bit_after_measure_on_inline_creg_accepted(self) -> None:
        """Print(inline[0]) after Measure(...) > inline[0] → OK."""
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            Measure(q[0]) > inline[0],
            Print(inline[0], tag="after_measure"),
        )
        SlrConverter(prog).hugr()

    def test_print_of_declared_creg_before_measure_accepted(self) -> None:
        """Declared CReg (in Main.vars) is user-acknowledged zero-init; Print OK."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),  # declared positional, not inline
            Print(c, tag="zero_init"),  # user knows c starts all-False
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c, tag="after"),
        )
        SlrConverter(prog).hugr()

    def test_print_bit_ref_before_measure_on_inline_creg_rejected(self) -> None:
        """Print(c[0]) where c is inline-only also rejected."""
        inline = CReg("inline", 2)
        prog = Main(
            q := QReg("q", 2),
            qb.X(q[0]),
            Print(inline[0], tag="early"),
            Measure(q[0]) > inline[0],
            Measure(q[1]) > inline[1],
        )
        with pytest.raises(GuppyCodegenError, match=r"references inline CReg"):
            SlrConverter(prog).hugr()

    def test_inline_creg_assigned_in_both_if_branches_propagates(self) -> None:
        """Measure-into-bit on both Then and Else marks the bit assigned after the If."""
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(qb.X(q[1]), Measure(q[1]) > inline[0]).Else(Measure(q[1]) > inline[0]),
            Print(inline[0], tag="post_if"),
        )
        SlrConverter(prog).hugr()

    def test_inline_creg_assigned_only_in_then_does_not_propagate(self) -> None:
        """Measure-into-bit in Then only → after-If, that bit still not definitely assigned."""
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(Measure(q[0]) > inline[0]),
            Print(inline[0], tag="maybe"),
        )
        with pytest.raises(GuppyCodegenError, match=r"references inline CReg"):
            SlrConverter(prog).hugr()

    def test_inline_creg_assigned_in_repeat_body_propagates(self) -> None:
        """Repeat(n) with n>=1 runs body at least once; bit assignment propagates out."""
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 1),
            Repeat(3).block(qb.X(q[0]), Measure(q[0]) > inline[0], qb.Prep(q[0])),
            Print(inline[0], tag="post_repeat"),
        )
        SlrConverter(prog).hugr()

    def test_inline_creg_assigned_in_static_for_propagates(self) -> None:
        """Static For with trip>=1 propagates bit assignment."""
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 1),
            For("i", 0, 2).Do(qb.X(q[0]), Measure(q[0]) > inline[0], qb.Prep(q[0])),
            Print(inline[0], tag="post_for"),
        )
        SlrConverter(prog).hugr()

    def test_print_inline_bit_with_only_other_bit_measured_rejected(self) -> None:
        """Bit-level soundness gap: Measure(inline[0]); Print(inline[1])
        should be rejected, not silently emit a runtime out-of-bounds read.

        Before bit-level tracking the validator marked the whole inline CReg as
        "assigned" after any bit was measured, so Print(inline[1]) compiled and
        Selene panicked at runtime. Bit-level tracking + the inferred-size
        bound make this a clean construct-time rejection.
        """
        inline = CReg("inline", 2)
        prog = Main(
            q := QReg("q", 2),
            qb.X(q[0]),
            Measure(q[0]) > inline[0],
            Print(inline[1], tag="bit1"),
        )
        with pytest.raises(GuppyCodegenError, match=r"references inline CReg"):
            SlrConverter(prog).hugr()

    def test_whole_inline_creg_print_rejected_outright(self) -> None:
        """Phase 1 rejects whole-CReg Print of an inline CReg unconditionally.

        Even when every inferred bit has been Measure-assigned, whole-CReg
        Print of an inline CReg silently shrinks the register relative to the
        user's CReg(name, size) intent -- inline-from-Measure inference only
        sees Measure-targeted indices, so the inferred RegisterDecl.size can
        be smaller than what the user wrote. Phase 1 fail-fast: require
        explicit Main(...) declaration or per-bit Print.
        """
        inline = CReg("inline", 2)
        prog = Main(
            q := QReg("q", 2),
            qb.X(q[0]),
            Measure(q[0]) > inline[0],
            Measure(q[1]) > inline[1],
            Print(inline, tag="full"),
        )
        with pytest.raises(GuppyCodegenError, match=r"Print\(whole-CReg\) of inline CReg"):
            SlrConverter(prog).hugr()

    def test_whole_inline_creg_print_shrink_case_rejected(self) -> None:
        """Regression for the tracer-bullet shrink case: whole-CReg Print of an
        inline CReg silently emitting a register smaller than the user declared.

        Before this fix, `inline = CReg("inline", 2)` with only inline[0]
        measured + `Print(inline)` would compile and emit `result("result.whole",
        inline)` against a size-1 inferred register (max_measured_index + 1),
        silently losing the user's stated size 2.
        """
        inline = CReg("inline", 2)
        prog = Main(
            q := QReg("q", 2),
            qb.X(q[0]),
            Measure(q[0]) > inline[0],
            Print(inline, tag="whole"),
        )
        with pytest.raises(GuppyCodegenError, match=r"Print\(whole-CReg\) of inline CReg"):
            SlrConverter(prog).hugr()

    def test_whole_declared_creg_print_accepted(self) -> None:
        """When the CReg is declared in Main(...), it's no longer inline and
        whole-CReg Print is allowed (user-acknowledged size).
        """
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),  # declared positional -> not inline
            qb.X(q[0]),
            qb.X(q[1]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Print(c, tag="whole"),
        )
        SlrConverter(prog).hugr()


# ── Additional path-signature edge cases ───────────────────────────────


class TestPathSignatureEdgeCases:
    """Coverage called out as missing in f2ebb32c."""

    def test_while_with_print_rejected(self) -> None:
        """While body with Print is rejected (v1 also rejects While outright)."""
        from pecos.slr import While

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            While(c[0] == 0).Do(
                Print(c, tag="loop"),
                qb.Prep(q[0]),
                Measure(q[0]) > c[0],
            ),
        )
        with pytest.raises(GuppyCodegenError):
            SlrConverter(prog).hugr()

    def test_parallel_block_with_print_compiles(self) -> None:
        """`Parallel(Print(c[0]), Print(c[1]))` compiles -- Parallel is sequential."""
        from pecos.slr import Parallel

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            qb.X(q[1]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Parallel(
                Print(c[0], tag="bit0"),
                Print(c[1], tag="bit1"),
            ),
        )
        SlrConverter(prog).hugr()

    def test_repeat_zero_with_print_walks_body_for_validation(self) -> None:
        """Repeat(0) body never runs at runtime, but invalid Print constructs
        inside it are still rejected at validation time. Assignments inside do
        not propagate to the outer scope.
        """
        inline = CReg("inline", 1)
        prog = Main(
            q := QReg("q", 1),
            Repeat(0).block(
                Print(inline[0], tag="never_runs"),  # inline[0] never written anywhere
            ),
            Measure(q[0]) > inline[0],
        )
        with pytest.raises(GuppyCodegenError, match=r"references inline CReg"):
            SlrConverter(prog).hugr()

    def test_non_static_for_with_print_via_ast_rejected(self) -> None:
        """Direct AST-level test for non-static For body containing Print.

        v1's public SLR API doesn't allow non-static For bounds (start/stop
        must be int literals), so this case isn't reachable via Main/Print
        construction. Build the AST directly to pin the validator's defensive
        rejection in `_static_for_trip_count`.
        """
        from pecos.slr.ast import (
            AllocatorDecl,
            ForStmt,
            LiteralExpr,
            PrintOp,
            Program,
            VarExpr,
        )
        from pecos.slr.ast.codegen.guppy import AstToGuppy

        # main(q: array[qubit, 1] @ owned):
        #   for i in range(x, 3):     # start is a VarExpr (non-literal) -> non-static
        #       Print(c, tag="loop")  # rejected: Print inside non-static For
        prog = Program(
            name="main",
            declarations=(AllocatorDecl(name="q", capacity=1, parent=None),),
            body=(
                ForStmt(
                    variable="i",
                    start=VarExpr(name="x"),
                    stop=LiteralExpr(value=3),
                    body=(PrintOp(value="c", tag="loop", namespace="result"),),
                ),
            ),
        )

        emitter = AstToGuppy()
        with pytest.raises(GuppyCodegenError, match=r"non-static `For`"):
            emitter.generate(prog)


# ── Cross-codegen byte-identity (v2-blockcall-resource-effects) ─────────


class TestCrossCodegenPrintEmission:
    """Pin the behavior of each non-Guppy codegen when Print is inserted.

    Empirically probed 2026-05-14 against a Bell-state program with and without
    a trailing `Print(c, tag="debug")`:

    - **QASM**: emits `// Print result.debug c` as a comment line; output is
      *not* byte-identical (intentional, per the doc's "comment-only across
      non-Guppy" plan).
    - **QIR**, **Stim**, **QuantumCircuit**: silently skip PrintOp via their
      isinstance-dispatch fallback (no else clause). Output is byte-identical.

    These tests pin both behaviors. If a future change starts emitting in
    QIR/Stim/QC, this catches it. If QASM stops emitting the comment, this
    catches it.
    """

    @staticmethod
    def _bell_no_print() -> Main:
        return Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            Measure(q) > c,
        )

    @staticmethod
    def _bell_with_print() -> Main:
        return Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.H(q[0]),
            qb.CX(q[0], q[1]),
            Measure(q) > c,
            Print(c, tag="debug"),
        )

    def test_qasm_emits_print_as_exactly_one_comment_line(self) -> None:
        """QASM adds exactly one `// Print result.debug c` line; otherwise identical.

        "in `added_lines`" is too loose -- a future
        regression could add extra lines silently and the test would still pass.
        Assert the added line is exactly `[expected_comment]` (one line, no more).
        """
        a = SlrConverter(self._bell_no_print()).qasm()
        b = SlrConverter(self._bell_with_print()).qasm()
        assert a != b, "expected QASM to emit a Print comment, not silently skip"
        expected_added = "// Print result.debug c"
        # Set-diff the line lists to detect ALL additions, not just whether
        # the expected line is somewhere among them.
        added_lines = [line for line in b.splitlines() if line not in a.splitlines()]
        assert added_lines == [expected_added], f"expected exactly [{expected_added!r}] added lines, got {added_lines}"

    def test_qir_byte_identical(self) -> None:
        """QIR silently drops Print; pre/post output is byte-identical."""
        a = SlrConverter(self._bell_no_print()).qir()
        b = SlrConverter(self._bell_with_print()).qir()
        assert a == b, "QIR Print emission must be a silent skip (Phase 1 policy)"

    def test_stim_byte_identical(self) -> None:
        a = str(SlrConverter(self._bell_no_print()).stim())
        b = str(SlrConverter(self._bell_with_print()).stim())
        assert a == b, "Stim Print emission must be a silent skip (Phase 1 policy)"

    def test_quantum_circuit_byte_identical(self) -> None:
        a = str(SlrConverter(self._bell_no_print()).quantum_circuit())
        b = str(SlrConverter(self._bell_with_print()).quantum_circuit())
        assert a == b, "QuantumCircuit Print emission must be a silent skip (Phase 1 policy)"


# ── Selene shape edge cases (probed 2026-05-14) ──────────────────────────


class TestSeleneShapeEdgeCases:
    """Pin non-obvious Selene `to_dict()` shapes that were easy to get wrong.

    These tests use empirically-probed reference shapes. If Selene's
    representation changes, the test fails loudly and the doc text in
    `v2-print.md` needs updating.
    """

    def test_multibit_creg_print_in_repeat_flattens_iterations_into_inner_list(self) -> None:
        """Selene flattens iteration x bit into a single inner list per shot.

        Empirical shape for 2-shot x 2-iter x 2-bit:
            {'result.iter': [[1, 0, 1, 0], [1, 0, 1, 0]]}
        NOT nested per-iteration like `[[[1,0],[1,0]], [[1,0],[1,0]]]`.

        Pinning matters: a future Selene change could nest by iteration, which
        would break user code that consumes the flat layout.
        """
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            Repeat(2).block(
                qb.X(q[0]),
                Measure(q[0]) > c[0],
                qb.Prep(q[0]),
                Print(c, tag="iter"),
            ),
        )
        raw = _run_and_get_result_dict(prog, shots=3, qubits=2)
        assert "result.iter" in raw
        events = raw["result.iter"]
        assert len(events) == 3, f"expected 3 per-shot lists, got {len(events)}"
        # Each shot: 2 iterations x 2 bits = 4 ints flat.
        # X on q[0] each iter -> c[0] = 1; c[1] never measured -> 0.
        # Iteration order: [c[0]_iter0, c[1]_iter0, c[0]_iter1, c[1]_iter1].
        expected_per_shot = [1, 0, 1, 0]
        for shot_events in events:
            assert list(shot_events) == expected_per_shot, shot_events

    def test_empty_main_returns_empty_dict_no_measurements(self) -> None:
        """Empty Main() compiles to a no-op program; Selene to_dict() is empty."""
        from pecos import Hugr, selene_engine, sim

        prog = Main()
        package = SlrConverter(prog).hugr()
        hugr_bytes = package.to_str().encode("utf-8")
        result = sim(Hugr(hugr_bytes)).classical(selene_engine()).qubits(1).seed(42).run(2)
        raw = result.to_dict() if hasattr(result, "to_dict") else result
        assert raw == {}, f"expected empty result dict for empty Main(), got {raw}"

    def test_single_print_single_bit_creg_flat_per_shot_list(self) -> None:
        """Selene shape: ONE Print of a single-bit declared CReg → flat int per shot.

        Empirical shape: `{'result.zero_init': [0, 0]}` for 2 shots.

        NOT `[[0], [0]]` -- when there's exactly one event per shot and the
        event is a single bit, Selene flattens to a plain int. Pin this so
        users writing `result.tag[shot]` get a stable contract.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Print(c, tag="zero_init"),  # before Measure: bit is the False init
            Measure(q[0]) > c[0],
        )
        raw = _run_and_get_result_dict(prog, shots=2)
        assert "result.zero_init" in raw
        events = raw["result.zero_init"]
        assert len(events) == 2
        # Each shot is a bare int, not a list.
        for shot_value in events:
            assert isinstance(shot_value, int), f"expected int per shot, got {type(shot_value).__name__}"
            assert shot_value == 0  # Print before Measure -> auto-init False

    def test_single_print_multibit_creg_list_per_shot(self) -> None:
        """Selene shape: ONE Print of a multi-bit declared CReg → list of bits per shot.

        Empirical: `{'result.tag': [[1, 0], [1, 0]]}` for 2 shots x 2-bit CReg.
        Contrast with the single-bit case above which returns a flat int per shot.
        """
        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 2),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Measure(q[1]) > c[1],
            Print(c, tag="pair"),
        )
        raw = _run_and_get_result_dict(prog, shots=2, qubits=2)
        assert "result.pair" in raw
        events = raw["result.pair"]
        assert len(events) == 2
        for shot_value in events:
            # Multi-bit: list of 2 ints per shot.
            assert hasattr(shot_value, "__len__"), f"expected list per shot, got {type(shot_value).__name__}"
            assert len(shot_value) == 2
            # X on q[0] -> c[0]=1; c[1]=0.
            assert int(shot_value[0]) == 1
            assert int(shot_value[1]) == 0
