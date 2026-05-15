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
from pecos.slr import CReg, For, Main, Print, QReg, Repeat, SlrConverter
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
