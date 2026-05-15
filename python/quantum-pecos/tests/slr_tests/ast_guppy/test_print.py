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

    def test_print_in_repeat_emits_event_list(self) -> None:
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
        raw = _run_and_get_result_dict(prog, shots=5)
        # tag emitted thrice per shot; Selene returns event-list under the key
        assert "result.iter" in raw, f"expected 'result.iter' in {list(raw.keys())}"

    def test_print_in_static_for_emits_event_list(self) -> None:
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            For("i", 0, 2).Do(
                qb.X(q[0]),
                Measure(q[0]) > c[0],
                qb.Prep(q[0]),
                Print(c, tag="loop"),
            ),
        )
        raw = _run_and_get_result_dict(prog, shots=5)
        assert "result.loop" in raw


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


# ── Print does not change program return shape ──────────────────────────


class TestPrintScopeOrthogonal:
    """Print is a side-effect; it does not allocate, does not affect main's return type."""

    def test_print_does_not_break_implicit_return(self) -> None:
        """v1 implicit return of result CRegs still works alongside Print."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Print(c),
        )
        raw = _run_and_get_result_dict(prog)
        # Both Print's tag and the implicit return's tag should be present.
        assert "result.c" in raw

    def test_multiple_prints_same_tag_become_event_list(self) -> None:
        """Two Print(c) calls in the same body emit the tag twice per shot."""
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
        raw = _run_and_get_result_dict(prog)
        assert "result.early" in raw
        assert "result.late" in raw
