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

"""Phase 2 deprecation warnings for v2 SLR output-model redesign.

v2's breaking transition (Phase 3b in
`~/Repos/pecos-docs/design/slr/v2-breaking-migration.md`) will:

- Remove the `result=` kwarg from `CReg`.
- Make `Return(...)` mandatory for any exposed output; drop the implicit
  return-of-all-result-CRegs rule.

Phase 2 (this commit) gives users one release of warning before the breaking
flip. Both warnings are `DeprecationWarning` so CI surfaces them under
`-W error::DeprecationWarning` without breaking default usage.
"""

from __future__ import annotations

import warnings

import pytest
from pecos.slr import CReg, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure


class TestCRegResultKwargDeprecation:
    """`CReg(..., result=False)` warns at construction time."""

    def test_creg_with_result_false_emits_deprecation_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match=r"`CReg\(\.\.\., result=False\)`"):
            CReg("c", 1, result=False)

    def test_creg_with_default_result_does_not_warn(self) -> None:
        """The default (result=True) is the legacy behavior; don't warn on it.

        That avoids spamming every existing SLR program with deprecation
        noise. Programs that explicitly opt into result=False get the warning.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn any warning into a test failure
            CReg("c", 1)

    def test_creg_with_explicit_result_true_does_not_warn(self) -> None:
        """Explicit result=True is also the default behavior; not deprecated."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CReg("c", 1, result=True)

    def test_warning_points_at_user_call_site(self) -> None:
        """The DeprecationWarning's stacklevel should attribute to the user code,
        not to CReg.__init__ itself. Use stacklevel=2 so the warning's filename
        matches this test file.
        """
        with pytest.warns(DeprecationWarning, match=r"`CReg\(\.\.\., result=False\)`") as record:
            CReg("c", 1, result=False)
        assert len(record) == 1
        # The recorded warning's filename should be this test file.
        assert "test_deprecation_warnings.py" in record[0].filename


class TestImplicitReturnDeprecation:
    """Programs with measurements but no explicit `Return(...)` warn at
    `.guppy()` / `.hugr()` time.
    """

    def test_implicit_return_warns_on_program_with_measure_no_return(self) -> None:
        """Classic v1 pattern: Measure into result CReg, no explicit Return.
        v2 will require Return; warn now.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )
        with pytest.warns(DeprecationWarning, match=r"Implicit return"):
            SlrConverter(prog).guppy()

    def test_explicit_return_silences_warning(self) -> None:
        """With explicit Return, no deprecation warning fires."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            Return(c),
        )
        with warnings.catch_warnings():
            # Promote DeprecationWarning to error so any fire = test failure.
            warnings.simplefilter("error", DeprecationWarning)
            SlrConverter(prog).guppy()

    def test_no_measurement_does_not_warn(self) -> None:
        """A program with no Measure has nothing to implicitly return.

        Don't pester users with the implicit-return warning when there's
        no implicit-return behavior to deprecate.
        """
        prog = Main(
            q := QReg("q", 1),
            qb.H(q[0]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            SlrConverter(prog).guppy()

    def test_empty_main_does_not_warn(self) -> None:
        prog = Main()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            SlrConverter(prog).guppy()

    def test_measure_inside_repeat_triggers_warning(self) -> None:
        """The walker descends into Repeat / For / While / Parallel / If bodies."""
        from pecos.slr import Repeat

        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            Repeat(2).block(qb.X(q[0]), Measure(q[0]) > c[0], qb.Prep(q[0])),
        )
        with pytest.warns(DeprecationWarning, match=r"Implicit return"):
            SlrConverter(prog).guppy()

    def test_measure_inside_if_triggers_warning(self) -> None:
        from pecos.slr import If

        prog = Main(
            q := QReg("q", 2),
            c := CReg("c", 1),
            d := CReg("d", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
            If(c[0]).Then(Measure(q[1]) > d[0]).Else(Measure(q[1]) > d[0]),
        )
        with pytest.warns(DeprecationWarning, match=r"Implicit return"):
            SlrConverter(prog).guppy()

    def test_warning_does_not_block_compilation(self) -> None:
        """The deprecation warning is non-blocking; .guppy() and .hugr() still
        succeed. Only Phase 3b makes the breaking change.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            source = SlrConverter(prog).guppy()
            assert "result.c" not in source  # no Print emitted; only return
            package = SlrConverter(prog).hugr()
            assert package is not None

    def test_warning_points_at_user_call_site(self) -> None:
        """Stacklevel must attribute to the user's .guppy()/.hugr()/etc. call site,
        not to internals of slr_converter.py. Empirically verified:
        stacklevel=3 from `_maybe_warn_implicit_return_deprecation`
        walks past the warn line, past the helper, past the public method,
        landing on the user.
        """
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )
        with pytest.warns(DeprecationWarning, match=r"Implicit return") as record:
            SlrConverter(prog).guppy()
        assert len(record) == 1
        assert (
            "test_deprecation_warnings.py" in record[0].filename
        ), f"expected warning attributed to this test file; got {record[0].filename}"

    def test_warning_fires_only_once_per_converter(self) -> None:
        """Cached on the SlrConverter instance: multiple codegen calls share one warning."""
        prog = Main(
            q := QReg("q", 1),
            c := CReg("c", 1),
            qb.X(q[0]),
            Measure(q[0]) > c[0],
        )
        converter = SlrConverter(prog)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            converter.guppy()
            converter.qasm()
            converter.hugr()
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        # Exactly one implicit-return warning across the three codegen calls.
        implicit_return_warnings = [w for w in deprecation if "Implicit return" in str(w.message)]
        assert (
            len(implicit_return_warnings) == 1
        ), f"expected 1 implicit-return warning across 3 codegen calls, got {len(implicit_return_warnings)}"

    def test_warning_fires_for_qasm_qir_stim_quantum_circuit_too(self) -> None:
        """Phase 2 deprecation is SLR-wide (not Guppy/HUGR-only): every public
        codegen entry point on SlrConverter fires the warning, because the
        v2 breaking change is an SLR-API change downstream consumers all see.
        """
        for entry in ("qasm", "guppy", "hugr", "qir", "stim", "quantum_circuit"):
            prog = Main(
                q := QReg("q", 1),
                c := CReg("c", 1),
                qb.X(q[0]),
                Measure(q[0]) > c[0],
            )
            with pytest.warns(DeprecationWarning, match=r"Implicit return"):
                getattr(SlrConverter(prog), entry)()

    def test_measure_with_no_result_does_not_warn(self) -> None:
        """`Measure(q[0])` without a `> c[i]` results tuple writes to no CReg,
        so the implicit-return rule is not engaged. Don't warn.

        Tracer-bullet regression: the original walker treated *any*
        MeasureOp as evidence of implicit-return reliance, which was a
        false positive for measure-and-discard patterns.
        """
        prog = Main(
            q := QReg("q", 1),
            qb.X(q[0]),
            Measure(q[0]),  # discard the result
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            SlrConverter(prog).guppy()

    def test_measure_into_only_result_false_creg_does_not_warn(self) -> None:
        """When the only Measure target is a declared `result=False` CReg,
        nothing flows through the implicit-return rule. Don't warn.

        (The CReg(result=False) construction itself emits its own deprecation;
        we suppress that here so the test focuses on the implicit-return check.)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            scratch = CReg("scratch", 1, result=False)

        prog = Main(
            q := QReg("q", 1),
            scratch,  # declared positional, result=False -> opt-out
            qb.X(q[0]),
            Measure(q[0]) > scratch[0],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            SlrConverter(prog).guppy()
        # No implicit-return warning should fire.
        implicit_return_warnings = [w for w in caught if "Implicit return" in str(w.message)]
        assert len(implicit_return_warnings) == 0, (
            "result=False-only Measure should not trigger implicit-return warning; "
            f"got {[str(w.message) for w in implicit_return_warnings]}"
        )

    def test_declared_result_creg_without_measure_still_warns(self) -> None:
        """Declared `is_result=True` CRegs are implicitly returned even if never
        measured (they auto-init to all-False). v2 will require Return; warn.

        This is a positive case where the *Measure-presence* alone is not the
        trigger -- the trigger is "any is_result=True CReg + no explicit Return".
        """
        prog = Main(
            q := QReg("q", 1),
            CReg("c", 1),  # declared, default is_result=True; not bound (just registered in Main.vars)
            qb.X(q[0]),  # no Measure
        )
        with pytest.warns(DeprecationWarning, match=r"Implicit return"):
            SlrConverter(prog).guppy()
