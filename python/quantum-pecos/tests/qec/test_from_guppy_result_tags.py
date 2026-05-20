# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for tag-referenced detectors in ``DetectorErrorModel.from_guppy``.

Covers:

1. **Correspondence cross-check (load-bearing):** for a scrambled straight-line
   Guppy program where ``result()`` calls are declared in non-source order, a
   DEM built via ``result_tags`` (which goes through
   ``pecos_hugr_qis::extract_result_tag_measurements`` to recover the
   reorder-immune tag -> measurement binding from the compiled HUGR) is
   **byte-identical** to the DEM built via the equivalent positional
   ``records``. This is the committed verification of the
   HUGR-traversal-ordinal == traced-``MeasId``-order property the prior
   review (proposal 001 item #7) flagged as unproven.
2. **Runtime-loop guard:** a Guppy program with a runtime loop (the surface
   code) using ``result_tags`` fails loud with the documented "static N vs
   traced M" message, instead of silently misbinding.
3. **Wrapper-input rejection:** ``result_tags`` requires a ``@guppy``-decorated
   function -- a compiled program (e.g. the object ``make_surface_code``
   returns) is rejected fail-loud upfront, not crashed inside the HUGR
   compile step.
4. **Unknown-tag rejection:** referencing a tag the program never records is
   an error.
"""

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit
from pecos.guppy import get_num_qubits, make_surface_code
from pecos.qec import DetectorErrorModel


# A scrambled straight-line program: three measurements in source order
# qa, qb, qc; but ``result()`` is called for them in reverse-then-mixed order
# (c, a, b). The HUGR-side extractor binds tag_a -> [0], tag_b -> [1],
# tag_c -> [2] (ordinals of the measurements the tags actually record, not the
# order of the result() calls). If the HUGR traversal ordinal of the i-th
# measurement op equals the traced MeasId order position of the same
# measurement (the property under test), then result_tags resolution produces
# the same DEM as the obvious positional one.
@guppy
def _scrambled_three_measurements() -> None:
    qa = qubit()
    qb = qubit()
    qc = qubit()
    a = measure(qa)
    b = measure(qb)
    c = measure(qc)
    result("tag_c", c)
    result("tag_a", a)
    result("tag_b", b)


_NOISE = {"p1": 0.0, "p2": 0.0, "p_meas": 0.1, "p_prep": 0.0}


def _from_guppy(detectors_json: str, *, observables_json: str = "[]") -> str:
    """Build the scrambled-program DEM with the given metadata and return it as a string."""
    dem = DetectorErrorModel.from_guppy(
        _scrambled_three_measurements,
        num_qubits=3,
        detectors_json=detectors_json,
        observables_json=observables_json,
        seed=0,
        **_NOISE,
    )
    return dem.to_string()


# ---------------------------------------------------------------------------
# 1. Correspondence: result_tags DEM == positional-records DEM (byte-identical)
# ---------------------------------------------------------------------------

# Three measurements (a, b, c) in trace order; record offsets are
# (a -> -3, b -> -2, c -> -1) under the Stim convention. If
# HUGR-traversal-ordinal == traced-MeasId-order, then result_tags=["tag_X"]
# resolves to the same record offset as the positional form for tag X.


@pytest.mark.parametrize(
    ("tag", "record"),
    [("tag_a", -3), ("tag_b", -2), ("tag_c", -1)],
)
def test_result_tags_match_positional_records(tag: str, record: int) -> None:
    """Each tag resolves to the same DEM as its positional-record equivalent."""
    via_tags = _from_guppy(f'[{{"id":0,"result_tags":["{tag}"]}}]')
    via_records = _from_guppy(f'[{{"id":0,"records":[{record}]}}]')
    assert via_tags == via_records, (
        f"result_tags={tag!r} should resolve to records={record}; HUGR-ordinal != traced-MeasId-order for this case"
    )


def test_result_tags_multi_tag_detector_matches_positional() -> None:
    """A detector referencing multiple tags resolves to the same DEM as
    the positional equivalent (asserts the property for combined refs too)."""
    via_tags = _from_guppy('[{"id":0,"result_tags":["tag_a","tag_c"]}]')
    via_records = _from_guppy('[{"id":0,"records":[-3,-1]}]')
    assert via_tags == via_records


def test_result_tags_observables_path_matches_positional() -> None:
    """The observables_json path resolves result_tags identically."""
    via_tags = _from_guppy(
        "[]",
        observables_json='[{"id":0,"result_tags":["tag_b"]}]',
    )
    via_records = _from_guppy(
        "[]",
        observables_json='[{"id":0,"records":[-2]}]',
    )
    assert via_tags == via_records


# ---------------------------------------------------------------------------
# 2. Runtime-loop guard: surface code fails loud, not silent
# ---------------------------------------------------------------------------


def test_result_tags_with_runtime_loop_program_fails_loud() -> None:
    """The surface code uses ``for _ in range(comptime(n))`` rounds; the HUGR
    has one static measure op per loop body, not per occurrence. The Rust
    static-vs-traced count guard rejects this case rather than silently
    misbinding (per-occurrence tag binding requires CFG-interpreter-class
    machinery; see proposal 001)."""
    with pytest.raises(ValueError, match=r"runtime loops|not supported"):
        DetectorErrorModel.from_guppy(
            make_surface_code(distance=3, num_rounds=3, basis="Z"),
            num_qubits=get_num_qubits(3),
            detectors_json='[{"id":0,"result_tags":["any_tag"]}]',
            **_NOISE,
        )


# ---------------------------------------------------------------------------
# 3. Wrapper-input rejection: result_tags requires @guppy directly
# ---------------------------------------------------------------------------


def test_result_tags_with_non_guppy_callable_fails_loud_upfront() -> None:
    """``result_tags`` requires a ``@guppy``-decorated function (or a
    ``GuppyFunctionDefinition`` such as ``make_surface_code`` returns).
    A plain Python callable cannot be compiled to a HUGR; ``from_guppy``
    must reject it upfront with the clear ``@guppy`` message instead of
    crashing later inside the HUGR compile step (the upfront guard the
    review flagged as needed)."""

    def not_a_guppy_function() -> None:
        pass

    with pytest.raises(ValueError, match=r"@guppy-decorated function"):
        DetectorErrorModel.from_guppy(
            not_a_guppy_function,
            num_qubits=1,
            detectors_json='[{"id":0,"result_tags":["any_tag"]}]',
            **_NOISE,
        )


# ---------------------------------------------------------------------------
# 4. Unknown-tag rejection
# ---------------------------------------------------------------------------


def test_result_tags_unknown_tag_fails_loud() -> None:
    with pytest.raises(ValueError, match=r"never records|result_tag"):
        _from_guppy('[{"id":0,"result_tags":["nonexistent_tag"]}]')
