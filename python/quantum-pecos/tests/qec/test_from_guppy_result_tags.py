# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Guppy tags bind to lowered runtime measurements, including inlined helpers.

Asymmetric noise histories make incorrect tag bindings change the DEM string.
"""

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.lang import owned
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import Detector, DetectorErrorModel, build_dem_from_guppy, rec, result_ref


# A scrambled straight-line program: three measurements in source order
# qa, qb, qc; ``result()`` is called for them in reverse-then-mixed order
# (c, a, b). The QIS analyzer binds tag_a -> [0], tag_b -> [1],
# tag_c -> [2] (ordinals of the measurements the tags actually record, not
# the order of the result() calls).
#
# The nontrivial pre-history entangles qb and qc after an H on qb. With
# p1/p2 > 0 this produces distinct mechanisms for detectors anchored to qa,
# qb, and qc, so the DEMs for
# detectors anchored to records [-3], [-2], [-1] differ in their (number of)
# mechanisms / probabilities. A test asserting result_tags equals positional
# records is then load-bearing: a wrong ordinal mapping would produce a
# different DEM string, not coincidentally equal as it does for symmetric
# programs. The test asserts up front that the three positional DEMs differ
# (so a future-symmetric refactor self-fails rather than silently passing).
@guppy
def _scrambled_three_measurements() -> None:
    qa = qubit()
    qb = qubit()
    qc = qubit()
    h(qb)
    cx(qb, qc)
    a = measure(qa).read()
    b = measure(qb).read()
    c = measure(qc).read()
    result("tag_c", c)
    result("tag_a", a)
    result("tag_b", b)


_NOISE = {"p1": 0.01, "p2": 0.02, "p_meas": 0.1, "p_prep": 0.005}


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

# Three measurements (a, b, c) in trace order have record offsets (-3, -2, -1).


def test_result_tags_match_positional_records() -> None:
    """Each tag matches its positional DEM, with distinguishable noise histories."""
    via_records = {
        -3: _from_guppy('[{"id":0,"records":[-3]}]'),
        -2: _from_guppy('[{"id":0,"records":[-2]}]'),
        -1: _from_guppy('[{"id":0,"records":[-1]}]'),
    }
    # Sanity: the three reference DEMs must differ -- otherwise the test
    # is symmetric and a wrong ordinal mapping would pass spuriously.
    assert via_records[-3] != via_records[-2] != via_records[-1] != via_records[-3], (
        "scrambled program is DEM-symmetric across the three measurements; the "
        "correspondence test is no longer load-bearing. Restore asymmetric gates."
    )

    # Now the substantive claim: each tag's DEM matches the positional one.
    assert _from_guppy('[{"id":0,"result_tags":["tag_a"]}]') == via_records[-3]
    assert _from_guppy('[{"id":0,"result_tags":["tag_b"]}]') == via_records[-2]
    assert _from_guppy('[{"id":0,"result_tags":["tag_c"]}]') == via_records[-1]


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
    """Measurements in runtime loops cannot be certified from one trace."""
    with pytest.raises(
        ValueError,
        match=(
            r"^Cannot certify measurement order from the lowered program: "
            r"order-relevant block %\S+ is on a CFG cycle$"
        ),
    ):
        DetectorErrorModel.from_guppy(
            make_surface_code(distance=3, num_rounds=3, basis="Z"),
            num_qubits=get_num_qubits(3),
            detectors_json='[{"id":0,"result_tags":["any_tag"]}]',
            **_NOISE,
        )


# ---------------------------------------------------------------------------
# 3. Non-Guppy callable rejection: result_tags requires a HUGR-compilable input
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


# ---------------------------------------------------------------------------
# 5. result_tags + records: strict + redundancy-checked, not additive
# ---------------------------------------------------------------------------


def test_result_tags_with_redundant_records_builds_unchanged() -> None:
    """When ``records`` exactly matches the resolved ``result_tags`` offsets
    (sorted-set equality), the entry is accepted and the DEM equals the
    records-only equivalent (no double-reference / parity corruption)."""
    # tag_a resolves to record -3 in the asymmetric three-measurement program.
    with_both = _from_guppy('[{"id":0,"records":[-3],"result_tags":["tag_a"]}]')
    records_only = _from_guppy('[{"id":0,"records":[-3]}]')
    assert with_both == records_only


def test_result_tags_non_redundant_with_records_fails_loud() -> None:
    """When ``records`` and ``result_tags`` reference *different* measurements
    they are not redundant; the rewriter must fail loud rather than silently
    combine them additively (which would either weaken the DEM or, on
    accidental duplicate, XOR-cancel the detector's mechanisms)."""
    # tag_c resolves to -1; records=-3 is a different measurement.
    with pytest.raises(ValueError, match=r"alternatives|not additive|different measurements"):
        _from_guppy('[{"id":0,"records":[-3],"result_tags":["tag_c"]}]')


def test_result_tags_with_malformed_records_fails_loud() -> None:
    """The rewriter must strict-parse existing ``records`` -- not silently
    drop malformed entries via filter_map (a regression a previous review
    caught). Both a non-integer entry and a non-array shape must fail loud."""
    with pytest.raises(ValueError, match=r"records entries must be integers"):
        _from_guppy('[{"id":0,"records":["bad"],"result_tags":["tag_a"]}]')
    with pytest.raises(ValueError, match=r"records must be a JSON array"):
        _from_guppy('[{"id":0,"records":-3,"result_tags":["tag_a"]}]')


@guppy
def _measure_owned(q: qubit @ owned) -> bool:
    return measure(q).read()


@guppy
def _inlined_three_measurements() -> None:
    q0 = qubit()
    q1 = qubit()
    q2 = qubit()
    h(q1)
    cx(q1, q2)
    x(q2)
    x(q2)
    b = _measure_owned(q1)
    a = measure(q0).read()
    c = measure(q2).read()
    result("a", a)
    result("b", b)
    result("c", c)


def test_inlined_result_tags_match_positional_records() -> None:
    def build(field: str, value: str) -> str:
        return DetectorErrorModel.from_guppy(
            _inlined_three_measurements,
            num_qubits=3,
            detectors_json=f'[{{"id":0,"{field}":[{value}]}}]',
            seed=0,
            **_NOISE,
        ).to_string()

    positional = {offset: build("records", str(offset)) for offset in (-3, -2, -1)}
    assert len(set(positional.values())) == 3
    for tag, offset in (("b", -3), ("a", -2), ("c", -1)):
        assert build("result_tags", f'"{tag}"') == positional[offset], tag


def test_inlined_typed_result_refs_match_positional_records() -> None:
    positional = {
        offset: build_dem_from_guppy(
            _inlined_three_measurements,
            num_qubits=3,
            detectors=[Detector(rec[offset])],
            seed=0,
            **_NOISE,
        )
        for offset in (-3, -2, -1)
    }
    assert len({build.dem.to_string() for build in positional.values()}) == 3
    for tag, offset in (("b", -3), ("a", -2), ("c", -1)):
        tagged = build_dem_from_guppy(
            _inlined_three_measurements,
            num_qubits=3,
            detectors=[Detector(result_ref(tag))],
            seed=0,
            **_NOISE,
        )
        assert tagged.detectors_json == positional[offset].detectors_json, tag
        assert tagged.dem.to_string() == positional[offset].dem.to_string(), tag
