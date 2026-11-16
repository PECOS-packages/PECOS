"""Order-preserving compaction checked against exhaustive contiguous partitions."""

from dataclasses import replace
from itertools import product

import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import MeasurementParity, OperationOrigin, ScheduledCircuit, SchedulePolicy
from pecos.qec.generic.guppy_comparison import _rescheduling_contract


def _plan(operations):
    return ScheduledCircuit(
        3,
        (tuple(Op("R", (q,)) for q in range(3)), *((op,) for op in operations), tuple(Op("M", (q,)) for q in range(3))),
        (),
        (),
    )


def _listing(plan):
    return [(op, op.origin) for moment in plan.moments for op in moment]


def _minimum_contiguous_partition(operations):
    # Independent oracle: enumerate every possible boundary set. No greedy
    # placement or per-wire depth calculation from the implementation is used.
    best = len(operations)
    for boundaries in product((False, True), repeat=len(operations) - 1):
        groups = [[]]
        for i, op in enumerate(operations):
            if i and boundaries[i - 1]:
                groups.append([])
            groups[-1].extend(op.qubits)
        if all(len(set(g)) == len(g) for g in groups):
            best = min(best, len(groups))
    return best


def test_compaction_matches_exhaustive_ordered_partition_oracle():
    vocabulary = (Op("H", (0,)), Op("X", (1,)), Op("Z", (2,)), Op("CX", (0, 1)))
    for sequence in product(vocabulary, repeat=5):
        plan = _plan(sequence)
        compact = plan.rescheduled(SchedulePolicy.COMPACT_UNITARIES)
        assert len(compact.moments) - 2 == _minimum_contiguous_partition(sequence)
        assert _listing(compact) == _listing(plan)
        assert _rescheduling_contract(compact) == _rescheduling_contract(plan)
        assert len(compact.moments) <= len(plan.moments)


def test_asap_overtakes_while_compaction_preserves_order_and_origins():
    operations = (Op("H", (0,), OperationOrigin("first", "H", 0, 0)), Op("X", (0,)), Op("H", (1,)))
    plan = _plan(operations)
    asap = plan.rescheduled(SchedulePolicy.ASAP_UNITARIES)
    compact = plan.rescheduled(SchedulePolicy.COMPACT_UNITARIES)
    assert asap.moments[1:-1] == ((operations[0], operations[2]), (operations[1],))
    assert compact.moments[1:-1] == ((operations[0],), (operations[1], operations[2]))
    assert _listing(compact) == _listing(plan)
    assert compact.moments[1][0] is plan.moments[1][0]
    assert all("preserve source order" in row.placement for row in compact.inspect())
    assert compact.rescheduled(SchedulePolicy.COMPACT_UNITARIES) is compact
    with pytest.raises(ValueError, match="cannot be recovered"):
        compact.rescheduled(SchedulePolicy.REFERENCE)


def test_boundaries_mixed_moments_empty_regions_relabel_and_reuse_are_retained():
    plan = ScheduledCircuit(
        2,
        (
            (),
            (Op("R", (0,)), Op("R", (1,))),
            (),
            (Op("H", (0,)),),
            (Op("X", (1,)),),
            (Op("RELABEL", (0, 1)),),
            (Op("M", (0,)), Op("H", (1,))),
            (Op("R", (0,)), Op("X", (1,))),
            (),
            (Op("M", (1,)), Op("M", (0,))),
            (),
        ),
        (MeasurementParity((0, 2), "affine", True, (1.0, 2.0)),),
        (MeasurementParity((1,), "L", True),),
        citations=("reference",),
        role_names=("a", "b"),
        measurement_names=("first", "second", "third"),
    )
    compact = plan.rescheduled(SchedulePolicy.COMPACT_UNITARIES)

    def boundaries(p):
        return [m for m in p.moments if any(op.name in {"R", "M", "RELABEL"} for op in m)]

    assert boundaries(plan) == boundaries(compact)
    assert _listing(plan) == _listing(compact)
    assert (compact.detectors, compact.observables, compact.measurement_aliases, compact.output_carriers) == (
        plan.detectors,
        plan.observables,
        plan.measurement_aliases,
        plan.output_carriers,
    )
    assert compact.citations == plan.citations
    assert compact.resources.physical_operations == plan.resources.physical_operations
    assert not ScheduledCircuit(1, ((), (), ()), (), ()).rescheduled(SchedulePolicy.COMPACT_UNITARIES).moments


@pytest.mark.parametrize("policy", [SchedulePolicy.REFERENCE, None, "compact_unitaries", True])
def test_comparison_rejects_invalid_candidate_before_capture(policy):
    with pytest.raises(ValueError, match="candidate_policy"):
        _plan(()).compare_guppy_schedules(runtime=None, candidate_policy=policy)


def test_compact_comparison_checks_full_listing_order_before_capture(monkeypatch):
    plan = _plan((Op("H", (0,)), Op("X", (0,)), Op("H", (1,))))
    wrong = replace(plan.rescheduled(SchedulePolicy.ASAP_UNITARIES), schedule=SchedulePolicy.COMPACT_UNITARIES)
    monkeypatch.setattr(ScheduledCircuit, "rescheduled", lambda _self, _policy: wrong)
    with pytest.raises(ValueError, match="complete source listing order"):
        plan.compare_guppy_schedules(runtime=None, candidate_policy=SchedulePolicy.COMPACT_UNITARIES)
