"""Dependency coverage is distinct from calibrated physical moment timing."""

import pytest
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyFence, GuppyOrdering, ScheduledCircuit


def _plan():
    return ScheduledCircuit(
        3,
        (
            tuple(Op("R", (q,)) for q in range(3)),
            (Op("X", (0,)),),
            (Op("H", (2,)),),
            (Op("CX", (1, 2)),),
            (Op("M", (2,)),),
            (Op("R", (2,)),),
            (Op("H", (2,)),),
            (Op("Z", (0,)),),
            (Op("RELABEL", (0, 1)),),
            (Op("X", (0,)),),
            tuple(Op("M", (q,)) for q in range(3)),
        ),
        (),
        (),
    )


@pytest.mark.parametrize("n", range(1, 21))
def test_fence_has_all_input_to_all_output_dependencies(n):
    roles = tuple(range(n))
    dependencies = {q: {q} for q in roles}
    fence = GuppyFence(0, roles)
    for operands in fence.joins:
        union = set().union(*(dependencies[q] for q in operands))
        for q in operands:
            dependencies[q] = union.copy()
    assert all(dependencies[q] == set(roles) for q in roles)
    assert len(fence.joins) == (1 if n == 1 else 2 * n - 3)


def test_regions_do_not_claim_to_join_lifetime_or_relabel_boundaries():
    plan = _plan().guppy_ordering_plan(GuppyOrdering.UNITARY_REGIONS)
    assert [f.after_moment for f in plan.fences] == [1, 2, 6]
    assert all(f.qubits == (0, 1, 2) for f in plan.fences)
    report = plan.to_dict()
    assert report["uncovered_cuts"] == [0, 3, 4, 5, 7, 8, 9]
    assert report["runtime_barrier_calls"] == 9
    assert not report["physical_schedule_verified"]
    assert not report["noise_equivalence_verified"]
    assert "3/10" in plan.describe()


def test_strict_fixed_lifetime_rejects_memory_reuse():
    with pytest.raises(ValueError, match="no intermediate R/M"):
        _plan().guppy_program(ordering=GuppyOrdering.FIXED_LIFETIME)


def test_fixed_lifetime_covers_relabel_and_empty_moment_cuts():
    p = ScheduledCircuit(
        2,
        (
            (Op("R", (0,)), Op("R", (1,))),
            (Op("RELABEL", (0, 1)),),
            (),
            (Op("X", (0,)),),
            (Op("M", (0,)), Op("M", (1,))),
        ),
        (),
        (),
    )
    contract = p.guppy_ordering_plan(GuppyOrdering.FIXED_LIFETIME)
    assert [f.after_moment for f in contract.fences] == [0, 1, 2, 3]
    assert contract.to_dict()["uncovered_cuts"] == []
    source = p.guppy_source(ordering=GuppyOrdering.FIXED_LIFETIME)
    assert "q1, q0 = pecos_qis_runtime_barrier_qubits2_hugr(q1, q0)" in source
    assert "swap(" not in source


def test_default_remains_functional_and_bad_mode_fails_early():
    assert _plan().guppy_ordering_plan().fences == ()
    assert "runtime_barrier" not in _plan().guppy_source()
    with pytest.raises(TypeError, match="GuppyOrdering"):
        _plan().guppy_source(ordering="unitary_regions")


def test_all_moments_reports_measurement_dependencies_and_empty_live_fences():
    plan = ScheduledCircuit(
        1,
        ((), (Op("R", (0,)),), (Op("M", (0,)),), (), (Op("R", (0,)),), (Op("M", (0,)),), ()),
        (),
        (),
    )
    contract = plan.guppy_ordering_plan(GuppyOrdering.ALL_MOMENTS)
    report = contract.to_dict()
    assert report["uncovered_cuts"] == []
    assert report["runtime_barrier_calls"] == 6
    assert report["ordered_allocation_calls"] == 2
    assert report["dependency_helper_calls"] == 6  # two collect/scatter pairs, two outcomes
    assert [(f.qubits, f.measurements) for f in contract.fences] == [
        ((), ()),
        ((0,), ()),
        ((), (0,)),
        ((), ()),
        ((0,), ()),
        ((), (1,)),
    ]
    assert not report["physical_schedule_verified"]
    assert not report["noise_equivalence_verified"]


def test_all_moments_relabels_only_aliases():
    source = _plan().guppy_source(ordering=GuppyOrdering.ALL_MOMENTS)
    assert "q1, ordering_token = pecos_qis_ordering_join_hugr(q1, ordering_token)" in source
    assert "swap(" not in source
    assert "ordering_token = pecos_qis_ordering_observe_hugr(ordering_token, m0)" in source


@pytest.mark.parametrize("name", ["int", "bool", "tuple", "pecos_qis_ordered_alloc_hugr"])
def test_lifetime_source_rejects_shadowed_helper_types_and_names(name):
    with pytest.raises(ValueError, match="shadow"):
        _plan().guppy_source(function_name=name, ordering=GuppyOrdering.ALL_MOMENTS)
