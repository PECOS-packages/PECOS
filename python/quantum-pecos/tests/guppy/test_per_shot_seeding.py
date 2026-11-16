"""Each shot has its own stream, independent of workers and earlier branches."""

import pecos_rslib as prs
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import h, measure, qubit, x
from pecos import sim


@guppy
def measured_branch() -> None:
    q = qubit()
    h(q)
    before = measure(q).read()
    result("before", before)
    tail = qubit()
    if before:
        h(tail)
        x(tail)
    else:
        x(tail)
    result("after", measure(tail).read())


@guppy
def forced_branch() -> None:
    q = qubit()
    h(q)
    before = measure(q).read()
    result("before", before)
    tail = qubit()
    h(tail)
    x(tail)
    result("after", measure(tail).read())


def run(program, workers: int) -> dict:
    return (
        sim(program)
        .classical(prs.qis_engine().selene_runtime().interface(prs.qis_helios_interface()))
        .quantum(prs.stab_vec())
        .qubits(2)
        .noise(prs.depolarizing_noise().with_uniform_probability(0.1))
        .seed(42)
        .workers(workers)
        .run(64)
        .to_dict()
    )


def test_pre_branch_records_match_across_variants() -> None:
    measured = run(measured_branch, 1)
    forced = run(forced_branch, 1)
    assert set(measured["before"]) == {0, 1}
    assert measured["before"] == forced["before"]


def test_results_do_not_depend_on_worker_count() -> None:
    assert run(measured_branch, 1) == run(measured_branch, 3)
