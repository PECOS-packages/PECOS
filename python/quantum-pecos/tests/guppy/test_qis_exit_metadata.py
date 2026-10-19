"""Ordinary exit diagnostics are shot metadata, separate from user registers."""

import pytest
from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.platform import exit as guppy_exit
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import measure, qubit
from guppylang.std.random import seeded_pcg32
from pecos import Guppy, selene_engine, sim
from pecos.compilation_pipeline import compile_guppy_to_hugr
from selene_sim import build
from selene_sim.backends import IdealErrorModel, Quest, SimpleRuntime
from selene_sim.exceptions import SelenePanicError


def reference(program: object, shots: int) -> list[dict]:
    """Run the same HUGR in live Selene; these programs use unique output tags."""
    instance = build(compile_guppy_to_hugr(program))
    try:
        return [
            dict(shot)
            for shot in instance.run_shots(
                simulator=Quest(random_seed=42),
                n_qubits=1,
                runtime=SimpleRuntime(random_seed=42),
                error_model=IdealErrorModel(),
                n_shots=shots,
                random_seed=42,
            )
        ]
    finally:
        instance.delete_files()


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("feedback", [False, True])
def test_exit_metadata_matches_selene_without_changing_outputs(code: int, workers: int, feedback: bool) -> None:
    """Collection-time and post-measurement exits retain data and their diagnostics."""

    @guppy
    def program() -> None:
        result("shot", get_current_shot())
        # A user register named like the metadata is still an ordinary register.
        result("program_exit", 7)
        if feedback:
            result("det", measure(qubit()).read())
        guppy_exit("done", code)

    shots = sim(Guppy(program)).classical(selene_engine()).qubits(1).workers(workers).run(4)
    expected = [{"code": code, "message": "done"}] * 4
    assert shots.program_exits == expected
    columns = shots.to_shot_map()
    assert columns.program_exits == expected
    assert columns.shots == len(shots) == 4
    data = shots.to_dict()
    assert data["program_exit"] == [7] * 4
    oracle = reference(program, 4)
    for i, row in enumerate(oracle):
        assert row.pop("exit: done") == shots.program_exits[i]["code"]
        assert row == {name: values[i] for name, values in data.items()}
    # Returned dictionaries are copies, not mutable access to the stored result.
    exposed = shots.program_exits
    exposed[0]["code"] = 99
    assert shots.program_exits == expected


@pytest.mark.parametrize("workers", [1, 3])
def test_native_rng_zero_bound_keeps_diagnostic_and_empty_shot_count(workers: int) -> None:
    """The #1371 reproducer no longer looks like an unexplained empty result."""

    @guppy
    def program() -> None:
        rng = seeded_pcg32(nat(42))
        result("sample", rng.next_int_bounded(nat(0)))

    message = "PCG32.next_int_bounded: bound must be positive"
    shots = sim(Guppy(program)).classical(selene_engine()).qubits(1).workers(workers).run(4)
    assert shots.to_dict() == {}
    assert shots.program_exits == [{"code": 1, "message": message}] * 4
    assert shots.to_shot_map().shots == 4
    assert shots.to_shot_map().program_exits == shots.program_exits
    assert reference(program, 4) == [{f"exit: {message}": 1}] * 4


@pytest.mark.parametrize("workers", [1, 3])
def test_ragged_results_do_not_hide_exits_or_contaminate_reused_simulation(workers: int) -> None:
    """Metadata remains readable when early exit changes output register counts."""

    @guppy
    def program() -> None:
        index = get_current_shot()
        result("shot", index)
        result("measured", measure(qubit()).read())
        if index % 2 == 1:
            guppy_exit("odd shot", 1)
        result("after", 9)

    simulation = sim(Guppy(program)).classical(selene_engine()).qubits(1).workers(workers).build()
    shots = simulation.run(6)
    expected = [None, {"code": 1, "message": "odd shot"}] * 3
    assert shots.program_exits == expected
    with pytest.raises(RuntimeError, match="registers"):
        shots.to_dict()
    with pytest.raises(RuntimeError, match="registers"):
        shots.to_shot_map()
    assert shots.program_exits == expected
    for i, row in enumerate(reference(program, 6)):
        assert row["shot"] == i
        assert ("after" in row) == (i % 2 == 0)
        assert row.get("exit: odd shot") == (None if i % 2 == 0 else 1)
    simulation.reset()
    for _ in range(2):
        normal = simulation.run(1)
        assert normal.program_exits == [None]
        assert normal.to_dict() == {"shot": [0], "measured": [0], "after": [9]}
    # Reset and subsequent jobs must not change already returned diagnostics.
    assert shots.program_exits == expected


def test_exit_like_output_names_are_not_metadata() -> None:
    """There are no reserved diagnostic names in user output registers."""

    @guppy
    def program() -> None:
        result("exit: done", 7)
        result("program_exits", 8)
        guppy_exit("done", 1)

    shots = sim(Guppy(program)).classical(selene_engine()).qubits(1).run(1)
    assert shots.to_dict() == {"exit: done": [7], "program_exits": [8]}
    assert shots.program_exits == [{"code": 1, "message": "done"}]


def test_exit_code_1000_records_existing_reference_boundary_difference() -> None:
    """Adding metadata preserves PECOS's existing inclusive exit boundary."""

    @guppy
    def program() -> None:
        guppy_exit("boundary", 1000)

    shots = sim(Guppy(program)).classical(selene_engine()).qubits(1).run(2)
    assert shots.program_exits == [{"code": 1000, "message": "boundary"}] * 2
    assert shots.to_dict() == {}
    # Live Selene 0.3.4/0.3.5 classifies 1000 as a panic despite its >1000 docs.
    # This is a documented compatibility gap, not a claimed parity result.
    with pytest.raises(SelenePanicError) as error:
        reference(program, 2)
    assert error.value.code == 1000
