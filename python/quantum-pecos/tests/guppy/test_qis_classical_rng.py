"""Program-seeded classical randomness on the QIS route, with a live Selene oracle."""

from collections.abc import Callable

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.qsystem.random import RNG, make_discrete_distribution
from pecos import Guppy, selene_engine, sim
from pecos.compilation_pipeline import compile_guppy_to_hugr
from selene_sim import build
from selene_sim.backends import IdealErrorModel, Quest, SimpleRuntime

EXPECTED = {"a": 1085446021, "b": 0, "f": 0.18373215361498296, "c": 1312053665, "back": 1793278615}


def rng_program(seed: int) -> object:
    """Compile a program whose seed is independent of the simulation seed."""

    @guppy
    def main() -> None:
        rng = RNG(seed)
        a = rng.random_int()
        b = rng.random_int_bounded(10)
        f = rng.random_float()
        rng.random_advance(3)
        c = rng.random_int()
        rng.random_advance(-2)
        back = rng.random_int()
        rng.discard()
        result("a", a)
        result("b", b)
        result("f", f)
        result("c", c)
        result("back", back)

    return main


def helper_program(seed: int) -> object:
    """Exercise the standard library helpers that consume the same RNG."""

    @guppy
    def main() -> None:
        rng = RNG(seed)
        values = array(0, 1, 2, 3, 4)
        rng.shuffle(values)
        result("shuffled", values)
        dist = make_discrete_distribution(array(1.0, 2.0, 3.0))
        samples = array(dist.sample(rng) for _ in range(5))
        result("samples", samples)
        rng.discard()

    return main


def run_pecos(program: object, seed: int, shots: int = 3) -> dict:
    """Return the QIS route's named result columns."""
    return sim(Guppy(program)).classical(selene_engine()).qubits(1).seed(seed).run(shots).to_dict()


def run_selene(program: object, seed: int, shots: int = 3) -> list[dict]:
    """Run the same HUGR through selene_sim's reference runtime."""
    instance = build(compile_guppy_to_hugr(program))
    try:
        return [
            dict(shot)
            for shot in instance.run_shots(
                simulator=Quest(random_seed=seed),
                n_qubits=1,
                runtime=SimpleRuntime(random_seed=seed),
                error_model=IdealErrorModel(),
                n_shots=shots,
                random_seed=seed,
            )
        ]
    finally:
        instance.delete_files()


@pytest.mark.parametrize("instance_seed", [1, 987])
def test_reference_values_every_shot(instance_seed: int) -> None:
    """Every shot restarts the program seed's stream."""
    assert run_pecos(rng_program(42), instance_seed) == {key: [value] * 3 for key, value in EXPECTED.items()}


def test_second_program_seed_changes_stream() -> None:
    """The program seed controls the stream independently of the instance seed."""
    first = run_pecos(rng_program(42), 1)
    second = run_pecos(rng_program(43), 1)
    assert first != second
    assert second == run_pecos(rng_program(43), 987)
    assert all(values == [values[0]] * 3 for values in second.values())


@pytest.mark.parametrize("program_seed", [42, 43])
@pytest.mark.parametrize("instance_seed", [1, 987])
@pytest.mark.parametrize("factory", [rng_program, helper_program])
def test_live_selene_parity(factory: Callable[[int], object], program_seed: int, instance_seed: int) -> None:
    """Keep the classical stream and standard library helpers tied to Selene."""
    program = factory(program_seed)
    reference = run_selene(program, instance_seed)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    assert run_pecos(program, instance_seed) == expected


@pytest.mark.parametrize(
    ("program_seed", "shuffled", "samples"),
    [
        (42, [3, 4, 0, 2, 1], [1, 1, 1, 0, 2]),
        (43, [4, 1, 0, 3, 2], [1, 2, 0, 1, 0]),
    ],
)
def test_helper_pins(program_seed: int, shuffled: list[int], samples: list[int]) -> None:
    """Pins whose Selene verification is maintained by test_live_selene_parity."""
    expected = {"shuffled": [shuffled] * 3, "samples": [samples] * 3}
    assert run_pecos(helper_program(program_seed), 1) == expected
    assert run_pecos(helper_program(program_seed), 987) == expected


def constant_program(value: int) -> object:
    """Generate distinct programs with the same Python function name."""

    @guppy
    def main() -> None:
        # Keep the body out of line: a single result call is inlined into qmain,
        # eliminating the interposable call that this regression must exercise.
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)
        result("x", value)

    return main


def test_same_named_guppy_programs_keep_their_own_constants() -> None:
    """Loading a second same-named program must execute its own body."""
    first = constant_program(13)
    second = constant_program(29)
    assert run_pecos(first, 1) == {"x": [[13] * 32] * 3}
    assert run_pecos(second, 1) == {"x": [[29] * 32] * 3}
