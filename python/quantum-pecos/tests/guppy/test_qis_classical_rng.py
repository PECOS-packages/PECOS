"""Program-seeded classical randomness on the QIS route, with a live Selene oracle."""

from collections import defaultdict
from collections.abc import Callable

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.num import nat
from guppylang.std.qsystem.random import RNG, make_discrete_distribution
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import measure, qubit, x
from guppylang.std.random import seeded_pcg32
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


def run_pecos(program: object, seed: int, shots: int = 3, workers: int = 1) -> dict:
    """Return the QIS route's named result columns."""
    return sim(Guppy(program)).classical(selene_engine()).qubits(1).seed(seed).workers(workers).run(shots).to_dict()


def run_selene(program: object, seed: int, shots: int = 3) -> list[dict]:
    """Run the same HUGR through selene_sim's reference runtime."""
    instance = build(compile_guppy_to_hugr(program))
    try:
        shots_out = []
        for shot in instance.run_shots(
            simulator=Quest(random_seed=seed),
            n_qubits=1,
            runtime=SimpleRuntime(random_seed=seed),
            error_model=IdealErrorModel(),
            n_shots=shots,
            random_seed=seed,
        ):
            # Selene emits an ordered list of (label, value) pairs. Preserve every
            # repeated label, matching PECOS result aggregation instead of dict's
            # last-value-only behavior. A single array remains one array value.
            grouped = defaultdict(list)
            for key, value in shot:
                grouped[key].append(value)
            shots_out.append({key: values[0] if len(values) == 1 else values for key, values in grouped.items()})
        return shots_out
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


def shot_rng_program() -> object:
    """Use the program index both as an RNG stream and an array subscript."""

    @guppy
    def main() -> None:
        shot = get_current_shot()
        rng = RNG(42 + shot)
        value = rng.random_int()
        rng.discard()
        parameters = array(11, 23, 37, 41, 59, 67, 79)
        result("shot", shot)
        result("rng", value)
        result("parameter", parameters[shot])

    return main


@pytest.mark.parametrize("instance_seed", [1, 987])
def test_program_shot_rng_and_parameters_match_live_selene(instance_seed: int) -> None:
    """Global indices select the same parameters and RNG streams with any worker count."""
    program = shot_rng_program()
    reference = run_selene(program, instance_seed, shots=7)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    assert expected["shot"] == list(range(7))
    assert expected["parameter"] == [11, 23, 37, 41, 59, 67, 79]
    assert len(set(expected["rng"])) == 7
    for workers in (1, 2, 3):
        assert run_pecos(program, instance_seed, shots=7, workers=workers) == expected


def platform_rng_features(seed: int) -> object:
    """Exercise every platform RNG method, including signed seeds and large jumps."""

    @guppy
    def main() -> None:
        rng = RNG(seed)
        result("integers", array(rng.random_int() for _ in range(16)))
        result("floats", array(rng.random_float() for _ in range(8)))
        result("angles", array(rng.random_angle().halfturns for _ in range(8)))
        result("clifford_angles", array(rng.random_clifford_angle().halfturns for _ in range(8)))
        result("unit_bound", rng.random_int_bounded(1))
        result("odd_bound", array(rng.random_int_bounded(3) for _ in range(8)))
        # This bound rejects almost one quarter of the unsigned PCG32 outputs.
        result("rejection_bound", array(rng.random_int_bounded(1073741825) for _ in range(16)))
        result("max_bound", array(rng.random_int_bounded(2147483647) for _ in range(8)))
        before = rng.random_int()
        rng.random_advance(-1)
        rng.random_advance(0)
        result("before", before)
        result("repeated", rng.random_int())
        rng.random_advance(1099511627793)
        result("after_jump", rng.random_int())
        rng.random_advance(-1099511627794)
        result("after_rewind", rng.random_int())
        rng.discard()
        replacement = RNG(seed)
        result("recreated", replacement.random_int())
        replacement.discard()

    return main


@pytest.mark.parametrize("program_seed", [-(2**63), -1, 0, 42, 2**63 - 1])
def test_all_platform_rng_methods_match_live_selene(program_seed: int) -> None:
    """Match full mixed-call streams, not just each method's first output."""
    program = platform_rng_features(program_seed)
    reference = run_selene(program, 1)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    for shot in reference:
        assert all(-(2**31) <= value < 2**31 for value in shot["integers"])
        assert all(0.0 <= value < 1.0 for value in shot["floats"])
        assert all(-1.0 <= value < 1.0 for value in shot["angles"])
        assert set(shot["clifford_angles"]) <= {0.0, 0.5, 1.0, 1.5}
        assert shot["unit_bound"] == 0
        assert all(0 <= value < 3 for value in shot["odd_bound"])
        assert all(0 <= value < 1073741825 for value in shot["rejection_bound"])
        assert all(0 <= value < 2147483647 for value in shot["max_bound"])
        assert shot["before"] == shot["repeated"]
    # Simulation seed changes and two concurrent workers cannot change explicit seeds.
    assert run_pecos(program, 987, workers=2) == expected


def native_rng_program(seed: int) -> object:
    """Interleave two native generators with platform state and an isolated control."""

    @guppy
    def main() -> None:
        first = seeded_pcg32(nat(seed))
        second = seeded_pcg32(nat(seed + 1))
        control = seeded_pcg32(nat(seed))
        platform = RNG(seed)
        for _ in range(12):
            result("first", first.next_int())
            result("second", second.next_int())
            result("platform", platform.random_int())
            result("control", control.next_int())
            result("first_bounded", first.next_int_bounded(nat(1073741825)))
            result("second_bounded", second.next_int_bounded(nat(3)))
            result("control_bounded", control.next_int_bounded(nat(1073741825)))
        platform.discard()

    return main


@pytest.mark.parametrize("program_seed", [0, 42, 2**62])
def test_native_guppy_rng_matches_live_selene(program_seed: int) -> None:
    """Pure Guppy PCG32 instances stay independent of one another and platform RNG."""
    program = native_rng_program(program_seed)
    reference = run_selene(program, 1)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    for shot in reference:
        assert shot["first"] == shot["control"]
        assert shot["first_bounded"] == shot["control_bounded"]
        assert shot["first"] != shot["second"]
        assert all(0 <= value < 1073741825 for value in shot["first_bounded"])
        assert all(0 <= value < 3 for value in shot["second_bounded"])
    assert run_pecos(program, 987, workers=2) == expected


def feedback_rng_program() -> object:
    """Feed random choices into gates, and measurements back into RNG control flow."""

    @guppy
    def main() -> None:
        rng = RNG(42 + get_current_shot())
        for _ in range(8):
            bit = rng.random_int_bounded(2)
            q = qubit()
            if bit == 1:
                x(q)
            observed = measure(q).read()
            result("chosen", bit)
            result("measured", observed)
            # The measurement read suspends the program and selects draw count.
            if observed:
                result("branch_draw", rng.random_int())
            else:
                rng.random_advance(3)
            result("after_feedback", rng.random_int())
        rng.discard()

    return main


def test_rng_state_survives_quantum_feedback() -> None:
    """Both drivers execute the same deterministic quantum branches and later RNG draws."""
    program = feedback_rng_program()
    reference = run_selene(program, 1, shots=7)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    choices = {bit for shot in reference for bit in shot["chosen"]}
    assert choices == {0, 1}
    for shot in reference:
        assert shot["chosen"] == [int(bit) for bit in shot["measured"]]
    for workers in (1, 3):
        assert run_pecos(program, 987, shots=7, workers=workers) == expected


def invalid_distribution_program(weight: float) -> object:
    """Exercise Guppy's checked distribution construction."""

    @guppy
    def main() -> None:
        rng = RNG(42)
        distribution = make_discrete_distribution(array(weight, 0.0, 0.0))
        result("sample", distribution.sample(rng))
        rng.discard()

    return main


def oversized_bound_program() -> object:
    """A platform bound must fit the signed 32-bit input ABI."""

    @guppy
    def main() -> None:
        rng = RNG(42)
        result("sample", rng.random_int_bounded(2147483648))
        rng.discard()

    return main


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: invalid_distribution_program(-1.0), "Negative weight included in discrete distribution"),
        (lambda: invalid_distribution_program(0.0), "No positive weights included in discrete distribution"),
        (oversized_bound_program, "bound must be a 32-bit integer"),
    ],
)
def test_rng_input_errors_match_live_selene(factory: Callable[[], object], message: str) -> None:
    """Invalid Guppy RNG inputs must fail instead of returning fabricated samples."""
    from selene_sim.exceptions import SelenePanicError

    program = factory()
    with pytest.raises(SelenePanicError, match=message):
        run_selene(program, 1, shots=1)
    with pytest.raises(RuntimeError, match=message):
        run_pecos(program, 1, shots=1)
    # An aborted invocation must not poison the next simulation's program RNG.
    assert run_pecos(rng_program(42), 1, shots=1) == {key: [value] for key, value in EXPECTED.items()}


def test_multiple_platform_rng_handles_match_live_selene() -> None:
    """Platform handles share global state; do not promise independent streams."""

    @guppy
    def main() -> None:
        first = RNG(42)
        result("draw", first.random_int())
        second = RNG(43)
        for _ in range(5):
            result("draw", first.random_int())
            result("draw", second.random_int())
        first.discard()
        result("draw", second.random_int())
        second.discard()

    reference = run_selene(main, 1)
    expected = {"draw": [shot["draw"] for shot in reference]}
    assert all(len(draws) == 12 for draws in expected["draw"])
    assert run_pecos(main, 987, workers=2) == expected


@pytest.mark.parametrize("delta", [0, 1, -1, 2**63 - 1, -(2**63)])
def test_rng_jump_boundaries_match_live_selene(delta: int) -> None:
    """Extreme signed jumps compose modulo 2**64, including the self-inverse minimum."""
    reverse = delta if delta == -(2**63) else -delta

    @guppy
    def main() -> None:
        rng = RNG(42)
        result("first", rng.random_int())
        rng.random_advance(delta)
        result("jumped", rng.random_int())
        rng.random_advance(-1)
        rng.random_advance(reverse)
        result("restored", rng.random_int())
        rng.discard()

    reference = run_selene(main, 1)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    assert expected["first"] == [1085446021] * 3
    assert expected["restored"] == [176895750] * 3
    assert run_pecos(main, 987, workers=2) == expected


@pytest.mark.parametrize("seed", [0, 2**63 - 1, 2**63, 2**64 - 1])
def test_native_rng_unsigned_seed_boundaries(seed: int) -> None:
    """Native PCG32 handles the unsigned seed boundaries, including the high bit."""
    high, low = divmod(seed, 2**32)

    @guppy
    def main() -> None:
        rng = seeded_pcg32((nat(high) << nat(32)) | nat(low))
        result("integers", array(rng.next_int() for _ in range(16)))
        result("unit_bound", rng.next_int_bounded(nat(1)))
        result("max_bound", array(rng.next_int_bounded(nat(2147483647)) for _ in range(8)))

    reference = run_selene(main, 1)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    for shot in reference:
        assert shot["unit_bound"] == 0
        assert all(0 <= value < 2147483647 for value in shot["max_bound"])
    assert run_pecos(main, 987, workers=2) == expected


@pytest.mark.parametrize("size", [0, 1])
def test_rng_degenerate_helpers_match_live_selene(size: int) -> None:
    """Empty/singleton shuffles and deterministic distributions retain stream behavior."""

    @guppy
    def main() -> None:
        rng = RNG(42)
        values = array(i for i in range(size))
        rng.shuffle(values)
        # Check the operation without relying on empty-array output (unsupported
        # by Selene) or singleton-array presentation (scalar in PECOS).
        result("length", len(values))
        if size == 1:
            result("element", values[0])
        single = make_discrete_distribution(array(3.0))
        sparse = make_discrete_distribution(array(0.0, 2.0, 0.0))
        result("single", array(single.sample(rng) for _ in range(8)))
        result("sparse", array(sparse.sample(rng) for _ in range(8)))
        result("after_helpers", rng.random_int())
        rng.discard()

    reference = run_selene(main, 1)
    expected = {key: [shot[key] for shot in reference] for key in reference[0]}
    for shot in reference:
        assert shot["length"] == size
        if size == 1:
            assert shot["element"] == 0
        assert shot["single"] == [0] * 8
        assert shot["sparse"] == [1] * 8
    assert run_pecos(main, 987, workers=2) == expected


@pytest.mark.parametrize("bound", [0, -1, -(2**31)])
def test_nonpositive_platform_rng_bounds_fail_cleanly(bound: int) -> None:
    """Do not emulate Selene's crash/unsigned reinterpretation for invalid bounds."""

    @guppy
    def main() -> None:
        rng = RNG(42)
        result("sample", rng.random_int_bounded(bound))
        rng.discard()

    with pytest.raises(RuntimeError, match=r"random_rng.*bound must be positive"):
        run_pecos(main, 1, shots=1)
    assert run_pecos(rng_program(42), 1, shots=1) == {key: [value] for key, value in EXPECTED.items()}


def test_reusable_simulation_recovers_from_rng_error_after_feedback() -> None:
    """An error on the second global shot must not poison a reset simulation."""

    @guppy
    def main() -> None:
        shot = get_current_shot()
        rng = RNG(42 + shot)
        q = qubit()
        x(q)
        observed = measure(q).read()
        if observed and shot == 1:
            result("invalid", rng.random_int_bounded(0))
        result("value", rng.random_int())
        rng.discard()

    simulation = sim(Guppy(main)).classical(selene_engine()).qubits(1).seed(42).workers(1).build()
    with pytest.raises(RuntimeError, match=r"random_rng.*bound must be positive"):
        simulation.run(3)
    simulation.reset()
    assert simulation.run(1).to_dict() == {"value": [1085446021]}
    assert simulation.run(1).to_dict() == {"value": [1085446021]}
