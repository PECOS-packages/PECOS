"""Behavioral coverage for Guppy execution through HUGR and Selene."""

import re

import pecos as pc
import pecos_rslib_llvm
import pytest
from guppylang import guppy
from guppylang.std.builtins import comptime, result
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos.guppy_gen import variant_scoped


@guppy
def hadamard() -> None:
    q = qubit()
    h(q)
    result("outcome", measure(q).read())


@guppy
def bell_state() -> None:
    q0, q1 = qubit(), qubit()
    h(q0)
    cx(q0, q1)
    result("left", measure(q0).read())
    result("right", measure(q1).read())


@pytest.fixture(params=[pc.hugr_engine, pc.selene_engine], ids=["hugr", "selene"])
def engine_factory(request):
    return request.param


def _run(program, engine_factory, qubits: int, shots: int, noise=None) -> dict:
    if engine_factory is pc.hugr_engine:
        builder = engine_factory().hugr_bytes(program.compile().to_bytes()).to_sim()
    else:
        builder = pc.sim(pc.Guppy(program)).classical(engine_factory())
    builder = builder.quantum(pc.state_vector()).qubits(qubits).seed(42)
    if noise is not None:
        builder = builder.noise(noise)
    return builder.run(shots).to_dict()


@pytest.mark.parametrize("initial_bit", [False, True], ids=["zero", "one"])
def test_basis_state(engine_factory, initial_bit: bool) -> None:
    def basis_state() -> None:
        q = qubit()
        if comptime(initial_bit):
            x(q)
        result("outcome", measure(q).read())

    program = guppy(variant_scoped(basis_state, initial_bit))
    results = _run(program, engine_factory, qubits=1, shots=16)
    assert results["outcome"] == [int(initial_bit)] * 16


def test_hadamard_distribution(engine_factory) -> None:
    shots = 1024
    values = _run(hadamard, engine_factory, qubits=1, shots=shots)["outcome"]
    assert len(values) == shots
    assert set(values) == {0, 1}
    # Six binomial standard errors for P(1) = 1/2.
    assert sum(values) / shots == pytest.approx(0.5, abs=6 * (0.25 / shots) ** 0.5, rel=0)


def test_bell_state_correlations(engine_factory) -> None:
    shots = 1024
    results = _run(bell_state, engine_factory, qubits=2, shots=shots)
    left, right = results["left"], results["right"]
    assert len(left) == len(right) == shots
    assert left == right
    assert set(left) == {0, 1}
    assert sum(left) / shots == pytest.approx(0.5, abs=6 * (0.25 / shots) ** 0.5, rel=0)


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("target", [False, True])
def test_cnot_truth_table(engine_factory, control: bool, target: bool) -> None:
    def cnot() -> None:
        q0, q1 = qubit(), qubit()
        if comptime(control):
            x(q0)
        if comptime(target):
            x(q1)
        cx(q0, q1)
        result("control", measure(q0).read())
        result("target", measure(q1).read())

    program = guppy(variant_scoped(cnot, control, target))
    results = _run(program, engine_factory, qubits=2, shots=16)
    assert results["control"] == [int(control)] * 16
    assert results["target"] == [int(control ^ target)] * 16


@pytest.mark.parametrize("probability", [0.0, 0.25, 1.0])
def test_measurement_noise(engine_factory, probability: float) -> None:
    @guppy
    def noisy_x() -> None:
        q = qubit()
        x(q)
        result("outcome", measure(q).read())

    shots = 1024
    noise = pc.depolarizing_noise().with_uniform_probability(0.0).with_p_meas(probability)
    values = _run(noisy_x, engine_factory, qubits=1, shots=shots, noise=noise)["outcome"]
    assert len(values) == shots
    assert set(values) <= {0, 1}
    tolerance = 6 * (probability * (1 - probability) / shots) ** 0.5
    assert values.count(0) / shots == pytest.approx(probability, abs=tolerance, rel=0)


@pytest.mark.parametrize("iterations", [0, 3])
def test_parametric_loop_execution(engine_factory, iterations: int) -> None:
    @guppy
    def count_ones(n: int) -> int:
        count = 0
        for _ in range(n):
            q = qubit()
            x(q)
            if measure(q).read():
                count += 1
        return count

    def loop_program() -> None:
        count = count_ones(comptime(iterations))
        q = qubit()
        if count == comptime(iterations):
            x(q)
        result("count_matches", measure(q).read())

    program = guppy(variant_scoped(loop_program, iterations))
    results = _run(program, engine_factory, qubits=1, shots=16)
    assert results["count_matches"] == [1] * 16


def test_three_qubit_parity(engine_factory) -> None:
    @guppy
    def parity() -> None:
        q0, q1, q2 = qubit(), qubit(), qubit()
        h(q0)
        h(q1)
        cx(q0, q2)
        cx(q1, q2)
        result("left", measure(q0).read())
        result("right", measure(q1).read())
        result("parity", measure(q2).read())

    shots = 256
    results = _run(parity, engine_factory, qubits=3, shots=shots)
    left, right, parity = results["left"], results["right"], results["parity"]
    assert len(left) == len(right) == len(parity) == shots
    assert set(zip(left, right, strict=True)) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert parity == [a ^ b for a, b in zip(left, right, strict=True)]


@pytest.mark.parametrize(("program", "num_measurements"), [(hadamard, 1), (bell_state, 2)], ids=["hadamard", "bell"])
def test_hugr_to_qis_compilation(program, num_measurements: int) -> None:
    output = pecos_rslib_llvm.compile_hugr_to_qis(program.compile().to_bytes())
    assert re.search(r"define\b[^\n]*@qmain\(", output)
    assert len(re.findall(r"\bcall\b[^\n]*@___lazy_measure\(", output)) == num_measurements
