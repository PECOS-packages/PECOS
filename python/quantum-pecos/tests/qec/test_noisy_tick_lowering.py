"""Executable channel payloads, physical timing, and native runtime integration."""

import itertools
import json
from collections import defaultdict

import pytest
import stim
from pecos.qec.generic import SI1000, CircuitOp, MeasurementParity, ScheduledCircuit, evaluate_tick_circuit
from pecos.quantum import TickCircuit
from pecos_rslib_exp import depolarizing, monte_carlo, sim_neo, stabilizer

from qec.test_native_scheduled_noise import _pair_plan
from qec.test_scheduled_si1000 import CASES, FIXTURES, _case_plan, _wire_moments


def _payload(channel):
    words = ("I", "X") if channel["kind"] == "X_ERROR" else (
        "".join(w) for w in itertools.product("IXYZ", repeat=len(channel["qubits"]))
    )
    denominator = {"X_ERROR": 1, "DEPOLARIZE1": 3, "DEPOLARIZE2": 15}[channel["kind"]]
    return {
        tuple(sorted((p, q) for q, p in zip(channel["qubits"], word, strict=True) if p != "I")):
            channel["probability"] / denominator if set(word) != {"I"} else 1 - channel["probability"]
        for word in words
    }


def _wire_steps(circuit):
    """Project execution steps to physical moments, checking actual channel payloads."""
    result, ids = defaultdict(list), set()
    names = {"PZ": "R", "MZ": "M", "SZ": "S", "SZdg": "S_DAG", "F": "C_XYZ", "Fdg": "C_ZYX", "SX": "SQRT_X"}
    for i in range(circuit.num_ticks()):
        moment = circuit.get_tick_meta(i, "qec_moment")
        phase = circuit.get_tick_meta(i, "qec_execution_phase")
        tick = circuit.get_tick(i)
        seen = set()
        for j, gate in enumerate(tick.gate_batches()):
            assert not seen.intersection(gate.qubits)
            seen.update(gate.qubits)
            if gate.is_channel():
                entry = tick.get_gate_attr(j, "qec_noise_channel")
                assert entry["id"] not in ids
                ids.add(entry["id"])
                assert moment == entry["moment"]
                assert phase == (entry["reason"] if entry["placement"] == "end" else entry["placement"])
                assert set(gate.qubits) == set(entry["qubits"])
                actual = {tuple(sorted(terms)): p for p, terms in gate.channel_mixed_pauli_terms()}
                assert actual == _payload(entry)
                name, qs, args = entry["kind"], tuple(entry["qubits"]), (entry["probability"],)
                for q in qs:
                    result[moment, q].append((name, qs, args))
            else:
                assert phase == "gates"
                name = names.get(gate.gate_type.name, gate.gate_type.name)
                arity = 2 if name in {"CX", "CZ"} else 1
                for start in range(0, len(gate.qubits), arity):
                    qs = tuple(gate.qubits[start:start+arity])
                    for q in qs:
                        result[moment, q].append((name, qs, ()))
    return dict(result), ids


def _run(circuit, shots=32, seed=42):
    return tuple(tuple(bool(x) for x in row) for row in sim_neo(circuit).quantum(stabilizer()).sampling(
        monte_carlo(shots),
    ).seed(seed).run().to_list())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["fixture"])
def test_published_channels_physical_order_and_native_runtime(case):
    noisy = _case_plan(case)
    circuit = noisy.to_tick_circuit()
    wire, ids = _wire_steps(circuit)
    reference, moments = _wire_moments(stim.Circuit((FIXTURES / case["fixture"]).read_text()))
    assert wire == reference
    assert ids == {c.id for c in noisy.channels}
    assert circuit.get_meta("qec_physical_moments") == moments
    assert circuit.get_meta("qec_noise_channels") == len(ids)
    assert circuit.get_meta("qec_tick_semantics") == "execution_steps"
    assert circuit.measurement_emission() == noisy.ideal.to_tick_circuit().measurement_emission()
    assert circuit.circuit_definitions() == noisy.ideal.to_tick_circuit().circuit_definitions()
    for record in _run(circuit, shots=3):
        assert len(record) == noisy.ideal.num_measurements
        assert evaluate_tick_circuit(circuit, record) == noisy.ideal.evaluate(record)


@pytest.mark.parametrize("p", [0, .001, .2, 1e-20])
def test_partial_collapse_wait_empty_moments_affine_values_and_retired_sites(p):
    noisy = SI1000(p).apply(_pair_plan())
    circuit = noisy.to_tick_circuit()
    assert _wire_steps(circuit)[0] == _wire_moments(noisy.to_stim())[0]
    moments = [circuit.get_tick_meta(i, "qec_moment") for i in range(circuit.num_ticks())]
    assert moments == sorted(moments)
    gates = [i for i in range(circuit.num_ticks()) if circuit.get_tick_meta(i, "qec_execution_phase") == "gates"]
    assert len(gates) == len(noisy.ideal.moments)
    for i, moment in zip(gates, noisy.ideal.moments, strict=True):
        assert circuit.get_tick(i).is_empty() == (not moment)
    assert json.loads(circuit.get_meta("observables"))[0]["parity_constant"] == 1
    for record in _run(circuit):
        assert evaluate_tick_circuit(circuit, record) == noisy.ideal.evaluate(record)
        if p == 0:
            assert record == (False, True, False)
    if p == 0:
        assert circuit.num_ticks() == len(noisy.ideal.moments)
    else:
        with pytest.raises(ValueError, match="do not also pass"):
            sim_neo(circuit).quantum(stabilizer()).noise(depolarizing().p1(.01)).run()


@pytest.mark.parametrize(("method", "targets", "branches"), [
    ("x_error", 3, 2), ("depolarize1", 3, 4), ("depolarize2", (3, 1), 16),
])
@pytest.mark.parametrize("p", [0, .125, 1])
def test_public_channel_payloads(method, targets, branches, p):
    circuit = TickCircuit()
    tick = circuit.tick()
    getattr(tick, method)(p, targets).meta("physical_location", "test")
    gate = circuit.get_tick(0).gate_batches()[0]
    assert gate.is_channel()
    terms = gate.channel_mixed_pauli_terms()
    assert len(terms) == branches
    assert sum(prob for prob, _ in terms) == pytest.approx(1.)
    assert sum(prob for prob, ops in terms if ops) == pytest.approx(p)
    assert circuit.get_tick(0).get_gate_attr(0, "physical_location") == "test"


@pytest.mark.parametrize(("method", "targets"), [("x_error", 0), ("depolarize1", 0), ("depolarize2", (0, 1))])
@pytest.mark.parametrize("p", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_channel_probability_does_not_mutate_tick(method, targets, p):
    circuit = TickCircuit()
    tick = circuit.tick().h([2])
    with pytest.raises(ValueError, match="probability"):
        getattr(tick, method)(p, targets)
    assert circuit.get_tick(0).gate_count() == 1


def test_channel_qubit_validation_and_conflict_do_not_mutate_tick():
    circuit = TickCircuit()
    tick = circuit.tick().h([0])
    with pytest.raises(ValueError, match="distinct"):
        tick.depolarize2(.1, (1, 1))
    with pytest.raises(ValueError, match=r"conflict|already"):
        tick.depolarize2(.1, (1, 0))
    assert circuit.get_tick(0).gate_count() == 1
    assert circuit.get_tick(0).active_qubits() == [0]


def test_direct_pre_measurement_and_post_reset_channels_are_executable():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().x_error(1., 0)
    circuit.tick().mz([0])
    circuit.tick().pz([0])
    circuit.tick().x_error(1., 0)
    circuit.tick().x_error(1., 0)
    circuit.tick().mz([0])
    assert _run(circuit) == ((True, False),) * 32


def test_categorical_runtime_distribution_against_exact_classical_enumeration():
    op = CircuitOp
    plan = ScheduledCircuit(2, (
        (op("R", (0,)), op("R", (1,))), (op("CZ", (0, 1)),), (op("M", (1,)), op("M", (0,))),
    ), (MeasurementParity((0, 1), "check"),), ())
    p = .16
    expected = dict.fromkeys(itertools.product((False, True), repeat=2), 0.)
    for r0, r1, m0, m1 in itertools.product((0, 1), repeat=4):
        rate = (2*p if r0 else 1-2*p) * (2*p if r1 else 1-2*p)
        rate *= (5*p if m0 else 1-5*p) * (5*p if m1 else 1-5*p)
        for a, b in itertools.product("IXYZ", repeat=2):
            branch = 1-p if a == b == "I" else p/15
            expected[bool(r1 ^ m1 ^ (b in "XY")), bool(r0 ^ m0 ^ (a in "XY"))] += rate*branch
    circuit = SI1000(p).apply(plan).to_tick_circuit()
    shots = _run(circuit, shots=20_000, seed=38)
    assert {key: shots.count(key)/len(shots) for key in expected} == pytest.approx(expected, abs=.012)
    assert _run(circuit) == _run(circuit)


def test_empty_support_rejection_matches_ideal_tick_boundary():
    plan = ScheduledCircuit(1, ((CircuitOp("R", (0,)),), (CircuitOp("M", (0,)),)), (
        MeasurementParity((), "constant", constant=True),
    ), ())
    with pytest.raises(ValueError, match="nonempty"):
        SI1000(.001).apply(plan).to_tick_circuit()


@pytest.mark.parametrize("p", [0, 1e-20])
def test_clifford_aliases_and_nonascending_measurements(p):
    op, parity = CircuitOp, MeasurementParity
    names = ["H", "H", "S", "S_DAG", "C_XYZ", "C_ZYX", "SQRT_X", "SQRT_X", "SQRT_X", "SQRT_X", "Z", "Z", "X"]
    plan = ScheduledCircuit(2, (
        (op("R", (0,)), op("R", (1,))),
        *((op(name, (0,)),) for name in names),
        (op("M", (1,)), op("M", (0,))),
    ), (parity((1,), "signed", constant=True),), (parity((0,), "zero"),))
    noisy = SI1000(p).apply(plan)
    assert _run(noisy.to_tick_circuit()) == noisy.compile_native().sample_records(32, seed=42)
    assert _run(noisy.to_tick_circuit()) == ((False, True),) * 32
