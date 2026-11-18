"""Signed native-CZ paper reference, physical schedule, and instrument checks."""

import hashlib
import json
from pathlib import Path

import pytest
import stim
from pecos.qec.color import periodic_488_code, periodic_488_implementation, periodic_488_memory
from pecos.qec.generic import SchedulePolicy, evaluate_tick_circuit
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

FIXTURES = Path(__file__).parent / "fixtures/periodic488"


def _moments(circuit):
    moments, current = [], []
    for inst in circuit:
        if inst.name == "TICK":
            moments.append(sorted(current))
            current = []
        elif inst.name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "MPAD"}:
            targets = [t.value for t in inst.targets_copy()]
            arity = 2 if inst.name == "CZ" else 1
            current.extend((inst.name, tuple(targets[i : i + arity])) for i in range(0, len(targets), arity))
    if current:
        moments.append(sorted(current))
    return moments


def _parities(circuit):
    count, detectors, observables = 0, [], [set() for _ in range(4)]
    for inst in circuit:
        if inst.name == "M":
            count += len(inst.targets_copy())
        elif inst.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            records = {count + t.value for t in inst.targets_copy()}
            if inst.name == "DETECTOR":
                detectors.append(tuple(sorted(records)))
            else:
                observables[int(inst.gate_args_copy()[0])] ^= records
    return detectors, [tuple(sorted(s)) for s in observables]


@pytest.mark.parametrize("distance", [4, 8, 12])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_exact_native_author_memory_and_independent_reference_signs(distance, basis, rounds):
    impl = periodic_488_implementation(periodic_488_code(distance), gate_set="native-cz")
    plan = periodic_488_memory(impl, rounds, basis)
    reference = stim.Circuit((FIXTURES / f"native_d{distance}_{basis}_r{rounds}.stim").read_text())
    assert len(plan.moments) == 25 * rounds
    assert _moments(plan.to_stim()) == _moments(reference)
    dets, obs = _parities(reference)
    assert [p.records for p in plan.detectors] == dets
    expected_coords = reference.get_detector_coordinates()
    for actual in (
        plan.to_stim().get_detector_coordinates(),
        {i: list(p.coords) for i, p in enumerate(plan.detectors)},
        {i: list(d.coords) for i, d in enumerate(plan.dem_spec()[0])},
        stim.Circuit(tick_circuit_to_stim(plan.to_tick_circuit())).get_detector_coordinates(),
    ):
        assert actual.keys() == expected_coords.keys()
        for index, coords in expected_coords.items():
            # The external Stim text fixtures round centroid coordinates to six
            # significant digits; the in-memory generator retains full precision.
            assert actual[index] == pytest.approx(coords, rel=5e-6, abs=5e-7)
    assert [p.records for p in plan.observables] == obs
    sample = reference.reference_sample()
    for p in (*plan.detectors, *plan.observables):
        assert sum(int(sample[i]) for i in p.records) % 2 == p.constant
    assert any(p.constant for p in plan.detectors)
    tick = plan.to_tick_circuit()
    assert tick.num_ticks() == 25 * rounds
    assert tick.num_measurements() == plan.num_measurements
    names = {"F": "C_XYZ", "Fdg": "C_ZYX", "SX": "SQRT_X", "SZ": "S", "SZdg": "S_DAG", "PZ": "R", "MZ": "M"}
    actual = []
    for i in range(tick.num_ticks()):
        ops = []
        for gate in tick.get_tick(i).gate_batches():
            name = names.get(gate.gate_type.name, gate.gate_type.name)
            qs = list(map(int, gate.qubits))
            arity = 2 if name == "CZ" else 1
            ops.extend((name, tuple(qs[j : j + arity])) for j in range(0, len(qs), arity))
        actual.append(sorted(ops))
    assert actual == _moments(reference)


@pytest.mark.parametrize("distance", [4, 8, 12, 16])
def test_signed_gadget_flows_in_physical_h_frame(distance):
    code = periodic_488_code(distance)
    impl = periodic_488_implementation(code, gate_set="native-cz")
    gadget = impl.gadget()
    assert len(gadget.moments) == 25
    assert gadget.input_h_frame == gadget.output_h_frame
    assert gadget.input_h_frame
    indices = {role: i for i, role in enumerate(impl.roles)}
    frames = {indices[r] for r in gadget.input_h_frame}
    events = {name: i for i, name in enumerate(gadget.outcomes)}
    circuit = stim.Circuit()
    for moment in gadget.moments:
        for gate in moment:
            circuit.append(gate.name, [indices[r] for r in gate.roles])

    def pauli(axis, support):
        result = stim.PauliString(impl.num_qubits)
        for q in support:
            result[q] = ("Z" if axis == "X" else "X") if q in frames else axis
        return result

    for flow in gadget.flows:
        incoming = stim.PauliString(impl.num_qubits)
        outgoing = stim.PauliString(impl.num_qubits)
        if not flow.name.startswith("constraint"):
            kind, op = flow.name.split(":")
            axis, index = op[0], int(op[2:-1])
            if kind == "logical":
                support = (code.logical_x_supports if axis == "X" else code.logical_z_supports)[index]
                incoming = pauli(axis, support)
                outgoing = pauli(axis, support)
            elif kind == "input":
                incoming = pauli(axis, code.faces[index])
            else:
                outgoing = pauli(axis, code.faces[index])
        if flow.parity.constant:
            outgoing *= -1
        assert circuit.has_flow(
            stim.Flow(input=incoming, output=outgoing, measurements=[events[t] for t in flow.parity.terms]),
        ), flow.name


@pytest.mark.parametrize("distance", [4, 8])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2, 4])
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_native_signed_memory_both_schedules_and_backends(distance, basis, rounds, schedule):
    impl = periodic_488_implementation(periodic_488_code(distance), gate_set="native-cz")
    plan = periodic_488_memory(impl, rounds, basis, schedule=schedule)
    tick = plan.to_tick_circuit()
    for circuit in (plan.to_stim(), stim.Circuit(tick_circuit_to_stim(tick))):
        circuit.detector_error_model()
        for raw in circuit.compile_sampler(seed=15).sample(32):
            assert plan.evaluate_stim_record(raw) == ((False,) * len(plan.detectors), (False,) * 4)
            assert evaluate_tick_circuit(tick, raw[:-1]) == plan.evaluate_stim_record(raw)


def test_native_source_provenance():
    from pecos.qec.color import _shutty_torus488_cz

    root = Path(_shutty_torus488_cz.__file__).parent
    for name, info in json.loads((root / "provenance.json").read_text())["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == info["sha256"]


def test_gate_set_must_be_an_explicit_supported_construction():
    with pytest.raises(ValueError, match="gate_set"):
        periodic_488_implementation(periodic_488_code(4), gate_set="cz")
