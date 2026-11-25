"""Author references and independent signed-instrument checks for periodic 488."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import stim
from pecos.qec.color import (
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
    periodic_488_memory_program,
)
from pecos.qec.generic import NamedParity, SchedulePolicy
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

FIXTURES = Path(__file__).parent / "fixtures/periodic488"


def _rank(rows):
    pivots = {}
    for original in rows:
        row = original
        while row:
            bit = row.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = row
                break
            row ^= pivots[bit]
    return len(pivots)


def _physical(circuit):
    """Retain exact moments, ignoring annotation placement and listing order."""
    moments, current = [], []
    for inst in circuit:
        if inst.name == "TICK":
            moments.append(sorted(current))
            current = []
        elif inst.name in {"R", "H", "CX", "M"}:
            targets = [t.value for t in inst.targets_copy()]
            arity = 2 if inst.name == "CX" else 1
            current.extend((inst.name, tuple(targets[i : i + arity])) for i in range(0, len(targets), arity))
    if current:
        moments.append(sorted(current))
    return moments


def _parities(circuit):
    count, detectors, observables, measured = 0, [], [set() for _ in range(4)], []
    for inst in circuit:
        if inst.name == "M":
            measured.extend(t.value for t in inst.targets_copy())
            count += len(inst.targets_copy())
        elif inst.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            refs = {count + t.value for t in inst.targets_copy()}
            if inst.name == "DETECTOR":
                detectors.append(tuple(sorted(refs)))
            else:
                observables[int(inst.gate_args_copy()[0])] ^= refs
    return detectors, [tuple(sorted(s)) for s in observables], measured


@pytest.mark.parametrize("distance", [4, 8, 12])
def test_geometry_logical_pairing_and_spatial_schedule(distance):
    code = periodic_488_code(distance)
    reference = json.loads((FIXTURES / f"geometry_d{distance}.json").read_text())
    assert code.num_data == distance**2
    assert code.num_logicals == 4
    assert code.data_coordinates == tuple(tuple(r["coord"]) for r in reference["data_qubits"])
    assert code.faces == tuple(tuple(f["data_support"]) for f in reference["plaquettes"])
    assert code.period_vectors == ((-distance, -distance), (-distance, distance))
    assert code.logical_x_supports == tuple(tuple(s) for s in reference["canonical_logicals"]["x"])
    assert code.logical_z_supports == tuple(tuple(s) for s in reference["canonical_logicals"]["z"])
    masks = [sum(1 << q for q in face) for face in code.faces]
    assert _rank(masks) == code.num_data // 2 - 2
    assert all((a & b).bit_count() % 2 == 0 for a in masks for b in masks)
    for i, xs in enumerate(code.logical_x_supports):
        assert len(xs) == distance
        for j, zs in enumerate(code.logical_z_supports):
            assert len(set(xs) & set(zs)) % 2 == (i == j)
        assert all(len(set(xs) & set(f)) % 2 == 0 for f in code.faces)
    impl = periodic_488_implementation(code)
    gadget = impl.gadget()
    assert sum(any(g.name == "CX" for g in m) for m in gadget.moments) == 12
    assert sum(g.name == "CX" for m in gadget.moments for g in m) == 120 * (distance // 4) ** 2
    at = {name: q for q, name in enumerate(impl.roles)}
    for moment in gadget.moments:
        for gate in moment:
            if gate.name == "CX":
                assert tuple(sorted(at[r] for r in gate.roles)) in impl.hardware_edges
    assert "logical:X[3]" in code.describe()
    assert "logical:Z[3]" in gadget.describe()


@pytest.mark.parametrize("distance", [4, 8, 12])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_exact_author_memory_reference(distance, basis, rounds):
    impl = periodic_488_implementation(periodic_488_code(distance))
    plan = periodic_488_memory(impl, rounds, basis)
    reference = stim.Circuit((FIXTURES / f"d{distance}_{basis}_r{rounds}.stim").read_text())
    assert _physical(plan.to_stim()) == _physical(reference)
    dets, obs, measured = _parities(reference)
    assert [p.records for p in plan.detectors] == dets
    expected_coords = reference.get_detector_coordinates()
    assert plan.to_stim().get_detector_coordinates() == expected_coords
    assert {i: list(p.coords) for i, p in enumerate(plan.detectors)} == expected_coords
    assert {i: list(d.coords) for i, d in enumerate(plan.dem_spec()[0])} == expected_coords
    assert stim.Circuit(tick_circuit_to_stim(plan.to_tick_circuit())).get_detector_coordinates() == expected_coords
    assert [p.records for p in plan.observables] == obs
    tick = plan.to_tick_circuit()
    assert tick.measurement_emission() == [(q, i) for i, q in enumerate(measured)]
    assert _physical(stim.Circuit(tick_circuit_to_stim(tick))) == _physical(reference)
    assert len(tick.circuit_definitions()["observables"]) == 4


@pytest.mark.parametrize("distance", [4, 8, 12])
def test_complete_signed_choi_instrument_against_stim(distance):
    """All face input/output signs and eight logical correlations fix the channel.

    The ancillas are sampled on an entangled data/reference input, then reset.
    No author Pauli-propagation or memory helper is used by this oracle.
    """
    code = periodic_488_code(distance)
    impl = periodic_488_implementation(code)
    gadget = impl.gadget()
    n = code.num_data
    at = {name: q for q, name in enumerate(impl.roles)}
    for seed in (7, 31, 52):
        sim = stim.TableauSimulator(seed=seed)
        sim.set_num_qubits(3 * n)
        for q in range(n):
            sim.h(q)
            sim.cx(q, 2 * n + q)
        outcomes = {}
        for moment in gadget.moments:
            for gate in moment:
                qs = [at[r] for r in gate.roles]
                if gate.name == "M":
                    outcomes[gate.outcome] = bool(sim.measure(qs[0]))
                    sim.reset(qs[0])
                else:
                    sim.do(stim.Circuit(gate.name + " " + " ".join(map(str, qs))))

        def sign(parity, outcomes=outcomes):
            value = parity.constant
            for term in parity.terms:
                value ^= outcomes[term]
            return -1 if value else 1

        flows = {flow.name: flow.parity for flow in gadget.flows}
        generators = []
        for axis in "XZ":
            for i, face in enumerate(code.faces):
                for kind, shift in (("input", 2 * n), ("output", 0)):
                    pauli = stim.PauliString(3 * n)
                    for q in face:
                        pauli[shift + q] = axis
                    assert sim.peek_observable_expectation(pauli) == sign(flows[f"{kind}:{axis}[{i}]"])
                    x, z = pauli.to_numpy()
                    generators.append(sum(int(b) << j for j, b in enumerate([*x, *z])))
            supports = code.logical_x_supports if axis == "X" else code.logical_z_supports
            for i, support in enumerate(supports):
                pauli = stim.PauliString(3 * n)
                for q in support:
                    pauli[q] = axis
                    pauli[2 * n + q] = axis
                assert sim.peek_observable_expectation(pauli) == sign(flows[f"logical:{axis}[{i}]"])
                x, z = pauli.to_numpy()
                generators.append(sum(int(b) << j for j, b in enumerate([*x, *z])))
        assert _rank(generators) == 2 * n
        for flow in gadget.flows:
            if flow.name.startswith("constraint"):
                assert sign(flow.parity) == 1


@pytest.mark.parametrize("distance", [4, 8])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2, 4])
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_complete_deterministic_parities_and_memory_composition(distance, basis, rounds, schedule):
    impl = periodic_488_implementation(periodic_488_code(distance))
    authored = periodic_488_memory_program(impl, rounds, basis)
    assert len(authored.calls) == rounds
    plan = authored.compile(schedule)
    circuit = plan.to_stim()
    constraints = [sum(1 << i for i in p.records) for p in (*plan.detectors, *plan.observables)]
    assert _rank(constraints) == circuit.count_determined_measurements()
    for record in circuit.compile_sampler(seed=55).sample(32):
        dets, obs = plan.evaluate(record)
        assert not any(dets)
        assert obs == (False, False, False, False)
    circuit.detector_error_model()
    definitions = plan.to_tick_circuit().circuit_definitions()
    assert [o["id"] for o in definitions["observables"]] == [0, 1, 2, 3]
    assert [o["measurements"] for o in definitions["observables"]] == [list(p.records) for p in plan.observables]


def test_all_four_logical_ids_respond_to_final_data_faults():
    code = periodic_488_code(4)
    for basis, supports in (("X", code.logical_x_supports), ("Z", code.logical_z_supports)):
        plan = periodic_488_memory(periodic_488_implementation(code), 2, basis)
        record = list(plan.to_stim().reference_sample())
        for q in range(code.num_data):
            changed = record.copy()
            changed[-code.num_data + q] ^= True
            assert plan.evaluate(changed)[1] == tuple(q in support for support in supports)


@pytest.mark.parametrize("distance", [True, 0, 3, 6, 4.0])
def test_distance_validation_precedes_cache(distance):
    periodic_488_code(4)
    with pytest.raises(ValueError, match="distance must"):
        periodic_488_code(distance)


def test_factories_reject_noncanonical_replacements():
    code = periodic_488_code(4)
    with pytest.raises(ValueError, match="canonical code"):
        periodic_488_implementation(replace(code, logical_x_supports=code.logical_x_supports[::-1]))
    impl = periodic_488_implementation(code)
    with pytest.raises(ValueError, match="canonical periodic"):
        periodic_488_memory(replace(impl, extraction=replace(impl.extraction, flows=())), 2)
    for rounds in (0, True, 2.0):
        with pytest.raises(ValueError, match="rounds must"):
            periodic_488_memory(impl, rounds)
    with pytest.raises(ValueError, match="basis must"):
        periodic_488_memory(impl, 2, "Y")


def test_source_and_reference_provenance():
    from pecos.qec.color import _shutty_torus488

    root = Path(_shutty_torus488.__file__).parent
    for name, entry in json.loads((root / "provenance.json").read_text())["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entry["sha256"]
    for name, digest in json.loads((FIXTURES / "sha256.json").read_text()).items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("distance", [4, 8, 12, 16])
def test_every_named_signed_flow_with_independent_stim_checker(distance):
    """Stim's signed flow check uses randomized trials (false positive <= 2^-256)."""
    code = periodic_488_code(distance)
    impl = periodic_488_implementation(code)
    gadget = impl.gadget()
    indices = {r: q for q, r in enumerate(impl.roles)}
    events = {name: i for i, name in enumerate(gadget.outcomes)}
    circuit = stim.Circuit()
    for moment in gadget.moments:
        for gate in moment:
            circuit.append(gate.name, [indices[r] for r in gate.roles])

    def pauli(axis, support):
        result = stim.PauliString(impl.num_qubits)
        for q in support:
            result[q] = axis
        return result

    for flow in gadget.flows:
        incoming = stim.PauliString(impl.num_qubits)
        outgoing = stim.PauliString(impl.num_qubits)
        if not flow.name.startswith("constraint"):
            kind, operator = flow.name.split(":")
            axis = operator[0]
            index = int(operator[2:-1])
            if kind == "logical":
                supports = code.logical_x_supports if axis == "X" else code.logical_z_supports
                incoming = pauli(axis, supports[index])
                outgoing = pauli(axis, supports[index])
            elif kind == "input":
                incoming = pauli(axis, code.faces[index])
            else:
                assert kind == "output"
                outgoing = pauli(axis, code.faces[index])
        if flow.parity.constant:
            outgoing *= -1
        assert circuit.has_flow(
            stim.Flow(
                input=incoming,
                output=outgoing,
                measurements=[events[t] for t in flow.parity.terms],
            ),
        ), flow.name


def test_recurring_calls_reuse_the_named_extraction_contract():
    impl = periodic_488_implementation(periodic_488_code(4))
    program = periodic_488_memory_program(impl, 4, "Z")
    assert program.calls[1].gadget is impl.gadget()
    assert program.calls[2].gadget is impl.gadget()
