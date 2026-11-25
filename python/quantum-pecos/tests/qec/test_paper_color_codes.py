"""Independent circuit/instrument oracles for the two cited color-code papers."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest
from pecos.qec.color import ColorExtraction, color_code_implementation, color_code_spec, color_memory
from pecos.qec.color.paper_codes import bell_flag_projection
from pecos.qec.generic.scheduled import CircuitOp, MeasurementParity, ScheduledCircuit, SchedulePolicy

stim = pytest.importorskip("stim")
FIXTURES = Path(__file__).parent / "fixtures" / "paper_color_codes"


def _parts(circuit):
    """Canonical physical moments and absolute-record parities, independent of emitter."""
    moments, current, detectors, observable = [], [], [], set()
    count = 0
    for instruction in circuit.flattened():
        name = instruction.name
        targets = instruction.targets_copy()
        if name == "TICK":
            moments.append(tuple(sorted(current)))
            current = []
        elif name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            records = {count + target.value for target in targets}
            if name == "DETECTOR":
                detectors.append(tuple(sorted(records)))
            else:
                observable ^= records
        elif name not in {"QUBIT_COORDS", "SHIFT_COORDS"}:
            arity = 2 if name in {"CX", "CZ"} else 1
            current.extend(
                (name, tuple(t.value for t in targets[i : i + arity])) for i in range(0, len(targets), arity)
            )
            if name == "M":
                count += len(targets)
    if current:
        moments.append(tuple(sorted(current)))
    return moments, detectors, tuple(sorted(observable))


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("488_*.stim")), ids=lambda path: path.stem)
def test_brickwork_matches_independent_release(path):
    _, distance, gates, basis, phase = path.stem.split("_")
    choice = ColorExtraction.BRICKWORK_CX if gates == "cnot" else ColorExtraction.BRICKWORK_CZ
    impl = color_code_implementation(color_code_spec("488", int(distance[1:])), choice)
    plan = color_memory(impl, 3, basis, start_phase=int(phase[1:]))
    assert _parts(plan.to_stim()) == _parts(stim.Circuit(path.read_text()))
    if choice == ColorExtraction.BRICKWORK_CZ:
        assert all(len(cycle.moments) == (23 if impl.code.distance == 3 else 24) for cycle in impl.cycles)


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_hexagonal_geometry_and_refresh_coverage(distance):
    oracle = json.loads((FIXTURES / "666_geometry.json").read_text())[str(distance)]
    code = color_code_spec("666", distance)
    assert code.num_data == oracle["n"] == (3 * distance**2 + 1) // 4
    assert code.face_slots == tuple(map(tuple, oracle["faces"]))
    impl = color_code_implementation(code, ColorExtraction.SWAP_SUPERDENSE)
    counts = Counter()
    for cycle in impl.cycles:
        phase = Counter(op.qubits[0] for moment in cycle.moments for op in moment if op.name == "RELABEL")
        assert max(phase.values()) == 1
        counts.update(phase)
        assert sum(any(op.name == "CX" for op in moment) for moment in cycle.moments) == 7
    assert dict(counts) == {q: sum(q in face for face in code.faces) for q in range(code.num_data)}


def _choi(n, total):
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(total + n)
    simulator.h(*range(n))
    for q in range(n):
        simulator.cx(q, total + q)
    return simulator


def _project(simulator, axis, face, n, bit):
    pauli = stim.PauliString(n)
    for q in face:
        pauli[q] = axis
    simulator.postselect_observable(pauli, desired_value=bool(bit))


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("choice", [ColorExtraction.SUPERDENSE, ColorExtraction.SWAP_SUPERDENSE])
def test_hexagonal_choi_instrument_against_ideal_projectors(distance, choice):
    """Checks the full conditional channel, not merely an initialized codeword.

    Oracle: Fig. 1 / Supplement B's explicit X-odd-slot feedback followed by
    comparison to ideal face projectors. No CSSFlow maps are used here.
    """
    code = color_code_spec("666", distance)
    impl = color_code_implementation(code, choice)
    n, total, nf = code.num_data, impl.num_qubits, len(code.faces)
    rng = random.Random(901)
    for cycle in impl.cycles:
        for _ in range(4):
            bits = [rng.randrange(2) for _ in range(2 * nf)]
            actual = _choi(n, total)
            carriers = list(range(total))
            event = 0
            for moment in cycle.moments:
                for op in moment:
                    if op.name == "RELABEL":
                        a, b = op.qubits
                        carriers[a], carriers[b] = carriers[b], carriers[a]
                    elif op.name == "M":
                        q = carriers[op.qubits[0]]
                        actual.postselect_z(q, desired_value=bool(bits[event]))
                        actual.reset(q)
                        event += 1
                    else:
                        actual.do(stim.Circuit(f"{op.name} " + " ".join(str(carriers[q]) for q in op.qubits)))
            # Canonicalize carrier coordinates for comparison only; no export gate.
            for slot in range(total):
                if carriers[slot] != slot:
                    other = carriers[slot]
                    actual.swap(slot, other)
                    carriers = [other if q == slot else slot if q == other else q for q in carriers]
            for face, slots in enumerate(code.face_slots):
                if bits[nf + face]:
                    for q in slots[1::2]:
                        if q >= 0:
                            actual.x(q)
            expected = _choi(n, total)
            for axis, outcomes in (("X", bits[:nf]), ("Z", bits[nf:])):
                for face, bit in zip(code.faces, outcomes, strict=True):
                    _project(expected, axis, face, n, bit)
            assert actual.canonical_stabilizers() == expected.canonical_stabilizers()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_bell_flag_projector_has_deterministic_flags_on_arbitrary_input(basis):
    code = color_code_spec("666", 3)
    n, nf, total = code.num_data, len(code.faces), 13
    for branch in range(1 << nf):
        bits = [(branch >> i) & 1 for i in range(nf)]
        actual = _choi(n, total)
        event = 0
        for moment in bell_flag_projection(code, basis):
            for op in moment:
                if op.name == "M":
                    q = op.qubits[0]
                    if event < nf:
                        actual.postselect_z(q, desired_value=bool(bits[event]))
                    else:
                        assert actual.peek_z(q) == 1
                    actual.reset(q)
                    event += 1
                else:
                    actual.do(stim.Circuit(f"{op.name} " + " ".join(map(str, op.qubits))))
        expected = _choi(n, total)
        for face, bit in zip(code.faces, bits, strict=True):
            _project(expected, "X" if basis == "Z" else "Z", face, n, bit)
        assert actual.canonical_stabilizers() == expected.canonical_stabilizers()


CASES = [
    ("488", 3, ColorExtraction.BRICKWORK_CX),
    ("488", 7, ColorExtraction.BRICKWORK_CZ),
    ("666", 3, ColorExtraction.SUPERDENSE),
    ("666", 5, ColorExtraction.SWAP_SUPERDENSE),
    ("666", 7, ColorExtraction.SWAP_SUPERDENSE),
]


@pytest.mark.parametrize(("family", "distance", "choice"), CASES)
@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_memory_boundaries_and_alternate_schedule(family, distance, choice, rounds, basis):
    impl = color_code_implementation(color_code_spec(family, distance), choice)
    for phase in range(impl.period):
        reference = color_memory(impl, rounds, basis, start_phase=phase)
        packed = reference.rescheduled(SchedulePolicy.ASAP_UNITARIES)
        assert packed.output_carriers == reference.output_carriers
        assert packed.detectors == reference.detectors
        assert len(packed.moments) <= len(reference.moments)
        for plan in (reference, packed):
            circuit = plan.to_stim()
            circuit.detector_error_model()
            assert not circuit.compile_detector_sampler(seed=17).sample(shots=32, append_observables=True).any()
            raw = circuit.compile_sampler(seed=18).sample(shots=8)
            for record in raw:
                detectors, observables = plan.evaluate(record)
                assert not any(detectors)
                assert not any(observables)
        if choice == ColorExtraction.SWAP_SUPERDENSE:
            assert reference.output_carriers != tuple(range(impl.num_qubits))


def test_scheduling_is_a_real_separate_choice():
    op = CircuitOp
    plan = ScheduledCircuit(
        2,
        (
            (op("R", (0,)), op("R", (1,))),
            (op("H", (0,)),),
            (op("X", (1,)),),
            (op("M", (0,)), op("M", (1,))),
        ),
        (),
        (),
    )
    packed = plan.rescheduled(SchedulePolicy.ASAP_UNITARIES)
    assert len(packed.moments) == 3
    assert plan.num_measurements == packed.num_measurements
    with pytest.raises(ValueError, match="cannot be recovered"):
        packed.rescheduled(SchedulePolicy.REFERENCE)


@pytest.mark.parametrize("bad", [True, 0, 2, 5, 6])
def test_invalid_brickwork_distance(bad):
    with pytest.raises(ValueError, match="distance must be"):
        color_code_spec("488", bad)


@pytest.mark.parametrize("family", ["488", "666"])
def test_distance_validation_is_independent_of_cache(family):
    color_code_spec(family, 3)
    with pytest.raises(ValueError, match="distance must be"):
        color_code_spec(family, 3.0)


@pytest.mark.parametrize("name", ["qubit", "measure", "result", "cx", "not a name", "def"])
def test_guppy_function_name_validation(name):
    implementation = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SUPERDENSE)
    plan = color_memory(implementation, 1)
    with pytest.raises(ValueError, match="function_name"):
        plan.guppy_source(function_name=name)


def test_invalid_choices_and_lifetimes():
    code = color_code_spec("666", 3)
    with pytest.raises(ValueError, match="does not implement"):
        color_code_implementation(code, ColorExtraction.BRICKWORK_CX)
    with pytest.raises(TypeError, match="ColorExtraction"):
        color_code_implementation(code, "superdense")
    impl = color_code_implementation(code, ColorExtraction.SWAP_SUPERDENSE)
    for bad in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="rounds must be"):
            color_memory(impl, bad)
    for bad in [-1, 3, True]:
        with pytest.raises(ValueError, match="start_phase is outside"):
            color_memory(impl, 1, start_phase=bad)
    with pytest.raises(ValueError, match="non-live"):
        ScheduledCircuit(1, ((CircuitOp("H", (0,)),),), (), ())
    with pytest.raises(ValueError, match="overlapping"):
        ScheduledCircuit(1, ((CircuitOp("R", (0,)), CircuitOp("R", (0,))),), (), ())
    with pytest.raises(ValueError, match="nonexistent"):
        ScheduledCircuit(1, ((CircuitOp("R", (0,)),), (CircuitOp("M", (0,)),)), (MeasurementParity((1,), "bad"),), ())
