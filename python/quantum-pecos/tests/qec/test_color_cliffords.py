"""Independent signed logical-action and six-state color-code checks."""

import itertools
import random

import pytest
from pecos.qec.color import (
    ColorCliffordExperiment,
    ColorExtraction,
    LogicalState,
    color_code_implementation,
    color_code_spec,
    color_state_memory,
    color_state_memory_program,
    logical_clifford_gadget,
)
from pecos.qec.generic import NamedParity, SchedulePolicy

stim = pytest.importorskip("stim")


def _logical(n, axis):
    x, z = stim.PauliString("X" * n), stim.PauliString("Z" * n)
    return {"I": stim.PauliString(n), "X": x, "Y": 1j * x * z, "Z": z}[axis]


def _pauli(n, axis, support):
    p = stim.PauliString(n)
    for q in support:
        p[q] = axis
    return p


@pytest.mark.parametrize("distance", [3, 5, 7, 9])
@pytest.mark.parametrize("gate", ["H", "S", "S_DAG", "X", "Z"])
def test_exact_signed_logical_and_stabilizer_conjugation(distance, gate):
    code = color_code_spec("666", distance)
    gadget = logical_clifford_gadget(code, gate)
    at = {role: q for q, role in enumerate(gadget.roles)}
    circuit = stim.Circuit()
    for moment in gadget.moments:
        for op in moment:
            circuit.append(op.name, [at[r] for r in op.roles])
    tableau = circuit.to_tableau()
    # Independent oracle: the ideal one-qubit gate, including every Pauli sign.
    ideal = stim.Tableau.from_named_gate(gate)
    for axis in "XYZ":
        image = ideal(stim.PauliString(axis))
        expected = image.sign * _logical(code.num_data, "IXYZ"[image[0]])
        assert tableau(_logical(code.num_data, axis)) == expected
    # Balanced S/Sdag support must preserve the positive stabilizer group.
    for support in code.faces:
        x, z = (_pauli(code.num_data, axis, support) for axis in "XZ")
        if gate == "H":
            assert tableau(x) == z
            assert tableau(z) == x
        elif gate in {"S", "S_DAG"}:
            assert tableau(x) == x * z
            assert tableau(z) == z
        else:
            assert tableau(x) == x
            assert tableau(z) == z


@pytest.mark.parametrize("distance", [3, 5, 7])
def test_cx_full_signed_two_logical_pauli_action(distance):
    code = color_code_spec("666", distance)
    n = code.num_data
    gadget = logical_clifford_gadget(code, "CX")
    at = {role: q for q, role in enumerate(gadget.roles)}
    c = stim.Circuit()
    for layer in gadget.moments:
        for op in layer:
            c.append(op.name, [at[r] for r in op.roles])
    actual, ideal = c.to_tableau(), stim.Tableau.from_named_gate("CX")
    for a, b in itertools.product("IXYZ", repeat=2):
        image = ideal(stim.PauliString(a + b))
        expected = image.sign * (_logical(n, "IXYZ"[image[0]]) + _logical(n, "IXYZ"[image[1]]))
        assert actual(_logical(n, a) + _logical(n, b)) == expected


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("choice", [ColorExtraction.SUPERDENSE, ColorExtraction.SWAP_SUPERDENSE])
@pytest.mark.parametrize("state", list(LogicalState))
def test_six_state_memories_all_phases_and_schedules(distance, choice, state):
    impl = color_code_implementation(color_code_spec("666", distance), choice)
    for phase in range(impl.period):
        program = color_state_memory_program(impl, 4, state, start_phase=phase)
        for schedule in SchedulePolicy:
            plan = program.compile(schedule)
            circuit = plan.to_stim()
            circuit.detector_error_model()
            assert not circuit.compile_detector_sampler(seed=123).sample(32, append_observables=True).any()
            assert all(not p.constant for p in (*plan.detectors, *plan.observables))
            raw = circuit.compile_sampler(seed=234).sample(8)
            for shot in raw:
                detectors, observables = plan.evaluate(shot)
                assert not any(detectors)
                assert not any(observables)


def _execute_open_program(program, impl, blocks, seed):
    """Independent named-gate interpreter; no production compiler or renderer."""
    roles = [f"{block}:{role}" for block in blocks for role in impl.roles]
    carriers = {role: i for i, role in enumerate(roles)}
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(len(roles))
    events = {}
    for call in program.calls:
        bound = dict(call.bindings)
        for moment in call.gadget.moments:
            for op in moment:
                targets = [carriers[bound[r]] for r in op.roles]
                if op.name == "RELABEL":
                    a, b = [bound[r] for r in op.roles]
                    carriers[a], carriers[b] = carriers[b], carriers[a]
                elif op.name == "M":
                    events[f"{call.label}/{op.outcome}"] = sim.measure(targets[0])
                else:
                    primitive = stim.Circuit()
                    primitive.append(op.name, targets)
                    sim.do(primitive)
    return sim, events, carriers


def _eval(parity, events):
    value = parity.constant
    for term in parity.terms:
        value ^= events[term]
    return value


def _physical_logical(impl, block, axis, carriers, size):
    x = _pauli(size, "X", [carriers[f"{block}:data[{q}]"] for q in impl.code.logical_support])
    z = _pauli(size, "Z", [carriers[f"{block}:data[{q}]"] for q in impl.code.logical_support])
    return {"I": stim.PauliString(size), "X": x, "Y": 1j * x * z, "Z": z}[axis]


@pytest.mark.parametrize("state", list(LogicalState))
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_preparation_is_the_requested_signed_state_before_any_inverse(state, distance):
    impl = color_code_implementation(color_code_spec("666", distance), ColorExtraction.SWAP_SUPERDENSE)
    p = ColorCliffordExperiment(impl)
    p.prepare("memory", state)
    for seed in (17, 18):
        actual, events, carriers = _execute_open_program(p, impl, ("memory",), seed)
        for axis in "XYZ":
            expectation = actual.peek_observable_expectation(
                _physical_logical(impl, "memory", axis, carriers, impl.num_qubits),
            )
            frame = p.frame("memory")
            sign = {"X": frame.x_sign, "Z": frame.z_sign, "Y": frame.x_sign ^ frame.z_sign}[axis]
            expectation *= (-1) ** _eval(sign, events)
            expected = (-1 if state.value[0] == "-" else 1) if state.value[1] == axis else 0
            assert expectation == expected


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("choice", [ColorExtraction.SUPERDENSE, ColorExtraction.SWAP_SUPERDENSE])
def test_interleaved_cliffords_and_extraction_against_ideal_two_qubit_state(distance, choice):
    """Full logical Pauli tomography, including entangled/Y states and random branches."""
    impl = color_code_implementation(color_code_spec("666", distance), choice)
    p = ColorCliffordExperiment(impl, ("alice", "bob"))
    p.prepare("alice", "+Y", start_phase=impl.period - 1)
    p.prepare("bob", "-X")
    ideal = stim.TableauSimulator()
    ideal.set_num_qubits(2)
    ideal.h(0)
    ideal.s(0)
    ideal.x(1)
    ideal.h(1)
    rng = random.Random(67)
    sequence = [("CX", 0, 1), ("H", 0), ("S", 1), ("CX", 1, 0)]
    sequence += [(rng.choice(("H", "S", "S_DAG", "X", "Z")), rng.randrange(2)) for _ in range(8)]
    blocks = ("alice", "bob")
    for gate, *targets in sequence:
        p.extract("alice")
        p.extract("bob")
        p.gate(gate, *(blocks[q] for q in targets))
        c = stim.Circuit()
        c.append(gate, targets)
        ideal.do(c)
    # Distinct phases/round counts also move data to different reference carriers.
    p.extract("alice", 2)
    p.extract("bob", 1)
    for seed in (731, 732, 733):
        actual, events, carriers = _execute_open_program(p, impl, blocks, seed)
        assert all(not _eval(det.parity, events) for det in p.detectors)
        for block in blocks:
            frame = p.frame(block)
            for axis, checks in (("X", frame.check_x), ("Z", frame.check_z)):
                for support, expected in zip(impl.code.faces, checks, strict=True):
                    operator = _pauli(2 * impl.num_qubits, axis, [carriers[f"{block}:data[{q}]"] for q in support])
                    assert actual.peek_observable_expectation(operator) == (-1) ** _eval(expected, events)
        for a, b in itertools.product("IXYZ", repeat=2):
            physical = stim.PauliString(2 * impl.num_qubits)
            correction = NamedParity()
            for block, axis in zip(blocks, (a, b), strict=True):
                physical *= _physical_logical(impl, block, axis, carriers, 2 * impl.num_qubits)
                frame = p.frame(block)
                if axis in "XY":
                    correction ^= frame.x_sign
                if axis in "ZY":
                    correction ^= frame.z_sign
            observed = actual.peek_observable_expectation(physical) * (-1) ** _eval(correction, events)
            assert observed == ideal.peek_observable_expectation(stim.PauliString(a + b))


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_bell_correlations_after_relabeling_and_extraction(basis):
    impl = color_code_implementation(color_code_spec("666", 5), ColorExtraction.SWAP_SUPERDENSE)
    p = ColorCliffordExperiment(impl, ("alice", "bob"))
    p.prepare("alice", "+X", start_phase=2)
    p.prepare("bob")
    p.extract("alice", 2)
    p.extract("bob", 3)
    p.gate("CX", "alice", "bob")
    p.extract("alice")
    p.extract("bob", 2)
    parity = p.readout("alice", basis) ^ p.readout("bob", basis)
    # Bell Phi+ has XX=ZZ=+1, YY=-1, independently of code distance.
    p.observable("Bell correlation", parity ^ NamedParity(constant=basis == "Y"))
    _, events, _ = _execute_open_program(p, impl, ("alice", "bob"), 73)
    assert all(not _eval(det.parity, events) for det in (*p.detectors, *p.observables))


def test_invalid_inputs_leave_protocol_unchanged():
    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    p = ColorCliffordExperiment(impl, ("alice", "bob"))
    for invalid in ("Y", "unknown", 1):
        with pytest.raises(ValueError, match="not a valid LogicalState"):
            p.prepare("alice", invalid)
        assert not p.calls
    p.prepare("alice")
    before = p.calls, p.frame("alice")
    for gate, target, error in (
        ("CX", "alice", "distinct live target"),
        ("CX", "bob", "must be prepared"),
        ("H", "bob", "Only CX"),
        ("T", None, "Unsupported logical Clifford"),
    ):
        with pytest.raises(ValueError, match=error):
            p.gate(gate, "alice", target)
        assert (p.calls, p.frame("alice")) == before
    with pytest.raises(ValueError, match="positive integer"):
        p.extract("alice", True)
    with pytest.raises(ValueError, match="live roles"):
        p.compile()
    with pytest.raises(ValueError, match="readout basis"):
        p.readout("alice", "bad")
    p.readout("alice")
    with pytest.raises(ValueError, match="not yet read out"):
        p.gate("H", "alice")
    with pytest.raises(ValueError, match="already prepared"):
        p.prepare("alice")
    with pytest.raises(ValueError, match="canonical triangular 666"):
        logical_clifford_gadget(color_code_spec("488", 3), "S")


def test_phase_counter_is_continuous_across_cliffords():
    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    program = ColorCliffordExperiment(impl)
    program.prepare("memory", start_phase=2)
    assert program.next_phase("memory") == 2
    program.extract("memory")
    assert program.next_phase("memory") == 0
    program.gate("S", "memory")
    assert program.next_phase("memory") == 0
    program.extract("memory", 2)
    assert program.next_phase("memory") == 2


def test_block_namespace_cannot_collide_with_generated_calls():
    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    with pytest.raises(ValueError, match="Block names must not contain"):
        ColorCliffordExperiment(impl, ("a:prepare", "gate[4]:H:a"))
