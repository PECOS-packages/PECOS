# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Smoke tests for the traced-QIS surface-code route after Clifford lowering."""

import random

from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos.quantum import TickCircuit
from pecos_rslib_exp import (
    Mast,
    StabMps,
    depolarizing,
    fault_catalog,
    meas_sampling,
    sim_neo,
    stabilizer,
    statevec,
)

ONE_Q_INVERSES = {
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "H": "H",
    "F": "Fdg",
    "Fdg": "F",
    "SX": "SXdg",
    "SXdg": "SX",
    "SY": "SYdg",
    "SYdg": "SY",
    "SZ": "SZdg",
    "SZdg": "SZ",
}

TWO_Q_INVERSES = {
    "CX": "CX",
    "CY": "CY",
    "CZ": "CZ",
    "SXX": "SXXdg",
    "SXXdg": "SXX",
    "SYY": "SYYdg",
    "SYYdg": "SYY",
    "SZZ": "SZZdg",
    "SZZdg": "SZZ",
    "SWAP": "SWAP",
}

TICK_1Q_METHODS = {
    "X": "x",
    "Y": "y",
    "Z": "z",
    "H": "h",
    "F": "f",
    "Fdg": "fdg",
    "SX": "sx",
    "SXdg": "sxdg",
    "SY": "sy",
    "SYdg": "sydg",
    "SZ": "sz",
    "SZdg": "szdg",
}

TICK_2Q_METHODS = {
    "CX": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SXX": "sxx",
    "SXXdg": "sxxdg",
    "SYY": "syy",
    "SYYdg": "syydg",
    "SZZ": "szz",
    "SZZdg": "szzdg",
    "SWAP": "swap",
}


def build_lowered_traced_qis_surface_code(rounds=3):
    patch = SurfacePatch.create(distance=3)
    tc = _build_surface_tick_circuit_for_native_model(patch, rounds, "Z", circuit_source="traced_qis")
    tc.lower_clifford_rotations()
    return tc


def traced_qis_noise():
    return depolarizing().p1(0.0003).p2(0.003).p_meas(0.0015).p_prep(0.0015)


def zero_noise():
    return depolarizing().p1(0).p2(0).p_meas(0).p_prep(0)


def build_explicit_clifford_gate_circuit():
    tc = TickCircuit()
    tc.tick().szdg([0])
    tc.tick().sx([0])
    tc.tick().sxdg([1])
    tc.tick().sy([0])
    tc.tick().sydg([1])
    tc.tick().f([0])
    tc.tick().fdg([1])
    tc.tick().cy([(0, 1)])
    tc.tick().cz([(0, 1)])
    tc.tick().sxx([(0, 1)])
    tc.tick().sxxdg([(0, 1)])
    tc.tick().syy([(0, 1)])
    tc.tick().syydg([(0, 1)])
    tc.tick().szz([(0, 1)])
    tc.tick().szzdg([(0, 1)])
    tc.tick().swap([(0, 1)])
    tc.tick().mz([0, 1])
    tc.set_meta("num_measurements", "2")
    tc.set_meta("detectors", "[]")
    tc.set_meta("observables", "[]")
    return tc


def random_standard_clifford_sequence(seed, depth=14, num_qubits=3):
    rng = random.Random(seed)
    sequence = []
    for _ in range(depth):
        if rng.random() < 0.55:
            gate = rng.choice(tuple(ONE_Q_INVERSES))
            qubits = (rng.randrange(num_qubits),)
        else:
            gate = rng.choice(tuple(TWO_Q_INVERSES))
            q0, q1 = rng.sample(range(num_qubits), 2)
            qubits = (q0, q1)
        sequence.append((gate, qubits))
    return sequence


def inverse_standard_clifford_sequence(sequence):
    inverse = []
    for gate, qubits in reversed(sequence):
        if len(qubits) == 1:
            inverse.append((ONE_Q_INVERSES[gate], qubits))
        else:
            inverse.append((TWO_Q_INVERSES[gate], qubits))
    return inverse


def apply_tick_gate(tc, gate, qubits):
    tick = tc.tick()
    if len(qubits) == 1:
        getattr(tick, TICK_1Q_METHODS[gate])([qubits[0]])
    else:
        getattr(tick, TICK_2Q_METHODS[gate])([(qubits[0], qubits[1])])


def build_mirrored_random_clifford_circuit(seed, num_qubits=3):
    sequence = random_standard_clifford_sequence(seed, num_qubits=num_qubits)
    tc = TickCircuit()
    tc.tick().pz(list(range(num_qubits)))
    for gate, qubits in sequence + inverse_standard_clifford_sequence(sequence):
        apply_tick_gate(tc, gate, qubits)
    tc.tick().mz(list(range(num_qubits)))
    tc.set_meta("num_measurements", str(num_qubits))
    tc.set_meta("detectors", "[]")
    tc.set_meta("observables", "[]")
    return tc, sequence


def run_direct_wrapper_mirrored_circuit(sim, sequence, num_qubits=3):
    for q in range(num_qubits):
        sim.run_1q_gate("PZ", q)

    for gate, qubits in sequence + inverse_standard_clifford_sequence(sequence):
        if len(qubits) == 1:
            sim.run_1q_gate(gate, qubits[0])
        else:
            sim.run_2q_gate(gate, qubits)

    return [sim.run_1q_gate("MZ", q) for q in range(num_qubits)]


def test_meas_sampling_runs_on_lowered_traced_qis_surface_code():
    tc = build_lowered_traced_qis_surface_code()
    shots = 8

    result = sim_neo(tc).quantum(meas_sampling()).noise(traced_qis_noise()).shots(shots).seed(123).run()

    assert result.num_shots == shots
    assert result.num_measurements == int(tc.get_meta("num_measurements"))
    assert len(result[0]) == result.num_measurements


def test_fault_catalog_builds_on_lowered_traced_qis_surface_code():
    tc = build_lowered_traced_qis_surface_code()

    catalog = fault_catalog(tc, traced_qis_noise())
    first = next(catalog.fault_configurations(1))

    assert len(catalog) > 0
    assert len(first.locations) == 1
    assert len(first.faults) == 1
    assert first.locations[0] is catalog.locations[first.location_indices[0]]


def test_lowered_traced_qis_pipeline_sampling_and_catalog_smoke():
    tc = build_lowered_traced_qis_surface_code(rounds=2)
    noise = traced_qis_noise()

    result = sim_neo(tc).quantum(meas_sampling()).noise(noise).shots(3).seed(321).run()
    catalog = fault_catalog(tc, noise)
    first_fault = next(catalog.fault_configurations(1))

    assert result.num_shots == 3
    assert result.num_measurements == int(tc.get_meta("num_measurements"))
    assert len(catalog) > 0
    assert len(first_fault.locations) == 1
    assert len(first_fault.faults) == 1


def test_explicit_python_gate_names_map_to_rust_clifford_gates():
    tc = build_explicit_clifford_gate_circuit()
    noise = depolarizing().p1(0.03).p2(0.15).p_meas(0).p_prep(0)

    result = sim_neo(tc).quantum(meas_sampling()).noise(noise).shots(3).seed(123).run()
    assert result.num_shots == 3
    assert result.num_measurements == 2

    catalog = fault_catalog(tc, noise)
    # Structural catalog includes all locations (including p_meas=0 and p_prep=0).
    # Count only alternatives at locations with nonzero channel probability.
    nonzero_alts = sum(len(loc.faults) for loc in catalog if loc.channel_probability > 0.0)
    assert nonzero_alts == 156


def test_sim_neo_native_backends_accept_face_gates():
    tc = TickCircuit()
    tc.tick().pz([0])
    tc.tick().f([0])
    tc.tick().fdg([0])
    tc.tick().mz([0])
    tc.set_meta("num_measurements", "1")
    tc.set_meta("detectors", "[]")
    tc.set_meta("observables", "[]")

    for backend in (stabilizer(), statevec()):
        result = sim_neo(tc).quantum(backend).noise(zero_noise()).shots(2).seed(123).run()
        assert result.num_measurements == 1
        assert all(result[shot][0] == 0 for shot in range(result.num_shots))


def test_direct_exp_wrappers_accept_standard_clifford_names():
    for sim in (StabMps(2, seed=123), Mast(2, 1, seed=123)):
        for gate in ("I", "X", "Y", "Z", "H", "F", "Fdg", "SX", "SXdg", "SY", "SYdg", "SZ", "SZdg"):
            sim.run_1q_gate(gate, 0)

        for gate in (
            "CX",
            "CY",
            "CZ",
            "SXX",
            "SXXdg",
            "SYY",
            "SYYdg",
            "SZZ",
            "SZZdg",
            "SWAP",
        ):
            sim.run_2q_gate(gate, (0, 1))


def test_direct_exp_wrappers_face_inverse_is_deterministic():
    for sim in (StabMps(1, seed=123), Mast(1, 1, seed=123)):
        sim.run_1q_gate("PZ", 0)
        sim.run_1q_gate("F", 0)
        sim.run_1q_gate("Fdg", 0)
        assert sim.run_1q_gate("MZ", 0) == 0


def test_random_mirrored_standard_clifford_circuits_match_across_backends():
    expected = [0, 0, 0]

    for seed in (7, 19, 41):
        tc, sequence = build_mirrored_random_clifford_circuit(seed)

        backend_results = {}
        for name, backend in (("stabilizer", stabilizer()), ("statevec", statevec())):
            result = sim_neo(tc).quantum(backend).noise(zero_noise()).shots(4).seed(seed).run()
            backend_results[name] = [list(row) for row in result.to_list()]

        backend_results["StabMps"] = [run_direct_wrapper_mirrored_circuit(StabMps(3, seed=seed), sequence)]
        backend_results["Mast"] = [run_direct_wrapper_mirrored_circuit(Mast(3, 1, seed=seed), sequence)]

        for name, rows in backend_results.items():
            assert all(row == expected for row in rows), (seed, name, rows)
