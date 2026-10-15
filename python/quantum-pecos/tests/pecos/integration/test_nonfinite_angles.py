# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Non-finite angles must fail at Python boundaries without a Rust panic."""

import math

import pecos_rslib as rust
import pecos_rslib_exp as experimental
import pytest


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "convert",
    [
        rust.angle64.from_radians,
        rust.angle64.from_turns,
        rust.AngleSource.fixed,
        lambda angle: rust.Gate(rust.GateType.RZ, [angle], [0]),
        lambda angle: rust.GateRegistry().decompose("RZ", [0], [angle]),
        lambda angle: rust.TickCircuit().tick().rz(angle, [0]),
        lambda angle: rust.TickCircuit().tick().crz(angle, [(0, 1)]),
        lambda angle: rust.StateVec(1).run_gate("RZ", {0}, angle=angle),
        lambda angle: rust.StateVec(2).run_gate("CRZ", {(0, 1)}, angle=angle),
    ],
)
def test_binding_rejects_nonfinite_angle(convert, angle):
    with pytest.raises(ValueError, match=r"(?i)(nan|inf)"):
        convert(angle)


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
def test_experimental_coherent_noise_rejects_nonfinite_angle(angle):
    with pytest.raises(ValueError, match=r"(?i)(nan|inf)"):
        experimental.NoiseModelBuilder().idle_rz(angle)


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
def test_experimental_simulator_rejects_nonfinite_angle(angle):
    with pytest.raises(ValueError, match=r"(?i)(nan|inf)"):
        experimental.StabMps(1).run_gate("RZ", {0}, angle=angle)


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("consumer", ["sim_neo", "exact_detection_rates"])
def test_experimental_circuit_extraction_rejects_nonfinite_angle(angle, consumer):
    from types import SimpleNamespace

    gate = SimpleNamespace(gate_type=rust.GateType.RZ, qubits=[0], angles=[angle])
    tick = SimpleNamespace(gate_batches=lambda: [gate])
    circuit = SimpleNamespace(num_ticks=lambda: 1, get_tick=lambda _index: tick)
    with pytest.raises(ValueError, match=r"(?i)(nan|inf)"):
        getattr(experimental, consumer)(circuit)


def test_phir_parser_rejects_unit_overflow():
    source = (
        '{"format":"PHIR/JSON","version":"0.1.0","ops":['
        '{"data":"qvar_define","data_type":"qubits","variable":"q","size":1},'
        '{"qop":"RZ","angles":[[1e308],"pi"],"args":[["q",0]]}]}'
    )
    with pytest.raises(ValueError, match="inf"):
        rust.PhirJsonEngine(source).process_program()


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_composite_angles_are_validated_atomically(angle, position):
    from pecos import QuantumCircuit

    circuit = QuantumCircuit()
    angles = [0.0, 0.0, 0.0]
    angles[position] = angle
    with pytest.raises(ValueError, match=r"(?i)(nan|inf)"):
        circuit.append({"RXXRYYRZZ": {(0, 1)}}, angles=tuple(angles))
    assert len(circuit) == 0


@pytest.mark.parametrize(("angle", "expected"), [(math.nan, "NaN"), (math.inf, "inf"), (-math.inf, "-inf")])
def test_python_phir_guard_matches_rust_message(angle, expected):
    from pecos.reps.pyphir.pyphir import PyPHIR

    with pytest.raises(ValueError, match="angle must be finite") as error:
        PyPHIR.handle_op({"qop": "RZ", "angles": [[angle], "rad"], "args": [["q", 0]]}, PyPHIR())
    assert str(error.value) == f"angle must be finite, got {expected} radians"


@pytest.mark.parametrize(("angle", "expected"), [(math.nan, "NaN"), (math.inf, "inf"), (-math.inf, "-inf")])
def test_python_composite_guard_matches_rust_message(angle, expected):
    from pecos import QuantumCircuit

    with pytest.raises(ValueError, match="angle must be finite") as error:
        QuantumCircuit().append({"RXXRYYRZZ": {(0, 1)}}, angles=(0.0, angle, 0.0))
    assert str(error.value) == f"angle must be finite, got {expected} radians"
