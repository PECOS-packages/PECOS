"""User-facing QEC entry point and metadata ergonomics tests."""

from __future__ import annotations

import json

import pytest


def test_main_pecos_namespace_exports_sim_neo_stack() -> None:
    from pecos import depolarizing, fault_catalog, meas_sampling, sim_neo, stabilizer
    from pecos.quantum import TickCircuit

    tc = TickCircuit()
    tc.tick().mz([0])

    result = sim_neo(tc).quantum(stabilizer()).noise(depolarizing()).shots(2).seed(123).run()
    assert result.num_shots == 2
    assert meas_sampling() is not None
    assert callable(fault_catalog)


def test_build_memory_circuit_is_public_surface_helper() -> None:
    from pecos.qec.surface import build_memory_circuit

    tc = build_memory_circuit(distance=3, rounds=2, basis="Z")

    assert int(tc.get_meta("num_measurements")) > 0
    assert json.loads(tc.get_meta("detectors"))
    assert json.loads(tc.get_meta("observables"))


def test_surface_code_memory_runs_native_zero_noise_quick_start() -> None:
    from pecos.qec.surface import surface_code_memory

    result = surface_code_memory(
        distance=3,
        physical_error_rate=0.0,
        shots=4,
        rounds=1,
        seed=123,
    )

    assert result.distance == 3
    assert result.num_shots == 4
    assert result.num_rounds == 1
    assert result.logical_error_rate == 0.0
    assert result.raw_error_rate == 0.0


def test_surface_code_memory_rejects_ambiguous_noise_inputs() -> None:
    from pecos.qec.surface import NoiseModel, surface_code_memory

    with pytest.raises(ValueError, match="either physical_error_rate or noise_model"):
        surface_code_memory(
            physical_error_rate=0.0,
            noise_model=NoiseModel.uniform(0.001),
            shots=0,
            rounds=1,
        )


def test_tick_circuit_metadata_helpers_build_detector_and_observable_json() -> None:
    from pecos.quantum import TickCircuit

    tc = TickCircuit()
    det_id = tc.add_detector(records=[-1], coords=[0.0, 1.0, 2.0], label="d0")
    obs_id = tc.add_observable(records=[-1, -2], label="L2")

    detectors = json.loads(tc.get_meta("detectors"))
    observables = json.loads(tc.get_meta("observables"))

    assert det_id == 0
    assert detectors == [{"id": 0, "records": [-1], "coords": [0.0, 1.0, 2.0], "label": "d0"}]
    assert int(tc.get_meta("num_detectors")) == 1

    assert obs_id == 2
    assert observables == [{"id": 2, "records": [-1, -2], "label": "L2"}]
    assert int(tc.get_meta("num_observables")) == 3


def test_tick_circuit_observable_helper_rejects_conflicting_label_id() -> None:
    from pecos.quantum import TickCircuit

    tc = TickCircuit()
    with pytest.raises(ValueError, match="conflicts"):
        tc.add_observable(records=[-1], observable_id=1, label="L2")


def test_tick_circuit_reset_clears_annotations_and_measurement_records() -> None:
    from pecos.quantum import TickCircuit, Z

    tc = TickCircuit()
    measurements = tc.tick().mz([0])
    tc.detector(measurements)
    tc.observable(measurements)
    tc.tracked_operator(Z(0))

    assert tc.num_measurements() == 1
    assert len(tc.annotations()) == 3

    tc.reset()

    assert tc.num_ticks() == 0
    assert tc.num_measurements() == 0
    assert tc.annotations() == []
