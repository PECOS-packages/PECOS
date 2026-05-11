# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Inline TickCircuit channel tests for sim_neo Python bindings."""

import pytest
from pecos_rslib.quantum import TickCircuit
from pecos_rslib_exp import depolarizing, meas_sampling, sim_neo, stabilizer


def prep_measure_circuit() -> TickCircuit:
    tc = TickCircuit()
    tc.tick().pz([0])
    tc.tick().mz([0])
    return tc


def measurement_rows(result) -> list[list[int]]:
    return [list(row) for row in result.to_list()]


def test_tick_circuit_with_noise_inserts_channel_payload() -> None:
    noisy = prep_measure_circuit().with_noise(p_prep=1.0)
    channel_gates = [gate for _, gate in noisy.gate_batches() if gate.is_channel()]

    assert len(channel_gates) == 1
    assert channel_gates[0].channel_mixed_pauli_terms() == [
        (0.0, []),
        (1.0, [("X", 0)]),
    ]


def test_tick_circuit_with_noise_rejects_measurement_readout_noise() -> None:
    with pytest.raises(ValueError, match="measurement readout noise"):
        prep_measure_circuit().with_noise(p_meas=0.1)


def test_tick_circuit_with_noise_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="p1 must be in \\[0, 1\\]"):
        prep_measure_circuit().with_noise(p1=-0.1)

    with pytest.raises(ValueError, match="p2 must be in \\[0, 1\\]"):
        prep_measure_circuit().with_noise(p2=1.1)


def test_sim_neo_default_routes_inline_channels_through_density_matrix() -> None:
    noisy = prep_measure_circuit().with_noise(p_prep=1.0)

    result = sim_neo(noisy).shots(5).seed(123).run()

    assert measurement_rows(result) == [[1], [1], [1], [1], [1]]


def test_sim_neo_stabilizer_samples_inline_pauli_channels() -> None:
    noisy = prep_measure_circuit().with_noise(p_prep=1.0)

    result = sim_neo(noisy).quantum(stabilizer()).shots(5).seed(123).run()

    assert measurement_rows(result) == [[1], [1], [1], [1], [1]]


def test_sim_neo_rejects_noise_builder_with_inline_channels() -> None:
    noisy = prep_measure_circuit().with_noise(p_prep=1.0)
    noise = depolarizing().p1(0.1)

    with pytest.raises(ValueError, match="do not also pass"):
        sim_neo(noisy).noise(noise).run()


def test_sim_neo_meas_sampling_rejects_inline_channels() -> None:
    noisy = prep_measure_circuit().with_noise(p_prep=1.0)

    with pytest.raises(ValueError, match="does not consume inline channel"):
        sim_neo(noisy).quantum(meas_sampling()).run()
