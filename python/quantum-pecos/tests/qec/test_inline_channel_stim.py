"""Stim compatibility preserves categorical payloads without constraining PECOS."""

from itertools import product
from types import SimpleNamespace

import pytest
import stim
from pecos.qec._stim_channels import channel_to_stim
from pecos.qec.generic import SI1000, evaluate_tick_circuit
from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_stim, tick_circuit_to_stim
from pecos.quantum import TickCircuit

from qec.test_native_scheduled_noise import _pair_plan
from qec.test_scheduled_si1000 import CASES, FIXTURES, _case_plan, _mechanisms


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["fixture"])
def test_published_inline_export_matches_reference(case):
    noisy = _case_plan(case)
    ticks = noisy.to_tick_circuit()
    circuit = stim.Circuit(tick_circuit_to_stim(ticks))
    reference = stim.Circuit((FIXTURES / case["fixture"]).read_text())
    # PECOS retains authored affine parity signs. Stim export encodes their
    # constant with an extra MPAD bit absent from the published files.
    padding = int(any(p.constant for p in (*noisy.ideal.detectors, *noisy.ideal.observables)))
    assert circuit.num_measurements == reference.num_measurements + padding
    assert circuit.num_detectors == reference.num_detectors
    assert circuit.num_observables == reference.num_observables
    assert circuit.get_detector_coordinates() == {
        d["id"]: d["coords"] or [] for d in ticks.circuit_definitions()["detectors"]
    }
    assert _mechanisms(circuit) == pytest.approx(_mechanisms(reference), rel=1e-10, abs=1e-14)
    # Export retains execution boundaries, not the original hardware-time count.
    assert circuit.num_ticks == ticks.num_ticks() - 1
    sample = circuit.compile_sampler(seed=23).sample(3)
    converter = circuit.compile_m2d_converter(skip_reference_sample=True)
    dets, obs = converter.convert(measurements=sample, separate_observables=True)
    for record, ds, os in zip(sample, dets, obs, strict=True):
        assert evaluate_tick_circuit(ticks, record[:reference.num_measurements]) == (tuple(ds), tuple(os))
        if padding:
            assert record[-1]


@pytest.mark.parametrize("p", [0.0, 1e-25, 0.001, 0.2])
def test_affine_records_coordinates_and_explicit_zero_noise(p):
    ticks = SI1000(p).apply(_pair_plan()).to_tick_circuit()
    circuit = stim.Circuit(tick_circuit_to_stim(ticks))
    assert circuit.num_measurements == len(ticks.measurement_emission()) + 1
    assert circuit.get_detector_coordinates() == {0: [1.0, 2.0, 3.0, 4.0]}
    samples = circuit.compile_sampler(seed=14).sample(8)
    ds, os = circuit.compile_m2d_converter(skip_reference_sample=True).convert(
        measurements=samples,
        separate_observables=True,
    )
    for record, d, o in zip(samples, ds, os, strict=True):
        assert record[-1]
        assert evaluate_tick_circuit(ticks, record[:-1]) == (tuple(d), tuple(o))
    # No silent external noise is introduced by the DEM convenience wrapper.
    dem = stim.DetectorErrorModel(generate_dem_from_tick_circuit_via_stim(ticks, decompose_errors=False))
    assert dem.num_detectors == 1
    if p == 0:
        assert dem.num_errors == 0


@pytest.mark.parametrize("name", ["x_error", "depolarize1", "depolarize2"])
@pytest.mark.parametrize("p", [0.0, 1e-25, 0.5, 0.75, 0.9375, 1.0])
def test_actual_payload_endpoints_and_metadata_not_authoritative(name, p):
    ticks = TickCircuit()
    ticks.tick().pz([2, 5])
    getattr(ticks.tick(), name)(p, (5, 2) if name == "depolarize2" else 5)
    ticks.tick().mz([5, 2])
    ticks.set_meta("qec_noise_probability", 0.123)
    actual = stim.Circuit(tick_circuit_to_stim(ticks))
    noise = [inst for inst in actual if inst.name in {"X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"}]
    if p == 0:
        assert noise == []
    else:
        assert len(noise) == 1
        assert noise[0].gate_args_copy() == pytest.approx([p], rel=1e-15, abs=0)
        assert {t.value for t in noise[0].targets_copy()} == ({2, 5} if name == "depolarize2" else {5})


def _payload(qubits, branches):
    # Exercise the payload protocol with distributions not constructible by the
    # current Python TickHandle convenience methods.
    return SimpleNamespace(qubits=qubits, channel_mixed_pauli_terms=lambda: branches)


@pytest.mark.parametrize("word", ["".join(p) for p in product("IXYZ", repeat=2) if p != ("I", "I")])
def test_asymmetric_binary_physical_target_order(word):
    gate = _payload([7, 2], [(0.625, []), (0.375, [(p, q) for p, q in zip(word, [7, 2], strict=True) if p != "I"])])
    instruction = stim.Circuit(channel_to_stim(gate))[0]
    if "I" in word:
        position = 0 if word[0] != "I" else 1
        assert instruction.name == word[position] + "_ERROR"
        assert instruction.targets_copy()[0].value == [7, 2][position]
    else:
        assert instruction.name == "E"
        assert [str(t) for t in instruction.targets_copy()] == [
            f"stim.target_{word[0].lower()}(7)",
            f"stim.target_{word[1].lower()}(2)",
        ]
    assert instruction.gate_args_copy() == [0.375]


def test_nonuniform_channels_keep_exclusive_branches_and_stim_argument_order():
    # XI, IY, YZ and ZZ are deliberately asymmetric and nonuniform.
    gate = _payload(
        [7, 2],
        [(0.4, []), (0.1, [("X", 7)]), (0.2, [("Y", 2)]), (0.25, [("Y", 7), ("Z", 2)]), (0.05, [("Z", 7), ("Z", 2)])],
    )
    instruction = stim.Circuit(channel_to_stim(gate))[0]
    assert instruction.name == "PAULI_CHANNEL_2"
    assert instruction.gate_args_copy() == [0, 0.2, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.05]
    assert [t.value for t in instruction.targets_copy()] == [7, 2]
    # In Z readout these categorical branches have four distinct outcomes.
    circuit = stim.Circuit("R 7 2\n" + channel_to_stim(gate) + "\nM 7 2")
    rows = circuit.compile_sampler(seed=91).sample(40000)
    observed = {bits: sum(tuple(row) == bits for row in rows) / len(rows) for bits in product((False, True), repeat=2)}
    assert observed == pytest.approx(
        {(False, False): 0.45, (False, True): 0.2, (True, False): 0.35, (True, True): 0}, abs=0.012,
    )
    one = _payload([3], [(0.4, []), (0.1, [("X", 3)]), (0.2, [("Y", 3)]), (0.3, [("Z", 3)])])
    assert stim.Circuit(channel_to_stim(one))[0].gate_args_copy() == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "kwargs", [{"p1": 0.1}, {"p2": 0.1}, {"p_meas": 0.1}, {"p_prep": 0.1}, {"p1_gate_rates": {"H": 0.1}}],
)
@pytest.mark.parametrize("p", [0.0, 0.01])
def test_additive_noise_rejected_even_at_explicit_zero(p, kwargs):
    ticks = SI1000(p).apply(_pair_plan()).to_tick_circuit()
    for export in (tick_circuit_to_stim, generate_dem_from_tick_circuit_via_stim):
        with pytest.raises(ValueError, match="external"):
            export(ticks, **kwargs)


def test_wrapper_legacy_defaults_and_with_noise_zero():
    ticks = TickCircuit()
    ticks.tick().pz([0])
    ticks.tick().mz([0])
    ticks.add_detector([-1])
    assert stim.DetectorErrorModel(generate_dem_from_tick_circuit_via_stim(ticks)).num_errors == 1
    assert stim.DetectorErrorModel(generate_dem_from_tick_circuit_via_stim(ticks.with_noise())).num_errors == 0


@pytest.mark.parametrize(
    "gate",
    [
        _payload([0, 1, 2], [(1.0, [])]),
        _payload([0], [(0.8, []), (0.4, [("X", 0)])]),
        _payload([0], [(1.0, [("X", 1)])]),
        _payload([0], [(float("nan"), [])]),
    ],
)
def test_unrepresentable_or_invalid_payload_fails(gate):
    with pytest.raises(ValueError, match=r"channel|Pauli"):
        channel_to_stim(gate)
