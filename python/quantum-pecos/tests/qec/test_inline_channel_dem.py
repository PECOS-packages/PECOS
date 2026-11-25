"""Generic native DEMs consume actual inline payloads, independent of QEC plans."""

import json

import pytest
import stim
from pecos.qec.generic import SI1000, CircuitOp, MeasurementParity, ScheduledCircuit
from pecos.quantum import TickCircuit
from pecos_rslib.qec import DemSampler, DetectorErrorModel

from qec.test_native_scheduled_noise import _native_mechanisms, _pair_plan
from qec.test_scheduled_si1000 import CASES, FIXTURES, _case_plan, _mechanisms


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["fixture"])
def test_generic_inline_dem_matches_published_reference_and_scheduled_native(case):
    noisy = _case_plan(case)
    ticks = noisy.to_tick_circuit()
    model = DetectorErrorModel.from_circuit(ticks)
    expected = _mechanisms(stim.Circuit((FIXTURES / case["fixture"]).read_text()))
    assert _native_mechanisms(model) == pytest.approx(expected, rel=1e-10, abs=1e-14)
    independent = noisy.compile_native().detector_error_model()
    assert _native_mechanisms(model) == pytest.approx(_native_mechanisms(independent), rel=1e-12, abs=1e-15)
    assert model.detector_coordinates() == independent.detector_coordinates()
    assert model.num_detectors == independent.num_detectors
    assert model.num_observables == independent.num_observables


@pytest.mark.parametrize("p", [0, 0.001, 0.2, 1e-25])
def test_tick_dag_sampler_and_affine_parities(p):
    noisy = SI1000(p).apply(_pair_plan())
    ticks = noisy.to_tick_circuit()
    for circuit in (ticks, ticks.to_dag_circuit()):
        model = DetectorErrorModel.from_circuit(circuit)
        assert _native_mechanisms(model) == pytest.approx(
            _native_mechanisms(noisy.compile_native().detector_error_model()),
            rel=1e-12,
            abs=0.0,
        )
        assert model.detector_coordinates() == [(0, [1.0, 2.0, 3.0, 4.0])]
        sampler = DemSampler.from_circuit(circuit)
        assert sampler.num_detectors == model.num_detectors
        assert sampler.num_observables == model.num_observables
        assert sampler.sample_batch(8, seed=52).num_shots == 8
        if p == 0:
            assert not any(sampler.sample_batch(8, seed=52).observable_flips()[0])


def _binary(p=0.125):
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().x_error(p, 0)
    circuit.tick().mz_with_ids([0], [17])
    circuit.add_detector([-1], coords=[1, 2, 3, 4, 5])
    circuit.add_observable([-1], observable_id=0)
    return circuit


def test_payload_is_authority_and_metadata_not_needed_to_identify_inline_noise():
    circuit = _binary()
    assert circuit.get_meta("pecos_noise_source") is None
    circuit.set_meta("qec_noise_probability", 0.999)
    circuit.set_meta("qec_noise_channels", -5)
    assert DetectorErrorModel.from_circuit(circuit).independent_mechanisms() == [(0.125, [0], [0])]
    # Stable ids are not interpreted as positional indices.
    entries = json.loads(circuit.get_meta("detectors"))
    entries[0]["meas_ids"] = [17]
    entries[0].pop("records")
    circuit.set_meta("detectors", json.dumps(entries))
    assert DetectorErrorModel.from_circuit(circuit).independent_mechanisms() == [(0.125, [0], [0])]


@pytest.mark.parametrize("constructor", [DetectorErrorModel.from_circuit, DemSampler.from_circuit])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"p1": 0.001},
        {"p2": 0.01},
        {"p_meas": 0.1},
        {"p_prep": 0.1},
        {"p_idle": 0.1},
        {"idle_rz": 0.1},
        {"p_idle_linear_rate": 0.1},
        {"p_idle_quadratic_rate": 0.1},
        {"p_meas_crosstalk_local": 0.1},
        {"p1_gate_rates": {"H": 0.1}},
        {"p2_gate_rates": {"CZ": 0.1}},
    ],
)
def test_explicit_additional_noise_rejected(constructor, kwargs):
    with pytest.raises(ValueError, match="external"):
        constructor(_binary(), **kwargs)


def test_legacy_defaults_and_explicit_zero_noise_producers():
    ideal = TickCircuit()
    ideal.tick().pz([0])
    ideal.tick().mz([0])
    ideal.add_detector([-1])
    assert DetectorErrorModel.from_circuit(ideal).independent_mechanisms()
    assert DetectorErrorModel.from_circuit(ideal, p1=0, p2=0, p_meas=0, p_prep=0).independent_mechanisms() == []
    noisy = ideal.with_noise()
    assert noisy.get_meta("pecos_noise_source") == "inline"
    assert not any(g.is_channel() for _, g in noisy.gate_batches())
    for circuit in (noisy, noisy.to_dag_circuit()):
        assert DetectorErrorModel.from_circuit(circuit).independent_mechanisms() == []
        with pytest.raises(ValueError, match="external"):
            DetectorErrorModel.from_circuit(circuit, p_meas=0.1)
    assert ideal.get_meta("pecos_noise_source") is None


@pytest.mark.parametrize(("method", "p"), [("depolarize1", 0.751), ("depolarize2", 0.94)])
def test_unavailable_exact_conversion_fails_without_approximation(method, p):
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    getattr(circuit.tick(), method)(p, 0 if method == "depolarize1" else (0, 1))
    circuit.tick().mz([0, 1])
    circuit.add_detector([-1])
    with pytest.raises(ValueError, match="exact independent-channel range"):
        DetectorErrorModel.from_circuit(circuit)


@pytest.mark.parametrize(("method", "p"), [("depolarize1", 0.75), ("depolarize2", 0.9375)])
def test_fully_mixed_channels_are_exact(method, p):
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    getattr(circuit.tick(), method)(p, 0 if method == "depolarize1" else (0, 1))
    circuit.tick().mz([0, 1])
    circuit.add_detector([-2])
    assert DetectorErrorModel.from_circuit(circuit).independent_mechanisms() == [(0.5, [0], [])]


def test_nondeterministic_detector_and_observable_rejected_but_correlations_allowed():
    op, parity = CircuitOp, MeasurementParity
    plan = ScheduledCircuit(
        1,
        (
            (op("R", (0,)),),
            (op("H", (0,)),),
            (op("M", (0,)),),
        ),
        (parity((0,), "random"),),
        (),
    )
    with pytest.raises(ValueError, match="deterministic ideal detector"):
        DetectorErrorModel.from_circuit(SI1000(0.001).apply(plan).to_tick_circuit())
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().h([0])
    circuit.tick().cx([(0, 1)])
    circuit.tick().depolarize2(0.01, (0, 1))
    circuit.tick().mz([1, 0])
    circuit.add_detector([-1, -2])
    assert DetectorErrorModel.from_circuit(circuit).independent_mechanisms()
    circuit.add_observable([-1], observable_id=0)
    with pytest.raises(ValueError, match="deterministic ideal observable"):
        DemSampler.from_circuit(circuit)


def test_invalid_record_metadata_is_not_bypassed_by_inline_route():
    circuit = _binary()
    circuit.set_meta("detectors", '[{"id":0,"records":[-2]}]')
    with pytest.raises(ValueError, match="out of range"):
        DetectorErrorModel.from_circuit(circuit)
