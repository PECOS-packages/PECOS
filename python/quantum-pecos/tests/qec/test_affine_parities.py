"""Literal XOR constants across authoring, raw evaluation, and DEM supports."""

import json

import pytest
import stim
from pecos import sim, stabilizer
from pecos.qec import Detector, Observable, build_dem_from_guppy, rec
from pecos.qec.generic import CircuitOp as O
from pecos.qec.generic import MeasurementParity as P
from pecos.qec.generic import NamedParity, ScheduledCircuit, evaluate_tick_circuit
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim


@pytest.mark.parametrize("cls", [Detector, Observable])
def test_literal_bits_and_xor_syntax(cls):
    original = cls("a", rec[-1], id=3, metadata={"label": "named"})
    assert cls("a", 1, rec[-1], id=3, metadata={"label": "named"}) == original ^ 1
    assert 1 ^ original == original ^ True
    assert cls(1, "a", 0, rec[-1], 1, id=3, metadata={"label": "named"}) == original
    assert cls("a", rec[-1], 1, constant=True, id=3, metadata={"label": "named"}) == original
    assert original ^ 1 ^ 1 == original
    assert not original.constant
    assert cls("a", constant=True) == cls("a", 1)
    with pytest.raises(ValueError, match="at least one measurement"):
        cls(1)
    for bad in (2, -1):
        with pytest.raises(ValueError, match="0 or 1"):
            cls("a", bad)
    with pytest.raises(TypeError, match="bool"):
        cls("a", constant=1)


def test_named_and_indexed_parity_xor_preserves_support():
    named = NamedParity(("a", "b"))
    assert str(named ^ 1) == "a ^ b ^ 1"
    assert 1 ^ named ^ 1 == named
    assert named ^ named ^ 1 == NamedParity(constant=True)
    indexed = P((2, 5), "check")
    assert (1 ^ indexed).records == (2, 5)
    assert indexed ^ 1 ^ 1 == indexed


def _plan(*, constant=True):
    return ScheduledCircuit(
        1,
        ((O("R", (0,)),), (O("X", (0,)),), (O("M", (0,)),)),
        (P((0,), "check", constant),),
        (P((0,), "logical", constant),),
    )


@pytest.mark.parametrize("constant", [False, True])
def test_signed_export_raw_records_and_support_only_native_definitions(constant):
    plan = _plan(constant=constant)
    expected = ((not constant,), (not constant,))
    assert plan.evaluate([1]) == expected
    circuit = plan.to_stim()
    assert circuit.num_measurements == 1 + constant
    assert circuit.num_ticks == 3
    for raw in circuit.compile_sampler(seed=2).sample(8):
        assert plan.evaluate_stim_record(raw) == expected
    tick = plan.to_tick_circuit()
    assert tick.num_measurements() == 1
    assert tick.num_ticks() == 3
    assert tick.circuit_definitions()["detectors"][0]["measurements"] == [0]
    assert evaluate_tick_circuit(tick, [1]) == expected
    converted = stim.Circuit(tick_circuit_to_stim(tick))
    assert converted.num_measurements == 1 + constant
    for raw in converted.compile_sampler(seed=2).sample(8):
        assert plan.evaluate_stim_record(raw) == expected
    # Stim reports flips relative to its reference, even if an authored expression is one.
    assert not circuit.compile_detector_sampler().sample(8, append_observables=True).any()


def test_synthetic_padding_is_validated_and_not_a_physical_measurement():
    plan = _plan()
    with pytest.raises(ValueError, match="MPAD"):
        plan.evaluate_stim_record([1])
    with pytest.raises(ValueError, match="MPAD"):
        plan.evaluate_stim_record([1, 0])
    with pytest.raises(ValueError, match="record length"):
        plan.evaluate([1, 1])
    assert plan.num_measurements == 1
    assert plan.to_tick_circuit().measurement_emission() == [(0, 0)]


def test_nontrivial_padding_offsets_and_constant_only_stim_expression():
    plan = ScheduledCircuit(
        2,
        (
            (O("R", (0,)), O("R", (1,))),
            (O("X", (0,)),),
            (O("M", (1,)), O("M", (0,))),
        ),
        (P((0, 1), "different", True), P((), "always one", True)),
        (P((1,), "one", True),),
    )
    circuit = plan.to_stim()
    record = circuit.reference_sample()
    assert list(record) == [False, True, True]
    assert plan.evaluate_stim_record(record) == ((False, True), (False,))
    assert str(circuit).count("MPAD") == 1
    assert "DETECTOR rec[-3] rec[-2] rec[-1]" in str(circuit)
    with pytest.raises(ValueError, match="nonempty"):
        plan.to_tick_circuit()


def test_signed_guppy_build_evaluation_fingerprint_and_identical_fault_dem():
    plan = _plan()
    program = plan.guppy_program()
    noise = {"p1": 0.001, "p2": 0.002, "p_meas": 0.003, "p_prep": 0.004}
    signed = build_dem_from_guppy(
        program,
        num_qubits=1,
        detectors=[Detector("qec:m:0", 1)],
        observables=[Observable("qec:m:0", 1, id=5)],
        **noise,
    )
    raw = build_dem_from_guppy(
        program,
        num_qubits=1,
        detectors=[Detector("qec:m:0")],
        observables=[Observable("qec:m:0", id=5, metadata={"reference": 1})],
        **noise,
    )
    assert signed.schema_fingerprint != raw.schema_fingerprint
    assert signed.dem.to_string() == raw.dem.to_string()
    assert signed.evaluate_runtime_record([1]) == ([0], 0)
    assert raw.evaluate_runtime_record([1]) == ([1], 1 << 5)  # reference is not an XOR instruction
    assert signed.evaluate_results({"qec:m:0": True}) == ([0], 0)
    assert signed.evaluate_results({"qec:m:0": False}) == ([1], 1 << 5)
    columns = sim(program).qubits(1).quantum(stabilizer()).seed(5).run(8).to_shot_map().to_dict()
    assert signed.evaluate_result_columns(columns) == [([0], 0)] * 8
    assert json.loads(signed.detectors_json)[0]["parity_constant"] == 1
    assert "reference" not in json.loads(signed.observables_json)[0]
    detectors, observables = plan.dem_spec()
    assert detectors[0].constant
    assert observables[0].constant


def test_malformed_native_constants_are_not_silently_dropped():
    tick = _plan().to_tick_circuit()
    entries = json.loads(tick.get_meta("detectors"))
    entries[0]["parity_constant"] = 2
    tick.set_meta("detectors", json.dumps(entries))
    with pytest.raises(ValueError, match="parity_constant"):
        evaluate_tick_circuit(tick, [1])
    with pytest.raises(ValueError, match="parity_constant"):
        tick_circuit_to_stim(tick)


def test_whitespace_native_metadata_remains_an_empty_schema():
    from pecos.quantum import TickCircuit

    tick = TickCircuit()
    tick.tick().pz([0])
    tick.tick().mz([0])
    tick.set_meta("detectors", " \n ")
    tick.set_meta("observables", "\t")
    assert evaluate_tick_circuit(tick, [0]) == ((), ())
    assert "MPAD" not in tick_circuit_to_stim(tick)


@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("prefixed", [False, True])
@pytest.mark.parametrize("constant", [0, 1])
def test_native_legacy_ids_and_prefixes_preserve_signs(aliased, prefixed, constant):
    from pecos.quantum import TickCircuit

    tick = TickCircuit()
    tick.tick().pz([0])
    tick.tick().mz([0])
    tick.set_meta(
        "detectors",
        json.dumps(
            [
                {
                    "detector_id" if aliased else "id": "D0" if prefixed else 0,
                    "records": [-1],
                    "parity_constant": constant,
                },
            ],
        ),
    )
    tick.set_meta(
        "observables",
        json.dumps(
            [
                {
                    "observable_id" if aliased else "id": "L2" if prefixed else 2,
                    "records": [-1],
                    "parity_constant": constant,
                },
            ],
        ),
    )
    assert evaluate_tick_circuit(tick, [0]) == ((bool(constant),), (bool(constant),))
    converted = stim.Circuit(tick_circuit_to_stim(tick))
    assert converted.num_measurements == 1 + constant
    detector = next(op for op in converted if op.name == "DETECTOR")
    assert len(detector.targets_copy()) == 1 + constant


@pytest.mark.parametrize("cls", [Detector, Observable])
def test_inferred_definition_xor_preserves_selector_and_defaults(cls):
    original = cls.inferred("derived", element=2, id=3, metadata={"label": "named"})
    toggled = original ^ 1
    assert not original.constant
    assert toggled.constant
    assert toggled.inferred_output == original.inferred_output
    assert toggled.refs == ()
    assert toggled.id == 3
    assert toggled.metadata == {"label": "named"}
    assert toggled ^ 1 == original
