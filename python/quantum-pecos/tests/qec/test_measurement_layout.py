# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Shared abstract measurement reading and program-bound certificates."""

import hashlib
import json
from functools import partial
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import measure, qubit
from pecos._compilation import guppy_to_hugr
from pecos.guppy_gen._certificate import certify_measurement_layout
from pecos.qec.measurement_layout import abstract_result_refs
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch
from pecos.qec.surface.decode import _surface_abstract_measurement_result_refs
from pecos_rslib.quantum import DagCircuit, Gate, TickCircuit

SURFACE_REFS = json.loads((Path(__file__).parent / "fixtures" / "surface_abstract_result_refs.json").read_text())


@pytest.mark.parametrize("key", SURFACE_REFS)
def test_surface_result_refs_match_fixture(key):
    distance, basis, rounds, *budget = key.split("-")
    circuit = generate_tick_circuit_from_patch(
        SurfacePatch.create(distance=int(distance.removeprefix("d"))),
        int(rounds.removeprefix("r")),
        basis,
        ancilla_budget=int(budget[0].removeprefix("budget")) if budget else None,
        add_typed_annotations=False,
    )
    expected = [tuple(ref) for ref in SURFACE_REFS[key]]
    for _ in range(2):
        refs = _surface_abstract_measurement_result_refs(circuit)
        assert refs == expected


def test_generic_reader_walk_and_metadata():
    circuit = TickCircuit()
    tick = circuit.tick()
    tick.h([8])
    tick.add_gate("MZ", [7]).metas({"label": "omega", "ordinal": 31})
    tick.add_gate("MX", [2]).metas({"label": "alpha", "ordinal": 19})
    circuit.tick()
    tick = circuit.tick()
    tick.add_gate("MZ", [5]).metas({"ordinal": 7})
    tick.add_gate("MPZ", [3]).metas({"label": "middle", "ordinal": 23})
    tick.add_gate("MeasureFree", [1]).metas({"label": "last", "ordinal": 42})
    seen = []

    def classify(label, meta):
        seen.append((label, meta("ordinal"), meta("absent")))
        return ("scalar", label) if label else ("array", "readout", meta("ordinal"))

    assert abstract_result_refs(circuit, lambda: classify, family="custom") == [
        ("scalar", "omega"),
        ("scalar", "alpha"),
        ("array", "readout", 7),
        ("scalar", "middle"),
        ("scalar", "last"),
    ]
    assert seen == [("omega", 31, None), ("alpha", 19, None), ("", 7, None), ("middle", 23, None), ("last", 42, None)]


@pytest.mark.parametrize("gate_type", ["MZ", "MX", "MPZ", "MeasureFree"])
@pytest.mark.parametrize("label", ["custom", None])
def test_generic_reader_rejects_multi_qubit_before_classifying(gate_type, label):
    template = TickCircuit()
    template.tick().add_gate(gate_type, [4])
    dag = DagCircuit()
    dag.add_gate(Gate(template.get_tick(0).gate_batches()[0].gate_type, qubits=[4, 2]))
    circuit = dag.to_tick_circuit()
    if label is not None:
        circuit.set_gate_meta(0, 0, "label", label)

    def classify(_label, _meta):
        pytest.fail("multi-qubit measurement reached classifier")

    with pytest.raises(ValueError, match="names one measurement but its gate measures 2 qubits") as exc:
        abstract_result_refs(circuit, lambda: classify, family="custom")
    assert str(exc.value) == (
        f"custom measurement label {label or ''!r} names one measurement but its gate measures 2 qubits"
    )


def test_generic_reader_propagates_classifier_error():
    circuit = TickCircuit()
    circuit.tick().add_gate("MZ", [0]).metas({"label": "unknown"})
    error = ValueError("custom classifier rejected unknown")

    def classify(_label, _meta):
        raise error

    with pytest.raises(ValueError, match="custom classifier rejected unknown") as exc:
        abstract_result_refs(circuit, lambda: classify, family="custom")
    assert exc.value is error


@guppy
def _tagged_program() -> None:
    result("alpha", measure(qubit()).read())
    result("beta", measure(qubit()).read())
    result("alpha", measure(qubit()).read())
    second = measure(qubit()).read()
    first = measure(qubit()).read()
    result("readout", array(first, second))


@guppy
def _uncertified_program() -> None:
    result("alpha", measure(qubit()).read())
    result("beta", measure(qubit()).read())


def test_generic_certificate_layout_and_digest():
    circuit = TickCircuit()
    for index, label in enumerate(["alpha", "beta", "alpha", "readout[1]", "readout[0]"]):
        circuit.tick().add_gate("MZ", [index]).metas({"label": label})

    def classify(label, _meta):
        if label.startswith("readout["):
            return ("array", "readout", int(label.removeprefix("readout[").removesuffix("]")))
        return ("scalar", label)

    certify_measurement_layout(_tagged_program, circuit, lambda: classify, family="custom")
    digest, layout = _tagged_program.__pecos_named_measurement_layout_v2__
    assert layout == (("alpha", 0), ("beta", 0), ("alpha", 1), ("readout:meas:1", 0), ("readout:meas:0", 0))
    assert (
        digest
        == hashlib.sha256(
            guppy_to_hugr(_tagged_program) + b"\0" + json.dumps(layout, separators=(",", ":")).encode(),
        ).hexdigest()
    )


@pytest.mark.parametrize("emission", [[(1, None), (0, None)], [(0, None)]])
@pytest.mark.parametrize("certify", [False, True])
def test_reader_and_certificate_reject_emission_disagreement(monkeypatch, emission, certify):
    circuit = TickCircuit()
    circuit.tick().add_gate("MZ", [0]).metas({"label": "alpha"})
    circuit.tick().add_gate("MZ", [1]).metas({"label": "beta"})
    monkeypatch.setattr(TickCircuit, "measurement_emission", lambda _self: emission)

    def classify(label, _meta):
        return ("scalar", label)

    read = partial(certify_measurement_layout, _uncertified_program) if certify else abstract_result_refs
    with pytest.raises(ValueError, match="disagree with the shared reader's measurement emission") as exc:
        read(circuit, lambda: classify, family="custom")
    assert str(exc.value) == "custom measurement result refs disagree with the shared reader's measurement emission"
    if certify:
        # A rejected layout must not leave a certificate behind.
        assert not hasattr(_uncertified_program, "__pecos_named_measurement_layout_v2__")
