# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Python integration coverage for the shared Rust circuit renderer."""

import math
import xml.etree.ElementTree as ET

import pytest
from pecos_rslib.quantum import DagCircuit, TickCircuit


@pytest.fixture(params=["tick", "dag"])
def circuit(request):
    tick = TickCircuit()
    tick.tick().h([2])
    tick.tick().cx([(2, 7)])
    tick.tick().rz(math.pi / 2, [7])
    tick.tick().mz_with_ids([2, 7], [40, 41])
    return tick if request.param == "tick" else tick.to_dag_circuit()


def test_python_svg_preserves_sparse_qubit_labels_and_gates(circuit):
    root = ET.fromstring(circuit.to_svg())  # noqa: S314 - locally generated SVG
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "viewBox" in root.attrib
    text = " ".join(root.itertext())
    assert "q2" in text
    assert "q7" in text
    assert "q0" not in text
    assert "RZ(π/2)" in text
    assert "MZ" in text
    assert circuit._repr_svg_() == circuit.to_svg()


@pytest.mark.parametrize("format", ["ascii", "unicode", "svg", "tikz", "dot"])
def test_render_formats_and_angle_units(circuit, format):
    output = circuit.render(format, angle_unit="turns", color=False)
    assert "RZ(1/4)" in output
    assert "\x1b[" not in output
    assert getattr(circuit, f"to_{format}")() == circuit.render(format)
    if format == "svg":
        ET.fromstring(output)  # noqa: S314 - locally generated SVG
        assert "#FFB0B0" not in output
        assert "#A8C8F0" not in output
    elif format == "tikz":
        assert r"\begin{tikzpicture}" in output
    elif format == "dot":
        assert "digraph circuit" in output


def test_ticks_and_dependency_layers_are_distinct():
    tick = TickCircuit()
    tick.tick().h([0])
    tick.tick()
    tick.tick().x([1])
    assert "3 ticks" in tick.to_ascii()
    assert "1 layer" in tick.to_dag_circuit().to_ascii()
    # Displaying the DAG must not compact or otherwise modify the source ticks.
    assert tick.num_ticks() == 3


@pytest.mark.parametrize("circuit_type", [TickCircuit, DagCircuit])
def test_empty_circuit_svg_is_well_formed(circuit_type):
    root = ET.fromstring(circuit_type().to_svg())  # noqa: S314 - locally generated SVG
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert float(root.attrib["width"]) > 0
    assert float(root.attrib["height"]) > 0


def test_render_rejects_unknown_options(circuit):
    with pytest.raises(ValueError, match="format must be"):
        circuit.render("png")
    with pytest.raises(ValueError, match="angle_unit must be"):
        circuit.render(angle_unit="degrees")


@pytest.mark.parametrize("as_dag", [False, True])
def test_custom_labels_and_operand_indices_survive_graphical_exports(as_dag):
    circuit = TickCircuit()
    circuit.tick().h([0, 1, 2])
    name = 'A<&>_"%'
    circuit.tick().custom_gate(name, [2, 0])
    if as_dag:
        circuit = circuit.to_dag_circuit()
    svg = circuit.to_svg()
    root = ET.fromstring(svg)  # noqa: S314 - locally generated SVG
    labels = " ".join(root.itertext())
    assert f"{name}[0]" in labels
    assert f"{name}[1]" in labels
    # Three horizontal wires and one connector across the custom operation.
    assert len(root.findall("{http://www.w3.org/2000/svg}line")) == 4
    assert r"A<&>_\"%" in circuit.to_dot()
    assert r"A<\&>\_\"\%".replace(r"\"", '"') in circuit.to_tikz()


def test_correlated_noise_keeps_joint_support_in_svg():
    circuit = TickCircuit()
    circuit.tick().cx([(2, 0)])
    noisy = circuit.with_noise(p2=0.1)
    root = ET.fromstring(noisy.to_svg())  # noqa: S314 - locally generated SVG
    labels = " ".join(root.itertext())
    assert "Ch[0]" in labels
    assert "Ch[1]" in labels
    # Two wires, a CX connector, and a separate channel connector.
    assert len(root.findall("{http://www.w3.org/2000/svg}line")) == 4


def test_tick_range_preserves_indices_empty_ticks_and_full_qubit_context():
    circuit = TickCircuit()
    circuit.tick().h([2, 40])
    circuit.tick()
    circuit.tick().x([2])
    circuit.tick().z([40])
    selected = circuit.render(start=1, stop=3, title="Window <A>")
    root = ET.fromstring(selected)  # noqa: S314 - locally generated SVG
    labels = list(root.itertext())
    assert "t1" in labels
    assert "t2" in labels
    assert "t0" not in labels
    assert "t3" not in labels
    assert "q40" in labels
    assert "q2" in labels
    assert "X" in labels
    assert "H" not in labels
    assert "Z" not in labels
    assert "Window <A> | TickCircuit: 2 qubits, ticks 1..3 of 4" in labels
    assert circuit.num_ticks() == 4
    assert "ticks 3..4 of 4" in circuit.render(start=3)
    assert "ticks 0..1 of 4" in circuit.render(stop=1)
    assert "ticks 4..4 of 4" in circuit.render(start=4)


def test_dag_range_uses_layers_not_original_ticks():
    tick = TickCircuit()
    tick.tick().h([0])
    tick.tick().x([1])
    tick.tick().cx([(0, 1)])
    dag = tick.to_dag_circuit()
    labels = list(ET.fromstring(dag.render(start=1)).itertext())  # noqa: S314
    assert "l1" in labels
    assert "l0" not in labels
    assert "DagCircuit: 2 qubits, layers 1..2 of 2" in labels
    with pytest.raises(ValueError, match="range"):
        dag.render(stop=3)


@pytest.mark.parametrize("bounds", [{"start": -1}, {"stop": -1}, {"start": 2, "stop": 1}, {"stop": 5}])
def test_render_rejects_invalid_ranges(bounds):
    circuit = TickCircuit()
    circuit.tick().h([0])
    with pytest.raises(ValueError, match="range"):
        circuit.render(**bounds)


@pytest.mark.parametrize("output_format", ["ascii", "unicode", "svg", "tikz", "dot"])
def test_measurement_ids_preserve_sparse_operand_mapping_and_ranges(output_format):
    circuit = TickCircuit()
    circuit.tick().h([2, 8])
    circuit.tick().mz_with_ids([8, 2], [42, 7])
    for source in [circuit, circuit.to_dag_circuit()]:
        assert source.render(output_format) == source.render(output_format, show_measurement_ids=False)
        shown = source.render(output_format, start=1, stop=2, show_measurement_ids=True)
        assert "MZ[m42]" in shown
        assert "MZ[m7]" in shown
        if output_format == "ascii":
            rows = {row.split(":", 1)[0].strip(): row for row in shown.splitlines() if row.startswith("q")}
            assert "MZ[m42]" in rows["q8"]
            assert "MZ[m7]" in rows["q2"]


@pytest.mark.parametrize(("gate_type", "symbol"), [("MZ", "MZ"), ("MeasureLeaked", "MZ_L")])
def test_measurement_labels_do_not_assign_missing_ids(gate_type, symbol):
    circuit = TickCircuit()
    circuit.tick().add_gate(gate_type, [2, 8])
    before = circuit.to_ascii()
    shown = circuit.render("ascii", show_measurement_ids=True)
    assert shown.count(f"{symbol}[m?]") == 2
    assert circuit.to_ascii() == before


def test_explicit_leakage_ids_keep_operand_order_metadata_and_reserve_ids():
    circuit = TickCircuit()
    handle = circuit.tick()
    assert handle.measure_leaked_with_ids([8, 2], [42, 7]) is handle
    handle.meta("round", 3)
    gate = circuit.get_tick(0).gate_batches()[0]
    assert gate.gate_type.name == "MeasureLeaked"
    assert gate.qubits == [8, 2]
    assert gate.meas_ids == [42, 7]
    assert circuit.get_tick(0).get_gate_attr(0, "round") == 3
    rows = {
        row.split(":", 1)[0].strip(): row
        for row in circuit.render("ascii", show_measurement_ids=True).splitlines()
        if row.startswith("q")
    }
    assert "MZ_L[m42]" in rows["q8"]
    assert "MZ_L[m7]" in rows["q2"]
    circuit.tick().mz([0])
    assert circuit.get_tick(1).gate_batches()[0].meas_ids == [43]


@pytest.mark.parametrize(("qubits", "ids"), [([0], []), ([0, 1], [7, 7]), ([0], [2**64 - 1])])
def test_invalid_explicit_leakage_ids_leave_gate_and_counter_unchanged(qubits, ids):
    circuit = TickCircuit()
    handle = circuit.tick()
    with pytest.raises(ValueError, match="meas_ids"):
        handle.measure_leaked_with_ids(qubits, ids)
    assert circuit.get_tick(0).gate_batches() == []
    handle.mz([2])
    assert circuit.get_tick(0).gate_batches()[0].meas_ids == [0]


def test_conflicting_explicit_leakage_measurement_does_not_reserve_ids():
    circuit = TickCircuit()
    handle = circuit.tick().h([0])
    with pytest.raises(ValueError, match="already in use"):
        handle.measure_leaked_with_ids([0], [42])
    circuit.tick().mz([1])
    assert circuit.get_tick(1).gate_batches()[0].meas_ids == [0]
