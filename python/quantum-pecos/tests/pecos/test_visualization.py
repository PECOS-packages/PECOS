# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Circuit diagram snapshots and audited trace rendering."""

import copy
import json
import xml.etree.ElementTree as ET

import pytest
from pecos.quantum import TickCircuit
from pecos.visualization import circuit_diagram


def completed_trace():
    return [
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": 5,
            "shot_index": 0,
            "chunk_index": 0,
            "stage": "execute",
            "operations": [
                {"AllocateQubit": {"id": 0}},
                {"Quantum": {"H": 0}},
                {"Quantum": {"Measure": [0, 7]}},
            ],
            "num_operations": 3,
            "lowered_quantum_ops": [
                {"gate_type": "PZ", "qubits": [0], "angles": [], "params": []},
                {"gate_type": "RZ", "qubits": [0], "angles": [0.5], "params": []},
                {
                    "gate_type": "MZ",
                    "qubits": [0],
                    "angles": [],
                    "params": [],
                    "measurement_result_ids": [7],
                },
            ],
            "lowered_quantum_ops_complete": True,
        },
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": 5,
            "shot_index": 0,
            "chunk_index": 1,
            "stage": "trace_complete",
            "operations": [],
            "num_operations": 0,
            "lowered_quantum_ops": [],
            "named_result_traces": [],
        },
    ]


def test_diagram_is_a_snapshot_and_exports_utf8(tmp_path):
    circuit = TickCircuit()
    circuit.tick().h([0])
    diagram = circuit_diagram(circuit)
    circuit.tick().x([1])
    root = ET.fromstring(str(diagram))  # noqa: S314 - locally generated SVG
    labels = " ".join(root.itertext())
    assert "q1" not in labels
    assert diagram._repr_svg_() == str(diagram)
    path = tmp_path / "circuit.svg"
    diagram.save(path)
    assert path.read_text(encoding="utf-8") == str(diagram)


def test_trace_uses_lowered_operations_without_capture_or_mutation(monkeypatch):
    import pecos.tracing

    def unexpected_capture(*_args, **_kwargs):
        pytest.fail("rendering a saved trace must not execute a program")

    monkeypatch.setattr(pecos.tracing, "capture_qis_operation_trace", unexpected_capture)
    trace = json.loads(json.dumps(completed_trace()))
    original = copy.deepcopy(trace)
    diagram = circuit_diagram(trace, format="unicode")
    assert "RZ(" in str(diagram)
    assert "[H]" not in str(diagram)
    assert "3 ticks" in str(diagram)
    assert diagram._repr_svg_() is None
    assert trace == original


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("incomplete", "terminal trace_complete"),
        ("mixed_shots", "shot"),
        ("unaudited", "complete lowered gate stream"),
        ("missing_measurement_id", "missing measurement_result_ids"),
    ],
)
def test_diagram_rejects_invalid_traces(malformation, message):
    trace = completed_trace()
    if malformation == "incomplete":
        trace.pop()
    elif malformation == "mixed_shots":
        trace[-1]["shot_index"] = 1
    elif malformation == "unaudited":
        trace[0]["lowered_quantum_ops_complete"] = False
    else:
        del trace[0]["lowered_quantum_ops"][-1]["measurement_result_ids"]
    with pytest.raises(ValueError, match=message):
        circuit_diagram(trace)


def test_dag_and_text_notebook_display():
    class Printer:
        def __init__(self):
            self.output = ""

        def text(self, text):
            self.output += text

    tick = TickCircuit()
    tick.tick().h([0])
    diagram = circuit_diagram(tick.to_dag_circuit(), format="ascii")
    assert "DagCircuit" in str(diagram)
    printer = Printer()
    diagram._repr_pretty_(printer, cycle=False)
    assert printer.output == str(diagram)
    assert diagram._repr_svg_() is None


def test_unsupported_inputs_and_options():
    with pytest.raises(TypeError, match="source must be"):
        circuit_diagram("program.hugr")
    with pytest.raises(ValueError, match="format must be"):
        circuit_diagram(TickCircuit(), format="png")
    with pytest.raises(ValueError, match="angle_unit must be"):
        circuit_diagram(TickCircuit(), angle_unit="degrees")


def test_named_panels_replay_complete_trace_once(monkeypatch):
    import pecos.tracing
    from pecos.tracing import _qis_operation_trace_to_visualization_circuit
    from pecos.visualization import circuit_diagram_panels

    original = _qis_operation_trace_to_visualization_circuit
    calls = []

    def replay(trace):
        calls.append(trace)
        return original(trace)

    monkeypatch.setattr(pecos.tracing, "_qis_operation_trace_to_visualization_circuit", replay)
    trace = completed_trace()
    # Overlap, repeated titles, and an empty range remain distinct panels.
    panels = circuit_diagram_panels(trace, [("A", 0, 2), ("A", 1, 3), ("Empty", 3, 3)])
    assert len(calls) == 1
    assert len(panels) == 3
    for panel, selection in zip(panels, ["0..2", "1..3", "3..3"], strict=True):
        assert f"ticks {selection} of 3" in str(panel)
        assert panel._repr_svg_() == str(panel)
    assert "A | TickCircuit" in str(panels[0])


def test_range_cannot_bypass_trace_validation():
    from pecos.visualization import circuit_diagram_panels

    trace = completed_trace()[:-1]
    with pytest.raises(ValueError, match="terminal trace_complete"):
        circuit_diagram(trace, start=0, stop=1)
    with pytest.raises(ValueError, match="terminal trace_complete"):
        circuit_diagram_panels(trace, [])


@pytest.mark.parametrize("gate_type", ["MZ", "MeasureLeaked"])
def test_trace_measurement_labels_keep_source_identity(gate_type):
    from pecos.visualization import circuit_diagram_panels

    trace = completed_trace()
    trace[0]["lowered_quantum_ops"][-1]["gate_type"] = gate_type
    original = copy.deepcopy(trace)
    diagram = circuit_diagram(trace, start=2, stop=3, show_measurement_ids=True)
    labels = list(ET.fromstring(str(diagram)).itertext())  # noqa: S314 - generated SVG
    symbol = "MZ_L" if gate_type == "MeasureLeaked" else "MZ"
    assert f"{symbol}[m7]" in labels
    panels = circuit_diagram_panels(trace, [("Readout", 2, 3)], format="tikz", show_measurement_ids=True)
    assert f"{symbol.replace('_', r'\_')}[m7]" in str(panels[0])
    assert "Readout" in str(panels[0])
    assert "Boolean projection" not in str(diagram)
    assert trace == original


def mixed_measurement_trace(kinds=("MZ", "MeasureLeaked", "MZ"), *, different_metadata=False):
    trace = completed_trace()
    trace[0]["operations"] = [{"Quantum": {"Measure": [q, mid]}} for q, mid in [(4, 42), (1, 7), (3, 20)]]
    trace[0]["lowered_quantum_ops"] = [
        {
            "gate_type": kind,
            "qubits": [q],
            "angles": [],
            "params": [],
            "measurement_result_ids": [mid],
            "metadata": {"round": index if different_metadata else 2},
        }
        for index, (kind, q, mid) in enumerate(zip(kinds, [4, 1, 3], [42, 7, 20], strict=True))
    ]
    return trace


@pytest.mark.parametrize("kinds", [("MZ", "MeasureLeaked", "MZ"), ("MeasureLeaked", "MZ", "MeasureLeaked")])
@pytest.mark.parametrize("different_metadata", [False, True])
def test_visual_replay_preserves_native_measurements_and_metadata(kinds, different_metadata):
    from pecos.tracing import _qis_operation_trace_to_visualization_circuit, qis_operation_trace_to_tick_circuit

    trace = mixed_measurement_trace(kinds, different_metadata=different_metadata)
    original = copy.deepcopy(trace)
    circuit = _qis_operation_trace_to_visualization_circuit(trace)
    for _ in range(2):
        circuit.compact_ticks()
        batches = list(circuit.gate_batches())
        assert [(g.gate_type.name, g.qubits, g.meas_ids) for _, g in batches] == [
            (kind, [q], [mid]) for kind, q, mid in zip(kinds, [4, 1, 3], [42, 7, 20], strict=True)
        ]
        tick = circuit.get_tick(0)
        assert [tick.get_gate_attr(i, "round") for i in range(3)] == ([0, 1, 2] if different_metadata else [2, 2, 2])
    analysis = qis_operation_trace_to_tick_circuit(trace)
    assert all(g.gate_type.name == "MZ" for _, g in analysis.gate_batches())
    assert trace == original


@pytest.mark.parametrize("output_format", ["ascii", "unicode", "svg", "tikz", "dot"])
def test_mixed_trace_exports_keep_labels_and_titles(output_format):
    output = str(
        circuit_diagram(mixed_measurement_trace(), format=output_format, title="Readout", show_measurement_ids=True),
    )
    for label in ["MZ[m42]", "MZ_L[m7]", "MZ[m20]"]:
        assert (label.replace("_", r"\_") if output_format == "tikz" else label) in output
    assert "Readout" in output
    assert "Boolean projection" not in output


@pytest.mark.parametrize("invalid", ["duplicate", "cardinality", "missing_ids", "source_mismatch", "incomplete"])
def test_preserved_leakage_trace_rejects_invalid_provenance_even_outside_panel(invalid):
    from pecos.visualization import circuit_diagram_panels

    trace = mixed_measurement_trace()
    leakage = trace[0]["lowered_quantum_ops"][1]
    if invalid == "duplicate":
        leakage["measurement_result_ids"] = [42]
    elif invalid == "cardinality":
        leakage["measurement_result_ids"] = [7, 8]
    elif invalid == "missing_ids":
        del leakage["measurement_result_ids"]
    elif invalid == "source_mismatch":
        leakage["measurement_result_ids"] = [99]
    else:
        trace.pop()
    message = {
        "duplicate": "duplicate measurement result ids",
        "cardinality": "measurement_result_ids for",
        "missing_ids": "missing measurement_result_ids",
        "source_mismatch": "identities do not match",
        "incomplete": "terminal trace_complete",
    }[invalid]
    with pytest.raises(ValueError, match=message):
        circuit_diagram(trace, start=0, stop=0)
    with pytest.raises(ValueError, match=message):
        circuit_diagram_panels(trace, [])
