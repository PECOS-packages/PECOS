# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Source associations across circuit rendering and audited trace replay."""

import copy
import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pytest
from pecos.quantum import TickCircuit
from pecos.visualization import circuit_diagram, inspect_circuit


def trace_for(gates, *, engine_id=5):
    """A completed lowered stream without measurements or source-node guesses."""
    return [
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": engine_id,
            "shot_index": 0,
            "chunk_index": 0,
            "stage": "execute",
            "operations": [],
            "num_operations": 0,
            "lowered_quantum_ops": gates,
            "lowered_quantum_ops_complete": True,
        },
        {
            "format": "pecos_qis_operation_trace_v1",
            "engine_trace_id": engine_id,
            "shot_index": 0,
            "chunk_index": 1,
            "stage": "trace_complete",
            "operations": [],
            "num_operations": 0,
            "lowered_quantum_ops": [],
        },
    ]


def svg_sources(svg):
    root = ET.fromstring(svg)  # noqa: S314 - locally generated SVG
    return [node.attrib["data-pecos-operation"] for node in root.iter() if "data-pecos-operation" in node.attrib]


def test_snapshot_ranges_links_and_independent_records(tmp_path):
    circuit = TickCircuit()
    circuit.tick().h([2, 8])
    circuit.tick().cx([(8, 2)]).meta("note", '<metadata & "quotes">')
    circuit.tick().mz_with_ids([8, 2], [42, 7])
    snapshot = inspect_circuit(circuit, show_measurement_ids=True)
    selected = inspect_circuit(circuit, start=1, stop=2)
    assert selected.operations[0]["id"] == snapshot.operations[1]["id"] == "tick:1:batch:0"
    assert selected.operations[0]["source"] == {"kind": "tick_batch", "tick": 1, "batch": 0}
    assert selected.operations[0]["qubits"] == [8, 2]
    assert selected.operations[0]["metadata"]["note"] == '<metadata & "quotes">'
    assert svg_sources(selected.svg) == ["tick:1:batch:0"] * 3
    assert snapshot.operations[-1]["measurement_ids"] == [42, 7]
    assert "MZ[m42]" in snapshot.svg
    assert "MZ[m7]" in snapshot.svg
    records = snapshot.operations
    records[0]["qubits"].append(99)
    circuit.tick().h([99])
    assert 99 not in snapshot.operations[0]["qubits"]
    assert "q99" not in snapshot.svg
    assert "data-pecos-operation" not in circuit.to_svg()
    snapshot.save(tmp_path / "inspection.json")
    snapshot.save_svg(tmp_path / "inspection.svg")
    assert json.loads((tmp_path / "inspection.json").read_text())["format"] == "pecos_circuit_inspection_v1"
    assert (tmp_path / "inspection.svg").read_text() == snapshot._repr_svg_()


def test_batch_expansion_and_dag_node_sources():
    circuit = TickCircuit()
    circuit.tick().cx([(0, 2), (1, 3)])
    snapshot = inspect_circuit(circuit)
    assert len(snapshot.operations) == 1
    assert len(svg_sources(snapshot.svg)) == 6
    dag = circuit.to_dag_circuit()
    snapshot = inspect_circuit(dag)
    for operation in snapshot.operations:
        assert operation["source"]["kind"] == "dag_node"
        node = operation["source"]["node"]
        assert dag.gate(node).gate_type.name == operation["gate_type"]
    assert set(svg_sources(snapshot.svg)) == {operation["id"] for operation in snapshot.operations}


@pytest.mark.parametrize("as_dag", [False, True])
@pytest.mark.parametrize("bounds", [{}, {"start": 0, "stop": 2}])
def test_annotations_do_not_change_full_or_range_rendering(as_dag, bounds):
    circuit = TickCircuit()
    circuit.tick().h([0, 1])
    circuit.tick().cx([(0, 1)])
    if as_dag:
        circuit = circuit.to_dag_circuit()
    snapshot = inspect_circuit(circuit, **bounds)
    without_groups = (
        "\n".join(
            line
            for line in snapshot.svg.splitlines()
            if not line.startswith("<g data-pecos-operation=") and line != "</g>"
        )
        + "\n"
    )
    assert without_groups == str(circuit_diagram(circuit, **bounds))


def test_trace_origins_survive_compaction_and_override_metadata_collisions():
    trace = trace_for(
        [
            {"gate_type": "H", "qubits": [0], "metadata": {"_pecos_trace_origin": "forged"}},
            {"gate_type": "H", "qubits": [1]},
        ],
        engine_id=2**64 - 1,
    )
    original = copy.deepcopy(trace)
    snapshot = inspect_circuit(trace)
    assert trace == original
    assert len(snapshot.operations) == 2
    assert {op["source"]["tick"] for op in snapshot.operations} == {0}
    origins = [op["metadata"]["_pecos_trace_origin"] for op in snapshot.operations]
    assert [origin["lowered_op_index"] for origin in origins] == [0, 1]
    assert all(origin["engine_trace_id"] == 2**64 - 1 for origin in origins)
    assert all(origin["chunk_index"] == 0 and origin["shot_index"] == 0 for origin in origins)


def test_each_decomposition_stage_retains_trace_origin_and_metadata():
    trace = trace_for(
        [
            {"gate_type": "CRZ", "qubits": [0, 1], "angles": [0.5], "metadata": {"source_label": "rotation"}},
        ],
    )
    snapshot = inspect_circuit(trace)
    assert len(snapshot.operations) >= 2
    assert all(op["gate_type"] != "CRZ" for op in snapshot.operations)
    origins = [op["metadata"]["_pecos_trace_origin"] for op in snapshot.operations]
    assert all(origin == origins[0] for origin in origins)
    assert all(op["metadata"]["source_label"] == "rotation" for op in snapshot.operations)
    assert len({op["id"] for op in snapshot.operations}) == len(snapshot.operations)
    selected = inspect_circuit(trace, start=1, stop=2)
    assert selected.operations[0]["id"] in {op["id"] for op in snapshot.operations}
    assert selected.operations[0]["metadata"]["_pecos_trace_origin"] == origins[0]


def test_invalid_trace_still_rejected():
    trace = trace_for([{"gate_type": "H", "qubits": [0]}])
    trace.pop()
    with pytest.raises(ValueError, match="terminal trace_complete"):
        inspect_circuit(trace)


def test_empty_and_invalid_ranges():
    circuit = TickCircuit()
    circuit.tick().h([0])
    empty = inspect_circuit(circuit, start=1, stop=1)
    assert empty.operations == []
    assert svg_sources(empty.svg) == []
    for start, stop in [(-1, 1), (1, 0), (0, 2)]:
        with pytest.raises(ValueError, match="range"):
            inspect_circuit(circuit, start=start, stop=stop)


def test_nonfinite_metadata_is_not_silently_lost():
    circuit = TickCircuit()
    circuit.tick().h([0]).meta("unknown", float("nan"))
    with pytest.raises(ValueError, match="JSON"):
        inspect_circuit(circuit)


@pytest.mark.parametrize("as_dag", [False, True])
def test_interactive_html_is_optional_in_native_snapshot_and_separate_from_json(as_dag, tmp_path):
    circuit = TickCircuit()
    circuit.tick().cx([(0, 2), (1, 3)])
    if as_dag:
        circuit = circuit.to_dag_circuit()
    assert "html" not in circuit.inspect()
    native = circuit.inspect(include_html=True)
    inspection = inspect_circuit(circuit)
    assert inspection.to_html() == native["html"]
    assert "html" not in json.loads(inspection.to_json())
    assert inspection._repr_svg_() == native["svg"]
    inspection.save_html(tmp_path / "inspection.html")
    inspection.interactive().save(tmp_path / "viewer.html")
    assert (tmp_path / "inspection.html").read_text() == native["html"]
    assert (tmp_path / "viewer.html").read_text() == native["html"]


def test_notebook_frame_is_isolated_and_metadata_cannot_break_out():
    circuit = TickCircuit()
    attack = '</pre></iframe><script>globalThis.injected = true</script> & "quotes"'
    circuit.tick().h([0]).meta("note", attack)
    inspection = inspect_circuit(circuit)
    assert attack not in inspection.to_html()
    assert "&lt;script&gt;globalThis.injected = true&lt;/script&gt;" in inspection.to_html()

    class FrameParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append((tag, dict(attrs)))

    parser = FrameParser()
    parser.feed(inspection.interactive()._repr_html_())
    assert len(parser.tags) == 1
    tag, attrs = parser.tags[0]
    assert tag == "iframe"
    assert attrs["sandbox"] == "allow-scripts"
    assert attrs["srcdoc"] == inspection.to_html()


def test_html_keeps_large_trace_ids_as_exact_text():
    trace = trace_for([{"gate_type": "H", "qubits": [0]}], engine_id=2**64 - 1)
    html = inspect_circuit(trace).to_html()
    assert "18446744073709551615" in html
    assert "JSON.parse" not in html
    assert "No stored operations in this view." in inspect_circuit(TickCircuit()).to_html()


def html_operation_records(html):
    class RecordParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.records = []

        def handle_starttag(self, tag, attrs):
            if tag == "article":
                self.records.append(dict(attrs))

    parser = RecordParser()
    parser.feed(html)
    return parser.records


def test_html_links_decomposed_trace_stages_with_exact_origins_and_scoped_ranges():
    trace = trace_for([{"gate_type": "CRZ", "qubits": [0, 1], "angles": [0.5]}], engine_id=2**64 - 1)
    full = html_operation_records(inspect_circuit(trace).to_html())
    assert len(full) >= 2
    assert {record["data-pecos-trace-origin"] for record in full} == {"qis:18446744073709551615:0:0:0"}
    selected = html_operation_records(inspect_circuit(trace, start=1, stop=2).to_html())
    assert len(selected) == 1
    assert selected[0] in full


def test_html_trace_groups_distinguish_every_coordinate_and_ignore_missing_origins():
    circuit = TickCircuit()
    origin = {
        "format": "pecos_qis_lowered_origin_v1",
        "engine_trace_id": 2**64 - 1,
        "shot_index": 0,
        "chunk_index": 0,
        "lowered_op_index": 0,
    }
    circuit.tick().h([0]).meta("_pecos_trace_origin", origin)
    for field, value in [
        ("engine_trace_id", 2**64 - 2),
        ("shot_index", 1),
        ("chunk_index", 1),
        ("lowered_op_index", 1),
    ]:
        circuit.tick().h([0]).meta("_pecos_trace_origin", {**origin, field: value})
    circuit.tick().h([0])
    circuit.tick().h([0]).meta("_pecos_trace_origin", {"engine_trace_id": 0})
    circuit.tick().h([0]).meta("_pecos_trace_origin", {**origin, "format": "unknown"})
    records = html_operation_records(inspect_circuit(circuit).to_html())
    assert len({record["data-pecos-trace-origin"] for record in records[:5]}) == 5
    assert all("data-pecos-trace-origin" not in record for record in records[5:])


def test_html_search_aliases_preserve_qubit_and_measurement_identity():
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([3, 30], [7, 42])
    records = html_operation_records(inspect_circuit(circuit).to_html())
    assert records[0]["data-pecos-search-ids"] == "q3 q30 m7 m42"
