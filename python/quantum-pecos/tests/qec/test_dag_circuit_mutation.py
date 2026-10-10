# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""DAG mutations preserve dependencies through the Python bindings."""

from pecos.quantum import DagCircuit, TickCircuit


def test_append_after_tick_conversion() -> None:
    ticks = TickCircuit()
    ticks.tick().x([0])
    ticks.tick().mz([0])
    dag = ticks.to_dag_circuit()
    dag.h([0])
    assert (1, 2, 0) in dag.wires()
    assert dag.topological_order() == [0, 1, 2]


def test_remove_bridges_before_reusing_slot() -> None:
    dag = DagCircuit()
    dag.h([0]).h([0]).h([0])
    dag.remove_gate(1)
    assert dag.wires() == [(0, 2, 0)]
    dag.remove_gate(2)
    dag.h([0])
    assert dag.wires() == [(0, 2, 0)]
    assert dag.topological_order() == [0, 2]


def assert_rendered_program_order(dag: DagCircuit) -> None:
    """Renderer node IDs follow the input program; each qubit must stay ordered."""
    heads = {}
    wires = set(dag.wires())
    order = {node: position for position, node in enumerate(dag.topological_order())}
    for node in dag.nodes():
        for qubit in dag.gate(node).qubits:
            if qubit in heads:
                previous = heads[qubit]
                assert (previous, node, qubit) in wires
                assert order[previous] < order[node]
            heads[qubit] = node


def test_surface_szz_renderer_preserves_program_order() -> None:
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.circuit_builder import generate_dag_circuit_from_patch

    patch = SurfacePatch.create(distance=3)
    dag = generate_dag_circuit_from_patch(patch, num_rounds=1, interaction_basis="szz")
    names = {dag.gate(node).gate_type.name for node in dag.nodes()}
    assert {"SX", "SXdg"} <= names
    assert_rendered_program_order(dag)


def test_state_injection_renderer_preserves_program_order() -> None:
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.circuit_builder import DagCircuitRenderer
    from pecos.qec.surface.injection import state_injection

    patch = SurfacePatch.create(distance=3)
    for state, name in [("T", "T"), ("TDG", "Tdg")]:
        injection = state_injection(patch, state=state)
        dag = DagCircuitRenderer().render(
            list(injection.seed.steps),
            injection.seed.allocations[0],
            patch,
            0,
            "Z",
        )
        assert any(dag.gate(node).gate_type.name == name for node in dag.nodes())
        assert_rendered_program_order(dag)
