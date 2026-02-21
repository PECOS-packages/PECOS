# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for the HUGR to DAG converter."""

from __future__ import annotations

import pytest

# Check if guppylang is available
try:
    from guppylang import guppy
    from guppylang.std.quantum import cx, cz, h, measure, qubit, s, t, x, y, z

    HAS_GUPPYLANG = True
except ImportError:
    HAS_GUPPYLANG = False

# Check if hugr_to_dag is available (requires hugr)
try:
    from pecos.circuit_converters.hugr_to_dag import (
        UnsupportedHugrStructureError,
        dag_to_gate_sequence,
        guppy_to_dag,
        hugr_to_dag,
    )

    HAS_HUGR_TO_DAG = True
except ImportError:
    HAS_HUGR_TO_DAG = False
pytestmark = pytest.mark.skipif(
    not (HAS_GUPPYLANG and HAS_HUGR_TO_DAG),
    reason="guppylang or hugr not available",
)


class TestBasicConversion:
    """Tests for basic HUGR to DAG conversion."""

    def test_single_qubit_circuit(self) -> None:
        """Test conversion of a single-qubit circuit."""

        @guppy
        def single_h() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        package = single_h.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        # Should have 3 nodes: QAlloc, H, MeasureFree
        assert len(dag.nodes()) == 3

        # Check node attributes
        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "QAlloc" in op_names
        assert "H" in op_names
        assert "MeasureFree" in op_names

    def test_bell_state_circuit(self) -> None:
        """Test conversion of a Bell state circuit."""

        @guppy
        def bell() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        package = bell.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        # Should have 6 nodes: 2 QAlloc, H, CX, 2 MeasureFree
        assert len(dag.nodes()) == 6

        # Check that edges exist (dependencies)
        assert dag.edge_count() >= 4  # At least QAlloc->H, H->CX, QAlloc->CX, CX->Measure

    def test_multi_gate_circuit(self) -> None:
        """Test conversion of a circuit with multiple gate types."""

        @guppy
        def multi_gate() -> bool:
            q = qubit()
            h(q)
            t(q)
            s(q)
            x(q)
            y(q)
            z(q)
            return measure(q)

        package = multi_gate.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        # Check all gate types are present
        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "H" in op_names
        assert "T" in op_names
        assert "S" in op_names
        assert "X" in op_names
        assert "Y" in op_names
        assert "Z" in op_names

    def test_two_qubit_gates(self) -> None:
        """Test conversion with two-qubit gates."""

        @guppy
        def two_qubit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            cx(q0, q1)
            cz(q0, q1)
            return measure(q0), measure(q1)

        package = two_qubit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "CX" in op_names
        assert "CZ" in op_names


class TestFilteringOptions:
    """Tests for filtering options (include_alloc, include_measure)."""

    def test_gates_only(self) -> None:
        """Test extracting only gates (no alloc/measure)."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            t(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr, include_alloc=False, include_measure=False)

        # Should only have H and T gates
        assert len(dag.nodes()) == 2

        op_types = [dag.node_attrs(n).get("op_type") for n in dag.nodes()]
        assert all(t == "gate" for t in op_types)

    def test_no_alloc(self) -> None:
        """Test excluding allocation nodes."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr, include_alloc=False)

        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "QAlloc" not in op_names
        assert "H" in op_names
        assert "MeasureFree" in op_names

    def test_no_measure(self) -> None:
        """Test excluding measurement nodes."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr, include_measure=False)

        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "QAlloc" in op_names
        assert "H" in op_names
        assert "MeasureFree" not in op_names


class TestNodeAttributes:
    """Tests for node attribute correctness."""

    def test_op_type_classification(self) -> None:
        """Test that op_type is correctly set for different operations."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        for node in dag.nodes():
            attrs = dag.node_attrs(node)
            op_name = attrs.get("op_name")
            op_type = attrs.get("op_type")

            if op_name == "QAlloc":
                assert op_type == "alloc"
            elif op_name == "H":
                assert op_type == "gate"
            elif op_name == "MeasureFree":
                assert op_type == "measure"

    def test_extension_attribute(self) -> None:
        """Test that extension attribute is set correctly."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        for node in dag.nodes():
            attrs = dag.node_attrs(node)
            ext = attrs.get("extension")
            assert ext == "tket.quantum"

    def test_hugr_node_idx_attribute(self) -> None:
        """Test that hugr_node_idx attribute is set."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        for node in dag.nodes():
            attrs = dag.node_attrs(node)
            hugr_idx = attrs.get("hugr_node_idx")
            assert hugr_idx is not None
            assert isinstance(hugr_idx, int)


class TestTopologicalSort:
    """Tests for topological sort functionality."""

    def test_dag_to_gate_sequence(self) -> None:
        """Test that dag_to_gate_sequence returns operations in valid order."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            t(q)
            s(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)
        sequence = dag_to_gate_sequence(dag)

        # Check sequence is a list of dicts
        assert isinstance(sequence, list)
        assert all(isinstance(g, dict) for g in sequence)

        # Check each entry has required keys
        for gate in sequence:
            assert "op_name" in gate
            assert "op_type" in gate
            assert "node_idx" in gate

    def test_topological_order_validity(self) -> None:
        """Test that topological order respects dependencies."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            t(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)
        sequence = dag_to_gate_sequence(dag)

        # Get indices in sequence order
        op_order = {g["op_name"]: i for i, g in enumerate(sequence)}

        # QAlloc should come before H
        assert op_order.get("QAlloc", 0) < op_order.get("H", 999)
        # H should come before T (in single-qubit chain)
        assert op_order.get("H", 0) < op_order.get("T", 999)
        # T should come before MeasureFree
        assert op_order.get("T", 0) < op_order.get("MeasureFree", 999)


class TestEdgeDependencies:
    """Tests for edge/dependency correctness."""

    def test_single_qubit_chain_edges(self) -> None:
        """Test edge structure for a single-qubit chain."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            t(q)
            return measure(q)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr, include_alloc=False, include_measure=False)

        # Should have H -> T edge for gates-only
        edges = list(dag.edges())
        assert len(edges) >= 1

    def test_two_qubit_parallel_edges(self) -> None:
        """Test edge structure for parallel operations on two qubits."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            h(q1)  # Parallel to first H
            cx(q0, q1)
            return measure(q0), measure(q1)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        # Both H gates should connect to CX
        # CX should connect to both measurements
        assert dag.edge_count() >= 4


class TestEmptyAndMinimalCircuits:
    """Tests for edge cases with minimal circuits."""

    def test_single_allocation_and_measure(self) -> None:
        """Test circuit with just allocation and measurement."""

        @guppy
        def minimal() -> bool:
            q = qubit()
            return measure(q)

        package = minimal.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        # Should have QAlloc and MeasureFree
        assert len(dag.nodes()) == 2

        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "QAlloc" in op_names
        assert "MeasureFree" in op_names

    def test_gates_only_on_minimal_circuit(self) -> None:
        """Test gates-only mode on circuit with no gates."""

        @guppy
        def minimal() -> bool:
            q = qubit()
            return measure(q)

        package = minimal.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr, include_alloc=False, include_measure=False)

        # Should have no nodes (no gates)
        assert len(dag.nodes()) == 0


class TestDAGProperties:
    """Tests for DAG properties and structure."""

    def test_dag_is_acyclic(self) -> None:
        """Test that the resulting DAG has no cycles."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            h(q1)
            return measure(q0), measure(q1)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        # If it's a valid DAG, topological_sort should work without error
        sorted_nodes = dag.topological_sort()
        assert len(sorted_nodes) == len(dag.nodes())

    def test_roots_are_allocations(self) -> None:
        """Test that DAG roots are allocation nodes."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        roots = dag.roots()
        for root in roots:
            op_type = dag.node_attrs(root).get("op_type")
            assert op_type == "alloc"

    def test_leaves_are_measurements(self) -> None:
        """Test that DAG leaves are measurement nodes."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        package = circuit.compile()
        hugr = package.modules[0]

        dag = hugr_to_dag(hugr)

        leaves = dag.leaves()
        for leaf in leaves:
            op_type = dag.node_attrs(leaf).get("op_type")
            assert op_type == "measure"


class TestGuppyToDag:
    """Tests for the guppy_to_dag convenience wrapper."""

    def test_basic_usage(self) -> None:
        """Test basic guppy_to_dag usage."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            return measure(q)

        dag = guppy_to_dag(circuit)

        # Should have 3 nodes: QAlloc, H, MeasureFree
        assert len(dag.nodes()) == 3

        op_names = [dag.node_attrs(n).get("op_name") for n in dag.nodes()]
        assert "QAlloc" in op_names
        assert "H" in op_names
        assert "MeasureFree" in op_names

    def test_with_filtering_options(self) -> None:
        """Test guppy_to_dag with filtering options."""

        @guppy
        def circuit() -> bool:
            q = qubit()
            h(q)
            t(q)
            return measure(q)

        dag = guppy_to_dag(circuit, include_alloc=False, include_measure=False)

        # Should only have H and T gates
        assert len(dag.nodes()) == 2

        op_types = [dag.node_attrs(n).get("op_type") for n in dag.nodes()]
        assert all(t == "gate" for t in op_types)

    def test_equivalent_to_manual_compilation(self) -> None:
        """Test that guppy_to_dag produces same result as manual compilation."""

        @guppy
        def circuit() -> tuple[bool, bool]:
            q0 = qubit()
            q1 = qubit()
            h(q0)
            cx(q0, q1)
            return measure(q0), measure(q1)

        # Using guppy_to_dag
        dag1 = guppy_to_dag(circuit)

        # Manual compilation
        package = circuit.compile()
        dag2 = hugr_to_dag(package.modules[0])

        # Should have same structure
        assert len(dag1.nodes()) == len(dag2.nodes())
        assert dag1.edge_count() == dag2.edge_count()

        # Same operations
        ops1 = sorted(dag1.node_attrs(n).get("op_name") for n in dag1.nodes())
        ops2 = sorted(dag2.node_attrs(n).get("op_name") for n in dag2.nodes())
        assert ops1 == ops2
