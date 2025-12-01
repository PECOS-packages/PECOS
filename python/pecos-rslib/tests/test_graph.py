# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for graph module (MWPM decoder)."""


import _pecos_rslib as pc


class TestGraphCreation:
    """Test Graph creation and basic operations."""

    def test_graph_new(self):
        """Test creating a new empty graph."""
        graph = pc.graph.Graph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_graph_with_capacity(self):
        """Test creating a graph with pre-allocated capacity."""
        graph = pc.graph.Graph.with_capacity(10, 20)
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_graph_repr(self):
        """Test graph string representation."""
        graph = pc.graph.Graph()
        assert str(graph) == "Graph(nodes=0, edges=0)"

        graph.add_node()
        graph.add_node()
        assert str(graph) == "Graph(nodes=2, edges=0)"


class TestGraphNodes:
    """Test node operations."""

    def test_add_single_node(self):
        """Test adding a single node."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()

        assert n0 == 0
        assert graph.node_count() == 1

    def test_add_multiple_nodes(self):
        """Test adding multiple nodes."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        assert n0 == 0
        assert n1 == 1
        assert n2 == 2
        assert graph.node_count() == 3

    def test_add_many_nodes(self):
        """Test adding many nodes."""
        graph = pc.graph.Graph()
        nodes = [graph.add_node() for _ in range(100)]

        assert len(nodes) == 100
        assert nodes == list(range(100))
        assert graph.node_count() == 100

    def test_nodes_empty_graph(self):
        """Test nodes() on empty graph."""
        graph = pc.graph.Graph()
        assert graph.nodes() == []

    def test_nodes_with_nodes(self):
        """Test nodes() returns correct node indices."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        nodes = graph.nodes()
        assert nodes == [0, 1, 2]
        assert n0 in nodes
        assert n1 in nodes
        assert n2 in nodes


class TestGraphEdges:
    """Test edge operations."""

    def test_add_single_edge(self):
        """Test adding a single edge."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 1.0)
        assert graph.edge_count() == 1

    def test_add_multiple_edges(self):
        """Test adding multiple edges."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(n1, n2)
        edge_id = graph.find_edge(n1, n2)
        graph.set_edge_weight(edge_id, 2.0)

        graph.add_edge(n0, n2)
        edge_id = graph.find_edge(n0, n2)
        graph.set_edge_weight(edge_id, 3.0)

        assert graph.edge_count() == 3

    def test_edge_weights(self):
        """Test edges with different weights."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 5.5)
        edges = graph.edges()

        assert len(edges) == 1
        assert edges[0] == (n0, n1, 5.5)

    def test_edges_list(self):
        """Test retrieving list of all edges."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(n1, n2)
        edge_id = graph.find_edge(n1, n2)
        graph.set_edge_weight(edge_id, 20.0)

        edges = graph.edges()
        assert len(edges) == 2

        # Check that both edges are present (order may vary)
        edge_set = {(e[0], e[1]) for e in edges}
        assert (n0, n1) in edge_set or (n1, n0) in edge_set
        assert (n1, n2) in edge_set or (n2, n1) in edge_set


class TestMaxWeightMatching:
    """Test maximum weight matching algorithm."""

    def test_matching_simple_pair(self):
        """Test matching with a single pair of nodes."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 10.0)
        matching = graph.max_weight_matching(False)

        # Both nodes should be matched to each other
        assert len(matching) == 2
        assert matching[n0] == n1
        assert matching[n1] == n0

    def test_matching_two_pairs(self):
        """Test matching with two separate pairs."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()
        n3 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(n2, n3)
        edge_id = graph.find_edge(n2, n3)
        graph.set_edge_weight(edge_id, 20.0)

        matching = graph.max_weight_matching(False)

        # All four nodes should be matched
        assert len(matching) == 4
        assert matching[n0] == n1
        assert matching[n1] == n0
        assert matching[n2] == n3
        assert matching[n3] == n2

    def test_matching_chooses_heaviest_edge(self):
        """Test that matching chooses the heaviest edge."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        # Triangle with different weights
        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(n1, n2)  # Heaviest edge
        edge_id = graph.find_edge(n1, n2)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(n0, n2)
        edge_id = graph.find_edge(n0, n2)
        graph.set_edge_weight(edge_id, 2.0)

        matching = graph.max_weight_matching(False)

        # Should match n1-n2 (heaviest edge) and leave n0 unmatched
        assert len(matching) == 2
        assert matching[n1] == n2
        assert matching[n2] == n1
        assert n0 not in matching

    def test_matching_complex_graph(self):
        """Test matching with a more complex graph."""
        graph = pc.graph.Graph()
        nodes = [graph.add_node() for _ in range(6)]

        # Create a graph with multiple possible matchings
        graph.add_edge(nodes[0], nodes[1])
        edge_id = graph.find_edge(nodes[0], nodes[1])
        graph.set_edge_weight(edge_id, 5.0)

        graph.add_edge(nodes[2], nodes[3])
        edge_id = graph.find_edge(nodes[2], nodes[3])
        graph.set_edge_weight(edge_id, 8.0)

        graph.add_edge(nodes[4], nodes[5])
        edge_id = graph.find_edge(nodes[4], nodes[5])
        graph.set_edge_weight(edge_id, 3.0)

        graph.add_edge(nodes[0], nodes[2])
        edge_id = graph.find_edge(nodes[0], nodes[2])
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(nodes[1], nodes[3])
        edge_id = graph.find_edge(nodes[1], nodes[3])
        graph.set_edge_weight(edge_id, 1.0)

        matching = graph.max_weight_matching(False)

        # Should match all 6 nodes into 3 pairs
        assert len(matching) == 6

        # Each node should be matched to exactly one other node
        for node in nodes:
            assert node in matching
            matched_node = matching[node]
            assert matching[matched_node] == node

    def test_matching_with_odd_nodes(self):
        """Test matching with odd number of nodes (one node unmatched)."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        # Only one edge
        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 10.0)

        matching = graph.max_weight_matching(False)

        # Only n0 and n1 should be matched
        assert len(matching) == 2
        assert matching[n0] == n1
        assert matching[n1] == n0
        assert n2 not in matching

    def test_matching_empty_graph(self):
        """Test matching on an empty graph."""
        graph = pc.graph.Graph()
        matching = graph.max_weight_matching(False)

        assert len(matching) == 0

    def test_matching_nodes_no_edges(self):
        """Test matching on graph with nodes but no edges."""
        graph = pc.graph.Graph()
        graph.add_node()
        graph.add_node()
        graph.add_node()

        matching = graph.max_weight_matching(False)

        # No edges means no matching
        assert len(matching) == 0

    def test_matching_max_cardinality_false(self):
        """Test matching with max_cardinality=False (default)."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()
        n3 = graph.add_node()

        # Two heavy edges and two light edges
        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 100.0)

        graph.add_edge(n2, n3)
        edge_id = graph.find_edge(n2, n3)
        graph.set_edge_weight(edge_id, 100.0)

        graph.add_edge(n0, n2)
        edge_id = graph.find_edge(n0, n2)
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(n1, n3)
        edge_id = graph.find_edge(n1, n3)
        graph.set_edge_weight(edge_id, 1.0)

        matching = graph.max_weight_matching(False)

        # Should prefer the heavy edges
        assert len(matching) == 4
        assert matching[n0] == n1
        assert matching[n2] == n3

    def test_matching_deterministic(self):
        """Test that matching is deterministic (uses BTreeMap)."""
        # Run the same matching multiple times and verify results are identical
        results = []
        for _ in range(5):
            graph = pc.graph.Graph()
            n0 = graph.add_node()
            n1 = graph.add_node()
            n2 = graph.add_node()
            n3 = graph.add_node()

            graph.add_edge(n0, n1)
            edge_id = graph.find_edge(n0, n1)
            graph.set_edge_weight(edge_id, 10.0)

            graph.add_edge(n2, n3)
            edge_id = graph.find_edge(n2, n3)
            graph.set_edge_weight(edge_id, 20.0)

            matching = graph.max_weight_matching(False)
            results.append(matching)

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]


class TestGraphUseCases:
    """Test graph usage for MWPM decoder scenarios."""

    def test_mwpm_decoder_scenario(self):
        """Test a typical MWPM decoder scenario.

        In quantum error correction, detection events (syndrome measurements)
        are matched in pairs. The algorithm maximizes total weight.

        Note: In practice, MWPM decoders may use inverted distances (1/distance)
        or log-likelihood ratios as weights to ensure higher weights for better matches.
        """
        graph = pc.graph.Graph()

        # Create 4 detection events
        d0 = graph.add_node()
        d1 = graph.add_node()
        d2 = graph.add_node()
        d3 = graph.add_node()

        # Add edges with weights inversely proportional to distance
        # High weight = close together = good match
        graph.add_edge(d0, d1)  # Close together, high weight
        edge_id = graph.find_edge(d0, d1)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(d2, d3)  # Close together, high weight
        edge_id = graph.find_edge(d2, d3)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(d0, d2)  # Far apart, low weight
        edge_id = graph.find_edge(d0, d2)
        graph.set_edge_weight(edge_id, 2.0)

        graph.add_edge(d1, d3)  # Far apart, low weight
        edge_id = graph.find_edge(d1, d3)
        graph.set_edge_weight(edge_id, 2.0)

        matching = graph.max_weight_matching(False)

        # Should match d0-d1 and d2-d3 (highest total weight)
        assert len(matching) == 4
        assert matching[d0] == d1
        assert matching[d2] == d3

    def test_empty_matching_use_case(self):
        """Test when no detection events occur (empty graph)."""
        graph = pc.graph.Graph()
        matching = graph.max_weight_matching(False)

        # Empty matching is valid (no errors detected)
        assert len(matching) == 0


class TestEdgeData:
    """Test edge data/attributes functionality."""

    def test_get_edge_data_simple(self):
        """Test retrieving edge data for a simple edge."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 5.5)

        # Get edge data
        data = graph.get_edge_data(n0, n1)
        assert data is not None
        assert data["weight"] == 5.5

    def test_edge_endpoints(self):
        """Test getting edge endpoints from edge ID."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)

        # Get endpoints from edge ID
        endpoints = graph.edge_endpoints(edge_id)
        assert endpoints is not None
        a, b = endpoints
        assert (a, b) == (n0, n1)

    def test_edge_endpoints_nonexistent(self):
        """Test edge_endpoints with invalid edge ID."""
        graph = pc.graph.Graph()

        # Non-existent edge ID should return None
        endpoints = graph.edge_endpoints(9999)
        assert endpoints is None

    def test_add_edge_weight_kwarg(self):
        """Test add_edge with weight set via method."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        # Set weight using method-based API
        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 7.5)

        data = graph.get_edge_data(n0, n1)
        assert data is not None
        assert data["weight"] == 7.5

    def test_get_edge_data_nonexistent(self):
        """Test getting edge data for non-existent edge."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        # No edge added
        data = graph.get_edge_data(n0, n1)
        assert data is None

    def test_get_edge_data_undirected(self):
        """Test that edge data works in both directions (undirected graph)."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 10.0)

        # Should work in both directions
        data1 = graph.get_edge_data(n0, n1)
        data2 = graph.get_edge_data(n1, n0)

        assert data1 is not None
        assert data2 is not None
        assert data1["weight"] == 10.0
        assert data2["weight"] == 10.0


class TestSubgraph:
    """Test subgraph extraction functionality."""

    def test_subgraph_simple(self):
        """Test creating a simple subgraph."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()
        n3 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(n1, n2)
        edge_id = graph.find_edge(n1, n2)
        graph.set_edge_weight(edge_id, 2.0)

        graph.add_edge(n2, n3)
        edge_id = graph.find_edge(n2, n3)
        graph.set_edge_weight(edge_id, 3.0)

        # Create subgraph with just n0 and n1
        sub = graph.subgraph([n0, n1])

        assert sub.node_count() == 2
        assert sub.edge_count() == 1

        # Edges in subgraph should maintain weights
        edges = sub.edges()
        assert len(edges) == 1
        assert edges[0][2] == 1.0  # weight

    def test_subgraph_disconnected(self):
        """Test subgraph with disconnected nodes."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()
        n3 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(n2, n3)
        edge_id = graph.find_edge(n2, n3)
        graph.set_edge_weight(edge_id, 20.0)

        # Create subgraph with n0 and n2 (not connected)
        sub = graph.subgraph([n0, n2])

        assert sub.node_count() == 2
        assert sub.edge_count() == 0  # No edge between n0 and n2

    def test_subgraph_empty(self):
        """Test creating an empty subgraph."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 5.0)

        # Empty subgraph
        sub = graph.subgraph([])

        assert sub.node_count() == 0
        assert sub.edge_count() == 0


class TestShortestPath:
    """Test shortest path functionality."""

    def test_single_source_shortest_path_simple(self):
        """Test shortest paths in a simple graph."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(n1, n2)
        edge_id = graph.find_edge(n1, n2)
        graph.set_edge_weight(edge_id, 1.0)

        paths = graph.single_source_shortest_path(n0)

        assert len(paths) == 3
        assert paths[n0] == [n0]
        assert paths[n1] == [n0, n1]
        assert paths[n2] == [n0, n1, n2]

    def test_single_source_shortest_path_disconnected(self):
        """Test shortest paths with disconnected components."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()
        n3 = graph.add_node()

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 1.0)

        graph.add_edge(n2, n3)
        edge_id = graph.find_edge(n2, n3)
        graph.set_edge_weight(edge_id, 1.0)

        # From n0, can only reach n0 and n1
        paths = graph.single_source_shortest_path(n0)

        assert len(paths) == 2
        assert n0 in paths
        assert n1 in paths
        assert n2 not in paths
        assert n3 not in paths

    def test_single_source_shortest_path_weighted(self):
        """Test that shortest path considers weights."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        n2 = graph.add_node()

        # Direct path n0->n2 has weight 10
        # Path via n1 has weight 2+3=5 (shorter)
        graph.add_edge(n0, n2)
        edge_id = graph.find_edge(n0, n2)
        graph.set_edge_weight(edge_id, 10.0)

        graph.add_edge(n0, n1)
        edge_id = graph.find_edge(n0, n1)
        graph.set_edge_weight(edge_id, 2.0)

        graph.add_edge(n1, n2)
        edge_id = graph.find_edge(n1, n2)
        graph.set_edge_weight(edge_id, 3.0)

        paths = graph.single_source_shortest_path(n0)

        # Should take the shorter path through n1
        assert paths[n2] == [n0, n1, n2]


class TestAttrsBuilder:
    """Test mutable attribute views and dict-like access."""

    def test_edge_attrs_view_chainable_insert(self):
        """Test EdgeAttrsView chainable insert method."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        graph.add_edge(n0, n1)

        # Test chainable insert
        attrs = graph.edge_attrs(n0, n1)
        attrs.insert("weight", 5.0).insert("label", "boundary").insert(
            "path", [1, 2, 3]
        )

        # Verify all values were set
        assert attrs["weight"] == 5.0
        assert attrs["label"] == "boundary"
        assert attrs["path"] == [1, 2, 3]

    def test_edge_attrs_view_mixed_access(self):
        """Test mixing dict-like and chainable access."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        graph.add_edge(n0, n1)

        # Mix dict-like and chainable style
        attrs = graph.edge_attrs(n0, n1)
        attrs["x"] = 1.0
        attrs.insert("y", 2.0).insert("z", 3.0)
        attrs["w"] = 4.0

        # Verify all values
        assert attrs["x"] == 1.0
        assert attrs["y"] == 2.0
        assert attrs["z"] == 3.0
        assert attrs["w"] == 4.0

    def test_edge_attrs_view_update_from_dict(self):
        """Test EdgeAttrsView.update() with a dict."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        graph.add_edge(n0, n1)

        # Update from dict
        attrs = graph.edge_attrs(n0, n1)
        attrs.update({"weight": 5.0, "label": "boundary", "path": [1, 2, 3]})

        # Verify all values were set
        assert attrs["weight"] == 5.0
        assert attrs["label"] == "boundary"
        assert attrs["path"] == [1, 2, 3]

    def test_edge_attrs_view_update_multiple_times(self):
        """Test multiple updates to EdgeAttrsView."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()
        n1 = graph.add_node()
        graph.add_edge(n0, n1)

        # First update
        attrs = graph.edge_attrs(n0, n1)
        attrs.update({"a": 1, "b": 2})

        # Second update (should merge/overwrite)
        attrs.update({"b": 20, "c": 3})

        # Verify
        assert attrs["a"] == 1
        assert attrs["b"] == 20  # overwritten
        assert attrs["c"] == 3

    def test_node_attrs_view_dict_like(self):
        """Test NodeAttrsView dict-like interface."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()

        # Test dict-like setting and getting
        attrs = graph.node_attrs(n0)
        attrs["label"] = "qubit"
        attrs["position"] = [1.0, 2.0, 3.0]
        attrs["active"] = True

        assert attrs["label"] == "qubit"
        assert attrs["position"] == [1.0, 2.0, 3.0]
        assert attrs["active"] is True

    def test_node_attrs_view_insert(self):
        """Test NodeAttrsView.insert() chainable method."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()

        # Test chainable insert
        attrs = graph.node_attrs(n0)
        attrs.insert("x", 1.0).insert("y", 2.0).insert("z", 3.0)

        assert attrs["x"] == 1.0
        assert attrs["y"] == 2.0
        assert attrs["z"] == 3.0

    def test_node_attrs_view_update(self):
        """Test NodeAttrsView.update() with a dict."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()

        # Update from dict
        attrs = graph.node_attrs(n0)
        attrs.update({"label": "qubit", "index": 5, "coords": [1.0, 2.0]})

        # Verify all values were set
        assert attrs["label"] == "qubit"
        assert attrs["index"] == 5
        assert attrs["coords"] == [1.0, 2.0]

    def test_node_attrs_view_get(self):
        """Test NodeAttrsView.get() with default values."""
        graph = pc.graph.Graph()
        n0 = graph.add_node()

        attrs = graph.node_attrs(n0)
        attrs["existing"] = "value"

        # Test get with existing key
        assert attrs.get("existing") == "value"

        # Test get with non-existing key (default None)
        assert attrs.get("nonexistent") is None

        # Test get with custom default
        assert attrs.get("nonexistent", "default") == "default"

    def test_graph_attrs_view_dict_like(self):
        """Test GraphAttrsView dict-like interface."""
        graph = pc.graph.Graph()

        # Test dict-like setting and getting
        attrs = graph.attrs()
        attrs["name"] = "test_graph"
        attrs["version"] = 1
        attrs["metadata"] = ["tag1", "tag2"]

        assert attrs["name"] == "test_graph"
        assert attrs["version"] == 1
        assert attrs["metadata"] == ["tag1", "tag2"]

    def test_graph_attrs_view_insert(self):
        """Test GraphAttrsView.insert() chainable method."""
        graph = pc.graph.Graph()

        # Test chainable insert
        attrs = graph.attrs()
        attrs.insert("author", "Alice").insert("date", "2025-01-01").insert(
            "version", 2
        )

        assert attrs["author"] == "Alice"
        assert attrs["date"] == "2025-01-01"
        assert attrs["version"] == 2

    def test_graph_attrs_view_update(self):
        """Test GraphAttrsView.update() with a dict."""
        graph = pc.graph.Graph()

        # Update from dict
        attrs = graph.attrs()
        attrs.update({"name": "my_graph", "size": 100, "tags": ["important"]})

        # Verify all values were set
        assert attrs["name"] == "my_graph"
        assert attrs["size"] == 100
        assert attrs["tags"] == ["important"]

    def test_graph_attrs_view_get(self):
        """Test GraphAttrsView.get() with default values."""
        graph = pc.graph.Graph()

        attrs = graph.attrs()
        attrs["existing"] = "value"

        # Test get with existing key
        assert attrs.get("existing") == "value"

        # Test get with non-existing key (default None)
        assert attrs.get("nonexistent") is None

        # Test get with custom default
        assert attrs.get("nonexistent", "default") == "default"

    def test_all_three_attr_levels(self):
        """Test that graph, node, and edge attributes all work together."""
        graph = pc.graph.Graph()

        # Set graph-level attributes
        graph.attrs()["name"] = "test"
        graph.attrs()["version"] = 1

        # Create nodes with attributes
        n0 = graph.add_node()
        n1 = graph.add_node()
        graph.node_attrs(n0)["label"] = "qubit_0"
        graph.node_attrs(n1)["label"] = "qubit_1"

        # Create edge with attributes
        graph.add_edge(n0, n1)
        graph.edge_attrs(n0, n1)["weight"] = 5.0
        graph.edge_attrs(n0, n1)["type"] = "coupling"

        # Verify all levels
        assert graph.attrs()["name"] == "test"
        assert graph.attrs()["version"] == 1
        assert graph.node_attrs(n0)["label"] == "qubit_0"
        assert graph.node_attrs(n1)["label"] == "qubit_1"
        assert graph.edge_attrs(n0, n1)["weight"] == 5.0
        assert graph.edge_attrs(n0, n1)["type"] == "coupling"
