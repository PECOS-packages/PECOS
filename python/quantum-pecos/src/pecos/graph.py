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

"""Graph algorithms for PECOS.

This module provides graph data structures and algorithms, including
minimum-weight perfect matching (MWPM) for quantum error correction decoders.

Graph Types
-----------
Graph : Undirected graph
    For general undirected graph algorithms and MWPM decoder.

DiGraph : Directed graph
    For directed graphs that may contain cycles.

DAG : Directed Acyclic Graph
    For directed graphs with enforced acyclicity (useful for circuit DAGs,
    dependency graphs). Raises DagWouldCycleError if adding an edge would
    create a cycle.

Features
--------
- Node and edge management with attributes
- max_weight_matching() for MWPM decoder (Graph)
- Dijkstra's shortest path algorithms (Graph)
- Topological sort, predecessors/successors (DiGraph, DAG)
- Longest path, ancestors/descendants, roots/leaves (DAG)

Attribute view classes provide dict-like access to graph/node/edge attributes.
"""

# DAG exceptions are in the pecos_rslib root module
from pecos_rslib import DagHasCycleError, DagWouldCycleError
from pecos_rslib.graph import (
    # Directed Acyclic Graph
    DAG,
    DagEdgeAttrsView,
    DagGraphAttrsView,
    DagNodeAttrsView,
    # Directed Graph
    DiGraph,
    DiGraphAttrsView,
    DiGraphEdgeAttrsView,
    DiGraphNodeAttrsView,
    # Undirected Graph
    EdgeAttrsView,
    Graph,
    GraphAttrsView,
    NodeAttrsView,
)

__all__ = [  # noqa: RUF022
    # Undirected Graph
    "Graph",
    "EdgeAttrsView",
    "NodeAttrsView",
    "GraphAttrsView",
    # Directed Graph
    "DiGraph",
    "DiGraphEdgeAttrsView",
    "DiGraphNodeAttrsView",
    "DiGraphAttrsView",
    # Directed Acyclic Graph
    "DAG",
    "DagEdgeAttrsView",
    "DagNodeAttrsView",
    "DagGraphAttrsView",
    # DAG Exceptions
    "DagWouldCycleError",
    "DagHasCycleError",
]
