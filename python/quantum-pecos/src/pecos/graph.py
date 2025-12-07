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

The Graph class provides:
- Node and edge management with attributes
- max_weight_matching() method for MWPM decoder
- Dijkstra's shortest path algorithms

Attribute view classes provide dict-like access to graph/node/edge attributes.
"""

from pecos_rslib.graph import EdgeAttrsView, Graph, GraphAttrsView, NodeAttrsView

__all__ = ["EdgeAttrsView", "Graph", "GraphAttrsView", "NodeAttrsView"]
