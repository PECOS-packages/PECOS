# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Minimum-weight Perfect-matching decoder for 2D surface codes.

This module implements a 2D minimum-weight perfect-matching (MWPM) decoder for
surface code quantum error correction, designed for code capacity modeling
and ideal decoding scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pecos.circuits import QuantumCircuit
from pecos.decoders.mwpm2d import precomputing
from pecos.graph import Graph

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pecos.engines.std_output import StdOutput
    from pecos.protocols import QECCProtocol


class MWPM2D:
    """2D minimum weight perfect matching for surface capacity assuming code capacity. (Only data error.).

    A simple Minimum Weight Perfect Matching decoder. It is for 2D decoding either for code capacity modeling or ideal
    decoding.

    For code capacity, data errors are sprinkled before each logical gate. Then the decoder takes in syndrome
    measurements to come up with a recovery operation.

    """

    # Basic subpackage required attributes
    output = None
    input = None

    def __init__(self, qecc: QECCProtocol) -> None:
        """Initialize the MWPM2D decoder.

        Args:
        ----
            qecc: The quantum error correcting code protocol to use for decoding.
        """
        instr = qecc.instruction("instr_syn_extract")

        self.instr = instr

        self.recorded_recovery = {}  # previous: syndrome => recovery

        precomputed_data = precomputing.precompute(instr)

        self.precomputed_data = precomputed_data

        # Real check labels only: virtual boundary nodes have no virtual-edge data.
        self.syndrome_maps = {
            check_type: {label: node for label, node in data["node_map"].items() if node in data["virtual_edge_data"]}
            for check_type, data in precomputed_data.items()
        }
        self.syndrome_labels = self.syndrome_maps["X"].keys() | self.syndrome_maps["Z"].keys()

    def decode(
        self,
        measurements: StdOutput,
        _error_params: dict[str, Any] | None = None,
    ) -> QuantumCircuit:
        """Takes measurement results and outputs a result.

        logic_range identifies over what part of self.logic we are decoding over.

        Raises:
            ValueError: If a syndrome label is not a real check known to the precomputed data.
        """
        syndromes = set(measurements.simplified(last=True))

        decode_data = self.precomputed_data
        unknown = syndromes - self.syndrome_labels
        if unknown:
            msg = f"Unknown syndrome labels: {sorted(unknown, key=str)!r}"
            raise ValueError(msg)

        tuple_key = frozenset(syndromes)

        if tuple_key in self.recorded_recovery:
            return self.recorded_recovery[tuple_key]

        recovery = QuantumCircuit(1)

        correction_x = []
        correction_z = []

        # Decode 'X' and Z separately.
        for check_type in ["X", "Z"]:
            correction = correction_z if check_type == "X" else correction_x

            check_type_decode = decode_data[check_type]

            distance_graph = check_type_decode["dist_graph"]
            virtual_edge_data = check_type_decode["virtual_edge_data"]

            node_map = self.syndrome_maps[check_type]
            active_syn = {node_map[label] for label in syndromes if label in node_map}

            # Build a new graph instead of using subgraph (which renumbers nodes)
            # We need to keep the original node IDs from distance_graph
            real_graph = Graph()

            # First, ensure we have all the syndrome nodes with their original IDs
            # by adding nodes until we reach the highest ID we need
            max_syndrome = max(active_syn) if active_syn else 0
            for _ in range(max_syndrome + 1):
                real_graph.add_node()

            # Add edges between active syndrome nodes from the distance graph
            for n1 in active_syn:
                for n2 in active_syn:
                    if n1 < n2:  # Only check each pair once
                        # Check if edge exists in distance graph
                        weight = distance_graph.get_weight(n1, n2)
                        if weight is not None:
                            # Copy edge to real_graph with all attributes
                            real_graph.add_edge(n1, n2)
                            real_graph.set_weight(n1, n2, weight)

                            # Copy all other attributes
                            edge_data = distance_graph.get_edge_data(n1, n2)
                            if edge_data:
                                edge_attrs = real_graph.edge_attrs(n1, n2)
                                for key, value in edge_data.items():
                                    if key != "weight":  # weight already set
                                        edge_attrs[key] = value

            # Add virtual nodes
            new_name = self.itr_v_name()
            active_virt = set()
            for s in active_syn:
                edge_data = virtual_edge_data[s]
                next(new_name)
                # Create the virtual node and optionally store the name as an attribute
                v_id = real_graph.add_node()
                # Store name as attribute if needed for debugging
                # real_graph.node_attrs(v_id)['name'] = v_name
                active_virt.add(v_id)
                # Add edge with attributes from precomputed edge_data
                real_graph.add_edge(s, v_id)
                edge_attrs = real_graph.edge_attrs(s, v_id)
                for key, value in edge_data.items():
                    if key == "weight":
                        real_graph.set_weight(s, v_id, value)
                    else:
                        edge_attrs[key] = value

            # Add edges between virtual nodes to allow pairing of un-needed virtual nodes
            for vi in active_virt:
                for vj in active_virt:
                    if vi != vj:
                        real_graph.add_edge(vi, vj)
                        real_graph.set_weight(vi, vj, 0.0)

            # Find a matching using pecos.graph
            matching = real_graph.max_weight_matching(max_cardinality=True)
            matching_edges = list(matching.items())

            matching = {n1: n2 for n2, n1 in matching_edges}
            matching.update(dict(matching_edges))

            nodes_paired = set()
            # Matching and recovery paths use graph node IDs, not syndrome labels.
            for n1 in active_syn:
                n2 = matching[n1]

                # n1 is always a real syndrome, so only skip pairs already covered from the other end.
                if n1 in nodes_paired:
                    continue

                nodes_paired.add(n2)

                # Get data_path attribute from the matched edge
                edge_attrs = real_graph.edge_attrs(n1, n2)
                correction.extend(edge_attrs["data_path"])

        correction_x = set(correction_x)
        correction_z = set(correction_z)

        correction_y = correction_x & correction_z
        correction_x -= correction_y
        correction_z -= correction_y

        if correction_z:
            recovery.update({"Z": correction_z})

        if correction_x:
            recovery.update({"X": correction_x})

        if correction_y:
            recovery.update({"Y": correction_y})

        self.recorded_recovery[tuple_key] = recovery

        return recovery

    @staticmethod
    def itr_v_name() -> Iterator[str]:
        """Generate unique vertex names for matching graph.

        Yields:
            Unique vertex name strings in the format 'vu1', 'vu2', etc.
        """
        i = 0

        while True:
            i += 1
            yield "vu" + str(i)
