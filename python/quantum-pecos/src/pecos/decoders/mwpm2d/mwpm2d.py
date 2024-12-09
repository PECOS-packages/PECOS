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

import networkx as nx

from pecos.circuits import QuantumCircuit
from pecos.decoders.mwpm2d import precomputing


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

    def __init__(self, qecc) -> None:
        instr = qecc.instruction("instr_syn_extract")

        self.instr = instr

        self.recorded_recovery = {}  # previous: syndrome => recovery

        precomputed_data = precomputing.precompute(instr)

        self.precomputed_data = precomputed_data

    def decode(self, measurements, error_params=None):
        """Takes measurement results and outputs a result.

        logic_range identifies over what part of self.logic we are decoding over.
        """
        syndromes = set(measurements.simplified(True))

        tuple_key = frozenset(syndromes)

        if tuple_key in self.recorded_recovery:
            return self.recorded_recovery[tuple_key]

        else:
            recovery = QuantumCircuit(1)

            decode_data = self.precomputed_data

            correction_x = []
            correction_z = []

            # Decode 'X' and Z separately.
            for check_type in ["X", "Z"]:
                correction = correction_z if check_type == "X" else correction_x

                check_type_decode = decode_data[check_type]

                distance_graph = check_type_decode["dist_graph"]
                virtual_edge_data = check_type_decode["virtual_edge_data"]

                active_syn = set(syndromes)
                # Get the real graph
                real_graph = nx.Graph(distance_graph.subgraph(active_syn))
                active_syn = set(real_graph.nodes())

                # Add virtual nodes
                new_name = self.itr_v_name()
                active_virt = set()
                for s in active_syn:
                    edge_data = virtual_edge_data[s]
                    v_name = next(new_name)
                    active_virt.add(v_name)
                    real_graph.add_edge(s, v_name, **edge_data)

                # Add edges between virtual nodes to allow pairing of un-needed virtual nodes
                for vi in active_virt:
                    for vj in active_virt:
                        if vi != vj:
                            real_graph.add_edge(vi, vj, weight=0)

                # Find a matching
                matching_edges = nx.max_weight_matching(real_graph, maxcardinality=True)
                matching = {n1: n2 for n2, n1 in matching_edges}
                matching.update(dict(matching_edges))

                nodes_paired = set()
                ## for n1 in real_graph.nodes():
                real_syn = set(real_graph.nodes())
                for n1 in syndromes & real_syn:
                    n2 = matching[n1]

                    # Don't continue if node has already been covered or path starts and ends with virtuals.
                    if n1 in nodes_paired or (
                        str(n1).startswith("v") and str(n2).startswith("v")
                    ):
                        continue

                    nodes_paired.add(n2)

                    path_attr = real_graph.get_edge_data(n1, n2)
                    correction.extend(path_attr["data_path"])

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
    def itr_v_name():
        i = 0

        while True:
            i += 1
            yield "vu" + str(i)
