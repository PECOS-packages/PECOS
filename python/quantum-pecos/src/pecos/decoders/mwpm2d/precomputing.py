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

"""These functions build distance graphs for logical gates of qeccs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pecos

if TYPE_CHECKING:
    from pecos.protocols import LogicalInstructionProtocol
    from pecos.typing import GraphProtocol, Node, Path


def precompute(instr: LogicalInstructionProtocol) -> dict[str, Any]:
    """Precompute decoder information for the given instruction.

    Args:
    ----
        instr: The logical instruction to precompute decoder information for.
    """
    qecc = instr.qecc

    if (
        qecc.name == "4.4.4.4 Surface Code"
        and qecc.circuit_compiler.name == "Check2Circuits"
    ):
        precomputed_data = code_surface4444(instr)

    elif (
        qecc.name == "Medial 4.4.4.4 Surface Code"
        and qecc.circuit_compiler.name == "Check2Circuits"
    ):
        precomputed_data = code_surface4444medial(instr)

    else:
        msg = "Can only handle the non-medial surface code!"
        raise Exception(msg)

    return precomputed_data


def code_surface4444(instr: LogicalInstructionProtocol) -> dict[str, Any]:
    """Pre-computing for surface4444 class.

    This decoder is for 2D slices. It is assumed that it can decode logical instruction by logical instruction.

    Parameters:
        instr: Instruction
    """
    if instr.symbol == "instr_syn_extract":
        # In the future go through different instructions
        decoder_data = surface4444_identity(instr)

    else:
        msg = 'Can currently only handle "instr_init_zero".'
        raise Exception(msg)

    return decoder_data


def code_surface4444medial(instr: LogicalInstructionProtocol) -> dict[str, Any]:
    """Pre-computing for surface4444 class.

    This decoder is for 2D slices. It is assumed that it can decode logical instruction by logical instruction.

    Parameters:
        instr: Instruction
    """
    if instr.symbol == "instr_syn_extract":
        # In the future go through different instructions
        decoder_data = surface4444medial_identity(instr)

    else:
        msg = 'Can currently only handle "instr_init_zero".'
        raise Exception(msg)

    return decoder_data


def compute_all_shortest_paths(graph: GraphProtocol) -> dict[Node, dict[Node, Path]]:
    """Compute all shortest paths in a graph.

    This function will explicitly generate the all-pairs shortest paths.

    Args:
        graph: A graph object with nodes() and single_source_shortest_path() methods

    Returns:
        Dictionary of dictionaries with path[source][target] = list of nodes in path
    """
    # Compute all-pairs shortest paths explicitly
    all_paths = {}
    for source in graph.nodes():
        # For each source, get paths to all targets
        source_paths = graph.single_source_shortest_path(source)
        all_paths[source] = source_paths

    return all_paths


def surface4444_identity(instr: LogicalInstructionProtocol) -> dict[str, Any]:
    """Compute decoder information for Surface 4444 identity gate.

    For X and Z decoding separately:

    - Create dictionary:

    - Determine how virtual nodes connect to data qubits.
    - find syndromes(and virtual) edges -> data
    - Generate distance graph
    - Determine syn -> closest v and weight

    Parameters:
        instr: Instruction
    """
    # In the end... need:
    # For x and z separately:
    #
    # syn -> closest virtual node
    # edge -> data
    # distance graph

    qecc = instr.qecc

    virtual_edge_data_x = {}
    virtual_edge_data_z = {}

    # Create a dictionary to store precomputed information that will be used for decoding
    info = {
        "X": {
            "dist_graph": pecos.graph.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_x,
        },
        "Z": {
            "dist_graph": pecos.graph.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_z,
        },
    }

    # Record what data qudits the syndrome to syndrome edges correspond to.
    edges_x = {}
    edges_z = {}

    # The sides of the QECC patch
    sides = qecc.sides  # t, r, b, l

    # checks of the logical instruction
    instr = qecc.instruction("instr_syn_extract")
    abs_circ = instr.abstract_circuit

    # Dictionary of data qudit to syndrome-to-syndrome edge
    d2edge_x = {}
    d2edge_z = {}

    # Temporary graphs that will store the direct syndrome-to-syndrome edges. This will be used to create the fully
    # connected, distance graph.
    temp_graph_x = pecos.graph.Graph()
    temp_graph_z = pecos.graph.Graph()

    # Assume the QECC uses checks
    # add edges based on checks
    for gate_symbol, _, params in abs_circ.items():
        ancilla = params["ancillas"]
        data_qubits = params["datas"]

        for data in data_qubits:
            if gate_symbol == "X check":
                edges = d2edge_x
            elif gate_symbol == "Z check":
                edges = d2edge_z
            else:
                msg = f"This decoder can only handle check of purely X or Z type rather than {gate_symbol}!"
                raise Exception(
                    msg,
                )

            syn_list = edges.setdefault(data, [])
            syn_list.append(ancilla)

    # ---- Create virtual nodes ----- #

    virt_x = set()
    virt_z = set()

    # side: top, right, bottom, left
    # For the non-medial surface code patch, a x virtual node is paired with data qubits on the right and left sides
    # and a z virtual node is paired with data qubits on the top and bottom sides (should say what Pauli type the side
    # is.

    vi = 0

    for side_label, side_qubits in sides.items():
        for data in side_qubits:
            vi += 1

            virt_node = "v" + str(vi)

            # X virtual nodes (sides left and right)
            if side_label in {"left", "right"}:  # 1 for i = 1 and 3 => left and right
                syn_list = d2edge_x.setdefault(data, [])
                syn_list.append(virt_node)
                virt_x.add(virt_node)

            # Z virtual nodes (sides top and bottom)
            elif side_label in {"top", "bottom"}:  # 0 for i = 0 and 2 => top and bottom
                syn_list = d2edge_z.setdefault(data, [])
                syn_list.append(virt_node)
                virt_z.add(virt_node)
            else:
                msg = f'side_label "{side_label}" not understood!'
                raise Exception(msg)

    return invert_data(
        info,
        d2edge_x,
        d2edge_z,
        edges_x,
        edges_z,
        temp_graph_x,
        temp_graph_z,
        virt_x,
        virt_z,
    )


def surface4444medial_identity(instr: LogicalInstructionProtocol) -> dict[str, Any]:
    """Compute decoder information for Surface 4444 medial identity gate.

    For X and Z decoding separately:

    - Create dictionary:

    - Determine how virtual nodes connect to data qubits.
    - find syndromes(and virtual) edges -> data
    - Generate distance graph
    - Determine syn -> closest v and weight

    :param instr:
    :return:
    """
    # In the end... need:
    # For x and z separately:
    #
    # syn -> closest virtual node
    # edge -> data
    # distance graph

    qecc = instr.qecc

    virtual_edge_data_x = {}
    virtual_edge_data_z = {}

    # Create a dictionary to store precomputed information that will be used for decoding
    info = {
        "X": {
            "dist_graph": pecos.graph.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_x,
        },
        "Z": {
            "dist_graph": pecos.graph.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_z,
        },
    }

    # Record what data qudits the syndrome to syndrome edges correspond to.
    edges_x = {}
    edges_z = {}

    # The sides of the QECC patch
    sides = qecc.sides  # t, r, b, l

    # checks of the logical instruction
    instr = qecc.instruction("instr_syn_extract")
    abs_circ = instr.abstract_circuit

    # Dictionary of data qudit to syndrome-to-syndrome edge
    d2edge_x = {}
    d2edge_z = {}

    # Temporary graphs that will store the direct syndrome-to-syndrome edges. This will be used to create the fully
    # connected, distance graph.
    temp_graph_x = pecos.graph.Graph()
    temp_graph_z = pecos.graph.Graph()

    # Assume the QECC uses checks
    # add edges based on checks
    for gate_symbol, _, params in abs_circ.items():
        data_qudits = params["datas"]
        ancilla = params["ancillas"]

        for data in data_qudits:
            if gate_symbol == "X check":
                edges = d2edge_x
            elif gate_symbol == "Z check":
                edges = d2edge_z
            else:
                msg = f"This decoder can only handle check of purely X or Z type rather than {gate_symbol}!"
                raise Exception(
                    msg,
                )

            syn_list = edges.setdefault(data, [])
            syn_list.append(ancilla)

    # VIRTUAL NODES...
    # ---- Create virtual nodes ----- #
    virt_x = set()
    virt_z = set()

    # side: top, right, bottom, left
    # For the non-medial surface code patch, a x virtual node is paired with data qubits on the right and left sides
    # and a z virtual node is paired with data qubits on the top and bottom sides (should say what Pauli type the side
    # is.

    distance_width = qecc.width
    distance_height = qecc.height

    vi = 0
    virt_node = None
    for side_label, side_qubits in sides.items():
        for i, data in enumerate(side_qubits):
            if side_label == "top":
                if distance_width % 2 == 1:  # odd
                    if i == 0 or i % 2 == 1:
                        vi += 1
                        virt_node = "v" + str(vi)

                elif i % 2 == 0:
                    vi += 1
                    virt_node = "v" + str(vi)

                syn_list = d2edge_z.setdefault(data, [])
                syn_list.append(virt_node)
                virt_z.add(virt_node)

            elif side_label == "bottom":
                if i == 0 or i % 2 == 1:
                    vi += 1
                    virt_node = "v" + str(vi)

                syn_list = d2edge_z.setdefault(data, [])
                syn_list.append(virt_node)
                virt_z.add(virt_node)

            elif side_label == "left":
                if i == 0 or i % 2 == 1:
                    vi += 1
                    virt_node = "v" + str(vi)

                syn_list = d2edge_x.setdefault(data, [])
                syn_list.append(virt_node)
                virt_x.add(virt_node)

            elif side_label == "right":
                if distance_height % 2 == 1:
                    if i == 0 or i % 2 == 1:
                        vi += 1
                        virt_node = "v" + str(vi)
                elif i % 2 == 0:
                    vi += 1
                    virt_node = "v" + str(vi)

                syn_list = d2edge_x.setdefault(data, [])
                syn_list.append(virt_node)
                virt_x.add(virt_node)
            else:
                msg = f'side_label "{side_label}" not understood!'
                raise Exception(msg)

    return invert_data(
        info,
        d2edge_x,
        d2edge_z,
        edges_x,
        edges_z,
        temp_graph_x,
        temp_graph_z,
        virt_x,
        virt_z,
    )


def invert_data(
    info: dict[str, Any],
    d2edge_x: dict[Any, list[Any]],
    d2edge_z: dict[Any, list[Any]],
    edges_x: dict[tuple[Any, Any], Any],
    edges_z: dict[tuple[Any, Any], Any],
    temp_graph_x: GraphProtocol,
    temp_graph_z: GraphProtocol,
    virt_x: set[Any],
    virt_z: set[Any],
) -> dict[str, Any]:
    """Invert data-to-edge mappings and construct distance graphs.

    This function processes the data-to-edge mappings to create edge-to-data mappings,
    builds temporary graphs, and constructs fully connected distance graphs for both
    X and Z type checks. It also finds the closest virtual node for each syndrome.

    Args:
        info: Dictionary containing precomputed decoder information.
        d2edge_x: Data qubit to X-type edge mapping.
        d2edge_z: Data qubit to Z-type edge mapping.
        edges_x: Dictionary to store X-type edge to data mappings.
        edges_z: Dictionary to store Z-type edge to data mappings.
        temp_graph_x: Temporary graph for X-type connections.
        temp_graph_z: Temporary graph for Z-type connections.
        virt_x: Set of X-type virtual nodes.
        virt_z: Set of Z-type virtual nodes.

    Returns:
        The updated info dictionary with distance graphs and closest virtual nodes.
    """
    # invert data -> edge and make sure len(edge) = 2
    # Store node mappings for both X and Z
    node_map_x = {}
    node_map_z = {}

    for check_type in ["X", "Z"]:
        if check_type == "X":
            edge_dict = d2edge_x
            edges = edges_x
            temp_graph = temp_graph_x
            node_map = node_map_x
        else:
            edge_dict = d2edge_z
            edges = edges_z
            temp_graph = temp_graph_z
            node_map = node_map_z

        # Collect all unique node identifiers (both integers and strings)
        # and create nodes for them in the graph
        all_nodes = set()
        for edge in edge_dict.values():
            all_nodes.update(edge)

        # Create a mapping from original identifier to node ID
        for node_id in sorted(all_nodes, key=str):
            # Add node and create mapping
            idx = temp_graph.add_node()
            node_map[node_id] = idx
            # Optionally store original identifier as attribute for debugging
            # temp_graph.node_attrs(idx)['original_id'] = str(node_id)

        for data, edge in edge_dict.items():
            if len(edge) != 2:
                msg = (
                    f"There should be exactly two syndromes (virtual or not) connected to each data qudit. Instead,"
                    f" q: {data} edge: {edge}"
                )
                raise Exception(msg)

            # Store edges keyed by node IDs (integers)
            idx0 = node_map[edge[0]]
            idx1 = node_map[edge[1]]
            edges[(idx0, idx1)] = data
            edges[(idx1, idx0)] = data
            # Add edges using node IDs
            temp_graph.add_edge(idx0, idx1)

    # Convert virt_x and virt_z to use node IDs
    virt_x_ids = {node_map_x[v] for v in virt_x if v in node_map_x}
    virt_z_ids = {node_map_z[v] for v in virt_z if v in node_map_z}

    # Create distance graph
    for check_type in ["X", "Z"]:
        if check_type == "X":
            temp_graph = temp_graph_x
            g = info["X"]["dist_graph"]
            closest = info["X"]["closest_virt"]
            virt = virt_x_ids  # Use node IDs instead of original labels
            edge2d = edges_x
            virtual_edge_data = info["X"]["virtual_edge_data"]

        else:
            temp_graph = temp_graph_z
            g = info["Z"]["dist_graph"]
            closest = info["Z"]["closest_virt"]
            virt = virt_z_ids  # Use node IDs instead of original labels
            edge2d = edges_z
            virtual_edge_data = info["Z"]["virtual_edge_data"]

        # Use a future-proof approach to get all shortest paths
        paths = compute_all_shortest_paths(temp_graph)

        # Create nodes in the distance graph for all nodes in temp_graph
        # We need to ensure the node IDs in g match those in temp_graph
        for node_id in temp_graph.nodes():
            # Add nodes until we reach the required node_id
            while g.node_count() <= node_id:
                g.add_node()

        for n1, wdict in paths.items():
            for n2, syn_path in wdict.items():
                weight = len(syn_path) - 1

                if weight != 0:
                    # Get list of datas corresponding to the connected path between syndromes
                    data_path = []
                    s1 = syn_path[0]
                    for s2 in syn_path[1:]:
                        data = edge2d[s1, s2]
                        data_path.append(data)
                        s1 = s2

                    if (n1 not in virt) and (n2 not in virt):
                        g.add_edge(n1, n2)
                        g.set_weight(n1, n2, -weight)
                        edge_attrs = g.edge_attrs(n1, n2)
                        edge_attrs["syn_path"] = syn_path
                        edge_attrs["data_path"] = data_path

        syn = set(g.nodes())
        syn -= virt

        # Find closest virtual node

        for s in syn:
            shortest_len = float("inf")
            closest_v = None
            for v in virt:
                sv_len = len(paths[s][v])
                if sv_len < shortest_len:
                    shortest_len = sv_len
                    closest_v = v
            closest[s] = closest_v

        for s, v in closest.items():
            syn_path = paths[s][v]
            weight = len(syn_path) - 1

            data_path = []
            s1 = syn_path[0]
            for s2 in syn_path[1:]:
                data = edge2d[s1, s2]
                data_path.append(data)
                s1 = s2

            virtual_edge_data[s] = {
                "virtual_node": v,
                "weight": -weight,
                "syn_path": syn_path,
                "data_path": data_path,
            }

    return info
