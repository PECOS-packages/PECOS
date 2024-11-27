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

import networkx as nx


def precompute(instr):
    """Args:
    ----
        instr:

    Returns:
    -------

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


def code_surface4444(instr):
    """Pre-computing for surface4444 class.

    This decoder is for 2D slices. It is assumed that it can decode logical instruction by logical instruction.

    :param logical_gate:
    :param precomputed_data: A dictionary of precomputed data used to decode syndromes of logical instructions.
    :return:
    """
    if instr.symbol == "instr_syn_extract":
        # In the future go through different instructions
        decoder_data = surface4444_identity(instr)

    else:
        msg = 'Can currently only handle "instr_init_zero".'
        raise Exception(msg)

    return decoder_data


def code_surface4444medial(instr):
    """Pre-computing for surface4444 class.

    This decoder is for 2D slices. It is assumed that it can decode logical instruction by logical instruction.

    :param logical_gate:
    :param precomputed_data: A dictionary of precomputed data used to decode syndromes of logical instructions.
    :return:
    """
    if instr.symbol == "instr_syn_extract":
        # In the future go through different instructions
        decoder_data = surface4444medial_identity(instr)

    else:
        msg = 'Can currently only handle "instr_init_zero".'
        raise Exception(msg)

    return decoder_data


def surface4444_identity(instr):
    """For X and Z decoding separately:

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
            "dist_graph": nx.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_x,
        },
        "Z": {
            "dist_graph": nx.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_z,
        },
    }

    # Record what data qudits the syndrome to syndrome edges correspond to.
    edges_x = {}
    edges_z = {}

    # syndrome-to-syndrome, fully-connected graph
    graph_x = info["X"]["dist_graph"]
    graph_z = info["Z"]["dist_graph"]

    # The closest virtual node to each syndrome
    closest_x = info["X"]["closest_virt"]
    closest_z = info["Z"]["closest_virt"]

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
    temp_graph_x = nx.Graph()
    temp_graph_z = nx.Graph()

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
                raise Exception(
                    "This decoder can only handle check of purely X or Z type rather than %s!"
                    % gate_symbol,
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
            if side_label in ["left", "right"]:  # 1 for i = 1 and 3 => left and right
                syn_list = d2edge_x.setdefault(data, [])
                syn_list.append(virt_node)
                virt_x.add(virt_node)

            # Z virtual nodes (sides top and bottom)
            elif side_label in ["top", "bottom"]:  # 0 for i = 0 and 2 => top and bottom
                syn_list = d2edge_z.setdefault(data, [])
                syn_list.append(virt_node)
                virt_z.add(virt_node)
            else:
                raise Exception('side_label "%s" not understood!' % side_label)

    # invert data -> edge and make sure len(edge) = 2
    for check_type in ["X", "Z"]:
        if check_type == "X":
            edge_dict = d2edge_x
            edges = edges_x
            temp_graph = temp_graph_x
        else:
            edge_dict = d2edge_z
            edges = edges_z
            temp_graph = temp_graph_z

        for data, edge in edge_dict.items():
            if len(edge) != 2:
                msg = (
                    f"There should be exactly two syndromes (virtual or not) connected to each data qudit. Instead,"
                    f" q: {data} edge: {edge}"
                )
                raise Exception(msg)

            edges[tuple(edge)] = data
            edges[(edge[1], edge[0])] = data
            temp_graph.add_edge(edge[0], edge[1])

    # Create distance graph
    for check_type in ["X", "Z"]:
        if check_type == "X":
            temp_graph = temp_graph_x
            g = graph_x
            closest = closest_x
            virt = virt_x
            edge2d = edges_x
            virtual_edge_data = virtual_edge_data_x

        else:
            temp_graph = temp_graph_z
            g = graph_z
            closest = closest_z
            virt = virt_z
            edge2d = edges_z
            virtual_edge_data = virtual_edge_data_z

        paths = dict(nx.shortest_path(temp_graph))

        for n1, wdict in paths.items():
            for n2, syn_path in wdict.items():
                weight = len(syn_path) - 1

                if weight != 0:
                    # Get list of datas corresponding to the connected path between syndromes
                    data_path = []
                    s1 = syn_path[0]
                    for s2 in syn_path[1:]:
                        data = edge2d[(s1, s2)]
                        data_path.append(data)
                        s1 = s2

                    if (n1 not in virt) and (n2 not in virt):
                        g.add_edge(
                            n1,
                            n2,
                            weight=-weight,
                            syn_path=syn_path,
                            data_path=data_path,
                        )

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
                data = edge2d[(s1, s2)]
                data_path.append(data)
                s1 = s2

            virtual_edge_data[s] = {
                "virtual_node": v,
                "weight": -weight,
                "syn_path": syn_path,
                "data_path": data_path,
            }

    return info


def surface4444medial_identity(instr):
    """For X and Z decoding separately:

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
            "dist_graph": nx.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_x,
        },
        "Z": {
            "dist_graph": nx.Graph(),
            "closest_virt": {},
            "virtual_edge_data": virtual_edge_data_z,
        },
    }

    # Record what data qudits the syndrome to syndrome edges correspond to.
    edges_x = {}
    edges_z = {}

    # syndrome-to-syndrome, fully-connected graph
    graph_x = info["X"]["dist_graph"]
    graph_z = info["Z"]["dist_graph"]

    # The closest virtual node to each syndrome
    closest_x = info["X"]["closest_virt"]
    closest_z = info["Z"]["closest_virt"]

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
    temp_graph_x = nx.Graph()
    temp_graph_z = nx.Graph()

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
                raise Exception(
                    "This decoder can only handle check of purely X or Z type rather than %s!"
                    % gate_symbol,
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
    for side_label, side_qubits in sides.items():
        for i, data in enumerate(side_qubits):
            if side_label == "top":
                if distance_width % 2 == 1:  # odd
                    if i == 0 or i % 2 == 1:
                        vi += 1
                        virt_node = "v" + str(vi)

                else:  # even
                    if i % 2 == 0:
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
                else:
                    if i % 2 == 0:
                        vi += 1
                        virt_node = "v" + str(vi)

                syn_list = d2edge_x.setdefault(data, [])
                syn_list.append(virt_node)
                virt_x.add(virt_node)
            else:
                raise Exception('side_label "%s" not understood!' % side_label)

    # invert data -> edge and make sure len(edge) = 2
    for check_type in ["X", "Z"]:
        if check_type == "X":
            edge_dict = d2edge_x
            edges = edges_x
            temp_graph = temp_graph_x
        else:
            edge_dict = d2edge_z
            edges = edges_z
            temp_graph = temp_graph_z

        for data, edge in edge_dict.items():
            if len(edge) != 2:
                msg = (
                    f"There should be exactly two syndromes (virtual or not) connected to each data qudit. Instead,"
                    f" q: {data} edge: {edge}"
                )
                raise Exception(msg)

            edges[tuple(edge)] = data
            edges[(edge[1], edge[0])] = data
            temp_graph.add_edge(edge[0], edge[1])

    # Create distance graph
    for check_type in ["X", "Z"]:
        if check_type == "X":
            temp_graph = temp_graph_x
            g = graph_x
            closest = closest_x
            virt = virt_x
            edge2d = edges_x
            virtual_edge_data = virtual_edge_data_x

        else:
            temp_graph = temp_graph_z
            g = graph_z
            closest = closest_z
            virt = virt_z
            edge2d = edges_z
            virtual_edge_data = virtual_edge_data_z

        paths = dict(nx.shortest_path(temp_graph))

        for n1, wdict in paths.items():
            for n2, syn_path in wdict.items():
                weight = len(syn_path) - 1

                if weight != 0:
                    # Get list of datas corresponding to the connected path between syndromes
                    data_path = []
                    s1 = syn_path[0]
                    for s2 in syn_path[1:]:
                        data = edge2d[(s1, s2)]
                        data_path.append(data)
                        s1 = s2

                    if (n1 not in virt) and (n2 not in virt):
                        g.add_edge(
                            n1,
                            n2,
                            weight=-weight,
                            syn_path=syn_path,
                            data_path=data_path,
                        )

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
                data = edge2d[(s1, s2)]
                data_path.append(data)
                s1 = s2

            virtual_edge_data[s] = {
                "virtual_node": v,
                "weight": -weight,
                "syn_path": syn_path,
                "data_path": data_path,
            }

    return info
