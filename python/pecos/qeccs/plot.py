# Copyright 2019 The PECOS Developers
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
from matplotlib import pyplot as plt


# plot intsructions
def plot_qecc(
    qecc,
    figsize=(9, 9),
    dpi=80,
    filename=None,
    title_font_size=16,
    axis_font_size=14,
    legend_font_size=14,
    **kwargs,
):
    """Produces a plot of a qecc.

    Args:
    ----
        qecc(QECC): The ``qecc`` instance that is to be plotted.
        figsize(tuple of int): The size of the plotted figure.

    Returns:
    -------

    """
    if len(kwargs):
        raise Exception("keys %s not recognized!" % kwargs.keys())

    g = nx.DiGraph()

    mapping = qecc.mapping

    if mapping is None:
        mapping = NoMap()

    pos_old = qecc.layout
    pos = {mapping[q]: loc for q, loc in pos_old.items()}

    qudit_nodes_data = mapset(mapping, qecc.data_qudit_set)
    qudit_nodes_ancilla = mapset(mapping, qecc.ancilla_qudit_set)
    qudit_nodes_qudit = mapset(mapping, qecc.data_qudit_set)

    data_labels = {}
    for i in qudit_nodes_data:
        data_labels[i] = "$" + str(i) + "$"

    ancilla_labels = {}
    for i in qudit_nodes_ancilla:
        ancilla_labels[i] = "$" + str(i) + "$"

    g.add_nodes_from(qudit_nodes_qudit)
    plt.figure(num=None, figsize=figsize, dpi=dpi, edgecolor="k")
    plt.title("QECC layout: %s" % qecc.name, size=title_font_size)

    # Draw data qudits
    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        nodelist=qudit_nodes_data,
        node_color="white",
        node_shape="o",
        node_size=700,
        label="data qubit",
    )
    nodes.set_edgecolor("black")

    # Draw ancilla qudits
    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        nodelist=qudit_nodes_ancilla,
        node_color="black",
        node_shape="s",
        node_size=700,
        label="ancilla qubit",
    )
    nodes.set_edgecolor("black")

    # Label ancilla qudits
    nx.draw_networkx_labels(
        g,
        pos=pos,
        labels=ancilla_labels,
        font_size=16,
        font_color="white",
    )

    # Label data qudits
    nx.draw_networkx_labels(g, pos=pos, labels=data_labels, font_size=16)

    # Label nodes

    ax = plt.gca()
    ax.set_facecolor("lightgray")
    ax.set_xlabel("x (arbitrary length units)", size=axis_font_size)
    ax.set_ylabel("y (arbitrary length units)", size=axis_font_size)
    ax.invert_yaxis()

    plt.legend(
        labelspacing=2.5,
        borderpad=1.5,
        loc="upper left",
        bbox_to_anchor=(1, 1.01),
        fontsize=legend_font_size,
    )

    if filename:
        plt.savefig(filename)

    plt.show()


def plot_instr(
    instr,
    figsize=(9, 9),
    dpi=80,
    filename=None,
    title_font_size=16,
    axis_font_size=14,
    legend_font_size=14,
    **kwargs,
):
    """Args:
    ----
        instr(LogicalInstruction):

    Returns:
    -------

    """
    if len(kwargs):
        raise Exception("keys %s not recognized!" % kwargs.keys())

    g = nx.DiGraph()

    mapping = instr.qecc.mapping

    if mapping is None:
        mapping = NoMap()

    pos_old = instr.qecc.layout
    pos = {mapping[q]: loc for q, loc in pos_old.items()}

    qudit_nodes_data = mapset(mapping, instr.data_qudit_set)
    qudit_nodes_x = mapset(mapping, instr.ancilla_x_check)
    qudit_nodes_z = mapset(mapping, instr.ancilla_z_check)

    g.add_nodes_from(qudit_nodes_data)
    g.add_nodes_from(qudit_nodes_x)
    g.add_nodes_from(qudit_nodes_z)

    edge_labels, _, _ = graph_add_directed_cnots(instr, g)

    labels = {}
    for i in qudit_nodes_data:
        labels[i] = "$" + str(i) + "$"

    for i in qudit_nodes_x:
        labels[i] = "$" + str(i) + "$"

    for i in qudit_nodes_z:
        labels[i] = "$" + str(i) + "$"

    plt.figure(num=None, figsize=figsize, dpi=dpi, edgecolor="k")
    plt.title(
        f"Logical Instruction: '{instr.symbol}'  QECC: {instr.qecc.name}",
        size=title_font_size,
    )

    nx.draw_networkx_edges(g, pos=pos, arrowsize=30)
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels)

    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        nodelist=qudit_nodes_data,
        node_color="lightyellow",
        node_size=700,
        label="data qubit",
    )
    nodes.set_edgecolor("black")

    try:
        nodes = nx.draw_networkx_nodes(
            g,
            pos=pos,
            nodelist=qudit_nodes_x,
            node_color="lightcoral",
            node_shape="s",
            node_size=600,
            label="X ancilla",
        )

        nodes.set_edgecolor("black")
    except AttributeError:
        pass

    try:
        nodes = nx.draw_networkx_nodes(
            g,
            pos=pos,
            nodelist=qudit_nodes_z,
            node_color="powderblue",
            node_shape="s",
            node_size=600,
            label="Z ancilla",
        )
        nodes.set_edgecolor("black")
    except AttributeError:
        pass

    nx.draw_networkx_labels(g, pos=pos, labels=labels, font_size=16)

    ax = plt.gca()
    ax.set_xlabel("x (arbitrary length units)", size=axis_font_size)
    ax.set_ylabel("y (arbitrary length units)", size=axis_font_size)
    ax.invert_yaxis()

    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)

    plt.legend(
        labelspacing=2.5,
        borderpad=1.5,
        loc="upper left",
        bbox_to_anchor=(1, 1.01),
        fontsize=legend_font_size,
    )

    if filename:
        plt.savefig(filename)

    plt.show()


def get_ancilla_types(instr):
    x_ancillas = set()
    z_ancillas = set()
    abs_circuit = instr.abstract_circuit

    for gate_symbol, _, params in abs_circuit.items():
        if gate_symbol == "X check":
            ancilla = params["ancillas"]
            x_ancillas.add(ancilla)
        elif gate_symbol == "Z check":
            ancilla = params["ancillas"]
            z_ancillas.add(ancilla)

    return x_ancillas, z_ancillas


def graph_add_directed_cnots(instr, g):
    circuit = instr.circuit
    edge_labels = {}
    cys = []
    czs = []

    for i in range(len(circuit)):
        for sym, qudits, _ in circuit.items(tick=i):
            if sym in {"CNOT", "CZ", "CY"}:
                g.add_edges_from(qudits)
                for e in qudits:
                    edge_labels[e] = str(i)

                if sym == "CZ":
                    czs.append(qudits)
                elif sym == "CY":
                    cys.append(qudits)

    return edge_labels, czs, cys


def mapset(mapping, oldset):
    """Applies a mapping to a set.

    Args:
    ----
        mapping:
        oldset (set):

    Returns:
    -------

    """
    newset = set()

    for e in oldset:
        newset.add(mapping[e])

    return newset


class NoMap:
    """Default Mapping: item -> item."""

    def __init__(self) -> None:
        pass

    def __getitem__(self, item):
        return item
