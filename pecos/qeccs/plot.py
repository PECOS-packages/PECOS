#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from matplotlib import pyplot as plt
import networkx as nx


# plot intsructions
def plot_qecc(qecc, figsize=(9, 9), dpi=80, filename=None, **kwargs):
    """Produces a plot of a qecc.

    Args:
        qecc(QECC): The ``qecc`` instance that is to be plotted.
        figsize(tuple of int): The size of the plotted figure.

    Returns:

    """

    if len(kwargs):
        raise Exception('keys %s not recognized!' % kwargs.keys())

    G = nx.DiGraph()

    # x_ancillas, z_ancillas = get_ancilla_types(instr)

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
        data_labels[i] = '$' + str(i) + '$'

    ancilla_labels = {}
    for i in qudit_nodes_ancilla:
        ancilla_labels[i] = '$' + str(i) + '$'

    G.add_nodes_from(qudit_nodes_qudit)
    plt.figure(num=None, figsize=figsize, dpi=dpi, edgecolor='k')
    plt.title("QECC layout: %s" % qecc.name)

    # Draw data qudits
    nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=qudit_nodes_data, node_color='white',
                                   node_shape='o', node_size=700, label='data qubit')
    nodes.set_edgecolor('black')

    # Draw ancilla qudits
    nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=qudit_nodes_ancilla, node_color='black',
                                   node_shape='s', node_size=700, label='ancilla qubit')
    nodes.set_edgecolor('black')

    # Label ancilla qudits
    nx.draw_networkx_labels(G, pos=pos, labels=ancilla_labels, font_size=16, font_color='white')

    # Label data qudits
    nx.draw_networkx_labels(G, pos=pos, labels=data_labels, font_size=16)

    # Label nodes
    # nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)

    ax = plt.gca()
    ax.set_facecolor('lightgray')
    ax.set_xlabel('x (arbitrary length units)')
    ax.set_ylabel('y (arbitrary length units)')
    # ax.invert_yaxis()
    # ax.invert_xaxis()

    plt.legend(labelspacing=2.5, borderpad=1.5, loc='upper left', bbox_to_anchor=(1, 1.01))

    if filename:
        plt.savefig(filename)

    plt.show()


def plot_instr(instr, figsize=(9, 9), dpi=80, filename=None, **kwargs):
    """

    Args:
        instr(LogicalInstruction):

    Returns:

    """

    if len(kwargs):
        raise Exception('keys %s not recognized!' % kwargs.keys())

    G = nx.DiGraph()

    # x_ancillas, z_ancillas = get_ancilla_types(instr)

    mapping = instr.qecc.mapping

    if mapping is None:
        mapping = NoMap()

    pos_old = instr.qecc.layout
    pos = {mapping[q]: loc for q, loc in pos_old.items()}

    qudit_nodes_data = mapset(mapping, instr.data_qudit_set)
    qudit_nodes_x = mapset(mapping, instr.ancilla_x_check)
    qudit_nodes_z = mapset(mapping, instr.ancilla_z_check)

    G.add_nodes_from(qudit_nodes_data)
    G.add_nodes_from(qudit_nodes_x)
    G.add_nodes_from(qudit_nodes_z)

    edge_labels, _, _ = graph_add_directed_cnots(instr, G)

    # print(czs)
    # print(cys)

    labels = {}
    for i in qudit_nodes_data:
        labels[i] = '$' + str(i) + '$'

    for i in qudit_nodes_x:
        labels[i] = '$' + str(i) + '$'

    for i in qudit_nodes_z:
        labels[i] = '$' + str(i) + '$'

    plt.figure(num=None, figsize=figsize, dpi=dpi, edgecolor='k')
    plt.title("Logical Instruction: '%s'  QECC: %s" % (instr.symbol, instr.qecc.name))

    nx.draw_networkx_edges(G, pos=pos, edge_labels=edge_labels, arrowsize=30)

    nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=qudit_nodes_data, node_color='lightyellow', node_size=700,
                                   label='data qubit')
    nodes.set_edgecolor('black')

    try:
        nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=qudit_nodes_x, node_color='lightcoral',
                                       node_shape='s', node_size=600, label='X ancilla')

        nodes.set_edgecolor('black')
    except AttributeError:
        pass

    try:
        nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=qudit_nodes_z, node_color='powderblue',
                                       node_shape='s', node_size=600, label='Z ancilla')
        nodes.set_edgecolor('black')
    except AttributeError:
        pass

    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=16)


    ax = plt.gca()
    ax.set_xlabel('x (arbitrary length units)')
    ax.set_ylabel('y (arbitrary length units)')
    # ax.invert_yaxis()
    # ax.invert_xaxis()

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.legend(labelspacing=2.5, borderpad=1.5, loc='upper left', bbox_to_anchor=(1, 1.01))

    if filename:
        plt.savefig(filename)

    plt.show()


def get_ancilla_types(instr):

    x_ancillas = set([])
    z_ancillas = set([])
    abs_circuit = instr.abstract_circuit

    for gate_symbol, _, params in abs_circuit.items():
        if gate_symbol == 'X check':
            ancilla = params['ancillas']
            x_ancillas.add(ancilla)
        elif gate_symbol == 'Z check':
            ancilla = params['ancillas']
            z_ancillas.add(ancilla)

    return x_ancillas, z_ancillas


def graph_add_directed_cnots(instr, G):

    circuit = instr.circuit
    edge_labels = {}
    cys = []
    czs = []

    # print(circuit)
    for i in range(len(circuit)):
        for sym, qudits, _ in circuit.items(tick=i):
            if sym == 'CNOT' or sym == 'CZ' or sym == 'CY':
                G.add_edges_from(qudits)
                for e in qudits:
                    edge_labels[e] = str(i)

                if sym == 'CZ':
                    czs.append(qudits)
                elif sym == 'CY':
                    cys.append(qudits)

    return edge_labels, czs, cys


def mapset(mapping, oldset):
    """
    Applies a mapping to a set.

    Args:
        mapping:
        oldset (set):

    Returns:

    """
    newset = set()

    for e in oldset:
        newset.add(mapping[e])

    return newset


class NoMap:
    """
    Default Mapping: item -> item.
    """

    def __init__(self):
        pass

    def __getitem__(self, item):
        return item
