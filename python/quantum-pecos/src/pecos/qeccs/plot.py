"""Plotting utilities for quantum error correction codes.

This module provides visualization tools for quantum error correction codes,
including lattice plots, syndrome visualization, and code structure diagrams
for various QECC implementations in PECOS.
"""

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

from typing import TYPE_CHECKING, TypeVar

import networkx as nx

if TYPE_CHECKING:

    from pecos.protocols import LogicalInstructionProtocol, QECCProtocol

T = TypeVar("T")


# plot intsructions
def plot_qecc(
    qecc: "QECCProtocol",
    figsize: tuple[int, int] = (9, 9),
    dpi: int = 80,
    filename: str | None = None,
    title_font_size: int = 16,
    axis_font_size: int = 14,
    legend_font_size: int = 14,
    **kwargs: object,
) -> None:
    """Produces a plot of a qecc.

    Args:
    ----
        qecc(QECC): The ``qecc`` instance that is to be plotted.
        figsize(tuple of int): The size of the plotted figure.
        dpi: Dots per inch resolution for the plot.
        filename: Optional filename to save the plot. If None, the plot is displayed but not saved.
        title_font_size: Font size for the plot title.
        axis_font_size: Font size for axis labels.
        legend_font_size: Font size for legend text.
        **kwargs: Additional keyword arguments (will raise exception if any are provided).

    """
    from matplotlib import pyplot as plt

    if kwargs:
        msg = f"keys {kwargs.keys()} not recognized!"
        raise Exception(msg)

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
    plt.title(f"QECC layout: {qecc.name}", size=title_font_size)

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
    instr: "LogicalInstructionProtocol",
    figsize: tuple[int, int] = (9, 9),
    dpi: int = 80,
    filename: str | None = None,
    title_font_size: int = 16,
    axis_font_size: int = 14,
    legend_font_size: int = 14,
    **kwargs: object,
) -> None:
    """Plot syndrome extraction using the provided configuration.

    Args:
    ----
        instr(LogicalInstruction): The logical instruction to plot
        figsize(tuple of int): The size of the plotted figure
        dpi: Dots per inch resolution for the plot
        filename: Optional filename to save the plot. If None, the plot is displayed but not saved
        title_font_size: Font size for the plot title
        axis_font_size: Font size for axis labels
        legend_font_size: Font size for legend text
        **kwargs: Additional keyword arguments (will raise exception if any are provided)

    """
    from matplotlib import pyplot as plt

    if kwargs:
        msg = f"keys {kwargs.keys()} not recognized!"
        raise Exception(msg)

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


def get_ancilla_types(instr: "LogicalInstructionProtocol") -> tuple[set, set]:
    """Extract X and Z ancilla qubits from logical instruction.

    Analyzes the abstract circuit of a logical instruction to identify
    which ancilla qubits are used for X-type and Z-type stabilizer checks.

    Args:
        instr: Logical instruction containing abstract circuit with check operations.

    Returns:
        Tuple of (X ancillas set, Z ancillas set) containing qubit identifiers.
    """
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


def graph_add_directed_cnots(
    instr: "LogicalInstructionProtocol",
    g: nx.DiGraph,
) -> tuple[dict, list, list]:
    """Add directed CNOT edges to graph from logical instruction.

    Processes the circuit of a logical instruction to add directed edges
    for two-qubit gates (CNOT, CZ, CY) to a NetworkX directed graph,
    with edge labels indicating the time step.

    Args:
        instr: Logical instruction containing the quantum circuit.
        g: NetworkX directed graph to add edges to.

    Returns:
        Tuple of (edge labels dict, CZ gate list, CY gate list).
    """
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


class NoMap:
    """Default Mapping: item -> item."""

    def __init__(self) -> None:
        """Initialize the NoMap identity mapping."""

    def __getitem__(self, item: T) -> T:
        """Return the item unchanged (identity mapping)."""
        return item


def mapset(mapping: dict | NoMap, oldset: set) -> set:
    """Applies a mapping to a set.

    Args:
    ----
        mapping: A dictionary-like object that maps elements from the old set to new values.
        oldset (set): The original set whose elements will be mapped to new values.

    """
    newset = set()

    for e in oldset:
        newset.add(mapping[e])

    return newset
