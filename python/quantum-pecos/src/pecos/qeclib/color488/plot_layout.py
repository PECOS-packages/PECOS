# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Plotting utilities for Color488 code layouts."""

from typing import TYPE_CHECKING

import networkx as nx
from matplotlib import pyplot as plt

if TYPE_CHECKING:

    from pecos.qeclib.color488 import Color488


def plot_layout(
    color488: "Color488",
    *,
    numbered_qubits: bool = False,
    numbered_poly: bool = False,
) -> plt:
    """Plot the layout of a Color488 code.

    Args:
        color488: A Color488 instance
        numbered_qubits: Whether to number the data qubits
        numbered_poly: Whether to number the polygons

    Returns:
        The matplotlib pyplot module with the plot rendered.
    """
    import matplotlib.pyplot as plt

    positions, polygons = color488.get_layout()

    # Calculate the mid-point for each polygon
    pos_poly = []
    for polygon in polygons:
        node_ids = polygon[:-1]  # Exclude the color
        coords = [positions[node_id] for node_id in node_ids]
        mid_x = sum(x for x, _ in coords) / len(coords)
        mid_y = sum(y for _, y in coords) / len(coords)
        pos_poly.append((mid_x, mid_y))

    # Re-write data structures to plot
    pos_poly = sorted(pos_poly, key=lambda point: (-point[1], point[0]))
    ployid2pos = {i: pos_poly[i] for i in range(len(pos_poly))}

    g = nx.Graph()

    # Add nodes representing data qubits
    for node_id, (x, y) in positions.items():
        g.add_node(node_id, pos=(x, y))

    def get_edges_from_polygon(node_ids: list[int]) -> list[tuple[int, int]]:
        """Extract edges from polygon node ids.

        Args:
            node_ids: List of node IDs that form the polygon.

        Returns:
            Edges as pairs of consecutive nodes (including the polygon)
        """
        edges = []
        for i in range(len(node_ids)):
            edge = (node_ids[i], node_ids[(i + 1) % len(node_ids)])
            edges.append(edge)
        return edges

    # Collect all edges in set to not double count
    polygon_edges = []
    for polygon in polygons:
        node_ids = polygon[:-1]  # Exclude the color
        edges = get_edges_from_polygon(node_ids)
        polygon_edges.append((edges, polygon[-1]))  # Add edges with their color

    shared_edges = set()
    unique_edges = []

    for edges, color in polygon_edges:
        for edge in edges:
            if edge in shared_edges or (edge[1], edge[0]) in shared_edges:
                continue
            shared_edges.add(edge)
            unique_edges.append((edge, color))

    # Plot edges as black lines
    for edge, _ in unique_edges:
        x_coords = [positions[edge[0]][0], positions[edge[1]][0]]
        y_coords = [positions[edge[0]][1], positions[edge[1]][1]]
        plt.plot(x_coords, y_coords, "k-", lw=2)  # Black line for shared edges

    # Plot filled in polygons
    for polygon in polygons:
        node_ids = polygon[:-1]
        color = polygon[-1]

        polygon_coords = [positions[node_id] for node_id in node_ids]
        polygon_coords.append(
            polygon_coords[0],
        )  # Close the polygon by repeating the first point

        x_coords, y_coords = zip(*polygon_coords, strict=False)

        plt.fill(
            x_coords,
            y_coords,
            color=color,
            alpha=0.5,
        )  # Fill polygon with color and transparency

    # Plot the graph nodes on top (with black borders)
    pos = nx.get_node_attributes(g, "pos")
    if numbered_qubits:
        nx.draw(
            g,
            pos,
            with_labels=True,
            node_size=250,
            node_color="white",
            edgecolors="black",
            font_size=10,
            linewidths=2,
        )
    else:
        nx.draw(g, pos, with_labels=False, node_size=20, node_color="black")

    if numbered_poly:
        # Add white nodes with black borders for ployid2pos
        for ploy_id, (x, y) in ployid2pos.items():
            plt.scatter(
                x,
                y,
                s=200,
                c="lightgrey",
                edgecolors="black",
                zorder=3,
                marker="s",
                linewidths=1.5,
            )  # Draw node
            plt.text(
                x,
                y,
                str(ploy_id),
                ha="center",
                va="center",
                fontsize=10,
                zorder=4,
            )  # Add label

    # Set equal aspect ratio
    plt.axis("equal")

    return plt
