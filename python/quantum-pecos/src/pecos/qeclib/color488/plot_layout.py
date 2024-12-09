import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pecos.qeclib.color488.abtract_layout import gen_layout


def plot_layout(distance, numbered_qubits=False, numbered_poly=False):

    positions, polygons = gen_layout(distance)

    # Calculate the mid-point for each polygon
    pos_poly = []
    for polygon in polygons:
        node_ids = polygon[:-1]  # Exclude the color
        coords = [positions[node_id] for node_id in node_ids]
        mid_x = sum(x for x, y in coords) / len(coords)
        mid_y = sum(y for x, y in coords) / len(coords)
        pos_poly.append((mid_x, mid_y))

    # Re-write data structures to plot
    pos_poly = sorted(pos_poly, key=lambda point: (-point[1], point[0]))
    ployid2pos = {i: pos_poly[i] for i in range(len(pos_poly))}

    G = nx.Graph()

    # Add nodes representing data qubits
    for node_id, (x, y) in positions.items():
        G.add_node(node_id, pos=(x, y))

    def get_edges_from_polygon(node_ids):
        """Extract edges from polygon node ids

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
        polygon_coords.append(polygon_coords[0])  # Close the polygon by repeating the first point

        x_coords, y_coords = zip(*polygon_coords)

        plt.fill(x_coords, y_coords, color=color, alpha=0.5)  # Fill polygon with color and transparency

    # Plot the graph nodes on top (with black borders)
    pos = nx.get_node_attributes(G, "pos")
    if numbered_qubits:
        nx.draw(
            G, pos, with_labels=True, node_size=250, node_color="white", edgecolors="black", font_size=10, linewidths=2
        )
    else:
        nx.draw(G, pos, with_labels=False, node_size=20, node_color="black")

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
            plt.text(x, y, str(ploy_id), ha="center", va="center", fontsize=10, zorder=4)  # Add label

    # Set equal aspect ratio
    plt.axis("equal")

    return plt
