"""Generate triangular 4.8.8 hardware for every distance ``d = 4L - 1``.

The coordinate, plaquette-support, and ``gate_edges`` fields define the
physical problem. ``canonical_superdense_pairs`` and ``routing_ancillas`` are
included only so the bundled reference-format verifier has a runnable default;
they are not restrictions on a new construction.
"""

from __future__ import annotations

from typing import Any, Iterable


UPPER_LEFT = ((0, 2), (1, 2), (2, 0), (2, 2))
SQUARE = ((0, 0), (0, 2), (1, 0), (1, 2))
OCTAGON = (
    (0, 0),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 2),
    (3, 0),
    (3, 2),
)
LOWER_LEFT = ((0, 0), (1, 0), (2, 0), (2, 2))


def distance_for_level(level: int) -> int:
    if type(level) is not int or level < 1:
        raise ValueError("level must be a positive integer")
    return 4 * level - 1


def level_for_distance(distance: int) -> int:
    if type(distance) is not int or distance < 3 or distance % 4 != 3:
        raise ValueError("distance must have the triangular-family form 4L-1")
    return (distance + 1) // 4


def _data_coordinates(level: int) -> list[tuple[int, int]]:
    right = -(4 * level + 1)
    result = []
    for row in range(4 * level - 1):
        width = min(2 * row + 3, 8 * level - 2 * row - 3)
        left = right - width + 1
        result.extend((x, 6 + 2 * row) for x in range(left, right + 1))
    if len(result) != 8 * level * level - 1:
        raise AssertionError("closed-form data census failed")
    return result


def _ancilla_coordinates(
    level: int,
    data: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    right = -(4 * level + 1)
    left_by_row = [
        min(x for x, candidate_y in data if candidate_y == 6 + 2 * row)
        for row in range(4 * level - 1)
    ]
    result = []
    for row in range(4 * level - 2):
        # The brickwork rail alternates which of the adjacent data-row cuts it
        # follows.  This is the compact form of the three straight boundaries.
        left = left_by_row[row + 1] if row % 2 == 0 else left_by_row[row]
        row_right = right + 1 if row % 2 == 0 else right
        result.extend((x, 7 + 2 * row) for x in range(left, row_right + 1))
    expected = 8 * level * level + 2 * level - 3
    if len(result) != expected:
        raise AssertionError("closed-form ancilla census failed")
    return result


def _translated(
    shape: Iterable[tuple[int, int]], anchor: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    x, y = anchor
    return tuple(sorted((x + dx, y + dy) for dx, dy in shape))


def _face_rows(
    level: int,
    data: list[tuple[int, int]],
) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]]:
    """Return `(support coordinates, ordered record-pair coordinates)` rows."""

    right = -(4 * level + 1)
    data_set = set(data)
    raw: list[
        tuple[
            tuple[tuple[int, int], ...],
            tuple[int, int],
            str,
        ]
    ] = []

    # Upper-left truncated boundary faces.
    for offset in range(level - 1):
        raw.append(
            (
                UPPER_LEFT,
                (-(4 * level + 5 + 4 * offset), 6 + 4 * offset),
                "upper-left",
            )
        )

    # Interior and sloped-boundary squares.
    for x, y in data:
        if (x - y) % 4 == 3 and all((x + dx, y + dy) in data_set for dx, dy in SQUARE):
            raw.append((SQUARE, (x, y), "square"))

    # The vertical boundary contributes a second square phase.
    for offset in range(2 * level - 1):
        raw.append((SQUARE, (right - 1, 6 + 4 * offset), "right-square"))

    # Bulk octagons.  The two even coordinate phases are the two colors in
    # the brickwork unit cell.
    for x, y in data:
        if (
            x % 2 == 0
            and x % 4 == y % 4
            and all((x + dx, y + dy) in data_set for dx, dy in OCTAGON)
        ):
            raw.append((OCTAGON, (x, y), "octagon"))

    # Lower-left truncated boundary and the final three-boundary tip.
    for offset in range(level):
        raw.append(
            (
                LOWER_LEFT,
                (-(8 * level - 1) + 4 * offset, 4 * level + 4 + 4 * offset),
                "lower-left",
            )
        )

    result = []
    for shape, (x, y), kind in raw:
        support = _translated(shape, (x, y))
        if not set(support) <= data_set:
            raise AssertionError(f"{kind} face leaves the triangular data patch")
        if kind == "upper-left":
            pair = ((x, y + 1), (x + 1, y + 1))
        elif kind in ("right-square", "octagon"):
            pair = ((x + 1, y + 1), (x + 2, y + 1))
        elif kind == "lower-left":
            is_tip = x == right - 2 and y == 8 * level
            pair = (
                ((x + 1, y + 1), (x + 2, y + 1))
                if is_tip
                else ((x, y + 1), (x + 1, y + 1))
            )
        else:
            # The record ordering flips on the two sloped boundary phases.
            # These boundary equations are local family predicates; unlike a
            # column-wise min/max, they do not inspect the other faces in the
            # finite patch to decide this face's orientation.
            reverse = (x % 4 == 3 and x + y == right + 4) or (
                x % 4 == 1 and y - x == -3 * right - 2
            )
            pair = (
                ((x + 1, y + 1), (x, y + 1))
                if reverse
                else ((x, y + 1), (x + 1, y + 1))
            )
        result.append((support, pair))

    result.sort(
        key=lambda row: (
            min(y for _, y in row[0]),
            min(x for x, _ in row[0]),
            len(row[0]),
            row[0],
        )
    )
    if len(result) != 4 * level * level - 1:
        raise AssertionError("closed-form face census failed")
    if len({support for support, _ in result}) != len(result):
        raise AssertionError("closed-form face generator emitted a duplicate")
    return result


def _induced_grid_edges(
    coords: list[tuple[int, int]],
) -> list[list[int]]:
    index_at = {point: index for index, point in enumerate(coords)}
    if len(index_at) != len(coords):
        raise AssertionError("geometry contains coincident qubits")
    edges = []
    for index, (x, y) in enumerate(coords):
        for neighbor in ((x + 1, y), (x, y + 1)):
            other = index_at.get(neighbor)
            if other is not None:
                edges.append(sorted((index, other)))
    return sorted(edges)


def generate_base_layout(level: int) -> dict[str, Any]:
    """Generate a base layout for every triangular level `L >= 1`."""

    distance_for_level(level)
    data = _data_coordinates(level)
    ancillas = _ancilla_coordinates(level, data)
    coords = [*data, *ancillas]
    index_at = {point: index for index, point in enumerate(coords)}
    face_rows = _face_rows(level, data)
    try:
        pairs = [[index_at[left], index_at[right]] for _, (left, right) in face_rows]
    except KeyError as exc:
        raise AssertionError("record pair is absent from the ancilla rail") from exc
    paired = [qubit for pair in pairs for qubit in pair]
    if len(paired) != len(set(paired)):
        raise AssertionError("closed-form record pairs overlap")
    n_data = len(data)
    all_ancillas = set(range(n_data, len(coords)))
    routing = sorted(all_ancillas - set(paired))
    if len(routing) != 2 * level - 1:
        raise AssertionError("closed-form routing-ancilla census failed")
    edges = _induced_grid_edges(coords)
    if len(edges) != 32 * level * level - 10 * level - 3:
        raise AssertionError("closed-form coupler census failed")
    return {
        "qubit_coords": [list(point) for point in data],
        "ancilla_coords": [list(point) for point in ancillas],
        "gate_edges": edges,
        "plaquettes": [
            {
                "color": "face",
                "qubits": sorted(index_at[point] for point in support),
            }
            for support, _ in face_rows
        ],
        "canonical_superdense_pairs": pairs,
        "routing_ancillas": routing,
        "torus_vectors": [[0, 0], [0, 0]],
    }
