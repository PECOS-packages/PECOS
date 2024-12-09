def gen_layout(distance: int):
    """Creates an abstract layout which represents the location of the qubits in the code for a fixed 2D geometry."""
    lattice_height = 4 * distance - 4
    lattice_width = 2 * distance - 2
    pos_qubits = []
    pos_checks = []

    for y in range(lattice_width + 1):
        for x in range(lattice_height + 1):
            if ((x, y) == (x, x + 2) and x % 2 == 1 and y % 8 == 3) or (
                (x, y) == (4 * distance - y, y) and x % 2 == 1 and y % 8 == 7
            ):
                pass

            elif (x, y) > (x, x) or (x, y) > (4 * distance - y - 2, y):
                continue

            if x % 2 == 0 and y % 2 == 0:
                if (y / 2) % 4 == 1 or (y / 2) % 4 == 2:
                    if (x / 2) % 4 == 2 or (x / 2) % 4 == 3:
                        pos_qubits.append((x, y))

                else:
                    if (x / 2) % 4 == 0 or (x / 2) % 4 == 1:
                        pos_qubits.append((x, y))

            if x % 4 == 1 and y % 4 == 3:
                pos_checks.append((x, y))

            if y == 0 and x % 8 == 5:
                pos_checks.append((x, y))

    pos_qubits = sorted(pos_qubits, key=lambda point: (-point[1], point[0]))
    pos_checks = sorted(pos_checks, key=lambda point: (-point[1], point[0]))
    nodeid2pos = {i: pos_qubits[i] for i in range(len(pos_qubits))}
    pos2nodeid = {v: k for k, v in nodeid2pos.items()}

    polygons = []
    for x, y in pos_checks:
        if square := found_square(x, y, pos2nodeid):
            polygons.append(square)
        elif y == 0 and (gon := found_bottomgon(x, y, pos2nodeid)):
            polygons.append(gon)
        elif octo := found_octogon(x, y, pos2nodeid):
            polygons.append(octo)
        else:
            pass

    return nodeid2pos, polygons


def get_boundaries(nodeid2pos):
    """Determines the nodes that lie along the left, right, and bottom boundaries

    Returns:
        Three lists: 1) nodes of the left, 2) bottom, and  3) right boundaries of a triangular 4.8.8 color code.
    """
    # TODO: do this...
    pass


def found_square(x, y, pos2nodeid):
    square = [(x - 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)]
    square_ids = []
    for coord in square:
        nid = pos2nodeid.get(coord)
        if nid is None:
            return False
        square_ids.append(nid)

    square_ids.append("red")

    return square_ids


def found_octogon(x, y, pos2nodeid):
    octogon = [
        (x - 1, y + 3),
        (x - 3, y + 1),
        (x - 3, y - 1),
        (x - 1, y - 3),
        (x + 1, y - 3),
        (x + 3, y - 1),
        (x + 3, y + 1),
        (x + 1, y + 3),
    ]
    octo_ids = []

    for coord in octogon:
        nid = pos2nodeid.get(coord)
        if nid is not None:
            octo_ids.append(nid)

    if not octo_ids:
        return False

    if (x - 1) // 4 % 2:
        octo_ids.append("green")
    else:
        octo_ids.append("blue")

    return octo_ids


def found_bottomgon(x, y, pos2nodeid):
    coords = [
        (x - 1, y + 2),
        (x - 3, y),
        (x + 3, y),
        (x + 1, y + 2),
    ]
    found_ids = []
    for coord in coords:
        nid = pos2nodeid.get(coord)
        if nid is None:
            return False
        found_ids.append(nid)

    if (x - 1) // 4 % 2:
        found_ids.append("green")
    else:
        found_ids.append("blue")

    return found_ids
