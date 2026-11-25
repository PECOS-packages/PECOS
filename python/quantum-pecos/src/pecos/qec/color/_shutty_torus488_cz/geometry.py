"""Uniform distance-4m 4.8.8 brickwork torus and depth-12 schedule.

The quotient periods are (-d,-d), (-d,d), without added vertices or edges.
The primitive rule in rule_seed.json is extracted once from the verified d8
construction, using *physical nearest-neighbor displacements*. In particular,
it does not attempt to lift a wrapped edge just from its endpoint residues.

Public API: build_geometry(d), build_schedule(d), verify_geometry(g),
verify_schedule(g, schedule).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

SEED = json.loads(Path(__file__).with_name('rule_seed.json').read_text())


def _width(d):
    if not isinstance(d, int) or d < 4 or d % 4:
        raise ValueError('distance must be a positive multiple of four')
    return d // 4


def _key(coord, d):
    x, y = coord
    return (int(x+y) % (2*d), int(x-y) % (2*d))


def _rows(d):
    w = _width(d)
    n = d*d
    data, ancillas = [], []
    for i in range(w):
        for j in range(w):
            shift = (-4*i-4*j, -4*i+4*j)
            for local, c in enumerate(SEED['base_data_coords']):
                q = 16*(i*w+j)+local
                data.append({'index': q, 'coord': [c[k]+shift[k] for k in range(2)]})
            for local, c in enumerate(SEED['base_ancilla_coords']):
                q = n+16*(i*w+j)+local
                ancillas.append({'index': q, 'coord': [c[k]+shift[k] for k in range(2)]})
    return data, ancillas


def _add(basis, row):
    while row:
        pivot = row.bit_length()-1
        if pivot not in basis:
            basis[pivot] = row
            return True
        row ^= basis[pivot]
    return False


def _reduce(basis, row):
    for pivot in sorted(basis, reverse=True):
        if row >> pivot & 1:
            row ^= basis[pivot]
    return row


def _mask(support):
    return sum(1 << q for q in support)


def _support(mask):
    return [q for q in range(mask.bit_length()) if mask >> q & 1]


def _logicals(d, faces):
    """Find four weight-d periodic cycles with an antidiagonal Gram matrix.

    The four X rows and the reversed four Z rows then have the usual identity
    pairing. Candidate cycles repeat four primitive data vertices along a
    noncontractible sequence of cells. Exhausting this finite candidate family
    is for selecting representatives, not a circuit-distance lower bound.
    """
    if d == 8:
        return deepcopy(SEED['original_d8_logicals'])
    w = _width(d)
    checks = [_mask(f['data_support']) for f in faces]
    basis = {}
    for h in checks:
        _add(basis, h)
    candidates = set()
    for local in combinations(range(16), 4):
        for di, dj in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            v = _mask([16*(((k*di) % w)*w+(k*dj) % w)+q
                       for k in range(w) for q in local])
            if all((v & h).bit_count() % 2 == 0 for h in checks):
                if _reduce(basis, v):
                    candidates.add(v)
    candidates = sorted(candidates)
    # The quotient has dimension four. Find an isotropic pair and their
    # commuting duals directly among the weight-d cycles.
    for a in candidates:
        for b in candidates:
            if (a & b).bit_count() % 2 or not _reduce(basis, a ^ b):
                continue
            for c in candidates:
                if (a & c).bit_count() % 2 or (b & c).bit_count() % 2 != 1:
                    continue
                for e in candidates:
                    if ((a & e).bit_count() % 2 == 1
                            and (b & e).bit_count() % 2 == 0
                            and (c & e).bit_count() % 2 == 0):
                        xs = [_support(v) for v in [a, b, c, e]]
                        return {'x': xs, 'z': list(reversed(xs))}
    raise AssertionError('periodic candidate family failed to span four logicals')


def build_geometry(d):
    """Construct the fixed hardware/code family with the frozen geometry schema.

    At d8 this reproduces the original qubit indices, coordinates, faces,
    edges, record pairs and canonical logicals exactly.
    """
    w = _width(d)
    n = d*d
    data, ancillas = _rows(d)
    dlookup = {_key(r['coord'], d): r['index'] for r in data}
    alookup = {_key(r['coord'], d): r['index'] for r in ancillas}
    assert len(dlookup) == n and len(alookup) == n
    faces, pairs = [], []
    for i in range(w):
        for j in range(w):
            shift = (-4*i-4*j, -4*i+4*j)
            for f in SEED['base_faces']:
                def lookup(coords, table):
                    return [table[_key([c[k]+shift[k] for k in range(2)], d)]
                            for c in coords]
                faces.append({'index': len(faces), 'color': f['color'],
                              'data_support': lookup(f['support_coords'], dlookup)})
                pairs.append(lookup(f['record_coords'], alookup))
    lookup = dlookup | alookup
    edges = set()
    for row in data+ancillas:
        x, y = row['coord']
        for dx, dy in [(1, 0), (0, 1)]:
            other = lookup[_key([x+dx, y+dy], d)]
            edges.add(tuple(sorted((row['index'], other))))
    edges = sorted(edges)
    geometry = {
        'schema': 'tessellate.torus_488_periodic_geometry/1',
        'code': {'distance': d, 'face_rank': n//2-2, 'k': 4, 'n': n},
        'period_vectors': [[-d, -d], [-d, d]],
        'data_qubits': data, 'ancillas': ancillas, 'plaquettes': faces,
        'canonical_logicals': _logicals(d, faces),
        'reference_record_pairs': pairs,
        'complete_local_edges': [list(e) for e in edges],
        'extraction_edges': [list(e) for e in edges if e[1] >= n],
        'optional_data_data_edges': [list(e) for e in edges if e[1] < n],
        'source_sha256': deepcopy(SEED['source_sha256']),
    }
    # The original boundary source hash refers to d8. Keep its provenance,
    # but explicitly distinguish the scaled boundary in generated files.
    geometry['source_sha256']['boundary'] = (
        SEED['source_sha256']['boundary'] if d == 8 else None)
    verify_geometry(geometry)
    return geometry


MIDDLE_RECIPE = [[(0, 1), (2, 3)], [(1, 2)], [(2, 1)], [(1, 0), (3, 2)]]


def data_block(layers, code, coords, n_data):
    """Orient four even cycles on the fixed primitive quotient.

    Codes have precisely the same bit meaning as the original construction's
    data_block: sorted quotient components, start at their smallest vertex,
    visit its smallest neighbor first, alternate matching colors, XOR bit j.
    """
    if not isinstance(code, int) or not 0 <= code < 16:
        raise ValueError('matching orientation code must be in range(16)')
    def key(q):
        x, y = coords[q]
        return (int(q >= n_data), int(x+y) % 8, int(x-y) % 8)
    edges = [tuple(e) for layer in layers for e in layer]
    graph = defaultdict(set)
    for a, b in edges:
        aa, bb = key(a), key(b)
        graph[aa].add(bb)
        graph[bb].add(aa)
    assert all(len(ns) == 2 for ns in graph.values())
    left, components = set(graph), []
    while left:
        stack, component = [min(left)], set()
        while stack:
            q = stack.pop()
            if q in component:
                continue
            component.add(q)
            stack.extend(graph[q]-component)
        left.difference_update(component)
        components.append(sorted(component))
    components.sort(key=lambda c: c[0])
    assert len(components) == 4
    color = {}
    for j, component in enumerate(components):
        start = current = min(component)
        previous, step = None, 0
        while True:
            nxt = min(q for q in graph[current] if q != previous)
            color[frozenset((current, nxt))] = (step % 2) ^ ((code >> j) & 1)
            previous, current = current, nxt
            step += 1
            if current == start:
                break
        assert step % 2 == 0
    out = [[], []]
    for a, b in edges:
        out[color[frozenset((key(a), key(b)))]].append([a, b])
    assert all(len(layer) == n_data and len({q for e in layer for q in e}) == 2*n_data
               for layer in out)
    return out


def build_schedule(d, x_code=5, z_code=10, middle=None):
    """Instantiate the periodic rule, optionally changing matching orientations.

    Defaults reproduce the verified d8 circuit's directed layer gate sets.
    x_code and z_code independently orient the four quotient cycles in the
    two data blocks. `middle`, when supplied, is a list of matching layers of
    directed local-index pairs on every disjoint path (l,a,b,r), indices 0..3.
    Its length need not equal four; the rest of the cycle is preserved.
    """
    w = _width(d)
    n = d*d
    data, ancillas = _rows(d)
    lookup = [{_key(r['coord'], d): r['index'] for r in rows}
              for rows in [data, ancillas]]
    coords = {r['index']: r['coord'] for r in data+ancillas}
    layers = []
    for template in SEED['layers']:
        layer = []
        for i in range(w):
            for j in range(w):
                for kind, local, dx, dy, other_kind in template:
                    q = kind*n+16*(i*w+j)+local
                    x, y = coords[q]
                    r = lookup[other_kind][_key([x+dx, y+dy], d)]
                    layer.append([q, r])
        layers.append(layer)
    layers[2:4] = data_block(layers[2:4], x_code, coords, n)
    layers[8:10] = data_block(layers[8:10], z_code, coords, n)
    if middle is not None:
        leaf = dict(layers[1])
        paths = [(leaf[a], a, b, leaf[b]) for a, b in layers[0]
                 if a in leaf and b in leaf]
        assert len(paths) == 4*w*w and len({q for p in paths for q in p}) == n
        for layer in middle:
            used = [q for a, b in layer for q in (a, b)]
            if len(used) != len(set(used)) or any(q not in range(4) for q in used):
                raise ValueError('middle layers must be matchings on local indices 0..3')
            if any(abs(a-b) != 1 for a, b in layer):
                raise ValueError('middle gates must use existing adjacent path couplers')
        replacement = [[[p[a], p[b]] for p in paths for a, b in layer]
                       for layer in middle]
        layers[4:8] = replacement
    def hadamards(key):
        return [n+16*(i*w+j)+q for i in range(w) for j in range(w)
                for q in SEED[key]]
    return {'schema': 'tessellate.native_schedule/1',
            'h_prepare': hadamards('h_prepare'),
            'h_measure': hadamards('h_measure'),
            'measurement_order': list(range(n, 2*n)), 'layers': layers}


def verify_geometry(g):
    """Exact CSS, rank, logical and hardware consistency checks."""
    n = g['code']['n']
    d = g['code']['distance']
    rows = [_mask(f['data_support']) for f in g['plaquettes']]
    basis = {}
    for f, h in zip(g['plaquettes'], rows):
        assert h.bit_count() == len(f['data_support']) in [4, 8]
        assert all((h & k).bit_count() % 2 == 0 for k in rows)
        _add(basis, h)
    assert len(rows) == n//2 and len(basis) == n//2-2
    xs = [_mask(s) for s in g['canonical_logicals']['x']]
    zs = [_mask(s) for s in g['canonical_logicals']['z']]
    assert len(xs) == len(zs) == 4
    for v in xs+zs:
        assert v.bit_count() == d
        assert all((v & h).bit_count() % 2 == 0 for h in rows)
    assert [[(x & z).bit_count() % 2 for z in zs] for x in xs] == [
        [int(i == j) for j in range(4)] for i in range(4)]
    assert len(g['data_qubits']) == len(g['ancillas']) == n
    assert len(g['complete_local_edges']) == 4*n
    assert len(g['extraction_edges']) == 3*n
    assert len(g['optional_data_data_edges']) == n
    weights = Counter(map(int.bit_count, rows))
    return {'data': n, 'ancillas': n, 'faces': len(rows),
            'face_rank': len(basis), 'k': n-2*len(basis),
            'face_weight_counts': dict(weights),
            'logical_weights_x': [x.bit_count() for x in xs],
            'logical_weights_z': [z.bit_count() for z in zs],
            'static_distance_lower_bound_claimed_here': False}


def verify_schedule(g, schedule):
    edges = {tuple(e) for e in g['complete_local_edges']}
    incidence = Counter()
    for layer in schedule['layers']:
        qubits = [q for e in layer for q in e]
        assert len(qubits) == len(set(qubits)), 'layer is not a matching'
        for a, b in layer:
            assert tuple(sorted((a, b))) in edges
        incidence.update(qubits)
    assert len(schedule['layers']) == 12
    assert sum(map(len, schedule['layers'])) == 120*(g['code']['distance']//4)**2
    ancillas = {q['index'] for q in g['ancillas']}
    assert set(schedule['measurement_order']) == ancillas
    assert set(schedule['h_prepare']) <= ancillas
    assert set(schedule['h_measure']) <= ancillas
    return {'cnot_depth': len(schedule['layers']),
            'cnots_per_cycle': sum(map(len, schedule['layers'])),
            'gate_incidence_histogram': dict(Counter(incidence.values())),
            'allowed_couplers_only': True, 'each_layer_is_matching': True}
