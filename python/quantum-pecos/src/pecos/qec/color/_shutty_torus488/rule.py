"""Periodic mixed-polarity 12-CNOT-layer superdense extraction rule."""
from .geometry import build_geometry, build_schedule as base_schedule, MIDDLE_RECIPE
from .instrument import instrument
from .make_memory import memory


def build_schedule(d):
    s = base_schedule(d, x_code=3, z_code=3, middle=MIDDLE_RECIPE)
    n = d*d
    primitive = base_schedule(4)
    leaf = dict(primitive['layers'][1])
    paths = [(leaf[a], a, b, leaf[b]) for a,b in primitive['layers'][0]
             if a in leaf and b in leaf]
    selected_local = {q % 16 for j,path in enumerate(paths) if j in (0,2) for q in path}
    selected = {q for q in range(n,2*n) if q % 16 in selected_local}
    for key in ('h_prepare','h_measure'):
        s[key] = sorted(set(s[key]) ^ selected)
    for layer in s['layers']:
        for gate in layer:
            a,b = gate
            flip = max(a,b) in selected if min(a,b) < n else a in selected and b in selected
            if flip:
                gate.reverse()
    return s


def build_memory(d, basis, p=.001):
    g = build_geometry(d)
    s = build_schedule(d)
    ins = instrument(s,[f['data_support'] for f in g['plaquettes']],g['canonical_logicals'])
    c,metadata = memory(g,[s]*d,basis,p=p,checked=[ins]*d)
    return g,s,ins,c,metadata


