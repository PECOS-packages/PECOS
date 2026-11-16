#!/usr/bin/env python3
"""Spatially periodic 25-moment common-reset/common-readout CZ cycle.

The 16+16 primitive cell repeats for every distance d=4m.  Complete X/Z
extraction occurs every round, with no temporal alternation and no new edges.
"""
from pathlib import Path
import argparse
import json
from . import native_gates
from .core import baseline, compact, cycle_census, gate, atoms, asap_events, periodic_key

ROOT = Path(__file__).resolve().parent
NAMES = ['I', 'H', 'S', 'C_XYZ', 'C_ZYX', 'SQRT_X']


def load_rule():
    raw = json.loads((ROOT/'rule25.json').read_text())
    codes = {tuple(map(int,k.split(':'))):v for k,v in raw.items()}
    assert set(codes) == {(p,q) for p in range(1,26,2) for q in range(16)}
    assert all(type(v) is int and 0 <= v < len(NAMES) for v in codes.values())
    return codes


def make_cycle(distance: int, *, compacted: bool = True) -> dict:
    c = baseline(distance)
    n = distance*distance
    codes = load_rule()
    for phase in range(1,26,2):
        groups = {}
        for operation in c['moments'][phase]:
            for q in operation['targets']:
                if q < n:
                    groups.setdefault('H', []).append(q)
        for q in range(n,2*n):
            code = codes[phase,q%16]
            if code:
                groups.setdefault(NAMES[code], []).append(q)
        c['moments'][phase] = [gate(name,qs) for name,qs in groups.items()]
    if compacted:
        c = compact(c)
        assert len(c['moments']) == 25
    return c


