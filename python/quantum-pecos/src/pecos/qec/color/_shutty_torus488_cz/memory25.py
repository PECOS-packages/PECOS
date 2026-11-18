#!/usr/bin/env python3
"""Full r-round memories of exactly 25r physical moments, including boundaries.

All ancillas reset and measure together, once per complete cycle.  The initial
data reset/frame and terminal data frame/readout overlap disjoint ancilla
operations.  Every per-wire operation and every record is retained.
"""
from copy import deepcopy
from pathlib import Path
import argparse,json
from . import native_gates
from .family import make_cycle
from .check import (Inst, cb, census, decorate, normalize, validate, ANNOTATIONS)
from .cliffords import SINGLE
from .geometry import build_geometry
from .make_memory import memory


def parity_annotations(reference, record_map, total_records):
    assert sorted(record_map)==list(range(total_records))
    old_count=0;out=[]
    for inst in reference:
        if inst.name=='M':old_count+=len(inst.targets)
        elif inst.name in {'DETECTOR','OBSERVABLE_INCLUDE'}:
            out.append(Inst(inst.name,tuple(('r',record_map[old_count+v]-total_records)
                                           for typ,v in inst.targets),inst.args,inst.tag))
    assert old_count==total_records
    return out


def wire_words(circuit,nq):
    words=[[] for _ in range(nq)]
    for inst in circuit:
        if inst.name not in SINGLE|{'R','M','CZ'}:continue
        width=2 if inst.name=='CZ' else 1
        for k in range(0,len(inst.targets),width):
            qs=inst.targets[k:k+width]
            for q in qs:words[q].append((inst.name,qs))
    return words


def build_memory(distance,basis,rounds=None,p=.001,*,geometry=None,checked=None):
    if basis not in {'X','Z'}:raise ValueError('basis must be X or Z')
    if rounds is None:rounds=distance
    if type(rounds) is not int or rounds < 1:raise ValueError('rounds must be a positive integer')
    if not 0<p<.2:raise ValueError('noise p must satisfy 0<p<0.2')
    n=distance*distance;c=make_cycle(distance)
    g=build_geometry(distance) if geometry is None else geometry
    a=validate(c,g) if checked is None else checked
    reference,_=memory(g,[c]*rounds,basis,checked=[a]*rounds)
    reference=normalize(reference)
    moments=[]
    for _ in range(rounds):moments.extend(deepcopy(c['moments']))
    rotated=sorted((set(range(n)) if basis=='X' else set())^set(c['data_h_frame']))
    # Active data begin at moment >=4 and end at moment <=19 in each cycle.
    moments[0].insert(0,{'gate':'R','targets':list(range(n))})
    if rotated:
        moments[1].insert(0,{'gate':'H','targets':rotated})
        moments[-2].insert(0,{'gate':'H','targets':rotated})
    moments[-1].append({'gate':'M','targets':list(range(n))})
    clean=[i for i in reference if i.name=='QUBIT_COORDS']
    for t,moment in enumerate(moments):
        clean.extend(Inst(op['gate'],tuple(op['targets'])) for op in moment)
        if t+1<len(moments):clean.append(Inst('TICK',()))
    nm=(rounds+1)*n
    clean.extend(parity_annotations(reference,list(range(nm)),nm))
    clean=normalize(clean)
    assert wire_words(reference,2*n)==wire_words(clean,2*n)
    noisy=decorate(clean,2*n,p)
    resources=census(clean,2*n)
    assert resources['physical_moments']==25*rounds
    return dict(clean=clean,noisy=noisy,reference=reference,
                cycle=c,geometry=g,instrument=a,census=resources)


