"""Independent exact binary Pauli propagation for the bundle's native circuits.
Not a substitute implementation of Stim: only the explicitly supported gate set
is accepted. Fault signatures are exact GF(2), not sampling estimates.
"""
from dataclasses import dataclass
import re, json, hashlib
from pathlib import Path
from itertools import product

NOISE={'X_ERROR','Y_ERROR','Z_ERROR','DEPOLARIZE1','DEPOLARIZE2'}
@dataclass
class Inst:
    name:str
    targets:tuple
    args:tuple=()
    tag:str=''
    def __str__(self):
        a=self.name+('['+self.tag+']' if self.tag else '')
        if self.args: a+='('+', '.join(format(v,'.15g') for v in self.args)+')'
        return a+(' '+' '.join(str(t) if isinstance(t,int) else f'rec[{t[1]}]' for t in self.targets) if self.targets else '')

def parse(text):
    out=[]
    for l in text.splitlines():
        l=l.split('#')[0].strip()
        if not l:continue
        m=re.fullmatch(r'([A-Z_0-9]+)(?:\[([^\]]*)\])?(?:\(([^)]*)\))?(?:\s+(.*))?',l)
        if not m:raise ValueError(l)
        n,t,a,tail=m.groups()
        if n not in NOISE|{'R','H','CX','M','TICK','QUBIT_COORDS','DETECTOR','OBSERVABLE_INCLUDE','X','Y','Z'}:raise ValueError(n)
        ts=[]
        for p in (tail or '').split():
            if p.startswith('rec['): ts.append(('r',int(p[4:-1])))
            else:ts.append(int(p))
        out.append(Inst(n,tuple(ts),tuple(float(v) for v in a.split(',')) if a else (), t or ''))
    return out

def write(c,path):Path(path).write_text('\n'.join(map(str,c))+'\n')
def bits(xs):
    v=0
    for x in xs:v^=1<<x
    return v

def indices(v):
    while v:
        b=v&-v;yield b.bit_length()-1;v^=b

def rank(rows):
    piv={}
    for r in rows:
        while r:
            b=r.bit_length()-1
            if b not in piv:piv[b]=r;break
            r^=piv[b]
    return len(piv)



