"""Independent exact validation using the bundle's unmodified symbolic modules.
No Stim/DEM claim is made by this program. Full channel enumeration comes from
linear_circuit.Analysis. Inputs/outputs use standard Stim text syntax.
"""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parents[1]/'starter'
from .geometry import build_geometry,verify_geometry
from .instrument import instrument
from .linear_circuit import Inst,rank,bits,indices
from .make_memory import memory
from .cliffords import SINGLE
ANNOTATIONS={'QUBIT_COORDS','DETECTOR','OBSERVABLE_INCLUDE','SHIFT_COORDS'}

def validate(candidate,g):
 n=g['code']['n'];assert candidate['distance']==g['code']['distance'];assert candidate['schema']=='torus488.cz_cycle/1'
 edges={tuple(e) for e in g['complete_local_edges']};ms=candidate['moments'];measured=[]
 frame=candidate.get('data_h_frame',[]);assert len(set(frame))==len(frame) and all(type(q)is int and 0<=q<n for q in frame)
 assert len(ms)>=3
 for mi,m in enumerate(ms):
  active=set()
  for op in m:
   assert set(op)=={'gate','targets'}
   name,qs=op['gate'],op['targets'];assert name in SINGLE|{'CZ','R','M'}
   assert qs and len(set(qs))==len(qs) and all(type(q)is int and 0<=q<2*n for q in qs)
   assert not active.intersection(qs);active.update(qs)
   if name=='CZ':assert len(qs)%2==0 and all(tuple(sorted(p)) in edges for p in zip(qs[::2],qs[1::2]))
   if name in {'R','M'}:
    assert mi==(0 if name=='R' else len(ms)-1)
    assert all(n<=q<2*n for q in qs)
   if name=='M':measured+=qs
  if mi in {0,len(ms)-1}:
   name='R' if mi==0 else 'M';assert {q for op in m if op['gate']==name for q in op['targets']}==set(range(n,2*n))
 assert candidate['measurement_order']==measured
 return instrument(candidate,[f['data_support'] for f in g['plaquettes']],g['canonical_logicals'],n)

def normalize(c):
 out=[]
 for i in c:
  if not i.targets and i.name!='TICK':
   # DETECTOR with empty support is legal.
   if i.name!='DETECTOR':continue
  if out and i.name in SINGLE|{'R','M','CZ','DEPOLARIZE1','DEPOLARIZE2','X_ERROR'} and (i.name,i.args,i.tag)==(out[-1].name,out[-1].args,out[-1].tag):
   prev=out[-1];out[-1]=Inst(i.name,prev.targets+i.targets,i.args,i.tag)
  else:out.append(i)
 return out

def split(c):
 out=[];current=[]
 for i in c:
  if i.name=='TICK':out.append(current);current=[]
  else:current.append(i)
 if current and any(i.name not in ANNOTATIONS for i in current):out.append(current)
 elif current:out[-1]+=current
 return out

def decorate(c,nq,p=.001):
 out=[];ms=split(c)
 def add(name,qs,args=(),tag=''):out.append(Inst(name,tuple(qs),tuple(args),tag))
 for mi,m in enumerate(ms):
  active=set();collapsed=set()
  for i in m:
   if i.name in ANNOTATIONS:out.append(i);continue
   name,qs=i.name,i.targets
   assert name in SINGLE|{'CZ','R','M'} and not i.args
   assert len(set(qs))==len(qs) and not active.intersection(qs);active.update(qs)
   if name=='M':add('X_ERROR',qs,[5*p],'si1000:measurement')
   out.append(i)
   if name=='CZ':add('DEPOLARIZE2',qs,[p],'si1000:CZ')
   elif name=='R':add('X_ERROR',qs,[2*p],'si1000:reset');collapsed.update(qs)
   elif name=='M':collapsed.update(qs)
   else:add('DEPOLARIZE1',qs,[p/10],'si1000:1q')
  idle=sorted(set(range(nq))-active)
  if idle:add('DEPOLARIZE1',idle,[p/10],'si1000:idle')
  wait=sorted(set(range(nq))-collapsed)
  if collapsed and wait:add('DEPOLARIZE1',wait,[2*p],'si1000:collapse_wait')
  if mi+1<len(ms):add('TICK',[])
 return normalize(out)

def cb(c):return ('\n'.join(map(str,c))+'\n').encode()
def sha(c):return hashlib.sha256(cb(c)).hexdigest()

# Exact independent deterministic-measurement count by pure stabilizer updates.
# Pure input is |0> on all physical qubits, including unused ancillas.

def census(c,nq):
 counts=Counter();gates=Counter()
 for m in split(c):
  active=set();collapsed=set();names=set();counts['physical_moments']+=1
  for i in m:
   if i.name in ANNOTATIONS:continue
   assert not active.intersection(i.targets);active.update(i.targets);names.add(i.name)
   if i.name in {'R','M'}:collapsed.update(i.targets)
   gates[i.name]+=len(i.targets)//(2 if i.name=='CZ' else 1)
  counts['ordinary_idle_qubit_moments']+=nq-len(active)
  counts['CZ_layers']+=int('CZ'in names);counts['H_moments']+=int('H'in names)
  if collapsed:counts['collapse_moments']+=1;counts['extra_collapse_wait_qubit_moments']+=nq-len(collapsed)
 counts['gates']=dict(sorted(gates.items()));return dict(counts)

