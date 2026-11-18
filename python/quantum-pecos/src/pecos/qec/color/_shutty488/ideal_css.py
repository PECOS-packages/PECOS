"""Exact CSS Choi instrument checks for the frozen CX reference.
Adapted from the hash-pinned direct-local source; no fault search code.
"""
from .binary import bits, indices, rank

def pivots(rows):
 p={};rel=[]
 for v,r in rows:
  while v:
   b=v.bit_length()-1
   if b not in p:p[b]=(v,r);break
   a,s=p[b];v^=a;r^=s
  else:
   if r:rel.append(r)
 return p,rel

def reduce(v,p):
 for b,(a,r) in sorted(p.items(),reverse=True):
  if v>>b&1:v^=a
 return v

def query(v,p):
 s=0
 while v:
  b=v.bit_length()-1
  if b not in p:raise ValueError('Missing target operator '+hex(v))
  a,r=p[b];v^=a;s^=r
 return s

def xor(xs):
 v=0
 for x in xs:v^=x
 return v

def static(g):
 n=len(g['qubit_coords']);fs=[bits(f['qubits']) for f in g['plaquettes']]
 assert rank(fs)==(n-1)//2
 assert all((a&b).bit_count()%2==0 for a in fs for b in fs)
 p,_=pivots((r,0) for r in fs)
 lo=(1<<n)-1
 # All-ones is the declared logical representative at every size. No search.
 assert (lo.bit_count()&1) and all((f&lo).bit_count()%2==0 for f in fs)
 return fs,lo

def hardware(g,s):
 n=len(g['qubit_coords']);N=n+len(g['ancilla_coords']);E={tuple(sorted(e)) for e in g['gate_edges']}
 assert len(s['measurement_order'])==len(set(s['measurement_order']))
 assert set(s['measurement_order'])==set(range(n,N))
 assert set(s['h_prepare'])<=set(s['measurement_order']) and set(s['h_measure'])<=set(s['measurement_order'])
 for l in s['layers']:
  assert len({q for e in l for q in e})==2*len(l)
  for a,b in l:assert tuple(sorted((a,b))) in E and n<=max(a,b)
 return {'status':'PASS','data':n,'ancillas':N-n,'collar':0,'cx_depth':len(s['layers']),'cx_count':sum(map(len,s['layers'])),'data_data_cx':0}

def instrument(g,s,lo=None):
 """Project the exact doubled-input CSS Choi state onto all terminal records.
 Choi references are algebraic indices only, never physical circuit wires.
 """
 n=len(g['qubit_coords']);na=len(g['ancilla_coords']);N=n+na
 fs=[bits(f['qubits']) for f in g['plaquettes']]
 if lo is None:lo=static(g)[1]
 hp=set(s['h_prepare']);hm=set(s['h_measure']);mo=s['measurement_order']
 all_maps={};rels=[]
 for ax in 'XZ':
  rows=[(1<<q)|(1<<(N+q)) for q in range(n)]
  rows += [1<<q for q in range(n,N) if (q in hp)==(ax=='X')]
  for l in s['layers']:
   for a,b in l:
    c,t=(a,b) if ax=='X' else (b,a)
    rows=[v^(1<<t) if v>>c&1 else v for v in rows]
  tagged=[(v,0) for v in rows]
  tagged += [(1<<q,1<<j) for j,q in enumerate(mo) if (q in hm)==(ax=='X')]
  p,rr=pivots(tagged);rels.extend(rr)
  sin=[query(f<<N,p) for f in fs]
  sout=[query(f,p) for f in fs]
  lp=query(lo^(lo<<N),p)
  all_maps[ax]=(sin,sout,lp)
 rp,_=pivots((r,0) for r in rels)
 def red(x):return reduce(x,rp)
 si={ax:list(map(red,all_maps[ax][0])) for ax in 'XZ'}
 so={ax:list(map(red,all_maps[ax][1])) for ax in 'XZ'}
 lg={ax:[red(all_maps[ax][2])] for ax in 'XZ'}
 assert rank([v for ax in 'XZ' for v in si[ax]]+list(v for v,r in rp.values()))==2*len(fs)+len(rp), 'syndrome rank deficient'
 # Solve an explicit physical-data Pauli frame from syndrome and logical signs.
 frames={}
 for ax in 'XZ':
  eqs=[(f,si[ax][j]^so[ax][j]) for j,f in enumerate(fs)]+[(lo,lg[ax][0])]
  ep,er=pivots(eqs);assert all(red(x)==0 for x in er)
  sol=[0]*n
  for b,(v,r) in sorted(ep.items()):sol[b]=red(r^xor(sol[j] for j in indices(v^(1<<b))))
  assert all(red(xor(sol[j] for j in indices(v))^r)==0 for v,r in eqs)
  frames['z' if ax=='X' else 'x']=sol
 return dict(status='PASS',n_data=n,n_ancilla=na,input_face_record_maps=si,output_face_record_maps=so,logical_choi_signs=lg,pauli_frame_x=frames['x'],pauli_frame_z=frames['z'],record_constraints=[v for v,r in sorted(rp.values(),reverse=True)],input_syndrome_rank=2*len(fs),choi_group_rank=2*n,independent_random_measurements_on_maximally_mixed_input=na-len(rp),branches_per_valid_syndrome=2**(na-len(rp)-2*len(fs)),logical_support=list(indices(lo)))
