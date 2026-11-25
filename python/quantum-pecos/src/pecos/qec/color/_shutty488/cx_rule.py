"""Direct radius-five, 4x4-periodic twelve-slot extraction: no scheduling solve."""
from pathlib import Path
import json
PERIOD=2
RULE=json.loads((Path(__file__).parent/'templates.json').read_text())
TABLE={tuple(z['key']):z for z in RULE['classes']}
assert len(TABLE)==len(RULE['classes'])
OFFSETS=[(dx,dy)for dx in range(-5,6)for dy in range(-5,6)if abs(dx)+abs(dy)<=5]

def build_cycle(geometry,phase):
 data=geometry['qubit_coords'];co=data+geometry['ancilla_coords'];nd=len(data);right=max(x for x,y in data);at={tuple(p):q for q,p in enumerate(co)};layers=[[]for _ in range(12)];hp=[];hm=[]
 for a in range(nd,len(co)):
  x,y=co[a];sig=''.join('0'if(x+dx,y+dy)not in at else'1'if at[x+dx,y+dy]<nd else'2'for dx,dy in OFFSETS);rule=TABLE[((x-right)%4,(y-7)%4,sig)]
  if rule['h_prepare']:hp.append(a)
  if rule['h_measure']:hm.append(a)
  for slot,cx,cy,tx,ty in rule['gates']:
   layers[slot].append([at[x+cx,y+cy],at[x+tx,y+ty]])
 s=dict(layers=[sorted(l)for l in layers],h_prepare=hp,h_measure=hm,measurement_order=list(range(nd,len(co))))
 if phase%2:s=dict(s,layers=list(reversed(s['layers'])),h_prepare=hm,h_measure=hp)
 return s
