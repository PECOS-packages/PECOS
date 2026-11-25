"""Independent signed Pauli conjugation, not a Stim replacement.
Only the named primitives below are supported by this validation path.
Conventions: each entry is (x,z,k) for i**k X**x Z**z.
"""
def table(ximage,zimage):
 t={}
 for x in (0,1):
  for z in (0,1):
   a=ximage if x else (0,0,0);b=zimage if z else (0,0,0)
   t[x,z]=(a[0]^b[0],a[1]^b[1],(a[2]+b[2]+2*a[1]*b[0])%4)
 return t
IMAGES={
 'I':table((1,0,0),(0,1,0)),
 'H':table((0,1,0),(1,0,0)),
 'X':table((1,0,0),(0,1,2)),
 'Y':table((1,0,2),(0,1,2)),
 'Z':table((1,0,2),(0,1,0)),
 'S':table((1,1,1),(0,1,0)),
 'S_DAG':table((1,1,3),(0,1,0)),
}
SINGLE=set(IMAGES)
