"""The additional signed native Clifford actions; no Stim shim.
Convention: (x,z,k) denotes i**k X**x Z**z.
Checked against quantumlib/Stim doc/gates.md (2026-09-09).
"""
from .cliffords import IMAGES, SINGLE, table
IMAGES.update({
    "C_XYZ": table((1, 1, 1), (1, 0, 0)),   # X -> Y; Z -> X
    "C_ZYX": table((0, 1, 0), (1, 1, 1)),   # X -> Z; Z -> Y
    "SQRT_X": table((1, 0, 0), (1, 1, 3)),  # X -> X; Z -> -Y
})
SINGLE.update(IMAGES)
