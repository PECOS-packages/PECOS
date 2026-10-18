# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Helix tables from Berthusen et al., arXiv:2609.03194, appendix Table I.

For C4-Helix, X0 has the same support as Z1, not Z0. Redundant
stabilizer rows and their paper ordering are intentionally retained.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HelixCode:
    """Dense paper representatives, with logical indices ordered 0, 1."""

    name: str
    n: int
    k: int
    distance: int
    stabilizers: tuple[str, ...]
    logical_x: tuple[str, str]
    logical_z: tuple[str, str]
    blocks: tuple[tuple[int, ...], ...]


def helix_code() -> HelixCode:
    """Return the [[10, 2, 3]] Helix code."""
    return HelixCode(
        name="Helix",
        n=10,
        k=2,
        distance=3,
        stabilizers=(
            "XIIXIIXXII",
            "IXIIXIIXXI",
            "XIXIIIIIXX",
            "IXIXIXIIIX",
            "IIXIXXXIII",
            "IZZIIZIIZI",
            "IIZZIIZIIZ",
            "IIIZZZIZII",
            "ZIIIZIZIZI",
            "ZZIIIIIZIZ",
        ),
        logical_x=("IIXXIXIIII", "XIIIIIXIIX"),
        logical_z=("IZIIZZIIII", "ZIIIIIIZZI"),
        blocks=(),
    )


def c4_helix_code() -> HelixCode:
    """Return the [[20, 2, 6]] C4-Helix code."""
    return HelixCode(
        name="C4-Helix",
        n=20,
        k=2,
        distance=6,
        stabilizers=(
            "XXXXIIIIIIIIIIIIIIII",
            "IIIIXXXXIIIIIIIIIIII",
            "IIIIIIIIXXXXIIIIIIII",
            "IIIIIIIIIIIIXXXXIIII",
            "IIIIIIIIIIIIIIIIXXXX",
            "XIIXIIXXIIXXXIIXIIII",
            "IIIIXIIXXXIIIIXXXIIX",
            "IXXIIIIIXIIXXXIIIIXX",
            "IIXXIXXIIIIIIXXIXXII",
            "XXIIXXIIIXXIIIIIIXXI",
            "ZZZZIIIIIIIIIIIIIIII",
            "IIIIZZZZIIIIIIIIIIII",
            "IIIIIIIIZZZZIIIIIIII",
            "IIIIIIIIIIIIZZZZIIII",
            "IIIIIIIIIIIIIIIIZZZZ",
            "ZIIZIIZZIIZZZIIZIIII",
            "IIIIZIIZZZIIIIZZZIIZ",
            "IZZIIIIIZIIZZZIIIIZZ",
            "IIZZIZZIIIIIIZZIZZII",
            "ZZIIZZIIIZZIIIIIIZZI",
        ),
        logical_x=("IIXXIIIIXIIXXIIXIIII", "XIIXIIXXIIIIIIIIIIXX"),
        logical_z=("ZIIZIIZZIIIIIIIIIIZZ", "IIZZIIIIZIIZZIIZIIII"),
        blocks=((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19)),
    )
