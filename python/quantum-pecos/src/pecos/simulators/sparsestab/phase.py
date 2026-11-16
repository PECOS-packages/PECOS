"""Product phases for sparse stabilizer generators in the W = XZ convention."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.sparsestab.state import Gens


def product_phase(gens: Gens, rows: set[int]) -> int:
    """Return the exponent of i for a product of commuting generator rows.

    Each row stores (-1)**minus * i**imag times its XZ support. Left
    multiplication contributes (-1)**|Z_row & X_accumulated| when reordering
    ZX to XZ. Since the rows commute, their iteration order does not matter.
    """
    phase = 2 * len(rows & gens.signs_minus) + len(rows & gens.signs_i)
    cumulative_x = set()
    for row in rows:
        phase += 2 * len(gens.row_z[row] & cumulative_x)
        cumulative_x ^= gens.row_x[row]
    return phase % 4
