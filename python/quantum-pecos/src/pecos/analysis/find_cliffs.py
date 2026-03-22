# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Functions to identify Clifford gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc

if TYPE_CHECKING:
    from pecos import Array

dtype = "complex"

cliff_str2matrix = {
    "I": pc.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype),
    "X": pc.array([[0.0, 1.0], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    "Y": pc.array([[0.0 + 0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    "Z": pc.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "SX": pc.array([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 - 1.0j, 1.0 + 0.0j]], dtype=dtype),
    "SXdg": pc.array([[1.0 + 0.0j, 0.0 + 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]], dtype=dtype),
    "SY": pc.array([[1.0 + 0.0j, -1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]], dtype=dtype),
    "SYdg": pc.array(
        [[1.0 + 0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 1.0 + 0.0j]],
        dtype=dtype,
    ),
    "SZ": pc.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "SZdg": pc.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
    "H": pc.array([[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "H2": pc.array(
        [[1.0 + 0.0j, -1.0 + 0.0j], [-1.0 + 0.0j, -1.0 + 0.0j]],
        dtype=dtype,
    ),
    "H3": pc.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=dtype),
    "H4": pc.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 - 1.0j, 0.0 + 0.0j]], dtype=dtype),
    "H5": pc.array([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, -1.0 + 0.0j]], dtype=dtype),
    "H6": pc.array([[1.0 + 0.0j, 0.0 + 1.0j], [0.0 - 1.0j, -1.0 + 0.0j]], dtype=dtype),
    "F": pc.array([[1.0 + 0.0j, 0.0 - 1.0j], [1.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "Fdg": pc.array([[1.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 0.0 - 1.0j]], dtype=dtype),
    "F2": pc.array([[1.0 + 0.0j, -1.0 + 0.0j], [0.0 + 1.0j, 0.0 + 1.0j]], dtype=dtype),
    "F2dg": pc.array(
        [[1.0 + 0.0j, 0.0 - 1.0j], [-1.0 + 0.0j, 0.0 - 1.0j]],
        dtype=dtype,
    ),
    "F3": pc.array([[1.0 + 0.0j, 0.0 + 1.0j], [-1.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "F3dg": pc.array(
        [[1.0 + 0.0j, -1.0 + 0.0j], [0.0 - 1.0j, 0.0 - 1.0j]],
        dtype=dtype,
    ),
    "F4": pc.array([[1.0 + 0.0j, 1.0 + 0.0j], [0.0 - 1.0j, 0.0 + 1.0j]], dtype=dtype),
    "F4dg": pc.array([[1.0 + 0.0j, 0.0 + 1.0j], [1.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
}

r1xy_ang2str = {
    (3.141592653589793, 3.141592653589793): "X",
    (3.141592653589793, 1.5707963267948966): "Y",
    (3.141592653589793, 0): "X",
    (3.141592653589793, -1.5707963267948966): "Y",
    (3.141592653589793, -3.141592653589793): "X",
    (1.5707963267948966, 3.141592653589793): "SXdg",
    (1.5707963267948966, 1.5707963267948966): "SY",
    (1.5707963267948966, 0): "SX",
    (1.5707963267948966, -1.5707963267948966): "SYdg",
    (1.5707963267948966, -3.141592653589793): "SXdg",
    (-1.5707963267948966, 3.141592653589793): "SX",
    (-1.5707963267948966, 1.5707963267948966): "SYdg",
    (-1.5707963267948966, 0): "SXdg",
    (-1.5707963267948966, -1.5707963267948966): "SY",
    (-3.141592653589793, 3.141592653589793): "X",
    (-3.141592653589793, 1.5707963267948966): "Y",
    (-3.141592653589793, 0): "X",
    (-3.141592653589793, -1.5707963267948966): "Y",
    (-3.141592653589793, -3.141592653589793): "X",
}


rz_ang2str = {
    (3.141592653589793,): "Z",
    (1.5707963267948966,): "SZ",
    (-1.5707963267948966,): "SZdg",
    (-3.141592653589793,): "Z",
    (4.71238898038469,): "SZdg",
    (-4.71238898038469,): "SZ",
    (6.283185307179586,): "I",
    (0.0,): "I",
}


def r1xy_matrix(theta: float, phi: float) -> Array:
    """Creates a Array matrix for a R1XY gate."""
    c = pc.cos(theta * 0.5)
    s = pc.sin(theta * 0.5)

    return pc.array(
        [
            [c, -1j * pc.exp(-1j * phi) * s],
            [-1j * pc.exp(1j * phi) * s, c],
        ],
        dtype=dtype,
    )


def rz_matrix(theta: float) -> Array:
    """Creates a Array matrix for a RZ gate."""
    return pc.array(
        [
            [pc.exp(-1j * theta * 0.5), 0.0],
            [0.0, pc.exp(1j * theta * 0.5)],
        ],
        dtype=dtype,
    )


def mnormal(m: Array, *, atol: float = 1e-12) -> Array:
    """Normalizes a Array to help with comparing matrices up to global phases."""
    # Use isclose for complex comparison (from pecos.num)
    unit = m[0, 0] if not pc.isclose(m[0, 0], 0.0, atol=atol) else m[0, 1]

    return m / unit


def m2cliff(m: Array, *, atol: float = 1e-12) -> str | bool:
    """Identifies (ignoring global phases) a Clifford given a matrix."""
    m = mnormal(m)

    for sym, c in cliff_str2matrix.items():
        if pc.isclose(c, m, atol=atol).all():
            return sym
    return False


def r1xy2cliff(
    theta: float,
    phi: float,
    *,
    atol: float = 1e-12,
    use_conv_table: bool = True,
) -> str | bool:
    """Identifies (ignoring global phases) a Clifford given the angles of a R1XY gate."""
    if use_conv_table:
        if pc.isclose(theta % pc.f64.tau, 0.0, atol=atol):
            return "I"
        for cangs, csym in r1xy_ang2str.items():
            a, b = cangs
            if pc.isclose(a, theta, atol=atol) and pc.isclose(b, phi, atol=atol):
                return csym

    m = r1xy_matrix(theta, phi)

    return m2cliff(m)


def rz2cliff(
    theta: float,
    *,
    atol: float = 1e-12,
    use_conv_table: bool = True,
) -> str | bool:
    """Identifies (ignoring global phases) a Clifford given the angles of a RZ gate."""
    if use_conv_table:
        if pc.isclose(theta % pc.f64.tau, 0.0, atol=atol):
            return "I"
        for cangs, csym in rz_ang2str.items():
            a = cangs[0]
            if pc.isclose(a, theta, atol=atol):
                return csym

    m = rz_matrix(theta)

    return m2cliff(m)
