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

import numpy as np

dtype = "complex"

cliff_str2matrix = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype),
    "X": np.array([[0.0, 1.0], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    "Y": np.array([[0.0 + 0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    "Z": np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "SX": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 - 1.0j, 1.0 + 0.0j]], dtype=dtype),
    "SXdg": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]], dtype=dtype),
    "SY": np.array([[1.0 + 0.0j, -1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]], dtype=dtype),
    "SYdg": np.array(
        [[1.0 + 0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 1.0 + 0.0j]],
        dtype=dtype,
    ),
    "SZ": np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "SZdg": np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
    "H": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "H2": np.array(
        [[1.0 + 0.0j, -1.0 + 0.0j], [-1.0 + 0.0j, -1.0 + 0.0j]],
        dtype=dtype,
    ),
    "H3": np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=dtype),
    "H4": np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 - 1.0j, 0.0 + 0.0j]], dtype=dtype),
    "H5": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, -1.0 + 0.0j]], dtype=dtype),
    "H6": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [0.0 - 1.0j, -1.0 + 0.0j]], dtype=dtype),
    "F": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [1.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "Fdg": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 0.0 - 1.0j]], dtype=dtype),
    "F2": np.array([[1.0 + 0.0j, -1.0 + 0.0j], [0.0 + 1.0j, 0.0 + 1.0j]], dtype=dtype),
    "F2dg": np.array(
        [[1.0 + 0.0j, 0.0 - 1.0j], [-1.0 + 0.0j, 0.0 - 1.0j]],
        dtype=dtype,
    ),
    "F3": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [-1.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "F3dg": np.array(
        [[1.0 + 0.0j, -1.0 + 0.0j], [0.0 - 1.0j, 0.0 - 1.0j]],
        dtype=dtype,
    ),
    "F4": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [0.0 - 1.0j, 0.0 + 1.0j]], dtype=dtype),
    "F4dg": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [1.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
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


def r1xy_matrix(theta: float, phi: float) -> np.ndarray:
    """Creates a np.array matrix for a R1XY gate."""
    c = np.cos(theta * 0.5)
    s = np.sin(theta * 0.5)

    m = np.array(
        [
            [c, -1j * np.exp(-1j * phi) * s],
            [-1j * np.exp(1j * phi) * s, c],
        ],
        dtype=dtype,
    )

    return m


def rz_matrix(theta: float) -> np.ndarray:
    """Creates a np.array matrix for a RZ gate."""
    m = np.array(
        [
            [np.exp(-1j * theta * 0.5), 0.0],
            [0.0, np.exp(1j * theta * 0.5)],
        ],
        dtype=dtype,
    )

    return m


def mnormal(m: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Normalizes a np.array to help with comparing matrices up to global phases."""
    if not np.isclose(m[0, 0], 0.0, atol=atol):
        unit = m[0, 0]
    else:
        unit = m[0, 1]

    return m / unit


def m2cliff(m: np.array, *, atol: float = 1e-12) -> str | bool:
    """Identifies (ignoring global phases) a Clifford given a matrix."""
    m = mnormal(m)

    for sym, c in cliff_str2matrix.items():
        if np.isclose(c, m, atol=atol).all():
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
        if np.isclose(theta % 2 * np.pi, 0.0, atol=atol):
            return "I"
        else:
            for cangs, csym in r1xy_ang2str.items():
                a, b = cangs
                if np.isclose(a, theta, atol=atol) and np.isclose(b, phi, atol=atol):
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
        if np.isclose(theta % 2 * np.pi, 0.0, atol=atol):
            return "I"
        else:
            for cangs, csym in rz_ang2str.items():
                a = cangs[0]
                if np.isclose(a, theta, atol=atol):
                    return csym

    m = rz_matrix(theta)

    return m2cliff(m)
