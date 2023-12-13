import numpy as np

dtype = "complex"

cliff_dict = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype),
    "X": np.array([[0.0, 1.0], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    "Y": np.array([[0.0 + 0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype),
    "Z": np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "SqrtX": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 - 1.0j, 1.0 + 0.0j]], dtype=dtype),
    "SqrtXdg": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]], dtype=dtype),
    "SqrtY": np.array([[1.0 + 0.0j, -1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j]], dtype=dtype),
    "SqrtYdg": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [-1.0 + 0.0j, 1.0 + 0.0j]], dtype=dtype),
    "SqrtZ": np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "SqrtZdg": np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
    "H1": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "H2": np.array([[1.0 + 0.0j, -1.0 + 0.0j], [-1.0 + 0.0j, -1.0 + 0.0j]], dtype=dtype),
    "H3": np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 0.0 + 0.0j]], dtype=dtype),
    "H4": np.array([[0.0 + 0.0j, 1.0 + 0.0j], [0.0 - 1.0j, 0.0 + 0.0j]], dtype=dtype),
    "H5": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, -1.0 + 0.0j]], dtype=dtype),
    "H6": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [0.0 - 1.0j, -1.0 + 0.0j]], dtype=dtype),
    "F1": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [1.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "F1d": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [0.0 + 1.0j, 0.0 - 1.0j]], dtype=dtype),
    "F2": np.array([[1.0 + 0.0j, -1.0 + 0.0j], [0.0 + 1.0j, 0.0 + 1.0j]], dtype=dtype),
    "F2d": np.array([[1.0 + 0.0j, 0.0 - 1.0j], [-1.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
    "F3": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [-1.0 + 0.0j, 0.0 + 1.0j]], dtype=dtype),
    "F3d": np.array([[1.0 + 0.0j, -1.0 + 0.0j], [0.0 - 1.0j, 0.0 - 1.0j]], dtype=dtype),
    "F4": np.array([[1.0 + 0.0j, 1.0 + 0.0j], [0.0 - 1.0j, 0.0 + 1.0j]], dtype=dtype),
    "F4d": np.array([[1.0 + 0.0j, 0.0 + 1.0j], [1.0 + 0.0j, 0.0 - 1.0j]], dtype=dtype),
}

u1q_conv = {
    (3.141592653589793, 3.141592653589793): "X",
    (3.141592653589793, 1.5707963267948966): "Y",
    (3.141592653589793, 0): "X",
    (3.141592653589793, -1.5707963267948966): "Y",
    (3.141592653589793, -3.141592653589793): "X",
    (1.5707963267948966, 3.141592653589793): "SqrtXdg",
    (1.5707963267948966, 1.5707963267948966): "SqrtY",
    (1.5707963267948966, 0): "SqrtX",
    (1.5707963267948966, -1.5707963267948966): "SqrtYdg",
    (1.5707963267948966, -3.141592653589793): "SqrtXdg",
    (-1.5707963267948966, 3.141592653589793): "SqrtX",
    (-1.5707963267948966, 1.5707963267948966): "SqrtYdg",
    (-1.5707963267948966, 0): "SqrtXdg",
    (-1.5707963267948966, -1.5707963267948966): "SqrtY",
    (-3.141592653589793, 3.141592653589793): "X",
    (-3.141592653589793, 1.5707963267948966): "Y",
    (-3.141592653589793, 0): "X",
    (-3.141592653589793, -1.5707963267948966): "Y",
    (-3.141592653589793, -3.141592653589793): "X",
}


rz_conv = {
    (3.141592653589793,): "Z",
    (1.5707963267948966,): "SqrtZ",
    (-1.5707963267948966,): "SqrtZdg",
    (-3.141592653589793,): "Z",
    (4.71238898038469,): "SqrtZdg",
    (-4.71238898038469,): "SqrtZ",
    (6.283185307179586,): "I",
    (0.0,): "I",
}


def u1q(theta, phi):
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


def rz(theta):
    m = np.array(
        [
            [np.exp(-1j * theta * 0.5), 0.0],
            [0.0, np.exp(1j * theta * 0.5)],
        ],
        dtype=dtype,
    )

    return m


def mnormal(m):
    if not np.isclose(m[0, 0], 0.0, atol=1e-10):
        unit = m[0, 0]
    else:
        unit = m[0, 1]

    return m / unit


def m2cliff(m, atol=1e-9):
    m = mnormal(m)

    for sym, c in cliff_dict.items():
        if np.isclose(c, m, atol=atol).all():
            return sym
    return False


def u1q2cliff(theta, phi, atol=1e-9, use_conv_table=True):
    if use_conv_table:
        if np.isclose(theta % 2 * np.pi, 0.0, atol=atol):
            return "I"
        else:
            for cangs, csym in u1q_conv.items():
                a, b = cangs
                if np.isclose(a, theta, atol=atol) and np.isclose(b, phi, atol=atol):
                    return csym

    m = u1q(theta, phi)

    return m2cliff(m)


def rz2cliff(theta, atol=1e-9, use_conv_table=True):
    if use_conv_table:
        if np.isclose(theta % 2 * np.pi, 0.0, atol=atol):
            return "I"
        else:
            for cangs, csym in rz_conv.items():
                a = cangs[0]
                if np.isclose(a, theta, atol=atol):
                    return csym

    m = rz(theta)

    return m2cliff(m)
