"""This file builds up the matrix definitions of unitary gates based on the definitions defined in hqslib1.inc.

The intention is to verify simulator definitions (withing epsilon).
"""

import pecos as pc

i = 1j
pi = pc.f64.pi


# Paulis (The defining gates)
# ==================================
# First we have to define the Paulis to define everything else:
I = pc.array(
    [
        [1, 0],
        [0, 1],
    ],
    dtype=pc.dtypes.complex128,
)

X = pc.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=pc.dtypes.complex128,
)

Y = pc.array(
    [
        [0, -i],
        [i, 0],
    ],
    dtype=pc.dtypes.complex128,
)

Z = pc.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=pc.dtypes.complex128,
)

# Checking Pauli algebra:
assert (I.dot(I) == I).all()
assert (X.dot(X) == I).all()
assert (Y.dot(Y) == I).all()
assert (Z.dot(Z) == I).all()
assert (i * X.dot(Z) == Y).all()
assert (i * Z.dot(Y) == X).all()
assert (i * Y.dot(X) == Z).all()
assert (X.dot(Y) == -1 * Y.dot(X)).all()
assert (X.dot(Z) == -1 * Z.dot(X)).all()
assert (Y.dot(Z) == -1 * Z.dot(Y)).all()

# Projectors
# ==================================
project_zero = pc.array([[1, 0], [0, 0]], dtype=pc.dtypes.complex128)

project_one = pc.array([[0, 0], [0, 1]], dtype=pc.dtypes.complex128)

assert (project_zero + project_one == I).all(), "Something up with identity or the projectors"

# We will use the tensor/Kronecker. It verify that it is of the right type:
cnot_def = pc.kron(project_zero, I) + pc.kron(project_one, X)
cnot_verify = pc.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=pc.dtypes.complex128,
)
assert (cnot_verify == cnot_def).all(), "There is something wrong with the CNOT!"
cnot_def_rev = pc.kron(I, project_zero) + pc.kron(X, project_one)

# Helper functions
# ==================================


def eqv2phase(ma, mb) -> bool:
    """Show two matrices are equivalent update to a phase."""
    if ma.shape != mb.shape:
        return False
    if len(ma.shape) != 2:
        return False

    phase = None

    for ai, bi in zip(ma, mb, strict=False):
        for aij, bij in zip(ai, bi, strict=False):
            if pc.isclose(aij, 0.0):
                if not pc.isclose(aij, bij):
                    return False
            else:
                if aij == 0.0:
                    return False
                if bij == 0.0:
                    return False
                p = aij / bij

                if phase is None:
                    if not pc.isclose(p * p.conjugate(), 1.0):
                        return False
                    phase = p
                else:
                    if not pc.isclose(p, phase):
                        return False

    if not isinstance(phase, (int, float, complex)):
        return False
    return pc.isclose(ma / phase, mb).all()


def oporder_multiply(args):
    """Multiply matrices in order of operations. (Opposite of matrix multiplication order.)."""
    matrices = reversed(args)

    m_total = None

    for ms in matrices:

        # Kronecker product matrices on different qubits
        if isinstance(ms, tuple):
            k_total = None
            for mk in ms:
                k_total = mk if k_total is None else pc.kron(k_total, mk)
            ms = k_total

        if m_total is None:
            m_total = ms
        else:
            m_total = m_total.dot(ms)

    return m_total


# SINGLE-QUBIT UNITARY PRIMITIVES
# ==================================


def U(theta: float, phi: float, lamb: float):
    """As defined in OpenQASM arXiv:1707.03429."""
    return pc.linalg.expm(-i * Z * phi / 2).dot(pc.linalg.expm(-i * Y * theta / 2)).dot(pc.linalg.expm(-i * Z * lamb / 2))


assert pc.isclose(U(0.0, 0.0, 0.0), I).all()
assert eqv2phase(U(pi, 0.0, 0.0), Y)
assert eqv2phase(U(0.0, pi, 0.0), Z)
assert eqv2phase(U(0.0, 0.0, pi), Z)
assert eqv2phase(U(pi, pi / 2, -pi / 2), X)


def RZ(theta: float):
    """Opaque Rz(lambda) q;
    //gate Rz(lambda) q
    //{
    //   U(0,0,lambda) q;
    //}.
    """
    return pc.exp(i * (theta / 2)) * pc.linalg.expm(-i * Z * theta / 2)


assert pc.isclose(RZ(0.0), I).all()
assert eqv2phase(RZ(pi), Z)
assert eqv2phase(RZ(pi / 2), U(0, 0, pi / 2))
assert eqv2phase(RZ(pi / 3), U(0, 0, pi / 3))

# try 5 random attempts to equivocate to gate definition
for _ in range(5):
    lamb = pc.random.random()
    assert eqv2phase(RZ(lamb), U(0, 0, lamb))


def U1q(theta: float, phi: float):
    """Opaque U1q(theta, phi) q;
    //gate U1q(theta, phi) q
    //{
    //   U(theta, phi-pi/2, pi/2-phi) q;
    //}.
    """
    uxy = X * pc.cos(phi) + Y * pc.sin(phi)
    return pc.linalg.expm(-i * uxy * theta / 2)


# try 5 random attempts to equivocate to gate definition
for _ in range(5):
    theta, phi = pc.random.random(), pc.random.random()
    assert eqv2phase(U(theta, phi - pi / 2, pi / 2 - phi), U1q(theta, phi))


# TWO-QUBIT UNITARY PRIMITIVES
# ==================================
def SqrtZZ():
    """Opaque ZZ() q1,q2;
    //gate ZZ() q1,q2
    //{
    //	U1q(pi/2, pi/2) q2;
    //	CX q1, q2;
    //	Rz(pi/2) q1;
    //	U1q(pi/2, 0) q2;
    //	U1q(pi/2, -pi/2) q2;
    //}.
    """
    return pc.exp(i * pi / 4) * pc.linalg.expm(-i * pc.kron(Z, Z) * pi / 4)


assert pc.isclose(SqrtZZ(), pc.diag(pc.array([1, i, i, 1], dtype=pc.dtypes.complex128))).all()
assert pc.isclose(pc.linalg.matrix_power(SqrtZZ(), 2), pc.kron(Z, Z)).all()
assert pc.isclose(pc.linalg.matrix_power(SqrtZZ(), 4), pc.kron(I, I)).all()
sqrtzz_circ = oporder_multiply(
    [
        (I, U1q(pi / 2, pi / 2)),
        cnot_def,
        (RZ(pi / 2), I),
        (I, U1q(pi / 2, 0)),
        (I, U1q(pi / 2, -pi / 2)),
    ],
)
sqrtzz_def = pc.diag(pc.array([1, i, i, 1], dtype=pc.dtypes.complex128))
assert eqv2phase(sqrtzz_circ, sqrtzz_def)
assert eqv2phase(SqrtZZ(), sqrtzz_def)


def RZZ(theta: float):
    """Opaque RZZ(theta) q1,q2;

    # Not defined in header, but defined as:
    """
    return pc.exp(i * theta / 2) * pc.linalg.expm(-i * pc.kron(Z, Z) * theta / 2)


assert pc.isclose(RZZ(0.0), pc.kron(I, I)).all()
assert pc.isclose(RZZ(pi), pc.kron(Z, Z)).all()
assert pc.isclose(RZZ(pi / 2), SqrtZZ()).all()


# STANDARD GATES
# ==================================


def CX():
    """// Clifford gate: CNOT
    gate CX() c,t
    {
       U1q(-pi/2, pi/2) t;
       ZZ() c, t;
       Rz(-pi/2) c;
       U1q(pi/2, pi) t;
       Rz(-pi/2) t;
    }.
    """
    return oporder_multiply(
        [
            (I, U1q(-pi / 2, pi / 2)),
            SqrtZZ(),
            (RZ(-pi / 2), I),
            (I, U1q(pi / 2, pi)),
            (I, RZ(-pi / 2)),
        ],
    )


assert eqv2phase(CX(), cnot_def)


def H():
    """// Clifford gate: Hadamard
    gate h() a
    {
       U1q(pi/2, -pi/2) a;
       Rz(pi) a;
    }.
    """
    return oporder_multiply([U1q(pi / 2, -pi / 2), RZ(pi)])


# Standard def. from web
h_def = pc.array(
    [
        [1, 1],
        [1, -1],
    ],
    dtype=pc.dtypes.complex128,
) / pc.sqrt(2)

assert eqv2phase(H(), h_def)


def S():
    """// Clifford gate: sqrt(Z) phase gate
    gate s() a
    {
       Rz(pi/2) a;
    }.
    """
    return RZ(pi / 2)


assert eqv2phase(pc.linalg.matrix_power(S(), 2), Z)
assert eqv2phase(pc.linalg.matrix_power(S(), 4), I)


def Sdg():
    """// Clifford gate: conjugate of sqrt(Z)
    gate sdg() a
    {
       Rz(-pi/2) a;
    }.
    """
    return RZ(-pi / 2)


assert eqv2phase(pc.linalg.matrix_power(Sdg(), 2), Z)
assert eqv2phase(pc.linalg.matrix_power(Sdg(), 3), S())
assert pc.isclose(Sdg().conj().T, S()).all()
assert eqv2phase(pc.linalg.matrix_power(Sdg(), 4), I)


def T():
    """// C3 gate: sqrt(S) phase gate
    gate t() a
    {
       Rz(pi/4) a;
    }.
    """
    return pc.diag(pc.array([1, pc.exp(i * pi / 4)], dtype=pc.dtypes.complex128))


assert eqv2phase(T(), RZ(pi / 4))


def Tdg():
    """// C3 gate: conjugate of sqrt(S)
    gate tdg() a
    {
       Rz(-pi/4) a;
    }.
    """
    return pc.diag(pc.array([1, pc.exp(-i * pi / 4)], dtype=pc.dtypes.complex128))


assert eqv2phase(Tdg(), RZ(-pi / 4))
assert not eqv2phase(Tdg(), T())

# // --- Standard rotations ---


def RX(theta):
    """// Rotation around X-axis
    gate rx(theta) a
    {
       U1q(theta, 0) a;
    }.
    """
    return pc.exp(i * (theta / 2)) * pc.linalg.expm(-i * (theta / 2) * X)


# try 5 random attempts to equivocate to gate definition
for _ in range(5):
    theta = pc.random.random()
    assert eqv2phase(U1q(theta, 0), RX(theta))


def RY(theta):
    """// Rotation around Y-axis
    gate ry(theta) a
    {
       U1q(theta, pi/2) a;
    }.
    """
    return pc.exp(i * (theta / 2)) * pc.linalg.expm(-i * (theta / 2) * Y)


# try 5 random attempts to equivocate to gate definition
for _ in range(5):
    theta = pc.random.random()
    assert eqv2phase(U1q(theta, pi / 2), RY(theta))


# Already defined:
"""
// Rotation around Z-axis
gate rz(phi) a
{
   Rz(phi) a;
}
"""

# // --- QE Standard User-Defined Gates  ---


def CZ():
    """// controlled-Phase
    gate cz() a,b
    {
       h b;
       cx a,b;
       h b;
    }.
    """
    return oporder_multiply(
        [
            (I, H()),
            cnot_def,
            (I, H()),
        ],
    )


cz_def = pc.kron(project_zero, I) + pc.kron(project_one, Z)

assert eqv2phase(CZ(), cz_def)


def CY():
    """// controlled-Y
    gate cy() a,b
    {
       sdg b;
       cx a,b;
       s b;
    }.
    """
    return oporder_multiply(
        [
            (I, Sdg()),
            cnot_def,
            (I, S()),
        ],
    )


cy_def = pc.kron(project_zero, I) + pc.kron(project_one, Y)

assert eqv2phase(CY(), cy_def)


def CH():
    """// controlled-H
    gate ch() a,b
    {
       h b; sdg b;
       cx a,b;
       h b; t b;
       cx a,b;
       t b; h b; s b; x b; s a;
    }.
    """
    ch = oporder_multiply(
        [
            # Why is this so long? Simplify?
            (I, H()),
            (I, Sdg()),
            cnot_def,
            (I, H()),
            (I, T()),
            cnot_def,
            (I, T()),
            (I, H()),
            (I, S()),
            (I, X),
            (S(), I),
        ],
    )

    """
    ch = oporder_multiply([
        (I, RY(pi / 4)),

        # CX,
        (RZ(pi / 2), RZ(-pi / 2)),  # SZ, SZd
        (I, U1q(pi / 2, 0)),  # I, SX
        SqrtZZ(),  # SZZ
        (I, U1q(pi / 2, -pi / 2)),

        (I, RY(-pi / 4)),

    ])
    """

    """
    ch = oporder_multiply([
        (I, RY(pi / 4)),
        cnot_def,
        (I, RY(-pi / 4)),
    ])
    """

    return ch


# Note: !!! Can not use H() in definition due to phase
ch_def = pc.kron(project_zero, I) + pc.kron(project_one, h_def)

assert eqv2phase(CH(), ch_def)


def Toffoli():
    """// C3 gate: Toffoli
    gate ccx() a,b,c
    {
       h c;
       cx b,c; tdg c;
       cx a,c; t c;
       cx b,c; tdg c;
       cx a,c; t b; t c; h c;
       cx a,b; t a; tdg b;
       cx a,b;
    }.
    """
    cnot_def_a_c = pc.kron(pc.kron(project_zero, I), I) + pc.kron(
        pc.kron(project_one, I),
        X,
    )

    return oporder_multiply(
        [
            (I, I, h_def),
            (I, cnot_def),
            (I, I, Tdg()),
            cnot_def_a_c,
            (I, I, T()),
            (I, cnot_def),
            (I, I, Tdg()),
            cnot_def_a_c,
            (I, T(), T()),
            (I, I, H()),
            (cnot_def, I),
            (T(), Tdg(), I),
            (cnot_def, I),
        ],
    )


tof_def = pc.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=pc.dtypes.complex128,
)

assert eqv2phase(Toffoli(), tof_def)


def CU1(lamb):
    """// controlled phase rotation
    gate cu1(lambda) a,b
    {
       Rz(lambda/2) a;
       cx a, b;
       Rz(-lambda/2) b;
       cx a, b;
       Rz(lambda/2) b;
    }.
    """
    return oporder_multiply(
        [
            (RZ(lamb / 2), I),
            cnot_def,
            (I, RZ(-lamb / 2)),
            cnot_def,
            (I, RZ(lamb / 2)),
        ],
    )


def cu1_def(lamb):
    """Controlled U1 by def."""

    def U1(lamb):
        """As defined in OpenQASM arXiv:1707.03429."""
        return pc.exp(i * lamb / 2) * U(0, 0, lamb)

    return pc.kron(project_zero, I) + pc.kron(project_one, U1(lamb))


for _ in range(5):
    lamb = pc.random.random()
    assert eqv2phase(CU1(lamb), cu1_def(lamb))


def CU3(theta, phi, lamb):
    """// controlled-U
    gate cu3(theta, phi, lambda) c, t
    {
       Rz((lambda-phi)/2) t;
       cx c, t;
       Rz(-(phi+lambda)/2) t;
       U1q(-theta/2, pi/2) t;
       cx c, t;
       U1q(theta/2, pi/2) t;
       Rz(phi) t;
    }.
    """
    return oporder_multiply(
        [
            (I, RZ((lamb - phi) / 2)),
            cnot_def,
            (I, RZ(-(phi + lamb) / 2)),
            (I, U1q(-theta / 2, pi / 2)),
            cnot_def,
            (I, U1q(theta / 2, pi / 2)),
            (I, RZ(phi)),
        ],
    )


def cu3_def(theta, phi, lamb):
    """Controlled U3 by def."""

    def U3(theta, phi, lamb):
        """As defined in OpenQASM arXiv:1707.03429."""
        return U(theta, phi, lamb)

    return pc.kron(project_zero, I) + pc.kron(project_one, U3(theta, phi, lamb))


for _ in range(10):
    theta, phi, lamb = pc.random.random(), pc.random.random(), pc.random.random()
    assert eqv2phase(CU3(theta, phi, lamb), cu3_def(theta, phi, lamb))
