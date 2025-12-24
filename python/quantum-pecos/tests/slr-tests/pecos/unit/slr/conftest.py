"""Pytest fixtures for SLR tests."""

import pytest
from pecos.qeclib import qubit
from pecos.slr import CReg, Main, Permute, QReg


@pytest.fixture
def basic_permutation_program() -> tuple:
    """Create a basic program with permutation of classical registers."""
    a = CReg("a", 2)
    b = CReg("b", 2)

    prog = Main(
        a,
        b,
        Permute(
            [a[0], b[1]],
            [b[1], a[0]],
        ),
        a[0].set(1),  # Should become b[1] = 1 after permutation
    )

    return prog, a, b


@pytest.fixture
def same_register_permutation_program() -> tuple:
    """Create a program with permutation within the same register."""
    a = CReg("a", 3)

    prog = Main(
        a,
        Permute(
            [a[0], a[1], a[2]],
            [a[2], a[0], a[1]],
        ),
        a[0].set(1),  # Should become a[2] = 1
        a[1].set(0),  # Should become a[0] = 0
        a[2].set(1),  # Should become a[1] = 1
    )

    return prog, a


@pytest.fixture
def quantum_permutation_program() -> tuple:
    """Create a program with permutation of quantum registers."""
    a = QReg("a", 2)
    b = QReg("b", 2)

    prog = Main(
        a,
        b,
        Permute(
            [a[0], b[0]],
            [b[0], a[0]],
        ),
        qubit.H(a[0]),  # Should become H(b[0]) after permutation
        qubit.CX(a[0], a[1]),  # Should become CX(b[0], a[1]) after permutation
    )

    return prog, a, b


@pytest.fixture
def measurement_program() -> tuple:
    """Create a program with permutation and measurements."""
    a = QReg("a", 2)
    b = QReg("b", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        m,
        n,
        # Apply permutations to both quantum and classical registers
        Permute(
            [a[0], b[0]],
            [b[0], a[0]],
        ),
        Permute(
            [m[0], n[0]],
            [n[0], m[0]],
        ),
        # Apply quantum operations
        qubit.H(a[0]),
        qubit.CX(a[0], b[0]),
    )

    return prog, a, b, m, n


@pytest.fixture
def individual_measurement_program() -> tuple:
    """Create a program with permutation and individual measurements."""
    a = QReg("a", 2)
    b = QReg("b", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        m,
        n,
        # Apply permutations to both quantum and classical registers
        Permute(
            [a[0], b[0]],
            [b[0], a[0]],
        ),
        Permute(
            [m[0], n[0]],
            [n[0], m[0]],
        ),
        # Apply quantum operations
        qubit.H(a[0]),
        qubit.CX(a[0], b[0]),
        # Add individual measurements
        qubit.Measure(a[0]) > m[0],
        qubit.Measure(a[1]) > m[1],
    )

    return prog, a, b, m, n


@pytest.fixture
def register_measurement_program() -> tuple:
    """Create a program with permutation and register-wide measurements."""
    a = QReg("a", 2)
    b = QReg("b", 2)
    m = CReg("m", 2)
    n = CReg("n", 2)

    prog = Main(
        a,
        b,
        m,
        n,
        # Apply permutations to both quantum and classical registers
        Permute(
            [a[0], b[0]],
            [b[0], a[0]],
        ),
        Permute(
            [m[0], n[0]],
            [n[0], m[0]],
        ),
        # Apply quantum operations
        qubit.H(a[0]),
        qubit.CX(a[0], b[0]),
        # Add register-wide measurement
        qubit.Measure(a) > m,
    )

    return prog, a, b, m, n
