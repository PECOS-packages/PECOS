# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from typing import Any

from pecos.simulators.sparsesim.state import SparseSim


def Identity(state: SparseSim, qubit: int, **params: Any) -> None:
    """Identity, which does nothing.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """


def X(state: SparseSim, qubit: int, **params: Any) -> None:
    """X
    Returns:

    X -> X
    Z -> -Z
    W -> -W
    Y -> -Y
    => If you have a Z component, add a -1.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_z[qubit]


def Y(state: SparseSim, qubit: int, **params: Any) -> None:
    """Pauli Y.

    X -> -X
    Z -> -Z
    W -> W
    Y -> Y
    => If you have an X or Z component but not both, add a -1.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X or Z (exclusive) -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] ^ stabs.col_z[qubit]


def Z(state: SparseSim, qubit: int, **params: Any) -> None:
    """Z
    Returns:

    X -> -X
    Z -> Z
    W -> -W
    Y -> -Y
    => If you have a X component, add a -1.

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit]


def SX(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Q rotation to stabilizers and destabilizers.

    Q = \sqrt{X} = HSH

    Q = Q
    X = Q^2
    Q^{\dagger} = Q^3
    I = Q^4

    X -> X
    Z -> -iW = -Y
    W -> -iZ
    Y -> Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z -> -1
    # ---------------------
    stabs.signs_minus ^= stabs.col_z[qubit]

    # Z -> i
    # ---------------------

    # Now we need to deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]

        for i in g.col_z[qubit]:
            g.row_x[i] ^= {qubit}


def SXdg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Q^{\dagger} rotation to stabilizers and destabilizers.

    Qd = \sqrt{X}^{\dagger} = H S^{\dagger}H

    Q = Q
    X = Q^2
    Q^{\dagger} = Q^3
    I = Q^4

    X -> X
    Z -> iW = Y
    W -> iZ
    Y -> -Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z -> i
    # ---------------------

    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]

        for i in g.col_z[qubit]:
            g.row_x[i] ^= {qubit}


def SY(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a R rotation to stabilizers and destabilizers.

    R = \sqrt{XZ} = SQS^{\dagger}

    R = R
    XZ = R^2
    R^{\dagger} = R^3
    I = R^4

    X -> -Z
    Z -> X
    W -> W
    Y -> Y

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X not Z -> -1
    # ---------------------
    stabs.signs_minus ^= stabs.col_x[qubit] - stabs.col_z[qubit]

    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]


def SYdg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a R rotation to stabilizers and destabilizers.

    R^{\dagger} = \sqrt{XZ} = SQ^{\dagger}S^{\dagger}

    R = R
    XZ = R^2
    R^{\dagger} = R^3
    I = R^4

    X -> Z
    Z -> -X
    W -> W
    Y -> Y

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z not X -> -1
    # ---------------------
    stabs.signs_minus ^= stabs.col_z[qubit] - stabs.col_x[qubit]

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]


def SZ(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a phase gate (S) rotation to stabilizers and destabilizers.

    S = \sqrt{Z}

    This is P in CHP.

    S = S
    Z = S^2
    S^{\dagger} = S^3
    I = S^4

    X -> iW = Y
    Z -> Z
    W -> iX
    Y -> -X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X -> i
    # ---------------------
    # i * i = -1
    stabs.signs_minus ^= stabs.signs_i & stabs.col_x[qubit]
    # For each X add an i unless there is already an i there then delete it.
    stabs.signs_i ^= stabs.col_x[qubit]

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # Z += X
        g.col_z[qubit] ^= g.col_x[qubit]

        # Update row
        for i in g.col_x[qubit]:
            g.row_z[i] ^= {qubit}


def SZdg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hermitian adjoint phase gate (S^{\dagger}) rotation to stabilizers and destabilizers.

    S = S
    Z = S^2
    S^{\dagger} = S^3
    I = S^4

    X -> -iW = -Y
    Z -> Z
    W -> -iX
    Y -> X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X -> -1
    # ---------------------
    stabs.signs_minus ^= stabs.col_x[qubit]

    # X -> i
    # ---------------------
    # Now we need to deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in X can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # Z += X
        g.col_z[qubit] ^= g.col_x[qubit]

        for i in g.col_x[qubit]:
            g.row_z[i] ^= {qubit}


def H(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hadamard gate (H) rotation to stabilizers and destabilizers.

    Same as H1 in some places in PECOS.

    X + Z

    X -> Z
    Z -> X
    W -> -W
    Y -> -Y

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X and Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] & stabs.col_z[qubit]

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]


def H2(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hadamard gate (H4) rotation to stabilizers and destabilizers.

    X - Z

    X -> -Z
    Z -> -X
    W -> -W
    Y -> -Y

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X or Z (inclusive) -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] | stabs.col_z[qubit]

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]


def H3(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hadamard gate (H3) rotation to stabilizers and destabilizers.

    Y + X

    X -> iW = Y
    Z -> -Z
    W -> -iX
    Y -> X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_z[qubit]

    # X -> i
    # ----------
    # For Xs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # X += Z
        g.col_z[qubit] ^= g.col_x[qubit]

        for i in g.col_x[qubit]:
            g.row_z[i] ^= {qubit}


def H4(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hadamard gate (H6) rotation to stabilizers and destabilizers.

    Y - X

    X -> -iW = -Y
    Z -> -Z
    W -> iX
    Y -> -X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X or Z (exclusive) -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] ^ stabs.col_z[qubit]

    # X -> i
    # ----------
    # For Xs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # X += Z
        g.col_z[qubit] ^= g.col_x[qubit]

        for i in g.col_x[qubit]:
            g.row_z[i] ^= {qubit}


def H5(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hadamard gate (H2) rotation to stabilizers and destabilizers.

    Z + Y

    X -> -X
    Z -> iW = Y
    W -> -iZ
    Y -> Z
    """
    stabs = state.stabs

    # Change the sign appropriately
    # If X apply -1
    # If Z apply i

    # X -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit]

    # Z -> i
    # ----------
    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]

        for i in g.col_z[qubit]:
            g.row_x[i] ^= {qubit}


def H6(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a Hadamard gate (H5) rotation to stabilizers and destabilizers.

    Z - Y

    X -> -X
    Z -> -iW = -Y
    W -> iZ
    Y -> -Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X or Z (exclusive) -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] ^ stabs.col_z[qubit]

    # Z -> i
    # ----------
    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Update column
        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]

        for i in g.col_z[qubit]:
            g.row_x[i] ^= {qubit}


def F(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F1) about a stabilizer octahedron face to stabilizers  and destabilizers.

    X -> iW = Y
    Z -> X
    W -> -iZ
    Y -> Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # both X and Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] & stabs.col_z[qubit]

    # X -> i
    # ----------
    # For Xs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_x[i].discard(qubit)

        for i in xonly:
            g.row_z[i].add(qubit)

        # Remove only Z
        # Z -> X
        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]


def F2(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F2) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> -Z
    Z -> iW = Y
    W -> iX
    Y -> -X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X not Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] - stabs.col_z[qubit]

    # Z -> i
    # ----------
    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_z[i].discard(qubit)

        for i in zonly:
            g.row_x[i].add(qubit)

        # Remove only Z
        # X -> Z
        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # Z += X
        g.col_z[qubit] ^= g.col_x[qubit]


def F3(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F3) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> iW = Y
    Z -> -X
    W -> iZ
    Y -> -Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z not X -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_z[qubit] - stabs.col_x[qubit]

    # X -> i
    # ----------
    # For Xs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_x[i].discard(qubit)

        for i in xonly:
            g.row_z[i].add(qubit)

        # Remove only Z
        # Z -> X
        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]


def F4(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F4) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> Z
    Z -> -iW = -Y
    W -> iX
    Y -> -X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # Z not X -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_z[qubit] - stabs.col_x[qubit]

    # Z -> i
    # ----------
    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_z[i].discard(qubit)

        for i in zonly:
            g.row_x[i].add(qubit)

        # Remove only Z
        # X -> Z
        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # Z += X
        g.col_z[qubit] ^= g.col_x[qubit]


def Fdg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F1^{\dagger}) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> Z
    Z -> iW = Y
    W -> -iX
    Y -> X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X and Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] & stabs.col_z[qubit]

    # Z -> i
    # ----------
    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_z[i].discard(qubit)

        for i in zonly:
            g.row_x[i].add(qubit)

        # Remove only Z
        # X -> Z
        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # Z += X
        g.col_z[qubit] ^= g.col_x[qubit]


def F2dg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F2^{\dagger}) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> -iW = -Y
    Z -> -X
    W -> -iZ
    Y -> Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X or Z (inclusive) -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] | stabs.col_z[qubit]

    # X -> i
    # ----------
    # For Xs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_x[i].discard(qubit)

        for i in xonly:
            g.row_z[i].add(qubit)

        # Remove only Z
        # Z -> X
        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]


def F3dg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F3^{\dagger}) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> -Z
    Z -> -iW = -Y
    W -> -iX
    Y -> X

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X or Z (inclusive) -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] | stabs.col_z[qubit]

    # Z -> i
    # ----------
    # For Zs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_z[qubit]

    # Generators only in Z column
    gens_only_z = stabs.col_z[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_z

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_z[i].discard(qubit)

        for i in zonly:
            g.row_x[i].add(qubit)

        # Remove only Z
        # X -> Z
        for i in xonly:
            g.row_x[i].discard(qubit)
            g.row_z[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # Z += X
        g.col_z[qubit] ^= g.col_x[qubit]


def F4dg(state: SparseSim, qubit: int, **params: Any) -> None:
    r"""Applies a rotation (F4^{\dagger}) about a stabilizer octahedron face to stabilizers and destabilizers.

    X -> -iW = -Y
    Z -> X
    W -> iZ
    Y -> -Z

    state (SparseSim): Instance representing the stabilizer state.
    qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    stabs = state.stabs

    # Change the sign appropriately

    # X not Z -> -1
    # ----------
    stabs.signs_minus ^= stabs.col_x[qubit] - stabs.col_z[qubit]

    # X -> i
    # ----------
    # For Xs in the qubit column we want to add i to the signs...

    # Deal with the i's ...

    # Generators common to both
    gens_common = stabs.signs_i & stabs.col_x[qubit]

    # Generators only in Z column
    gens_only_x = stabs.col_x[qubit] - stabs.signs_i

    # Generators that are common => i*i = -1
    # => Update the minus signs
    stabs.signs_minus ^= gens_common

    # Remove them from i's
    stabs.signs_i -= gens_common

    # Generators that are only in Z can just be added => i*1 = i
    stabs.signs_i |= gens_only_x

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gens:
        # Swap X and Z for rows
        xonly = g.col_x[qubit] - g.col_z[qubit]

        zonly = g.col_z[qubit] - g.col_x[qubit]

        xzshared = g.col_x[qubit] & g.col_z[qubit]

        for i in xzshared:
            g.row_x[i].discard(qubit)

        for i in xonly:
            g.row_z[i].add(qubit)

        # Remove only Z
        # Z -> X
        for i in zonly:
            g.row_z[i].discard(qubit)
            g.row_x[i].add(qubit)

        # Swap X and Z for cols
        g.col_x[qubit], g.col_z[qubit] = g.col_z[qubit], g.col_x[qubit]

        # X += Z
        g.col_x[qubit] ^= g.col_z[qubit]
