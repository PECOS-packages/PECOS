#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from .cmd_one_qubit import S, Sd


def cnot(state, qubits):
    """
    :param gens:
    :param qubits:
    :return:

    II -> II
    XI -> XX
    ZI -> ZI
    WI -> WX
    IX -> IX
    IZ -> ZZ
    IW -> ZW
    XX -> XI
    XZ -> WW
    XW -> WZ
    ZX -> ZX
    ZZ -> IZ
    ZW -> IW
    WX -> WI
    WZ -> XW
    WW -> XZ

    II -> II
    XI -> XX
    ZI -> ZI
    YI -> YX
    IX -> IX
    IZ -> ZZ
    IY -> ZY
    XX -> XI
    XZ -> -YY
    XY -> YZ
    ZX -> ZX
    ZZ -> IZ
    ZY -> IY
    YX -> YI
    YZ -> XY
    YY -> -XZ
    """

    qubit1, qubit2 = qubits

    set_q1 = {qubit1}
    set_q2 = {qubit2}

    # Update Paulis
    # -------------------------------------------------------------------
    for g in state.gen_list:
        # X2 += X1 for rows
        for i in g.col_x[qubit1]:
            g.row_x[i].symmetric_difference_update(set_q2)

        # Z1 += Z2 for rows
        for i in g.col_z[qubit2]:
            g.row_z[i].symmetric_difference_update(set_q1)

        # Update the X column
        # X2 += X1
        g.col_x[qubit2] ^= g.col_x[qubit1]

        # Update the Z column
        # Z1 += Z2
        g.col_z[qubit1] ^= g.col_z[qubit2]


def CZ(state, qubits):
    """
    Applies a Controlled-X gate (CNOT) rotation to generators

    This version is best for a large number of qubits (aboa ut >= 150)

    II -> II
    XI -> XZ
    ZI -> ZI
    WI -> WZ
    IX -> ZX
    IZ -> IZ
    IW -> ZW
    XX -> -WW
    XZ -> XI
    XW -> -WX
    ZX -> IX
    ZZ -> ZZ
    ZW -> IW
    WX -> -XW
    WZ -> WI
    WW -> -XX

    II -> II
    XI -> XZ
    ZI -> ZI
    YI -> YZ
    IX -> ZX
    IZ -> IZ
    IY -> ZY
    XX -> YY
    XZ -> XI
    XY -> -YX
    ZX -> IX
    ZZ -> ZZ
    ZY -> IY
    YX -> -XY
    YZ -> YI
    YY -> XX
    """

    qubit1, qubit2 = qubits

    stabs = state.stabs

    # Change the sign appropriately
    stabs.signs_minus ^= stabs.col_x[qubit1] & stabs.col_x[qubit2]

    # Update Paulis
    # -------------------------------------------------------------------
    for gens in state.gen_list:

        old_z1_col = set(gens.col_z[qubit1])
        old_z2_col = set(gens.col_z[qubit2])

        # Update columns

        # Z1 += X2
        gens.col_z[qubit1] ^= gens.col_x[qubit2]

        # Z2 += X1
        gens.col_z[qubit2] ^= gens.col_x[qubit1]

        # Update rows

        # Z gens added
        z1_added = gens.col_z[qubit1] - old_z1_col
        z2_added = gens.col_z[qubit2] - old_z2_col

        for i in z1_added:
            gens.row_z[i].add(qubit1)

        for i in z2_added:
            gens.row_z[i].add(qubit2)

        del z1_added
        del z2_added

        # Z gens removed
        z1_removed = old_z1_col - gens.col_z[qubit1]
        z2_removed = old_z2_col - gens.col_z[qubit2]

        for i in z1_removed:
            gens.row_z[i].discard(qubit1)

        for i in z2_removed:
            gens.row_z[i].discard(qubit2)


def CY(state, qubits):
    """
    Applies a Controlled-Y gate (
    """

    _, qubit2 = qubits

    S(state, qubit2)
    cnot(state, qubits)
    Sd(state, qubit2)


def SWAP(state, qubits):  # qubit1 => control, qubit2 => target
    """
    Applies a SWAP gate (CNOT) to the generators

    II -> II
    XI -> IX
    ZI -> IZ
    WI -> IW
    IX -> XI
    IZ -> ZI
    IW -> WI
    XX -> XX
    XZ -> ZX
    XW -> WX
    ZX -> XZ
    ZZ -> ZZ
    ZW -> WZ
    WX -> WX
    WZ -> ZW
    WW -> WW
    """

    qubit1, qubit2 = qubits

    # Update Paulis
    # -------------------------------------------------------------------
    for gens in state.gen_list:

        # Update rows

        # Swap the Xs of qubit1 and qubit2 for rows

        # set of gens with Xs on qubit1 but not qubit2
        xin1 = gens.col_x[qubit1] - gens.col_x[qubit2]

        # set of gens with Xs on qubit2 but not qubit1
        xin2 = gens.col_x[qubit2] - gens.col_x[qubit1]

        for i in xin1:
            gens.row_x[i].discard(qubit1)
            gens.row_x[i].add(qubit2)

        for i in xin2:
            gens.row_x[i].discard(qubit2)
            gens.row_x[i].add(qubit1)

        del xin1
        del xin2

        # Swap the Zs of qubit1 and qubit2 for cols

        # set of gens with Zs on qubit1 but not qubit2
        zin1 = gens.col_z[qubit1] - gens.col_z[qubit2]

        # set of gens with Zs on qubit2 but not qubit1
        zin2 = gens.col_z[qubit2] - gens.col_z[qubit1]

        for i in zin1:
            gens.row_z[i].discard(qubit1)
            gens.row_z[i].add(qubit2)

        for i in zin2:
            gens.row_z[i].discard(qubit2)
            gens.row_z[i].add(qubit1)

        del zin1
        del zin2

        # Update columns

        # Swap the Xs of qubit1 and qubit2 for cols
        gens.col_x[qubit1], gens.col_x[qubit2] = gens.col_x[qubit2], gens.col_x[qubit1]

        # Swap the Zs of qubit1 and qubit2 for cols
        gens.col_z[qubit1], gens.col_z[qubit2] = gens.col_z[qubit2], gens.col_z[qubit1]


def G2(state, qubits):
    """
    Applies a CZ.H(1).H(2).CZ to the generators

    This gate has a Schmidt rank of 4

    II -> II
    XI -> IX
    ZI -> XZ
    WI -> XW
    IX -> XI
    IZ -> ZX
    IW -> WX
    XX -> XX
    XZ -> ZI
    XW -> WI
    ZX -> IZ
    ZZ -> -WW
    ZW -> -ZW
    WX -> IW
    WZ -> -WZ
    WW -> ZZ
    """

    qubit1, qubit2 = qubits

    stabs = state.stabs

    # Change the sign appropriately
    # sign += Z1*Z2
    stabs.signs_minus ^= stabs.col_z[qubit1] & stabs.col_z[qubit2]

    # Update Paulis
    # -------------------------------------------------------------------
    for gens in state.gen_list:

        # set of gens with Zs on qubit1 but not qubit2
        zin1 = gens.col_z[qubit1] - gens.col_z[qubit2]

        # set of gens with Zs on qubit2 but not qubit1
        zin2 = gens.col_z[qubit2] - gens.col_z[qubit1]

        old_x1_col = set(gens.col_x[qubit1])
        old_x2_col = set(gens.col_x[qubit2])

        # Update columns
        # ---------------------

        # Swap X1 and X2 columns
        # Swap the Xs of qubit1 and qubit2 for cols
        gens.col_x[qubit1], gens.col_x[qubit2] = gens.col_x[qubit2], gens.col_x[qubit1]

        # X1 += Z1
        gens.col_x[qubit1] ^= gens.col_z[qubit1]

        # X2 += Z2
        gens.col_x[qubit2] ^= gens.col_z[qubit2]

        # Swap Z1 and Z2
        # Swap the Zs of qubit1 and qubit2 for cols
        gens.col_z[qubit1], gens.col_z[qubit2] = gens.col_z[qubit2], gens.col_z[qubit1]

        # Update the rows
        # ---------------------

        # Swap the Zs of qubit1 and qubit2 for cols

        for i in zin1:
            gens.row_z[i].discard(qubit1)
            gens.row_z[i].add(qubit2)

        for i in zin2:
            gens.row_z[i].discard(qubit2)
            gens.row_z[i].add(qubit1)

        del zin1
        del zin2

        # Z gens added
        x1_added = gens.col_x[qubit1] - old_x1_col
        x2_added = gens.col_x[qubit2] - old_x2_col

        for i in x1_added:
            gens.row_x[i].add(qubit1)

        for i in x2_added:
            gens.row_x[i].add(qubit2)

        del x1_added
        del x2_added

        # Z gens removed
        x1_removed = old_x1_col - gens.col_x[qubit1]
        x2_removed = old_x2_col - gens.col_x[qubit2]

        for i in x1_removed:
            gens.row_x[i].discard(qubit1)

        for i in x2_removed:
            gens.row_x[i].discard(qubit2)

        del x1_removed
        del x2_removed


def II(state, qubits, **kwargs):
    pass
