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


def circ2set(circuit):
    qudit_xs = set()
    qudit_zs = set()
    for gate, qubits in circuit:
        if gate == "X":
            qudit_xs = qubits
        elif gate == "Z":
            qudit_zs = qubits
        else:
            raise Exception('Operator "%s" not handled for logical ops!' % gate)

    return qudit_xs, qudit_zs


def op_commutes(stab_xs, stab_zs, commute_with):
    # Does the stabilizer anti-commute with any of the stabilizers in the stabilizer state?

    anticom_stabs = set()

    for q in stab_xs:
        anticom_stabs ^= commute_with.col_z[q]  # stabilizers that anti-commute with q

    for q in stab_zs:
        anticom_stabs ^= commute_with.col_x[q]  # stabilizers that anti-commute with q

    return not len(anticom_stabs)


def find_stab(state, stab_xs, stab_zs):
    """Find the sign of the logical operator.

    Args:
    ----
        state:
        stab_xs:
        stab_zs:

    Returns:
    -------

    """

    if len(stab_xs) == 0 and len(stab_zs) == 0:
        return True

    stab_xs = set(stab_xs)
    stab_zs = set(stab_zs)

    stabs = state.stabs

    # Attempt to build stabilizers from the stabilizers in the stabilizer state.

    built_up_xs = set()
    built_up_zs = set()
    for q in stab_xs:
        for stab_id in state.destabs.col_z[q]:
            built_up_xs ^= stabs.row_x[stab_id]
            built_up_zs ^= stabs.row_z[stab_id]

    for q in stab_zs:
        for stab_id in state.destabs.col_x[q]:
            built_up_xs ^= stabs.row_x[stab_id]
            built_up_zs ^= stabs.row_z[stab_id]

    # Compare with logical operator
    built_up_xs ^= stab_xs
    built_up_zs ^= stab_zs

    # Whether a stabilizer has been found
    return not (len(built_up_xs) != 0 or len(built_up_zs) != 0)


def remove_stab(state, stab_xs, stab_zs, destab_xs, destab_zs):
    # make sure stabs and destabs anti-commute
    # ----------------------------------------
    if (len(stab_xs & destab_zs) + len(stab_zs & destab_xs)) % 2 == 0:
        msg = "Stabilizer and destabilizers do not anti-commute."
        raise Exception(msg)

    logical_xs = stab_xs
    logical_zs = stab_zs
    delogical_xs = destab_xs
    delogical_zs = destab_zs

    stabs = state.stabs
    destabs = state.destabs

    anticom_x = len(logical_xs & delogical_zs) % 2
    anticom_z = len(logical_zs & delogical_xs) % 2

    if not (anticom_x + anticom_z % 2):
        msg = "Logical and delogical supplied do not anti-commute!"
        raise Exception(msg)

    # We want the supplied logical operator to be in the stabilizer group and
    #  the supplied delogical to not be in the stabilizers (we want it to end up being the logical op's destabilizer)

    # The following two function calls are wasteful because we will need some of what they discover... such as all the
    #  stabilizers that have destabilizers that anti-commute with the logical operator...
    #  But it is assumed that the user is not calling this function that often... so we can be wasteful...

    # Check logical is a stabilizer (we want to remove it from the stabilizers)
    if is_not_stabilizer(state, stab_xs, stab_zs):
        msg = "Supplied stabilizer is NOT in the current stabilizer group!"
        raise Exception(msg)

    # Check delogical is not a stabilizer
    if not is_not_stabilizer(state, destab_xs, destab_zs):
        msg = "Supplied delogical IS in the current stabilizer group!"
        raise Exception(msg)

    # Now that everything seems fine so far...
    # We need to remove the logical operator and insure the delogical partner is also a logical operator and not an
    # error

    # Step 1 - Get the logical operator as a stabilizer... well we are going to remove it so we actually don't need to
    #   group multiply to get it back... rather we just need to determine which stabilizer we will remove and update the
    #   destabilizers of the other stabilizers appropriately.

    #   We do need to determine the logical operator's sign so that we can remove it in step 4.

    # Step 2 - For the stabilizers that had destabilizers that anti-commuted with the logical operator but are not being
    #   removed, update their destabilizers.

    # Step 3 - Remove the stabilizer we have singled out to remove and its destabilizer

    # Step 4 - Find any stabilizers that anti-commute with the delogical operator and multiply those with the logical
    #   operator. We would normally have to update the logical operator's destabilizer, but we are going to throw these
    #   away anyway.

    # --------------------------------
    # Step 1 - Identify destabilizers that anti-commute, a stabilizer to remove, and the total logical sign.

    build_stabs = set()

    for q in logical_xs:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Use |= or ^= ??????????????????
        build_stabs ^= destabs.col_z[
            q
        ]  # These should point to all the stabilizers that will combine to give X on
        # qubit q

    for q in logical_zs:
        build_stabs ^= destabs.col_x[
            q
        ]  # These should point to all the stabilizers that will combine to give Z on
        # qubit q

    # Stabilizer to remove
    removed_id = build_stabs.pop()

    # Get sign of the logical operator
    logical_minus = len(build_stabs & stabs.signs_minus) % 2

    if removed_id in stabs.signs_minus:
        logical_minus += 1

    logical_i = len(build_stabs & stabs.signs_i)

    if removed_id in stabs.signs_i:
        logical_i += 1

    logical_i %= 4

    if logical_i == 2:
        logical_i = 0
        logical_minus += 1
    elif logical_i == 3:
        logical_i = 1
        logical_minus += 1

    logical_minus %= 2

    # --------------------------------
    # Step 2 - Update destabilizers

    # Row
    # ---
    for gen in build_stabs:
        destabs.row_x[gen] ^= destabs.row_x[removed_id]
        destabs.row_z[gen] ^= destabs.row_z[removed_id]

    # Column
    # ---
    for q in destabs.row_x[removed_id]:
        destabs.col_x[q] ^= build_stabs

    for q in destabs.row_z[removed_id]:
        destabs.col_z[q] ^= build_stabs

    # --------------------------------
    # Step 3 - Remove the stabilizer and its destabilizer

    # Col update

    # Remove stabilizer from column
    for q in stabs.row_x[removed_id]:
        stabs.col_x[q].discard(removed_id)

    for q in stabs.row_z[removed_id]:
        stabs.col_z[q].discard(removed_id)

    # Remove stabs from row
    stabs.row_x[removed_id].clear()
    stabs.row_z[removed_id].clear()

    # ----
    # Remove destabilizer from column
    for q in destabs.row_x[removed_id]:
        destabs.col_x[q].discard(removed_id)

    for q in destabs.row_z[removed_id]:
        destabs.col_z[q].discard(removed_id)

    # Remove destabs from row
    destabs.row_x[removed_id].clear()
    destabs.row_z[removed_id].clear()

    # --------------------------------
    # Step 4 - Find stabilizers that anti-commute with the delogical. Then multiply these with the logical.

    delog_anticom = set()

    for s, qubits in enumerate(stabs.row_z):
        overlap = delogical_xs & qubits

        if len(overlap) % 2:  # Odd overlap
            delog_anticom.add(s)

    for s, qubits in enumerate(stabs.row_x):
        overlap = delogical_zs & qubits

        if len(overlap) % 2:  # Odd overlap
            delog_anticom.add(s)

    # Now do rowsum on these with the logical operator

    for gen in delog_anticom:
        stabs.row_x[gen] ^= logical_xs
        stabs.row_z[gen] ^= logical_zs

    for q in logical_xs:
        stabs.col_x[q] ^= delog_anticom

    for q in logical_zs:
        stabs.col_z[q] ^= delog_anticom

    # Update the sign on these
    if logical_minus:
        stabs.signs_minus ^= delog_anticom

    if logical_i:
        # Find two is (carry operation)
        two_is = stabs.signs_i & delog_anticom
        stabs.signs_minus ^= two_is

        stabs.signs_i ^= delog_anticom


def is_not_stabilizer(state, qubits_x, qubits_z):
    stabs = state.stabs
    destabs = state.destabs

    # ----- Anti-commutes with stabilizers? ----- #

    for qubits in stabs.row_z:
        overlap = qubits_x & qubits

        if len(overlap) % 2:  # Odd overlap
            return 1  # Zs do not commute

    for qubits in stabs.row_x:
        overlap = qubits_z & qubits

        if len(overlap) % 2:  # Odd overlap
            return 1  # Xs do not commute

    # ----- In the stabilizer group? ----- #

    build_stabs = set()

    for q in qubits_x:
        build_stabs ^= destabs.col_z[
            q
        ]  # These should point to all the stabilizers that will combine to give X on
        # qubit q

    for q in qubits_z:
        build_stabs ^= destabs.col_x[
            q
        ]  # These should point to all the stabilizers that will combine to give Z on
        # qubit q

    # Build up the X and Z Paulis
    build_x_paulis = set()
    build_z_paulis = set()
    for stab in build_stabs:
        build_x_paulis ^= stabs.row_x[stab]

    for stab in build_stabs:
        build_z_paulis ^= stabs.row_z[stab]

    # Now lets see if the Paulis we built and the Paulis in the supplied logical operator agree
    sym_diff_x = build_x_paulis ^ qubits_x

    sym_diff_z = build_z_paulis ^ qubits_z

    if len(sym_diff_x) > 0 or len(sym_diff_z) > 0:
        # Was not able to get supplied Paulis from the stabilizers
        # (But did commute)
        return 2  # Not in the stabilizers group

    else:  # Was able to get supplied Paulis from the stabilizers
        return 0  # In the stabilizer group
