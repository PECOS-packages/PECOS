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

import numpy as np

from pecos.simulators.sparsesim.cmd_one_qubit import H5, H
from pecos.simulators.sparsesim.state import SparseSim


def meas_x(
    state: SparseSim,
    qubit: int,
    *,
    forced_outcome: int = -1,
    collapse: bool = True,
    **params,
) -> int:
    """Measurement in the X basis.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome (int):  Integer that will be outputted by the measurement if the measurement is
            non-deterministic. If equal to -1, however, the outcome will be uniformly chosen from {0, 1}.
        collapse (bool): Whether state should be collapsed.

    Returns:
        Measurement outcome (0 or 1).

    """
    H(state, qubit)

    meas_outcome = meas_z(
        state,
        qubit,
        forced_outcome=forced_outcome,
        collapse=collapse,
    )

    H(state, qubit)

    return meas_outcome


def meas_y(
    state: SparseSim,
    qubit: int,
    *,
    forced_outcome: int = -1,
    collapse: bool = True,
    **params,
) -> int:
    """Measurement in the Y basis.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome (int):  Integer that will be outputted by the measurement if the measurement is
            non-deterministic. If equal to -1, however, the outcome will be uniformly chosen from {0, 1}.
        collapse (bool): Whether to collapse the state if measurement is not already determined.

    Returns:
        Measurement outcome (0 or 1).

    """
    H5(state, qubit)

    meas_outcome = meas_z(
        state,
        qubit,
        forced_outcome=forced_outcome,
        collapse=collapse,
    )

    H5(state, qubit)

    return meas_outcome


def meas_z(
    state: SparseSim,
    qubit: int,
    *,
    forced_outcome: int = -1,
    collapse: bool = True,
    **params,
) -> int:
    """Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome (int):  Integer that will be outputted by the measurement if the measurement is
            non-deterministic. If equal to -1, however, the outcome will be uniformly chosen from {0, 1}.
        collapse (bool): Whether to collapse the state if measurement is not already determined.

    Returns:
        Measurement outcome (0 or 1).

    """
    # Determine if any stabilizer gens anti-commute with the Z measurement on the qubit
    # => Get the stabilizer generators that have Xs on the qubit we are measuring.

    # Choose an anti-commuting stabilizer to replace.

    stabs = state.stabs
    destabs = state.destabs

    anticom_stabs_col = stabs.col_x[qubit]
    anticom_destabs_col = destabs.col_x[qubit]

    if len(anticom_stabs_col) == 0:  # No anti-commuting stabilizer => determined sign
        stabs_row_x = stabs.row_x
        stabs_row_z = stabs.row_z

        num_minuses = len(anticom_destabs_col & stabs.signs_minus)
        num_is = len(anticom_destabs_col & stabs.signs_i)

        # Sign correction due ZX -> -XZ
        cumulative_x = set()
        for row in anticom_destabs_col:
            num_minuses += len(stabs_row_z[row] & cumulative_x)

            # Update the row sum Paulis
            cumulative_x ^= stabs_row_x[row]

        if num_is % 4:  # Can only be 0 or 2
            num_minuses += 1

        meas_outcome = num_minuses % 2

    else:  # There is at least one anti-commuting stabilizer. => indetermined sign
        if collapse:
            return nondeterministic_meas(
                state,
                qubit,
                anticom_stabs_col,
                anticom_destabs_col,
                forced_outcome,
            )

        else:
            if forced_outcome is not None:
                if forced_outcome in {0, 1}:
                    meas_outcome = forced_outcome
                else:
                    raise Exception(
                        "forced_outcome can only be 0 or 1 and not %s" % forced_outcome,
                    )
            else:
                meas_outcome = np.random.randint(2)

    return meas_outcome


def nondeterministic_meas(
    state: SparseSim,
    qubit: int,
    anticom_stabs_col: set[int],
    anticom_destabs_col: set[int],
    forced_outcome: int,
) -> int:
    """Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        anticom_stabs_col (Set[int]):
        anticom_destabs_col (Set[int]):
        forced_outcome (int):  Integer that will be outputted by the measurement if the measurement is
            non-deterministic. If equal to -1, however, the outcome will be uniformly chosen from {0, 1}.

    Returns:
        Measurement outcome (0 or 1).

    """
    # Removing dots
    stabs_row_x = state.stabs.row_x
    stabs_row_z = state.stabs.row_z
    destabs_row_x = state.destabs.row_x
    destabs_row_z = state.destabs.row_z
    stabs_col_x = state.stabs.col_x
    stabs_col_z = state.stabs.col_z
    destabs_col_x = state.destabs.col_x
    destabs_col_z = state.destabs.col_z

    anticom_stabs_col = set(
        anticom_stabs_col,
    )  # Stabilizers that anti-commute with the measurement
    anticom_destabs_col = set(anticom_destabs_col)  # Destabilizers that anti-commute

    smallest_wt = 2 * state.num_qubits + 2
    removed_id = None

    for stab_id in anticom_stabs_col:
        if len(stabs_row_x[stab_id]) + len(stabs_row_z[stab_id]) < smallest_wt:
            smallest_wt = len(stabs_row_x[stab_id]) + len(stabs_row_z[stab_id])
            removed_id = stab_id

    anticom_stabs_col.discard(removed_id)

    removed_row_x = set(stabs_row_x[removed_id])
    removed_row_z = set(stabs_row_z[removed_id])

    # -----------------------------------------------
    # Signs
    # -----------------------------------------------
    if removed_id in state.stabs.signs_minus:
        state.stabs.signs_minus ^= anticom_stabs_col

    if removed_id in state.stabs.signs_i:
        # Remove i from the removed stabilizer.
        state.stabs.signs_i.discard(removed_id)

        # Generators common to both
        gens_common = state.stabs.signs_i & anticom_stabs_col

        # Generators only in Z column
        gens_only_stabs = anticom_stabs_col - state.stabs.signs_i

        # Generators that are common => i*i = -1
        state.stabs.signs_minus ^= gens_common

        # Remove them from i's
        state.stabs.signs_i -= gens_common

        # Generators that are only in Z can just be added => i*1 = i
        state.stabs.signs_i |= gens_only_stabs

    # ----------------------------------------------------------------
    # Multiply anti-commuting stabs with removed stab.
    # ----------------------------------------------------------------
    for gen in anticom_stabs_col:
        # ZX -> -XZ sign correction
        num_minuses = len(removed_row_z & stabs_row_x[gen])
        if num_minuses % 2:  # An overall minus occurred when multiply by removed row.
            state.stabs.signs_minus ^= {gen}

        # row sum stabilizers
        stabs_row_x[gen] ^= removed_row_x
        stabs_row_z[gen] ^= removed_row_z

        # As we go through this wont we modify removed_row?
        # Need to at least not row sum on it.

    for q in removed_row_x:
        stabs_col_x[q] ^= anticom_stabs_col

    for q in removed_row_z:
        stabs_col_z[q] ^= anticom_stabs_col

    # ---------------------------------------------------------------------
    # Replace the removed stabilizer with the measured stabilizer.
    # ---------------------------------------------------------------------

    # Col update

    # Remove stabilizer
    for q in stabs_row_x[removed_id]:
        stabs_col_x[q].discard(removed_id)

    for q in stabs_row_z[removed_id]:
        stabs_col_z[q].discard(removed_id)

    # Remove replaced stabilizer with the measured stabilizer
    stabs_col_z[qubit].add(removed_id)

    # Row update
    stabs_row_x[removed_id].clear()
    stabs_row_z[removed_id] = {qubit}

    # ---------------------------------------------------------------------
    # Multiply all other anticommuting destabilizers with the removed one.
    # ---------------------------------------------------------------------

    # Anything that anticommutes with the new stabilizer gets multiplied by the new desabilizer, which is the
    # removed stabilizer.

    # Clear removed destabilizer
    for q in state.destabs.row_x[removed_id]:
        destabs_col_x[q].discard(removed_id)

    for q in state.destabs.row_z[removed_id]:
        destabs_col_z[q].discard(removed_id)

    # Add in/Multiply by the new destabilizer
    # This makes all destabilizers commute with the new stabilizer.

    anticom_destabs_col.discard(removed_id)

    for q in removed_row_x:
        destabs_col_x[q].add(removed_id)
        destabs_col_x[q] ^= anticom_destabs_col

    for q in removed_row_z:
        destabs_col_z[q].add(removed_id)
        destabs_col_z[q] ^= anticom_destabs_col

    for row in anticom_destabs_col:
        destabs_row_x[row] ^= removed_row_x
        destabs_row_z[row] ^= removed_row_z

    # The new destabilizer is the removed stabilizer
    destabs_row_x[removed_id] = removed_row_x
    destabs_row_z[removed_id] = removed_row_z

    # ---------------------------------------------------------------------
    # Measurements
    # ---------------------------------------------------------------------

    """
    if forced_outcome is not None:

        if forced_outcome == 0 or forced_outcome == 1:
            meas_outcome = forced_outcome
        else:
            raise Exception('forced_outcome can only be 0 or 1 and not %s' % forced_outcome)
    else:
        meas_outcome = np.random.randint(2)
    """

    meas_outcome = forced_outcome if forced_outcome > -1 else np.random.randint(2)

    # Use the random outcome as the sign of the replaced stabilizer
    if meas_outcome:
        state.stabs.signs_minus.add(removed_id)
    else:
        state.stabs.signs_minus.discard(removed_id)

    return meas_outcome


def force_output(
    state: SparseSim,
    qubit: int,
    forced_output: int = -1,
    **params: Any,
) -> int:
    """Outputs value.

    Used for error generators to generate outputs when replacing measurements.

    Args:
    ----
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_output (int): Integer that will be outputted.

    Returns:
        Measurement outcome that is force to be a particular value.

    """
    return forced_output
