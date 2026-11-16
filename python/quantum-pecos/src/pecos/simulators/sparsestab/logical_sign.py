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

"""Functions for logical sign computation.

find_logical_signs
logical_flip
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.simulators.sparsestab.phase import product_phase

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.simulators.sparsestab.state import SparseStabPy


def find_logical_signs(
    state: SparseStabPy,
    logical_circuit: QuantumCircuit,
    delogical_circuit: QuantumCircuit | None = None,
) -> int:
    """Find the sign of the logical operator.

    Args:
        state (SparseStabPy): The SparseStabPy state instance.
        logical_circuit (QuantumCircuit): The logical circuit to find the sign of.
        delogical_circuit (QuantumCircuit | None): Optional delogical circuit to check anti-commutation with.
    """
    if len(logical_circuit) != 1:
        msg = "Logical operators are expected to only have one tick."
        raise Exception(msg)

    stabs = state.stabs
    destabs = state.destabs

    logical_xs = set()
    logical_zs = set()
    logical_ys = set()

    for symbol, gate_locations, _ in logical_circuit.items():
        if symbol == "X":
            logical_xs.update(gate_locations)
        elif symbol == "Z":
            logical_zs.update(gate_locations)
        elif symbol == "Y":
            logical_xs.update(gate_locations)
            logical_zs.update(gate_locations)
            logical_ys.update(gate_locations)
        else:
            msg = f'Can not currently handle logical operator with operator "{symbol}"!'
            raise Exception(
                msg,
            )

    if delogical_circuit:  # Check the relationship between logical operator and delogical operator.
        if len(delogical_circuit) != 1:
            msg = "Delogical operators are expected to only have one tick."
            raise Exception(msg)

        delogical_xs = set()
        delogical_zs = set()

        for symbol, gate_locations, _ in delogical_circuit.items():
            if symbol == "X":
                delogical_xs.update(gate_locations)
            elif symbol == "Z":
                delogical_zs.update(gate_locations)
            elif symbol == "Y":
                delogical_xs.update(gate_locations)
                delogical_zs.update(gate_locations)
            else:
                msg = f'Can not currently handle logical operator with operator "{symbol}"!'
                raise Exception(
                    msg,
                )

        # Make sure the logical and delogical anti-commute

        anticom_x = len(logical_xs & delogical_zs) % 2  # Number of common elements modulo 2
        anticom_z = len(logical_zs & delogical_xs) % 2  # Number of common elements modulo 2

        if not ((anticom_x + anticom_z) % 2):
            print(f"logical Xs: {logical_xs} logical Zs: {logical_zs}")
            print(f"delogical Xs: {delogical_xs} delogical Zs: {delogical_zs}")
            msg = "Logical and delogical operators supplied do not anti-commute!"
            raise Exception(msg)

    # We want the supplied logical operator to be in the stabilizer group and
    #  the supplied delogical to not be in the stabilizers (we want it to end up being the logical op's destabilizer)

    # The following two function calls are wasteful because we will need some of what they discover... such as all the
    #  stabilizers that have destabilizers that anti-commute with the logical operator...
    #  But it is assumed that the user is not calling this function that often... so we can be wasteful...

    # Check logical is a stabilizer (we want to remove it from the stabilizers)

    # Find the anti-commuting destabilizers => stabilizers to give the logical operator
    # --------------------------
    build_stabs = set()

    for q in logical_xs:  # For qubits that have Xs in for the logical operator...
        build_stabs ^= destabs.col_z[q]  # Add in stabilizers that anti-commute for the logical operator's Xs

    for q in logical_zs:
        build_stabs ^= destabs.col_x[q]  # Add in stabilizers that anti-commute for the logical operator's Zs

    # If a stabilizer anticommutes an even number of times for the X and/or Z Paulis... it will not appear due to ^=

    # Confirm that the stabilizers chosen give the logical operator. If not... return with a failure = 1
    # --------------------------
    test_x = set()
    test_z = set()

    for stab in build_stabs:
        test_x ^= stabs.row_x[stab]
        test_z ^= stabs.row_z[stab]

    # Compare with logical operator
    test_x ^= logical_xs
    test_z ^= logical_zs

    if len(test_x) != 0 or len(test_z) != 0:
        # for stab in build_stabs:

        print(f"Logical op: xs - {logical_xs} and zs - {logical_zs}")
        msg = f"Failure due to not finding logical op! x... {test_x ^ logical_xs!s} z... {test_z ^ logical_zs!s}"
        raise Exception(msg)

    # Since Y = iW, converting the product from W to explicit Y subtracts
    # one power of i for every Y, including pairs that contribute a minus.
    phase = (product_phase(stabs, build_stabs) - len(logical_ys)) % 4
    if phase % 2:
        msg = "Final sign should not have an imaginary sign!"
        raise Exception(msg)

    return phase // 2
