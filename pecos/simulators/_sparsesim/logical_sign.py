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

"""
Functions:

find_logical_signs
logical_flip
"""
from typing import Optional
from ...circuits import QuantumCircuit


def find_logical_signs(state,
                       logical_circuit: QuantumCircuit,
                       delogical_circuit: Optional[QuantumCircuit] = None
                       ) -> int:
    """
    Find the sign of the logical operator.

    Args:
        state:
        logical_circuit:

    Returns:

    """

    if len(logical_circuit) != 1:
        raise Exception('Logical operators are expected to only have one tick.')

    stabs = state.stabs
    destabs = state.destabs

    logical_xs = set([])
    logical_zs = set([])

    for symbol, gate_locations, _ in logical_circuit.items():

        if symbol == 'X':
            logical_xs.update(gate_locations)
        elif symbol == 'Z':
            logical_zs.update(gate_locations)
        elif symbol == 'Y':
            logical_xs.update(gate_locations)
            logical_zs.update(gate_locations)
        else:
            raise Exception('Can not currently handle logical operator with operator "%s"!' % symbol)

    if delogical_circuit:  # Check the relationship between logical operator and delogical operator.

        if len(delogical_circuit) != 1:
            raise Exception('Delogical operators are expected to only have one tick.')

        delogical_xs = set([])
        delogical_zs = set([])
        
        for symbol, gate_locations, _ in delogical_circuit.items():

            if symbol == 'X':
                delogical_xs.update(gate_locations)
            elif symbol == 'Z':
                delogical_zs.update(gate_locations)
            elif symbol == 'Y':
                delogical_xs.update(gate_locations)
                delogical_zs.update(gate_locations)
            else:
                raise Exception('Can not currently handle logical operator with operator "%s"!' % symbol)

        # Make sure the logical and delogical anti-commute

        anticom_x = len(logical_xs & delogical_zs) % 2  # Number of common elements modulo 2
        anticom_z = len(logical_zs & delogical_xs) % 2  # Number of common elements modulo 2

        if not ((anticom_x + anticom_z) % 2):
            print('logical Xs: %s logical Zs: %s' % (logical_xs, logical_zs))
            print('delogical Xs: %s delogical Zs: %s' % (delogical_xs, delogical_zs))
            raise Exception("Logical and delogical operators supplied do not anti-commute!")

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
        #    print('stab ... ', stab)

        print(('Logical op: xs - %s and zs - %s' % (logical_xs, logical_zs)))
        raise Exception('Failure due to not finding logical op! x... %s z... %s' %
                        (str(test_x ^ logical_xs), str(test_z ^ logical_zs)))

    # Get the sign of the logical operator
    # --------------------------

    # First, the minus sign
    logical_minus = len(build_stabs & stabs.signs_minus)

    # Second, the number of imaginary numbers
    logical_i = len(build_stabs & stabs.signs_i)

    # Translate the Ws to Ys... W = -i(iW) = -iY => For each Y add another -1 and +i.
    logical_ws = logical_xs & logical_zs
    num_ys = len(logical_ws)

    logical_minus += num_ys
    logical_i += num_ys

    # Do (-1)^even = 1 -> 0, (-1)^odd = -1 -> 1
    logical_minus %= 2

    # Reinterpret number of is
    logical_i %= 4

    # num_is %4 = 0 => +1 => logical_i = 0, logical_minus += 0
    # num_is %4 = 1 => +i => logical_i = 1, logical_minus += 0

    if logical_i == 2:  # num_is %4 = 2 => -1 => logical_i = 0, logical_minus += 1
        logical_i = 0
        logical_minus += 1
    elif logical_i == 3:  # num_is %4 = 3 => -i => logical_i = 1, logical_minus += 1
        logical_i = 1
        logical_minus += 1

    if logical_i != 0:
        raise Exception('Logical operator has an imaginary sign... Not allowed if logical state is stabilized '
                        'by logical op!')

    return logical_minus
