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

"""Functions:

find_logical_signs
logical_flip
"""


def find_stab(state, xs: set[int], zs: set[int]):
    """Find a stabilizer in the stabilizer group.

    Args:
    ----
        state:
        xs:
        zs:

    Returns:
    -------

    """
    stabs = state.stabs
    destabs = state.destabs

    # Find the destabilizer generators that anticommute with the stabilizer indicated by xs and zs.

    # First the destabilizer generators that could *possibly* anticommute:
    possible_antidestabs = set()
    for q in xs:
        possible_antidestabs.update(destabs.col_z[q])

    for q in zs:
        possible_antidestabs.update(destabs.col_x[q])

    # Now we will confirm if they anticommute or not.
    antidestabs = set()
    for d in possible_antidestabs:
        if (len(xs & destabs.row_z[d]) + len(zs & destabs.row_x[d])) % 2 == 1:
            # They anticommute an odd number of times.
            antidestabs.add(d)

    # Now we will confirm that the supplied stabilizer is actually in the stabilizer group.
    confirm_xs = set()
    confirm_zs = set()
    for d in antidestabs:
        confirm_xs ^= stabs.row_x[d]
        confirm_zs ^= stabs.row_z[d]

    found = confirm_xs == xs and confirm_zs == zs

    return found, antidestabs


def refactor(state, xs, zs, choose=None, prefer=None, protected=None):
    """Find the sign of the logical operator.

    Args:
    ----
        state:
        xs:
        zs:
        choose (None, int): Order of stabilizer ids to choose from.
        prefer (None, set): Stabilizer ids that we should choose from.
        protected (None, set): Stabilizer ids not to choose from.

    Returns:
    -------

    """
    stabs = state.stabs
    destabs = state.destabs

    # Determine if the pruposed stabilizer is in the stabilizer group
    found, gens = find_stab(state, xs, zs)

    new_stab = None

    if found:
        # Now update the generators so the supplied stabilizer is a stabilizer generator.

        available = gens - protected if protected else gens

        # Pick a stabilizer generator to become the requested stabilizer generator.
        if choose is None and prefer is None:
            new_stab = available.pop()
            gens.remove(new_stab)

        elif prefer is None:  # Choose indicates what order to be stab ids from.
            new_stab = sorted(available)[choose]
            gens.remove(new_stab)

        else:  # choose is not None and prefer is not None. =>
            for i in prefer:
                if i in available:
                    new_stab = i
                    gens.remove(i)
                    break
            else:
                if choose is not None:
                    new_stab = sorted(available)[choose]
                    gens.remove(new_stab)
                else:
                    new_stab = available.pop()
                    gens.remove(new_stab)

            # What if everything is protected........

        # Now for each stabilizer/destabilizer generator pair we need to do:
        # stab_new -> stab_new * stab
        # destab -> destab * destab_new

        # Stab update
        for g in gens:
            for q in stabs.row_x[g]:
                stabs.col_x[q] ^= {new_stab}

            for q in stabs.row_z[g]:
                stabs.col_z[q] ^= {new_stab}

            stabs.row_x[new_stab] ^= stabs.row_x[g]
            stabs.row_z[new_stab] ^= stabs.row_z[g]

        # Destab update
        for q in destabs.row_x[new_stab]:
            destabs.col_x[q] ^= gens

        for q in destabs.row_z[new_stab]:
            destabs.col_z[q] ^= gens

        for g in gens:
            destabs.row_x[g] ^= destabs.row_x[new_stab]
            destabs.row_z[g] ^= destabs.row_z[new_stab]

        # Sign update
        gen_i = gens & stabs.signs_i
        gen_minus = gens & stabs.signs_minus

        num_minus = len(gen_minus)
        if len(gen_i) % 4 > 1:  # i.e. -1 or -i
            num_minus += 1

        if len(gen_i) % 2:  # i.e i or -i
            if new_stab in stabs.signs_i:
                num_minus += 1

            stabs.signs_i ^= {new_stab}

        if num_minus % 2:
            stabs.signs_minus ^= {new_stab}

    return found, new_stab
