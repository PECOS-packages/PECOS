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

from itertools import combinations, product
from ..circuits import QuantumCircuit
from .. import simulators


class VerifyStabilizers:
    """
    Used to define a stabilizer QECC.
    """

    def __init__(self):

        self.circ_sim = simulators.pySparseSim

        self.checks = []
        self.logical_zs = []
        self.logical_xs = []

        self.data_qubits = set([])
        self.ancilla_qubits = set([])
        self.circuit = None

        self.check_row_x = None
        self.check_row_z = None
        self.check_col_x = None
        self.check_col_z = None

        # stabilizer ids:
        self.data_gens = None
        self.ancilla_gens = None
        self.logical_gens = None

        # distance
        self.dist = None

        self.state = None

    def check(self, paulis, qubits):
        """

        Args:
            paulis(sequence of str):
            data_qubits(sequence of int):

        Returns: None

        """

        if not qubits:
            raise Exception('No qubit ids given.')

        check_string = 'check('+str(paulis)+', '+str(qubits)+')'

        if isinstance(paulis, str):
            if paulis not in ['X', 'x', 'Y', 'y', 'Z', 'z']:
                raise Exception('Paulis should be "X", "Y" or "Z"!')

            paulis_new = [paulis for _ in qubits]
            paulis = paulis_new

        if not isinstance(paulis, str) and len(paulis) != len(qubits):
            raise Exception('Number of Paulis and qubits do not match!!!')

        self.checks.append((paulis, qubits, check_string))
        self.data_qubits.update(qubits)

    def logicalz(self, paulis, qubits):
        """
        Used to define logical Z.

        Args:
            paulis:
            qubits:

        Returns:

        """

        if not qubits:
            raise Exception('No qubit ids given.')

        logical_string = 'check(' + str(paulis) + ', ' + str(qubits) + ')'

        if isinstance(paulis, str):
            if paulis not in ['X', 'x', 'Y', 'y', 'Z', 'z']:
                raise Exception('Paulis should be "X", "Y" or "Z"!')

            paulis_new = [paulis for _ in qubits]
            paulis = paulis_new

        if not isinstance(paulis, str) and len(paulis) != len(qubits):
            raise Exception('Number of Paulis and qubits do not match!!!')

        self.logical_zs.append((paulis, qubits, logical_string))

    def logicalx(self, paulis, qubits):
        """
        Used to define logical X.

        Args:
            paulis:
            qubits:

        Returns:

        """

        if not qubits:
            raise Exception('No qubit ids given.')

        logical_string = 'check(' + str(paulis) + ', ' + str(qubits) + ')'

        if isinstance(paulis, str):
            if paulis not in ['X', 'x', 'Y', 'y', 'Z', 'z']:
                raise Exception('Paulis should be "X", "Y" or "Z"!')

            paulis_new = [paulis for _ in qubits]
            paulis = paulis_new

        if not isinstance(paulis, str) and len(paulis) != len(qubits):
            raise Exception('Number of Paulis and qubits do not match!!!')

        self.logical_xs.append((paulis, qubits, logical_string))

    def num_logical_qubits(self):
        return len(self.data_qubits) - len(self.checks)

    def generators(self, print_y=True, verbose=True):
        """
        Evaluates the stabilizer generators that have been supplied via the `check` method.

        Args:
            print_y:
            verbose:

        Returns:

        """

        if self.circuit is None:
            raise Exception('Must compile circuits first!')

        state = self.state
        z, x, stab_strings, destab_strings = self.get_info(state, print_y=print_y, verbose=verbose)

        return z, x, stab_strings, destab_strings

    def _check_all_labels(self):
        """
        This checks to see that all the consecutive qubit ids have been used and none are missing.

        Returns:

        """

        qubit_labels = set()

        checks = self.checks
        for _, check in enumerate(checks):
            _, qs, _ = check
            qubit_labels.update(qs)

        largest_labels = max(qubit_labels)
        labels_should_have = set(range(largest_labels+1))

        dont_have = labels_should_have - qubit_labels

        if dont_have:
            raise Exception('Qubit ids missing: %s' % dont_have)

    def _check2rowcol(self):
        """
        Creates row and column matrices.

        Returns:

        """
        checks = self.checks

        num_checks = len(checks)
        row_x = [set() for _ in range(num_checks)]
        row_z = [set() for _ in range(num_checks)]
        col_x = [set() for _ in range(self.num_data_qubits)]
        col_z = [set() for _ in range(self.num_data_qubits)]

        for stab_id, check in enumerate(checks):
            ps, qs, _ = check

            for p, q in zip(ps, qs):

                if p == 'X' or p == 'x':
                    row_x[stab_id].add(q)
                    col_x[q].add(stab_id)

                elif p == 'Z' or p == 'z':
                    row_z[stab_id].add(q)
                    col_z[q].add(stab_id)

                elif p == 'Y' or p == 'y':
                    row_x[stab_id].add(q)
                    row_z[stab_id].add(q)
                    col_x[q].add(stab_id)
                    col_z[q].add(stab_id)

        self.check_row_x = row_x
        self.check_row_z = row_z
        self.check_col_x = col_x
        self.check_col_z = col_z

    def _check_commute(self):
        """
        Checks to see that all the stabilizer generators commute.

        Returns:
            Returns bool value if all the checks commute or not.
        """

        row_x = self.check_row_x
        row_z = self.check_row_z
        col_x = self.check_col_x
        col_z = self.check_col_z

        # print(row_x, row_z, col_x, col_z)

        for stab_id in range(len(self.checks)):

            anti_zs = set()
            for q in row_x[stab_id]:
                anti_zs ^= col_z[q]

            anti_xs = set()
            for q in row_z[stab_id]:
                anti_xs ^= col_x[q]

            anti = anti_xs ^ anti_zs
            anti.discard(stab_id)

            if anti:
                print('\nChecks anticommute!')
                print('\nCheck:')
                for s in anti:
                    # print('Xs-%s Zs-%s' % (row_x[s], row_z[s]))
                    print(self.checks[s][2])
                print('\nanticommutes with:')
                # print('Xs-%s Zs-%s' % (row_x[stab_id], row_z[stab_id]))
                print(self.checks[stab_id][2])

                raise Exception('Checks anticommute!')
        return True

    def compile(self):
        """
        Checks commutation relations and creates a circuit to measure the checks.

        Returns:

        """

        if self.circuit:
            raise Exception('Measurement encoding-circuit has already been compiled!')

        # Check the qubit ids
        self._check_all_labels()

        # Create row and column matrices.
        self._check2rowcol()

        # Checks that all the stabilizer generators (checks) commute.
        self._check_commute()

        # Create check circuits:
        # ----------------------
        ancilla_qubits = set([])
        qc = QuantumCircuit()

        ancilla_id = sorted(self.data_qubits)[-1]

        for (ps, qs, _) in self.checks:
            ancilla_id += 1
            ancilla_qubits.add(ancilla_id)
            qc.append('init |+>', {ancilla_id})

            for p, q in zip(ps, qs):
                symbol = None

                if p == 'X' or p == 'x':
                    symbol = 'CNOT'
                elif p == 'Z' or p == 'z':
                    symbol = 'CZ'
                elif p == 'Y' or p == 'y':
                    symbol = 'CY'

                qc.append(symbol, {(ancilla_id, q)})

            qc.append('measure X', {ancilla_id}, random_outcome=0)

        self.ancilla_qubits = ancilla_qubits

        self.circuit = qc

        # Run circuits
        # ------------
        # Separate the checks, logical stabilizers, and ancilla stabilizers.
        circuit = self.circuit
        state = simulators.pySparseSim(self.num_qubits)
        state.run_circuit(circuit)
        self.get_info(state, verbose=False)
        self.state = state

        self._verify_checks()
        self._check_logical_commute()

    def _check_logical_commute(self):

        logical_z_col_x = [set() for _ in range(self.num_data_qubits)]
        logical_z_col_z = [set() for _ in range(self.num_data_qubits)]
        logical_z_row_x = [set() for _ in range(len(self.logical_zs))]
        logical_z_row_z = [set() for _ in range(len(self.logical_zs))]

        logical_x_col_x = [set() for _ in range(self.num_data_qubits)]
        logical_x_col_z = [set() for _ in range(self.num_data_qubits)]
        logical_x_row_x = [set() for _ in range(len(self.logical_zs))]
        logical_x_row_z = [set() for _ in range(len(self.logical_zs))]

        for i, (ps, qs, _) in enumerate(self.logical_zs):
            for p, q in zip(ps, qs):
                if p == 'X' or p == 'Y':
                    logical_z_col_x[q].add(i)
                    logical_z_row_x[i].add(q)
                if p == 'Z' or p == 'Y':
                    logical_z_col_z[q].add(i)
                    logical_z_row_z[i].add(q)

        for i, (ps, qs, _) in enumerate(self.logical_xs):
            for p, q in zip(ps, qs):
                if p == 'X' or p == 'Y':
                    logical_x_col_x[q].add(i)
                    logical_x_row_x[i].add(q)
                if p == 'Z' or p == 'Y':
                    logical_x_col_z[q].add(i)
                    logical_x_row_z[i].add(q)

        for s in range(len(self.logical_zs)):
            anti_zs = set()
            for q in logical_z_row_x[s]:
                anti_zs ^= logical_z_col_z[q]

            anti_xs = set()
            for q in logical_z_row_z[s]:
                anti_xs ^= logical_z_col_x[q]

            anti = anti_xs ^ anti_zs
            anti.discard(s)

            if anti:
                print('\nLogical Zs anticommute!')
                print('\nLogical Zs:')
                for i in anti:
                    # print('Xs-%s Zs-%s' % (row_x[s], row_z[s]))
                    print(self.logical_zs[i][2])
                print('\nanticommutes with:')
                # print('Xs-%s Zs-%s' % (row_x[stab_id], row_z[stab_id]))
                print(self.logical_zs[s][2])

                raise Exception('Logical Zs anticommute!')

        for s in range(len(self.logical_xs)):
            anti_zs = set()
            for q in logical_x_row_x[s]:
                anti_zs ^= logical_x_col_z[q]

            anti_xs = set()
            for q in logical_x_row_z[s]:
                anti_xs ^= logical_x_col_x[q]

            anti = anti_xs ^ anti_zs
            anti.discard(s)

            if anti:
                print('\nLogical Xs anticommute!')
                print('\nLogical Xs:')
                for i in anti:
                    # print('Xs-%s Zs-%s' % (row_x[s], row_z[s]))
                    print(self.logical_xs[i][2])
                print('\nanticommutes with:')
                # print('Xs-%s Zs-%s' % (row_x[stab_id], row_z[stab_id]))
                print(self.logical_xs[s][2])

                raise Exception('Logical Xs anticommute!')

        # So far checked that all the logical Zs and logical Xs commute with themselves...
        # - Next check that they commute with the stabilizers...
        # - Then find if the there are anti-commuting pairs of logical Zs and Xs
        # - Then search for the logical operators and refactor...
        # This step might require switching logical Xs and Zs... Might be a bit complicated... as we can modify
        # "logical Xs" with destabilizers and swap those... So need to do a search through all stabilizers and
        # destabilizers and then determine if the required multiplication is valid with fix stabiliziers and whatever
        # has been fixed for the logical operators...

    def _verify_checks(self):

        # Stabilizers:
        checks = []

        for strings, qids, _ in self.checks:

            check_dict = {}
            for P, q in zip(strings, qids):
                if P == 'X':
                    qset = check_dict.setdefault('X', set())
                elif P == 'Z':
                    qset = check_dict.setdefault('Z', set())
                else:
                    qset = check_dict.setdefault('Y', set())
                qset.add(q)
            checks.append(check_dict)

        checks2 = []
        for i in range(len(self.checks)):

            xs = self.check_row_x[i]
            zs = self.check_row_z[i]

            stab_dict = {}

            if xs - zs:
                stab_dict['X'] = xs - zs

            if zs - xs:
                stab_dict['Z'] = zs - xs

            if xs & zs:
                stab_dict['Y'] = xs & zs

            checks2.append(stab_dict)

        if checks != checks2:
            print("WARNING: PECOS didn't refactor the stabilizers into the checks supplied!")

        return checks != checks2

    def eval(self, verbose=False):
        if self.circuit is None:
            self.compile()

        z, x, _, destab_strings = self.generators(verbose=verbose)

        if self.dist is None:
            self.distance(verbose=verbose)

        # Stabilizers:
        checks = []

        for strings, qids, _ in self.checks:

            check_dict = {}
            for P, q in zip(strings, qids):
                if P == 'X':
                    qset = check_dict.setdefault('X', set())
                elif P == 'Z':
                    qset = check_dict.setdefault('Z', set())
                else:
                    qset = check_dict.setdefault('Y', set())
                qset.add(q)
            checks.append(check_dict)

        # Destabilizers:
        destabs = []
        for i in self.data_gens:
            destab_dict = {}
            for j in range(len(destab_strings[i]) - self.num_ancilla_qubits):
                P = destab_strings[i][j]
                if P == 'X':
                    qset = destab_dict.setdefault('X', set())
                elif P == 'Z':
                    qset = destab_dict.setdefault('Z', set())
                elif P == 'Y':
                    qset = destab_dict.setdefault('Y', set())
                else:
                    continue
                qset.add(j-2)
            destabs.append(destab_dict)

        output_dict = {
            'num_datas': self.num_data_qubits,
            'num_logical_qubits': self.num_logical_qubits(),
            'distance': self.dist,
            '[[n, k, d]]': "[[%s, %s, %s]]" % (self.num_data_qubits, self.num_logical_qubits(), self.dist),
            'checks': checks,
            'destabilizers': destabs,
            'logical_xs': x,
            'logical_zs': z,
        }

        return output_dict

    @property
    def num_data_qubits(self):
        return len(self.data_qubits)

    @property
    def num_ancilla_qubits(self):
        return len(self.ancilla_qubits)

    @property
    def num_qubits(self):
        return len(self.data_qubits) + len(self.ancilla_qubits)

    def refactor(self, state):

        found_stab_ids = set()

        refactor_things = list(self.checks)
        refactor_things.extend(self.logical_zs)

        for (ps, qs, _) in refactor_things:
            xs = set([])
            zs = set([])

            for p, q in zip(ps, qs):

                if p == 'X' or p == 'x':
                    xs.add(q)
                elif p == 'Z' or p == 'z':
                    zs.add(q)
                elif p == 'Y' or p == 'y':
                    xs.add(q)
                    zs.add(q)

            #found, stab_id = state.refactor(xs, zs, choose=0, protected=found_stab_ids)

            try:
                found, stab_id = state.refactor(xs, zs, choose=0, protected=found_stab_ids)
            except IndexError:
                xonly = xs -zs
                zonly = zs - xs
                ys = xs & zs
                raise Exception("IndexError.\nThe stabilizer {'X': %s, 'Y': %s, 'Z': %s} is likely redundant!" %
                                (xonly, ys, zonly))

            found_stab_ids.add(stab_id)

            if not found:
                raise Exception('Could not find check:', (ps, qs))

            # print('Found Xs - %s, Zs - %s, id = %s' % (xs, zs, stab_id))
            # state.print_stabs(verbose=True, print_y=True)

        for q in self.ancilla_qubits:
            found, stab_id = state.refactor({q}, set(), choose=-1, protected=found_stab_ids)
            found_stab_ids.add(stab_id)

            if not found:
                raise Exception('Could not find ancilla %s' % q)

            # print('Found Xs - %s, Zs - %s, id = %s' % ({q}, set(), stab_id))
            # state.print_stabs(verbose=True, print_y=True)

    def get_check_ancilla(self):
        check_tuples = []
        ancilla_tuples = []

        for (ps, qs, _) in self.checks:
            xs = set([])
            zs = set([])

            for p, q in zip(ps, qs):

                if p == 'X' or p == 'x':
                    xs.add(q)
                elif p == 'Z' or p == 'z':
                    zs.add(q)
                elif p == 'Y' or p == 'y':
                    xs.add(q)
                    zs.add(q)

            check_tuples.append((xs, zs))

        for q in self.ancilla_qubits:
            ancilla_tuples.append(({q}, set([])))

        return check_tuples, ancilla_tuples

    def get_info(self, state, stop_search=1000, verbose=True, print_y=False):

        if self.circuit is None:
            return Exception('Must run `compile()` first!')

        self.refactor(state)
        stab_strs, destab_strs = state.print_stabs(verbose=False, print_y=print_y)

        num_ancillas = len(self.ancilla_qubits)

        num_logical = self.num_logical_qubits()
        num_checks = len(self.checks)

        if verbose:
            print('Number of data qubits: %s' % self.num_data_qubits)
            print('Number of checks: %s' % num_checks)
            print('Number of logical qubits: %s' % num_logical)

        check_tuples, ancilla_tuples = self.get_check_ancilla()
        # determine the gen_id of the checks and logicals
        check_gens = []
        logical_gens = []
        ancilla_gens = []

        # print(ancilla_tuples)

        missing_checks = list(check_tuples)
        missing_ancillas = list(ancilla_tuples)
        notmatched_gens = list(range(state.num_qubits))

        found_all = False
        search_count = 0
        while not found_all:
            if verbose:
                print('----')
            for g, gtuple in enumerate(zip(state.stabs.row_x, state.stabs.row_z)):
                if gtuple in missing_checks:
                    missing_checks.remove(gtuple)
                    # print(g)
                    # print(gtuple)
                    try:
                        notmatched_gens.remove(g)
                    except ValueError:
                        raise Exception('list.remove(x): x not in list.\nThe stabilizer %s is likely redundant!' %
                                        str(gtuple))

                    check_gens.append(g)
                elif gtuple in missing_ancillas:
                    missing_ancillas.remove(gtuple)
                    notmatched_gens.remove(g)
                    ancilla_gens.append(g)

            if len(notmatched_gens) == num_logical:
                logical_gens = notmatched_gens
                found_all = True
            else:
                for xs, zs in missing_checks:
                    state.refactor(xs, zs, choose=0, prefer=notmatched_gens)
                state.print_stabs(verbose=False, print_y=print_y)

            if search_count == stop_search:
                raise Exception('Can not refactor properly!')
            else:
                search_count += 1

        self.data_gens = set(check_gens)
        self.ancilla_gens = set(ancilla_gens)
        self.logical_gens = set(logical_gens)

        if verbose:

            if len(check_gens) != num_checks:
                print('Found:', check_gens)
                print('Want:', check_tuples)
                raise Exception('Did not find the correct number of stabilizer generators. %s/%s' %
                                (len(check_gens), num_checks))

            if len(logical_gens) != num_logical:
                print('Found:', logical_gens)
                raise Exception('Did not find the correct number of logical generators. %s/%s' %
                                (len(logical_gens), num_logical))

            print('\nStabilizer generators:')
            for gen in check_gens:
                print(stab_strs[gen][:len(stab_strs[gen])-num_ancillas])

            print('\nDestabilizer generators:')
            for gen in check_gens:
                print(destab_strs[gen][:len(destab_strs[gen]) - num_ancillas])

            print('\nLogical operators:')

            for i, gen in enumerate(logical_gens):
                print('\n. Logical Z #%s:' % str(i+1))
                print(stab_strs[gen][:len(stab_strs[gen])-num_ancillas])
                print('. Logical X #%s:' % str(i + 1))
                print(destab_strs[gen][:len(destab_strs[gen]) - num_ancillas])

        logical_z_strings = []
        logical_x_strings = []

        for i, gen in enumerate(logical_gens):

            z_string = stab_strs[gen][:len(stab_strs[gen]) - num_ancillas]
            x_string = destab_strs[gen][:len(destab_strs[gen]) - num_ancillas]

            check_dict = {}
            for q, P in enumerate(z_string):
                if P == 'X':
                    qset = check_dict.setdefault('X', set())
                elif P == 'Z':
                    qset = check_dict.setdefault('Z', set())
                elif P == 'Y':
                    qset = check_dict.setdefault('Y', set())
                else:
                    continue
                qset.add(q-2)
            logical_z_strings.append(check_dict)

            check_dict = {}
            for q, P in enumerate(x_string):
                if P == 'X':
                    qset = check_dict.setdefault('X', set())
                elif P == 'Z':
                    qset = check_dict.setdefault('Z', set())
                elif P == 'Y':
                    qset = check_dict.setdefault('Y', set())
                else:
                    continue

                qset.add(q-2)
            logical_x_strings.append(check_dict)

        return logical_z_strings, logical_x_strings, stab_strs, destab_strs

    def distance(self, css=False, verbose=True):
        """
        Checks the distance of the code.

        Returns:

        """

        if self.circuit is None:
            raise Exception('Must compile circuits first!')

        qudit_set = self.data_qubits

        state = self.state
        found = self._dist_mode_smallest(state, qudit_set, css=css, verbose=verbose)

        if verbose:
            if found:
                xs, zs = found
                distance = len(xs | zs)

                print('\nThis is a [[%s, %s, %s]] code.' % (self.num_data_qubits, self.num_logical_qubits(), distance))

        if not found:
            print('No logical errors found... Checks might describe a stabilizer state.')
        else:

            xs, zs = found
            self.dist = len(xs | zs)
            return found

    def _dist_mode_smallest(self, state, qudit_set, css=False, verbose=True):
        """
        Determine if a logical error can be found by starting with the smallest weight errors

        Args:
            state:
            qudit_set:

        Returns:

        """

        for lenq in range(1, len(qudit_set)+1):

            if verbose:
                print('Checking errors of length %s...' % lenq)

            for xs, zs in self.gen_errors(qudit_set, lenq, lenq, css=css):
                if self._is_logical_error(state, xs, zs):
                    if verbose:
                        print("Logical error found: Xs - %s Zs - %s" % (xs, zs))
                    return xs, zs

        # We should not be able to get to this point...
        return False

    def gen_errors(self, qubits, min_errors=1, max_errors=False, css=False):
        """

        Args:
            qubits (set of int):
            min_errors (int):
            max_errors (bool, int):
            css (bool):

        Returns:

        """

        paulis = ('X', 'Z', 'Y')

        num_qubits = len(qubits)

        for i in range(min_errors, num_qubits + 1):

            if max_errors and i > max_errors:
                break

            xs = next(product(('X',), repeat=i))
            zs = next(product(('Z',), repeat=i))

            xzs = [xs, zs]

            # print(xs)
            # print(zs)

            for b in combinations(qubits, i):

                for ps in xzs:

                    x_set = set()
                    z_set = set()
                    for p, q in zip(ps, b):
                        if p == 'X':
                            x_set.add(q)
                        else:
                            z_set.add(q)
                    yield x_set, z_set

            if not css:
                for a in product(paulis, repeat=i):
                    if a == xs or a == zs:
                        continue

                    for b in combinations(qubits, i):
                        x_set = set()
                        z_set = set()
                        for p, q in zip(a, b):
                            if p == 'X':
                                x_set.add(q)
                            elif p == 'Z':
                                z_set.add(q)
                            else:
                                x_set.add(q)
                                z_set.add(q)
                        yield x_set, z_set

    def _is_logical_error(self, state, xs, zs):

        # A trivial error anticommutes with the checks. (Might or might not anticommute with the logical stabilizers)
        # A logical error commutes with the checks and is not a product of checks.

        # Does the error anticommute with the checks?
        x_anticoms = set([])
        z_anticoms = set([])
        for q in xs:
            x_anticoms ^= state.stabs.col_z[q]

        for q in zs:
            z_anticoms ^= state.stabs.col_x[q]

        anticoms = x_anticoms ^ z_anticoms
        anticom_logical_zs = self.logical_gens & anticoms
        anticoms -= self.logical_gens

        if anticoms:
            return False
        else:

            # So the error commutes with all the stabilizers
            # Did it anticommute with any logical Z operations? If so... It is a product of logical Xs!
            # (and possibly other things)
            if anticom_logical_zs:
                return True
            else:
                # Let's see if the error anticommuted with any logical X operators:

                x_anticoms_destabs = set([])
                z_anticoms_destabs = set([])

                for q in xs:
                    x_anticoms_destabs ^= state.destabs.col_z[q]

                for q in zs:
                    z_anticoms_destabs ^= state.destabs.col_x[q]

                anticoms_destabs = x_anticoms_destabs ^ z_anticoms_destabs
                anticom_logical_xs = self.logical_gens & anticoms_destabs

                # The error is a product of logical Zs
                return bool(anticom_logical_xs)
