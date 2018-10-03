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

from itertools import product, combinations
import itertools as it
import numpy as np
from ..simulators import SparseSim as state_sparse
from ..circuit_runners import Standard
from ..circuits import QuantumCircuit, LogicalCircuit
from ..misc.stabilizer_funcs import op_commutes, remove_stab, circ2set, find_stab
from ..error_gens.parent_class_error_gen import ErrorCircuits
from ..decoders import MWPM2D


def powerset(iterable, bound=None):
    """
    Returns the power set of an iterable
    """
    powerlist = list(iterable)
    if bound is None:
        bound = len(powerlist)
    return it.chain.from_iterable(it.combinations(powerlist, t) for t in range(bound+1))


def t_errors_check(qecc, logical_gate=None, syn_extract=None, decoder=None, t_weight=None, error_set=None,
                   verbose=True, data_errors=True, ancilla_errors=False):
    """
    This checks that the exRec conditions for a fault-free error correction (EC) or logical gate (Ga) as described in
    arXiv:quant-ph/0504218.

    For fault-free EC, weight <= t errors in produce no errors out.

    For fault-free Ga, weight <= t errors in produce weight <= errors out.


    Fault-free EC:
                     ------------------
    error wt <= t -> |EC (fault free) | -> no errors => No syndrome in subsequent fault-free EC (+ no logical faults)
                     ------------------


    Fault-free Ga:
                     ------------------
    error wt <= t -> |Ga (fault free) | ->error wt <= t   => A following fault-free EC + Recovery will result in a state
                     ------------------
    with no logical fault.


    Args:
        qecc:
        logical_gate(QuantumCircuit):
        syn_extract(QuantumCircuit):
        decoder:
        t_weight:
        error_set:
        verbose:
        data_errors:
        ancilla_errors:

    Returns:
        tuple (bool, int): The bool is whether the check is passed. The int is the weight of error last checked. If the
        bool is True then int == t_weight. If bool == False, int == weight of error that caused a logical error.

    """

    qudit_set = set()

    if data_errors:
        qudit_set.update(qecc.data_qudit_set)

    if ancilla_errors:
        qudit_set.update(qecc.ancilla_qudit_set)

    if t_weight is None:
        t_weight = np.floor((qecc.distance-1)/2)

    if error_set is None:
        error_set = {'X', 'Y', 'Z'}

    circ_sim = Standard()

    # init |0> circuit
    initzero = LogicalCircuit(supress_warning=True)
    initzero.append(qecc.gate('ideal init |0>'))

    # init |+> circuit
    initplus = LogicalCircuit(supress_warning=True)
    initplus.append(qecc.gate('ideal init |+>'))

    if syn_extract is not None and logical_gate is not None:
        raise Exception('Both syn_extract and logical_gate cannot be set (not None).')

    if syn_extract is None:
        # Syndrome extraction
        syn_extract = LogicalCircuit(supress_warning=True)
        syn_extract.append(qecc.gate('I', num_syn_extract=1, random_outcome=1))

    if logical_gate is None:
        logic = syn_extract
    else:
        logic = logical_gate

    logical_ops_zero = qecc.instruction('instr_init_zero').logical_stabs[0]
    logical_ops_plus = qecc.instruction('instr_init_plus').logical_stabs[0]

    if decoder is None:
        decoder = MWPM2D(qecc)

    for qubit_comb in powerset(qudit_set):

        if len(qubit_comb) > t_weight:
            break

        error_combinations = product(error_set, repeat=len(qubit_comb))

        for error_comb in error_combinations:

            # print('>', error_comb, qubit_comb)
            error_circ = QuantumCircuit(1)
            errors = ErrorCircuits()

            errors.simple_add(0, 0, 0, before_errors=error_circ)

            for e, q in zip(error_comb, qubit_comb):
                # print(e, q)
                error_circ.update(e, {q})

            state_zero = circ_sim.init(qecc.num_qudits)
            state_plus = circ_sim.init(qecc.num_qudits)

            circ_sim.run_logic(state_zero, initzero)
            circ_sim.run_logic(state_plus, initplus)

            output, _ = circ_sim.run_logic(state_zero, logic, error_circuits=errors)
            circ_sim.run_logic(state_plus, logic, error_circuits=errors)

            syn = output.simplified(True)

            if syn:
                # Recovery operation
                recovery = decoder.decode(syn)
                circ_sim.run_circuit(state_zero, recovery)
                circ_sim.run_circuit(state_plus, recovery)

            sign_zero = state_zero.logical_sign(*logical_ops_zero)
            sign_plus = state_plus.logical_sign(*logical_ops_plus)
            # sign_plus = 0
            # sign_zero = 0

            if sign_zero or sign_plus:
                if verbose:
                    print(errors)
                return False, len(error_comb)

            if logical_gate is None:  # The following is only required for EC.

                # Any remaining syndromes?
                output, _ = circ_sim.run_logic(state_zero, syn_extract)
                # circ_sim.run_logic(state_plus, syn_extract, error_circuits=errors)
                syn = output.simplified(True)

                if syn:
                    if verbose:
                        print('syndromes = %s' % syn)
                        print(errors)
                    return False, len(error_comb)

    return True, int(t_weight)


def fault_check(qecc, logical_gate=None, decoder=None, t_weight=None, error_set=None, verbose=True, data_errors=True,
                ancilla_errors=False):
    """
    This checks that the exRec conditions for a faulty error correction (EC) or logical gate (Ga) as described in
    arXiv:quant-ph/0504218.

    For fault-free EC, weight <= t errors in produce no errors out.

    For fault-free Ga, weight <= t errors in produce weight <= errors out.


    Fault-free EC:
                     ------------------
    error wt <= t -> |EC (fault free) | -> no errors => No syndrome in subsequent fault-free EC (+ no logical faults)
                     ------------------


    Fault-free Ga:
                     ------------------
    error wt <= t -> |Ga (fault free) | ->error wt <= t   => A following fault-free EC + Recovery will result in a state
                     ------------------
    with no logical fault.


    Args:
        qecc:
        logical_gate(QuantumCircuit):
        decoder:
        t_weight:
        error_set:
        verbose:
        data_errors:
        ancilla_errors:

    Returns:
        tuple (bool, int): The bool is whether the check is passed. The int is the weight of error last checked. If the
        bool is True then int == t_weight. If bool == False, int == weight of error that caused a logical error.

    """

    qudit_set = set()

    if data_errors:
        qudit_set.update(qecc.data_qudit_set)

    if ancilla_errors:
        qudit_set.update(qecc.ancilla_qudit_set)

    if t_weight is None:
        t_weight = np.floor((qecc.distance - 1) / 2)

    if error_set is None:
        error_set = {'X', 'Y', 'Z'}

    circ_sim = Standard()

    # init |0> circuit
    initzero = LogicalCircuit(supress_warning=True)
    initzero.append(qecc.gate('ideal init |0>'))

    # init |+> circuit
    initplus = LogicalCircuit(supress_warning=True)
    initplus.append(qecc.gate('ideal init |+>'))

    if logical_gate is None:
        # Syndrome extraction
        syn_extract = LogicalCircuit(supress_warning=True)
        syn_extract.append(qecc.gate('I', num_syn_extract=1, random_outcome=1))
        logic = syn_extract
    else:
        logic = logical_gate

    logical_ops_zero = qecc.instruction('instr_init_zero').logical_stabs[0]
    logical_ops_plus = qecc.instruction('instr_init_plus').logical_stabs[0]

    if decoder is None:
        decoder = MWPM2D(qecc)

    for qubit_comb in powerset(qudit_set):

        if len(qubit_comb) > t_weight:
            break

        error_combinations = product(error_set, repeat=len(qubit_comb))

        for error_comb in error_combinations:

            # print('>', error_comb, qubit_comb)
            error_circ = QuantumCircuit(1)
            errors = ErrorCircuits()

            errors.simple_add(0, 0, 0, before_errors=error_circ)

            for e, q in zip(error_comb, qubit_comb):
                # print(e, q)
                error_circ.update(e, {q})

            state_zero = circ_sim.init(qecc.num_qudits)
            state_plus = circ_sim.init(qecc.num_qudits)

            circ_sim.run_logic(state_zero, initzero)
            circ_sim.run_logic(state_plus, initplus)

            output, _ = circ_sim.run_logic(state_zero, logic, error_circuits=errors)
            circ_sim.run_logic(state_plus, logic, error_circuits=errors)

            syn = output.simplified(True)

            if syn:
                # Recovery operation
                recovery = decoder.decode(syn)
                circ_sim.run_circuit(state_zero, recovery)
                circ_sim.run_circuit(state_plus, recovery)

            sign_zero = state_zero.logical_sign(*logical_ops_zero)
            sign_plus = state_plus.logical_sign(*logical_ops_plus)
            # sign_plus = 0
            # sign_zero = 0

            if sign_zero or sign_plus:
                if verbose:
                    print(errors)
                return False, len(error_comb)

    return True, int(t_weight)


def distance_check(qecc, mode=None, dist_mode=None):
    """
    Determines the distance of the code by looking for the smallest logical errors.

    Args:
        qecc:
        mode:
        dist_mode:

    Returns:
        Tuple (bool, int). The bool is whether the check is passed. The int is the weight of error last checked. If the
        bool is True then int == t_weight. If bool == False, int == weight of error that caused a logical error.

    """

    qudit_set = qecc.data_qudit_set

    circ_sim = Standard()
    state = circ_sim.init(qecc.num_qudits, simulator=state_sparse)

    ideal_initlogic = LogicalCircuit(supress_warning=True)
    ideal_initlogic.append(qecc.gate('ideal init |0>'))

    circ_sim.run_logic(state, ideal_initlogic)

    logical_op, delogical_op = qecc.instruction('instr_init_zero').logical_stabs[0]

    destab_xs, destab_zs = circ2set(delogical_op.items(params=False))
    stab_xs, stab_zs = circ2set(logical_op.items(params=False))

    remove_stab(state, stab_xs, stab_zs, destab_xs, destab_zs)

    if dist_mode is None:

        if mode == 'X' or mode == 'x':
            print('x')
            return dist_mode_x(state, qudit_set)
        elif mode == 'Z' or mode == 'z':
            print('z')
            return dist_mode_z(state, qudit_set)
        elif mode == 'power':
            return dist_mode_powerset(state, qudit_set)
        else:
            return dist_mode_smallest(state, qudit_set)

    else:
        return dist_mode(state, qudit_set)


def dist_mode_powerset(state, qudit_set):
    """

    Args:
        state:
        qudit_set:

    Returns:

    """

    for x_errors in powerset(qudit_set):
        for z_errors in powerset(qudit_set):
            if op_commutes(x_errors, z_errors, state.stabs) and not find_stab(state, x_errors, z_errors):
                return "Logical error found: Xs - %s Zs - %s" % (x_errors, z_errors)

    return False


def dist_mode_smallest(state, qudit_set):
    """

    Args:
        state:
        qudit_set:

    Returns:

    """
    for lenq in range(len(qudit_set)+1):
        for qs in combinations(qudit_set, lenq):
            # stab_x = qs, stab_zs = qs
            if op_commutes(qs, qs, state.stabs) and not find_stab(state, qs, qs):
                return "Logical error found: Xs - %s Zs - %s" % (qs, qs)

            for qs2 in powerset(qudit_set, len(qs) - 1):

                if op_commutes(qs2, qs, state.stabs) and not find_stab(state, qs2, qs):
                    return "Logical error found: Xs - %s Zs - %s" % (qs2, qs)

                if op_commutes(qs, qs2, state.stabs) and not find_stab(state, qs, qs2):
                    return "Logical error found: Xs - %s Zs - %s" % (qs, qs2)

    return False


def dist_mode_x(state, qudit_set):
    """

    Args:
        state:
        qudit_set:

    Returns:

    """
    z_errors = ()
    for x_errors in powerset(qudit_set):
        if op_commutes(x_errors, z_errors, state.stabs) and not find_stab(state, x_errors, z_errors):
            return "Logical error found: Xs - %s Zs - %s" % (x_errors, z_errors)

    return False


def dist_mode_z(state, qudit_set):
    """

    Args:
        state:
        qudit_set:

    Returns:

    """
    x_errors = ()
    for z_errors in powerset(qudit_set):
        if op_commutes(x_errors, z_errors, state.stabs) and not find_stab(state, x_errors, z_errors):
            return "Logical error found: Xs - %s Zs - %s" % (x_errors, z_errors)

    return False
