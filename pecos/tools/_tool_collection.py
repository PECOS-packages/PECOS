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
import numpy as np
from .. import circuits
from ..circuits import QuantumCircuit
from ..circuit_runners import Standard


def fault_tolerance_check(QECC, decoder):
    """
    Checks that the decoder can correct all Pauli errors of weight up to floor(distance/2).

    Args:
        QECC:
        decoder:

    Returns:

    """

    # The logical circuits:
    # ---------------------
    init_zero = circuits.LogicalCircuit(layout=QECC.layout)
    init_zero.append(QECC.gate('ideal init |0>'))

    init_plus = circuits.LogicalCircuit(layout=QECC.layout)
    init_plus.append(QECC.gate('ideal init |+>'))

    syn_extract = circuits.LogicalCircuit(layout=QECC.layout)
    syn_extract.append(QECC.gate('I', num_syn_extract=1))

    logical_ops = QECC.instruction('instr_syn_extract').final_logical_ops
    logical_z = logical_ops[0]['Z']
    # logical_x = logical_ops[0]['X']

    num_qudits = QECC.num_qudits
    data_qudits = QECC.data_qudit_set
    qudits = QECC.qudit_set

    t = int(np.floor((QECC.distance-1)*0.5))

    # circuit runner:
    circ_runner = Standard()

    # Check input errors
    for xs, zs in gen_pauli_errors(data_qudits, max_errors=t):
        err = QuantumCircuit([{'X': xs, 'Z': zs}])
        state = circ_runner.init(num_qudits)

        sign = _apply_err(state, circ_runner, init_zero, syn_extract, err, decoder, logical_z)

        if sign:

            raise Exception('Decoder failed to correct error: %s' % err)

        sign = _apply_err(state, circ_runner, init_zero, syn_extract, err, decoder, logical_z)

        if sign:
            raise Exception('Decoder failed to correct error: %s' % err)

    # Check circuit errors
    # Need to apply errors amongst any combination of ticks and qubits

    num_ticks = len(QECC.instruction('instr_syn_extract').circuit)

    spacetime = set(product(list(range(num_ticks)), qudits))
    for xs, zs in gen_pauli_errors(spacetime, max_errors=t):
        state = circ_runner.init(num_qudits)
        xs = list(xs)
        zs = list(zs)

        err_dict = form_errors(xs, zs)

        sign = _apply_err_spacetime(state, circ_runner, init_zero, syn_extract, err_dict, decoder, logical_z, QECC)

        if sign:

            raise Exception('Decoder failed to correct error: %s' % str(spacetime))

        sign = _apply_err_spacetime(state, circ_runner, init_zero, syn_extract, err_dict, decoder, logical_z, QECC)

        if sign:
            raise Exception('Decoder failed to correct error: %s' % str(spacetime))


def form_errors(xs, zs):

    errors = {}
    for t, q in xs:
        xerr = errors.setdefault(t, {}).setdefault('X', set())
        xerr.add(q)

    for t, q in zs:
        zerr = errors.setdefault(t, {}).setdefault('Z', set())
        zerr.add(q)

    return errors


def _apply_err_spacetime(state, circ_runner, init_circ, syn_circ, err_dict, decoder, logical_op, QECC):
    circ_runner.run(state, init_circ)

    syn_circ = QECC.instruction('instr_syn_extract', num_syn_extract=1)
    num_ticks = len(syn_circ.circuit)

    syn = set()
    for t in range(num_ticks):

        xerrs = err_dict[t].get('X', set())
        zerrs = err_dict[t].get('Z', set())

        for gate_sym, locations, _ in syn_circ.circuit.items(tick=t):

            if 'measure' in gate_sym and t in err_dict:
                before_xerrs = locations & xerrs
                if before_xerrs:
                    state.run_gate('X', before_xerrs)
                    xerrs -= before_xerrs

                before_zerrs = locations & zerrs
                if before_zerrs:

                    zerrs -= before_zerrs

            output = state.run_gate(gate_sym, locations)
            syn.update(output.keys())

        if xerrs:
            state.run_gate('X', xerrs)

        if zerrs:
            state.run_gate('Z', zerrs)

    if syn:
        recovery = decoder.decode(syn)
        circ_runner.run_circuit(state, recovery)

    sign = state.logical_sign(logical_op)

    return sign


def _apply_err(state, circ_runner, init_circ, syn_circ, error, decoder, logical_op):

    circ_runner.run(state, init_circ)
    circ_runner.run_circuit(state, error)
    output, _ = circ_runner.run(state, syn_circ)
    syn = output.simplified(True)

    if syn:
        recovery = decoder.decode(syn)
        circ_runner.run_circuit(state, recovery)

    sign = state.logical_sign(logical_op)

    return sign


def gen_pauli_errors(qubits, min_errors=1, max_errors=False, css=False):
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
