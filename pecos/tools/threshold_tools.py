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

import numpy as np
from .. import circuit_runners, circuits
from ..qeccs import Surface4444
from ..decoders import MWPM2D
from ..error_gens import XGen
from ..misc.threshold_curve import threshold_fit as default_fit, func as default_func


def threshold_code_capacity(qecc_class, error_gen, decoder_class, ps, ds, runs, verbose=False, mode=1,
                            threshold_fit=None, p0=None, func=None, circuit_runner=None, basis=None):
    """
    Function that generates p_logical values given a list of physical errors (ps) and distance (ds).
    Args:
        ps(list of float):
        ds(list of int):
        runs(int):
        error_gen:
        qecc_class:
        decoder_class:
        verbose:
        mode:
        threshold_fit:
        p0:
        func:
        circuit_runner:

    Returns:

    """

    if circuit_runner is None:
        circuit_runner = circuit_runners.Standard()

    if error_gen is None:
        error_gen = XGen(model_level='code_capacity')

    if qecc_class is None:
        qecc_class = Surface4444

    if decoder_class is None:
        decoder_class = MWPM2D

    if threshold_fit is None:
        threshold_fit = default_fit

    if func is None:
        func = default_func

        if p0 is None:
            p0 = (0.1, 1.5, 1, 1, 1)

    if basis not in [None, 'zero', 'plus', 'both']:
        raise Exception('`basis` can only be "None", "zero", "plus", "both"!')

    if mode == 1 and basis != 'both':
        determine_rate = codecapacity_logical_rate
    elif mode == 1 and basis == 'both':
        determine_rate = codecapacity_logical_rate2
    elif mode == 2:
        determine_rate = codecapacity_logical_rate3
    else:
        raise Exception('Mode "%s" is not handled!' % mode)

    plist = np.array(ps * len(ds))

    dlist = []
    for d in ds:
        for p in ps:
            dlist.append(d)
    dlist = np.array(dlist)

    plog = []
    for d in ds:

        qecc = qecc_class(distance=d)
        decoder = decoder_class(qecc)

        for p in ps:

            logical_error_rate, time = determine_rate(runs, qecc, d, error_gen, error_params={'p': p}, decoder=decoder,
                                                      verbose=verbose, circuit_runner=circuit_runner, basis=basis)
            if verbose:
                if time:
                    print('Runtime: %s s' % time)

                print('----')

            plog.append(logical_error_rate)

    plog = np.array(plog)

    # results = threshold_fit(plist, dlist, plog, func, p0)

    # return plist, dlist, plog, results
    # return {'distances': ds, 'ps': ps, 'plog': plog, 'opt': results[0], 'std': results[1]}
    return {'distances': ds, 'ps_physical': plist, 'p_logical': plog}


def threshold_code_capacity_calc(ps, ds, runs, error_gen=None, qecc_class=None, decoder_class=None, verbose=True,
                                 mode=1, threshold_fit=None, p0=None, func=None, circuit_runner=None):
    """
    Function that generates p_logical values given a list of physical errors (ps) and distance (ds).
    Args:
        ps(list of float):
        ds(list of int):
        runs(int):
        error_gen:
        qecc_class:
        decoder_class:
        verbose:
        mode:
        threshold_fit:
        p0:
        func:
        circuit_runner:

    Returns:

    """

    if circuit_runner is None:
        circuit_runner = circuit_runners.Standard()

    if error_gen is None:
        error_gen = XGen(model_level='code_capacity')

    if qecc_class is None:
        qecc_class = Surface4444

    if decoder_class is None:
        decoder_class = MWPM2D

    if threshold_fit is None:
        threshold_fit = default_fit

    if func is None:
        func = default_func

        if p0 is None:
            p0 = (0.1, 1.5, 1, 1, 1)

    if mode == 1:
        determine_rate = codecapacity_logical_rate
    elif mode == 2:
        determine_rate = codecapacity_logical_rate2
    elif mode == 3:
        determine_rate = codecapacity_logical_rate3
    else:
        raise Exception('Mode "%s" is not handled!' % mode)

    plist = np.array(ps * len(ds))

    dlist = []
    for d in ds:
        for p in ps:
            dlist.append(d)
    dlist = np.array(dlist)

    plog = []
    for d in ds:

        qecc = qecc_class(distance=d)
        decoder = decoder_class(qecc)

        for p in ps:

            logical_error_rate, time = determine_rate(runs, qecc, d, error_gen, error_params={'p': p}, decoder=decoder,
                                                      verbose=verbose, circuit_runner=circuit_runner)
            if verbose:
                if time:
                    print('Runtime: %s s' % time)

                print('----')

            plog.append(logical_error_rate)

    plog = np.array(plog)

    results = threshold_fit(plist, dlist, plog, func, p0)

    # return plist, dlist, plog, results
    return {'plist': plist, 'dlist': dlist, 'plog': plog, 'opt': results[0], 'std': results[1]}


def codecapacity_logical_rate(runs, qecc, distance, error_gen, error_params, decoder, seed=None, state_sim=None,
                              verbose=True, circuit_runner=None, basis=None):
    """
    A tool for determining the code-capacity logical-error rate for syndrome extraction.

    In this analysis only logical |0> is prepared and each run consists of an ideal logical |0> preparation followed by
    a single round of syndrome extraction. The error rate is determined by number of runs with logical failures divided
    by the total number of runs.

    Args:
        runs: Number of runs to evaluate the logical error rate.
        qecc:
        distance:
        error_gen:
        error_params:
        decoder:
        seed:
        state_sim:
        verbose:
        circuit_runner:

    Returns:

    """

    p = error_params['p']
    total_time = 0.0

    # Circuit simulator
    if circuit_runner is None:
        circuit_runner = circuit_runners.TimingRunner(seed=seed)
    # circ_sim = circuit_runners.Standard(seed=seed)

    # Syndrome extraction
    syn_extract = circuits.LogicalCircuit(supress_warning=True)
    syn_extract.append(qecc.gate('I', num_syn_extract=1))

    # Choosing basis
    if basis is None or basis == 'zero':
        basis = '|0>'
    elif basis == 'plus':
        basis = '|+>'
    else:
        raise Exception('Basis must be "zero", "plus", "None"!')

    # init circuit
    initzero = circuits.LogicalCircuit(supress_warning=True)
    instr_symbol = 'ideal init %s' % basis
    gate = qecc.gate(instr_symbol)
    initzero.append(gate)

    # print(gate)
    # print(initzero)

    logical_circ_dict = gate.final_instr().final_logical_ops
    logical_ops_sym = gate.final_instr().logical_stabilizers

    if len(logical_circ_dict) != 1:
        raise Exception('This tool expects a code that stores one logical qubit.')

    logical_circ = logical_circ_dict[0][logical_ops_sym[0]]

    num_failure = 0

    for _ in range(runs):
        # State
        state = circuit_runner.init(qecc.num_qudits)

        # Create ideal logical |0>
        circuit_runner.run(state, initzero)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        output, _ = circuit_runner.run(state, syn_extract, error_gen=error_gen, error_params=error_params)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        # syn = output.simplified(True)

        if output:

            # Recovery operation
            recovery = decoder.decode(output)

            # Apply recovery operation
            circuit_runner.run_circuit(state, recovery)

        sign = state.logical_sign(logical_circ)

        num_failure += sign

    logical_rate = float(num_failure) / float(runs)

    if verbose:
        print('\ndistance = %s' % distance)
        print('p = %s' % p)
        print('runs = %s' % runs)

        print('\nlogical error rate: %s' % logical_rate)
        r = float(logical_rate)/float(p)
        print('\nplog/p = %s' % r)

    return logical_rate, total_time


def codecapacity_logical_rate2(runs, qecc, distance, error_gen, error_params, decoder, seed=None, state_sim=None,
                               verbose=True, circuit_runner=None, basis=None):
    """
    A tool for determining the code-capacity logical-error rate for syndrome extraction.

    In this analysis only logical |0> is prepared and each run consists of an ideal logical |0> preparation followed by
    a single round of syndrome extraction. The error rate is determined by number of runs with logical failures divided
    by the total number of runs.

    Args:
        runs: Number of runs to evaluate the logical error rate.
        qecc:
        distance:
        error_gen:
        error_params:
        decoder:
        seed:
        state_sim:
        verbose:
        circuit_runner:

    Returns:

    """

    p = error_params['p']
    total_time = 0.0

    # Circuit simulator
    if circuit_runner is None:
        circuit_runner = circuit_runners.TimingRunner(seed=seed)

    # Syndrome extraction
    syn_extract = circuits.LogicalCircuit(supress_warning=True)
    syn_extract.append(qecc.gate('I', num_syn_extract=1))

    # init logical |0> circuit
    initzero = circuits.LogicalCircuit(supress_warning=True)
    initzero.append(qecc.gate('ideal init |0>'))

    # init logical |+> circuit
    initplus = circuits.LogicalCircuit(supress_warning=True)
    initplus.append(qecc.gate('ideal init |+>'))

    logical_ops_zero = qecc.instruction('instr_init_zero').logical_stabs[0]['Z']
    logical_ops_plus = qecc.instruction('instr_init_plus').logical_stabs[0]['X']

    num_failure = 0

    for _ in range(runs):
        # States
        state0 = circuit_runner.init(qecc.num_qudits, simulator=state_sim)
        state1 = circuit_runner.init(qecc.num_qudits, simulator=state_sim)

        # Create ideal logical |0>
        circuit_runner.run(state0, initzero)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        # Create ideal logical |+>
        circuit_runner.run(state1, initplus)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        output, error_circuits = circuit_runner.run(state0, syn_extract, error_gen=error_gen,
                                                          error_params=error_params)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        # syn = output.simplified(True)

        circuit_runner.run(state1, syn_extract, error_circuits=error_circuits)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        if output:

            # Recovery operation
            recovery = decoder.decode(output)

            # Apply recovery operation

            circuit_runner.run_circuit(state0, recovery)
            try:
                total_time += circuit_runner.total_time
            except AttributeError:
                pass
            circuit_runner.run_circuit(state1, recovery)
            try:
                total_time += circuit_runner.total_time
            except AttributeError:
                pass

        sign0 = state0.logical_sign(logical_ops_zero)
        sign1 = state1.logical_sign(logical_ops_plus)

        if sign0 or sign1:

            num_failure += 1

    logical_rate = float(num_failure) / float(runs)

    if verbose:
        print('\ndistance = %s' % distance)
        print('p = %s' % p)
        print('runs = %s' % runs)

        print('\nlogical error rate: %s' % logical_rate)
        r = float(logical_rate)/float(p)
        print('\nplog/p = %s' % r)
        # print('----')

    return logical_rate, total_time


def codecapacity_logical_rate3(runs, qecc, distance, error_gen, error_params, decoder, seed=None, state_sim=None,
                               max_syn_extract=1e7, circuit_runner=None, verbose=True, init_circuit=None,
                               init_logical_ops=None, basis=None):
    """
    A tool for determining the code-capacity logical-error rate for syndrome extraction.

    In this analysis only logical |0> is prepared and each run consists of an ideal logical |0> preparation followed by
    a single round of syndrome extraction. The error rate is determined by number of runs with logical failures divided
    by the total number of runs.

    !!! This version determines logical threshold from 1/avg(duration)

    Args:
        runs: Number of runs to evaluate the logical error rate.
        qecc:
        distance:
        error_gen:
        error_params:
        decoder:
        seed:
        state_sim:
        max_syn_extract:
        circuit_runner:
        verbose:
        init_circuit:
        init_logical_ops:

    Returns:

    """

    p = error_params['p']
    total_time = 0.0

    # Circuit simulator
    if circuit_runner is None:
        circuit_runner = circuit_runners.TimingRunner(seed=seed)

    if init_circuit is None:
        # init circuit
        init_circuit = circuits.LogicalCircuit(supress_warning=True)

        # Choosing basis
        if basis is None or basis == 'zero':
            basis = '|0>'
        elif basis == 'plus':
            basis = '|+>'
        else:
            raise Exception('Basis must be "zero", "plus", "None"!')

        gate = qecc.gate('ideal init %s' % basis)
        init_circuit.append(gate)

    if init_logical_ops is None:
        # logical_ops = qecc.instruction('instr_syn_extract').final_logical_ops[0]
        ## logical_ops = qecc.instruction('instr_init_zero').logical_stabs[0]

        if init_circuit is None:

            gate = qecc.gate('ideal init %s' % basis)

            # if len(gate.final_logical_stabs()) != 1:
            #    raise Exception('This tool expects a code that stores one logical qubit.')

            # logical_ops = qecc.instruction('instr_syn_extract').final_logical_ops[0]
            # logical_ops = gate.final_logical_stabs()[0]
            # logical_ops_plus = qecc.instruction('instr_init_plus').logical_stabs[0]

            logical_circ_dict = gate.final_instr().final_logical_ops
            logical_ops_sym = gate.final_instr().logical_stabilizers

            if len(logical_circ_dict) != 1:
                raise Exception('This tool expects a code that stores one logical qubit.')

            logical_ops = logical_circ_dict[0][logical_ops_sym[0]]
        else:
            raise Exception('This case is not handled!')
    else:
        logical_ops = init_logical_ops

    # Syndrome extraction
    syn_extract = circuits.LogicalCircuit(supress_warning=True)
    syn_extract.append(qecc.gate('I', num_syn_extract=1))

    # logical_ops = qecc.instruction('instr_syn_extract').final_logical_ops[0]

    run_durations = []

    for _ in range(runs):

        # State
        state = circuit_runner.init(qecc.num_qudits, simulator=state_sim)

        # Create ideal logical |0>
        circuit_runner.run(state, init_circuit)
        try:
            total_time += circuit_runner.total_time
        except AttributeError:
            pass

        for duration in range(max_syn_extract):

            # Run syndrome extraction
            output, _ = circuit_runner.run(state, syn_extract, error_gen=error_gen, error_params=error_params)
            try:
                total_time += circuit_runner.total_time
            except AttributeError:
                pass

            # syn = output.simplified(True)

            if output:
                # Recovery operation
                recovery = decoder.decode(output)

                # Apply recovery operation
                circuit_runner.run_circuit(state, recovery)
                try:
                    total_time += circuit_runner.total_time
                except AttributeError:
                    pass

            sign = state.logical_sign(logical_ops)

            if sign:
                break

        else:
            raise Exception('Max syndrome extraction (%s) met.' % max_syn_extract)

        run_durations.append(duration+1)  # duration + 1 == number of syndrome extractions.

    if verbose:
        print('\nTotal number of runs: %s' % sum(run_durations))

    run_durations = np.array(run_durations)
    duration_mean = np.mean(run_durations)

    logical_rate = 1.0 / duration_mean

    if verbose:
        print('\ndistance = %s' % distance)
        print('p = %s' % p)
        print('Number of failures = %s' % runs)

        print('\nlogical error rate: %s' % logical_rate)
        r = float(logical_rate)/float(p)
        print('\nplog/p = %s' % r)

    return logical_rate, total_time
