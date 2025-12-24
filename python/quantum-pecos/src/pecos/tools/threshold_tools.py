"""Threshold analysis tools for quantum error correction.

This module provides comprehensive tools for analyzing error thresholds
in quantum error correction codes, including threshold estimation,
curve fitting, and statistical analysis of error correction performance.
"""

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

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pecos as pc
from pecos.decoders import MWPM2D
from pecos.engines import circuit_runners
from pecos.error_models import XModel
from pecos.misc.threshold_curve import func as default_func
from pecos.misc.threshold_curve import threshold_fit as default_fit
from pecos.qeccs import Surface4444
from pecos.simulators import SparseSimPy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TypedDict

    from pecos import Array, f64
    from pecos.circuits import LogicalCircuit, QuantumCircuit
    from pecos.engines.circuit_runners import Standard
    from pecos.protocols import Decoder, ErrorGenerator, QECCProtocol, SimulatorProtocol
    from pecos.typing import ErrorParams, LogicalOperator

    ThresholdFitFunc = Callable[
        [
            tuple[Array[f64], Array[f64]],  # x = (p, dist)
            float,  # pth
            float,  # v0
            float,  # a
            float,  # b
            float,  # c
        ],
        float | Array[f64],
    ]

    ThresholdFitter = Callable[
        [
            Array[f64] | list[float],  # plist
            Array[f64] | list[float],  # dlist
            Array[f64] | list[float],  # plog
            ThresholdFitFunc,  # func
            Array[f64] | list[float],  # p0
        ],
        tuple[Array[f64], Array[f64]],
    ]

    class ThresholdResult(TypedDict):
        """Result from threshold calculations."""

        distances: Sequence[int]
        ps_physical: Array[f64]
        p_logical: Array[f64]

    class ThresholdCalcResult(TypedDict):
        """Result from threshold calculations with fitting."""

        plist: Array[f64]
        dlist: Array[f64]
        plog: Array[f64]
        opt: Array[f64]
        std: Array[f64]


def threshold_code_capacity(
    qecc_class: type[QECCProtocol],
    error_gen: type[ErrorGenerator],
    decoder_class: type[Decoder],
    ps: Sequence[float],
    ds: Sequence[int],
    runs: int,
    *,
    verbose: bool = False,
    mode: int = 1,
    threshold_fit: ThresholdFitter | None = None,
    p0: tuple[float, ...] | None = None,
    func: ThresholdFitFunc | None = None,
    circuit_runner: Standard | None = None,
    basis: str | None = None,
) -> ThresholdResult:
    """Function that generates p_logical values given a list of physical errors (ps) and distance (ds).

    Args:
    ----
        qecc_class: The quantum error correcting code class to use.
        error_gen: The error generator for creating noise models.
        decoder_class: The decoder class to use for error correction.
        ps: List of physical error probabilities to test.
        ds: List of code distances to test.
        runs: Number of Monte Carlo runs to perform for each (p, d) pair.
        verbose: If True, prints detailed progress and results.
        mode: The mode for logical rate calculation (1, 2, or 3).
        threshold_fit: Function to use for fitting the threshold curve.
        p0: Initial parameters for the threshold fitting function.
        func: The functional form to use for threshold fitting.
        circuit_runner: The circuit runner to use for simulations.
        basis: The basis for logical measurements (e.g., 'X' or 'Z').

    """
    if circuit_runner is None:
        circuit_runner = circuit_runners.Standard()

    if error_gen is None:
        error_gen = XModel(model_level="code_capacity")

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

    if basis not in {None, "zero", "plus", "both"}:
        msg = '`basis` can only be "None", "zero", "plus", "both"!'
        raise Exception(msg)

    if mode == 1 and basis != "both":
        determine_rate = codecapacity_logical_rate
    elif mode == 1 and basis == "both":
        determine_rate = codecapacity_logical_rate2
    elif mode == 2:
        determine_rate = codecapacity_logical_rate3
    else:
        msg = f'Mode "{mode}" is not handled!'
        raise Exception(msg)

    plist = pc.array(ps * len(ds))

    """
    dlist = []
    for d in ds:
        for p in ps:
            dlist.append(d)
    dlist = pc.array(dlist)
    """

    plog = []
    for d in ds:
        qecc = qecc_class(distance=d)
        decoder = decoder_class(qecc)

        for p in ps:
            logical_error_rate, time = determine_rate(
                runs,
                qecc,
                d,
                error_gen,
                error_params={"p": p},
                decoder=decoder,
                verbose=verbose,
                circuit_runner=circuit_runner,
                basis=basis,
            )
            if verbose:
                if time:
                    print(f"Runtime: {time} s")

                print("----")

            plog.append(logical_error_rate)

    plog = pc.array(plog)

    return {"distances": ds, "ps_physical": plist, "p_logical": plog}


def threshold_code_capacity_calc(
    ps: Sequence[float],
    ds: Sequence[int],
    runs: int,
    error_gen: type[ErrorGenerator] | None = None,
    qecc_class: type[QECCProtocol] | None = None,
    decoder_class: type[Decoder] | None = None,
    *,
    verbose: bool = True,
    mode: int = 1,
    threshold_fit: ThresholdFitter | None = None,
    p0: tuple[float, ...] | None = None,
    func: ThresholdFitFunc | None = None,
    circuit_runner: Standard | None = None,
) -> ThresholdCalcResult:
    """Function that generates p_logical values given a list of physical errors (ps) and distance (ds).

    Args:
    ----
        ps(list of float): List of physical error probabilities to test.
        ds(list of int): List of code distances to test.
        runs(int): Number of Monte Carlo runs to perform for each (p, d) pair.
        error_gen: The error generator for creating noise models.
        qecc_class: The quantum error correcting code class to use.
        decoder_class: The decoder class to use for error correction.
        verbose: If True, prints detailed progress and results.
        mode: The mode for logical rate calculation (1, 2, or 3).
        threshold_fit: Function to use for fitting the threshold curve.
        p0: Initial parameters for the threshold fitting function.
        func: The functional form to use for threshold fitting.
        circuit_runner: The circuit runner to use for simulations.

    """
    if circuit_runner is None:
        circuit_runner = circuit_runners.Standard()

    if error_gen is None:
        error_gen = XModel(model_level="code_capacity")

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
        msg = f'Mode "{mode}" is not handled!'
        raise Exception(msg)

    plist = pc.array(ps * len(ds))

    dlist = [d for d in ds for _p in ps]
    dlist = pc.array(dlist)

    plog = []
    for d in ds:
        qecc = qecc_class(distance=d)
        decoder = decoder_class(qecc)

        for p in ps:
            logical_error_rate, time = determine_rate(
                runs,
                qecc,
                d,
                error_gen,
                error_params={"p": p},
                decoder=decoder,
                verbose=verbose,
                circuit_runner=circuit_runner,
            )
            if verbose:
                if time:
                    print(f"Runtime: {time} s")

                print("----")

            plog.append(logical_error_rate)

    plog = pc.array(plog)

    results = threshold_fit(plist, dlist, plog, func, p0)

    return {
        "plist": plist,
        "dlist": dlist,
        "plog": plog,
        "opt": results[0],
        "std": results[1],
    }


def codecapacity_logical_rate(
    runs: int,
    qecc: QECCProtocol,
    distance: int,
    error_gen: type[ErrorGenerator],
    error_params: ErrorParams,
    decoder: Decoder,
    seed: int | None = None,
    _state_sim: SimulatorProtocol | None = None,
    *,
    verbose: bool = True,
    circuit_runner: Standard | None = None,
    basis: str | None = None,
) -> tuple[float, float]:
    """A tool for determining the code-capacity logical-error rate for syndrome extraction.

    In this analysis only logical |0> is prepared and each run consists of an ideal logical |0> preparation followed by
    a single round of syndrome extraction. The error rate is determined by number of runs with logical failures divided
    by the total number of runs.

    Args:
    ----
        runs: Number of runs to evaluate the logical error rate.
        qecc: The quantum error correcting code instance.
        distance: The code distance (used for creating the QECC instance if needed).
        error_gen: The error generator for creating noise models.
        error_params: Dictionary of error parameters (must include 'p' for error probability).
        decoder: The decoder instance for error correction.
        seed: Random seed for reproducibility.
        state_sim: The state simulator to use (deprecated parameter).
        verbose: If True, prints detailed progress and results.
        circuit_runner: The circuit runner to use for simulations.
        basis: The basis for logical measurements (e.g., 'X' or 'Z').

    """
    p = error_params["p"]
    total_time = 0.0

    # Circuit simulator
    if circuit_runner is None:
        circuit_runner = circuit_runners.TimingRunner(seed=seed)

    # Syndrome extraction
    syn_extract = pc.circuits.LogicalCircuit(suppress_warning=True)
    syn_extract.append(qecc.gate("I", num_syn_extract=1))

    # Choosing basis
    if basis is None or basis == "zero":
        basis = "|0>"
    elif basis == "plus":
        basis = "|+>"
    else:
        msg = 'Basis must be "zero", "plus", "None"!'
        raise Exception(msg)

    # init circuit
    initzero = pc.circuits.LogicalCircuit(suppress_warning=True)
    instr_symbol = f"ideal init {basis}"
    gate = qecc.gate(instr_symbol)
    initzero.append(gate)

    logical_circ_dict = gate.final_instr().final_logical_ops
    logical_ops_sym = gate.final_instr().logical_stabilizers

    if len(logical_circ_dict) != 1:
        msg = "This tool expects a code that stores one logical qubit."
        raise Exception(msg)

    logical_circ = logical_circ_dict[0][logical_ops_sym[0]]

    num_failure = 0

    for _ in range(runs):
        # State
        state = SparseSimPy(qecc.num_qudits)

        # Create ideal logical |0>
        circuit_runner.run(state, initzero)
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        output, _ = circuit_runner.run(
            state,
            syn_extract,
            error_gen=error_gen,
            error_params=error_params,
        )
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        if output:
            # Recovery operation
            recovery = decoder.decode(output)

            # Apply recovery operation
            circuit_runner.run(state, recovery)

        sign = state.logical_sign(logical_circ)

        num_failure += sign

    logical_rate = float(num_failure) / float(runs)

    if verbose:
        print(f"\ndistance = {distance}")
        print(f"p = {p}")
        print(f"runs = {runs}")

        print(f"\nlogical error rate: {logical_rate}")
        r = float(logical_rate) / float(p)
        print(f"\nplog/p = {r}")

    return logical_rate, total_time


def codecapacity_logical_rate2(
    runs: int,
    qecc: QECCProtocol,
    distance: int,
    error_gen: type[ErrorGenerator],
    error_params: ErrorParams,
    decoder: Decoder,
    seed: int | None = None,
    state_sim: SimulatorProtocol | None = None,
    *,
    verbose: bool = True,
    circuit_runner: Standard | None = None,
    _basis: str | None = None,
) -> tuple[float, float]:
    """A tool for determining the code-capacity logical-error rate for syndrome extraction.

    In this analysis only logical |0> is prepared and each run consists of an ideal logical |0> preparation followed by
    a single round of syndrome extraction. The error rate is determined by number of runs with logical failures divided
    by the total number of runs.

    Args:
    ----
        runs: Number of runs to evaluate the logical error rate.
        qecc: The quantum error correcting code instance.
        distance: The code distance (used for creating the QECC instance if needed).
        error_gen: The error generator for creating noise models.
        error_params: Dictionary of error parameters (must include 'p' for error probability).
        decoder: The decoder instance for error correction.
        seed: Random seed for reproducibility.
        state_sim: The state simulator to use.
        verbose: If True, prints detailed progress and results.
        circuit_runner: The circuit runner to use for simulations.
        basis: The basis for logical measurements (e.g., 'X' or 'Z').

    """
    p = error_params["p"]
    total_time = 0.0

    # Circuit simulator
    if circuit_runner is None:
        circuit_runner = circuit_runners.TimingRunner(seed=seed)

    # Syndrome extraction
    syn_extract = pc.circuits.LogicalCircuit(suppress_warning=True)
    syn_extract.append(qecc.gate("I", num_syn_extract=1))

    # init logical |0> circuit
    initzero = pc.circuits.LogicalCircuit(suppress_warning=True)
    initzero.append(qecc.gate("ideal init |0>"))

    # init logical |+> circuit
    initplus = pc.circuits.LogicalCircuit(suppress_warning=True)
    initplus.append(qecc.gate("ideal init |+>"))

    logical_ops_zero = qecc.instruction("instr_init_zero").logical_stabs[0]["Z"]
    logical_ops_plus = qecc.instruction("instr_init_plus").logical_stabs[0]["X"]

    num_failure = 0

    for _ in range(runs):
        # States
        state0 = state_sim(qecc.num_qudits)
        state1 = state_sim(qecc.num_qudits)

        # Create ideal logical |0>
        circuit_runner.run(state0, initzero)
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        # Create ideal logical |+>
        circuit_runner.run(state1, initplus)
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        output, error_circuits = circuit_runner.run(
            state0,
            syn_extract,
            error_gen=error_gen,
            error_params=error_params,
        )
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        circuit_runner.run(state1, syn_extract, error_circuits=error_circuits)
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        if output:
            # Recovery operation
            recovery = decoder.decode(output)

            # Apply recovery operation

            circuit_runner.run(state0, recovery)
            with contextlib.suppress(AttributeError):
                total_time += circuit_runner.total_time
            circuit_runner.run(state1, recovery)
            with contextlib.suppress(AttributeError):
                total_time += circuit_runner.total_time

        sign0 = state0.logical_sign(logical_ops_zero)
        sign1 = state1.logical_sign(logical_ops_plus)

        if sign0 or sign1:
            num_failure += 1

    logical_rate = float(num_failure) / float(runs)

    if verbose:
        print(f"\ndistance = {distance}")
        print(f"p = {p}")
        print(f"runs = {runs}")

        print(f"\nlogical error rate: {logical_rate}")
        r = float(logical_rate) / float(p)
        print(f"\nplog/p = {r}")

    return logical_rate, total_time


def codecapacity_logical_rate3(
    runs: int,
    qecc: QECCProtocol,
    distance: int,
    error_gen: type[ErrorGenerator],
    error_params: ErrorParams,
    decoder: Decoder,
    seed: int | None = None,
    state_sim: SimulatorProtocol | None = None,
    max_syn_extract: float = 1e7,
    circuit_runner: Standard | None = None,
    *,
    verbose: bool = True,
    init_circuit: QuantumCircuit | LogicalCircuit | None = None,
    init_logical_ops: LogicalOperator | None = None,
    basis: str | None = None,
) -> tuple[float, float]:
    """A tool for determining the code-capacity logical-error rate for syndrome extraction.

    In this analysis only logical |0> is prepared and each run consists of an ideal logical |0> preparation followed by
    a single round of syndrome extraction. The error rate is determined by number of runs with logical failures divided
    by the total number of runs.

    !!! This version determines logical threshold from 1/avg(duration)

    Args:
    ----
        runs: Number of runs to evaluate the logical error rate.
        qecc: The quantum error correcting code instance.
        distance: The code distance (used for creating the QECC instance if needed).
        error_gen: The error generator for creating noise models.
        error_params: Dictionary of error parameters (must include 'p' for error probability).
        decoder: The decoder instance for error correction.
        seed: Random seed for reproducibility.
        state_sim: The state simulator to use.
        max_syn_extract: Maximum number of syndrome extraction rounds before declaring success.
        circuit_runner: The circuit runner to use for simulations.
        verbose: If True, prints detailed progress and results.
        init_circuit: Custom initialization circuit (if None, uses default logical |0> or |+>).
        init_logical_ops: Custom logical operators for the initialized state.
        basis: The basis for logical measurements (e.g., 'X' or 'Z').

    """
    p = error_params["p"]
    total_time = 0.0

    # Circuit simulator
    if circuit_runner is None:
        circuit_runner = circuit_runners.TimingRunner(seed=seed)

    if init_circuit is None:
        # init circuit
        init_circuit = pc.circuits.LogicalCircuit(suppress_warning=True)

        # Choosing basis
        if basis is None or basis == "zero":
            basis = "|0>"
        elif basis == "plus":
            basis = "|+>"
        else:
            msg = 'Basis must be "zero", "plus", "None"!'
            raise Exception(msg)

        gate = qecc.gate(f"ideal init {basis}")
        init_circuit.append(gate)

    if init_logical_ops is None:
        if init_circuit is None:
            gate = qecc.gate(f"ideal init {basis}")

            # if len(gate.final_logical_stabs()) != 1:

            logical_circ_dict = gate.final_instr().final_logical_ops
            logical_ops_sym = gate.final_instr().logical_stabilizers

            if len(logical_circ_dict) != 1:
                msg = "This tool expects a code that stores one logical qubit."
                raise Exception(msg)

            logical_ops = logical_circ_dict[0][logical_ops_sym[0]]
        else:
            msg = "This case is not handled!"
            raise Exception(msg)
    else:
        logical_ops = init_logical_ops

    # Syndrome extraction
    syn_extract = pc.circuits.LogicalCircuit(suppress_warning=True)
    syn_extract.append(qecc.gate("I", num_syn_extract=1))

    run_durations = []

    for _ in range(runs):
        # State
        state = state_sim(qecc.num_qudits)

        # Create ideal logical |0>
        circuit_runner.run(state, init_circuit)
        with contextlib.suppress(AttributeError):
            total_time += circuit_runner.total_time

        for _duration in range(max_syn_extract):
            # Run syndrome extraction
            output, _ = circuit_runner.run(
                state,
                syn_extract,
                error_gen=error_gen,
                error_params=error_params,
            )
            with contextlib.suppress(AttributeError):
                total_time += circuit_runner.total_time

            if output:
                # Recovery operation
                recovery = decoder.decode(output)

                # Apply recovery operation
                circuit_runner.run(state, recovery)
                with contextlib.suppress(AttributeError):
                    total_time += circuit_runner.total_time

            sign = state.logical_sign(logical_ops)

            if sign:
                break

        else:
            msg = f"Max syndrome extraction ({max_syn_extract}) met."
            raise Exception(msg)

        run_durations.append(
            max_syn_extract,
        )  # duration + 1 == number of syndrome extractions.

    if verbose:
        print(f"\nTotal number of runs: {sum(run_durations)}")

    run_durations = pc.array(run_durations)
    duration_mean = pc.mean(run_durations)

    logical_rate = 1.0 / duration_mean

    if verbose:
        print(f"\ndistance = {distance}")
        print(f"p = {p}")
        print(f"Number of failures = {runs}")

        print(f"\nlogical error rate: {logical_rate}")
        r = float(logical_rate) / float(p)
        print(f"\nplog/p = {r}")

    return logical_rate, total_time
