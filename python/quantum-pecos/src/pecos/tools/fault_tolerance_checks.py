"""Comprehensive fault tolerance verification for quantum error correction.

This module provides advanced tools for systematic fault tolerance analysis,
including multi-level error injection, propagation analysis, and verification
of quantum error correction protocols under various fault models.
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

import itertools as it
from itertools import combinations, product
from typing import TYPE_CHECKING, TypeVar

import pecos as pc
from pecos.circuits import LogicalCircuit, QuantumCircuit
from pecos.decoders import MWPM2D
from pecos.engines.circuit_runners import Standard
from pecos.error_models.parent_class_error_gen import ErrorCircuits
from pecos.misc.stabilizer_funcs import circ2set, find_stab, op_commutes, remove_stab
from pecos.simulators import SparseSimPy

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos.protocols import Decoder, QECCProtocol, SimulatorProtocol

T = TypeVar("T")


def powerset(
    iterable: Iterable[T],
    bound: int | None = None,
) -> it.chain[tuple[T, ...]]:
    """Returns the power set of an iterable."""
    powerlist = list(iterable)
    if bound is None:
        bound = len(powerlist)
    return it.chain.from_iterable(
        it.combinations(powerlist, t) for t in range(bound + 1)
    )


def t_errors_check(
    qecc: QECCProtocol,
    logical_gate: QuantumCircuit | LogicalCircuit | None = None,
    syn_extract: QuantumCircuit | LogicalCircuit | None = None,
    decoder: Decoder | None = None,
    t_weight: int | None = None,
    error_set: Iterable[tuple[set[int], set[int]]] | None = None,
    *,
    verbose: bool = True,
    data_errors: bool = True,
    ancilla_errors: bool = False,
) -> tuple[bool, int]:
    """Check exRec conditions for fault-free error correction or logical gate.

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
    ----
        qecc: The quantum error correcting code instance.
        logical_gate(QuantumCircuit): The logical gate circuit to test (None for error correction only).
        syn_extract(QuantumCircuit): The syndrome extraction circuit to use.
        decoder: The decoder instance for error correction.
        t_weight: The maximum weight of errors to check (typically pc.floor((distance-1)/2)).
        error_set: Custom set of errors to check (if None, all Pauli errors are checked).
        verbose: If True, prints detailed information about failures.
        data_errors: If True, includes errors on data qubits.
        ancilla_errors: If True, includes errors on ancilla qubits.

    Returns:
    -------
        tuple (bool, int): The bool is whether the check is passed. The int is the weight of error last checked. If the
        bool is True then int == t_weight. If bool == False, int == weight of error that caused a logical error.

    """
    qudit_set = set()

    if data_errors:
        qudit_set.update(qecc.data_qudit_set)

    if ancilla_errors:
        qudit_set.update(qecc.ancilla_qudit_set)

    if t_weight is None:
        t_weight = pc.floor((qecc.distance - 1) / 2)

    if error_set is None:
        error_set = {"X", "Y", "Z"}

    circ_sim = Standard()

    # init |0> circuit
    initzero = LogicalCircuit(suppress_warning=True)
    initzero.append(qecc.gate("ideal init |0>"))

    # init |+> circuit
    initplus = LogicalCircuit(suppress_warning=True)
    initplus.append(qecc.gate("ideal init |+>"))

    if syn_extract is not None and logical_gate is not None:
        msg = "Both syn_extract and logical_gate cannot be set (not None)."
        raise Exception(msg)

    if syn_extract is None:
        # Syndrome extraction
        syn_extract = LogicalCircuit(suppress_warning=True)
        syn_extract.append(qecc.gate("I", num_syn_extract=1, forced_outcome=1))

    logic = syn_extract if logical_gate is None else logical_gate

    logical_ops_zero = qecc.instruction("instr_init_zero").logical_stabs[0]
    logical_ops_plus = qecc.instruction("instr_init_plus").logical_stabs[0]

    if decoder is None:
        decoder = MWPM2D(qecc)

    for qubit_comb in powerset(qudit_set):
        if len(qubit_comb) > t_weight:
            break

        error_combinations = product(error_set, repeat=len(qubit_comb))

        for error_comb in error_combinations:
            error_circ = QuantumCircuit(1)
            errors = ErrorCircuits()

            errors.simple_add(0, 0, 0, before_errors=error_circ)

            for e, q in zip(error_comb, qubit_comb, strict=False):
                error_circ.update(e, {q})

            state_zero = SparseSimPy(qecc.num_qudits)
            state_plus = SparseSimPy(qecc.num_qudits)

            circ_sim.run(state_zero, initzero)
            circ_sim.run(state_plus, initplus)

            output, _ = circ_sim.run(state_zero, logic, error_circuits=errors)
            circ_sim.run(state_plus, logic, error_circuits=errors)

            syn = output.simplified(last=True)

            if syn:
                # Recovery operation
                recovery = decoder.decode(syn)
                circ_sim.run(state_zero, recovery)
                circ_sim.run(state_plus, recovery)

            sign_zero = state_zero.logical_sign(*logical_ops_zero)
            sign_plus = state_plus.logical_sign(*logical_ops_plus)

            if sign_zero or sign_plus:
                if verbose:
                    print(errors)
                return False, len(error_comb)

            if logical_gate is None:  # The following is only required for EC.
                # Any remaining syndromes?
                output, _ = circ_sim.run(state_zero, syn_extract)
                syn = output.simplified(last=True)

                if syn:
                    if verbose:
                        print(f"syndromes = {syn}")
                        print(errors)
                    return False, len(error_comb)

    return True, int(t_weight)


def fault_check(
    qecc: QECCProtocol,
    logical_gate: QuantumCircuit | LogicalCircuit | None = None,
    decoder: Decoder | None = None,
    t_weight: int | None = None,
    error_set: Iterable[tuple[set[int], set[int]]] | None = None,
    *,
    verbose: bool = True,
    data_errors: bool = True,
    ancilla_errors: bool = False,
) -> tuple[bool, int]:
    """Check exRec conditions for faulty error correction or logical gate.

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
    ----
        qecc: The quantum error correcting code instance.
        logical_gate(QuantumCircuit): The logical gate circuit to test (None for error correction only).
        decoder: The decoder instance for error correction.
        t_weight: The maximum weight of errors to check (typically pc.floor((distance-1)/2)).
        error_set: Custom set of errors to check (if None, all Pauli errors are checked).
        verbose: If True, prints detailed information about failures.
        data_errors: If True, includes errors on data qubits.
        ancilla_errors: If True, includes errors on ancilla qubits.

    Returns:
    -------
        tuple (bool, int): The bool is whether the check is passed. The int is the weight of error last checked. If the
        bool is True then int == t_weight. If bool == False, int == weight of error that caused a logical error.

    """
    qudit_set = set()

    if data_errors:
        qudit_set.update(qecc.data_qudit_set)

    if ancilla_errors:
        qudit_set.update(qecc.ancilla_qudit_set)

    if t_weight is None:
        t_weight = pc.floor((qecc.distance - 1) / 2)

    if error_set is None:
        error_set = {"X", "Y", "Z"}

    circ_sim = Standard()

    # init |0> circuit
    initzero = LogicalCircuit(suppress_warning=True)
    initzero.append(qecc.gate("ideal init |0>"))

    # init |+> circuit
    initplus = LogicalCircuit(suppress_warning=True)
    initplus.append(qecc.gate("ideal init |+>"))

    if logical_gate is None:
        # Syndrome extraction
        syn_extract = LogicalCircuit(suppress_warning=True)
        syn_extract.append(qecc.gate("I", num_syn_extract=1, forced_outcome=1))
        logic = syn_extract
    else:
        logic = logical_gate

    logical_ops_zero = qecc.instruction("instr_init_zero").logical_stabs[0]
    logical_ops_plus = qecc.instruction("instr_init_plus").logical_stabs[0]

    if decoder is None:
        decoder = MWPM2D(qecc)

    for qubit_comb in powerset(qudit_set):
        if len(qubit_comb) > t_weight:
            break

        error_combinations = product(error_set, repeat=len(qubit_comb))

        for error_comb in error_combinations:
            error_circ = QuantumCircuit(1)
            errors = ErrorCircuits()

            errors.simple_add(0, 0, 0, before_errors=error_circ)

            for e, q in zip(error_comb, qubit_comb, strict=False):
                error_circ.update(e, {q})

            state_zero = SparseSimPy(qecc.num_qudits)
            state_plus = SparseSimPy(qecc.num_qudits)

            circ_sim.run(state_zero, initzero)
            circ_sim.run(state_plus, initplus)

            output, _ = circ_sim.run(state_zero, logic, error_circuits=errors)
            circ_sim.run(state_plus, logic, error_circuits=errors)

            syn = output.simplified(last=True)

            if syn:
                # Recovery operation
                recovery = decoder.decode(syn)
                circ_sim.run(state_zero, recovery)
                circ_sim.run(state_plus, recovery)

            sign_zero = state_zero.logical_sign(*logical_ops_zero)
            sign_plus = state_plus.logical_sign(*logical_ops_plus)

            if sign_zero or sign_plus:
                if verbose:
                    print(errors)
                return False, len(error_comb)

    return True, int(t_weight)


def distance_check(
    qecc: QECCProtocol,
    mode: str | None = None,
    dist_mode: str | None = None,
) -> int:
    """Determines the distance of the code by looking for the smallest logical errors.

    Args:
    ----
        qecc: The quantum error correcting code instance.
        mode: The mode for distance checking ('X', 'x', 'Z', 'z', or None for automatic).
        dist_mode: The specific distance checking mode to use (if None, uses default based on mode).

    Returns:
    -------
        Tuple (bool, int). The bool is whether the check is passed. The int is the weight of error last checked. If the
        bool is True then int == t_weight. If bool == False, int == weight of error that caused a logical error.

    """
    qudit_set = qecc.data_qudit_set

    circ_sim = Standard()
    state = SparseSimPy(qecc.num_qudits)

    ideal_initlogic = LogicalCircuit(suppress_warning=True)
    ideal_initlogic.append(qecc.gate("ideal init |0>"))

    circ_sim.run(state, ideal_initlogic)

    logical_op, delogical_op = qecc.instruction("instr_init_zero").logical_stabs[0]

    destab_xs, destab_zs = circ2set(delogical_op.items(params=False))
    stab_xs, stab_zs = circ2set(logical_op.items(params=False))

    remove_stab(state, stab_xs, stab_zs, destab_xs, destab_zs)

    if dist_mode is None:
        if mode in {"X", "x"}:
            print("x")
            return dist_mode_x(state, qudit_set)
        if mode in {"Z", "z"}:
            print("z")
            return dist_mode_z(state, qudit_set)
        if mode == "power":
            return dist_mode_powerset(state, qudit_set)
        return dist_mode_smallest(state, qudit_set)

    return dist_mode(state, qudit_set)


def dist_mode_powerset(state: SimulatorProtocol, qudit_set: set[int]) -> str | bool:
    """Check for logical errors using powerset of all possible X and Z errors.

    Args:
    state: Stabilizer state to check.
    qudit_set: Set of qudit/qubit indices to check for errors.
    """
    for x_errors in powerset(qudit_set):
        for z_errors in powerset(qudit_set):
            if op_commutes(x_errors, z_errors, state.stabs) and not find_stab(
                state,
                x_errors,
                z_errors,
            ):
                return f"Logical error found: Xs - {x_errors} Zs - {z_errors}"

    return False


def dist_mode_smallest(state: SimulatorProtocol, qudit_set: set[int]) -> str | bool:
    """Find smallest logical error by checking errors in increasing size.

    Args:
    ----
        state: Stabilizer state to check.
        qudit_set: Set of qudit/qubit indices to check for errors.
    """
    for lenq in range(len(qudit_set) + 1):
        for qs in combinations(qudit_set, lenq):
            if op_commutes(qs, qs, state.stabs) and not find_stab(state, qs, qs):
                return f"Logical error found: Xs - {qs} Zs - {qs}"

            for qs2 in powerset(qudit_set, len(qs) - 1):
                if op_commutes(qs2, qs, state.stabs) and not find_stab(state, qs2, qs):
                    return f"Logical error found: Xs - {qs2} Zs - {qs}"

                if op_commutes(qs, qs2, state.stabs) and not find_stab(state, qs, qs2):
                    return f"Logical error found: Xs - {qs} Zs - {qs2}"

    return False


def dist_mode_x(state: SimulatorProtocol, qudit_set: set[int]) -> str | bool:
    """Check for X-type logical errors only.

    Args:
    ----
        state: Stabilizer state to check.
        qudit_set: Set of qudit/qubit indices to check for errors.
    """
    z_errors = ()
    for x_errors in powerset(qudit_set):
        if op_commutes(x_errors, z_errors, state.stabs) and not find_stab(
            state,
            x_errors,
            z_errors,
        ):
            return f"Logical error found: Xs - {x_errors} Zs - {z_errors}"

    return False


def dist_mode_z(state: SimulatorProtocol, qudit_set: set[int]) -> str | bool:
    """Check for Z-type logical errors only.

    Args:
    ----
        state: Stabilizer state to check.
        qudit_set: Set of qudit/qubit indices to check for errors.
    """
    x_errors = ()
    for z_errors in powerset(qudit_set):
        if op_commutes(x_errors, z_errors, state.stabs) and not find_stab(
            state,
            x_errors,
            z_errors,
        ):
            return f"Logical error found: Xs - {x_errors} Zs - {z_errors}"

    return False
