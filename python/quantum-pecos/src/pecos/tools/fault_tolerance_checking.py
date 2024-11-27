# Copyright 2022 The PECOS Developers
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

from itertools import permutations, product
from typing import TYPE_CHECKING, Callable

from pecos import QuantumCircuit
from pecos.engines.circuit_runners import Standard
from pecos.simulators import SparseSim

if TYPE_CHECKING:
    from collections.abc import Sequence


def find_pauli_fault(
    qcirc: QuantumCircuit,
    wt: int,
    fail_func: Callable,
    num_qubits: int | None = None,
    simulator: str = "stabilizer",
    *,
    verbose: bool = True,
    failure_break: bool = True,
) -> list[dict | tuple[dict, dict]]:
    """Determines if there is a Pauli fault for the entire quantum circuit.

    TODO: Need to be able to only check a portion of the circuit.
    TODO: Need to add Wasm support.

    Args:
    ----
        qcirc: QuantumCircuit
        wt: Number of errors to apply.
        fail_func: A callable (e.g., function) that determines if a result fails.
        num_qubits: Number of qubits in the circuit.
        simulator: Simulator used to generate the state.
        verbose: Whether to print out information.
        failure_break: Whether to break at first failure.

    Returns:
    -------
        A list of errors that caused failure

    """
    failure = False

    failure_circs = []

    """
    if simulator is None:
        simulator = SparseSim
    """

    if num_qubits is None:
        num_qubits = qcirc.metadata["num_qubits"]

    for err in get_wt_paulis(qcirc, wt, make_qc=True):
        # output = run_sim(qcirc, shots=1, simulator=simulator, error_circuits=err)
        circ_sim = Standard()
        state = SparseSim(num_qubits)
        output, _ = circ_sim.run(state, qcirc, error_circuits=err)

        if fail_func(output):
            failure = True

            if verbose:
                print("Results:", output)
                print()
                print("Error:", err)
                print("-------------")
                print()

            failure_circs.append(err)

            if failure_break:
                break

    if not failure and verbose:
        print(f"Fault tolerant to {wt} gates going bad!")

    return failure_circs


def get_all_spacetime(
    qcirc: QuantumCircuit,
    initial_qubits: Sequence[int] | None = None,
):
    """Determine all the spacetime locations of gates/error events."""
    if initial_qubits is not None:
        for q in initial_qubits:
            yield {
                "tick": -1,
                "location": (q,),
                "before": True,
                "symbol": "init |0>",
                "metadata": {},
            }

    for gates, tick, _ in qcirc.iter_ticks():
        for sym, locations, metadata in gates.items():
            for loc in locations:
                if isinstance(loc, int):
                    loc = (loc,)

                yield {
                    "tick": tick,
                    "location": loc,
                    "before": sym.startswith("meas"),
                    "symbol": sym,
                    "metadata": metadata,
                }


def get_wt_paulis(
    circ: QuantumCircuit,
    wt: int,
    initial_qubits: Sequence[int] | None = None,
    *,
    make_qc: bool = True,
):
    """A generator of all combinations of Pauli faults of a given weight.

    Args:
    ----
        circ:
        wt:
        initial_qubits:
        make_qc:

    Returns:
    -------

    """
    # get the spacetime locations that will have errors
    for gate_data in permutations(get_all_spacetime(circ, initial_qubits), wt):
        iter_list = []
        tick_list = []
        loc_list = []
        before_list = []
        cond_list = []

        # for the error locations, create the pauli error iterators
        for gate_dict in gate_data:
            # tick, locs, after
            tick = gate_dict["tick"]
            locs = gate_dict["location"]
            before = gate_dict["before"]

            paulis = product("IXYZ", repeat=len(locs))
            next(paulis)  # skip ('I', ..., 'I'), which is the first element
            iter_list.append(paulis)

            tick_list.append(tick)
            loc_list.append(locs)
            before_list.append(before)
            cond = gate_dict["metadata"].get("cond")
            cond_list.append(cond)

        # get all combinations of these possible pauli errors
        for pauli_errs in product(*iter_list):
            tick_dict_after = {}
            tick_dict_before = {}
            cond_dict = {}

            for errs, tick, locs, before, cond in zip(
                pauli_errs,
                tick_list,
                loc_list,
                before_list,
                cond_list,
            ):
                tick_dict = tick_dict_before if before else tick_dict_after

                gate_dict = tick_dict.setdefault(tick, {})
                cond_dict[tick] = cond

                for p, q in zip(errs, locs):
                    if p != "I":
                        loc_set = gate_dict.setdefault(p, set())
                        loc_set.add(q)

            if make_qc:
                error = {}

                for t, pdict in tick_dict_after.items():
                    error_tick = error.setdefault(t, {})
                    if cond_dict.get(t):
                        qc = QuantumCircuit()
                        qc.append(pdict, cond=cond_dict.get(t))
                        error_tick["after"] = qc
                    else:
                        error_tick["after"] = QuantumCircuit([pdict])

                for t, pdict in tick_dict_before.items():
                    error_tick = error.setdefault(t, {})
                    if cond_dict.get(t):
                        qc = QuantumCircuit()
                        qc.append(pdict, cond=cond_dict.get(t))
                        error_tick["before"] = qc
                    else:
                        error_tick["before"] = QuantumCircuit([pdict])

                yield error

            else:
                yield tick_dict_after, tick_dict_before
