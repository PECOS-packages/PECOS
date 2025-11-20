"""Stabilizer verification tools for quantum error correction.

This module provides utilities for verifying stabilizer codes and analyzing
their properties, including stabilizer group verification, code distance
calculation, and logical operator validation.
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

from itertools import combinations, product
from typing import TYPE_CHECKING

import pecos as pc
from pecos.circuits import QuantumCircuit

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pecos.protocols import SimulatorProtocol
    from pecos.typing import LogicalOpInfo, StabilizerVerificationResult

# TODO: NEED TO ADD SIGN TRACKING TO DESTABILIZERS TO GET THE RIGHT SIGN FOR LOGICAL Xs


class VerifyStabilizers:
    """Used to define a stabilizer QECC."""

    def __init__(self) -> None:
        """Initialize the VerifyStabilizers instance.

        Sets up the circuit simulator and initializes empty data structures
        for stabilizer checks, logical operators, and qubit tracking.
        """
        self.circ_sim = pc.simulators.SparseSimPy

        self.checks = []
        self.logical_zs = []
        self.logical_xs = []
        self.logical_zs_defined = []  # User chosen logical Zs TODO: ...
        self.logical_xs_defined = []  # User chosen logical Xs TODO: ...
        self.logical_zs_reference = {}
        self.logical_xs_reference = {}

        self.data_qubits = set()
        self.ancilla_qubits = set()
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

    def check(self, paulis: str | Sequence[str], qubits: Sequence[int]) -> None:
        """Check if the given Pauli operators stabilize the state.

        Args:
            paulis: Sequence of Pauli operators to check (e.g., ['X', 'Z', 'Y']).
            qubits: Sequence of qubit indices corresponding to the Pauli operators.

        Returns: None

        """
        if not qubits:
            msg = "No qubit ids given."
            raise Exception(msg)

        check_string = "check(" + str(paulis) + ", " + str(qubits) + ")"

        if isinstance(paulis, str):
            if paulis not in {"X", "x", "Y", "y", "Z", "z"}:
                msg = 'Paulis should be "X", "Y" or "Z"!'
                raise Exception(msg)

            paulis_new = [paulis for _ in qubits]
            paulis = paulis_new

        if not isinstance(paulis, str) and len(paulis) != len(qubits):
            msg = "Number of Paulis and qubits do not match!!!"
            raise Exception(msg)

        self.checks.append((paulis, qubits, check_string))
        self.data_qubits.update(qubits)

    def logicalz(self, paulis: str | Sequence[str], qubits: Sequence[int]) -> None:
        """Used to define logical Z.

        Args:
        ----
            paulis: Either a single Pauli string ('X', 'Y', or 'Z') or a list of Pauli operators.
            qubits: List of qubit indices where the logical Z operator acts.
        """
        if not qubits:
            msg = "No qubit ids given."
            raise Exception(msg)

        logical_string = "check(" + str(paulis) + ", " + str(qubits) + ")"

        if isinstance(paulis, str):
            if paulis not in {"X", "x", "Y", "y", "Z", "z"}:
                msg = 'Paulis should be "X", "Y" or "Z"!'
                raise Exception(msg)

            paulis_new = [paulis for _ in qubits]
            paulis = paulis_new

        if not isinstance(paulis, str) and len(paulis) != len(qubits):
            msg = "Number of Paulis and qubits do not match!!!"
            raise Exception(msg)

        self.logical_zs.append((paulis, qubits, logical_string))

    def logicalx(self, paulis: str | Sequence[str], qubits: Sequence[int]) -> None:
        """Used to define logical X.

        Args:
        ----
            paulis: Either a single Pauli string ('X', 'Y', or 'Z') or a list of Pauli operators.
            qubits: List of qubit indices where the logical X operator acts.
        """
        if not qubits:
            msg = "No qubit ids given."
            raise Exception(msg)

        logical_string = "check(" + str(paulis) + ", " + str(qubits) + ")"

        if isinstance(paulis, str):
            if paulis not in {"X", "x", "Y", "y", "Z", "z"}:
                msg = 'Paulis should be "X", "Y" or "Z"!'
                raise Exception(msg)

            paulis_new = [paulis for _ in qubits]
            paulis = paulis_new

        if not isinstance(paulis, str) and len(paulis) != len(qubits):
            msg = "Number of Paulis and qubits do not match!!!"
            raise Exception(msg)

        self.logical_xs.append((paulis, qubits, logical_string))

    def num_logical_qubits(self) -> int:
        """Calculate the number of logical qubits in the stabilizer code.

        Returns:
            Number of logical qubits (data qubits minus stabilizer checks).
        """
        return len(self.data_qubits) - len(self.checks)

    def generators(
        self,
        *,
        print_y: bool = True,
        verbose: bool = True,
    ) -> tuple[
        list[dict[str, set[int]]],
        list[dict[str, set[int]]],
        list[str],
        list[str],
    ]:
        """Evaluates the stabilizer generators that have been supplied via the `check` method.

        Args:
        ----
            print_y: If True, includes Y operators in the output (otherwise converts to X and Z).
            verbose: If True, prints detailed information about the generators.
        """
        if self.circuit is None:
            msg = "Must compile circuits first!"
            raise Exception(msg)

        state = self.state
        z, x, stab_strings, destab_strings = self.get_info(
            state,
            print_y=print_y,
            verbose=verbose,
        )

        return z, x, stab_strings, destab_strings

    def _check_all_labels(self) -> None:
        """This checks to see that all the consecutive qubit ids have been used and none are missing."""
        qubit_labels = set()

        checks = self.checks
        for check in checks:
            _, qs, _ = check
            qubit_labels.update(qs)

        largest_labels = max(qubit_labels)
        labels_should_have = set(range(largest_labels + 1))

        dont_have = labels_should_have - qubit_labels

        if dont_have:
            msg = f"Qubit ids missing: {dont_have}"
            raise Exception(msg)

    def _check2rowcol(self) -> None:
        """Creates row and column matrices."""
        checks = self.checks

        num_checks = len(checks)
        row_x = [set() for _ in range(num_checks)]
        row_z = [set() for _ in range(num_checks)]
        col_x = [set() for _ in range(self.num_data_qubits)]
        col_z = [set() for _ in range(self.num_data_qubits)]

        for stab_id, check in enumerate(checks):
            ps, qs, _ = check

            for p, q in zip(ps, qs, strict=False):
                if p in {"X", "x"}:
                    row_x[stab_id].add(q)
                    col_x[q].add(stab_id)

                elif p in {"Z", "z"}:
                    row_z[stab_id].add(q)
                    col_z[q].add(stab_id)

                elif p in {"Y", "y"}:
                    row_x[stab_id].add(q)
                    row_z[stab_id].add(q)
                    col_x[q].add(stab_id)
                    col_z[q].add(stab_id)

        self.check_row_x = row_x
        self.check_row_z = row_z
        self.check_col_x = col_x
        self.check_col_z = col_z

    def _check_commute(self) -> bool:
        """Checks to see that all the stabilizer generators commute.

        Returns:
            Returns bool value if all the checks commute or not.
        """
        row_x = self.check_row_x
        row_z = self.check_row_z
        col_x = self.check_col_x
        col_z = self.check_col_z

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
                print("\nChecks anticommute!")
                print("\nCheck:")
                for s in anti:
                    print(self.checks[s][2])
                print("\nanticommutes with:")
                print(self.checks[stab_id][2])

                msg = "Checks anticommute!"
                raise Exception(msg)
        return True

    def compile(self) -> None:
        """Checks commutation relations and creates a circuit to measure the checks."""
        if self.circuit:
            msg = "Measurement encoding-circuit has already been compiled!"
            raise Exception(msg)

        # Check the qubit ids
        self._check_all_labels()

        # Create row and column matrices.
        self._check2rowcol()

        # Checks that all the stabilizer generators (checks) commute.
        self._check_commute()

        # Create check circuits:
        # ----------------------
        ancilla_qubits = set()
        qc = QuantumCircuit()

        ancilla_id = sorted(self.data_qubits)[-1]

        for ps, qs, _ in self.checks:
            ancilla_id += 1
            ancilla_qubits.add(ancilla_id)
            qc.append("init |+>", {ancilla_id})

            for p, q in zip(ps, qs, strict=False):
                symbol = None

                if p in {"X", "x"}:
                    symbol = "CNOT"
                elif p in {"Z", "z"}:
                    symbol = "CZ"
                elif p in {"Y", "y"}:
                    symbol = "CY"

                qc.append(symbol, {(ancilla_id, q)})

            qc.append("measure X", {ancilla_id}, random_outcome=0)

        self.ancilla_qubits = ancilla_qubits

        self.circuit = qc

        # Run circuits
        # ------------
        # Separate the checks, logical stabilizers, and ancilla stabilizers.
        circuit = self.circuit
        state = pc.simulators.SparseSimPy(self.num_qubits)
        state.run_circuit(circuit)
        self.get_info(state, verbose=False)
        self.state = state

        self._verify_checks()
        self._check_logical_commute()

    def _check_logical_commute(self) -> None:
        logical_z_col_x = [set() for _ in range(self.num_data_qubits)]
        logical_z_col_z = [set() for _ in range(self.num_data_qubits)]
        logical_z_row_x = [set() for _ in range(len(self.logical_zs))]
        logical_z_row_z = [set() for _ in range(len(self.logical_zs))]

        logical_x_col_x = [set() for _ in range(self.num_data_qubits)]
        logical_x_col_z = [set() for _ in range(self.num_data_qubits)]
        logical_x_row_x = [set() for _ in range(len(self.logical_xs))]
        logical_x_row_z = [set() for _ in range(len(self.logical_xs))]

        for i, (ps, qs, _) in enumerate(self.logical_zs):
            for p, q in zip(ps, qs, strict=False):
                if p in {"X", "Y"}:
                    logical_z_col_x[q].add(i)
                    logical_z_row_x[i].add(q)

                if p in {"Z", "Y"}:
                    logical_z_col_z[q].add(i)
                    logical_z_row_z[i].add(q)

        for i, (ps, qs, _) in enumerate(self.logical_xs):
            for p, q in zip(ps, qs, strict=False):
                if p in {"X", "Y"}:
                    logical_x_col_x[q].add(i)
                    logical_x_row_x[i].add(q)

                if p in {"Z", "Y"}:
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
                print("\nLogical Zs anticommute!")
                print("\nLogical Zs:")
                for i in anti:
                    print(self.logical_zs[i][2])
                print("\nanticommutes with:")
                print(self.logical_zs[s][2])

                msg = "Logical Zs anticommute!"
                raise Exception(msg)

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
                print("\nLogical Xs anticommute!")
                print("\nLogical Xs:")
                for i in anti:
                    print(self.logical_xs[i][2])
                print("\nanticommutes with:")
                print(self.logical_xs[s][2])

                msg = "Logical Xs anticommute!"
                raise Exception(msg)

        # So far checked that all the logical Zs and logical Xs commute with themselves...
        # - Next check that they commute with the stabilizers...
        # - Then find if the there are anti-commuting pairs of logical Zs and Xs
        # - Then search for the logical operators and refactor...
        # This step might require switching logical Xs and Zs... Might be a bit complicated... as we can modify
        # "logical Xs" with destabilizers and swap those... So need to do a search through all stabilizers and
        # destabilizers and then determine if the required multiplication is valid with fix stabiliziers and whatever
        # has been fixed for the logical operators...

    def _verify_checks(self) -> bool:
        # Stabilizers:
        checks = []

        for strings, qids, _ in self.checks:
            check_dict = {}
            for pauli, q in zip(strings, qids, strict=False):
                if pauli == "X":
                    qset = check_dict.setdefault("X", set())
                elif pauli == "Z":
                    qset = check_dict.setdefault("Z", set())
                else:
                    qset = check_dict.setdefault("Y", set())
                qset.add(q)
            checks.append(check_dict)

        checks2 = []
        for i in range(len(self.checks)):
            xs = self.check_row_x[i]
            zs = self.check_row_z[i]

            stab_dict = {}

            if xs - zs:
                stab_dict["X"] = xs - zs

            if zs - xs:
                stab_dict["Z"] = zs - xs

            if xs & zs:
                stab_dict["Y"] = xs & zs

            checks2.append(stab_dict)

        if checks != checks2:
            print(
                "WARNING: PECOS didn't refactor the stabilizers into the checks supplied!",
            )

        return checks != checks2

    def eval(self, *, verbose: bool = False) -> StabilizerVerificationResult:
        """Evaluate the stabilizer code verification.

        Args:
            verbose: Whether to print detailed output during evaluation.

        Returns:
            Verification result containing success status and error details.
        """
        if self.circuit is None:
            self.compile()

        z, x, _, destab_strings = self.generators(verbose=verbose)

        if self.dist is None:
            self.distance(verbose=verbose)

        # Stabilizers:
        checks = []

        for strings, qids, _ in self.checks:
            check_dict = {}
            for pauli, q in zip(strings, qids, strict=False):
                if pauli == "X":
                    qset = check_dict.setdefault("X", set())
                elif pauli == "Z":
                    qset = check_dict.setdefault("Z", set())
                else:
                    qset = check_dict.setdefault("Y", set())
                qset.add(q)
            checks.append(check_dict)

        # Destabilizers:
        destabs = []
        for i in self.data_gens:
            destab_dict = {}
            for j in range(len(destab_strings[i]) - self.num_ancilla_qubits):
                pauli = destab_strings[i][j]
                if pauli == "X":
                    qset = destab_dict.setdefault("X", set())
                elif pauli == "Z":
                    qset = destab_dict.setdefault("Z", set())
                elif pauli == "Y":
                    qset = destab_dict.setdefault("Y", set())
                else:
                    continue
                qset.add(j - 2)
            destabs.append(destab_dict)

        output_dict = {
            "num_datas": self.num_data_qubits,
            "num_logical_qubits": self.num_logical_qubits(),
            "distance": self.dist,
            "[[n, k, d]]": f"[[{self.num_data_qubits}, {self.num_logical_qubits()}, {self.dist}]]",
            "checks": checks,
            "destabilizers": destabs,
            "logical_xs": x,
            "logical_zs": z,
        }

        self.logical_xs_reference = {}
        self.logical_zs_reference = {}

        for i, xi in enumerate(x):
            self.logical_xs_reference["X" + str(i)] = xi

        for i, zi in enumerate(z):
            self.logical_zs_reference["Z" + str(i)] = zi

        return output_dict

    @property
    def num_data_qubits(self) -> int:
        """Get the number of data qubits.

        Returns:
            Number of data qubits in the stabilizer code.
        """
        return len(self.data_qubits)

    @property
    def num_ancilla_qubits(self) -> int:
        """Get the number of ancilla qubits.

        Returns:
            Number of ancilla qubits in the stabilizer code.
        """
        return len(self.ancilla_qubits)

    @property
    def num_qubits(self) -> int:
        """Get the total number of qubits.

        Returns:
            Total number of qubits (data + ancilla).
        """
        return len(self.data_qubits) + len(self.ancilla_qubits)

    def refactor(self, state: SimulatorProtocol) -> None:
        """Refactor the stabilizer state to match the expected generators.

        Args:
            state: Simulator state to refactor.
        """
        found_stab_ids = set()

        refactor_things = list(self.checks)
        refactor_things.extend(self.logical_zs)
        # TODO: NEED TO REFACTOR THE DESTABILIZER OF LOGICAL Z TO GET THE RIGHT LOGICAL X.....

        for ps, qs, _ in refactor_things:
            xs = set()
            zs = set()

            for p, q in zip(ps, qs, strict=False):
                if p in {"X", "x"}:
                    xs.add(q)
                elif p in {"Z", "z"}:
                    zs.add(q)
                elif p in {"Y", "y"}:
                    xs.add(q)
                    zs.add(q)

            try:
                found, stab_id = state.refactor(
                    xs,
                    zs,
                    choose=0,
                    protected=found_stab_ids,
                )
            except IndexError:
                xonly = xs - zs
                zonly = zs - xs
                ys = xs & zs
                msg = f"IndexError.\nThe stabilizer {{'X': {xonly}, 'Y': {ys}, 'Z': {zonly}}} is likely redundant!"
                raise Exception(msg) from IndexError

            found_stab_ids.add(stab_id)

            if not found:
                msg = "Could not find check:"
                raise Exception(msg, (ps, qs))

        for q in self.ancilla_qubits:
            found, stab_id = state.refactor(
                {q},
                set(),
                choose=-1,
                protected=found_stab_ids,
            )
            found_stab_ids.add(stab_id)

            if not found:
                msg = f"Could not find ancilla {q}"
                raise Exception(msg)

    def get_check_ancilla(
        self,
    ) -> tuple[list[tuple[set[int], set[int]]], list[tuple[set[int], set[int]]]]:
        """Get check and ancilla operator sets.

        Returns:
            Tuple containing lists of (X set, Z set) tuples for checks and ancillas.
        """
        check_tuples = []
        ancilla_tuples = []

        for ps, qs, _ in self.checks:
            xs = set()
            zs = set()

            for p, q in zip(ps, qs, strict=False):
                if p in {"X", "x"}:
                    xs.add(q)
                elif p in {"Z", "z"}:
                    zs.add(q)
                elif p in {"Y", "y"}:
                    xs.add(q)
                    zs.add(q)

            check_tuples.append((xs, zs))

        ancilla_tuples.extend(({q}, set()) for q in self.ancilla_qubits)

        return check_tuples, ancilla_tuples

    def get_info(
        self,
        state: SimulatorProtocol,
        stop_search: int = 1000,
        *,
        verbose: bool = True,
        print_y: bool = False,
    ) -> tuple[
        list[dict[str, set[int]]],
        list[dict[str, set[int]]],
        list[str],
        list[str],
    ]:
        """Get stabilizer information from the quantum state.

        Args:
            state: Simulator state to analyze.
            stop_search: Maximum number of refactoring attempts.
            verbose: Whether to print detailed information.
            print_y: Whether to include Y operators in output.

        Returns:
            Tuple of logical Z operators, logical X operators, stabilizer strings, destabilizer strings.
        """
        if self.circuit is None:
            return Exception("Must run `compile()` first!")

        self.refactor(state)
        stab_strs, destab_strs = state.print_stabs(
            verbose=False,
            print_y=print_y,
            print_destabs=True,
        )

        num_ancillas = len(self.ancilla_qubits)

        num_logical = self.num_logical_qubits()
        num_checks = len(self.checks)

        if verbose:
            print(f"Number of data qubits: {self.num_data_qubits}")
            print(f"Number of checks: {num_checks}")
            print(f"Number of logical qubits: {num_logical}")

        check_tuples, ancilla_tuples = self.get_check_ancilla()
        # determine the gen_id of the checks and logicals
        check_gens = []
        logical_gens = []
        ancilla_gens = []

        missing_checks = list(check_tuples)
        missing_ancillas = list(ancilla_tuples)
        notmatched_gens = list(range(state.num_qubits))

        found_all = False
        search_count = 0
        while not found_all:
            if verbose:
                print("----")
            for g, gtuple in enumerate(
                zip(state.stabs.row_x, state.stabs.row_z, strict=False),
            ):
                if gtuple in missing_checks:
                    missing_checks.remove(gtuple)
                    try:
                        notmatched_gens.remove(g)
                    except ValueError:
                        msg = f"list.remove(x): x not in list.\nThe stabilizer {gtuple!s} is likely redundant!"
                        raise Exception(
                            msg,
                        ) from ValueError

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
                state.print_stabs(verbose=False, print_y=print_y, print_destabs=True)

            if search_count == stop_search:
                msg = "Can not refactor properly!"
                raise Exception(msg)
            search_count += 1

        self.data_gens = set(check_gens)
        self.ancilla_gens = set(ancilla_gens)
        self.logical_gens = set(logical_gens)

        if verbose:
            if len(check_gens) != num_checks:
                print("Found:", check_gens)
                print("Want:", check_tuples)
                msg = f"Did not find the correct number of stabilizer generators. {len(check_gens)}/{num_checks}"
                raise Exception(msg)

            if len(logical_gens) != num_logical:
                print("Found:", logical_gens)
                msg = f"Did not find the correct number of logical generators. {len(logical_gens)}/{num_logical}"
                raise Exception(msg)

            print("\nStabilizer generators:")
            for gen in check_gens:
                print(stab_strs[gen][: len(stab_strs[gen]) - num_ancillas])

            print("\nDestabilizer generators:")
            for gen in check_gens:
                print(destab_strs[gen][: len(destab_strs[gen]) - num_ancillas])

            print("\nLogical operators:")

            for i, gen in enumerate(logical_gens):
                print(f"\n. Logical Z #{i + 1!s}:")
                print(stab_strs[gen][: len(stab_strs[gen]) - num_ancillas])
                print(f". Logical X #{i + 1!s}:")
                print(destab_strs[gen][: len(destab_strs[gen]) - num_ancillas])

        logical_z_strings = []
        logical_x_strings = []

        for gen in logical_gens:
            z_string = stab_strs[gen][: len(stab_strs[gen]) - num_ancillas]
            x_string = destab_strs[gen][: len(destab_strs[gen]) - num_ancillas]

            check_dict = {}
            for q, pauli in enumerate(z_string):
                if pauli == "X":
                    qset = check_dict.setdefault("X", set())
                elif pauli == "Z":
                    qset = check_dict.setdefault("Z", set())
                elif pauli == "Y":
                    qset = check_dict.setdefault("Y", set())
                else:
                    continue
                qset.add(q - 2)
            logical_z_strings.append(check_dict)

            check_dict = {}
            for q, pauli in enumerate(x_string):
                if pauli == "X":
                    qset = check_dict.setdefault("X", set())
                elif pauli == "Z":
                    qset = check_dict.setdefault("Z", set())
                elif pauli == "Y":
                    qset = check_dict.setdefault("Y", set())
                else:
                    continue

                qset.add(q - 2)
            logical_x_strings.append(check_dict)

        return logical_z_strings, logical_x_strings, stab_strs, destab_strs

    def distance(
        self,
        *,
        css: bool = False,
        verbose: bool = True,
    ) -> tuple[set[int], set[int]] | None:
        """Checks the distance of the code."""
        if self.circuit is None:
            msg = "Must compile circuits first!"
            raise Exception(msg)

        qudit_set = self.data_qubits

        state = self.state
        found = self._dist_mode_smallest(state, qudit_set, css=css, verbose=verbose)

        if verbose and found:
            xs, zs = found
            distance = len(xs | zs)

            print(
                f"\nThis is a [[{self.num_data_qubits}, {self.num_logical_qubits()}, {distance}]] code.",
            )

        if not found:
            print(
                "No logical errors found... Checks might describe a stabilizer state.",
            )
            return None
        xs, zs = found
        self.dist = len(xs | zs)
        return found

    def _dist_mode_smallest(
        self,
        state: SimulatorProtocol,
        qudit_set: set[int],
        *,
        css: bool = False,
        verbose: bool = True,
        start_len: int | None = None,
        end_len: int | None = None,
        list_ops: bool = False,
    ) -> Generator[tuple[set, set], None, None]:
        """Determine if a logical error can be found by starting with the smallest weight errors.

        Args:
        ----
            state: The quantum state to check for logical errors.
            qudit_set: Set of qudit indices to consider for errors.
            css: If True, only checks CSS (Calderbank-Shor-Steane) type errors.
            verbose: If True, prints progress and found logical operators.
            start_len: Starting weight of errors to check (default: 1).
            end_len: Maximum weight of errors to check (default: number of qudits).
            list_ops: If True, returns a list of all logical operators found.

        """
        ops = []

        if start_len is None:
            start_len = 1

        if end_len is None:
            end_len = len(qudit_set)

        for lenq in range(start_len, end_len + 1):
            if verbose:
                print(f"Checking Paulis of weight {lenq}...")

            for xs, zs in self.gen_errors(qudit_set, lenq, lenq, css=css):
                if self._is_logical_error(state, xs, zs):
                    if verbose:
                        print(f"Logical operator found: Xs - {xs} Zs - {zs}")

                    if list_ops:
                        ops.append({"X": xs, "Z": zs})
                    else:
                        return xs, zs

        return ops

    def gen_errors(
        self,
        qubits: set[int] | Sequence[int],
        min_errors: int = 1,
        *,
        max_errors: bool | int = False,
        css: bool = False,
    ) -> Generator[tuple[set[int], set[int]], None, None]:
        """Generate error patterns for testing stabilizer codes.

        Args:
        ----
            qubits (set of int): Set of qubit indices to generate errors on.
            min_errors (int): Minimum number of errors to generate.
            max_errors (bool, int): Maximum number of errors to generate. False for no limit.
            css (bool): If True, generate only CSS-compatible errors (X and Z only).

        """
        paulis = ("X", "Z", "Y")

        num_qubits = len(qubits)

        for i in range(min_errors, num_qubits + 1):
            if max_errors and i > max_errors:
                break

            xs = next(product(("X",), repeat=i))
            zs = next(product(("Z",), repeat=i))

            xzs = [xs, zs]

            for b in combinations(qubits, i):
                for ps in xzs:
                    x_set = set()
                    z_set = set()
                    for p, q in zip(ps, b, strict=False):
                        if p == "X":
                            x_set.add(q)
                        else:
                            z_set.add(q)
                    yield x_set, z_set

            if not css:
                for a in product(paulis, repeat=i):
                    if a in {xs, zs}:
                        continue

                    for b in combinations(qubits, i):
                        x_set = set()
                        z_set = set()
                        for p, q in zip(a, b, strict=False):
                            if p == "X":
                                x_set.add(q)
                            elif p == "Z":
                                z_set.add(q)
                            else:
                                x_set.add(q)
                                z_set.add(q)
                        yield x_set, z_set

    def _is_logical_error(
        self,
        state: SimulatorProtocol,
        xs: set[int],
        zs: set[int],
    ) -> bool:
        # A trivial error anticommutes with the checks. (Might or might not anticommute with the logical stabilizers)
        # A logical error commutes with the checks and is not a product of checks.

        # Does the error anticommute with the checks?
        x_anticoms = set()
        z_anticoms = set()
        for q in xs:
            x_anticoms ^= state.stabs.col_z[q]

        for q in zs:
            z_anticoms ^= state.stabs.col_x[q]

        anticoms = x_anticoms ^ z_anticoms
        anticom_logical_zs = self.logical_gens & anticoms
        anticoms -= self.logical_gens

        if anticoms:
            return False
        if anticom_logical_zs:
            # So the error commutes with all the stabilizers
            # Did it anticommute with any logical Z operations? If so... It is a product of logical Xs!
            # (and possibly other things)
            return True
        # Let's see if the error anticommuted with any logical X operators:

        x_anticoms_destabs = set()
        z_anticoms_destabs = set()

        for q in xs:
            x_anticoms_destabs ^= state.destabs.col_z[q]

        for q in zs:
            z_anticoms_destabs ^= state.destabs.col_x[q]

        anticoms_destabs = x_anticoms_destabs ^ z_anticoms_destabs
        anticom_logical_xs = self.logical_gens & anticoms_destabs

        # The error is a product of logical Zs
        return bool(anticom_logical_xs)

    def shortest_logicals(
        self,
        start_weight: int | None = None,
        delta: int = 0,
        *,
        verbose: bool = True,
        css: bool = False,
    ) -> tuple[
        list[LogicalOpInfo],
        dict[str, dict[str, set[int]]],
        dict[str, dict[str, set[int]]],
    ]:
        """Find the shortest logical operators.

        Args:
            start_weight (int): Weight of operators to begin searching.
            delta (int): Method will look for all logical ops with weight =< minimum weight + `delta`.
            verbose (bool): If True, print progress information during the search.
            css (bool): If True, restrict search to CSS-compatible operators (X and Z only).

        Returns:
        -------
            Dictionary of logical ops...

        """
        # if not self.logical_xs_reference and not self.logical_zs_reference:

        if start_weight is None:
            start_weight = self.dist if self.dist is not None else 1

        end_weight = start_weight + delta

        if self.circuit is None:
            msg = "Must compile circuits first!"
            raise Exception(msg)

        qudit_set = self.data_qubits

        end_weight = min(end_weight, len(qudit_set))

        state = self.state
        found = self._dist_mode_smallest(
            state,
            qudit_set,
            css=css,
            verbose=False,
            start_len=start_weight,
            end_len=end_weight,
            list_ops=True,
        )

        xs_labels = sorted(self.logical_xs_reference.keys())
        zs_labels = sorted(self.logical_zs_reference.keys())

        oplist = []

        if found:
            for paulis in found:
                op_product = []
                for xi, op_label in enumerate(xs_labels):
                    if self.op_anticommute(paulis, self.logical_xs_reference[op_label]):
                        op_product.append(zs_labels[xi])

                for zi, op_label in enumerate(zs_labels):
                    if self.op_anticommute(paulis, self.logical_zs_reference[op_label]):
                        op_product.append(xs_labels[zi])

                op_product = sorted(op_product)

                oplist.append(
                    {
                        "X": paulis["X"],
                        "Z": paulis["Z"],
                        "equiv_ops": tuple(op_product),
                    },
                )

        if verbose:
            print("Reference Logical Operators:")
            print("\nLogical Xs:")
            for op_label in xs_labels:
                op = self.logical_xs_reference[op_label]
                print(op_label, op)
            print("\nLogical Zs:")
            for op_label in zs_labels:
                op = self.logical_zs_reference[op_label]
                print(op_label, op)

            print("\nLogical Ops Found:\n")
            for foundop in oplist:
                print(
                    "X - {} Z - {} Equiv Ops - {}".format(
                        foundop["X"],
                        foundop["Z"],
                        foundop["equiv_ops"],
                    ),
                )

        return oplist, self.logical_xs_reference, self.logical_zs_reference

    @staticmethod
    def op_anticommute(op1: dict[str, set[int]], op2: dict[str, set[int]]) -> bool:
        """Check if two Pauli operators anticommute.

        Args:
            op1: First Pauli operator as dictionary with X, Y, Z keys and qubit sets.
            op2: Second Pauli operator as dictionary with X, Y, Z keys and qubit sets.

        Returns:
            True if the operators anticommute, False otherwise.
        """
        return bool(
            (
                len(op1.get("X", set()) & op2.get("Z", set()))
                + len(op2.get("X", set()) & op1.get("Z", set()))
            )
            % 2,
        )
