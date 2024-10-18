# Copyright 2018 The PECOS Developers
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

from typing import TYPE_CHECKING

from pecos.simulators.gate_syms import alt_symbols
from pecos.simulators.paulifaultprop import bindings
from pecos.simulators.paulifaultprop.logical_sign import find_logical_signs
from pecos.simulators.sim_class_types import PauliPropagation

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.circuits.quantum_circuit import ParamGateCollection


class PauliFaultProp(PauliPropagation):
    r"""A simulator that evolves Pauli faults through Clifford circuits.

    The unitary evolution of a Pauli follows :math:`PC = CP' \Leftrightarrow P' = C^{\dagger} P C`, where :math:`P` and
    :math:`P'` are Pauli operators and :math:`C` is a Clifford operator.

    Attributes:
    ----------
        num_qubits(int): Number of qubits.
        faults (Dict[str, Set[int]]):
        bindings (Dict[str, Callable]):

    """

    def __init__(self, *, num_qubits: int, track_sign: bool = False) -> None:
        """Args:
        ----
            num_qubits (int):

        Returns: None

        """
        super().__init__()

        self.num_qubits = num_qubits
        self.faults = {
            "X": set(),
            "Y": set(),
            "Z": set(),
        }
        # Here we will encode Y as the qubit id in faults_x and faults_z

        self.track_sign = track_sign
        self.sign = 0
        self.img = 0

        self.bindings = bindings.gate_dict
        for k, v in alt_symbols.items():
            if v in self.bindings:
                self.bindings[k] = self.bindings[v]

    def flip_sign(self):
        self.sign += 1
        self.sign %= 2

    def flip_img(self, num_is):
        self.img += num_is
        self.img %= 4

        if self.img in {2, 3}:
            self.flip_sign()

        self.img %= 2

    def logical_sign(self, logical_op: QuantumCircuit) -> int:
        """Find the sign of a logical operator, which is equivalent to determining if the faults commute (sign == 0) or
        anticommute (sign == 1) with the logical operator.

        That is, compare the commutation of `logical_op` with `faults.`

        Args:
        ----
            logical_op (QuantumCircuit): Quantum circuit representing a logical operator.

        Returns: int - sign.

        """
        return find_logical_signs(self, logical_op)

    def run_circuit(
        self,
        circuit: ParamGateCollection,
        removed_locations: set[int] | (set[tuple[int, ...]] | None) = None,
        *,
        apply_faults: bool = False,
    ):
        """Used to apply a quantum circuit to a state, whether the circuit represents an fault or ideal circuit.

        Args:
        ----
            circuit: A class representing a circuit. # TODO: Shouldn't this also include QuantumCircuit
            removed_locations : A set of qudit locations that correspond to
                ideal gates that should be removed.
            apply_faults: Whether to apply the `circuit` as a Pauli fault (True) or as a Clifford to update the
                faults (False).

        Returns: None

        """
        circuit_type = circuit.metadata.get("circuit_type")

        if circuit_type in {"faults", "recovery"}:
            self.add_faults(circuit)
            return None
        else:
            if self.faults["X"] or self.faults["Y"] or self.faults["Z"]:
                # Only apply gates if there are faults to act on
                return super().run_circuit(circuit, removed_locations)
            return None

            # need to return output?

    def add_faults(
        self,
        circuit: QuantumCircuit | ParamGateCollection,
        *,
        minus=False,
    ) -> None:
        """A methods to add faults to the state.

        Args:
        ----
            circuit (Union[QuantumCircuit, ParamGateCollection]): A quantum circuit representing Pauli faults.

        Returns: None

        """
        if self.track_sign and minus:
            self.flip_sign()

        for elem in circuit.items():
            if len(elem) == 2:
                symbol, locations = elem
            else:
                symbol, locations, _ = elem

            if symbol in ["X", "Y", "Z"]:
                if symbol == "X":
                    # X.I = X
                    # X.X = I
                    # X.Y = iZ
                    # X.Z = -iY

                    yoverlap = self.faults["Y"] & locations
                    zoverlap = self.faults["Z"] & locations

                    self.faults["Y"] -= yoverlap
                    self.faults["Z"] -= zoverlap

                    self.faults["Y"] ^= zoverlap
                    self.faults["Z"] ^= yoverlap

                    self.faults["X"] ^= locations - yoverlap - zoverlap

                    if self.track_sign:
                        if yoverlap:
                            # X.Y = i Z
                            self.flip_img(len(yoverlap))

                        if zoverlap:
                            # X.Z = -i Y
                            self.flip_img(len(zoverlap))

                            if len(zoverlap) % 2:
                                self.flip_sign()

                elif symbol == "Z":
                    # Z.I = Z
                    # Z.X = iY
                    # Z.Y = -iX
                    # Z.Z = I

                    xoverlap = self.faults["X"] & locations
                    yoverlap = self.faults["Y"] & locations

                    self.faults["X"] -= xoverlap
                    self.faults["Y"] -= yoverlap

                    self.faults["X"] ^= yoverlap
                    self.faults["Y"] ^= xoverlap

                    self.faults["Z"] ^= locations - xoverlap - yoverlap

                    if self.track_sign:
                        if xoverlap:
                            # Z.X = i Y
                            self.flip_img(len(xoverlap))

                        if yoverlap:
                            # Z.Y = -i X
                            self.flip_img(len(yoverlap))

                            if len(yoverlap) % 2:
                                self.flip_sign()

                else:
                    # Y.I = Y
                    # Y.X = -iZ
                    # Y.Y = I
                    # Y.Z = iX

                    xoverlap = self.faults["X"] & locations
                    zoverlap = self.faults["Z"] & locations

                    self.faults["X"] -= xoverlap
                    self.faults["Z"] -= zoverlap

                    self.faults["X"] ^= zoverlap
                    self.faults["Z"] ^= xoverlap

                    self.faults["Y"] ^= locations - xoverlap - zoverlap

                    if self.track_sign:
                        if zoverlap:
                            # Y Z = i X
                            self.flip_img(len(zoverlap))

                        if xoverlap:
                            # Y X = -i Z
                            self.flip_img(len(xoverlap))

                            if len(xoverlap) % 2:
                                self.flip_sign()

            else:
                msg = f"Got {symbol}. Can only handle Pauli errors."
                raise Exception(msg)

    def get_str(self):
        fault_dict = self.faults

        pstr = "-" if self.sign else "+"

        for q in range(self.num_qubits):
            if q in fault_dict.get("X", set()):
                pstr += "X"
            elif q in fault_dict.get("Y", set()):
                pstr += "Y"
            elif q in fault_dict.get("Z", set()):
                pstr += "Z"
            else:
                pstr += "I"
        return pstr

    def fault_str_sign(self, *, strip=False):
        fault_str = []

        if self.sign:
            fault_str.append("-")
        else:
            fault_str.append("+")

        if self.img:
            fault_str.append("i")
        else:
            fault_str.append(" ")

        fault_str = "".join(fault_str)

        if strip:
            fault_str = fault_str.strip()

        return fault_str

    def fault_str_operator(self):
        fault_str = []

        for q in range(self.num_qubits):
            if q in self.faults["X"]:
                fault_str.append("X")

            elif q in self.faults["Y"]:
                fault_str.append("Y")

            elif q in self.faults["Z"]:
                fault_str.append("Z")

            else:
                fault_str.append("I")

        return "".join(fault_str)

    def fault_string(self):
        return f"{self.fault_str_sign()}{self.fault_str_operator()}"

    def fault_wt(self):
        wt = len(self.faults["X"])
        wt += len(self.faults["Y"])
        wt += len(self.faults["Z"])

        return wt

    def __str__(self) -> str:
        return "{{'X': {}, 'Y': {}, 'Z': {}}}".format(
            self.faults["X"],
            self.faults["Y"],
            self.faults["Z"],
        )
