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

"""Simple error generator meant to demonstrate a basic error generator that produces errors."""

from __future__ import annotations

from typing import ClassVar

from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.error_models.class_errors_circuit import ErrorCircuits
from pecos.error_models.parent_class_error_gen import ParentErrorModel


class GatewiseModel(ParentErrorModel):
    """A simple error generator for the depolarizing model.

    This error generator does not allow much modification of the error model.
    """

    measurements: ClassVar[set[str]] = {"measure X", "measure Y", "measure Z"}
    inits: ClassVar[set[str]] = {
        "init |0>",
        "init |1>",
        "init |+>",
        "init |->",
        "init |+i>",
        "init |-i>",
    }
    two_qubits: ClassVar[set[str]] = {"CNOT", "CZ", "SWAP", "G"}
    one_qubits: ClassVar[set[str]] = {
        "I",
        "X",
        "Y",
        "Z",
        "Q",
        "Qd",
        "R",
        "Rd",
        "S",
        "Sd",
        "H",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H+z+x",
        "H-z-x",
        "H+y-z",
        "H-y-z",
        "H-x+y",
        "H-x-y",
        "F1",
        "F1d",
        "F2",
        "F2d",
        "F3",
        "F3d",
        "F4",
        "F4d",
    }

    inits_z: ClassVar[set[str]] = {"init |0>", "init |1>"}
    inits_x: ClassVar[set[str]] = {"init |+>", "init |->"}
    inits_y: ClassVar[set[str]] = {"init |+i>", "init |-i>"}

    def __init__(self) -> None:
        """ """

        super().__init__()
        self.gen = self.generator_class()
        self.gen.set_gate_group("measurements", self.measurements)
        self.gen.set_gate_group("preps", self.inits)
        self.gen.set_gate_group("two_qubits", self.two_qubits)
        self.gen.set_gate_group("one_qubits", self.one_qubits)

        # Inherit gen methods
        self.set_gate_error = self.gen.set_gate_error
        self.set_group_error = self.gen.set_group_error
        self.set_default_error = self.gen.set_default_error
        self.set_gate_group = self.gen.set_gate_group

        self.gate_groups = self.gen.gate_groups

        # Inherit error function classes
        self.ErrorStaticSymbol = self.gen.ErrorStaticSymbol
        self.ErrorSet = self.gen.ErrorSet
        self.ErrorSetTwoQuditTensorProduct = self.gen.ErrorSetTwoQuditTensorProduct

    def start(self, circuit, error_params):
        """Start up at the beginning of a circuit simulation.

        Args:
        ----
            circuit:
            error_params:

        Returns:
        -------

        """
        self.error_circuits = ErrorCircuits()
        self.circuit = circuit
        self.error_params = error_params

        return self.error_circuits

    def generate_tick_errors(self, tick_circuit, time, **params):
        """Returns before errors, after errors, and replaced locations for the given key (args).

        Returns:
        -------

        """
        tick_index = time[-1] if isinstance(time, tuple) else time

        circuit = tick_circuit.circuit

        # Simple model where for each gate there is a probability "p" for an X, Y, or Z error to occur.

        before = QuantumCircuit()
        after = QuantumCircuit()
        replace = set()

        # Data errors
        # -----------
        if tick_index == 0 and "data" in self.gen.error_func_dict:
            data_qudit_set = params["data_qudit_set"]
            self.gen.create_errors(self, "data", data_qudit_set, after, before, replace)

        # unitary and measurement errors
        # ------------------------------
        for symbol, gate_locations, _ in circuit.items(tick_index):
            self.gen.create_errors(self, symbol, gate_locations, after, before, replace)

        # idle errors
        # -----------
        if "idle" in self.gen.error_func_dict:
            inactive_qudits = circuit.qudits - circuit.active_qudits[tick_index]

            if tick_index == 0 and "data" in self.gen.error_func_dict:
                data_qudit_set = params["data_qudit_set"]
                inactive_qudits -= data_qudit_set

            self.gen.create_errors(
                self,
                "idle",
                inactive_qudits,
                after,
                before,
                replace,
            )

        self.error_circuits.add_circuits(time, before, after)

        return self.error_circuits

    def get_gate_error(self, symbol, gate_locations, error_params):
        """Args:
        ----
            symbol:
            gate_locations:
            error_params:

        Returns:
        -------

        """
        self.error_params = error_params

        before = QuantumCircuit()
        after = QuantumCircuit()
        replace = set()
        self.gen.create_errors(self, symbol, gate_locations, after, before, replace)
        return after, before, replace
