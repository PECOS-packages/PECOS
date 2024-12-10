# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np

from pecos.circuits import QuantumCircuit
from pecos.engines.cvm.classical import eval_condition
from pecos.error_models.class_errors_circuit import ErrorCircuits
from pecos.error_models.noise_impl_old.gate_groups import one_qubits, two_qubits
from pecos.error_models.noise_impl_old.init_noise import noise_init_bitflip
from pecos.error_models.noise_impl_old.meas_noise import noise_meas_bitflip
from pecos.error_models.noise_impl_old.memory_noise import noise_tq_mem
from pecos.error_models.noise_impl_old.sq_noise import noise_depolarizing_sq_gate
from pecos.error_models.noise_impl_old.tq_noise import (
    noise_two_qubit_gates_depolarizing_with_noiseless,
)
from pecos.error_models.parent_class_error_gen import ParentErrorModel


class DepolarizingErrorModel(ParentErrorModel):
    """Parameterized error model for Beta and ARC1."""

    def __init__(self) -> None:
        super().__init__()

        self.qubit_set = set()

        self.error_circuits = None
        self.error_params = None
        self.circuit = None

    def scaling(self):
        # conversion from average error to total error
        self.error_params["p1"] *= 3 / 2
        self.error_params["p2"] *= 5 / 4

        scale = self.error_params.get("scale", 1.0)
        self.error_params["p1"] *= scale
        self.error_params["p2"] *= scale

        if isinstance(self.error_params["p_meas"], tuple):
            self.error_params["p_meas"] = np.mean(self.error_params["p_meas"])

        self.error_params["p_meas"] *= scale
        self.error_params["p_init"] *= scale

    def start(self, circuit, error_params, reset_leakage=True):
        self.qubit_set = set(range(circuit.metadata["num_qubits"]))

        self.error_circuits = ErrorCircuits()
        self.circuit = circuit

        # if error_params is None:

        if not error_params:
            msg = "Error params not set!"
            raise Exception(msg)

        self.error_params = {}
        self.error_params.update(error_params)

        self.scaling()  # scale self.error_params params

        if not self.error_params.get("noiseless_qubits"):
            self.error_params["noiseless_qubits"] = set()
        self.error_params["noiseless_qubits"] = set(
            self.error_params["noiseless_qubits"],
        )

        return self.error_circuits

    def generate_tick_errors(
        self,
        tick_circuit,
        time,
        output=None,
        reset_leakage=False,
        **params,
    ):
        """The method that gets called each circuit tick to generate circuit noise for that tick."""
        # Get the tick
        tick_index = time[-1] if isinstance(time, tuple) else time

        circuit = tick_circuit.circuit

        before = QuantumCircuit()  # Faults that occur before the tick
        after = QuantumCircuit()  # Faults the occur after the tick
        remove_locations = set()  # gate locations to remove

        for symbol, locations, metadata in circuit.items(tick_index):
            cond = metadata.get("cond")

            if symbol == "cop":
                pass

            # ########################################
            # INITS WITH X NOISE
            elif symbol == "init |0>":
                noisy = set(locations) - self.error_params["noiseless_qubits"]
                noise_init_bitflip(noisy, after, "X", p=self.error_params["p_init"])

            # ########################################
            # ONE QUBIT GATES
            elif symbol in one_qubits:
                if eval_condition(cond, output):
                    noisy = set(locations) - self.error_params["noiseless_qubits"]
                    noise_depolarizing_sq_gate(noisy, after, p=self.error_params["p1"])

            # ########################################
            # TWO QUBIT GATES
            elif symbol in two_qubits:
                if eval_condition(cond, output):
                    noise_two_qubit_gates_depolarizing_with_noiseless(
                        locations,
                        after,
                        p=self.error_params["p2"],
                        noiseless_qubits=self.error_params["noiseless_qubits"],
                    )

                    if self.error_params["p2_mem"]:
                        noise_tq_mem(locations, after, p=self.error_params["p2_mem"])

            # ########################################
            # MEASURE X NOISE
            elif symbol == "measure Z":
                if eval_condition(cond, output):
                    noisy = set(locations) - self.error_params["noiseless_qubits"]
                    noise_meas_bitflip(
                        noisy,
                        metadata,
                        after,
                        p=self.error_params["p_meas"],
                    )

            elif symbol in {"repump", "leak", "unleak"}:
                pass

            else:
                raise Exception("This error model doesn't handle gate: %s!" % symbol)

        self.error_circuits.add_circuits(time, before, after, remove_locations)

        return self.error_circuits
