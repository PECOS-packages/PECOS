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

"""Integration tests for basic logical quantum circuit simulations."""
import pecos as pc


def test_surface() -> None:
    """This is to make sure no errors occur for running simple logical circuit.

    Returns:
        None

    """
    surface = pc.qeccs.Surface4444(distance=3)

    mwpm2d = pc.decoders.MWPM2D(surface)

    # Initialize ideal logical |0>
    logic = pc.circuits.LogicalCircuit(suppress_warning=True)
    logic.append(surface.gate("ideal init |0>"))

    # Syndrome extraction
    logic2 = pc.circuits.LogicalCircuit(suppress_warning=True)
    logic2.append(surface.gate("I", num_syn_extract=1))

    depolar = pc.error_models.DepolarModel(model_level="code_capacity")

    sim = pc.circuit_runners.Standard()

    state = pc.simulators.SparseSimPy(surface.num_qudits)

    _output1, _error_circuits1 = sim.run(state, logic)

    output2, _error_circuits2 = sim.run(
        state,
        logic2,
        error_gen=depolar,
        error_params={"p": 0.2},
    )

    # syn = output2.simplified(True)

    recovery = mwpm2d.decode(output2)

    sim.run(state, recovery)
