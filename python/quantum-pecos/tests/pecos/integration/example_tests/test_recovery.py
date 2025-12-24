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

"""Integration tests for quantum error recovery protocols."""

import pecos as pc


def test_recovery() -> None:
    """Test quantum error correction recovery process."""
    surface = pc.qeccs.Surface4444(distance=5)

    for _i in range(30):
        recovery_tester(surface)


def recovery_tester(qecc: pc.protocols.QECCProtocol) -> None:
    """Used to check recovery operations.

     - That ideal logical |0> and |+> return the same output for check measurements.
     - That whether a recovery operation flips a logical operator or not can be determined.

    Args:
        qecc: The quantum error correcting code instance.

    Returns:
        None

    """
    # Logical circuits
    # ----------------

    # logical |0>
    initzero = pc.circuits.LogicalCircuit(suppress_warning=True)
    initzero.append(qecc.gate("ideal init |0>"))

    # logical |+>
    initplus = pc.circuits.LogicalCircuit(suppress_warning=True)
    initplus.append(qecc.gate("ideal init |+>"))

    # syndrome extraction
    syn_ext = pc.circuits.LogicalCircuit(suppress_warning=True)
    syn_ext.append(qecc.gate("I", num_syn_extract=1))

    # Error Generator
    # ---------------
    depolar = pc.error_models.DepolarModel(model_level="code_capacity")

    # Circuit simulator
    # -----------------
    sim = pc.circuit_runners.Standard()

    # Run circuits
    # ------------
    state_zero = pc.simulators.SparseSimPy(qecc.num_qudits)
    output_zero, _ = sim.run(state_zero, initzero)

    assert not output_zero

    state_plus = pc.simulators.SparseSimPy(qecc.num_qudits)
    output_plus, _ = sim.run(state_plus, initplus)

    assert not output_plus

    output1, error_circuits1 = sim.run(state_zero, syn_ext)
    output2, error_circuits2 = sim.run(state_plus, syn_ext)

    assert error_circuits1 == error_circuits2
    assert not error_circuits1
    assert output1 == output2
    assert not output1

    output1, error_circuits1 = sim.run(
        state_zero,
        syn_ext,
        error_gen=depolar,
        error_params={"p": 0.3},
    )
    output2, error_circuits2 = sim.run(
        state_plus,
        syn_ext,
        error_circuits=error_circuits1,
    )

    assert error_circuits1 == error_circuits2
    assert output1 == output2

    # syn = output2.simplified(True)

    # Logical operations
    # ------------------
    logical_ops = qecc.instruction("instr_syn_extract").final_logical_ops[0]

    # Logical signs
    # -------------
    sign1 = state_zero.logical_sign(
        logical_ops["Z"],
        # logical_ops['X']
    )
    sign2 = state_plus.logical_sign(
        logical_ops["X"],
        # logical_ops['Z']
    )

    # Decoder
    # -------
    mwpm2d = pc.decoders.MWPM2D(qecc)

    # Recovery operation
    # ------------------
    recovery = mwpm2d.decode(output2)

    # Determine if the logical operators will flip due to the recovery

    commute1 = pc.misc.commute.qubit_pauli(logical_ops["Z"], recovery)
    commute2 = pc.misc.commute.qubit_pauli(logical_ops["X"], recovery)

    sim.run(state_zero, recovery)
    sim.run(state_plus, recovery)

    sign1_new = state_zero.logical_sign(
        logical_ops["Z"],
        # logical_ops['X']
    )
    sign2_new = state_plus.logical_sign(
        logical_ops["X"],
        # logical_ops['Z']
    )

    # Not commuting then should flip; otherwise, doesn't flip.
    if commute1:  # Commuting
        assert sign1 == sign1_new  # No flip
    else:  # Not commuting
        assert sign1 != sign1_new  # Flipped

    if commute2:  # Commuting
        assert sign2 == sign2_new  # No flip
    else:  # Not commuting
        assert sign2 != sign2_new  # Flipped
