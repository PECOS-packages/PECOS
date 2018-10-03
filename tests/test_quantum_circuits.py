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

from pecos.circuits import QuantumCircuit


def test_quantum_circuits():

    # Check the method append with check_overlap == True
    # ---------------------------------------------------
    qc = QuantumCircuit()

    assert len(qc) == 0
    assert qc.active_qudits == []

    qc.append({'int |0>': {0, 1}})

    assert len(qc) == 1
    assert qc.active_qudits == [{0, 1}]

    qc.append({'H': {0}})

    assert len(qc) == 2
    assert qc.active_qudits == [{0, 1}, {0}]

    qc.append({'CNOT': {(0, 1)}})

    assert len(qc) == 3
    assert qc.active_qudits == [{0, 1}, {0}, {0, 1}]

    qc.append({'measure Z': {0}})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0}, {0, 1}, {0}]

    # Check update, add, and discard with check_overlap == True
    # ---------------------------------------------------------

    qc.update({'X': {1}}, tick=1)

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0}]

    qc.update({'measure Z': {1}})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0, 1}]

    qc.discard({1})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0}]

    qc.update('X', {1})

    assert len(qc) == 4
    assert qc.active_qudits == [{0, 1}, {0, 1}, {0, 1}, {0, 1}]

    # Check the method append with check_overlap == False
    # ---------------------------------------------------
    qc = QuantumCircuit()

    assert len(qc) == 0
    assert qc.active_qudits == []

    qc.append({'int |0>': {0, 1}})

    assert len(qc) == 1
    assert qc.active_qudits == [{0, 1}]

    qc.append({'H': {0}})

    assert len(qc) == 2
    assert qc.active_qudits == [{0, 1}, {0}]

    # Check update with check_overlap == False
    # -----------------------------------------

    qc.update({'X': {1}}, tick=1)

    assert len(qc) == 2
    assert qc.active_qudits == [{0, 1}, {0, 1}]
