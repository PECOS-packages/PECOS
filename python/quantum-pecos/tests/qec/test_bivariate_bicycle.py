# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Python bindings for the bivariate-bicycle code and memory builder."""

import json

import pytest
from pecos.qec import BivariateBicycleCode, bb_memory_circuit
from pecos_rslib import DagCircuit
from pecos_rslib.qec import DetectorErrorModel

A_TERMS = [(3, 0), (0, 1), (0, 2)]
B_TERMS = [(0, 3), (1, 0), (2, 0)]


def test_bb_72_code_constructor_reports_n_and_k() -> None:
    code = BivariateBicycleCode(6, 6, A_TERMS, B_TERMS)

    assert code.num_qubits() == 72
    assert code.num_logical_qubits() == 12
    assert code.hx.num_checks() == code.hz.num_checks() == 36
    assert code.hx.num_qubits() == code.hz.num_qubits() == 72
    assert code.logical_x.num_checks() == code.logical_z.num_checks() == 12


def test_bb_memory_binding_is_deterministic_and_exports_metadata() -> None:
    first = bb_memory_circuit(6, 6, A_TERMS, B_TERMS, 2, "Z")
    second = bb_memory_circuit(6, 6, A_TERMS, B_TERMS, 2, "Z")

    assert repr(first) == repr(second)
    assert first.num_ticks() == 18
    assert first.get_meta("syndrome_extraction_depth") == "17"
    assert first.get_meta("num_detectors") == "144"
    assert first.get_meta("num_observables") == "12"


def test_bb_bindings_reject_invalid_basis_and_exponent() -> None:
    with pytest.raises(ValueError, match="basis must be"):
        bb_memory_circuit(6, 6, A_TERMS, B_TERMS, 2, "Y")
    with pytest.raises(ValueError, match="outside Z_6 x Z_6"):
        BivariateBicycleCode(6, 6, [(6, 0), (0, 1), (0, 2)], B_TERMS)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_bb_memory_dem_detector_bases_match_check_families(basis: str) -> None:
    circuit = bb_memory_circuit(6, 6, A_TERMS, B_TERMS, 3, basis)
    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0.001,
        p2=0.001,
        p_meas=0.001,
        p_prep=0.001,
    )
    mask = dem.detector_bases()
    # Each detector label starts with its check family.
    labels = {entry["id"]: entry["label"] for entry in json.loads(circuit.get_meta("detectors"))}
    assert mask == [labels[detector][0] for detector in range(dem.num_detectors)]
    assert set(mask) == {"X", "Z"}


def test_dem_detector_without_fault_sector_has_no_basis() -> None:
    circuit = DagCircuit()
    circuit.pz([0])
    circuit.mz([0])
    circuit.set_attr("num_measurements", "1")
    circuit.set_attr("detectors", '[{"id":0,"records":[-1]}]')
    dem = DetectorErrorModel.from_circuit(
        circuit,
        p1=0.0,
        p2=0.0,
        p_meas=0.001,
        p_prep=0.0,
    )
    with pytest.raises(ValueError, match="detector D0 is undefined or has no fault sector"):
        dem.detector_bases()
