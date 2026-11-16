"""Compiled measurements and runtime lowering are auditable without timing claims."""

import json

import pytest
from pecos import sim, stabilizer
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    lacroix_memory,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
)
from pecos.qec.generic import CircuitOp, MeasurementParity, ScheduledCircuit


@pytest.mark.parametrize(("family", "choice"), [
    ("488", ColorExtraction.BRICKWORK_CZ), ("666", ColorExtraction.SWAP_SUPERDENSE),
])
def test_audit_retains_measured_contract_and_distinguishes_runtime_units(family, choice):
    plan = color_memory(color_code_implementation(color_code_spec(family, 3), choice), 1)
    audit = plan.audit_guppy(seed=17)
    report = audit.to_dict()
    json.dumps(report)
    assert report["reference_moments"] == len(plan.moments)
    assert report["reference_peak_live"] == plan.resources.peak_live
    assert report["analysis_gate_counts"] == audit.build.circuit.gate_counts_by_type()
    assert report["measurement_binding"]["named_result_binding"] == "compiler_direct_scalar_complete"
    assert report["measurement_binding"]["measurement_count"] == plan.num_measurements
    assert report["runtime_circuit_representation"] == "clifford_normalized_for_dem_analysis"
    assert not report["physical_schedule_verified"]
    assert not report["noise_equivalence_verified"]
    assert audit.build.dem.independent_mechanisms() == []
    assert "not established" in audit.describe()
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(19).run(4).to_dict()
    for shot, (detectors, observable) in enumerate(audit.build.evaluate_result_columns(columns)):
        expected_d, expected_o = plan.evaluate_named({tag: values[shot] for tag, values in columns.items()})
        assert tuple(map(bool, detectors)) == expected_d
        assert observable == sum(int(bit) << i for i, bit in enumerate(expected_o))


def test_affine_constants_and_coordinates_are_retained_in_audit():
    plan = ScheduledCircuit(1, ((CircuitOp("R", (0,)),), (CircuitOp("X", (0,)),), (CircuitOp("M", (0,)),)),
                            (MeasurementParity((0,), "sign", True, (1., 2., 3., 4.)),), ())
    audit = plan.audit_guppy()
    assert audit.build.evaluate_results({"qec:m:0": True}) == ([0], 0)
    assert json.loads(audit.build.detectors_json)[0]["coords"] == [1., 2., 3., 4.]
    assert not audit.to_dict()["physical_schedule_verified"]


@pytest.mark.parametrize("family", ["lacroix", "periodic"])
def test_audit_supports_hardware_templates_and_multiple_logicals(family):
    if family == "lacroix":
        plan = lacroix_memory("d3_west", 2, "X")
    else:
        impl = periodic_488_implementation(periodic_488_code(4), gate_set="native-cz")
        plan = periodic_488_memory(impl, 2, "Z")
    audit = plan.audit_guppy(seed=31)
    assert audit.build.named_result_binding == "compiler_direct_scalar_complete"
    assert audit.build.dem.num_observables == len(plan.observables)
    assert audit.build.dem.num_detectors == len(plan.detectors)
    assert len(audit.build.measurement_ledger) == plan.num_measurements
    columns = sim(plan.guppy_program()).qubits(plan.num_qubits).quantum(stabilizer()).seed(47).run(4).to_dict()
    assert audit.build.evaluate_result_columns(columns) == [([0] * len(plan.detectors), 0)] * 4
