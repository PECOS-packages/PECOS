"""Actual compiled ordering cuts and named results for the Guppy backend."""

import math
import sys

import pytest
from pecos import capture_qis_operation_trace, sim, stabilizer
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
from pecos.qec.generic import CircuitOp as Op
from pecos.qec.generic import GuppyOrdering, MeasurementParity, ScheduledCircuit

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Native aggregate helper ABI adapter not implemented")


def _operations(trace):
    operations = []
    for chunk in trace:
        for source_op in chunk["operations"]:
            op = source_op
            native = op.get("RuntimeRequest", {}).get("request", {}).get("Native") if isinstance(op, dict) else None
            if isinstance(native, dict):
                if "Measure" in native:
                    m = native["Measure"]
                    assert not m["leaked"]
                    op = {"Quantum": {"Measure": [m["qubit"], m["result_id"]]}}
                elif "Reset" in native:
                    op = {"Quantum": {"Reset": native["Reset"]}}
                elif "Rxy" in native:
                    r = native["Rxy"]
                    op = {"Quantum": {"RXY": [r["theta"], r["phi"], r["qubit"]]}}
            operations.append(op)
    return operations


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_compiler_fences_order_disjoint_gates_and_keep_all_prep_before_readout(n):
    plan = ScheduledCircuit(
        n,
        (
            tuple(Op("R", (q,)) for q in range(n)),
            (Op("X", (n - 1,)),),
            (Op("SQRT_X", (min(1, n - 1),)),),
            (Op("H", (0,)),),
            tuple(Op("M", (q,)) for q in range(n)),
        ),
        (),
        (),
    )
    contract = plan.guppy_ordering_plan(GuppyOrdering.FIXED_LIFETIME)
    trace = capture_qis_operation_trace(plan.guppy_program(ordering=contract.mode), n, seed=7)
    ops = _operations(trace)
    assert ops.count("Barrier") == contract.to_dict()["runtime_barrier_calls"]
    measurements = {
        op["Quantum"]["Measure"][1]: op["Quantum"]["Measure"][0]
        for op in ops
        if isinstance(op, dict) and "Measure" in op.get("Quantum", {})
    }
    roles = {}
    for chunk in trace:
        for result in chunk["named_result_traces"]:
            roles[int(result["name"].split(":")[-1])] = measurements[result["result_ids"][0]]
    rotations = [op["Quantum"]["RXY"] for op in ops if isinstance(op, dict) and "RXY" in op.get("Quantum", {})]
    assert [row[2] for row in rotations] == [roles[n - 1], roles[min(1, n - 1)], roles[0]]
    assert [row[0] for row in rotations] == pytest.approx([math.pi, math.pi / 2, math.pi / 2])
    physical = [op["Quantum"] for op in ops if isinstance(op, dict) and "Quantum" in op]
    assert all("Reset" in op for op in physical[:n])
    assert all("Measure" in op for op in physical[-n:])


@pytest.mark.parametrize("ordering", [GuppyOrdering.UNITARY_REGIONS, GuppyOrdering.ALL_MOMENTS])
def test_runtime_fence_keeps_inverse_pulses_across_an_authored_cut(ordering):
    plan = ScheduledCircuit(
        1,
        ((Op("R", (0,)),), (Op("X", (0,)),), (Op("X", (0,)),), (Op("M", (0,)),)),
        (MeasurementParity((0,), "check"),),
        (),
    )
    ops = _operations(capture_qis_operation_trace(plan.guppy_program(ordering=ordering), 1))
    assert ops.count("Barrier") == (1 if ordering == GuppyOrdering.UNITARY_REGIONS else 3)
    assert sum(isinstance(op, dict) and "RXY" in op.get("Quantum", {}) for op in ops) == 2


@pytest.mark.parametrize("family", ["488", "swap666", "lacroix", "periodic"])
@pytest.mark.parametrize("ordering", [GuppyOrdering.UNITARY_REGIONS, GuppyOrdering.ALL_MOMENTS])
def test_paper_memory_regions_compile_with_complete_measurement_binding(family, ordering):
    if family == "488":
        plan = color_memory(color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ), 2, "X")
    elif family == "swap666":
        impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
        plan = color_memory(impl, 3, "Z")
    elif family == "lacroix":
        plan = lacroix_memory("d3_west", 2, "X")
    else:
        plan = periodic_488_memory(periodic_488_implementation(periodic_488_code(4), gate_set="native-cz"), 2, "Z")
    contract = plan.guppy_ordering_plan(ordering)
    assert contract.fences
    assert bool(contract.to_dict()["uncovered_cuts"]) == (ordering == GuppyOrdering.UNITARY_REGIONS)
    audit = plan.audit_guppy(ordering=contract.mode, seed=17)
    assert audit.build.named_result_binding == "compiler_direct_scalar_complete"
    assert audit.to_dict()["requested_ordering"] == contract.to_dict()
    assert not audit.to_dict()["physical_schedule_verified"]
    assert audit.build.dem.independent_mechanisms() == []
    program = plan.guppy_program(ordering=contract.mode)
    columns = sim(program).qubits(plan.num_qubits).quantum(stabilizer()).seed(19).run(8).to_dict()
    assert audit.build.evaluate_result_columns(columns) == [([0] * len(plan.detectors), 0)] * 8
    trace = capture_qis_operation_trace(program, plan.num_qubits, seed=17)
    assert _operations(trace).count("Barrier") == contract.to_dict()["runtime_barrier_calls"]


@pytest.mark.parametrize("opt_level", [0, 1, 2, 3])
def test_all_moments_orders_allocation_measurement_and_empty_live_intervals(opt_level):
    from pecos import Qis, compilation_pipeline

    plan = ScheduledCircuit(
        2,
        (
            (),
            (Op("R", (0,)),),
            (Op("X", (0,)),),
            (Op("R", (1,)),),
            (Op("M", (0,)),),
            (Op("X", (1,)),),
            (Op("M", (1,)),),
            (),
            (Op("R", (0,)),),
            (Op("SQRT_X", (0,)),),
            (Op("M", (0,)),),
            (),
        ),
        (),
        (),
    )
    program = plan.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS)
    compiled = Qis(compilation_pipeline.compile_hugr_to_qis(program.compile().to_bytes(), opt_level=opt_level))
    trace = capture_qis_operation_trace(compiled, plan.num_qubits, seed=13)
    # One runtime barrier per cut lets us compare source moments directly. IDs
    # may be reused and sites are implementation-defined, so compare operation kinds.
    segments = [[]]
    for op in _operations(trace):
        if op == "Barrier":
            segments.append([])
        elif isinstance(op, dict) and "Quantum" in op:
            segments[-1].append(next(iter(op["Quantum"])))
    assert segments == [
        [],
        ["Reset"],
        ["RXY"],
        ["Reset"],
        ["Measure"],
        ["RXY"],
        ["Measure"],
        [],
        ["Reset"],
        ["RXY"],
        ["Measure"],
        [],
    ]
    assert sum(len(chunk["named_result_traces"]) for chunk in trace) == 3


def test_all_moments_keeps_destructive_random_measurement_and_named_binding():
    plan = ScheduledCircuit(
        2,
        (
            (Op("R", (0,)), Op("R", (1,))),
            (Op("H", (0,)),),
            (Op("CX", (0, 1)),),
            (Op("M", (0,)),),
            (Op("M", (1,)),),
            (),
            (Op("R", (0,)),),
            (Op("M", (0,)),),
        ),
        (MeasurementParity((0, 1), "bell"), MeasurementParity((2,), "fresh")),
        (),
    )
    audit = plan.audit_guppy(ordering=GuppyOrdering.ALL_MOMENTS)
    assert audit.build.named_result_binding == "compiler_direct_scalar_complete"
    columns = (
        sim(plan.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS))
        .qubits(2)
        .quantum(
            stabilizer(),
        )
        .seed(15)
        .run(64)
        .to_dict()
    )
    assert set(columns["qec:m:0"]) == {0, 1}
    assert audit.build.evaluate_result_columns(columns) == [([0, 0], 0)] * 64


@pytest.mark.parametrize("num_moments", [0, 1, 3])
def test_all_moments_empty_program_has_no_quantum_work(num_moments):
    plan = ScheduledCircuit(1, ((),) * num_moments, (), ())
    trace = capture_qis_operation_trace(plan.guppy_program(ordering=GuppyOrdering.ALL_MOMENTS), 1)
    ops = _operations(trace)
    assert ops.count("Barrier") == max(0, num_moments - 1)
    assert not any(isinstance(op, dict) and "Quantum" in op for op in ops)
