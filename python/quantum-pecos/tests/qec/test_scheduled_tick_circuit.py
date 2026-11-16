"""Native backend, event order, provenance, and carrier-contract regressions."""

import pytest
import stim
from pecos.qec.color import (
    ColorExtraction,
    color_code_implementation,
    color_code_spec,
    color_memory,
    color_state_memory,
)
from pecos.qec.generic import CircuitOp as O
from pecos.qec.generic import MeasurementParity as P
from pecos.qec.generic import ScheduledCircuit, SchedulePolicy
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim


def _moments(circuit):
    result, pending = [], []
    for op in circuit:
        if op.name == "TICK":
            result.append(sorted(pending))
            pending = []
        elif op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            arity = 2 if op.name in {"CX", "CZ"} else 1
            targets = [t.value for t in op.targets_copy()]
            pending.extend((op.name, tuple(targets[i : i + arity])) for i in range(0, len(targets), arity))
    if pending:
        result.append(sorted(pending))
    return result


@pytest.mark.parametrize("choice", list(ColorExtraction))
@pytest.mark.parametrize("schedule", list(SchedulePolicy))
def test_all_paper_constructions_share_physical_moments_and_event_contract(choice, schedule):
    family = "488" if choice in {ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ} else "666"
    impl = color_code_implementation(color_code_spec(family, 3), choice)
    plan = color_memory(impl, 3, "X", schedule=schedule)
    tick = plan.to_tick_circuit()
    reconstructed = stim.Circuit(tick_circuit_to_stim(tick))
    assert _moments(reconstructed) == _moments(plan.to_stim())
    assert tick.num_ticks() == plan.resources.physical_moments
    assert tick.get_meta("qec_output_carriers") == list(plan.output_carriers)
    assert tick.get_meta("qec_measurement_names") == list(plan.measurement_aliases)
    assert [identity for _, identity in tick.measurement_emission()] == list(range(plan.num_measurements))
    definitions = tick.circuit_definitions()
    assert [d["measurements"] for d in definitions["detectors"]] == [list(p.records) for p in plan.detectors]
    assert [d["measurements"] for d in definitions["observables"]] == [list(p.records) for p in plan.observables]
    assert not reconstructed.compile_detector_sampler(seed=11).sample(64, append_observables=True).any()
    for i in range(tick.num_ticks()):
        assert isinstance(tick.get_tick_meta(i, "qec_moment"), int)
        assert isinstance(tick.get_tick_meta(i, "qec_operations"), list)


def test_relabel_reset_reuse_empty_moments_and_nonascending_measurement_ids():
    plan = ScheduledCircuit(
        3,
        (
            (),
            (O("R", (0,)), O("R", (1,)), O("R", (2,))),
            (O("RELABEL", (0, 1)),),
            (O("M", (0,)), O("RELABEL", (1, 2))),
            (O("R", (0,)),),
            (O("M", (1,)), O("H", (0,)), O("M", (2,))),
            (O("M", (0,)),),
            (),
        ),
        (P((0, 2), "check"),),
        (P((1,), "Logical Z"),),
    )
    tick = plan.to_tick_circuit()
    assert tick.num_ticks() == 7  # Pure relabel adds no physical tick.
    assert tick.get_tick(0).is_empty()
    assert tick.get_tick(6).is_empty()
    assert tick.measurement_emission() == [(1, 0), (2, 1), (0, 2), (1, 3)]
    assert tick.circuit_definitions()["detectors"][0]["measurements"] == [0, 2]
    assert tick.get_meta("qec_output_carriers") == [1, 2, 0]
    assert len(tick.get_meta("qec_relabels")) == 2
    assert all(g.gate_type.name != "SWAP" for _, g in tick.gate_batches())
    assert [tick.get_tick_meta(i, "qec_moment") for i in range(7)] == [0, 1, 3, 4, 5, 6, 7]
    assert len(tick.get_tick_meta(4, "qec_operations")) == 3


@pytest.mark.parametrize("label", ["Logical Z", "Lx", "L", "L+1", "L²", "L\u0660", "L99", "L0"])
def test_opaque_observable_labels_do_not_select_decoder_ids(label):
    plan = ScheduledCircuit(1, ((O("R", (0,)),), (O("M", (0,)),)), (), (P((0,), label),))
    tick = plan.to_tick_circuit()
    assert tick.circuit_definitions()["observables"][0]["id"] == 0
    assert tick.get_meta("qec_observable_labels") == [label]


@pytest.mark.parametrize("detector", [True, False])
@pytest.mark.parametrize(("parity", "match"), [(P((), "empty"), "nonempty")])
def test_unsupported_parities_fail_at_export_boundary(detector, parity, match):
    plan = ScheduledCircuit(
        1, ((O("R", (0,)),), (O("M", (0,)),)), (parity,) if detector else (), () if detector else (parity,),
    )
    with pytest.raises(ValueError, match=match):
        plan.to_tick_circuit()


def test_signed_y_memory_cliffords_and_native_dem_consumer():
    from pecos.qec import DetectorErrorModel

    impl = color_code_implementation(color_code_spec("666", 3), ColorExtraction.SWAP_SUPERDENSE)
    plan = color_state_memory(impl, 2, "-Y")
    tick = plan.to_tick_circuit()
    circuit = stim.Circuit(tick_circuit_to_stim(tick))
    assert _moments(circuit) == _moments(plan.to_stim())
    assert not circuit.compile_detector_sampler(seed=3).sample(64, append_observables=True).any()
    dem = DetectorErrorModel.from_circuit(tick, p1=0.001, p2=0.002, p_meas=0.003, p_prep=0.004)
    assert dem.num_detectors == len(plan.detectors)
    assert "error(" in dem.to_string()
