"""Python API coverage for stochastic Pauli-plus-leakage channels."""

from __future__ import annotations

import pytest

from pecos_rslib import (
    GeneralNoiseModelBuilder,
    P2PauliLeakageStep,
    PauliLeakageChannel,
    PauliLeakageDict,
    TwoQubitPauliLeakageChannel,
)


def test_pauli_leakage_dict_tensor_product_and_mapping_api() -> None:
    first = PauliLeakageDict({"X": 0.25, "L": 0.75})
    second = PauliLeakageDict({"Y": 0.4, "Z": 0.6})

    product = first * second

    assert product.arity == 2
    assert len(product) == 4
    assert product["XY"] == pytest.approx(0.1)
    assert product["LZ"] == pytest.approx(0.45)
    assert "LY" in product
    assert product.get("missing") is None
    assert product.to_dict() == product.events


def test_single_and_two_qubit_channels_round_trip_configuration() -> None:
    events = PauliLeakageDict({"X": 0.4, "Y": 0.2, "Z": 0.3, "L": 0.1})
    single = PauliLeakageChannel(0.001, events)
    joint = TwoQubitPauliLeakageChannel(
        0.01,
        {"IX": 0.2, "XI": 0.2, "XX": 0.2, "IL": 0.1, "LI": 0.1, "LL": 0.2},
    )

    assert single.probability == 0.001
    assert single.event_dict.events == events.events
    assert joint.probability == 0.01
    assert joint.events["LL"] == 0.2
    assert joint.event_dict.arity == 2


def test_channel_products_and_all_builder_hooks() -> None:
    first = PauliLeakageChannel(0.9, {"L": 1.0})
    second = PauliLeakageChannel(0.8, {"X": 1.0})
    independent = first * second
    joint = P2PauliLeakageStep.joint(TwoQubitPauliLeakageChannel(0.1, {"XL": 1.0}))

    builder = (
        GeneralNoiseModelBuilder()
        .with_p1_pauli_leakage_channels_before_gate([first])
        .add_p1_pauli_leakage_channel_before_gate(second)
        .with_p1_pauli_leakage_channels_after_gate([second])
        .add_p1_pauli_leakage_channel_after_gate(first)
        .with_p2_pauli_leakage_steps_before_gate([independent])
        .add_p2_pauli_leakage_step_before_gate(joint)
        .add_p2_pauli_leakage_channel_before_gate(first)
        .with_p2_pauli_leakage_steps_after_gate([joint])
        .add_p2_pauli_leakage_step_after_gate(independent)
        .add_p2_pauli_leakage_channel_after_gate(second)
    )

    assert P2PauliLeakageStep.tensor_product(first, second) is not None
    assert P2PauliLeakageStep.same_on_each(first) is not None
    assert isinstance(builder, GeneralNoiseModelBuilder)


@pytest.mark.parametrize("events", [{"I": 1.0}, {"I": 0.3, "X": 0.7}])
def test_explicit_identity_events(events: dict) -> None:
    assert PauliLeakageChannel(1.0, events).events == events
    assert TwoQubitPauliLeakageChannel(1.0, {"II": 1.0}).events == {"II": 1.0}


@pytest.mark.parametrize("events", [{}, {"X": 2.0}, {"Q": 1.0}, {"X": float("nan")}, {"X": -0.1, "Z": 1.1}])
def test_invalid_event_distributions(events: dict) -> None:
    with pytest.raises(ValueError, match="channel|weight|source rows|wildcard"):
        PauliLeakageChannel(1.0, events)


@pytest.mark.parametrize("weight", [1.000001, float.fromhex("0x1.0000000000001p+0")])
@pytest.mark.parametrize("joint", [False, True])
def test_event_weights_cannot_exceed_one(weight: float, joint: bool) -> None:
    cls = TwoQubitPauliLeakageChannel if joint else PauliLeakageChannel
    event = "XL" if joint else "X"
    with pytest.raises(ValueError, match="between 0 and 1"):
        cls(1.0, {event: weight})


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("placement", ["before", "after"])
@pytest.mark.parametrize("mixed", [False, True])
def test_identity_events_attach_and_execute(joint: bool, placement: str, mixed: bool) -> None:
    from pecos_rslib import Qasm, sim

    identity, flip = ("II", "XI") if joint else ("I", "X")
    events = {identity: 0.75, flip: 0.25} if mixed else {identity: 1.0}
    cls = TwoQubitPauliLeakageChannel if joint else PauliLeakageChannel
    channel = cls(0.8 if mixed else 1.0, PauliLeakageDict(events))
    builder = GeneralNoiseModelBuilder()
    if joint:
        noise = getattr(builder, f"add_p2_pauli_leakage_step_{placement}_gate")(P2PauliLeakageStep.joint(channel))
        gates = "cz q[0], q[1];" if mixed else "h q[0]; cz q[0], q[1]; h q[0];"
    else:
        noise = getattr(builder, f"add_p1_pauli_leakage_channel_{placement}_gate")(channel)
        gates = "z q[0];" if mixed else "h q[0]; h q[0];"
    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; ' + gates + " measure q -> c;"
    )
    outcomes = sim(program).noise(noise).seed(41).run(1024).to_dict()["c"]
    assert len(outcomes) == 1024
    assert set(outcomes) <= {0, 1}
    if mixed:
        assert 150 < sum(outcomes) < 260
    else:
        assert outcomes == [0] * 1024
