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
