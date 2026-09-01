"""Python API coverage for conditional population-transition channels."""

from __future__ import annotations

from pecos_rslib import (
    GeneralNoiseModelBuilder,
    P2TransitionStep,
    TransitionChannel,
    TransitionDict,
    TwoQubitTransitionChannel,
)


def test_transition_dict_tensor_product_and_mapping_api() -> None:
    recover_first = TransitionDict({"L": {"0": 1.0}})
    recover_second = TransitionDict({"L": {"1": 1.0}})

    product = recover_first * recover_second

    assert product.arity == 2
    assert len(product) == 5
    assert product["0L"] == {"01": 1.0}
    assert product["L0"] == {"00": 1.0}
    assert product["LL"] == {"01": 1.0}
    assert "LL" in product
    assert product.get("missing") is None
    assert product.keys() == ["0L", "1L", "L0", "L1", "LL"]
    assert product.values()[-1] == {"01": 1.0}
    assert product.to_dict() == product.transitions


def test_transition_dict_composes_in_matrix_order() -> None:
    leak_zero = TransitionDict({"0": {"L": 1.0}})
    recover_to_one = TransitionDict({"L": {"1": 1.0}})

    composed = recover_to_one @ leak_zero

    assert composed["0"] == {"1": 1.0}
    assert leak_zero.then(recover_to_one).transitions == composed.transitions


def test_transition_dict_and_channel_products_build_p2_steps() -> None:
    first_dict = TransitionDict({"L": {"0": 1.0}})
    second_dict = TransitionDict({"L": {"1": 1.0}})
    joint = TwoQubitTransitionChannel(0.25, first_dict * second_dict)

    first = TransitionChannel(0.9, first_dict)
    second = TransitionChannel(0.8, second_dict)
    independent = first * second

    builder = GeneralNoiseModelBuilder().with_p2_transition_steps_before_gate(
        [independent, P2TransitionStep.joint(joint)]
    )

    assert joint.transition_dict.arity == 2
    assert P2TransitionStep.tensor_product(first, second) is not None
    assert isinstance(builder, GeneralNoiseModelBuilder)


def test_single_qubit_transition_channel_round_trips_configuration() -> None:
    transitions = {
        "0": {"0": 0.05, "1": 0.05, "L": 0.90},
        "1": {"0": 0.10, "1": 0.80, "L": 0.10},
        "L": {"0": 0.45, "1": 0.45, "L": 0.10},
    }
    channel = TransitionChannel(0.01, transitions)

    assert channel.probability == 0.01
    assert channel.transitions == transitions


def test_builder_accepts_ordered_single_qubit_transition_stacks() -> None:
    leak = TransitionChannel(1.0, {"0": {"L": 1.0}})
    recover = TransitionChannel.leak_recovery(0.9, p_zero=0.25)

    builder = (
        GeneralNoiseModelBuilder()
        .with_p1_transition_channels_before_gate([leak, recover])
        .add_p1_transition_channel_after_gate(recover)
    )

    assert isinstance(builder, GeneralNoiseModelBuilder)


def test_p2_steps_accept_distinct_legs_and_joint_pair_states() -> None:
    first = TransitionChannel.leak_recovery(0.9, p_zero=0.75)
    second = TransitionChannel.leak_recovery(0.8, p_zero=0.25)
    independent = P2TransitionStep.independent(first, second)

    joint_channel = TwoQubitTransitionChannel(
        0.02,
        {
            "0L": {"L1": 0.2, "0L": 0.8},
            "LL": {"00": 0.5, "11": 0.5},
        },
    )
    joint = P2TransitionStep.joint(joint_channel)

    builder = (
        GeneralNoiseModelBuilder()
        .with_p2_transition_steps_before_gate([independent])
        .add_p2_transition_channel_before_gate(joint_channel)
        .with_p2_transition_steps_after_gate([joint])
        .add_p2_transition_channel_after_gate(joint_channel)
        .add_p2_transition_channel_after_gate(first)
    )

    assert joint_channel.transitions["0L"]["L1"] == 0.2
    assert isinstance(builder, GeneralNoiseModelBuilder)


def test_two_qubit_wildcards_are_identity_wires_on_either_leg() -> None:
    transitions = {
        "*L": {"*0": 0.45, "*1": 0.45, "*L": 0.10},
        "L*": {"0*": 0.45, "1*": 0.45, "L*": 0.10},
    }
    channel = TwoQubitTransitionChannel(1.0, transitions)

    builder = (
        GeneralNoiseModelBuilder()
        .add_p2_transition_channel_before_gate(channel)
        .add_p2_transition_channel_after_gate(channel)
    )

    assert channel.transitions == transitions
    assert isinstance(builder, GeneralNoiseModelBuilder)
