"""Python API coverage for conditional population-transition channels."""

from __future__ import annotations

import pytest

from pecos_rslib import (
    GeneralNoiseModelBuilder,
    P2TransitionStep,
    StateTransitionChannel,
    TransitionDict,
    TwoQubitStateTransitionChannel,
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


def test_transition_dict_and_channel_products_build_p2_steps() -> None:
    first_dict = TransitionDict({"L": {"0": 1.0}})
    second_dict = TransitionDict({"L": {"1": 1.0}})
    joint = TwoQubitStateTransitionChannel(0.25, first_dict * second_dict)

    first = StateTransitionChannel(0.9, first_dict)
    second = StateTransitionChannel(0.8, second_dict)
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
    channel = StateTransitionChannel(0.01, transitions)

    assert channel.probability == 0.01
    assert channel.transitions == transitions


def test_builder_accepts_ordered_single_qubit_transition_stacks() -> None:
    leak = StateTransitionChannel(1.0, {"0": {"L": 1.0}, "1": {"1": 1.0}})
    recover = StateTransitionChannel.leak_recovery(0.9, p_zero=0.25)

    builder = (
        GeneralNoiseModelBuilder()
        .with_p1_transition_channels_before_gate([leak, recover])
        .add_p1_transition_channel_after_gate(recover)
    )

    assert isinstance(builder, GeneralNoiseModelBuilder)


def test_p2_steps_accept_distinct_legs_and_joint_pair_states() -> None:
    first = StateTransitionChannel.leak_recovery(0.9, p_zero=0.75)
    second = StateTransitionChannel.leak_recovery(0.8, p_zero=0.25)
    independent = P2TransitionStep.independent(first, second)

    joint_channel = TwoQubitStateTransitionChannel(
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
    channel = TwoQubitStateTransitionChannel(1.0, transitions)

    builder = (
        GeneralNoiseModelBuilder()
        .add_p2_transition_channel_before_gate(channel)
        .add_p2_transition_channel_after_gate(channel)
    )

    assert channel.transitions == transitions
    assert isinstance(builder, GeneralNoiseModelBuilder)


@pytest.mark.parametrize("transitions", [{"0->0": 1.0}, {"1": {"1": 1.0}}])
def test_incomplete_rows_explain_measurement(transitions: dict) -> None:
    with pytest.raises(ValueError, match="both '0' and '1'.*MZ.*not the identity quantum map"):
        StateTransitionChannel(1.0, transitions)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_outer_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        StateTransitionChannel.leak_recovery(probability)


@pytest.mark.parametrize(
    "transitions",
    [
        {"L->0": 0.5},
        {"L->0": -1.0, "L->1": 2.0},
        {"L->0": float("nan")},
        {"L->Q": 1.0},
        {"L0": 1.0},
        {},
    ],
)
def test_invalid_transition_maps_raise_value_error(transitions: dict) -> None:
    with pytest.raises(ValueError, match="channel|weight|source rows|wildcard"):
        StateTransitionChannel(1.0, transitions)


@pytest.mark.parametrize(
    "transitions",
    [
        {"*L->00": 1.0},
        {"*L->L*": 1.0},
        {"*L->*0": 1.0, "LL->11": 1.0},
        {"*0->*0": 1.0},
    ],
)
def test_invalid_wildcard_groups(transitions: dict) -> None:
    with pytest.raises(ValueError, match="channel|weight|source rows|wildcard"):
        TwoQubitStateTransitionChannel(1.0, transitions)


def test_flat_transition_syntax_round_trips_as_rows() -> None:
    channel = StateTransitionChannel(0.4, {"0->L": 1.0, "1->1": 1.0})
    assert channel.transitions == {"0": {"L": 1.0}, "1": {"1": 1.0}}
    joint = TwoQubitStateTransitionChannel(1.0, {"*L->*0": 1.0})
    assert joint.transitions == {"*L": {"*0": 1.0}}


def test_mixed_family_insertion_order_changes_gate_site_state() -> None:
    from pecos_rslib import PauliLeakageChannel, Qasm, sim

    program = Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; z q[0]; measure q -> c;')
    leak = PauliLeakageChannel(1.0, {"L": 1.0})
    recover = StateTransitionChannel.leak_recovery(1.0, p_zero=1.0)
    recovered = (
        GeneralNoiseModelBuilder()
        .add_p1_pauli_leakage_channel_before_gate(leak)
        .add_p1_transition_channel_before_gate(recover)
    )
    leaked = (
        GeneralNoiseModelBuilder()
        .add_p1_transition_channel_before_gate(recover)
        .add_p1_pauli_leakage_channel_before_gate(leak)
    )
    assert sim(program).noise(recovered).seed(42).run(16).to_dict()["c"] == [0] * 16
    assert sim(program).noise(leaked).seed(42).run(16).to_dict()["c"] == [1] * 16


def test_identity_rows_collapse_in_sim_and_internal_results_stay_hidden() -> None:
    from pecos_rslib import Qasm, sim

    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; h q[0]; h q[0]; measure q -> c;'
    )
    channel = StateTransitionChannel(1.0, {"0->0": 1.0, "1->1": 1.0})
    noise = GeneralNoiseModelBuilder().add_p1_transition_channel_before_gate(channel)
    outcomes = sim(program).noise(noise).seed(51).run(256).to_dict()["c"]
    assert len(outcomes) == 256
    assert 85 < sum(outcomes) < 171


@pytest.mark.parametrize("weight", [1.000001, float.fromhex("0x1.0000000000001p+0")])
@pytest.mark.parametrize("joint", [False, True])
def test_transition_weights_cannot_exceed_one(weight: float, joint: bool) -> None:
    cls = TwoQubitStateTransitionChannel if joint else StateTransitionChannel
    label = "*L->*0" if joint else "L->0"
    with pytest.raises(ValueError, match="between 0 and 1"):
        cls(1.0, {label: weight})


@pytest.mark.parametrize("state_channel", [False, True])
@pytest.mark.parametrize("two_qubit", [False, True])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_explicit_leakage_ignores_legacy_depolarization_scale(
    state_channel: bool, two_qubit: bool, placement: str
) -> None:
    from pecos_rslib import PauliLeakageChannel, Qasm, sim

    gate = "cz q[0], q[1];" if two_qubit else "z q[0]; z q[1];"
    program = Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; ' + gate + " measure q -> c;")
    channel = (
        StateTransitionChannel(1.0, {"0->L": 1.0, "1->L": 1.0})
        if state_channel
        else PauliLeakageChannel(1.0, {"L": 1.0})
    )
    builder = GeneralNoiseModelBuilder().with_leakage_scale(0.0)
    arity = "p2" if two_qubit else "p1"
    family = "transition" if state_channel else "pauli_leakage"
    noise = getattr(builder, f"add_{arity}_{family}_channel_{placement}_gate")(channel)
    assert sim(program).noise(noise).seed(42).run(32).to_dict()["c"] == [3] * 32
