"""Python API coverage for conditional population-transition channels."""

from __future__ import annotations

import pytest

from pecos_rslib import (
    GeneralNoiseModelBuilder,
    P2TransitionStep,
    PauliLeakageChannel,
    Qasm,
    StateTransitionChannel,
    TransitionDict,
    TwoQubitStateTransitionChannel,
    sim,
)


def test_transition_dict_tensor_product_and_mapping_api() -> None:
    recover_first = TransitionDict({"L": {"0": 1.0}})
    recover_second = TransitionDict({"L": {"1": 1.0}})

    product = recover_first * recover_second

    assert product.arity == 2
    assert len(product) == 2
    assert product["L*"] == {"0*": 1.0}
    assert product["*L"] == {"*1": 1.0}
    assert "L*" in product
    assert product.get("missing") is None
    assert product.keys() == ["*L", "L*"]
    assert product.values()[-1] == {"0*": 1.0}
    assert product.to_dict() == product.transitions


@pytest.mark.parametrize("row_form", ["tensor", "transitions", "to_dict", "channel"])
def test_transition_tensor_plain_rows_preserve_partner_coherence(row_form: str) -> None:
    product = TransitionDict({"0": {"1": 1.0}, "1": {"0": 1.0}}) * TransitionDict({"L": {"L": 1.0}})
    original = TwoQubitStateTransitionChannel(1.0, product)
    rows = {
        "tensor": product,
        "transitions": product.transitions,
        "to_dict": product.to_dict(),
        "channel": original.transitions,
    }
    channel = TwoQubitStateTransitionChannel(1.0, rows[row_form])
    builder = GeneralNoiseModelBuilder().add_p2_transition_channel_before_gate(channel)
    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[1]; '
        "h q[1]; cx q[0],q[1]; h q[1]; measure q[1] -> c[0];"
    )
    assert sim(program).noise(builder).seed(1).run(400).to_dict()["c"] == [0] * 400


@pytest.mark.parametrize("step_form", ["product", "joint", "tensor", "stack"])
def test_transition_dict_and_channel_products_build_p2_steps(step_form: str) -> None:
    first_dict = TransitionDict({"L": {"0": 1.0}})
    second_dict = TransitionDict({"L": {"1": 1.0}})
    joint = TwoQubitStateTransitionChannel(1.0, first_dict * second_dict)

    first = StateTransitionChannel(1.0, first_dict)
    second = StateTransitionChannel(1.0, second_dict)
    independent = first * second
    steps = {
        "product": [independent],
        "joint": [P2TransitionStep.joint(joint)],
        "tensor": [P2TransitionStep.tensor_product(first, second)],
        "stack": [independent, P2TransitionStep.joint(joint)],
    }
    builder = GeneralNoiseModelBuilder().with_p2_transition_steps_before_gate(steps[step_form])
    # RZZ(0) is a native P2 identity; unlike cz, it adds no P1 decomposition hooks.
    # Prepare LL at the Z gates; each product must recover the ordered pair to 01.
    builder = builder.add_p1_pauli_leakage_channel_before_gate(PauliLeakageChannel(1.0, {"L": 1.0}))
    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
        "z q[0]; z q[1]; RZZ(0) q[0], q[1]; measure q -> c;"
    )
    assert joint.transition_dict.arity == 2
    assert sim(program).noise(builder).seed(42).run(16).to_dict()["c"] == [2] * 16


def test_single_qubit_transition_channel_round_trips_configuration() -> None:
    transitions = {
        "0": {"0": 0.05, "1": 0.05, "L": 0.90},
        "1": {"0": 0.10, "1": 0.80, "L": 0.10},
        "L": {"0": 0.45, "1": 0.45, "L": 0.10},
    }
    channel = StateTransitionChannel(0.01, transitions)

    assert channel.probability == 0.01
    assert channel.transitions == transitions


@pytest.mark.parametrize("placement", ["stack", "after"])
def test_builder_accepts_ordered_single_qubit_transition_stacks(placement: str) -> None:
    leak = StateTransitionChannel(1.0, {"0": {"L": 1.0}, "1": {"L": 1.0}})
    recover = StateTransitionChannel.leak_recovery(1.0, p_zero=1.0)
    builder = (
        GeneralNoiseModelBuilder()
        .with_p1_transition_channels_before_gate([leak, recover] if placement == "stack" else [leak])
        .add_p1_transition_channel_after_gate(recover)
    )
    # In stack order, each X acts on recovered zero: two X gates still yield one.
    # With recovery only after the gate, X is suppressed and recovery yields zero.
    gates, expected = ("x q[0]; x q[0];", 1) if placement == "stack" else ("x q[0];", 0)
    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; ' + gates + " measure q -> c;"
    )
    assert sim(program).noise(builder).seed(42).run(16).to_dict()["c"] == [expected] * 16


@pytest.mark.parametrize(
    "hook",
    ["before_steps", "before_channel", "after_steps", "after_joint", "after_single"],
)
def test_p2_steps_accept_distinct_legs_and_joint_pair_states(hook: str) -> None:
    first = StateTransitionChannel.leak_recovery(1.0, p_zero=1.0)
    second = StateTransitionChannel.leak_recovery(1.0, p_zero=0.0)
    independent = P2TransitionStep.independent(first, second)
    joint_channel = TwoQubitStateTransitionChannel(
        1.0,
        {"0L": {"L1": 1.0}, "1L": {"1L": 1.0}, "LL": {"00": 1.0}},
    )
    joint = P2TransitionStep.joint(joint_channel)
    builders = {
        "before_steps": GeneralNoiseModelBuilder().with_p2_transition_steps_before_gate([independent]),
        "before_channel": GeneralNoiseModelBuilder().add_p2_transition_channel_before_gate(joint_channel),
        "after_steps": GeneralNoiseModelBuilder().with_p2_transition_steps_after_gate([joint]),
        "after_joint": GeneralNoiseModelBuilder().add_p2_transition_channel_after_gate(joint_channel),
        "after_single": GeneralNoiseModelBuilder().add_p2_transition_channel_after_gate(first),
    }
    builder = builders[hook]
    builder = builder.add_p1_pauli_leakage_channel_before_gate(PauliLeakageChannel(1.0, {"L": 1.0}))
    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
        "z q[0]; z q[1]; RZZ(0) q[0], q[1]; measure q -> c;"
    )
    # Independent recovery distinguishes the legs (01); joint and same-leg recovery yield 00.
    expected = 2 if hook == "before_steps" else 0
    assert joint_channel.transitions["0L"]["L1"] == 1.0
    assert sim(program).noise(builder).seed(42).run(16).to_dict()["c"] == [expected] * 16


@pytest.mark.parametrize("leaked_leg", [0, 1])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_two_qubit_wildcards_are_identity_wires_on_either_leg(leaked_leg: int, placement: str) -> None:
    from pecos_rslib import P2PauliLeakageStep, TwoQubitPauliLeakageChannel

    transitions = {"*L": {"*0": 1.0}, "L*": {"0*": 1.0}}
    channel = TwoQubitStateTransitionChannel(1.0, transitions)
    # Leak only one operand at the native two-qubit identity RZZ(0), leaving the partner in |+>.
    preparation = GeneralNoiseModelBuilder().add_p2_pauli_leakage_step_before_gate(
        P2PauliLeakageStep.joint(TwoQubitPauliLeakageChannel(1.0, {"LI" if leaked_leg == 0 else "IL": 1.0}))
    )
    builder = (
        preparation.add_p2_transition_channel_before_gate(channel)
        if placement == "before"
        else preparation.add_p2_transition_channel_after_gate(channel)
    )
    partner = 1 - leaked_leg
    program = Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
        f"h q[{partner}]; RZZ(0) q[0], q[1]; h q[{partner}]; measure q -> c;"
    )
    # Recovery gives zero, and the wildcard preserves interference: H|+> = |0>.
    # Measuring/collapsing the wildcard leg would instead give random partner outcomes.
    assert channel.transitions == transitions
    assert sim(program).noise(builder).seed(42).run(32).to_dict()["c"] == [0] * 32


@pytest.mark.parametrize(("transitions", "missing"), [({"0->0": 1.0}, "1"), ({"1": {"1": 1.0}}, "0")])
def test_incomplete_rows_explain_measurement(transitions: dict, missing: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"pattern C: missing source rows {missing}.*MZ.*not the identity quantum map",
    ) as error:
        StateTransitionChannel(1.0, transitions)
    assert f"'{missing}->{missing}': 1.0" in str(error.value)


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


@pytest.mark.parametrize("joint", [False, True])
def test_overscaled_hook_probability_raises_value_error(joint: bool) -> None:
    from pecos_rslib import Qasm, sim

    program = Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; measure q -> c;')
    builder = GeneralNoiseModelBuilder().with_scale(2.0)
    if joint:
        channel = TwoQubitStateTransitionChannel(0.8, {"LL->00": 1.0})
        noise = builder.add_p2_transition_channel_before_gate(channel)
    else:
        channel = StateTransitionChannel.leak_recovery(0.8)
        noise = builder.add_p1_transition_channel_before_gate(channel)
    with pytest.raises(ValueError, match="scaled .*hook probabilities.*between zero and one"):
        sim(program).noise(noise).run(1)


@pytest.mark.parametrize(
    ("rows", "pattern", "missing"),
    [
        ({"0L->LL": 1.0}, "CL", "1L"),
        ({"L0->LL": 1.0}, "LC", "L1"),
        ({"00->LL": 1.0}, "CC", "01, 10, 11"),
    ],
)
def test_incomplete_joint_rows_name_missing_sources(rows: dict, pattern: str, missing: str) -> None:
    with pytest.raises(ValueError, match=f"pattern {pattern}: missing source rows {missing}"):
        TwoQubitStateTransitionChannel(1.0, rows)
