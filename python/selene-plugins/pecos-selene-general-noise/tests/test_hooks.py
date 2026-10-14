"""Hook configuration tests that require only the plugin's runtime dependency."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pecos_selene_general_noise import (
    GateNoise,
    GeneralNoiseParameters,
    GeneralNoisePlugin,
    IndependentHook,
    PauliLeakageHook,
    TransitionHook,
    TwoQubitGateNoise,
)


def config(parameters: GeneralNoiseParameters) -> dict:
    return json.loads(GeneralNoisePlugin(parameters=parameters).get_init_args()[0])


def test_hook_specs_preserve_order_and_freeze_rows() -> None:
    rows = {"L": {"0": 1.0}}
    recover = TransitionHook(1.0, rows)
    leak = PauliLeakageHook(1.0, {"L": 1.0})
    rows["L"]["0"] = 0.0
    parameters = (
        GeneralNoiseParameters()
        .add_p1_pauli_leakage_channel_before_gate(leak)
        .add_p1_transition_channel_before_gate(recover)
    )
    hooks = config(parameters)["single_qubit"]["hooks_before"]
    assert [hook["kind"] for hook in hooks] == ["pauli_leakage", "transition"]
    assert hooks[1]["transitions"] == {"L": {"0": 1.0}}
    replaced = parameters.with_p1_pauli_leakage_channels_before_gate((leak,))
    assert replaced.single_qubit.hooks_before == (recover, leak)
    assert parameters.single_qubit.hooks_before == (leak, recover)
    with pytest.raises(TypeError):
        TransitionHook(1.0, {}, unknown=True)


def test_hook_shape_and_effective_probability() -> None:
    leak = PauliLeakageHook(0.6, {"L": 1.0})
    with pytest.raises(ValueError, match="arity"):
        GeneralNoiseParameters(single_qubit=GateNoise(hooks_before=(TransitionHook(1.0, {"LL->00": 1.0}),)))
    with pytest.raises(ValueError, match=r"effective .* probability"):
        GeneralNoisePlugin(
            parameters=GeneralNoiseParameters(
                two_qubit=TwoQubitGateNoise(
                    hooks_after=(IndependentHook(leak, PauliLeakageHook(0.1, {"X": 1.0})),),
                    scale=2.0,
                ),
            ),
        )


@pytest.mark.parametrize("family", ["transition", "pauli_leakage"])
@pytest.mark.parametrize("timing", ["before", "after"])
def test_bridge_hook_calls(monkeypatch: pytest.MonkeyPatch, family: str, timing: str) -> None:
    import pecos_selene_general_noise.plugin as plugin

    def constructor(name):
        return lambda *args: (name, *args)

    fake = SimpleNamespace(
        **{
            name: constructor(name)
            for name in (
                "StateTransitionChannel",
                "TwoQubitStateTransitionChannel",
                "PauliLeakageChannel",
                "TwoQubitPauliLeakageChannel",
            )
        },
    )
    for name in ("P2TransitionStep", "P2PauliLeakageStep"):
        setattr(
            fake,
            name,
            SimpleNamespace(independent=constructor(name + ".independent"), joint=constructor(name + ".joint")),
        )
    monkeypatch.setattr(plugin, "import_module", lambda _name: fake)

    class RecordingBuilder:
        def __init__(self, calls=()):
            self.calls = calls

        def __getattr__(self, name):
            return lambda *args: RecordingBuilder((*self.calls, (name, args)))

    single = TransitionHook(0.25, {"L->0": 1.0}) if family == "transition" else PauliLeakageHook(0.25, {"L": 1.0})
    joint = TransitionHook(0.5, {"LL->00": 1.0}) if family == "transition" else PauliLeakageHook(0.5, {"LI": 1.0})
    other = PauliLeakageHook(0.1, {"X": 1.0}) if family == "transition" else TransitionHook(0.1, {"L->0": 1.0})
    parameters = GeneralNoiseParameters(
        single_qubit=GateNoise(**{f"hooks_{timing}": (single, other)}),
        two_qubit=TwoQubitGateNoise(**{f"hooks_{timing}": (single, joint, IndependentHook(single, single))}),
    )
    original = RecordingBuilder()
    calls = parameters.apply_to_pecos_builder(original).calls
    assert original.calls == ()
    assert [name for name, _ in calls] == [
        f"add_p1_{family}_channel_{timing}_gate",
        f"add_p1_{other.kind}_channel_{timing}_gate",
        f"add_p2_{family}_channel_{timing}_gate",
        f"add_p2_{family}_{'channel' if family == 'transition' else 'step'}_{timing}_gate",
        f"add_p2_{family}_step_{timing}_gate",
    ]
    assert calls[0][1][0][1:] == (0.25, {"L->0": 1.0} if family == "transition" else {"L": 1.0})
    assert calls[-1][1][0][0].endswith(".independent")


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_hook_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        GeneralNoiseParameters(single_qubit=GateNoise(hooks_before=(PauliLeakageHook(probability, {"X": 1.0}),)))


def test_hook_validation_leaves_rows_to_rust() -> None:
    parameters = GeneralNoiseParameters().add_p1_transition_channel_after_gate(TransitionHook(1.0, {"0->L": 1.0}))
    assert config(parameters)["single_qubit"]["hooks_after"][0]["transitions"] == {"0->L": 1.0}
    with pytest.raises(ValueError, match="family"):
        GeneralNoiseParameters(
            two_qubit=TwoQubitGateNoise(
                hooks_before=(IndependentHook(TransitionHook(1.0, {"L->0": 1.0}), PauliLeakageHook(1.0, {"L": 1.0})),),
            ),
        )


@pytest.mark.parametrize(
    "hook",
    [
        TransitionHook(0.25, {"L->0": 1.0}),
        TransitionHook(0.5, {"L": {"0": 1.0}}),
        PauliLeakageHook(0.75, {"LI": 1.0}),
        IndependentHook(PauliLeakageHook(0.25, {"L": 1.0}), PauliLeakageHook(0.5, {"X": 1.0})),
    ],
)
def test_hook_json_round_trip(hook) -> None:
    def decode(value):
        value = dict(value)
        kind = value.pop("kind")
        if kind == "independent":
            return IndependentHook(decode(value["first"]), decode(value["second"]))
        return {"transition": TransitionHook, "pauli_leakage": PauliLeakageHook}[kind](**value)

    parameters = GeneralNoiseParameters(two_qubit=TwoQubitGateNoise(hooks_after=(hook,)))
    encoded = config(parameters)["two_qubit"]["hooks_after"][0]
    assert decode(encoded) == hook
    if encoded["kind"] != "independent":
        encoded["unknown"] = True
        with pytest.raises(TypeError):
            decode(encoded)


@pytest.mark.parametrize("arity", [1, 2])
@pytest.mark.parametrize("timing", ["before", "after"])
@pytest.mark.parametrize("family", ["transition", "pauli_leakage"])
def test_all_family_replacements_preserve_other_family(arity: int, timing: str, family: str) -> None:
    transition = TransitionHook(1.0, {"L->0": 1.0})
    pauli = PauliLeakageHook(1.0, {"L": 1.0})
    hook, other = (transition, pauli) if family == "transition" else (pauli, transition)
    parameters = getattr(GeneralNoiseParameters(), f"add_p{arity}_{family}_channel_{timing}_gate")(hook)
    parameters = getattr(parameters, f"add_p{arity}_{other.kind}_channel_{timing}_gate")(other)
    plural = "channels" if arity == 1 else "steps"
    replace_hooks = getattr(parameters, f"with_p{arity}_{family}_{plural}_{timing}_gate")
    site = "single_qubit" if arity == 1 else "two_qubit"
    assert getattr(getattr(replace_hooks((hook,)), site), f"hooks_{timing}") == (other, hook)
    assert getattr(getattr(replace_hooks(()), site), f"hooks_{timing}") == (other,)
