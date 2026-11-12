"""Loss configuration shares one Rust parser with structured JSON."""

import json

import numpy as np
import pytest
from pecos_rslib import general_noise, sim
from pecos_rslib.programs import Qasm


def program() -> Qasm:
    """Measure one excited qubit twice, retaining both outcomes."""
    return Qasm.from_string(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg a[1]; creg b[1]; '
        "x q[0]; measure q[0] -> a[0]; measure q[0] -> b[0];",
    )


def test_t25_json_and_dict_parity() -> None:
    loss = {"enabled": True, "default_recovery": False, "measurement": 1.0}
    direct = general_noise().with_seed(7919).with_loss(loss)
    structured = general_noise().with_seed(7919).with_config_json(json.dumps({"loss": loss}))
    actual = sim(program()).noise(direct).seed(65537).run(20).to_dict()
    expected = sim(program()).noise(structured).seed(65537).run(20).to_dict()
    assert actual == expected
    assert actual["a"] == [1] * 20
    assert actual["b"] == [0] * 20


@pytest.mark.parametrize(
    "loss",
    [
        {"enabled": True, "measurement": True},
        {"enabled": True, "prep": "1"},
        {"enabled": False, "idle_rate": 0},
        {},
        {"enabled": True, "typo": 0},
    ],
)
def test_t25_identical_validation_text(loss: dict) -> None:
    with pytest.raises(ValueError, match=r"loss\.") as direct:
        general_noise().with_loss(loss)
    with pytest.raises(ValueError, match=r"loss\.") as structured:
        general_noise().with_config_json(json.dumps({"loss": loss}))
    assert str(direct.value) == str(structured.value)


@pytest.mark.parametrize(
    "value",
    [None, float("inf"), float("nan"), (1,), {1}, np.float64(0.1), np.int64(1), 2**64],
)
def test_strict_python_conversion(value: object) -> None:
    with pytest.raises(ValueError, match=r"loss\.measurement ="):
        general_noise().with_loss({"enabled": True, "measurement": value})


def test_non_string_keys() -> None:
    with pytest.raises(ValueError, match=r"loss\[1\] = 1: use str keys"):
        general_noise().with_loss({"enabled": True, 1: 0})
    with pytest.raises(ValueError, match=r"loss\.lost_partner\.pauli_model\[0\] = 0: use str keys"):
        general_noise().with_loss({"enabled": True, "lost_partner": {"pauli_model": {0: 1}}})


def test_lost_row_with_loss_builds_and_disabling_clears() -> None:
    noise = general_noise().with_loss({"enabled": True, "default_recovery": False})
    noise = noise.with_readout("MeasureZLost", {"lost": {"3": 1.0}})
    assert sim(program()).noise(noise).run(1).to_dict()["a"] == [1]
    with pytest.raises(ValueError, match="lost row needs qubit loss"):
        sim(program()).noise(noise.with_loss({"enabled": False})).run(1)


def test_recursive_channels_accept_lists_and_integer_probabilities() -> None:
    noise = general_noise().with_loss(
        {
            "enabled": True,
            "default_recovery": False,
            "channels": [
                {
                    "site": "after_single_qubit",
                    "kind": "events",
                    "probability": 1,
                    "events": {"L": 1},
                },
            ],
        },
    )
    result = sim(program()).noise(noise).run(1).to_dict()
    assert result["a"] == [0]


@pytest.mark.parametrize("index", [0, 1])
def test_nested_conversion_error_names_channel_path(index: int) -> None:
    channel = {
        "site": "before_single_qubit",
        "kind": "events",
        "probability": 0,
        "events": {"L": 1},
    }
    channels = [channel.copy() for _ in range(index + 1)]
    channels[index]["probability"] = (1, 2)
    with pytest.raises(ValueError, match="use dict, list, str, bool, i64 int or finite float") as caught:
        general_noise().with_loss({"enabled": True, "channels": channels})
    assert str(caught.value) == (
        f"loss.channels[{index}].probability = (1, 2): " "use dict, list, str, bool, i64 int or finite float"
    )


@pytest.mark.parametrize("structured", [False, True])
def test_scaled_loss_error_has_no_input_prefix(structured: bool) -> None:
    loss = {"enabled": True, "measurement": 0.6}
    noise = (
        general_noise().with_config_json(json.dumps({"loss": loss, "scaling": {"overall": 2}}))
        if structured
        else general_noise().with_loss(loss).with_scale(2)
    )
    with pytest.raises(ValueError, match="loss.measurement after scaling") as caught:
        sim(program()).noise(noise).run(1)
    assert str(caught.value) == (
        "loss.measurement after scaling = 1.2 (0.6 x scale 2): "
        "reduce the probability or scale factors to keep the scaled probability finite in [0, 1]"
    )
