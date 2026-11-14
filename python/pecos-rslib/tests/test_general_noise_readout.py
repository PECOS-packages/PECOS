"""Conditional readout configuration and Python error boundaries."""

import json

import pytest
from pecos_rslib import general_noise, sim
from pecos_rslib.programs import Qasm


def program() -> Qasm:
    """Read a qubit prepared in zero."""
    return Qasm.from_string('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; creg c[1]; measure q[0] -> c[0];')


def test_readout_builds_and_changes_results() -> None:
    noise = general_noise().with_seed(42).with_readout("Measure", {"0": {"1": 1.0}})
    results = sim(program()).noise(noise).run(20).to_dict()
    assert results["c"] == [1] * 20


@pytest.mark.parametrize("name", ["MZ", "MeasureZLeaked", "MeasureZLost", "MeasureZLeakedLost"])
def test_readout_flavor_configuration_is_accepted_by_builder(name: str) -> None:
    # This checks configuration acceptance. Engines' Python programs cannot emit
    # the new heralds; TickCircuit's neo path rejects configured readout matrices.
    # Per-flavor execution is covered by the Rust readout tests.
    noise = general_noise().with_readout(name, {"0": {"0": 1.0}})
    assert sim(program()).noise(noise).run(1).to_dict()["c"] == [0]


@pytest.mark.parametrize(
    "rows",
    [
        {"0": {"2": 1.0}},
        {"0": {"0": -0.1}},
        {"0": {"0": float("nan")}},
        {"0": {"0": float("inf")}},
        {"0": {"0": 0.1}},
        {"typo": {"0": 1.0}},
        {"0": {}},
        {},
        {0: {"0": 1.0}},
        {"0": {0: 1.0}},
    ],
)
def test_bad_readout_raises_value_error(rows: dict) -> None:
    with pytest.raises(ValueError, match="MZ"):
        general_noise().with_readout("MZ", rows)


@pytest.mark.parametrize("json_config", [False, True])
def test_lost_row_rejected_when_python_builds_model(json_config: bool) -> None:
    rows = {"lost": {"3": 1.0}}
    noise = (
        general_noise().with_config_json(json.dumps({"measurement": {"readout": {"MeasureZLost": rows}}}))
        if json_config
        else general_noise().with_readout("MeasureZLost", rows)
    )
    with pytest.raises(ValueError, match="lost row needs qubit loss"):
        sim(program()).noise(noise).run(1)


@pytest.mark.parametrize("value", [True, False])
def test_readout_rejects_boolean_probability(value: bool) -> None:
    with pytest.raises(ValueError, match="not bool") as caught:
        general_noise().with_readout("Measure", {"0": {"1": value}})
    message = str(caught.value)
    for part in ["Measure", 'row "0"', f'column "1" = {value}', "not bool"]:
        assert part in message


@pytest.mark.parametrize("value", ["not a number", None])
def test_readout_rejects_non_numeric_probability_with_context(value: str | None) -> None:
    with pytest.raises(ValueError, match="numeric probability") as caught:
        general_noise().with_readout("Measure", {"0": {"1": value}})
    message = str(caught.value)
    for part in ["Measure", 'row "0"', f'column "1" = {value}']:
        assert part in message


def test_readout_accepts_integer_probability() -> None:
    noise = general_noise().with_readout("MZ", {"0": {"1": 1}})
    assert sim(program()).noise(noise).run(1).to_dict()["c"] == [1]
