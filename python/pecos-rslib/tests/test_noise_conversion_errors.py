"""Invalid depolarizing builders report ValueError at simulation construction."""

import json
from functools import partial

import pytest
from pecos_rslib import (
    Qasm,
    biased_depolarizing_noise,
    depolarizing_noise,
    qasm_engine,
    run_phir_sim,
    sim,
)


@pytest.mark.parametrize("factory", [depolarizing_noise, biased_depolarizing_noise])
@pytest.mark.parametrize("probability", [1.5, -0.1, float("nan"), float("inf"), None])
@pytest.mark.parametrize("route", ["facade", "engine", "phir"])
def test_invalid_noise_builder_raises_value_error(factory, probability, route) -> None:
    noise = factory()
    if probability is not None:
        noise = noise.with_uniform_probability(probability)
    program = Qasm.from_string("OPENQASM 2.0; qreg q[1];")
    simulation = sim(program) if route == "facade" else qasm_engine().program(program).to_sim()
    message = (
        "Preparation probability must be set" if probability is None else r"Probability must be between 0\.0 and 1\.0"
    )
    if route == "phir":
        run = partial(
            run_phir_sim,
            json.dumps({"format": "PHIR/JSON", "version": "0.1.0", "ops": []}),
            noise_model=noise,
        )
    else:
        run = partial(simulation.noise(noise).run, 1)
    with pytest.raises(ValueError, match=message):
        run()


class GenericErrorModel:
    def __init__(self, probability: float) -> None:
        self.error_params = {"p1": probability}


@pytest.mark.parametrize("probability", [1.5, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("generic", [False, True])
def test_phir_noise_shortcuts_raise_value_error(probability: float, generic: bool) -> None:
    with pytest.raises(ValueError, match=r"Probability must be between 0\.0 and 1\.0"):
        run_phir_sim(
            json.dumps({"format": "PHIR/JSON", "version": "0.1.0", "ops": []}),
            noise_model=GenericErrorModel(probability) if generic else probability,
        )
