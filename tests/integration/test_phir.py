# Copyright 2023 The PECOS developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import json
from pathlib import Path

import pytest
from pecos.engines.hybrid_engine import HybridEngine
from pecos.error_models.generic_error_model import GenericErrorModel

try:
    from pecos.foreign_objects.wasmtime import WasmtimeObj
except ImportError:
    WasmtimeObj = None

try:
    from pecos.foreign_objects.wasmer import WasmerObj

    WASMER_ERR_MSG = None
except ImportError as e:
    WasmerObj = None
    WASMER_ERR_MSG = str(e)

# tools for converting wasm to wat: https://github.com/WebAssembly/wabt/releases/tag/1.0.33

this_dir = Path(__file__).parent

add_wat = this_dir / "wat/add.wat"
math_wat = this_dir / "wat/math.wat"
example1_phir = json.load(Path.open(this_dir / "phir/example1.json"))
example1_no_wasm_phir = json.load(Path.open(this_dir / "phir/example1_no_wasm.json"))
spec_example_phir = json.load(Path.open(this_dir / "phir/spec_example.json"))


# Select which marked tests to run by using the mark flag. See: https://docs.pytest.org/en/7.1.x/example/markers.html
# run only optional_dependency tests: pytest -v -m optional_dependency
# run all without optional_dependency tests: pytest -v -m "not optional_dependency"


def is_wasmer_supported():
    """A check on whether Wasmer is known to support OS/Python versions."""

    if WASMER_ERR_MSG == "Wasmer is not available on this system":
        return False

    return True


@pytest.mark.wasmtime()
@pytest.mark.optional_dependency()
def test_spec_example_wasmtime():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS."""

    wasm = WasmtimeObj(math_wat)
    HybridEngine().run(
        program=spec_example_phir,
        foreign_object=wasm,
        shots=1000,
    )


@pytest.mark.wasmtime()
@pytest.mark.optional_dependency()
def test_spec_example_noisy_wasmtime():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, with noise."""

    wasm = WasmtimeObj(str(math_wat))
    generic_errors = GenericErrorModel(
        error_params={
            "p1": 2e-1,
            "p2": 2e-1,
            "p_meas": 2e-1,
            "p_init": 1e-1,
            "p1_error_model": {
                "X": 0.25,
                "Y": 0.25,
                "Z": 0.25,
                "L": 0.25,
            },
        },
    )
    sim = HybridEngine(error_model=generic_errors)
    sim.run(
        program=spec_example_phir,
        foreign_object=wasm,
        shots=1000,
    )


@pytest.mark.wasmtime()
@pytest.mark.optional_dependency()
def test_example1_wasmtime():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS."""

    wasm = WasmtimeObj(add_wat)
    HybridEngine().run(
        program=example1_phir,
        foreign_object=wasm,
        shots=1000,
    )


@pytest.mark.wasmtime()
@pytest.mark.optional_dependency()
def test_example1_noisy_wasmtime():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, with noise."""

    wasm = WasmtimeObj(str(add_wat))
    generic_errors = GenericErrorModel(
        error_params={
            "p1": 2e-1,
            "p2": 2e-1,
            "p_meas": 2e-1,
            "p_init": 1e-1,
            "p1_error_model": {
                "X": 0.25,
                "Y": 0.25,
                "Z": 0.25,
                "L": 0.25,
            },
        },
    )
    sim = HybridEngine(error_model=generic_errors)
    sim.run(
        program=example1_phir,
        foreign_object=wasm,
        shots=1000,
    )


@pytest.mark.skipif(not is_wasmer_supported(), reason="Wasmer is not support on some OS/Python version combinations.")
@pytest.mark.wasmer()
@pytest.mark.optional_dependency()
def test_example1_wasmer():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS."""

    wasm = WasmerObj(add_wat)
    HybridEngine().run(
        program=example1_phir,
        foreign_object=wasm,
        shots=1000,
    )


@pytest.mark.skipif(not is_wasmer_supported(), reason="Wasmer is not support on some OS/Python version combinations.")
@pytest.mark.wasmer()
@pytest.mark.optional_dependency()
def test_example1_noisy_wasmer():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, with noise."""

    wasm = WasmerObj(str(add_wat))
    generic_errors = GenericErrorModel(
        error_params={
            "p1": 2e-1,
            "p2": 2e-1,
            "p_meas": 2e-1,
            "p_init": 1e-1,
            "p1_error_model": {
                "X": 0.25,
                "Y": 0.25,
                "Z": 0.25,
                "L": 0.25,
            },
        },
    )
    sim = HybridEngine(error_model=generic_errors)
    sim.run(
        program=example1_phir,
        foreign_object=wasm,
        shots=1000,
    )


def test_example1_no_wasm():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, without Wasm."""

    HybridEngine().run(program=example1_no_wasm_phir, shots=1000)


def test_example1_no_wasm_multisim():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, without Wasm."""

    HybridEngine().run_multisim(program=example1_no_wasm_phir, shots=1000, pool_size=2)


def test_example1_no_wasm_noisy():
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, without Wasm but with noise."""

    generic_errors = GenericErrorModel(
        error_params={
            "p1": 2e-1,
            "p2": 2e-1,
            "p_meas": 2e-1,
            "p_init": 1e-1,
            "p1_error_model": {
                "X": 0.25,
                "Y": 0.25,
                "Z": 0.25,
                "L": 0.25,
            },
        },
    )
    sim = HybridEngine(error_model=generic_errors)
    sim.run(
        program=example1_no_wasm_phir,
        shots=1000,
    )
