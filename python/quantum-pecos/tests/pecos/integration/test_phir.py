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

"""Integration tests for PHIR quantum program execution."""
import json
from pathlib import Path

import pytest
from pecos import WasmForeignObject
from pecos.classical_interpreters.phir_classical_interpreter import (
    PhirClassicalInterpreter,
)
from pecos.engines.hybrid_engine import HybridEngine
from pecos.error_models.generic_error_model import GenericErrorModel
from phir.model import PHIRModel
from pydantic import ValidationError

# tools for converting wasm to wat: https://github.com/WebAssembly/wabt/releases/tag/1.0.33

this_dir = Path(__file__).parent

add_wat = this_dir / "wat/add.wat"
math_wat = this_dir / "wat/math.wat"
example1_phir = json.load(Path.open(this_dir / "phir/example1.phir.json"))
example1_no_wasm_phir = json.load(
    Path.open(this_dir / "phir/example1_no_wasm.phir.json"),
)
spec_example_phir = json.load(Path.open(this_dir / "phir/spec_example.phir.json"))


# Select which marked tests to run by using the mark flag. See: https://docs.pytest.org/en/7.1.x/example/markers.html
# run only optional_dependency tests: pytest -v -m optional_dependency
# run all without optional_dependency tests: pytest -v -m "not optional_dependency"


def test_spec_example_wasmtime() -> None:
    """A random example showing that various basic aspects of PHIR is runnable by PECOS."""
    wasm = WasmForeignObject(math_wat)
    HybridEngine().run(
        program=spec_example_phir,
        foreign_object=wasm,
        shots=1000,
    )


def test_spec_example_noisy_wasmtime() -> None:
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, with noise."""
    wasm = WasmForeignObject(str(math_wat))
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


def test_example1_wasmtime() -> None:
    """A random example showing that various basic aspects of PHIR is runnable by PECOS."""
    wasm = WasmForeignObject(add_wat)
    HybridEngine().run(
        program=example1_phir,
        foreign_object=wasm,
        shots=1000,
    )


def test_example1_noisy_wasmtime() -> None:
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, with noise."""
    wasm = WasmForeignObject(str(add_wat))
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


def test_example1_no_wasm() -> None:
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, without Wasm."""
    HybridEngine().run(program=example1_no_wasm_phir, shots=1000)


def test_example1_no_wasm_multisim() -> None:
    """A random example showing that various basic aspects of PHIR is runnable by PECOS, without Wasm."""
    HybridEngine().run_multisim(program=example1_no_wasm_phir, shots=1000, pool_size=2)


def test_example1_no_wasm_noisy() -> None:
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


def test_record_random_bit() -> None:
    """Applying H and recording both 0 and 1."""
    results = HybridEngine(qsim="stabilizer").run(
        program=json.load(
            Path.open(this_dir / "phir" / "recording_random_meas.phir.json"),
        ),
        shots=100,
    )

    # print(results)
    results_dict = results
    c = results_dict["c"]
    assert c.count("01") + c.count("00") == len(c)


def test_classical_if_00_11() -> None:
    """Testing using an H + measurement and a conditional X gate to get 00 or 11."""
    results = HybridEngine(qsim="stabilizer").run(
        program=json.load(Path.open(this_dir / "phir" / "classical_00_11.phir.json")),
        shots=100,
    )

    results_dict = results
    c = results_dict["c"]
    assert c.count("00") + c.count("11") == len(c)


def test_throw_exception_with_bad_phir() -> None:
    """Making sure the bad PHIR throws an exception."""
    phir = json.load(Path.open(this_dir / "phir" / "bad_phir.phir.json"))
    with pytest.raises(ValidationError):
        PHIRModel.model_validate(phir)


def test_qparallel() -> None:
    """Testing the qparallel block of 2 Xs and 2 Ys gives an output of 1111."""
    results = HybridEngine(qsim="stabilizer").run(
        program=json.load(Path.open(this_dir / "phir" / "qparallel.phir.json")),
        shots=10,
    )

    results_dict = results
    m = results_dict["m"]
    assert m.count("1111") == len(m)


def test_bell_qparallel() -> None:
    """Testing a program creating and measuring a Bell state and using qparallel blocks returns expected results."""
    results = HybridEngine(qsim="state-vector").run(
        program=json.load(Path.open(this_dir / "phir" / "bell_qparallel.phir.json")),
        shots=20,
    )

    # Check either "c" (if Result command worked) or "m" (fallback)
    results_dict = results
    register = "c" if "c" in results_dict else "m"
    result_values = results_dict[register]
    assert result_values.count("00") + result_values.count("11") == len(result_values)


def test_bell_qparallel_cliff() -> None:
    """Test Bell state creation and measurement with qparallel blocks.

    Tests that a program creating and measuring a Bell state using qparallel blocks returns expected results
    with Clifford circuits and stabilizer simulator.
    """
    # Create an interpreter with validation disabled for testing Result instruction
    interp = PhirClassicalInterpreter()
    interp.phir_validate = False

    results = HybridEngine(qsim="stabilizer", cinterp=interp).run(
        program=json.load(
            Path.open(this_dir / "phir" / "bell_qparallel_cliff.phir.json"),
        ),
        shots=20,
    )

    # Check either "c" (if Result command worked) or "m" (fallback)
    results_dict = results
    register = "c" if "c" in results_dict else "m"
    result_values = results_dict[register]
    assert result_values.count("00") + result_values.count("11") == len(result_values)


def test_bell_qparallel_cliff_barrier() -> None:
    """Test Bell state creation and measurement with qparallel blocks and barriers.

    Tests that a program creating and measuring a Bell state using qparallel blocks and barriers returns expected
    results with Clifford circuits and stabilizer simulator.
    """
    interp = PhirClassicalInterpreter()
    interp.phir_validate = False

    results = HybridEngine(qsim="stabilizer", cinterp=interp).run(
        program=json.load(
            Path.open(this_dir / "phir" / "bell_qparallel_cliff_barrier.phir.json"),
        ),
        shots=20,
    )

    # Check either "c" (if Result command worked) or "m" (fallback)
    results_dict = results
    register = "c" if "c" in results_dict else "m"
    result_values = results_dict[register]
    assert result_values.count("00") + result_values.count("11") == len(result_values)


def test_bell_qparallel_cliff_ifbarrier() -> None:
    """Test Bell state creation and measurement with qparallel blocks and conditional barriers.

    Tests that a program creating and measuring a Bell state using qparallel blocks and conditional barriers
    returns expected results with Clifford circuits and stabilizer simulator.
    """
    interp = PhirClassicalInterpreter()
    interp.phir_validate = False

    results = HybridEngine(qsim="stabilizer", cinterp=interp).run(
        program=json.load(
            Path.open(this_dir / "phir" / "bell_qparallel_cliff_ifbarrier.phir.json"),
        ),
        shots=20,
    )

    # Check either "c" (if Result command worked) or "m" (fallback)
    results_dict = results
    register = "c" if "c" in results_dict else "m"
    result_values = results_dict[register]
    assert result_values.count("00") + result_values.count("11") == len(result_values)
