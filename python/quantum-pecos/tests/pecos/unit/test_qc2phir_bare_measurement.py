# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Bare measurements reach PHIR through a hidden scratch variable (#1497)."""

import json

import pecos as pc
import pytest
from pecos.engines.hybrid_engine import HybridEngine
from pecos.slr import CReg, Main, QReg, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.typing import PhirModel
from pecos_rslib import PhirJson, sim


def _run(program: str, runner: str) -> dict:
    if runner == "native":
        return {
            key: list(map(int, values)) for key, values in sim(PhirJson.from_string(program)).run(3).to_dict().items()
        }
    engine = HybridEngine(cinterp="python") if runner == "python" else HybridEngine()
    return {key: [int(value, 2) for value in values] for key, values in engine.run(program, shots=3).items()}


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_slr_bare_measurement_runs_and_reports_only_user_registers(runner) -> None:
    q, c = QReg("q", 2), CReg("c", 1)
    program = Main(q, c, qb.X(q[0]), qb.X(q[1]), qb.Measure(q[0]) > c[0], qb.Measure(q[1]))
    phir = SlrConverter(program).quantum_circuit().to_phir_json()
    PhirModel.model_validate(json.loads(phir))
    assert _run(phir, runner) == {"c": [1, 1, 1]}


def test_scratch_name_avoids_existing_names() -> None:
    qc = pc.QuantumCircuit(
        cvar_spec={"__pecos_scratch": 1},
        qvar_spec={"__pecos_scratch_": 1},
        qvars={0: ("__pecos_scratch_", 0)},
    )
    qc.append("measure Z", {0})
    ops = json.loads(qc.to_phir_json())["ops"]
    (measure,) = [op for op in ops if op.get("qop") == "Measure"]
    assert measure["returns"] == [["__pecos_scratch__", 0]]
    assert ops[-1] == {"data": "cvar_export", "variables": ["__pecos_scratch"]}


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_more_than_64_bare_measurements_use_several_hidden_scratch_variables(runner) -> None:
    qc = pc.QuantumCircuit(cvar_spec={"m": 1})
    qc.append("X", {0})
    qc.append("measure Z", set(range(70)))
    qc.append("measure Z", {0}, var=("m", 0))
    phir = qc.to_phir_json()
    PhirModel.model_validate(json.loads(phir))
    scratch = [op for op in json.loads(phir)["ops"] if op.get("data") == "cvar_define" and op["variable"] != "m"]
    assert [op["size"] for op in scratch] == [64, 6]
    assert _run(phir, runner) == {"m": [1, 1, 1]}


def test_existing_exports_are_kept_and_not_extended() -> None:
    qc = pc.QuantumCircuit(cvar_spec={"a": 1, "b": 1})
    qc.append("measure Z", {0})
    qc.append("measure Z", {1}, var=("a", 0))
    qc.append("cop", {}, cop_type="ExportCVar", export="a")
    exports = [op for op in json.loads(qc.to_phir_json())["ops"] if op.get("data") == "cvar_export"]
    assert exports == [{"data": "cvar_export", "variables": ["a"]}]


def test_programs_without_bare_measurements_are_unchanged() -> None:
    qc = pc.QuantumCircuit(cvar_spec={"m": 1})
    qc.append("measure Z", {0}, var=("m", 0))
    ops = json.loads(qc.to_phir_json())["ops"]
    assert not any(op.get("data") == "cvar_export" for op in ops)
    assert not any("__pecos_scratch" in json.dumps(op) for op in ops)


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_export_inside_a_conditional_counts_as_an_existing_export(runner) -> None:
    qc = pc.QuantumCircuit(cvar_spec={"c": 1, "d": 1})
    qc.append("X", {0})
    qc.append("measure Z", {0}, var=("c", 0))
    qc.append("measure Z", {1})
    qc.append("cop", {}, cop_type="ExportCVar", export="d", cond={"a": ("c", 0), "op": "==", "b": 1})
    assert _run(qc.to_phir_json(), runner) == {"d": [0, 0, 0]}


@pytest.mark.parametrize("runner", ["default", "python", "native"])
def test_size_zero_registers_are_not_exported(runner) -> None:
    qc = pc.QuantumCircuit(cvar_spec={"z": 0, "c": 1})
    qc.append("X", {0})
    qc.append("measure Z", {0}, var=("c", 0))
    qc.append("measure Z", {1})
    assert _run(qc.to_phir_json(), runner) == {"c": [1, 1, 1]}


def test_scratch_name_avoids_measurement_destinations() -> None:
    # A destination register outside cvar_spec must not be merged with the scratch.
    qc = pc.QuantumCircuit(cvar_spec={"c": 1})
    qc.append("X", {0})
    qc.append("measure Z", {0}, var=("__pecos_scratch", 0))
    qc.append("measure Z", {1})
    returns = [
        ret for op in json.loads(qc.to_phir_json())["ops"] if op.get("qop") == "Measure" for ret in op["returns"]
    ]
    assert ["__pecos_scratch", 0] in returns
    assert ["__pecos_scratch_", 0] in returns


def test_native_bare_measurement_no_longer_writes_a_user_register_m() -> None:
    # The native engine used to accept a returnless Measure and write its result into a register named m.
    qc = pc.QuantumCircuit(cvar_spec={"m": 2})
    qc.append("X", {0})
    qc.append("measure Z", {0})
    qc.append("measure Z", {1}, var=("m", 1))
    assert _run(qc.to_phir_json(), "native") == {"m": [0, 0, 0]}
