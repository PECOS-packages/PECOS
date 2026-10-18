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

"""The program-wide declaration namespace specified by PHIR-JSON v0.1."""

import json
import math
import reprlib

from pecos_rslib import validate_phir_classical_inputs


def validate_declaration_names(operations: list[dict]) -> None:
    """Reject a name declared more than once, regardless of declaration kind."""
    declarations = {}

    def register_names(ops: list[dict]) -> None:
        if not isinstance(ops, list):
            return
        for node in ops:
            if not isinstance(node, dict):
                continue
            kind = {"qvar_define": "quantum", "cvar_define": "classical"}.get(node.get("data"))
            if kind is not None:
                variable = node["variable"]
                if variable in declarations:
                    msg = (
                        f"Variable '{variable}' is already declared as {declarations[variable]}; "
                        f"cannot redeclare as {kind}"
                    )
                    raise ValueError(msg)
                declarations[variable] = kind
            if "block" in node:
                for field in ("ops", "true_branch", "false_branch"):
                    register_names(node.get(field) or [])

    register_names(operations)


def _json_declaration_value(value: object) -> object:
    """Keep native finite JSON scalars; mark other values for named Rust rejection."""
    if value is None or type(value) in (str, bool):
        return value
    if type(value) is int and -(2**63) <= value < 2**64:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return {"python": reprlib.repr(value)}


def validate_classical_inputs(program: dict) -> None:
    """Validate declaration and Result fields without serializing quantum ops or metadata.

    Metadata need not be JSON-serializable. Only declaration and Result fields
    are relevant here; structural errors belong to the schema/parser.
    """
    inputs = []

    def collect(operations: list[dict]) -> None:
        if not isinstance(operations, list):
            return
        for node in operations:
            if not isinstance(node, dict):
                continue
            if node.get("cop") == "Result":
                inputs.append({key: node[key] for key in ("cop", "args", "returns") if key in node})
            elif node.get("data") in ("cvar_define", "qvar_define"):
                declaration = {key: node[key] for key in ("data", "data_type", "variable", "size") if key in node}
                for field in ("data_type", "variable", "size"):
                    if field in declaration:
                        declaration[field] = _json_declaration_value(declaration[field])
                inputs.append(declaration)
            if "block" in node:
                for field in ("ops", "true_branch", "false_branch"):
                    collect(node.get(field) or [])

    collect(program.get("ops", []))
    validate_phir_classical_inputs(json.dumps({"ops": inputs}))
