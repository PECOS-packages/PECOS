# Copyright 2023 The PECOS Developers
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
from copy import deepcopy

import pecos

# TODO: Does this handle parallel gate ticks?

qsym_conv = {
    "init |0>": "Init",
    "measure Z": "Measure",
    "SqrtX": "SX",
    "SqrtXd": "SXdg",
    "SqrtY": "SY",
    "SqrtYd": "SYdg",
    "SqrtZ": "SZ",
    "SqrtZd": "SZdg",
    "SqrtZZ": "SZZ",
}


def conv_expr(expr):
    if "t" in expr:
        op = "="

        if expr["op"] == "=":
            # op = = -> "t = a"
            assert "b" not in expr  # noqa: S101
            left = expr["t"]
            right = expr["a"]
        else:
            # op != = -> "t = a op b"
            left = expr["t"]
            right = {"op": expr["op"], "a": expr["a"], "b": expr["b"]}

    else:
        op = expr["op"]
        left = expr["a"]
        right = expr.get("b")

    if isinstance(left, dict):
        left = conv_expr(left)

    if isinstance(right, dict):
        right = conv_expr(right)

    new_expr = {
        "cop": op,
    }

    if isinstance(left, tuple):
        left = list(left)

    if isinstance(right, tuple):
        right = list(right)

    if op == "=":
        new_expr["args"] = [right]
        new_expr["returns"] = [left]
    elif right is not None:
        new_expr["args"] = [left, right]
    else:
        new_expr["args"] = [left]

    return new_expr


def to_phir_dict(qc: "pecos.QuantumCircuit") -> dict:
    """Creates a json str representation of the QuantumCircuit listing all the gates. It does not preserve ticks or
    parallel gating of different gate types.
    """
    prog = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": deepcopy(qc.metadata),
        "ops": [],
    }
    prog["metadata"]["source_program_type"] = [
        "PECOS.QuantumCircuit",
        ["PECOS", str(pecos.__version__)],
    ]

    data = {"max_qid": 0}
    if "qvars" in qc.metadata:
        qid2qsym = {k: list(v) for k, v in prog["metadata"]["qvars"].items()}
        del prog["metadata"]["qvars"]
        prog["metadata"]["qid2qubit_sym"] = qid2qsym
    else:
        qid2qsym = {}

    def get_qsym(qid: int) -> list[str, int]:
        qsym = qid2qsym.get(qid)

        if qsym is None:
            if "qvar_spec" not in qc.metadata:
                qsym = ["q", data["max_qid"]]
                data["max_qid"] += 1
                qid2qsym[qid] = qsym
            else:
                msg = f"Couldn't find qsym for qid = {qid}"
                raise Exception(msg)
        return list(qsym)

    def find_qid2qsym(qubits):
        qs = []
        for loc in qubits:
            if isinstance(loc, (tuple, list)):
                qtup = []
                for q in loc:
                    qsym = get_qsym(q)

                    qtup.append(qsym)
                qs.append(qtup)
            else:
                qsym = get_qsym(loc)
                qs.append(qsym)
        return qs

    def flush_ops_buffer(ops, ops_buffer, current_cond) -> None:
        if current_cond is None:
            ops.extend(ops_buffer)
        else:
            block = {
                "block": "if",
                "condition": current_cond,
                "true_branch": ops_buffer,
            }
            ops.append(block)

    def handle_ops_buffer(
        current_op,
        ops,
        ops_buffer,
        cond,
        current_cond,
        force_flush=False,
    ):
        if (cond != current_cond or force_flush) and ops_buffer:
            flush_ops_buffer(ops, ops_buffer, current_cond)
            ops_buffer = []
            current_cond = cond

        if current_op:
            ops_buffer = ops_buffer_append(ops_buffer, current_op)

        return ops, ops_buffer, current_cond

    def ops_buffer_append(ops_buffer, current_op, compact_gates=True):
        # TODO: When you do that make sure gates don't overlap

        if ops_buffer and compact_gates:
            last_op = ops_buffer[-1]
            if (
                (
                    "qop" in last_op
                    and "qop" in current_op
                    and last_op["qop"] == current_op["qop"]
                )
                or (last_op.get("data") == last_op.get("data") == "cvar_export")
            ) and last_op.get("metadata") == current_op.get("metadata"):
                if "args" in current_op:
                    # TODO: Check no overlap!
                    last_op["args"].extend(current_op["args"])

                if "variables" in current_op:
                    last_op["variables"].extend(current_op["variables"])

                if "returns" in current_op:
                    last_op["returns"].extend(current_op["returns"])
            else:
                ops_buffer.append(current_op)

        else:
            ops_buffer.append(current_op)

        return ops_buffer

    ops = prog["ops"]

    for sym, size in qc.metadata.get("qvar_spec", {}).items():
        ops.append(
            {
                "data": "qvar_define",
                "data_type": "qubits",
                "variable": sym,
                "size": size,
            },
        )

    for sym, size in qc.metadata.get("cvar_spec", {}).items():
        ops.append(
            {
                "data": "cvar_define",
                "data_type": "i32",
                "variable": sym,
                "size": size,
            },
        )

    ops_buffer = []
    current_cond = None

    for sym, qubits, meta in qc.items():
        op = {}
        metadata = deepcopy(meta) if meta else {}

        if sym == "cop":
            # TODO: HOW to identify mops...., need to tag at higher level
            if metadata.get("cop_type") == "CFunc":
                op = {
                    "cop": "ffcall",
                    "function": metadata["func"],
                    "args": metadata["args"],
                }

                if metadata["assign_vars"]:
                    op["returns"] = metadata["assign_vars"]

            elif metadata.get("cop_type") == "ExportCVar":
                del metadata["cop_type"]
                if "duration" in metadata:
                    del metadata["duration"]

                op = {
                    "data": "cvar_export",
                    "variables": [metadata["export"]],
                }
                del metadata["export"]

            elif metadata.get("cop_type") in ["Idle", "idle"]:
                del metadata["cop_type"]
                op = {
                    "mop": "Idle",
                    "args": find_qid2qsym(qubits),
                }

            elif metadata.get("cop_type") in ["Transport", "transport"]:
                del metadata["cop_type"]
                op = {
                    "mop": "Transport",
                    "args": find_qid2qsym(qubits),
                }

            elif "expr" in metadata:
                op = conv_expr(metadata["expr"])
                del metadata["expr"]

            else:
                print("!!!!!", qubits, metadata)

        else:  # qop
            op.update(
                {
                    "qop": qsym_conv.get(sym, sym),
                    "args": find_qid2qsym(qubits),
                },
            )

            angles = None
            if "angle" in metadata:
                angles = [metadata["angle"]]
            elif "angles" in metadata:
                angles = metadata["angles"]

            if angles:
                op["angles"] = angles

            if sym.startswith("measure"):
                # Getting return values:
                if "var" in metadata:
                    var = metadata["var"]
                    op["returns"] = [list(var)]
                    del metadata["var"]
                elif "var_output" in metadata:
                    var_output = metadata["var_output"]
                    var_return = []
                    for q in qubits:
                        var = var_output[q]
                        if isinstance(var, str):
                            var_return.append(var)
                        else:
                            var_return.append(list(var))
                    op["returns"] = var_return
                    del metadata["var_output"]

        cond = metadata.get("cond")
        if cond is not None:
            cond = conv_expr(cond)

        if "cond" in metadata:
            del metadata["cond"]

        if metadata:
            op["metadata"] = metadata

        ops, ops_buffer, current_cond = handle_ops_buffer(
            op,
            ops,
            ops_buffer,
            cond,
            current_cond,
        )

    # Flush the buffer of any remaining operations
    if ops_buffer:
        ops, ops_buffer, current_cond = handle_ops_buffer(
            None,
            ops,
            ops_buffer,
            None,
            current_cond,
            force_flush=True,
        )

    num_qubits = len(qid2qsym)
    prog["metadata"]["num_qubits"] = num_qubits

    if "qvar_spec" not in qc.metadata:
        op = {
            "qop": "qvar_define",
            "data_type": "qubits",
            "variable": "q",
            "size": num_qubits,
        }

        ops.insert(0, op)

    if "cvar_spec" in prog["metadata"]:
        del prog["metadata"]["cvar_spec"]

    if "qvar_spec" in prog["metadata"]:
        del prog["metadata"]["qvar_spec"]

    return prog


def to_phir_json(qc: "pecos.QuantumCircuit"):
    """Convert the QuantumCircuit to the PHIR/JSON format."""
    return json.dumps(to_phir_dict(qc))
