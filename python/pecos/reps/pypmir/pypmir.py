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


from __future__ import annotations

from math import pi
from typing import TypeVar

from pecos.reps.pypmir import block_types as blk
from pecos.reps.pypmir import data_types as d
from pecos.reps.pypmir import op_types as op
from pecos.reps.pypmir.name_resolver import default_sim_name_resolver

TypeOp = TypeVar("TypeOp", bound=op.Op)


class PyPMIR:
    """Pythonic PECOS Middle-level IR. Used to convert PHIR into an object and optimize the data structure for
    simulations.
    """

    def __init__(self, metadata: dict | None = None, sim_name_resolver=None) -> None:
        self.ops = []
        self.metadata = metadata

        if sim_name_resolver is None:
            self.sim_name_resolver = default_sim_name_resolver
        else:
            self.sim_name_resolver = sim_name_resolver

        self.cvar_meta = []
        self.cvar_dtype_list = []
        self.csym2id = {}
        self.cvar_dtypes_set = set()
        self.qvar_meta = {}
        self.num_qubits = 0
        self.foreign_func_calls = set()

    @classmethod
    def handle_op(cls, o: dict, p: PyPMIR) -> TypeOp:
        if "block" in o:
            if o["block"] == "sequence":
                ops = []
                for so in o["ops"]:
                    ops.append(cls.handle_op(so, p))

                instr = blk.SeqBlock(
                    ops=ops,
                    metadata=o.get("metadata"),
                )
            elif o["block"] == "qparallel":
                ops = []
                for so in o["ops"]:
                    ops.append(cls.handle_op(so, p))

                instr = blk.QParallelBlock(
                    ops=ops,
                    metadata=o.get("metadata"),
                )
            elif o["block"] == "if":
                true_branch = []
                for so in o["true_branch"]:
                    true_branch.append(cls.handle_op(so, p))

                false_branch = []
                for so in o.get("false_branch", []):
                    false_branch.append(cls.handle_op(so, p))

                instr = blk.IfBlock(
                    condition=o["condition"],
                    true_branch=true_branch,
                    false_branch=false_branch,
                    metadata=o.get("metadata"),
                )
            else:
                msg = f"Block not recognized: {o}"
                raise Exception(msg)

        elif "qop" in o:
            # TODO: convert [qsym, qubit_init] to int
            # TODO: flatten to just list of ints even for TQ, etc.
            # TODO: Note size of gate?

            args = []
            for a in o["args"]:
                if isinstance(a[0], list):
                    tup = []
                    for b in a:
                        qsym, qid = b
                        qdata = p.qvar_meta[qsym]
                        tup.append(qdata.qubit_ids[qid])
                    args.append(tup)
                else:
                    qsym, qid = a
                    qdata = p.qvar_meta[qsym]
                    args.append(qdata.qubit_ids[qid])

            metadata = {} if o.get("metadata") is None else o["metadata"]

            if o.get("angles"):
                angles = tuple(
                    [angle * (pi if o["angles"][1] == "pi" else 1) for angle in o["angles"][0]],
                )
            else:
                angles = None

            # TODO: get rid of supplying angle or angles in syms and move to (sym, angles) or sym (or gate obj)
            if angles:
                if len(angles) == 1:
                    metadata["angle"] = angles[0]
                else:
                    metadata["angles"] = angles

            # TODO: Added to satisfy old-style error models. Remove when they not longer need this...
            if o.get("returns"):
                var_output = {}
                for q, cvar in zip(args, o["returns"], strict=False):
                    var_output[q] = cvar
                metadata["var_output"] = var_output

            instr = op.QOp(
                name=o["qop"],
                sim_name=None,
                angles=angles,
                args=args,
                returns=o.get("returns"),
                metadata=metadata,
            )

            instr.sim_name = p.sim_name_resolver(instr)

        elif "cop" in o:
            if o["cop"] == "ffcall":
                instr = op.FFCall(
                    name=o["function"],
                    args=o["args"],
                    returns=o.get("returns"),
                    metadata=o.get("metadata"),
                )
                p.foreign_func_calls.add(o["function"])
            else:
                instr = op.COp(
                    name=o["cop"],
                    args=o["args"],
                    returns=o.get("returns"),
                    metadata=o.get("metadata"),
                )

        elif "mop" in o:
            instr = op.MOp(
                name=o["mop"],
                args=o.get("args"),
                returns=o.get("returns"),
                metadata=o.get("metadata"),
            )

        elif "meta" in o:
            # TODO: Handle meta instructions
            name = o["meta"]
            if name == "barrier":
                instr = None
            else:
                msg = f"Meta instruction '{name}' not implemented/supported."
                raise NotImplementedError(msg)

        elif "//" in o:
            # Do not include comments
            instr = None

        else:
            msg = f"Instruction not recognized: {o}"
            raise Exception(msg)

        return instr

    @classmethod
    def from_phir(cls, phir: dict, sim_name_resolver=None) -> PyPMIR:
        """Takes a PHIR dictionary and converts it into a PyPMIR object."""
        p = PyPMIR(
            metadata=dict(
                phir.get("metadata", {}),
            ),
            sim_name_resolver=sim_name_resolver,
        )

        next_qvar_int = 0

        for o in phir["ops"]:
            if "data" in o:
                name = o["data"]

                if name == "cvar_define":
                    data = d.CVarDefine(
                        data_type=o["data_type"],
                        variable=o["variable"],
                        size=o["size"],
                        cvar_id=len(p.cvar_meta),
                        metadata=o.get("metadata"),
                    )

                    p.cvar_meta.append(data)
                    p.csym2id[data.variable] = data.cvar_id
                    p.cvar_dtypes_set.add(data.data_type)
                    p.cvar_dtype_list.append(data.data_type)

                if name == "qvar_define":
                    if o["data_type"] != "qubits":
                        msg = f"Do not know handle qvar type: {o['data_type']}"
                        raise Exception(msg)

                    qubit_ids = []
                    for _i in range(o["size"]):
                        qubit_ids.append(next_qvar_int)
                        next_qvar_int += 1

                    data = d.QVarDefine(
                        data_type=o["data_type"],
                        variable=o["variable"],
                        size=o["size"],
                        qubit_ids=qubit_ids,
                        metadata=o.get("metadata"),
                    )

                    p.qvar_meta[data.variable] = data
                    p.num_qubits += data.size
            else:
                instr = cls.handle_op(o, p)
                if instr:
                    p.ops.append(instr)

        return p
