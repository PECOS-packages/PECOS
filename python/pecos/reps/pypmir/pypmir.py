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

from typing import Optional

from . import block_types as blk
from . import data_types as d
from . import op_types as op


class PyPMIR:
    """Pythonic PECOS Middle-level IR. Used to convert PHIR into an object and optimize the data structure for
    simulations."""
    def __init__(self, metadata: Optional[dict] = None):

        self.ops = []
        self.metadata = metadata

        self.cvar_meta = []
        self.cvar_dtype_list = []
        self.csym2id = {}
        self.cvar_dtypes_set = set()
        self.qvar_meta = {}
        self.num_qubits = 0

    @classmethod
    def handle_op(cls, o, p):
        if "block" in o:
            if o["block"] == "sequence":

                ops = []
                for so in o["ops"]:
                    ops.append(cls.handle_op(so, p))

                instr = blk.SeqBlock(
                    ops=ops,
                    metadata=o.get("metadata"),
                )
            elif o["block"] == "if":
                """
                condition = o["condition"]
                if "cop" in condition:
                    # condition = cls.handle_op(condition, p)
                    pass
                else:
                    raise Exception(f"Condition is of unexpected form: {condition}")
                """

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
                raise Exception(f"Block not recognized: {o}")

        elif "qop" in o:
            # TODO: convert [qsym, qubit_init] to int
            # TODO: flatten to just list of ints even for TQ, etc.
            # TODO: Note size of gate?

            """
            data = d.QVarDefine(data_type=o["data_type"],
                                        variable=o["variable"],
                                        size=o["size"],
                                        qubit_ids=qubit_ids,
                                        metadata=o.get("metadata"), )

                    p.qvar_meta[data.variable] = data
            """

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

            if o.get("metadata") is None:
                metadata = {}
            else:
                metadata = o["metadata"]

            # TODO: Added to satisfy old-style error models. Remove when they not longer need this...
            if o.get("returns"):
                var_output = {}
                for q, cvar in zip(args, o["returns"]):
                    var_output[q] = cvar
                metadata["var_output"] = var_output

            instr = op.QOp(name=o["qop"],
                           args=args,
                           returns=o.get("returns"),
                           metadata=metadata, )

        elif "cop" in o:

            if o["cop"] == "ffcall":
                instr = op.FFCall(name=o["function"],
                                  args=o["args"],
                                  returns=o.get("returns"),
                                  metadata=o.get("metadata"), )
            else:
                instr = op.COp(name=o["cop"],
                               args=o["args"],
                               returns=o.get("returns"),
                               metadata=o.get("metadata"), )

        elif "mop" in o:
            instr = op.MOp(name=o["mop"],
                           args=o.get("args"),
                           returns=o.get("returns"),
                           metadata=o.get("metadata"), )

        else:
            raise Exception(f"Instruction not recognized: {o}")

        return instr

    @classmethod
    def from_phir(cls, phir: dict) -> "PyPMIR":

        # TODO: Build nested instructions.

        p = PyPMIR(metadata=dict(phir["metadata"]))

        next_qvar_int = 0

        for o in phir["ops"]:

            if "data" in o:
                name = o["data"]

                if name == "cvar_define":
                    data = d.CVarDefine(data_type=o["data_type"],
                                        variable=o["variable"],
                                        size=o["size"],
                                        cvar_id=len(p.cvar_meta),
                                        metadata=o.get("metadata"), )

                    p.cvar_meta.append(data)
                    p.csym2id[data.variable] = data.cvar_id
                    p.cvar_dtypes_set.add(data.data_type)
                    p.cvar_dtype_list.append(data.data_type)

                if name == "qvar_define":
                    if o["data_type"] != "qubits":
                        raise Exception(f"Do not know handle qvar type: {o['data_type']}")

                    qubit_ids = []
                    for i in range(o["size"]):
                        qubit_ids.append(next_qvar_int)
                        next_qvar_int += 1

                    data = d.QVarDefine(data_type=o["data_type"],
                                        variable=o["variable"],
                                        size=o["size"],
                                        qubit_ids=qubit_ids,
                                        metadata=o.get("metadata"), )

                    p.qvar_meta[data.variable] = data
                    p.num_qubits += data.size
            else:
                instr = cls.handle_op(o, p)
                p.ops.append(instr)

        return p
