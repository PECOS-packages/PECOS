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

import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from pecos.classical_interpreters.classical_interpreter_abc import ClassicalInterpreter
from pecos.reps.pypmir import PyPMIR
from pecos.reps.pypmir import types as pt

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pecos import QuantumCircuit
    from pecos.foreign_objects.foreign_object_abc import ForeignObject


def version2tuple(v):
    """Get version tuple from string."""
    return tuple(map(int, (v.split("."))))


data_type_map = {
    "i32": np.int32,
    "i64": np.int64,
    "u32": np.uint32,
    "u64": np.uint64,
}


class PHIRClassicalInterpreter(ClassicalInterpreter):
    """An interpreter that takes in a PHIR program and runs the classical side of the program."""

    def __init__(self) -> None:
        super().__init__()

        self.program = None
        self.foreign_obj = None
        self.cenv = None
        self.cid2dtype = None

        self.reset()

    def _reset_env(self):
        self.cenv = []
        self.cid2dtype = []

    def reset(self):
        """Reset the state to that at initialization."""
        self.program = None
        self.foreign_obj = None
        self._reset_env()

    def init(self, program: str | (dict | QuantumCircuit), foreign_obj: ForeignObject | None = None) -> int:
        """Initialize the interpreter to validate the format of the program, optimize the program representation,
        etc.
        """
        self.program = program
        self.foreign_obj = foreign_obj

        # Make sure we have `program` in the correct format or convert to PHIR/dict.
        if isinstance(program, str):  # Assume it is in the PHIR/JSON format and convert to dict
            self.program = json.loads(self.program)
        elif isinstance(self.program, (PyPMIR, dict)):
            pass
        else:
            self.program = self.program.to_phir_dict()

        if isinstance(self.program, dict):
            assert self.program["format"] in ["PHIR/JSON", "PHIR"]  # noqa: S101
            assert version2tuple(self.program["version"]) < (0, 2, 0)  # noqa: S101

        # convert to a format that will, hopefully, run faster in simulation
        if not isinstance(self.program, PyPMIR):
            self.program = PyPMIR.from_phir(self.program)

        self.check_ffc(self.program.foreign_func_calls, self.foreign_obj)

        self.initialize_cenv()

        return self.program.num_qubits

    def check_ffc(self, call_list: list[str], fobj: ForeignObject):
        if self.program.foreign_func_calls:
            func_names = set(fobj.get_funcs())
            not_supported = set(call_list) - func_names
            if not_supported:
                msg = (
                    f"The following foreign function calls are listed in the program but not supported by the "
                    f"supplied foreign object: {not_supported}"
                )
                raise Exception(msg)
        elif not self.program.foreign_func_calls and self.foreign_obj:
            msg = "No foreign function calls being made but foreign object is supplied."
            raise warnings.warn(msg, stacklevel=2)

    def shot_reinit(self):
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        self.initialize_cenv()

    def initialize_cenv(self) -> None:
        self._reset_env()
        if self.program:
            for cvar in self.program.cvar_meta:
                cvar: pt.data.CVarDefine
                dtype = data_type_map[cvar.data_type]
                self.cenv.append(dtype(0))
                self.cid2dtype.append(dtype)

    def execute(self, sequence: Sequence) -> Generator[list, Any, None]:
        """A generator that runs through and executes classical logic and yields other operations via a buffer."""

        op_buffer = []

        for op in sequence:
            if isinstance(op, pt.opt.QOp):
                op_buffer.append(op)

                if op.name == "Measure":
                    yield op_buffer
                    op_buffer.clear()

            elif isinstance(op, pt.opt.COp):
                self.handle_cops(op)

            elif isinstance(op, pt.block.IfBlock):
                yield from self.execute_block(op)

            elif isinstance(op, pt.opt.MOp):
                op_buffer.append(op)

            else:
                msg = f"Statement not recognized: {op}"
                raise TypeError(msg)

    def get_cval(self, cvar):
        cid = self.program.csym2id[cvar]
        return self.cenv[cid]

    def get_bit(self, cvar, idx):
        val = self.get_cval(cvar) & (1 << idx)
        val >>= idx
        return val

    def eval_expr(self, expr: int | (str | (list | dict))) -> int:
        if isinstance(expr, int):
            return expr
        elif isinstance(expr, str):
            return self.get_cval(expr)
        elif isinstance(expr, list):
            return self.get_bit(*expr)
        elif isinstance(expr, dict):
            # TODO: Expressions need to be converted to nested COps!
            sym = expr["cop"]
            args = expr["args"]

            if sym in {"~"}:  # Unary ops
                lhs = args[0]
                rhs = None
            else:
                lhs, rhs = args
                rhs = self.eval_expr(rhs)

            lhs = self.eval_expr(lhs)
            dtype = type(lhs)

            if sym == "^":
                return dtype(lhs ^ rhs)
            elif sym == "+":
                return dtype(lhs + rhs)
            elif sym == "-":
                return dtype(lhs - rhs)
            elif sym == "|":
                return dtype(lhs | rhs)
            elif sym == "&":
                return dtype(lhs & rhs)
            elif sym == ">>":
                return dtype(lhs >> rhs)
            elif sym == "<<":
                return dtype(lhs << rhs)
            elif sym == "*":
                return dtype(lhs * rhs)
            elif sym == "/":
                return dtype(lhs // rhs)
            elif sym == "==":
                return dtype(lhs == rhs)
            elif sym == "!=":
                return dtype(lhs != rhs)
            elif sym == "<=":
                return dtype(lhs <= rhs)
            elif sym == ">=":
                return dtype(lhs >= rhs)
            elif sym == "<":
                return dtype(lhs < rhs)
            elif sym == ">":
                return dtype(lhs > rhs)
            elif sym == "%":
                return dtype(lhs % rhs)
            elif sym == "~":
                return dtype(~lhs)
            else:
                msg = f"Unknown expression type: {sym}"
                raise ValueError(msg)
        return None

    def assign_int(self, cvar, val: int):
        i = None
        if isinstance(cvar, (tuple, list)):
            cvar, i = cvar

        cid = self.program.csym2id[cvar]
        dtype = self.cid2dtype[cid]
        size = self.program.cvar_meta[cid].size

        cval = self.cenv[cid]
        val = dtype(val)
        if i is None:
            cval = val
        else:
            cval &= ~(1 << i)
            cval |= (val & 1) << i

        # mask off bits give the size of the register
        cval &= (1 << size) - 1
        self.cenv[cid] = cval

    def handle_cops(self, op):
        if op.name == "=":
            (arg,) = op.args
            (rtn,) = op.returns
            val = self.eval_expr(arg)
            self.assign_int(rtn, val)

        elif isinstance(op, pt.opt.FFCall):
            args = []
            for a in op.args:
                val = self.get_cval(a) if isinstance(a, str) else self.get_bit(*a)

                args.append(int(val))

            if op.metadata and "namespace" in op.metadata:
                results = self.foreign_obj.exec(op.name, args, op.metadata["namespace"])
            elif self.foreign_obj is None:
                msg = f"Trying to call foreign function `{op.name}` but no foreign object supplied!"
                raise Exception(msg)
            else:
                results = self.foreign_obj.exec(op.name, args)

            if isinstance(results, int):
                (cvar,) = op.returns
                self.assign_int(cvar, results)
            else:
                for cvar, val in zip(op.returns, results):
                    self.assign_int(cvar, val)

        else:
            msg = f"Unsupported COp: {op}"
            raise Exception(msg)

    def execute_block(self, op):
        """Execute a block of ops."""
        if isinstance(op, pt.block.IfBlock):
            if self.eval_expr(op.condition):
                yield from self.execute(op.true_branch)

            elif op.false_branch:
                yield from self.execute(op.false_branch)

            else:
                yield from self.execute([])

        elif isinstance(op, pt.block.SeqBlock):
            yield from self.execute(op.ops)

        else:
            msg = f"block not implemented! {op}"
            raise NotImplementedError(msg)

    def receive_results(self, qsim_results: list[dict]):
        """Receive measurement results and assign as needed."""
        for meas in qsim_results:
            for cvar, val in meas.items():
                self.assign_int(cvar, val)

    def results(self, return_int=True) -> dict:
        """Dumps program final results."""
        result = {}
        for csym, cid in self.program.csym2id.items():
            cval = self.cenv[cid]
            if not return_int:
                size = self.program.cvar_meta[cid].size
                cval = "{:0{width}b}".format(cval, width=size)
            result[csym] = cval

        return result
