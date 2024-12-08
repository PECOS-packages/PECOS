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

from phir.model import PHIRModel

from pecos.classical_interpreters.classical_interpreter_abc import ClassicalInterpreter
from pecos.reps.pypmir import PyPMIR, signed_data_types, unsigned_data_types
from pecos.reps.pypmir import types as pt

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    from pecos import QuantumCircuit
    from pecos.foreign_objects.foreign_object_abc import ForeignObject


def version2tuple(v):
    """Get version tuple from string."""
    return tuple(map(int, (v.split("."))))


data_type_map = signed_data_types | unsigned_data_types

data_type_map_rev = {v: k for k, v in data_type_map.items()}


class PHIRClassicalInterpreter(ClassicalInterpreter):
    """An interpreter that takes in a PHIR program and runs the classical side of the program."""

    def __init__(self) -> None:
        super().__init__()

        self.program = None
        self.foreign_obj = None
        self.cenv = None
        self.cid2dtype = None
        self.csym2id = None
        self.cvar_meta = None

        self.phir_validate = True

        self.reset()

    def _reset_env(self):
        self.cenv = []
        self.cid2dtype = []

    def reset(self):
        """Reset the state to that at initialization."""
        self.program = None
        self.foreign_obj = None
        self._reset_env()

    def init(
        self,
        program: str | (dict | QuantumCircuit),
        foreign_obj: ForeignObject | None = None,
    ) -> int:
        """Initialize the interpreter to validate the format of the program, optimize the program representation,
        etc.
        """
        self.program = program
        self.foreign_obj = foreign_obj

        # Make sure we have `program` in the correct format or convert to PHIR/dict.
        if isinstance(
            program,
            str,
        ):  # Assume it is in the PHIR/JSON format and convert to dict
            self.program = json.loads(program)
        elif isinstance(self.program, (PyPMIR, dict)):
            pass
        else:
            self.program = self.program.to_phir_dict()

        # Assume PHIR dict format, validate PHIR
        if isinstance(self.program, dict) and self.phir_validate:
            PHIRModel.model_validate(self.program)

        if isinstance(self.program, dict):
            assert self.program["format"] in ["PHIR/JSON", "PHIR"]  # noqa: S101
            assert version2tuple(self.program["version"]) < (0, 2, 0)  # noqa: S101

        # convert to a format that will, hopefully, run faster in simulation
        if not isinstance(self.program, PyPMIR):
            self.program = PyPMIR.from_phir(self.program)

        self.check_ffc(self.program.foreign_func_calls, self.foreign_obj)

        self.csym2id = dict(self.program.csym2id)
        self.cvar_meta = list(self.program.cvar_meta)

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
            for cvar in self.cvar_meta:
                cvar: pt.data.CVarDefine
                dtype = data_type_map[cvar.data_type]
                self.cenv.append(dtype(0))
                self.cid2dtype.append(dtype)

    def add_cvar(self, cvar: str, dtype, size: int):
        """Adds a new classical variable to the interpreter."""
        if cvar not in self.csym2id:
            cid = len(self.csym2id)
            self.csym2id[cvar] = cid
            self.cenv.append(dtype(0))
            self.cid2dtype.append(dtype)
            self.cvar_meta.append(
                pt.data.CVarDefine(
                    size=size,
                    data_type=data_type_map_rev[dtype],
                    cvar_id=cid,
                    variable=cvar,
                ),
            )

    def _flatten_blocks(self, seq: Sequence):
        """Flattens the ops of blocks to be processed by the execute() method."""
        for op in seq:
            if isinstance(op, pt.block.SeqBlock):
                yield from self._flatten_blocks(op.ops)

            elif isinstance(op, pt.block.IfBlock):
                if self.eval_expr(op.condition):
                    yield from self._flatten_blocks(op.true_branch)
                elif op.false_branch:
                    yield from self._flatten_blocks(op.false_branch)
                else:  # For case of no false_branch (no else)
                    pass

            else:
                yield op

    def execute(self, seq: Sequence) -> Generator[list, Any, None]:
        """A generator that runs through and executes classical logic and yields other operations via a buffer."""

        op_buffer = []

        for op in self._flatten_blocks(seq):
            if isinstance(op, pt.opt.QOp):
                op_buffer.append(op)

                if op.name == "Measure":
                    yield op_buffer
                    op_buffer.clear()

            elif isinstance(op, pt.opt.COp):
                self.handle_cops(op)

            elif isinstance(op, pt.opt.MOp):
                op_buffer.append(op)

            elif op is None:
                # TODO: Make it so None ops are not included
                continue

            else:
                msg = f"Statement not recognized: {op} of type: {type(op)}"
                raise TypeError(msg)

        if op_buffer:
            yield op_buffer

    def get_cval(self, cvar):
        cid = self.csym2id[cvar]
        return self.cenv[cid]

    def get_bit(self, cvar, idx):
        val = self.get_cval(cvar) & (1 << idx)
        val >>= idx
        return val

    def eval_expr(self, expr: int | str | list | pt.opt.COp) -> int | None:
        """Evaluates integer expressions."""
        match expr:
            case int():
                return expr

            case str():
                return self.get_cval(expr)
            case list():
                return self.get_bit(*expr)
            case pt.opt.COp():
                sym = expr.name
                args = expr.args

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
            case _:
                return None

    def assign_int(self, cvar, val: int):
        i = None
        if isinstance(cvar, (tuple, list)):
            cvar, i = cvar

        cid = self.csym2id[cvar]
        dtype = self.cid2dtype[cid]

        cval = self.cenv[cid]
        val = dtype(val)
        if i is None:
            cval = val
        else:
            one = dtype(1)
            i = dtype(i)
            cval &= ~(one << i)
            cval |= (val & one) << i

        if type(cval) not in signed_data_types.values():
            # mask off bits given the size of the register
            # (only valid for unsigned data types)
            size = self.cvar_meta[cid].size
            cval &= (1 << size) - 1
        self.cenv[cid] = cval

    def handle_cops(self, op):
        """Handle the processing of classical operations."""

        if op.name == "=":
            args = []
            for a in op.args:
                args.append(self.eval_expr(a))

            for r, a in zip(op.returns, args):
                self.assign_int(r, a)

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

            if op.returns is not None:
                if isinstance(results, int):
                    (cvar,) = op.returns
                    self.assign_int(cvar, results)
                else:
                    for cvar, val in zip(op.returns, results):
                        self.assign_int(cvar, val)

        else:
            msg = f"Unsupported COp: {op}"
            raise Exception(msg)

    def receive_results(self, qsim_results: list[dict]):
        """Receive measurement results and assign as needed."""
        for meas in qsim_results:
            for cvar, val in meas.items():
                self.assign_int(cvar, val)

    def results(self, return_int=True) -> dict:
        """Dumps program final results."""
        result = {}
        for csym, cid in self.csym2id.items():
            cval = self.cenv[cid]
            if not return_int:
                size = self.cvar_meta[cid].size
                cval = "{:0{width}b}".format(cval, width=size)
            result[csym] = cval

        return result

    def result_bits(
        self,
        bits: Iterable[tuple[str, int]],
        *,
        filter_private=True,
    ) -> dict[tuple[str, int], int]:
        """Git a dictionary of bit values given an iterable of bits (which are encoded as tuple[str, int]
        for str[int])."""
        send_meas = {}
        for b in bits:
            for m, i in b:
                m: str
                i: int
                if filter_private and m.startswith("__"):
                    continue
                send_meas[(m, i)] = self.get_bit(m, i)
        return send_meas
