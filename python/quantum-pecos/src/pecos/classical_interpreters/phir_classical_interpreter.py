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

"""PHIR classical interpreter for quantum-classical hybrid computations.

This module provides a classical interpreter for PHIR (PECOS High-level Intermediate Representation) programs,
enabling the execution of classical logic and control flow within quantum-classical hybrid algorithms in the PECOS
framework.
"""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Any

from pecos.reps.pyphir import PyPHIR, signed_data_types, unsigned_data_types
from pecos.reps.pyphir import types as pt
from pecos.typing import PhirModel

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence

    from pecos import QuantumCircuit
    from pecos.protocols import ForeignObjectProtocol
    from pecos.typing import Integer


def version2tuple(v: str) -> tuple[int, ...]:
    """Get version tuple from string."""
    return tuple(map(int, (v.split("."))))


data_type_map = signed_data_types | unsigned_data_types

data_type_map_rev = {v: k for k, v in data_type_map.items()}


class PhirClassicalInterpreter:
    """An interpreter that takes in a PHIR program and runs the classical side of the program."""

    def __init__(self) -> None:
        """Initialize the PHIR classical interpreter.

        Sets up the interpreter with default values for program state,
        environment variables, and validation settings.
        """
        self.program = None
        self.foreign_obj = None
        self.cenv = None
        self.cid2dtype = None
        self.csym2id = None
        self.cvar_meta = None

        self.phir_validate = True

        self.reset()

    def _reset_env(self) -> None:
        self.cenv = []
        self.cid2dtype = []

    def reset(self) -> None:
        """Reset the state to that at initialization."""
        self.program = None
        self.foreign_obj = None
        self._reset_env()

    def init(
        self,
        program: str | (dict | QuantumCircuit),
        foreign_obj: ForeignObjectProtocol | None = None,
    ) -> int:
        """Initialize the interpreter to validate and optimize the program.

        Validates the format of the program and optimizes the program representation.
        """
        self.program = program
        self.foreign_obj = foreign_obj

        # Make sure we have `program` in the correct format or convert to PHIR/dict.
        if isinstance(
            program,
            str,
        ):  # Assume it is in the PHIR/JSON format and convert to dict
            self.program = json.loads(program)
        elif isinstance(self.program, PyPHIR | dict):
            pass
        else:
            self.program = self.program.to_phir_dict()

        # Assume PHIR dict format, validate PHIR
        if isinstance(self.program, dict) and self.phir_validate:
            PhirModel.model_validate(self.program)

        if isinstance(self.program, dict):
            if self.program["format"] not in {"PHIR/JSON", "PHIR"}:
                msg = f"Unsupported PHIR format: {self.program['format']}"
                raise ValueError(msg)
            if version2tuple(self.program["version"]) >= (0, 2, 0):
                msg = f"PHIR version {self.program['version']} not supported; only versions < 0.2.0 are supported"
                raise ValueError(msg)

        # convert to a format that will, hopefully, run faster in simulation
        if not isinstance(self.program, PyPHIR):
            self.program = PyPHIR.from_phir(self.program)

        self.check_ffc(self.program.foreign_func_calls, self.foreign_obj)

        self.csym2id = dict(self.program.csym2id)
        self.cvar_meta = list(self.program.cvar_meta)

        self.initialize_cenv()

        return self.program.num_qubits

    def check_ffc(self, call_list: list[str], fobj: ForeignObjectProtocol) -> None:
        """Check foreign function calls compatibility with the foreign object.

        Args:
            call_list: List of foreign function calls from the program.
            fobj: Foreign object protocol to check against.

        Raises:
            Exception: If foreign function calls are not supported by the object.
        """
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

    def shot_reinit(self) -> None:
        """Run all code needed at the beginning of each shot, e.g., resetting state."""
        self.initialize_cenv()

    def initialize_cenv(self) -> None:
        """Initialize the classical environment with program variables."""
        self._reset_env()
        if self.program:
            for cvar in self.cvar_meta:
                cvar: pt.data.CVarDefine
                dtype = data_type_map[cvar.data_type]
                self.cenv.append(dtype(0))
                self.cid2dtype.append(dtype)

    def add_cvar(self, cvar: str, dtype: type[Integer], size: int) -> None:
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

    def _flatten_blocks(self, seq: Sequence) -> Generator[Any, None, None]:
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

                if op.name in {"measure Z", "Measure", "Measure +Z"}:
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

    def get_cval(self, cvar: str) -> Integer:
        """Get the classical value of a variable.

        Args:
            cvar: Name of the classical variable.

        Returns:
            The classical value as a PECOS integer.
        """
        cid = self.csym2id[cvar]
        return self.cenv[cid]

    def get_bit(self, cvar: str, idx: int) -> int:
        """Get a specific bit from a classical variable.

        Args:
            cvar: Name of the classical variable.
            idx: Bit index to extract.

        Returns:
            The bit value (0 or 1).
        """
        cval = self.get_cval(cvar)
        dtype = type(cval)

        # Get bit width using Rust-backed dtype system
        bit_width = dtype.itemsize * 8

        # Check if idx is within the valid range for the data type
        if idx >= bit_width:
            msg = f"Bit index {idx} out of range for {dtype} (max {bit_width - 1})"
            raise ValueError(
                msg,
            )

        # Use Rust-backed bitwise operations
        one = dtype(1)
        mask = one << dtype(idx)

        return (cval & mask) >> dtype(idx)

    def eval_expr(
        self,
        expr: int | str | list | pt.opt.COp,
    ) -> int | Integer | None:
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
                    lhs = self.eval_expr(args[0])
                    dtype = type(lhs)
                    return dtype(~lhs)

                # Binary operators
                lhs, rhs = args
                lhs = self.eval_expr(lhs)
                rhs = self.eval_expr(rhs)
                dtype = type(lhs)

                # Map of operators to their functions
                ops = {
                    "^": lambda x, y: x ^ y,
                    "+": lambda x, y: x + y,
                    "-": lambda x, y: x - y,
                    "|": lambda x, y: x | y,
                    "&": lambda x, y: x & y,
                    ">>": lambda x, y: x >> y,
                    "<<": lambda x, y: x << y,
                    "*": lambda x, y: x * y,
                    "/": lambda x, y: x // y,
                    "==": lambda x, y: x == y,
                    "!=": lambda x, y: x != y,
                    "<=": lambda x, y: x <= y,
                    ">=": lambda x, y: x >= y,
                    "<": lambda x, y: x < y,
                    ">": lambda x, y: x > y,
                    "%": lambda x, y: x % y,
                }

                if sym in ops:
                    return dtype(ops[sym](lhs, rhs))

                msg = f"Unknown expression type: {sym}"
                raise ValueError(msg)
            case _:
                return None

    def assign_int(self, cvar: str | tuple | list, val: int) -> None:
        """Assign an integer value to a classical variable or specific bit.

        Args:
            cvar: Variable name or tuple/list containing (variable_name, bit_index).
            val: Integer value to assign.
        """
        i = None
        if isinstance(cvar, tuple | list):
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

    def handle_cops(self, op: pt.opt.COp) -> None:
        """Handle the processing of classical operations."""
        if op.name == "=":
            args = [self.eval_expr(a) for a in op.args]

            for r, a in zip(op.returns, args, strict=False):
                self.assign_int(r, a)

        elif op.name == "Result":
            # The "Result" instruction maps internal register names to external ones
            # For example: {"cop": "Result", "args": ["m"], "returns": ["c"]}
            # maps the "m" register to "c" for user-facing results
            for src_reg, dst_reg in zip(op.args, op.returns, strict=False):
                if isinstance(src_reg, str) and src_reg in self.csym2id:
                    # If source register exists, copy its value to the destination register
                    src_id = self.csym2id[src_reg]
                    src_val = self.cenv[src_id]
                    src_size = self.cvar_meta[src_id].size
                    src_type = self.cvar_meta[src_id].data_type

                    # Create destination register if it doesn't exist yet
                    if dst_reg not in self.csym2id:
                        # Use the correct method to create a new variable
                        dtype = data_type_map[src_type]
                        self.add_cvar(dst_reg, dtype, src_size)

                    # Copy the value
                    dst_id = self.csym2id[dst_reg]
                    self.cenv[dst_id] = src_val

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
                    for cvar, val in zip(op.returns, results, strict=False):
                        self.assign_int(cvar, val)

        else:
            msg = f"Unsupported COp: {op}"
            raise Exception(msg)

    def receive_results(self, qsim_results: list[dict]) -> None:
        """Receive measurement results and assign as needed."""
        for meas in qsim_results:
            for cvar, val in meas.items():
                self.assign_int(cvar, val)

    def results(self, *, return_int: bool = True) -> dict:
        """Dumps program final results."""
        result = {}
        for csym, cid in self.csym2id.items():
            cval = self.cenv[cid]
            if not return_int:
                size = self.cvar_meta[cid].size
                # Use native __format__() implementation from Rust scalars
                cval = "{:0{width}b}".format(cval, width=size)
            result[csym] = cval

        return result

    def result_bits(
        self,
        bits: Iterable[tuple[str, int]],
        *,
        filter_private: bool = True,
    ) -> dict[tuple[str, int], int]:
        """Get a dictionary of bit values given an iterable of bits.

        Bits are encoded as tuple[str, int] for str[int].
        """
        send_meas = {}
        for b in bits:
            for m, i in b:
                m: str
                i: int
                if filter_private and m.startswith("__"):
                    continue
                send_meas[m, i] = self.get_bit(m, i)
        return send_meas
