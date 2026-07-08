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


_MASK64 = (1 << 64) - 1


def _to_i64(bits: int) -> int:
    """Reinterpret the low 64 bits of ``bits`` as a signed two's-complement int."""
    bits &= _MASK64
    return bits - (1 << 64) if bits & (1 << 63) else bits


def _trunc_div(x: int, y: int) -> int:
    """Integer division truncated toward zero (C/Rust semantics).

    Python's ``//`` floors, so ``-7 // 2 == -4``; hardware/C integer division
    truncates toward zero, giving ``-3``. Matches the Rust interpreter's
    ``wrapping_div``.
    """
    q = abs(x) // abs(y)
    return -q if (x < 0) != (y < 0) else q


def _trunc_mod(x: int, y: int) -> int:
    """Integer remainder whose sign follows the dividend (C/Rust semantics).

    Consistent with :func:`_trunc_div`: ``x == _trunc_div(x, y) * y + _trunc_mod(x, y)``.
    So ``-7 % 3 == -1`` (not Python's ``2``). Matches Rust's ``wrapping_rem``.
    """
    r = abs(x) % abs(y)
    return -r if x < 0 else r


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

    @staticmethod
    def _check_register_width(variable: str, data_type: str, size: int, type_width: int) -> None:
        """Ensure a register's declared width fits its backing integer type.

        A signed size-S register is an ``i(S+1)`` integer (S data bits + a sign
        bit), so ``S + 1`` must fit the backing width N. An unsigned size-S
        register is a ``u(S)`` and must satisfy ``S <= N``.
        """
        needed = size + 1 if data_type in signed_data_types else size
        if needed > type_width:
            kind = "signed" if data_type in signed_data_types else "unsigned"
            limit = type_width - 1 if data_type in signed_data_types else type_width
            note = " (one bit is reserved for the sign)" if data_type in signed_data_types else ""
            msg = (
                f"Register {variable!r} declares {kind} type {data_type!r} with size {size}, "
                f"which does not fit its {type_width}-bit backing type{note}. "
                f"A size-{size} {kind} register needs {needed} bits. "
                f"Use size <= {limit} or a wider integer type."
            )
            raise ValueError(msg)

    def initialize_cenv(self) -> None:
        """Initialize the classical environment with program variables."""
        self._reset_env()
        if self.program:
            for cvar in self.cvar_meta:
                cvar: pt.data.CVarDefine
                dtype = data_type_map[cvar.data_type]
                self._check_register_width(cvar.variable, cvar.data_type, cvar.size, dtype.itemsize * 8)
                self.cenv.append(dtype(0))
                self.cid2dtype.append(dtype)

    def add_cvar(self, cvar: str, dtype: type[Integer], size: int) -> None:
        """Adds a new classical variable to the interpreter."""
        if cvar not in self.csym2id:
            data_type = data_type_map_rev[dtype]
            self._check_register_width(cvar, data_type, size, dtype.itemsize * 8)
            cid = len(self.csym2id)
            self.csym2id[cvar] = cid
            self.cenv.append(dtype(0))
            self.cid2dtype.append(dtype)
            self.cvar_meta.append(
                pt.data.CVarDefine(
                    size=size,
                    data_type=data_type,
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
    ) -> int:
        """Evaluate an integer expression at 64-bit backing precision.

        Mirrors the Rust interpreter: every operand is widened to 64 bits (signed
        values sign-extended, unsigned zero-extended), arithmetic wraps at 64
        bits, and division/modulo/shift/comparison follow signedness (an
        operation is signed only when BOTH operands are signed). The returned
        value is the 64-bit result reinterpreted as a signed integer; the
        assignment boundary (handle_cops/assign_int) clamps it to the
        destination register's i(S+1)/u(S) width.
        """
        bits, _signed, _is_bool = self._eval_bits(expr)
        return _to_i64(bits)

    def _eval_bits(self, expr: int | str | list | pt.opt.COp) -> tuple[int, bool, bool]:
        """Evaluate to a ``(64-bit bit pattern, is_signed, is_bool)`` triple.

        ``is_bool`` mirrors Rust's ``ExprValue::Boolean`` (a bit access or a
        ``&&``/``||`` result): unary ``~`` on a boolean is a logical NOT, not a
        full-width bit flip.
        """
        match expr:
            case int():
                # Literals that fit i64 are signed (Rust `ArgItem::Integer`);
                # larger values up to u64::MAX are unsigned (`ArgItem::UInteger`).
                signed = -(1 << 63) <= expr < (1 << 63)
                return expr & _MASK64, signed, False
            case str():
                cid = self.csym2id[expr]
                signed = self.cvar_meta[cid].data_type in signed_data_types
                return int(self.get_cval(expr)) & _MASK64, signed, False
            case list():
                # A single-bit access is a boolean 0/1.
                return int(self.get_bit(*expr)) & 1, False, True
            case pt.opt.COp():
                sym = expr.name
                if sym == "~":  # Unary NOT
                    (arg,) = expr.args
                    bits, signed, is_bool = self._eval_bits(arg)
                    if is_bool:
                        # Logical NOT of a boolean (matches Rust ~Boolean).
                        return int(not bits), False, True
                    # Bitwise NOT at eval width; keeps signedness.
                    return (~bits) & _MASK64, signed, False
                lhs, rhs = expr.args
                lbits, lsigned, _ = self._eval_bits(lhs)
                rbits, rsigned, _ = self._eval_bits(rhs)
                return self._eval_binop(sym, lbits, lsigned, rbits, rsigned)
            case _:
                msg = f"Unsupported expression: {expr!r}"
                raise ValueError(msg)

    @staticmethod
    def _eval_binop(op: str, lbits: int, lsigned: bool, rbits: int, rsigned: bool) -> tuple[int, bool, bool]:
        """Apply a binary op on 64-bit patterns, returning ``(bits, is_signed, is_bool)``.

        An operation is signed only when both operands are signed; division,
        modulo, shift-right and ordering comparisons then use the signed
        interpretation (matching the Rust interpreter). ``+ - * & | ^ << ==``
        depend only on the two's-complement bit pattern, so signedness only
        propagates the result tag for a later operation. ``&&``/``||`` return a
        boolean; the ordering/equality comparisons return an unsigned 0/1.
        """
        result_signed = lsigned and rsigned
        li = _to_i64(lbits)
        ri = _to_i64(rbits)

        match op:
            case "+":
                return (lbits + rbits) & _MASK64, result_signed, False
            case "-":
                return (lbits - rbits) & _MASK64, result_signed, False
            case "*":
                return (lbits * rbits) & _MASK64, result_signed, False
            case "&":
                return lbits & rbits, result_signed, False
            case "|":
                return lbits | rbits, result_signed, False
            case "^":
                return lbits ^ rbits, result_signed, False
            case "/":
                if rbits == 0:
                    msg = "division by zero"
                    raise ZeroDivisionError(msg)
                if result_signed:
                    return _trunc_div(li, ri) & _MASK64, True, False
                return (lbits // rbits) & _MASK64, False, False
            case "%":
                if rbits == 0:
                    msg = "modulo by zero"
                    raise ZeroDivisionError(msg)
                if result_signed:
                    return _trunc_mod(li, ri) & _MASK64, True, False
                return (lbits % rbits) & _MASK64, False, False
            case "<<":
                if ri < 0:
                    msg = f"Negative shift amount: {ri}"
                    raise ValueError(msg)
                shift = rbits & 0xFFFF
                return ((lbits << shift) & _MASK64 if shift < 64 else 0), result_signed, False
            case ">>":
                if ri < 0:
                    msg = f"Negative shift amount: {ri}"
                    raise ValueError(msg)
                if lsigned:  # arithmetic shift (sign-extends), tagged by the LHS
                    return (li >> (ri % 64)) & _MASK64, True, False
                shift = rbits & 0xFFFF
                return (lbits >> shift if shift < 64 else 0), False, False
            case "==":
                return int(lbits == rbits), False, False
            case "!=":
                return int(lbits != rbits), False, False
            case "<":
                return int(li < ri if result_signed else lbits < rbits), False, False
            case ">":
                return int(li > ri if result_signed else lbits > rbits), False, False
            case "<=":
                return int(li <= ri if result_signed else lbits <= rbits), False, False
            case ">=":
                return int(li >= ri if result_signed else lbits >= rbits), False, False
            case "&&":
                return int(lbits != 0 and rbits != 0), False, True
            case "||":
                return int(lbits != 0 or rbits != 0), False, True
            case _:
                msg = f"Unknown expression type: {op}"
                raise ValueError(msg)

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

        # Wrap the value to the register's declared width. In PHIR a signed
        # size-S register means an i(S+1) integer -- S data bits plus a sign
        # bit -- so it wraps to S+1 bits (capped at the backing type width),
        # and the sign bit is honored via two's complement. An unsigned size-S
        # register is a u(S) and wraps to S bits with no sign bit. Sign-extend
        # signed values before storing so later arithmetic and output see the
        # correct sign.
        meta = self.cvar_meta[cid]
        signed = meta.data_type in signed_data_types
        width = meta.size + 1 if signed else meta.size
        raw = int(cval) & ((1 << width) - 1)
        if signed and raw >> (width - 1):
            raw -= 1 << width
        self.cenv[cid] = dtype(raw)

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
                meta = self.cvar_meta[cid]
                # A signed size-S register is an i(S+1) integer, so it prints
                # S+1 bits (S data bits + a sign bit); an unsigned size-S
                # register prints S bits. Render the raw two's-complement bit
                # pattern rather than Python's sign-and-magnitude form (which
                # prints a leading "-" for negatives), so the sign bit shows up
                # as a "1"/"0" like every other bit.
                width = meta.size + 1 if meta.data_type in signed_data_types else meta.size
                raw = int(cval) & ((1 << width) - 1)
                cval = format(raw, f"0{width}b")
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
