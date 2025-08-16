# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Classical computation utilities for the PECOS virtual machine.

This module provides functions for evaluating classical operations, expressions,
and conditional logic in the PECOS classical virtual machine (CVM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.engines.cvm.binarray import BinArray

if TYPE_CHECKING:
    from typing import Any

    from pecos.circuits import QuantumCircuit
    from pecos.protocols import SimulatorProtocol


def set_output(
    state: SimulatorProtocol,
    circuit: QuantumCircuit,
    output_spec: dict[str, int] | None,
    output: dict[str, BinArray] | None,
) -> dict[str, BinArray]:
    """Set up output dictionary for classical variable storage.

    Initializes the output dictionary with BinArrays for storing classical
    computation results, using size specifications from the circuit metadata
    and provided output specification.

    Args:
        state: Quantum simulator state providing qubit count.
        circuit: Quantum circuit containing variable specifications.
        output_spec: Dictionary mapping variable names to bit sizes.
        output: Existing output dictionary to update, if any.

    Returns:
        Initialized output dictionary with BinArrays for each variable.
    """
    if output_spec is None:
        output_spec = {}

    output_spec["__pecos_scratch"] = state.num_qubits

    if circuit.metadata.get("cvar_spec"):
        output_spec_new = circuit.metadata["cvar_spec"]
        output_spec_new.update(output_spec)
        output_spec = output_spec_new

    if output is None:
        output = {}

        if output_spec:
            for symbol, size in output_spec.items():
                output[symbol] = BinArray(size)

    return output


def eval_op(
    op: str,
    a: BinArray | int,
    b: BinArray | int | None = None,
    width: int = 32,
) -> BinArray:
    """Evaluate a binary or unary operation on BinArrays.

    Performs arithmetic, logical, or comparison operations on binary arrays,
    supporting assignment, bitwise operations, arithmetic, and comparisons.

    Args:
        op: Operation string (e.g., '=', '+', '&', '==', '~').
        a: First operand as BinArray or integer.
        b: Second operand for binary operations, None for unary operations.
        width: Bit width for integer to BinArray conversion.

    Returns:
        Result of the operation as a BinArray.

    Raises:
        Exception: If operation is unsupported or arguments are invalid.
    """
    if isinstance(a, int):
        a = BinArray(width, a)

    if op == "=":
        if b:
            msg = "Assignment can only have one argument (only `a`)."
            raise Exception(msg)

        return a

    if op == "|":
        expr_eval = a | b
    elif op == "^":
        expr_eval = a ^ b
    elif op == "&":
        expr_eval = a & b
    elif op == "+":
        expr_eval = a + b
    elif op == "-":
        expr_eval = a - b
    elif op == ">>":
        expr_eval = a >> b
    elif op == "<<":
        expr_eval = a << b
    elif op == "*":
        expr_eval = a * b
    elif op == "/":
        expr_eval = a // b
    elif op == "==":
        expr_eval = a == b
    elif op == "!=":
        expr_eval = a != b
    elif op == "<=":
        expr_eval = a <= b
    elif op == ">=":
        expr_eval = a >= b
    elif op == "<":
        expr_eval = a < b
    elif op == ">":
        expr_eval = a > b
    elif op == "%":
        expr_eval = a % b

    elif op == "~":
        expr_eval = ~a

        if b:
            msg = "Unary operation but got another argument!!!."
            raise Exception(msg)

    else:
        msg = (
            f'Receive op "{op}". Only operators `=`, `~`, `|`, `^`, `&`, `+`, `-`, `<<`, and `>>` '
            f"have been implemented."
        )
        raise Exception(msg)

    return expr_eval


def get_val(
    a: BinArray | tuple[str, int] | list[str | int] | str | int,
    output: dict[str, BinArray],
    width: int,
    shot_id: int,
) -> BinArray:
    """Extract and convert a value to BinArray.

    Retrieves values from the output dictionary or converts literals to BinArrays,
    supporting indexed access for array variables.

    Args:
        a: Value to extract - can be BinArray, variable reference, or literal.
        output: Dictionary containing variable values.
        width: Bit width for value conversion.
        shot_id: The current instance's shot id

    Returns:
        Value converted to BinArray format.

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(a, BinArray):
        return a

    if isinstance(a, tuple | list):
        sym, idx = a
        val = output[sym][idx]

    elif isinstance(a, str):
        val = shot_id if a == "JOB_shotnum" else int(output[a])

    elif isinstance(a, int):
        val = a

    else:
        msg = f'Could not evaluate "{a!s}". Wrong type, got type: {type(a)}.'
        raise TypeError(msg)

    return BinArray(width, val)


def recur_eval_op(
    expr_dict: dict[str, Any],
    output: dict[str, BinArray],
    width: int,
    shot_id: int,
) -> BinArray:
    """Recursively evaluate a nested expression dictionary.

    Processes nested expressions by recursively evaluating sub-expressions
    and combining results using the specified operations.

    Args:
        expr_dict: Dictionary containing expression with 'op', 'a', 'b', 'c' keys.
        output: Dictionary containing variable values.
        width: Bit width for operations.
        shot_id: The current instance's shot id.

    Returns:
        Result of the evaluated expression as BinArray.
    """
    a = expr_dict.get("a")
    op = expr_dict.get("op")
    b = expr_dict.get("b")
    c = expr_dict.get("c")

    if isinstance(a, dict):
        a = recur_eval_op(a, output, width, shot_id=shot_id)

    elif c:  # c => unary operation
        c = (
            recur_eval_op(c, output, width, shot_id=shot_id)
            if isinstance(c, dict)
            else get_val(c, output, width, shot_id)
        )

        a = eval_op(op, c, width=width)

    else:
        a = get_val(a, output, width, shot_id)

    if b:
        b = (
            recur_eval_op(b, output, width, shot_id=shot_id)
            if isinstance(b, dict)
            else get_val(b, output, width, shot_id)
        )

        a = eval_op(op, a, b, width=width)

    return a


def eval_cop(
    cop_expr: dict[str, Any] | list[dict[str, Any]],
    output: dict[str, BinArray],
    width: int,
    shot_id: int,
) -> None:
    """Evaluate classical operation expression.

    Evaluate classical expression such as:

    assignment:
    t = a     BinArray = (BinArray | int)
    t[i] = a  BinArray[i] = (BinArray | int)

    binary operations:
    t = a o b
    t[i] = a[j] o b[k]
    """
    # Get `t` argument
    # ----------------
    t = cop_expr[
        "t"
    ]  # symbol of where the resulting value will be stored in the output

    if isinstance(t, str):
        t_sym = t
        t_index = None
    elif isinstance(t, tuple | list) and len(t) == 2:
        t_sym = t[0]
        t_index = t[1]
    else:
        msg = "`t` should be `str` or `Tuple[str, int]`!"
        raise Exception(msg)

    t_obj = output[t_sym]

    # Eval assignment
    # ---------------
    expr_eval = recur_eval_op(cop_expr, output, width, shot_id=shot_id)

    # Assign the final value:
    # -----------------------
    if t_index is not None:  # t[i] = ...
        t_obj[t_index] = expr_eval[0]

    else:  # t = ...
        t_obj.set_clip(expr_eval)


def eval_tick_conds(
    tick_circuit: QuantumCircuit,
    output: dict[str, BinArray],
) -> list[bool]:
    """Evaluate conditional expressions for each operation in a tick circuit.

    Processes each operation in the circuit and evaluates its conditional
    expression to determine if the operation should be executed.

    Args:
        tick_circuit: Quantum circuit containing operations with conditions.
        output: Dictionary containing variable values for condition evaluation.

    Returns:
        List of boolean values indicating whether each operation's condition is true.
    """
    conds = []

    for _symbol, _locations, params in tick_circuit.items():
        cond_eval = eval_condition(params.get("cond"), output)

        conds.append(cond_eval)
    return conds


def eval_condition(
    conditional_expr: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
    output: dict[str, BinArray],
) -> bool:
    """Evaluate a conditional expression to a boolean result.

    Processes conditional expressions supporting comparison operators,
    variable references, and nested conditions. Returns True for None conditions.

    Args:
        conditional_expr: Expression dictionary with 'op', 'a', 'b' keys,
                         tuple/list for complex conditions, or None.
        output: Dictionary containing variable values.

    Returns:
        Boolean result of the conditional evaluation.

    Raises:
        Exception: If expression format is invalid.
        TypeError: If operand types are unexpected.
    """
    # Handle if a condition might eval to something else (eval_to)
    if isinstance(conditional_expr, tuple | list):
        if len(conditional_expr) != 2:
            msg = "Not expected conditional to have more than 2 elements."
            raise Exception(msg)

        if not isinstance(conditional_expr[1], bool):
            msg = "Expecting the second conditional element to be bool."
            raise TypeError(msg)

        return eval_condition(conditional_expr[0], output) == eval_condition(
            conditional_expr[1],
            output,
        )

    if conditional_expr:
        a = conditional_expr["a"]
        b = conditional_expr["b"]
        op = conditional_expr["op"]
        if isinstance(a, str):
            a = output[a]  # str -> BinArray
        elif isinstance(a, tuple | list) and len(a) == 2:
            a = output[a[0]][a[1]]  # (str, int) -> int (1 or 0)
        else:
            msg = "`a` should be `str` or `Tuple[str, int]`!"
            raise Exception(msg)

        if isinstance(b, str):
            b = output[b]  # str -> BinArray
        elif isinstance(b, tuple | list) and len(b) == 2:
            b = output[b[0]][b[1]]  # (str, int) -> int (1 or 0)
        elif isinstance(b, int):
            pass
        else:
            msg = "`b` should be `str` or `Tuple[str, int]` or `int`!"
            raise Exception(msg)

        # Map of operators to their evaluation functions
        ops = {
            "==": lambda a, b: bool(a == b),
            "!=": lambda a, b: bool(a != b),
            "^": lambda a, b: bool(int(a ^ b)),
            "|": lambda a, b: bool(int(a | b)),
            "&": lambda a, b: bool(int(a & b)),
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            ">>": lambda a, b: a >> b,
            "<<": lambda a, b: a << b,
            "~": lambda a, _: ~a,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a // b,
        }

        if op in ops:
            return ops[op](a, b)

        msg = "Comparison operator not recognized!"
        raise Exception(msg)

    return True
