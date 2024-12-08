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

from __future__ import annotations

from pecos.engines.cvm.binarray2 import BinArray2 as BinArray


def set_output(state, circuit, output_spec, output):
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


def eval_op(op, a, b=None, width=32):
    if isinstance(a, int):
        a = BinArray(width, a)

    if op == "=":
        if b:
            msg = "Assignment can only have one argument (only `a`)."
            raise Exception(msg)

        return a

    elif op == "|":
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


def get_val(a, output, width):
    if isinstance(a, BinArray):
        return a

    if isinstance(a, (tuple, list)):
        sym, idx = a
        val = output[sym][idx]

    elif isinstance(a, str):
        val = int(output[a])

    elif isinstance(a, int):
        val = a

    else:
        msg = f'Could not evaluate "{str(a)}". Wrong type, got type: {type(a)}.'
        raise TypeError(msg)

    return BinArray(width, val)


def recur_eval_op(expr_dict, output, width):
    a = expr_dict.get("a")
    op = expr_dict.get("op")
    b = expr_dict.get("b")
    c = expr_dict.get("c")

    if isinstance(a, dict):
        a = recur_eval_op(a, output, width)

    elif c:  # c => unary operation
        c = (
            recur_eval_op(c, output, width)
            if isinstance(c, dict)
            else get_val(c, output, width)
        )

        a = eval_op(op, c, width=width)

    else:
        a = get_val(a, output, width)

    if b:
        b = (
            recur_eval_op(b, output, width)
            if isinstance(b, dict)
            else get_val(b, output, width)
        )

        a = eval_op(op, a, b, width=width)

    return a


def eval_cop(cop_expr, output, width):
    """Evaluate classical expression such as:

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
    elif isinstance(t, (tuple, list)) and len(t) == 2:
        t_sym = t[0]
        t_index = t[1]
    else:
        msg = "`t` should be `str` or `Tuple[str, int]`!"
        raise Exception(msg)

    t_obj = output[t_sym]

    # Eval assignment
    # ---------------
    expr_eval = recur_eval_op(cop_expr, output, width)

    # Assign the final value:
    # -----------------------
    if t_index is not None:  # t[i] = ...
        t_obj[t_index] = expr_eval[0]

    else:  # t = ...
        t_obj.set_clip(expr_eval)


def eval_tick_conds(tick_circuit, output):
    conds = []

    for _symbol, _locations, params in tick_circuit.items():
        cond_eval = eval_condition(params.get("cond"), output)

        conds.append(cond_eval)
    return conds


def eval_condition(conditional_expr, output) -> bool:
    # Handle if a condition might eval to something else (eval_to)
    if isinstance(conditional_expr, (tuple, list)):
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
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            a = output[a[0]][a[1]]  # (str, int) -> int (1 or 0)
        else:
            msg = "`a` should be `str` or `Tuple[str, int]`!"
            raise Exception(msg)

        if isinstance(b, str):
            b = output[b]  # str -> BinArray
        elif isinstance(b, (tuple, list)) and len(b) == 2:
            b = output[b[0]][b[1]]  # (str, int) -> int (1 or 0)
        elif isinstance(b, int):
            pass
        else:
            msg = "`b` should be `str` or `Tuple[str, int]` or `int`!"
            raise Exception(msg)

        if op == "==":
            return bool(a.__eq__(b))

        if op == "!=":
            return bool(a.__ne__(b))

        elif op == "^":
            return bool(a.__xor__(b).__int__())

        elif op == "|":
            return bool(a.__or__(b).__int__())

        elif op == "&":
            return bool(a.__and__(b).__int__())

        elif op == "<":
            return a.__lt__(b)

        elif op == ">":
            return a.__gt__(b)

        elif op == "<=":
            return a.__le__(b)

        elif op == ">=":
            return a.__ge__(b)

        elif op == ">>":
            return a.__rshift__(b)

        elif op == "<<":
            return a.__lshift__(b)

        elif op == "~":
            return a.__invert__()

        elif op == "*":
            return a.__mul__()

        elif op == "/":
            return a.__floordiv__()

        else:
            msg = "Comparison operator not recognized!"
            raise Exception(msg)

    else:
        return True
