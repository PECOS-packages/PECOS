# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Configurable classical semantics for the legacy hybrid engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pecos import BitInt, BitUInt
from pecos.engines.cvm import classical

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos.circuits import QuantumCircuit
    from pecos.protocols import SimulatorProtocol


class ClassicalSemantics(Protocol):
    """Classical operations required by the legacy :class:`HybridEngine`.

    Implementations can select arithmetic, comparison, and register-storage
    behavior without replacing or monkey-patching PECOS module functions.
    """

    def set_output(
        self,
        state: SimulatorProtocol,
        circuit: QuantumCircuit,
        output_spec: dict[str, int] | None,
        output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Initialize classical storage for a circuit execution."""
        ...

    def eval_condition(
        self,
        conditional_expr: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
        *,
        width: int,
    ) -> bool:
        """Evaluate whether a conditional operation should execute."""
        ...

    def eval_cop(
        self,
        cop_expr: dict[str, Any],
        output: dict[str, Any],
        *,
        width: int,
        shot_id: int,
    ) -> None:
        """Evaluate and store a classical expression."""
        ...


class DefaultClassicalSemantics:
    """PECOS's existing signed ``BitInt`` classical semantics."""

    @staticmethod
    def set_output(
        state: SimulatorProtocol,
        circuit: QuantumCircuit,
        output_spec: dict[str, int] | None,
        output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Initialize classical storage using PECOS's default behavior."""
        return classical.set_output(state, circuit, output_spec, output)

    @staticmethod
    def eval_condition(
        conditional_expr: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
        *,
        width: int,
    ) -> bool:
        """Evaluate a condition using PECOS's default behavior."""
        del width
        return classical.eval_condition(conditional_expr, output)

    @staticmethod
    def eval_cop(
        cop_expr: dict[str, Any],
        output: dict[str, Any],
        *,
        width: int,
        shot_id: int,
    ) -> None:
        """Evaluate an expression using PECOS's default behavior."""
        classical.eval_cop(cop_expr, output, width=width, shot_id=shot_id)


class UnsignedClassicalSemantics:
    """Unsigned fixed-width classical semantics.

    Arithmetic wraps modulo ``2**width``. Shifts are logical and mask their
    amount to ``width - 1``. Division and modulo by zero produce an all-ones
    word. Comparisons are unsigned unless an expression contains
    ``{"signed": True}``.

    By default, circuit variables whose ``cvar_spec_type`` is ``"cbitvar"``
    use :class:`BitUInt` storage so narrow values are zero-extended. Other
    variables retain :class:`BitInt` storage for two's-complement reporting.
    Alternate metadata type names can be supplied with
    ``unsigned_cvar_types``.

    A semantics instance has one authoritative word width. Explicit widths
    supplied by an engine or another caller must match the configured width.
    This prevents different components in one execution from silently
    evaluating the same expression at different widths.
    """

    _COMPARISONS = frozenset({"==", "!=", "<=", ">=", "<", ">"})

    def __init__(
        self,
        width: int = 32,
        *,
        unsigned_cvar_types: Iterable[str] = ("cbitvar",),
    ) -> None:
        """Create unsigned semantics for one classical word width.

        The default matches the legacy hybrid engine's default ``regwidth``.
        Consumers targeting another word size should pass that size to both
        the semantics object and the engine.

        Args:
            width: Default classical word width. Must be a positive power of
                two.
            unsigned_cvar_types: ``cvar_spec_type`` metadata values whose
                registers should use unsigned, zero-extending storage.
        """
        self._validate_width(width)
        self.width = width
        self.unsigned_cvar_types = frozenset(unsigned_cvar_types)

    @staticmethod
    def _validate_width(width: int) -> None:
        if width <= 0 or width & (width - 1):
            msg = f"Classical register width must be a positive power of two, got {width}."
            raise ValueError(msg)

    def _resolve_width(self, width: int | None) -> int:
        if width is None:
            return self.width
        self._validate_width(width)
        if width != self.width:
            msg = f"Classical register width {width} does not match the semantics width {self.width}."
            raise ValueError(msg)
        return width

    @staticmethod
    def _to_signed(value: int, width: int) -> int:
        value &= (1 << width) - 1
        if value >> (width - 1):
            return value - (1 << width)
        return value

    def get_val(
        self,
        value: BitInt | BitUInt | tuple[str, int] | list[str | int] | str | int,
        output: dict[str, Any],
        width: int | None,
        shot_id: int,
    ) -> BitUInt | BitInt:
        """Resolve a literal or variable reference as an unsigned word."""
        width = self._resolve_width(width)
        if isinstance(value, (BitInt, BitUInt)):
            return value
        if isinstance(value, (tuple, list)):
            symbol, index = value
            resolved = output[symbol][index]
        elif isinstance(value, str):
            resolved = shot_id if value == "JOB_shotnum" else int(output[value])
        elif isinstance(value, int):
            resolved = value
        else:
            msg = f'Could not evaluate "{value!s}". Wrong type, got type: {type(value)}.'
            raise TypeError(msg)
        return BitUInt(width, resolved)

    def eval_op(
        self,
        op: str,
        a: BitUInt | BitInt | int,
        b: BitUInt | BitInt | int | None = None,
        *,
        width: int | None = None,
        signed: bool = False,
    ) -> BitUInt:
        """Evaluate one operation with unsigned fixed-width arithmetic."""
        width = self._resolve_width(width)
        left = BitUInt(width, int(a))
        right = BitUInt(width, int(b)) if b is not None else None

        if op == "=":
            if right is not None:
                msg = "Assignment can only have one argument (only `a`)."
                raise ValueError(msg)
            return left
        if op == "~":
            if right is not None:
                msg = "Unary operation received a second argument."
                raise ValueError(msg)
            return ~left
        if op == "|":
            return left | right
        if op == "^":
            return left ^ right
        if op == "&":
            return left & right
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == ">>":
            return left >> (int(right) & (width - 1))
        if op == "<<":
            return left << (int(right) & (width - 1))
        if op in {"/", "%"}:
            if int(right) == 0:
                return BitUInt(width, (1 << width) - 1)
            return left // right if op == "/" else left % right
        if op in self._COMPARISONS:
            if signed:
                left_value = self._to_signed(int(left), width)
                right_value = self._to_signed(int(right), width)
            else:
                left_value = int(left)
                right_value = int(right)
            result = {
                "==": left_value == right_value,
                "!=": left_value != right_value,
                "<=": left_value <= right_value,
                ">=": left_value >= right_value,
                "<": left_value < right_value,
                ">": left_value > right_value,
            }[op]
            return BitUInt(width, int(result))
        msg = f"Unsupported classical operator: {op!r}."
        raise ValueError(msg)

    def _eval_expr(
        self,
        expr: dict[str, Any],
        output: dict[str, Any],
        width: int,
        shot_id: int,
    ) -> BitUInt:
        a = expr.get("a")
        op = expr.get("op")
        b = expr.get("b")
        c = expr.get("c")
        signed = bool(expr.get("signed", False))

        if isinstance(a, dict):
            a = self._eval_expr(a, output, width, shot_id)
        elif c is not None:
            c = (
                self._eval_expr(c, output, width, shot_id)
                if isinstance(c, dict)
                else self.get_val(
                    c,
                    output,
                    width,
                    shot_id,
                )
            )
            a = self.eval_op(op, c, width=width, signed=signed)
        else:
            a = self.get_val(a, output, width, shot_id)

        if b is not None:
            b = (
                self._eval_expr(b, output, width, shot_id)
                if isinstance(b, dict)
                else self.get_val(
                    b,
                    output,
                    width,
                    shot_id,
                )
            )
            a = self.eval_op(op, a, b, width=width, signed=signed)
        return a

    def set_output(
        self,
        state: SimulatorProtocol,
        circuit: QuantumCircuit,
        output_spec: dict[str, int] | None,
        output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Initialize signed and unsigned circuit-variable storage."""
        resolved_spec = dict(circuit.metadata.get("cvar_spec") or {})
        resolved_spec.update(output_spec or {})
        resolved_spec["__pecos_scratch"] = state.num_qubits

        if output is None:
            cvar_types = circuit.metadata.get("cvar_spec_type", {})
            output = {
                symbol: (BitUInt(size) if cvar_types.get(symbol) in self.unsigned_cvar_types else BitInt(size))
                for symbol, size in resolved_spec.items()
            }
        return output

    def eval_condition(
        self,
        conditional_expr: dict[str, Any] | tuple[Any, ...] | list[Any] | None,
        output: dict[str, Any],
        *,
        width: int | None = None,
    ) -> bool:
        """Evaluate a condition with optional signed comparison semantics."""
        width = self._resolve_width(width)
        if isinstance(conditional_expr, (tuple, list)):
            if len(conditional_expr) != 2:
                msg = "Expected a two-element conditional expression."
                raise ValueError(msg)
            if not isinstance(conditional_expr[1], bool):
                msg = "Expected the second conditional element to be bool."
                raise TypeError(msg)
            return self.eval_condition(conditional_expr[0], output, width=width) == conditional_expr[1]
        if conditional_expr:
            a = classical._resolve_condition_operand(  # noqa: SLF001
                conditional_expr["a"],
                output,
                "a",
            )
            b = classical._resolve_condition_operand(  # noqa: SLF001
                conditional_expr["b"],
                output,
                "b",
            )
            return bool(
                self.eval_op(
                    conditional_expr["op"],
                    a,
                    b,
                    width=width,
                    signed=bool(conditional_expr.get("signed", False)),
                ),
            )
        return True

    def eval_cop(
        self,
        cop_expr: dict[str, Any],
        output: dict[str, Any],
        *,
        width: int | None = None,
        shot_id: int,
    ) -> None:
        """Evaluate and store an expression using target register semantics."""
        width = self._resolve_width(width)
        target = cop_expr["t"]
        if isinstance(target, str):
            target_symbol = target
            target_index = None
        elif isinstance(target, (tuple, list)) and len(target) == 2:
            target_symbol, target_index = target
        else:
            msg = "`t` should be a string or a two-element variable reference."
            raise TypeError(msg)

        target_obj = output[target_symbol]
        value = self._eval_expr(cop_expr, output, width, shot_id)
        if target_index is not None:
            target_obj[target_index] = value[0]
        elif isinstance(target_obj, BitUInt):
            target_obj.set_clip(value)
        else:
            target_obj.set_clip(BitInt(width, self._to_signed(int(value), width)))


__all__ = [
    "ClassicalSemantics",
    "DefaultClassicalSemantics",
    "UnsignedClassicalSemantics",
]
