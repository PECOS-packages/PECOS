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

"""Install table-backed Clifford rotation bindings on Python simulators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib import lower_clifford_rotation

from pecos.exceptions import NotSupportedGateError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Protocol

    class _BindingState(Protocol):
        bindings: Mapping[str, Callable[..., object]]


_ONE_ANGLE_ROTATIONS = ("RZ", "RX", "RY", "RZZ", "RXX", "RYY")


def _rotation_binding(
    symbol: str,
) -> Callable[..., None]:
    def apply_rotation(state: _BindingState, location: int | tuple[int, ...], **params: object) -> None:
        if symbol in {"RXY1Q", "R1XY"}:
            if "angles" not in params:
                msg = "RXY1Q requires an 'angles' parameter"
                raise ValueError(msg)
            angles = params["angles"]
        else:
            if "angle" not in params:
                msg = f"{symbol} requires an 'angle' parameter"
                raise ValueError(msg)
            angles = (params["angle"],)

        operands = (location,) if isinstance(location, int) else tuple(location)
        for named, positions in lower_clifford_rotation(symbol, angles):
            named_operands = tuple(operands[position] for position in positions)
            named_location = named_operands[0] if len(named_operands) == 1 else named_operands
            try:
                named_binding = state.bindings[named]
            except KeyError as error:
                msg = f'The gate "{named}" is not available for this simulator: {type(state)}. Metadata: {params}'
                raise NotSupportedGateError(msg) from error
            named_binding(state, named_location)

    return apply_rotation


def install_clifford_rotation_bindings(bindings: dict[str, Callable[..., object]]) -> None:
    """Install projective rotation lowerings for stabilizer/tableau consumers.

    Results are equivalent only up to global phase and are unsuitable for
    phase-carrying simulation or matrix-exact rewriting. For example,
    ``RZZ(3*pi/2) = -SZZdg``, while the lowering installs ``SZZdg``.
    """
    for symbol in (*_ONE_ANGLE_ROTATIONS, "RXY1Q", "R1XY"):
        bindings[symbol] = _rotation_binding(symbol)
