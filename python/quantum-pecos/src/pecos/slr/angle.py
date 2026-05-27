"""Typed angle values for SLR rotation gates.

SLR rotation gates take a typed angle, not a bare float. `rad(x)` and
`turns(x)` construct an :class:`Angle` whose value is the exposed
``pecos.angle64`` fixed-point dtype -- the dtype carries all the math;
the wrapper adds only the source unit, used solely for pretty-printing.
Every backend unwraps to the underlying ``angle64`` before lowering, so
the wrapper is a display policy, not a parallel angle type.
"""

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pecos_rslib import angle64

__all__ = ["Angle", "rad", "turns"]


@dataclass(frozen=True)
class Angle:
    """A typed rotation angle: a ``pecos.angle64`` value plus its source unit.

    The ``value`` holds the exact fixed-point angle and owns every
    conversion (``to_radians``/``to_radians_signed``/``to_half_turns``/...).
    ``source_unit`` records whether the user wrote ``rad(...)`` or
    ``turns(...)`` so pretty-print can round-trip the unit label; it carries
    no math and is never consulted during backend lowering.
    """

    value: angle64
    source_unit: Literal["rad", "turns"]

    def slr_repr(self) -> str:
        """Render the SLR source form, e.g. ``rad(0.5)`` / ``turns(0.25)``.

        Uses the signed conversion so ordinary negative rotations read
        naturally; the numeric value is the canonicalized fixed-point angle,
        not the user's original literal (angle64 does not retain it).
        """
        if self.source_unit == "turns":
            return f"turns({self.value.to_turns_signed()})"
        return f"rad({self.value.to_radians_signed()})"


def rad(value: float) -> Angle:
    """Construct an :class:`Angle` from a value in radians."""
    return Angle(angle64.from_radians(float(value)), "rad")


def turns(value: float) -> Angle:
    """Construct an :class:`Angle` from a value in turns (1.0 = a full turn)."""
    return Angle(angle64.from_turns(float(value)), "turns")
