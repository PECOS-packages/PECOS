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

"""Canonical prep-basis lowering, single source for every codegen.

A prep gate is a Z-reset (|0>) followed by a fixed Clifford tail. The
tail is expressed as `GateKind`s so each backend reuses its existing
`GateKind -> name` map (`GATE_TO_QIR/STIM/QC/QASM`, guppy
`FUNCTIONAL_GATES`); there is exactly ONE tail table, not six. Pinned
by review (states experimentally verified;
S = diag(1, i)): the uniform symmetric model X-basis = H,
phase-flip via trailing Z; Y-basis = H then S(+)/Sdg(-).
"""

from __future__ import annotations

from pecos.slr.ast.nodes import GateKind

# basis -> Clifford tail applied AFTER a |0> reset.
PREP_TAIL: dict[str, tuple[GateKind, ...]] = {
    "PZ": (),  # |0>
    "PNZ": (GateKind.X,),  # |1>
    "PX": (GateKind.H,),  # |+>
    "PNX": (GateKind.H, GateKind.Z),  # |->
    "PY": (GateKind.H, GateKind.SZ),  # |+i>
    "PNY": (GateKind.H, GateKind.SZdg),  # |-i>
}


def prep_tail(basis: str) -> tuple[GateKind, ...]:
    """Tail for `basis`, or fail LOUD on an unknown basis.

    An unknown basis silently lowering as a bare |0> reset would be
    exactly the silent-miscompile class this lowering exists to kill.
    """
    try:
        return PREP_TAIL[basis]
    except KeyError:
        msg = f"codegen: unknown prep basis {basis!r} (expected one of {sorted(PREP_TAIL)})."
        raise NotImplementedError(msg) from None
