# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Dedicated prep gates -- BEHAVIORAL suite.

The 6 dedicated prep gates `PZ/PNZ/PX/PNX/PY/PNY` (basis is the gate
IDENTITY, not a string arg) lower to a Z-reset + a fixed Clifford
tail (single shared `_prep_tail` map). This suite asserts the
*behavior* (state / measured bit), not emitted text, across:

  (a) Stim peek_bloch -- exact single-qubit Bloch axis per gate.
  (b) AST -> Guppy -> Selene -- prep then measure-in-prep-basis is
      deterministic; assert the fixed bit over shots.
  (c) emitted-tail spot-check (QIR/QASM/QuantumCircuit) -- the
      distinguishing Clifford tail is present per the pinned table.
  (d) Soundness-critical: a non-PZ prep inside a
      BlockDecl invoked via BlockCall survives `flatten_block_calls`
      with its basis intact (QASM AND Guppy).
  (e) simulator: the Pn->PN-renamed entry points + human aliases
      drive the correct measured bit through StateVec.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import stim
from pecos.simulators import StateVec
from pecos.slr import Block, CReg, Main, QReg, Return, SlrConverter
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure

from ._selene_harness import run_ast_guppy_via_selene  # noqa: TID252

# gate -> (Stim Bloch axis, rotation-to-Z-eigenbasis gates, expected
# MZ bit after that rotation). Rotation maps the prep eigenstate to a
# computational-basis eigenstate so the measured bit is deterministic.
_GATES = {
    "PZ": ("+Z", [], 0),
    "PNZ": ("-Z", [], 1),
    "PX": ("+X", ["H"], 0),  # |+> --H--> |0>
    "PNX": ("-X", ["H"], 1),  # |-> --H--> |1>
    "PY": ("+Y", ["SZdg", "H"], 0),  # |+i> --SZdg(S†);H--> |0>
    "PNY": ("-Y", ["SZdg", "H"], 1),  # |-i> --SZdg(S†);H--> |1>
}


@pytest.mark.parametrize("gate", list(_GATES))
def test_prep_gate_stim_bloch(gate: str) -> None:
    """(a) Each dedicated prep gate yields its exact Bloch eigenstate."""
    axis = _GATES[gate][0]
    prog = Main(q := QReg("q", 1), getattr(qb, gate)(q[0]))
    s = SlrConverter(prog).stim()
    circ = s if isinstance(s, stim.Circuit) else stim.Circuit(str(s))
    sim = stim.TableauSimulator()
    sim.do(circ)
    assert axis in str(sim.peek_bloch(0)), f"{gate}: expected Bloch {axis}, got {sim.peek_bloch(0)}"


@pytest.mark.slow
@pytest.mark.parametrize("gate", list(_GATES))
def test_prep_gate_selene_guppy(gate: str) -> None:
    """(b) Prep then measure-in-prep-basis is deterministic through
    the AST -> Guppy -> Selene path (every shot the fixed bit)."""
    _axis, rot, expected = _GATES[gate]
    prog = Main(
        q := QReg("q", 1),
        c := CReg("c", 1),
        getattr(qb, gate)(q[0]),
        *[getattr(qb, g)(q[0]) for g in rot],
        Measure(q[0]) > c[0],
        Return(c),
    )
    records = run_ast_guppy_via_selene(prog, shots=8)
    bits = {r.get("measurement_0", 0) for r in records}
    assert bits == {expected}, f"{gate}: expected every shot == {expected}, got {sorted(bits)} ({records})"


@pytest.mark.parametrize(
    ("gate", "needles"),
    [
        ("PZ", {"qir": ["reset"], "qasm": ["reset "]}),
        ("PNZ", {"qir": ["reset", "__quantum__qis__x__body"], "qasm": ["reset ", "x "]}),
        ("PX", {"qir": ["reset", "__quantum__qis__h__body"], "qasm": ["reset ", "h "]}),
        (
            "PNX",
            {"qir": ["reset", "__quantum__qis__h__body", "__quantum__qis__z__body"], "qasm": ["reset ", "h ", "z "]},
        ),
        (
            "PY",
            # GateKind canonicalised to SZ/SZdg (S/Sdg removed): QIR
            # SZ->"s" (unchanged); QASM SZ->"rz(pi/2)" (the chosen
            # phase-gate QASM lowering -- s == rz(pi/2) up to phase).
            {
                "qir": ["reset", "__quantum__qis__h__body", "__quantum__qis__s__body"],
                "qasm": ["reset ", "h ", "rz(pi/2)"],
            },
        ),
        (
            "PNY",
            {
                "qir": ["reset", "__quantum__qis__h__body", "__quantum__qis__s__adj"],
                "qasm": ["reset ", "h ", "rz(-pi/2)"],
            },
        ),
    ],
)
def test_prep_gate_emitted_tail(gate: str, needles: dict[str, list[str]]) -> None:
    """(c) The distinguishing reset+Clifford tail is emitted per the
    pinned `_prep_tail` table (QIR + QASM spot-check)."""
    prog = Main(q := QReg("q", 1), c := CReg("c", 1), getattr(qb, gate)(q[0]), Measure(q[0]) > c[0], Return(c))
    qir = SlrConverter(prog).qir()
    for n in needles["qir"]:
        assert n in qir, f"{gate}: QIR missing {n!r}"
    qasm = SlrConverter(prog).qasm()
    for n in needles["qasm"]:
        assert n in qasm, f"{gate}: QASM missing {n!r}"


def test_b1_blockcall_preserves_prep_basis() -> None:
    """(d) Soundness-critical: a non-PZ prep inside a
    BlockDecl invoked via BlockCall must keep its basis through
    `flatten_block_calls` -- in QASM AND Guppy. PX = |0> reset + H;
    if the basis were dropped to PZ the H tail would vanish."""

    class PrepXBlock(Block):
        block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.extend(qb.PX(q[0]))

    prog = Main(
        outer := QReg("outer", 1),
        c := CReg("c", 1),
        PrepXBlock(outer),
        Measure(outer[0]) > c[0],
        Return(c),
    )

    qasm = SlrConverter(prog).qasm()
    assert "reset " in qasm, f"PX BlockCall->QASM lost the reset:\n{qasm}"
    assert "h " in qasm, f"PX basis lost through BlockCall->QASM (no H tail):\n{qasm}"

    guppy = SlrConverter(prog).guppy()
    assert "= h(" in guppy, f"PX basis lost through BlockCall->Guppy (no H tail):\n{guppy}"

    # And it stays behaviorally |+>: H then measure -> deterministic 0.
    class PrepXThenH(Block):
        block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

        def __init__(self, q: QReg) -> None:
            super().__init__()
            self.q = q
            self.extend(qb.PX(q[0]))

    prog2 = Main(
        outer := QReg("outer", 1),
        c := CReg("c", 1),
        PrepXThenH(outer),
        qb.H(outer[0]),
        Measure(outer[0]) > c[0],
        Return(c),
    )
    recs = run_ast_guppy_via_selene(prog2, shots=8)
    assert {r.get("measurement_0", 0) for r in recs} == {0}, f"PX|BlockCall not |+>: {recs}"


test_b1_blockcall_preserves_prep_basis = pytest.mark.slow(test_b1_blockcall_preserves_prep_basis)


# Each negative-eigenstate entry, measured natively in ITS OWN basis
# (the sim has MZ/MX/MY), is deterministically 1; positive is 0.
@pytest.mark.parametrize(
    ("entry", "meas", "expected"),
    [
        # Direct canonical keys (all 6 must dispatch
        # directly, not only via aliases).
        ("PZ", "MZ", 0),
        ("PNZ", "MZ", 1),
        ("PX", "MX", 0),
        ("PNX", "MX", 1),
        ("PY", "MY", 0),
        ("PNY", "MY", 1),
        # Human aliases must still resolve post Pn->PN rename.
        ("Init -Z", "MZ", 1),
        ("init |1>", "MZ", 1),
        ("Init -X", "MX", 1),
        ("init |->", "MX", 1),
        ("Init -Y", "MY", 1),
        ("init |-i>", "MY", 1),
    ],
)
def test_sim_pn_entry_points(entry: str, meas: str, expected: int) -> None:
    """(e) The Pn->PN-renamed canonical key + human aliases drive the
    correct state through StateVec (rebuilt rslib dispatch),
    confirmed by measuring natively in the prep's own basis."""
    s = StateVec(1)
    s.run_gate(entry, {0})
    r = s.run_gate(meas, {0})
    v = r.get(0, 0) if isinstance(r, dict) else r
    assert v == expected, f"{entry!r} measured {meas}: {v}, expected {expected}"
