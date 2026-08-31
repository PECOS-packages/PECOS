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

"""Callables handed out by a Rust-backed simulator's ``bindings`` must stay bound to their gate and simulator.

The bindings dictionary builds one Python callable per gate symbol. Each callable must capture its own gate
name and simulator instance: fetching another symbol, or the same symbol on another simulator, must not
redirect a callable that was fetched earlier.
"""

from __future__ import annotations

from pecos.simulators import SparseStab


def _tableau_after(gate: str, location: int | tuple[int, int]) -> tuple[str, str]:
    state = SparseStab(2)
    state.bindings[gate](state, location)
    return state.stab_tableau(), state.destab_tableau()


def test_held_callable_keeps_its_gate_after_another_symbol_is_fetched() -> None:
    state = SparseStab(2)
    apply_h = state.bindings["H"]
    state.bindings["CX"]
    apply_h(state, 0)
    assert (state.stab_tableau(), state.destab_tableau()) == _tableau_after("H", 0)


def test_held_callable_keeps_its_simulator_after_another_instance_fetches_the_symbol() -> None:
    first = SparseStab(2)
    second = SparseStab(2)
    apply_h_first = first.bindings["H"]
    second.bindings["H"]
    apply_h_first(first, 0)
    untouched = SparseStab(2)
    assert (first.stab_tableau(), first.destab_tableau()) == _tableau_after("H", 0)
    assert (second.stab_tableau(), second.destab_tableau()) == (
        untouched.stab_tableau(),
        untouched.destab_tableau(),
    )


def test_alias_assignment_does_not_redirect_existing_callables() -> None:
    state = SparseStab(2)
    apply_h = state.bindings["H"]
    state.bindings["CNOT"] = state.bindings["CX"]
    apply_h(state, 0)
    assert (state.stab_tableau(), state.destab_tableau()) == _tableau_after("H", 0)
    aliased = SparseStab(2)
    aliased.bindings["CNOT"] = aliased.bindings["CX"]
    aliased.bindings["CNOT"](aliased, (0, 1))
    assert (aliased.stab_tableau(), aliased.destab_tableau()) == _tableau_after("CX", (0, 1))
