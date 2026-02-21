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

"""Tests for TableauWrapper col_x, col_z, row_x, row_z properties."""

from pecos.simulators import SparseSim


def test_initial_state_tableau_properties() -> None:
    """Test tableau properties for the initial |0...0> state.

    For n qubits in |0...0>, stabilizers are Z_i for each qubit i,
    so col_z should have qubit i in generator i, and col_x should be empty.
    """
    n = 3
    state = SparseSim(n)

    stabs = state.stabs
    col_x = stabs.col_x
    col_z = stabs.col_z

    # Each stabilizer is Z_i, so col_z[i] should contain [i] and col_x[i] should be empty
    for i in range(n):
        assert col_x[i] == [], f"col_x[{i}] should be empty for |0> state"
        assert col_z[i] == [i], f"col_z[{i}] should be [{i}] for |0> state"


def test_bell_state_tableau_properties() -> None:
    """Test tableau properties after creating a Bell-like entangled state."""
    state = SparseSim(2)
    state.run_gate("H", {0})
    state.run_gate("CX", {(0, 1)})

    stabs = state.stabs
    col_x = stabs.col_x
    col_z = stabs.col_z

    # After H on qubit 0 and CX(0,1), stabilizers are XX and ZZ
    # Verify we get non-empty data back with the right shape
    assert len(col_x) == 2
    assert len(col_z) == 2


def test_tableau_row_properties() -> None:
    """Test that row_x and row_z are accessible and have correct dimensions."""
    n = 3
    state = SparseSim(n)

    stabs = state.stabs
    row_x = stabs.row_x
    row_z = stabs.row_z

    assert len(row_x) == n
    assert len(row_z) == n


def test_destabs_tableau_properties() -> None:
    """Test that destabilizer tableau properties are accessible."""
    n = 3
    state = SparseSim(n)

    destabs = state.destabs
    col_x = destabs.col_x
    col_z = destabs.col_z
    row_x = destabs.row_x
    row_z = destabs.row_z

    assert len(col_x) == n
    assert len(col_z) == n
    assert len(row_x) == n
    assert len(row_z) == n

    # For initial |0> state, destabilizers are X_i
    for i in range(n):
        assert col_x[i] == [i], f"destab col_x[{i}] should be [{i}] for |0> state"
        assert col_z[i] == [], f"destab col_z[{i}] should be empty for |0> state"


def test_gens_property() -> None:
    """Test that the gens property returns (stabs, destabs) tuple."""
    state = SparseSim(2)
    stabs_wrapper, destabs_wrapper = state.gens

    # Should be able to access col_x on both
    assert len(stabs_wrapper.col_x) == 2
    assert len(destabs_wrapper.col_x) == 2


def test_tableau_after_gate() -> None:
    """Test that tableau properties update correctly after applying gates."""
    state = SparseSim(1)

    # Initial |0>: stab is Z
    assert state.stabs.col_x == [[]]
    assert state.stabs.col_z == [[0]]

    # After H: stab is X
    state.run_gate("H", {0})
    assert state.stabs.col_x == [[0]]
    assert state.stabs.col_z == [[]]
