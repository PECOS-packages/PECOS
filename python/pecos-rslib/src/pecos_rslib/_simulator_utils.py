# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Shared utilities for Rust-backed quantum simulators.

This module provides common infrastructure used by simulator wrappers, including the gate bindings
dictionary that enables backwards compatibility while delegating to Rust implementations.
"""

from __future__ import annotations

from pecos_rslib._pecos_rslib import (
    adjust_tableau_string as _adjust_tableau_string_rust,
)


class TableauWrapper:
    """Wrapper for accessing stabilizer/destabilizer tableaus from simulators.

    Provides a consistent interface for printing and accessing tableau representations
    of quantum states in the stabilizer formalism.
    """

    def __init__(self, sim, *, is_stab: bool) -> None:
        """Initialize the tableau wrapper.

        Args:
            sim: The simulator instance (must have stab_tableau/destab_tableau methods).
            is_stab: True for stabilizers, False for destabilizers.
        """
        self._sim = sim
        self._is_stab = is_stab

    def print_tableau(self, *, verbose: bool = False) -> list[str]:
        """Print the tableau representation.

        Args:
            verbose: If True, print the tableau to stdout.

        Returns:
            List of tableau strings, one per row.
        """
        if self._is_stab:
            tableau = self._sim.stab_tableau()
        else:
            tableau = self._sim.destab_tableau()

        lines = tableau.strip().split("\n")
        adjusted_lines = [
            adjust_tableau_string(line, is_stab=self._is_stab) for line in lines
        ]

        if verbose:
            for line in adjusted_lines:
                print(line)

        return adjusted_lines


def adjust_tableau_string(line: str, *, is_stab: bool) -> str:
    """Adjust the tableau string to ensure the sign part always takes up two spaces
    and convert 'Y' to 'W'. For destabilizers, always use two spaces for the sign.

    This is a thin wrapper around the Rust implementation that always converts Y to W.

    Args:
        line: A single line from the tableau string.
        is_stab: True if this is a stabilizer, False if destabilizer.

    Returns:
        The adjusted line with proper spacing for signs and 'W' instead of 'Y'.
    """
    # Call Rust implementation with print_y=False (always convert Y to W)
    return _adjust_tableau_string_rust(line, is_stab=is_stab, print_y=False)


class GateBindingsDict(dict):
    """Special dict that delegates all gate lookups to Rust's run_gate().

    This provides backwards compatibility for code that accesses sim.bindings[gate_name].
    Instead of storing lambdas for every gate, we create them on-demand using __missing__.

    This class is used by all Rust-backed simulators (SparseSimRs, CppSparseSimRs, StateVecRs)
    to provide a consistent interface for gate execution while minimizing code duplication.
    """

    def __init__(self, sim):
        """Initialize the gate bindings dictionary.

        Args:
            sim: The simulator instance that wraps a Rust simulator (_sim attribute).
        """
        super().__init__()
        self._sim = sim

    def __missing__(self, key):
        """Create a lambda on-demand that calls Rust's run_gate().

        Args:
            key: The gate name (e.g., "H", "CX", "measure Z").

        Returns:
            A lambda function that executes the gate on the simulator.
        """

        # Create a lambda that delegates to run_gate
        # This handles both 1q and 2q gates automatically
        def gate_lambda(sim, location, **params):
            # Convert location to tuple (for single location in a set)
            if isinstance(location, int):
                loc_tuple = (location,)
            elif isinstance(location, list):
                loc_tuple = tuple(location)
            else:
                loc_tuple = location

            # Wrap in a set (run_gate expects a set of locations)
            loc_set = {loc_tuple}

            # Call run_gate with keyword arguments properly
            result_dict = self._sim.run_gate(key, loc_set, **params)

            # Extract the result for this specific location
            # run_gate returns a dict mapping locations to results
            # For single-location calls, return the value or None
            if result_dict:
                # Get the value for the location (could be keyed by int or tuple)
                return result_dict.get(location) or result_dict.get(loc_tuple)
            return None

        # Cache the lambda for future use
        self[key] = gate_lambda
        return gate_lambda

    def get(self, key, default=None):
        """Override get() to trigger __missing__ for non-existent keys.

        Args:
            key: The gate name to look up.
            default: Default value to return if lookup fails.

        Returns:
            The gate lambda or default value.
        """
        try:
            return self[key]  # This will trigger __missing__ if key doesn't exist
        except Exception:
            return default

    def __contains__(self, key):
        """Override 'in' operator to always return True for gate lookups.

        This allows runtime gate discovery - the simulator will attempt to execute
        any gate and let the Rust implementation decide if it's supported.

        Args:
            key: The gate name to check.

        Returns:
            True if the gate lambda can be created, False otherwise.
        """
        # Try to get the item - this will create it via __missing__ if needed
        try:
            _ = self[key]  # This will trigger __missing__ and create the lambda
            return True
        except Exception:
            return False


__all__ = ["GateBindingsDict", "TableauWrapper", "adjust_tableau_string"]
