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

"""Simulation function utilities for the classical virtual machine.

This module provides debugging and introspection functions for use within
the PECOS classical virtual machine during quantum circuit simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:

    class Runner(Protocol):
        """Protocol for runner objects used in simulation functions."""

        generate_errors: bool
        state: Any  # Has get_amps method


def sim_print(_runner: Runner, *args: tuple[str, Any]) -> None:
    """Print simulation variables for debugging.

    Outputs variable names and their values in a formatted way for debugging
    and inspection during simulation execution.

    Args:
        _runner: Simulation runner (unused).
        *args: Variable name-value pairs to print.
    """
    syms = [s for s, _ in args]
    syms = ", ".join(syms)
    print(f"sim_print({syms}):")
    for sym, b in args:
        print(f"    {sym}: {b!s} ({int(b)})")
    print()


def sim_test(
    _runner: Runner,
    *_args: object,
) -> None:
    """Test function for simulation debugging.

    Simple test function that prints a message to verify simulation
    function dispatch is working correctly.

    Args:
        _runner: Simulation runner (unused).
        *_args: Arguments (ignored).
    """
    print("SIM TEST!")


def sim_get_amp(
    runner: Runner,
    key_state: tuple[tuple[Any, Any], ...],
) -> dict[str, Any]:
    """Get amplitude for a specific quantum state.

    Retrieves the amplitude associated with a particular quantum state
    configuration from the simulation state.

    Args:
        runner: Simulation runner containing the quantum state.
        key_state: Tuple containing state key information.

    Returns:
        Dictionary containing amplitude information for the specified state.
    """
    st = str(key_state[0][1])
    return runner.state.get_amps(st)


def sim_get_amps(
    runner: Runner,
    *_args: object,
) -> dict[str, Any]:
    """Get all quantum state amplitudes.

    Retrieves all amplitude information from the current quantum state
    in the simulation.

    Args:
        runner: Simulation runner containing the quantum state.
        *_args: Arguments (ignored).

    Returns:
        Dictionary containing all state amplitudes.
    """
    return runner.state.get_amps()


def sim_noise(
    runner: Runner,
    *_args: object,
) -> int:
    """Get current noise generation status.

    Returns whether error generation is currently enabled in the simulation.

    Args:
        runner: Simulation runner containing noise settings.
        *_args: Arguments (ignored).

    Returns:
        1 if noise generation is enabled, 0 if disabled.
    """
    return int(runner.generate_errors)


def sim_noise_off(
    runner: Runner,
    *_args: object,
) -> int:
    """Disable noise generation in simulation.

    Turns off error generation and returns the updated noise status.

    Args:
        runner: Simulation runner to modify noise settings.
        *_args: Arguments (ignored).

    Returns:
        Updated noise status (0 indicating disabled).
    """
    runner.generate_errors = False
    return sim_noise(runner)


def sim_noise_on(
    runner: Runner,
    *_args: object,
) -> int:
    """Enable noise generation in simulation.

    Turns on error generation and returns the updated noise status.

    Args:
        runner: Simulation runner to modify noise settings.
        *_args: Arguments (ignored).

    Returns:
        Updated noise status (1 indicating enabled).
    """
    runner.generate_errors = True
    return sim_noise(runner)


sim_funcs = {
    "sim_test": sim_test,
    "sim_print": sim_print,
    "sim_get_amp": sim_get_amp,
    "sim_get_amps": sim_get_amps,
    "sim_noise": sim_noise,
    "sim_noise_off": sim_noise_off,
    "sim_noise_on": sim_noise_on,
}


def sim_exec(
    func: str,
    runner: Runner,
    *args: object,
) -> int | dict[str, object] | None:
    """Execute a simulation function by name.

    Dispatches to the appropriate simulation function based on the function name,
    enabling dynamic function calls from classical virtual machine operations.

    Args:
        func: Name of the simulation function to execute.
        runner: Simulation runner to pass to the function.
        *args: Arguments to pass to the simulation function.

    Returns:
        Result from the executed simulation function.

    Raises:
        KeyError: If the function name is not recognized.
    """
    return sim_funcs[func](runner, *args)
