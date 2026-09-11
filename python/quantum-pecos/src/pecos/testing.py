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

"""Testing utilities for PECOS.

This module provides testing utilities similar to NumPy's testing module,
but using pure PECOS arrays and functions.

Like numpy.testing, this module provides assertion functions for
comparing arrays with appropriate tolerance handling.

Noiseless TickCircuit replay and signed stabilizer-group helpers support
physical circuit oracles across the QEC tests.

Example:
    >>> import pecos as pc
    >>> from pecos.testing import assert_allclose, assert_array_equal
    >>>
    >>> x = pc.array([1.0, 2.0, 3.0])
    >>> y = pc.array([1.0, 2.0, 3.0])
    >>> assert_array_equal(x, y)
    >>>
    >>> z = pc.array([1.0, 2.0, 3.001])
    >>> assert_allclose(x, z, rtol=1e-2)  # Passes with relative tolerance
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pecos_rslib import SparseStab

import pecos as pc

if TYPE_CHECKING:
    from pecos_rslib.quantum import TickCircuit

    from pecos import Array


def assert_allclose(
    actual: Array,
    desired: Array,
    rtol: float = 1e-7,
    atol: float = 0.0,
    err_msg: str = "",
    *,
    verbose: bool = True,
) -> None:
    """Assert that two arrays are element-wise equal within tolerances.

    The test verifies that all elements satisfy:
        abs(actual - desired) <= (atol + rtol * abs(desired))

    This is similar to numpy.testing.assert_allclose but uses PECOS arrays.

    Args:
        actual: Array obtained.
        desired: Array desired.
        rtol: Relative tolerance parameter (default: 1e-7).
        atol: Absolute tolerance parameter (default: 0).
        err_msg: Error message to be printed in case of failure.
        verbose: If True, include detailed information in the error message.

    Raises:
        AssertionError: If actual and desired are not equal within the specified tolerances.

    Examples:
        >>> import pecos as pc
        >>> from pecos.testing import assert_allclose
        >>> x = pc.array([1.0, 2.0, 3.0])
        >>> y = pc.array([1.0, 2.0, 3.0])
        >>> assert_allclose(x, y)
        >>> z = pc.array([1.0, 2.0, 3.001])
        >>> assert_allclose(x, z, rtol=1e-2)  # This will pass
        >>> assert_allclose(x, z, rtol=1e-5)  # This will raise AssertionError
    """
    if not pc.allclose(actual, desired, rtol=rtol, atol=atol):
        # Compute the difference for error reporting
        diff = pc.abs(actual - desired)
        max_diff = float(pc.max(diff))

        # Build error message
        msg_parts = []
        if err_msg:
            msg_parts.append(err_msg)

        msg_parts.append(
            f"Arrays are not close (rtol={rtol}, atol={atol})",
        )
        msg_parts.append(f"Max absolute difference: {max_diff}")

        # Show a few example differences if verbose
        if verbose:
            # Convert to lists for element-wise comparison (PECOS arrays don't support > operator yet)
            diff_list = [float(d) for d in diff]
            abs_desired_list = [float(abs(d)) for d in desired]

            # Find mismatches
            mismatches = []
            for i, (d, ad) in enumerate(zip(diff_list, abs_desired_list, strict=False)):
                if d > atol + rtol * ad:
                    mismatches.append((i, actual[i], desired[i], d))
                    if len(mismatches) >= 5:  # Show up to 5 examples
                        break

            if mismatches:
                # Count total mismatches
                num_total_mismatches = sum(
                    1 for d, ad in zip(diff_list, abs_desired_list, strict=False) if d > atol + rtol * ad
                )
                msg_parts.append(
                    f"Mismatched elements: {num_total_mismatches} / {len(actual)}",
                )
                msg_parts.append("Examples of mismatched values:")
                for idx, act_val, des_val, diff_val in mismatches:
                    msg_parts.append(
                        f"  Index {idx}: actual={act_val}, desired={des_val}, diff={diff_val}",
                    )
                if num_total_mismatches > len(mismatches):
                    msg_parts.append(
                        f"  ... and {num_total_mismatches - len(mismatches)} more mismatches",
                    )

        raise AssertionError("\n".join(msg_parts))


def assert_array_equal(
    actual: Array,
    desired: Array,
    err_msg: str = "",
    *,
    verbose: bool = True,
) -> None:
    """Assert that two arrays are exactly equal.

    This is equivalent to assert_allclose with rtol=0 and atol=0,
    but provides clearer error messages for exact equality checks.

    Args:
        actual: Array obtained.
        desired: Array desired.
        err_msg: Error message to be printed in case of failure.
        verbose: If True, include detailed information in the error message.

    Raises:
        AssertionError: If actual and desired are not exactly equal.

    Examples:
        >>> import pecos as pc
        >>> from pecos.testing import assert_array_equal
        >>> x = pc.array([1, 2, 3])
        >>> y = pc.array([1, 2, 3])
        >>> assert_array_equal(x, y)
    """
    assert_allclose(actual, desired, rtol=0, atol=0, err_msg=err_msg, verbose=verbose)


def assert_array_less(
    x: Array,
    y: Array,
    err_msg: str = "",
    *,
    verbose: bool = True,
) -> None:
    """Assert that x < y element-wise.

    Args:
        x: First array to compare.
        y: Second array to compare.
        err_msg: Error message to be printed in case of failure.
        verbose: If True, include detailed information in the error message.

    Raises:
        AssertionError: If any element of x is >= the corresponding element of y.

    Examples:
        >>> import pecos as pc
        >>> from pecos.testing import assert_array_less
        >>> x = pc.array([1, 2, 3])
        >>> y = pc.array([2, 3, 4])
        >>> assert_array_less(x, y)
    """
    # Convert to lists for comparison (PECOS arrays don't support < operator yet)
    x_list = [float(val) for val in x]
    y_list = [float(val) for val in y]

    violations = [(i, xv, yv) for i, (xv, yv) in enumerate(zip(x_list, y_list, strict=False)) if xv >= yv]

    if violations:
        # Build error message
        msg_parts = []
        if err_msg:
            msg_parts.append(err_msg)

        msg_parts.append("Arrays do not satisfy x < y")
        msg_parts.append(f"Violations: {len(violations)} / {len(x)}")

        if verbose and violations:
            # Show some examples
            num_show = min(5, len(violations))

            msg_parts.append("Examples of violations:")
            for i in range(num_show):
                idx, xv, yv = violations[i]
                msg_parts.append(f"  Index {idx}: x={xv}, y={yv}")

            if len(violations) > num_show:
                msg_parts.append(
                    f"  ... and {len(violations) - num_show} more violations",
                )

        raise AssertionError("\n".join(msg_parts))


__all__ = [
    "assert_allclose",
    "assert_array_equal",
    "assert_array_less",
    "group_contains",
    "simulate_tick_circuit",
    "stabilizer_generators_after",
]


def _replay_tick_circuit(tc: TickCircuit, num_ticks: int, seed: int) -> tuple[SparseStab, list[int]]:
    """Replay supported Clifford operations, rejecting unknown operations and angles."""
    if not 0 <= num_ticks <= tc.num_ticks():
        msg = "Replay tick count is outside the circuit"
        raise ValueError(msg)
    max_q = max(
        (int(q) for i in range(tc.num_ticks()) for g in tc.get_tick(i).gate_batches() for q in g.qubits),
        default=0,
    )
    sim = SparseStab(max_q + 1)
    sim.set_seed(seed)
    flat = []
    for i in range(num_ticks):
        for gate in tc.get_tick(i).gate_batches():
            name = gate.gate_type.name
            qubits = [int(q) for q in gate.qubits]
            if gate.angles:
                msg = f"Unsupported replay operation: {name} with angles"
                raise NotImplementedError(msg)
            if name in {"QAlloc", "PZ"}:
                sim.run_gate("PZ", set(qubits))
            elif name == "MZ":
                flat.extend(sim.run_gate("MZ", {q}).get(q, 0) for q in qubits)
            elif gate.is_two_qubit():
                # Gate arity accessor: python/pecos-rslib/src/dag_circuit_bindings.rs.
                sim.run_gate(name, set(zip(qubits[::2], qubits[1::2], strict=True)))
            elif gate.is_single_qubit():
                # SparseStab.run_gate raises for unsupported symbols.
                sim.run_gate(name, set(qubits))
            else:
                msg = f"Unsupported replay operation: {name}"
                raise NotImplementedError(msg)
    return sim, flat


def simulate_tick_circuit(tc: TickCircuit, seed: int = 0) -> tuple[list[int], int, dict[int, int]]:
    """Replay noiselessly; return measurements, fired detector count, and observables."""
    _, flat = _replay_tick_circuit(tc, tc.num_ticks(), seed)
    num_meas = int(tc.get_meta("num_measurements"))
    if len(flat) != num_meas:
        msg = "Replay measurement count disagrees with circuit metadata"
        raise ValueError(msg)

    def parity(records: list[int]) -> int:
        value = 0
        for rec in records:
            index = num_meas + rec
            if not 0 <= index < len(flat):
                msg = f"Invalid measurement record: {rec}"
                raise ValueError(msg)
            value ^= flat[index]
        return value

    detectors = json.loads(tc.get_meta("detectors") or "[]")
    observables = json.loads(tc.get_meta("observables") or "[]")
    return (
        flat,
        sum(parity(det["records"]) for det in detectors),
        {obs["id"]: parity(obs["records"]) for obs in observables},
    )


def stabilizer_generators_after(tc: TickCircuit, num_ticks: int, *, seed: int = 0) -> tuple[str, ...]:
    """Return signed Hermitian Pauli generators after a noiseless circuit prefix.

    SparseStab.stab_tableau (sparse_stab_bindings.rs) includes the raw phase
    and prints XZ as Y. Physical Y = iXZ, so subtract one power of i per Y
    from that raw phase. Returned strings use '+'/'-' followed by I/X/Y/Z.
    """
    sim, _ = _replay_tick_circuit(tc, num_ticks, seed)
    generators = []
    for row in sim.stab_tableau().splitlines():
        phase = 2 if row[0] == "-" else 0
        body = row[1:]
        if body.startswith("i"):
            phase += 1
            body = body[1:]
        phase = (phase - body.count("Y")) % 4
        if phase not in {0, 2}:
            msg = f"Non-Hermitian stabilizer row: {row}"
            raise ValueError(msg)
        generators.append(("+" if phase == 0 else "-") + body)
    return tuple(generators)


def group_contains(generators: tuple[str, ...], pauli: str) -> bool:
    """Test signed Pauli membership by binary elimination, retaining product phases."""
    n = len(pauli) - 1

    def encode(signed: str) -> tuple[int, int, int]:
        if len(signed) != n + 1 or signed[0] not in "+-" or set(signed[1:]) - set("IXYZ"):
            msg = f"Invalid signed Pauli: {signed}"
            raise ValueError(msg)
        body = signed[1:]
        x = sum(1 << i for i, p in enumerate(body) if p in "XY")
        z = sum(1 << i for i, p in enumerate(body) if p in "ZY")
        return x, z, ((2 if signed[0] == "-" else 0) + body.count("Y")) % 4

    def multiply(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
        x, z, phase = a
        bx, bz, bp = b
        return x ^ bx, z ^ bz, (phase + bp + 2 * (z & bx).bit_count()) % 4

    pivots = {}
    for generator in generators:
        row = encode(generator)
        while vector := row[0] | (row[1] << n):
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row = multiply(row, pivots[pivot])
    row = encode(pauli)
    while vector := row[0] | (row[1] << n):
        pivot = vector.bit_length() - 1
        if pivot not in pivots:
            return False
        row = multiply(row, pivots[pivot])
    return row == (0, 0, 0)
