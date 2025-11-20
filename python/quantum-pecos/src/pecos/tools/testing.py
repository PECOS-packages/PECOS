# Copyright 2024 The PECOS Developers
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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc

if TYPE_CHECKING:
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
        >>> from pecos.tools.testing import assert_allclose
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

        # Find where the arrays differ
        threshold = atol + rtol * pc.abs(desired)
        mismatch = diff > threshold

        # Count mismatches
        n_mismatch = int(pc.sum(mismatch))

        # Build error message
        msg_parts = []
        if err_msg:
            msg_parts.append(err_msg)

        msg_parts.append(
            f"Arrays are not close (rtol={rtol}, atol={atol})",
        )
        msg_parts.append(f"Mismatched elements: {n_mismatch} / {len(actual)}")
        msg_parts.append(f"Max absolute difference: {max_diff}")

        if verbose and n_mismatch > 0:
            # Show some examples of mismatched values
            # Find indices of mismatches
            mismatch_indices = pc.where(mismatch)[0]
            n_show = min(5, len(mismatch_indices))  # Show up to 5 examples

            msg_parts.append("Examples of mismatched values:")
            for i in range(n_show):
                idx = int(mismatch_indices[i])
                msg_parts.append(
                    f"  Index {idx}: actual={actual[idx]}, desired={desired[idx]}, "
                    f"diff={diff[idx]}",
                )

            if n_mismatch > n_show:
                msg_parts.append(f"  ... and {n_mismatch - n_show} more mismatches")

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
        >>> from pecos.tools.testing import assert_array_equal
        >>> x = pc.array([1, 2, 3])
        >>> y = pc.array([1, 2, 3])
        >>> assert_array_equal(x, y)
    """
    if not pc.array_equal(actual, desired):
        # Find where the arrays differ
        mismatch = actual != desired
        n_mismatch = int(pc.sum(mismatch))

        # Build error message
        msg_parts = []
        if err_msg:
            msg_parts.append(err_msg)

        msg_parts.append("Arrays are not equal")
        msg_parts.append(f"Mismatched elements: {n_mismatch} / {len(actual)}")

        if verbose and n_mismatch > 0:
            # Show some examples of mismatched values
            mismatch_indices = pc.where(mismatch)[0]
            n_show = min(5, len(mismatch_indices))  # Show up to 5 examples

            msg_parts.append("Examples of mismatched values:")
            for i in range(n_show):
                idx = int(mismatch_indices[i])
                msg_parts.append(
                    f"  Index {idx}: actual={actual[idx]}, desired={desired[idx]}",
                )

            if n_mismatch > n_show:
                msg_parts.append(f"  ... and {n_mismatch - n_show} more mismatches")

        raise AssertionError("\n".join(msg_parts))


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
        >>> from pecos.tools.testing import assert_array_less
        >>> x = pc.array([1, 2, 3])
        >>> y = pc.array([2, 3, 4])
        >>> assert_array_less(x, y)
    """
    if not pc.all(x < y):
        # Find where the condition is violated
        violation = x >= y
        n_violations = int(pc.sum(violation))

        # Build error message
        msg_parts = []
        if err_msg:
            msg_parts.append(err_msg)

        msg_parts.append("Arrays do not satisfy x < y")
        msg_parts.append(f"Violations: {n_violations} / {len(x)}")

        if verbose and n_violations > 0:
            # Show some examples
            violation_indices = pc.where(violation)[0]
            n_show = min(5, len(violation_indices))

            msg_parts.append("Examples of violations:")
            for i in range(n_show):
                idx = int(violation_indices[i])
                msg_parts.append(f"  Index {idx}: x={x[idx]}, y={y[idx]}")

            if n_violations > n_show:
                msg_parts.append(f"  ... and {n_violations - n_show} more violations")

        raise AssertionError("\n".join(msg_parts))
