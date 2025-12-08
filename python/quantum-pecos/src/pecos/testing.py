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

"""Testing utilities for PECOS.

This module provides testing utilities similar to NumPy's testing module,
but using pure PECOS arrays and functions.

Like numpy.testing, this module provides assertion functions for
comparing arrays with appropriate tolerance handling.

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
                n_total_mismatches = sum(
                    1
                    for d, ad in zip(diff_list, abs_desired_list, strict=False)
                    if d > atol + rtol * ad
                )
                msg_parts.append(
                    f"Mismatched elements: {n_total_mismatches} / {len(actual)}",
                )
                msg_parts.append("Examples of mismatched values:")
                for idx, act_val, des_val, diff_val in mismatches:
                    msg_parts.append(
                        f"  Index {idx}: actual={act_val}, desired={des_val}, diff={diff_val}",
                    )
                if n_total_mismatches > len(mismatches):
                    msg_parts.append(
                        f"  ... and {n_total_mismatches - len(mismatches)} more mismatches",
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

    violations = [
        (i, xv, yv)
        for i, (xv, yv) in enumerate(zip(x_list, y_list, strict=False))
        if xv >= yv
    ]

    if violations:
        # Build error message
        msg_parts = []
        if err_msg:
            msg_parts.append(err_msg)

        msg_parts.append("Arrays do not satisfy x < y")
        msg_parts.append(f"Violations: {len(violations)} / {len(x)}")

        if verbose and violations:
            # Show some examples
            n_show = min(5, len(violations))

            msg_parts.append("Examples of violations:")
            for i in range(n_show):
                idx, xv, yv = violations[i]
                msg_parts.append(f"  Index {idx}: x={xv}, y={yv}")

            if len(violations) > n_show:
                msg_parts.append(
                    f"  ... and {len(violations) - n_show} more violations",
                )

        raise AssertionError("\n".join(msg_parts))


__all__ = [
    "assert_allclose",
    "assert_array_equal",
    "assert_array_less",
]
